"""小さい固定入力だけを共有する。数値ライブラリの import は明示的に行う。"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "jax", "jaxlib", "folx", "kfac_jax", "optax",
            "ml_collections", "chex", "attr", "absl", "typing_extensions")
ROUTES = ("psiformer", "slater", "screen", "lambda", "hf")


def require_numerics():
    missing = [name for name in PACKAGES if importlib.util.find_spec(name) is None]
    reasons = (["Python >= 3.10 が必要（本体の型注釈）"]
               if sys.version_info < (3, 10) else [])
    if missing:
        reasons.append("未導入: " + ", ".join(missing))
    if reasons:
        raise unittest.SkipTest("; ".join(reasons))
    # CPU/倍精度が標準。既存の本体やユーザー環境の設定は書き換えない。
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", os.environ.get("PW_TEST_X64", "1") != "0")
    jax.config.update("jax_default_matmul_precision", "float32")


def make_case(route, directory):
    """N=2、1層、2ヘッド。HF表は一時領域の合成平面波であり研究データではない。"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from periodicwave import CustomPsiformer as CP, CustomScreenformer as CS
    from periodicwave import SlaterNet, envelopes, networks
    from periodicwave.pbc import feature_layer, hamiltonian, lambda_jellium

    lattice = jnp.asarray([[3.2, -1.6], [0., 2.7712812921102037]])
    nspins = (1, 1) if route in ("psiformer", "slater", "screen") else (2, 0)
    options = dict(ndim=2, determinants=2, envelope=envelopes.make_null_envelope(),
                   feature_layer=feature_layer.make_pbc_feature_layer(
                       natoms=1, nspins=nspins, ndim=2, lattice=lattice, include_r_ae=False),
                   jastrow="NONE", jastrow_kwargs={"ndim": 2}, complex_output=True,
                   bias_orbitals=False, pbc_lattice=lattice, num_layers=1,
                   mlp_dim=8, num_perceptrons_per_layer=1, use_layer_norm=True,
                   mlp_activation_fct="GELU")
    if route != "slater":
        options.update(num_heads=2, attn_dim=4, value_dim=4)
    if route == "hf":
        # k はスーパーセルの逆格子。G=0 の2軌道、列順は本体の6列形式。
        k = 2 * np.pi * np.linalg.inv(np.asarray(lattice))
        table = np.array([[*k[0], 0., 0., 1., 0.], [*k[1], 0., 0., 1., 0.]])
        hf_path = Path(directory) / "synthetic_hf.txt"
        np.savetxt(hf_path, table, fmt="%.17g")
        options.update(lj_hf_path=str(hf_path), lj_lam=0.4,
                       lj_cusps=(0.2, 0.5, 0.3), lj_net_scale=0.05, lj_jastrow=True)
    maker = {"psiformer": CP, "slater": SlaterNet}.get(route, CS)
    net = maker.make_fermi_net(nspins, jnp.zeros(1), **options)
    params = net.init(jax.random.PRNGKey(7))
    spins = jnp.asarray([1., -1.])
    data = networks.FermiNetData(positions=jnp.asarray([0.23, 0.41, 1.17, 0.89]),
                               spins=spins, atoms=jnp.zeros((1, 2)), charges=jnp.zeros(1))
    energy_options = dict(lattice=lattice, potential_type="Coulomb", convergence_radius=2,
                          potential_kwargs={"interaction_energy_scale": 2.},
                          kinetic_kwargs={"laplacian_method": "folx"})
    energy_module = lambda_jellium if route in ("lambda", "hf") else hamiltonian
    if route in ("lambda", "hf"):
        energy_options.update(lam=0.4, delta=2., flip_batch=1)
    energy = energy_module.local_energy(net.apply, data.charges, nspins,
                                        complex_output=True, **energy_options)
    return net, params, data, lattice, options, energy


def batch_data(data):
    import jax
    import jax.numpy as jnp
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), data)
    batch.positions = batch.positions.at[1].add(jnp.asarray([0.13, -0.07, 0.03, 0.11]))
    return batch


def assert_tree_close(test, left, right, rtol=1e-9, atol=1e-9):
    import jax
    import numpy as np
    test.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
