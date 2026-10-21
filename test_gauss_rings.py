# -*- coding: utf-8 -*-
"""ガウスでぼかした輪切り（PW_ENV_KERNEL=gauss）の検算（CPU、倍精度）。

    JAX_PLATFORMS=cpu PYTHONPATH=. python test_gauss_rings.py [base_CustomScreenformer.py]

1. 階段 _ring_steps：1 の分割、d = 0 で傾き 0（偶関数）、負にならない
2. 既存の輪切り（sin）は変更前のコードとビット単位で一致（変更前のファイルを引数で渡したとき）
3. ガウス核の網で E_L（folx）と E_L（jax のヘッセ行列）が一致し、NaN が出ず、folx が erf を登録済みとして扱う
4. 1 電子を直線上で動かしたときの E_L：刻みを 1/8 にしたとき、隣り合う点の差の最大が
   ガウス核では 1/8 に縮み（連続）、sin の tau = 1 では縮まない（跳び）
"""
import importlib.util
import logging
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from periodicwave import networks, CustomScreenformer as CS, envelopes
from periodicwave.pbc import lambda_jellium as LJ, lattices, feature_layer as FL

A_WC = np.sqrt(2 * np.pi / np.sqrt(3))
ATOMS = jnp.zeros((1, 2)); CHARGES = jnp.zeros(1)
FAILS = []


def check(name, ok, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def cell(n):
    lat, _ = lattices._triangular_lattice_vecs_periodic_potential(A_WC, n)
    return jnp.asarray(lat)


def data_of(x, n):
    return networks.FermiNetData(positions=jnp.asarray(x), spins=jnp.ones(n), atoms=ATOMS, charges=CHARGES)


def build_net(mod, lat, n, rings, heads=8, seed=3):
    feat = FL.make_pbc_feature_layer(natoms=1, nspins=(n, 0), ndim=2, lattice=lat, include_r_ae=False)
    net = mod.make_fermi_net(
        (n, 0), CHARGES, ndim=2, determinants=2, envelope=envelopes.make_null_envelope(),
        feature_layer=feat, jastrow='NONE', jastrow_kwargs={'ndim': 2}, complex_output=True,
        bias_orbitals=False, num_layers=1, num_heads=heads, attn_dim=8, value_dim=8, mlp_dim=16,
        num_perceptrons_per_layer=1, use_layer_norm=True, mlp_activation_fct='GELU', pbc_lattice=lat,
        init_softmask_w_factor=0.7, init_softmask_w_spread=1.5, init_softmask_rings=rings)
    return net, net.init(jax.random.PRNGKey(seed))


def rings_tuple(dref, heads, kernel, width):
    """(境界/d_ref, 幅/d_ref, p[, 核])。width は sin なら tau、gauss なら sigma/輪の幅。"""
    eb = np.linspace(0.0, dref, heads + 1); D = dref / heads
    if kernel == 'gauss':
        return (eb / dref, width * D / dref, 3.0, 'gauss')
    return (eb / dref, width * D / 2 / dref, 3.0)


def kinetic(f, lat, method):
    return LJ.local_energy(f, CHARGES, (0, 0), complex_output=True, lattice=lat, lam=0.0,
                           potential_type='none', kinetic_kwargs={'laplacian_method': method})


def test_steps():
    print("[1] 階段 _ring_steps（N = 25 のセル、8 本、sigma = 輪の幅の半分）")
    dref, H = 3.2152, 8
    b = jnp.linspace(0.0, dref, H + 1); sig = 0.5 * dref / H
    d = jnp.linspace(0.0, dref, 4001)[None, :]                 # [1, M] を「対」の並びとみなす
    th = CS._ring_steps(d, b, sig, gauss=True)[:, 0, :]
    m = th[:-1] - th[1:]
    check("1 の分割 sum_h m_h = 1", float(jnp.abs(m.sum(0) - 1).max()) < 1e-12,
          f"max|sum - 1| {float(jnp.abs(m.sum(0) - 1).max()):.1e}")
    check("窓は負にならない", float(m.min()) > -1e-12, f"min {float(m.min()):.1e}")
    dm = jax.vmap(jax.grad(lambda x, h: (CS._ring_steps(x[None, None], b, sig, True)[h, 0, 0]
                                          - CS._ring_steps(x[None, None], b, sig, True)[h + 1, 0, 0])),
                  in_axes=(None, 0))(jnp.asarray(0.0), jnp.arange(H))
    check("d = 0 で傾き 0（偶関数）", float(jnp.abs(dm).max()) < 1e-12, f"max|m_h'(0)| {float(jnp.abs(dm).max()):.1e}")


def test_old_path(base_file):
    print("[2] sin の輪切りは変更前のコードとビット単位で一致")
    if not base_file:
        print("  （変更前のファイルが渡されていないので省く）"); return
    spec = importlib.util.spec_from_file_location("CS_base", base_file)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    lat = cell(3); n = 9; dref = CS._max_periodic_distance(np.asarray(lat))
    rings = rings_tuple(dref, 4, 'sin', 1.0)
    (na, pa), (nb, pb) = build_net(base, lat, n, rings, heads=4), build_net(CS, lat, n, rings, heads=4)
    same_tree = jax.tree_util.tree_all(jax.tree_util.tree_map(lambda u, v: bool(jnp.array_equal(u, v)), pa, pb))
    check("初期パラメータが同じ", same_tree and jax.tree_util.tree_structure(pa) == jax.tree_util.tree_structure(pb))
    rng = np.random.default_rng(0)
    ea, eb_ = jax.jit(kinetic(na.apply, lat, 'folx')), jax.jit(kinetic(nb.apply, lat, 'folx'))
    worst = 0.0
    for _ in range(3):
        x = rng.uniform(-2, 2, size=2 * n); dd = data_of(x, n)
        la, lb = na.apply(pa, dd.positions, dd.spins, ATOMS, CHARGES), nb.apply(pb, dd.positions, dd.spins, ATOMS, CHARGES)
        a, b = complex(ea(pa, None, dd)[0]), complex(eb_(pb, None, dd)[0])
        worst = max(worst, abs(a - b), float(abs(la[1] - lb[1])), float(abs(la[0] - lb[0])))
    check("ln psi と E_L がビット単位で一致", worst == 0.0, f"max|diff| {worst:.1e}")


def test_folx_vs_default():
    print("[3] ガウス核の網：E_L（folx）と E_L（jax のヘッセ行列）")
    lat = cell(3); n = 9; dref = CS._max_periodic_distance(np.asarray(lat))
    net, params = build_net(CS, lat, n, rings_tuple(dref, 4, 'gauss', 0.5), heads=4)
    check("パラメータに 'mask_gauss' がある", 'mask_gauss' in params['layers']['attention'][0])
    msgs = []

    class Grab(logging.Handler):
        def emit(self, r):
            msgs.append(r.getMessage())
    logging.getLogger().addHandler(Grab())
    ef, ed = jax.jit(kinetic(net.apply, lat, 'folx')), jax.jit(kinetic(net.apply, lat, 'default'))
    rng = np.random.default_rng(1); worst = 0.0; finite = True
    for _ in range(4):
        dd = data_of(rng.uniform(-2, 2, size=2 * n), n)
        a, b = complex(ef(params, None, dd)[0]), complex(ed(params, None, dd)[0])
        finite &= np.isfinite(a) and np.isfinite(b)
        worst = max(worst, abs(a - b) / max(1.0, abs(b)))
    check("folx と jax が一致", worst < 1e-10, f"相対差の最大 {worst:.1e}")
    check("NaN / inf なし", finite)
    check("folx が erf を登録済みとして扱う（ヘッセ行列の全計算なし）",
          not any("erf" in m and "registry" in m for m in msgs))


def test_continuity():
    print("[4] 1 電子を動かしたときの E_L の連続性（N = 9、4 本の輪）")
    lat = cell(3); n = 9; dref = CS._max_periodic_distance(np.asarray(lat))
    rng = np.random.default_rng(7)
    x0 = rng.uniform(-2, 2, size=2 * n)
    direction = np.zeros(2 * n); direction[0:2] = [1.0, 0.3]; direction /= np.linalg.norm(direction)
    out = {}
    for name, rings in (("sin tau=1", rings_tuple(dref, 4, 'sin', 1.0)),
                        ("gauss sigma=0.5", rings_tuple(dref, 4, 'gauss', 0.5))):
        net, params = build_net(CS, lat, n, rings, heads=4, seed=11)
        el = jax.jit(jax.vmap(lambda x: kinetic(net.apply, lat, 'folx')(params, None, data_of(x, n))[0].real))
        t = np.linspace(0.0, 1.5, 3001)
        E = np.asarray(el(jnp.asarray(x0[None] + t[:, None] * direction[None])))
        k = int(np.argmax(np.abs(np.diff(E))))                 # いちばん大きい段差のところ
        t2 = np.linspace(t[k], t[k + 1], 9)                    # その 1 区間を 8 等分
        E2 = np.asarray(el(jnp.asarray(x0[None] + t2[:, None] * direction[None])))
        j1, j2 = float(np.abs(np.diff(E)).max()), float(np.abs(np.diff(E2)).max())
        out[name] = (t, E, j1, j2)
        print(f"    {name:16s} 隣の点の差の最大 {j1:.4f}  -> 刻み 1/8 で {j2:.4f}（比 {j2 / j1:.3f}）")
    j = out["gauss sigma=0.5"]; check("ガウス核：刻み 1/8 で差も約 1/8（連続）", j[3] / j[2] < 0.2)
    j = out["sin tau=1"]; check("sin tau = 1：刻みを細かくしても差が残る（跳び）", j[3] / j[2] > 0.5)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "screening_analysis", "out_gauss_path.npz"),
             **{k.replace(" ", "_").replace("=", ""): np.stack([v[0], v[1]]) for k, v in out.items()})


if __name__ == "__main__":
    test_steps()
    test_old_path(sys.argv[1] if len(sys.argv) > 1 else "")
    test_folx_vs_default()
    test_continuity()
    print("\n失敗: " + (", ".join(FAILS) if FAILS else "なし"))
    sys.exit(1 if FAILS else 0)
