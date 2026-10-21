"""再構成したネットワークで局所エネルギーを測り、学習ログと突き合わせる。

    python screening_analysis/energy_check.py <folder> [--nconf 512]

これが合っていれば、sampler.build が学習時と同じ波動関数を組めていることの証拠になる。
"""
import argparse, glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import sampler as SM                                   # noqa
from periodicwave import envelopes, networks           # noqa
from periodicwave.pbc import feature_layer as FL, hamiltonian as HAM   # noqa
from periodicwave import CustomScreenformer as CS, CustomPsiformer as CP  # noqa


def build_signed(folder, checkpoint=None):
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True); data = z["data"].item()
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)[0]), z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    _m = SM._mats(cfg); L, MO = _m[0], (_m[1] if len(_m) > 1 else None)
    nspins = tuple(cfg["system"]["electrons"]); ntype = cfg["network"]["network_type"]
    charges = jnp.asarray(np.asarray(data["charges"][0])[0])
    fl = FL.make_pbc_feature_layer(natoms=charges.shape[0], nspins=nspins, ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    mk = CP.make_fermi_net if ntype == "CustomPsiformer" else CS.make_fermi_net
    net = mk(nspins, charges, ndim=2, determinants=cfg["network"]["determinants"],
             envelope=envelopes.make_null_envelope(), feature_layer=fl,
             jastrow=cfg["network"].get("jastrow", "NONE"),
             jastrow_kwargs=cfg["network"]["jastrow_kwargs"],
             bias_orbitals=cfg["network"]["bias_orbitals"],
             complex_output=cfg["network"].get("complex", False),
             pbc_lattice=jnp.asarray(L), **SM._net_kwargs(cfg, ntype, params))
    ke = dict(cfg["system"]["make_local_energy_kwargs"])
    ke["lattice"] = jnp.asarray(L)
    pk = dict(ke["potential_kwargs"])
    if MO is not None: pk["moire_lattice_vectors"] = jnp.asarray(MO)
    ke["potential_kwargs"] = pk
    le = HAM.local_energy(f=net.apply, charges=charges, nspins=nspins,
                          complex_output=cfg["network"].get("complex", False),
                          use_scan=False, **ke)
    return net, params, le, data, cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=512)
    ap.add_argument("--nwalk", type=int, default=64)
    a = ap.parse_args()
    net, params, le, data, cfg = build_signed(a.folder)
    R, pm = SM.sample(a.folder, a.nconf, a.nwalk, verbose=True)
    ne = sum(cfg["system"]["electrons"])
    sp = jnp.asarray(np.tile(np.asarray(data["spins"][0])[0], (len(R), 1)))
    at = jnp.asarray(np.tile(np.asarray(data["atoms"][0])[0], (len(R), 1, 1)))
    ch = jnp.asarray(np.tile(np.asarray(data["charges"][0])[0], (len(R), 1)))
    key = jax.random.split(jax.random.PRNGKey(0), len(R))
    x = jnp.asarray(R.reshape(len(R), -1))
    dat = networks.FermiNetData(positions=x, spins=sp, atoms=at, charges=ch)
    f = jax.jit(jax.vmap(le, in_axes=(None, 0, 0)))
    out = f(params, key, dat)
    E = np.asarray(out[0] if isinstance(out, tuple) else out)
    E = np.real(E)
    lo, hi = np.percentile(E, [1, 99])
    Ec = E[(E > lo) & (E < hi)]
    d = np.genfromtxt(os.path.join(a.folder, "train_stats.csv"), delimiter=",", names=True)
    print("\n  再構成したネットワーク: E = %.4f ± %.4f  (1%%外れ値を落として %.4f)"
          % (E.mean(), E.std() / np.sqrt(len(E)), Ec.mean()))
    print("  学習ログ（最後の 300 step）: E = %.4f ± %.4f   ewmean = %.4f"
          % (d["energy"][-300:].mean(), d["energy"][-300:].std() / np.sqrt(300),
             d["ewmean"][-1]))
    print("  局所エネルギーの標準偏差 %.3f （ログ %.3f）"
          % (E.std(), d["locstd"][-300:].mean()))


if __name__ == "__main__":
    main()
