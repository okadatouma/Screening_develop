"""学習済みチェックポイントから、局所エネルギーを運動項 T と Coulomb 項 V に分けて測る。

    python screening_analysis/tv_split.py results/2deg-Coulomb/CustomScreenformer/el5_4_rs5_tri_ctl
    python screening_analysis/tv_split.py <folder1> <folder2> ...        # まとめて表にする
    python screening_analysis/tv_split.py <folder> --blocks 40 --mcmc 20

train_stats.csv には合計 E しか残らないので、ここで pbc/hamiltonian.py と同じ式を
使って T と V を別々に評価し直す。<T>, <V>, |V|/T（結合強度）, ビリアル残差を出す。

単位について。configs/run_2deg_screen.py は長さを r_s a_B、エネルギーを Ha/r_s^2 に
とっている（interaction_energy_scale = r_s がその名残）。T と V を同じ尺度で割るので
比 |V|/T はこのスケーリングに依らない。参考のため Ha/electron 換算も出す。

注意: V は背景電荷を引いた後の値なので Hartree 項は入っていない。したがって
|V|/T は「交換相関 + Madelung」対「運動」の比であって、素の Hartree 比ではない。
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import jax
import jax.numpy as jnp
import folx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave import networks, envelopes, mcmc, CustomScreenformer, CustomPsiformer  # noqa: E402
from periodicwave.pbc import feature_layer as FL, hamiltonian as HAM  # noqa: E402


def _lattice(cfg):
    """config.json は numpy 配列を文字列で吐くので数値を拾い直す。列がベクトル。"""
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v[:4]).reshape(2, 2)


def build(folder, checkpoint=None):
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    data = z["data"].item()
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[0], z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))

    L = jnp.asarray(_lattice(cfg))
    nspins = tuple(cfg["system"]["electrons"])
    ndim = cfg["system"]["ndim"]
    charges = jnp.zeros(1)                      # 2DEG: ダミー原子 1 個、電荷 0
    r_s = float(cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
                ["interaction_energy_scale"])
    ntype = cfg["network"]["network_type"]

    fl = FL.make_pbc_feature_layer(natoms=1, nspins=nspins, ndim=ndim, lattice=L,
                                   include_r_ae=False)
    mod = CustomScreenformer if ntype == "CustomScreenformer" else CustomPsiformer
    net = mod.make_fermi_net(
        nspins, charges, ndim=ndim,
        determinants=cfg["network"]["determinants"],
        envelope=envelopes.make_null_envelope(), feature_layer=fl,
        jastrow=cfg["network"]["jastrow"], jastrow_kwargs=cfg["network"]["jastrow_kwargs"],
        bias_orbitals=cfg["network"]["bias_orbitals"],
        complex_output=cfg["network"]["complex"], pbc_lattice=L,
        **cfg["network"][ntype])

    signed = net.apply
    logabs = lambda *a, **k: signed(*a, **k)[1]
    pot = HAM.make_2DCoulomb_potential(L, jnp.array([0.0]), charges, 5, r_s)
    cplx = cfg["network"]["complex"]

    def kinetic(p, pos, spins, atoms, chg):
        """pbc/hamiltonian.py の folx 分岐と同一。T_L = -1/2 [ lap ln|psi| + |grad ln psi|^2 ]。"""
        w = folx.forward_laplacian(lambda x: signed(p, x, spins, atoms, chg),
                                   sparsity_threshold=0)(pos)
        out = -(w[1].laplacian + jnp.sum(w[1].jacobian.dense_array ** 2)) / 2
        if cplx:                                   # 位相の寄与（実部のみ効く）
            out = out + 0.5 * jnp.sum(w[0].jacobian.dense_array ** 2)
        return jnp.real(out)

    def potential(pos, atoms):
        ae, ee, _, _ = networks.construct_input_features(pos, atoms, ndim=ndim)
        return jnp.real(pot(ae, ee))

    return dict(params=params, data=data, logabs=logabs, kinetic=kinetic,
                potential=potential, nspins=nspins, ndim=ndim, r_s=r_s,
                ne=sum(nspins), ntype=ntype, ck=os.path.basename(ck), folder=folder)


def measure(b, blocks=30, mcmc_steps=20, width=0.2, seed=0):
    d = b["data"]
    pos = jnp.asarray(d["positions"][0])
    spins, atoms, chg = (jnp.asarray(d[k][0]) for k in ("spins", "atoms", "charges"))
    batch_net = jax.vmap(b["logabs"], in_axes=(None, 0, 0, 0, 0))
    step = mcmc.make_mcmc_step(batch_net, pos.shape[0], steps=mcmc_steps, ndim=b["ndim"])
    bT = jax.jit(jax.vmap(b["kinetic"], in_axes=(None, 0, 0, 0, 0)))
    bV = jax.jit(jax.vmap(b["potential"], in_axes=(0, 0)))

    key = jax.random.PRNGKey(seed)
    T, V = [], []
    for i in range(blocks):
        key, sub = jax.random.split(key)
        nd, _ = step(b["params"], _FND(pos, spins, atoms, chg), sub, width)
        pos = nd.positions
        T.append(np.asarray(bT(b["params"], pos, spins, atoms, chg)))
        V.append(np.asarray(bV(pos, atoms)))
    return np.concatenate(T), np.concatenate(V)


def _FND(pos, spins, atoms, chg):
    return networks.FermiNetData(positions=pos, spins=spins, atoms=atoms, charges=chg)


def _err(x, nblock=20):
    """ブロック平均で自己相関を潰した標準誤差。"""
    m = len(x) // nblock * nblock
    b = x[:m].reshape(nblock, -1).mean(1)
    return b.std(ddof=1) / np.sqrt(nblock)


def report(rows):
    print("\n  %-26s %5s %6s %10s %10s %10s %8s %9s"
          % ("run", "N", "r_s", "<T>", "<V>", "<E>", "|V|/T", "E/N [Ha]"))
    print("  " + "-" * 92)
    for r in rows:
        print("  %-26s %5d %6.4g %10.3f %10.3f %10.3f %8.2f %9.4f"
              % (r["name"], r["ne"], r["r_s"], r["T"], r["V"], r["T"] + r["V"],
                 abs(r["V"]) / r["T"], (r["T"] + r["V"]) / r["ne"] / r["r_s"] ** 2))
    print()
    for r in rows:
        print("  %-26s  T = %8.3f +- %.3f   V = %8.3f +- %.3f   virial 2T+V = %8.3f"
              % (r["name"], r["T"], r["eT"], r["V"], r["eV"], 2 * r["T"] + r["V"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--blocks", type=int, default=30)
    ap.add_argument("--mcmc", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for f in a.folders:
        b = build(f)
        T, V = measure(b, blocks=a.blocks, mcmc_steps=a.mcmc)
        rows.append(dict(name=os.path.basename(f), ne=b["ne"], r_s=b["r_s"],
                         T=T.mean(), V=V.mean(), eT=_err(T), eV=_err(V),
                         Traw=T, Vraw=V))
        print("  done %-30s T=%9.3f  V=%9.3f  |V|/T=%6.2f"
              % (os.path.basename(f), T.mean(), V.mean(), abs(V.mean()) / T.mean()))
    report(rows)
    if a.out:
        np.savez(a.out, **{f"{r['name']}_T": r["Traw"] for r in rows},
                 **{f"{r['name']}_V": r["Vraw"] for r in rows})
