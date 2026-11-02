# -*- coding: utf-8 -*-
"""λ-jellium の学習結果を項ごとに分解する（チェックポイントのウォーカーで評価）。

    python screening_analysis/lj_components.py <結果フォルダ> [--nconf 256] [--checkpoint PATH]

出力（N 電子の合計、コード単位。± はウォーカー間の標準誤差）
  T   = -1/2 sum_i lap_i Psi/Psi                 自由粒子の運動エネルギー
  K   = -Delta lam^2 sum_{s_i=+} lap_i Psi/Psi
  C   = Delta n_-
  F   = i Delta lam sum_i [Psi(S^(i))/Psi(S)] D_i log Psi(S^(i))
  V   = Ewald Coulomb
  E   = T + K + C + F + V（学習ログの energy と比べる）
  H_Delta = K + C + F（h の和の期待値、>= 0。下のバンドだけなら 0）
  w_up = <H_Delta> / (N Delta)                   上のバンドへの混ざりの目安
  n_-/N                                          下成分にいる電子の割合
"""
import argparse, glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import sampler as SM                                                   # noqa
from periodicwave import envelopes, networks                           # noqa
from periodicwave import CustomScreenformer as CS                      # noqa
from periodicwave.pbc import feature_layer as FL, lambda_jellium as LJ   # noqa


def build(folder, checkpoint=None):
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True); data = z["data"].item()
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)[0]), z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = SM._mats(cfg)[0]
    nspins = tuple(cfg["system"]["electrons"])
    charges = jnp.asarray(np.asarray(data["charges"][0])[0])
    fl = FL.make_pbc_feature_layer(natoms=charges.shape[0], nspins=nspins, ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    net = CS.make_fermi_net(nspins, charges, ndim=2, determinants=cfg["network"]["determinants"],
                            envelope=envelopes.make_null_envelope(), feature_layer=fl,
                            jastrow=cfg["network"].get("jastrow", "NONE"),
                            jastrow_kwargs=cfg["network"]["jastrow_kwargs"],
                            bias_orbitals=cfg["network"]["bias_orbitals"], complex_output=True,
                            pbc_lattice=jnp.asarray(L),
                            **SM._net_kwargs(cfg, "CustomScreenformer", params))
    ke = cfg["system"]["make_local_energy_kwargs"]
    return net, params, data, cfg, L, ke, ck


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=256)
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    net, params, data, cfg, L, ke, ck = build(a.folder, a.checkpoint)
    lam = float(ke["lam"]); delta = float(ke["delta"])
    r_s = float(ke["potential_kwargs"]["interaction_energy_scale"])
    coul = ke.get("potential_type", "Coulomb") == "Coulomb"
    comps = LJ.make_energy_components(net.apply, jnp.asarray(L), lam, delta, r_s, coulomb=coul)
    pos = np.asarray(data["positions"][0]); sp = np.asarray(data["spins"][0])
    at = np.asarray(data["atoms"][0]); ch = np.asarray(data["charges"][0])
    n = min(a.nconf, pos.shape[0]); N = sp.shape[1]
    f = jax.jit(jax.vmap(lambda d: comps(params, d)))
    out = []
    for i in range(0, n, 32):
        sl = slice(i, min(i + 32, n))
        d = networks.FermiNetData(positions=jnp.asarray(pos[sl]), spins=jnp.asarray(sp[sl]),
                                  atoms=jnp.asarray(at[sl]), charges=jnp.asarray(ch[sl]))
        out.append(np.asarray(f(d)))
    X = np.concatenate(out)                                    # (n, 5) 複素
    E = X.sum(1); HD = X[:, 1:4].sum(1)
    se = lambda v: v.real.std(ddof=1) / np.sqrt(len(v))
    print(f"{os.path.basename(os.path.normpath(a.folder))}   {os.path.basename(ck)}   walkers {n}   "
          f"N = {N}, r_s = {r_s:g}, lam = {lam:g}, Delta = {delta:g}")
    for k, name in enumerate(["T", "K", "C", "F", "V"]):
        print(f"  {name:8s} {X[:, k].real.mean():12.4f} ± {se(X[:, k]):.4f}   (Im 平均 {X[:, k].imag.mean():+.3f})")
    print(f"  {'E':8s} {E.real.mean():12.4f} ± {se(E):.4f}   σ(E_L) {np.sqrt(np.mean(np.abs(E - E.mean())**2)):.3f}"
          f"   Im 平均 {E.imag.mean():+.4f}")
    print(f"  {'H_Delta':8s} {HD.real.mean():12.4f} ± {se(HD):.4f}   w_up = <H_Delta>/(N Delta) = "
          f"{HD.real.mean() / (N * delta):.4f}")
    print(f"  n_-/N    {np.mean(sp[:n] < 0):.4f}")
    st = os.path.join(a.folder, "train_stats.csv")
    if os.path.exists(st):
        import pandas as pd
        s = pd.read_csv(st); w = s.tail(100)
        print(f"  学習ログ（最後の 100 step）energy {w.energy.mean():.4f}")


if __name__ == "__main__":
    main()
