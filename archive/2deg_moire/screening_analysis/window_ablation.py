"""波動関数はどこまでの距離を必要としているか — 注意の届く範囲を硬く切って測る。

    python screening_analysis/window_ablation.py <folder> [--nconf 384]

学習済みの波動関数をそのまま使い、注意の重みだけを半径 R で切る（窓の形は触らない）。
その波動関数から改めてサンプルして変分エネルギーを測ると、R を小さくしていったとき
波動関数が本当に必要としている距離のところでエネルギーが立ち上がる。再学習が要らない。
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import moire_analysis as MA                              # noqa
from energy_check import build_signed                    # noqa
from periodicwave import networks, mcmc as MC            # noqa
from periodicwave import CustomScreenformer as CS        # noqa


def _measure(net, params, le, data, nconf, nwalk, seed):
    """CLIP_R は呼び出し側で設定しておく。jit はここで作り直す（trace 時に読むため）。"""
    logabs = lambda p, x, s, a, c: net.apply(p, x, s, a, c)[1]
    bn = jax.vmap(logabs, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    step = jax.jit(MC.make_mcmc_step(bn, nwalk, steps=10, ndim=2))
    lef = jax.jit(jax.vmap(le, in_axes=(None, 0, 0)))
    dat = networks.FermiNetData(
        positions=jnp.asarray(np.asarray(data["positions"][0])[:nwalk]),
        spins=jnp.asarray(np.asarray(data["spins"][0])[:nwalk]),
        atoms=jnp.asarray(np.asarray(data["atoms"][0])[:nwalk]),
        charges=jnp.asarray(np.asarray(data["charges"][0])[:nwalk]))
    key = jax.random.PRNGKey(seed); w = jnp.asarray(0.15)
    for i in range(50):
        key, sk = jax.random.split(key)
        dat, pm = step(params, dat, sk, w)
        if (i + 1) % 2 == 0:
            w = w * (1.15 if pm > .55 else (1 / 1.15 if pm < .45 else 1.0))
    E = []
    while len(E) * nwalk < nconf:
        for _ in range(3):
            key, sk = jax.random.split(key)
            dat, pm = step(params, dat, sk, w)
        E.append(np.real(np.asarray(lef(params, jax.random.split(key, nwalk), dat)[0])))
    E = np.concatenate(E)
    lo, hi = np.percentile(E, [2, 98]); Ec = E[(E > lo) & (E < hi)]
    # ブロック平均で自己相関を粗く見込んだ誤差
    nb = max(4, len(Ec) // 64)
    blk = np.array_split(Ec, nb)
    err = float(np.std([b.mean() for b in blk], ddof=1) / np.sqrt(nb))
    return float(Ec.mean()), err, int(len(E))


def run(folder, nconf=384, nwalk=64, nw=12, seed=0, rmin=0.45):
    net, params, le, data, cfg = build_signed(folder)
    L = MA.load(folder)["L"]
    dmax = float(np.linalg.norm(L[:, 0]) / np.sqrt(3))
    Rs = list(np.geomspace(rmin, dmax * 0.99, nw))
    out = []
    CS.CLIP_R = None
    E0, e0, n0 = _measure(net, params, le, data, nconf, nwalk, seed)
    print("  切らない          E = %9.3f ± %.3f" % (E0, e0))
    # 対照: セルより広い R では襟が恒等的に 1 なので、E0 と一致しなければならない
    CS.CLIP_R = float(MA.euclid_to_w(dmax, L)) * 3.0
    Ec_, ec_, _ = _measure(net, params, le, data, nconf, nwalk, seed)
    print("  R = セル超（対照） E = %9.3f ± %.3f   ΔE = %+8.3f  ← 0 になるべき"
          % (Ec_, ec_, Ec_ - E0))
    for R in Rs:
        CS.CLIP_R = float(MA.euclid_to_w(R, L)) + 1e-9
        E, e, n = _measure(net, params, le, data, nconf, nwalk, seed)
        out.append(dict(R=float(R), E=E, Eerr=e, n=n, dE=E - E0))
        print("  R = %5.2f  E = %9.3f ± %.3f   ΔE = %+8.3f" % (R, E, e, E - E0))
    CS.CLIP_R = None
    res = dict(folder=folder, E0=E0, E0err=e0, dmax=dmax, points=out,
               control=dict(E=Ec_, Eerr=ec_, dE=Ec_ - E0))
    json.dump(res, open(os.path.join(folder, "window_ablation.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=384)
    ap.add_argument("--nwalk", type=int, default=64)
    ap.add_argument("--nw", type=int, default=12)
    a = ap.parse_args()
    run(a.folder, a.nconf, a.nwalk, a.nw)
