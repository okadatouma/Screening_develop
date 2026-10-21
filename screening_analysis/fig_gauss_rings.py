# -*- coding: utf-8 -*-
"""ガウスでぼかした輪切りの図（doc/gauss_rings.pdf）。

(a) N = 25 のセルでの 8 本の窓 m_h = g (T_h - T_{h+1})（sigma = 輪の幅の半分）と、その和 g
(b) 乱数の網（N = 9、4 本の輪）で電子 1 個を直線上で動かしたときの E_L（運動エネルギーだけ）。
    上：いまの輪切り tau = 1、下：ガウス核。上の縦の目盛りは、相手までの距離が輪の中心を横切る時刻
    （tau = 1 では窓の 2 階微分がそこで跳ぶ）。

    JAX_PLATFORMS=cpu PYTHONPATH=. python screening_analysis/fig_gauss_rings.py --compute   （jax のある環境で E_L を計算）
    python screening_analysis/fig_gauss_rings.py [--out DIR]                            （matplotlib のある環境で描く）
"""
import argparse, os, sys
import numpy as np
from scipy.special import ndtr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
NPZ = os.path.join(HERE, "out_gauss_rings", "path.npz")

DREF25, H25 = 3.2152, 8
SEED_PATH = 19                                   # 節（psi = 0）の近くを通らない道（test で探したもの）


def windows25(d, sf=0.5):
    D = DREF25 / H25; sig = sf * D
    b = np.arange(H25 + 1) * D
    T = [np.ones_like(d)] + [ndtr((d - b[k]) / sig) + ndtr((-d - b[k]) / sig) for k in range(1, H25)] + [np.zeros_like(d)]
    g = 0.5 * (1 + np.cos(np.pi * np.clip(d / DREF25, 0, 1) ** 3))
    return np.array([g * (T[h] - T[h + 1]) for h in range(H25)]), g


def path_energies():
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import test_gauss_rings as T
    from periodicwave import CustomScreenformer as CS
    lat = T.cell(3); n = 9; dref = CS._max_periodic_distance(np.asarray(lat))
    rng = np.random.default_rng(100 + SEED_PATH)
    x0 = rng.uniform(-2, 2, size=2 * n); ang = rng.uniform(0, 2 * np.pi)
    dirn = np.zeros(2 * n); dirn[0:2] = [np.cos(ang), np.sin(ang)]
    t = np.linspace(0.0, 1.2, 2401)
    X = x0[None] + t[:, None] * dirn[None]
    E = {}
    for name, rings in (("sin", T.rings_tuple(dref, 4, 'sin', 1.0)), ("gauss", T.rings_tuple(dref, 4, 'gauss', 0.5))):
        net, params = T.build_net(CS, lat, n, rings, heads=4, seed=11)
        el = jax.jit(jax.vmap(lambda x: T.kinetic(net.apply, lat, 'folx')(params, None, T.data_of(x, n))[0].real))
        E[name] = np.asarray(el(jnp.asarray(X)))
    # 電子 0 と相手 j の周期ノルム（networks.make_periodic_construct_input_features と同じ式）
    L = np.asarray(lat); M = L.T @ L; Linv = np.linalg.inv(L)
    r = X.reshape(len(t), n, 2)
    ee = r[:, 1:, :] - r[:, :1, :]
    s = np.einsum('il,tjl->tji', Linv, ee)
    a_, b_ = 1 - np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)
    dist = np.sqrt(np.einsum('tjm,mn,tjn->tj', a_, M, a_) + np.einsum('tjm,mn,tjn->tj', b_, M, b_)) / (2 * np.pi)
    D = dref / 4; centres = (np.arange(4) + 0.5) * D
    cross = []
    for c in centres:
        sgn = np.sign(dist - c)
        k, j = np.nonzero(np.diff(sgn, axis=0))
        cross += list(t[k])
    return t, E, np.array(sorted(cross))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "doc"))
    ap.add_argument("--compute", action="store_true")
    a = ap.parse_args()
    if a.compute:
        t, E, cross = path_energies()
        os.makedirs(os.path.dirname(NPZ), exist_ok=True)
        np.savez(NPZ, t=t, sin=E["sin"], gauss=E["gauss"], cross=cross)
        print("wrote", NPZ); return
    import style as S
    import matplotlib.pyplot as plt
    S.use()
    plt.rcParams.update({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8.5})
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], left=0.075, right=0.985, top=0.9, bottom=0.16,
                          wspace=0.28, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    d = np.linspace(0, DREF25, 4001)
    m, g = windows25(d)
    for h in range(H25):
        ax.plot(d, m[h], color=S.BLUE, lw=1.6 if h == 4 else 0.8, alpha=1.0 if h == 4 else 0.5)
    ax.plot(d, m.sum(0), color=S.INK, lw=0.9, ls="--")
    ax.text(0.15, 1.04, "窓の和 $=g$（破線）", fontsize=7.8, color=S.INK)
    ax.set_xlim(0, DREF25); ax.set_ylim(-0.03, 1.13)
    ax.set_xlabel("相手までの距離 $d$"); ax.set_ylabel("窓 $m_h(d)$")
    ax.set_title("(a) ガウス核の輪切り（$\\sigma=\\Delta/2$）", fontsize=9.2, pad=4)

    z = np.load(NPZ); t, cross = z["t"], z["cross"]; E = {"sin": z["sin"], "gauss": z["gauss"]}
    j = np.abs(np.diff(E["sin"]))
    print(f"tau = 1：|差| > 0.5 の跳びの時刻 {np.round(t[1:][j > 0.5], 4)}")
    print(f"輪の中心を横切る時刻 {np.round(cross, 4)}")
    print(f"ガウス核：隣の点の差の最大 {np.abs(np.diff(E['gauss'])).max():.4f}")
    a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a1.plot(t, E["sin"], color=S.GREEN, lw=1.0)
    a2.plot(t, E["gauss"], color=S.BLUE, lw=1.0)
    for x in cross:
        a1.axvline(x, color=S.LINE, lw=0.6, zorder=0)
    a1.text(0.02, 0.84, "いまの輪切り（$\\tau=1$）", transform=a1.transAxes, fontsize=7.8, color=S.GREEN)
    a2.text(0.02, 0.84, "ガウス核（$\\sigma=\\Delta/2$）", transform=a2.transAxes, fontsize=7.8, color=S.BLUE)
    a1.set_title("(b) 電子 1 個を動かしたときの $E_L$（運動エネルギー）", fontsize=9.2, pad=4)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.set_xlabel("動かした距離 $t$"); a1.set_xlim(0, t[-1])
    for axx in (a1, a2):
        lo, hi = np.percentile(E["sin"] if axx is a1 else E["gauss"], [0, 100])
        pad = 0.12 * (hi - lo); axx.set_ylim(lo - pad, hi + 2.2 * pad)
    fig.savefig(os.path.join(a.out, "fig_gauss_rings.pdf")); plt.close(fig)
    print("wrote fig_gauss_rings.pdf")


if __name__ == "__main__":
    main()
