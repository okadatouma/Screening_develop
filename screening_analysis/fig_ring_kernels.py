# -*- coding: utf-8 -*-
"""輪切りの窓を核で比べる図：いまの輪切り（tau = 1）、raised cosine 核（幅 2 Delta）、3 次 B スプライン。
上：8 本の窓、中：c_h = h（直線）のときの f''、下：c_h = cos(2 pi x_h / a) のときの f''（破線は真の値）。
輪は両側に続くとした（端の扱いは除く）。全体の窓 g は 1。

    python screening_analysis/fig_ring_kernels.py [--out DIR]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt
from check_ring_kernels import DREF, A, DL, kernels, step

COLS = [("いまの輪切り（$\\tau=1$）", "halfsine tau=1", S.GREEN),
        ("raised cosine 核（幅 $2\\Delta$）", "raised-cos 2D", S.BLUE),
        ("3 次 B スプライン", "B2 (cubic B)", S.PURPLE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "doc"))
    a = ap.parse_args(); S.use()
    plt.rcParams.update({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8.5})
    d = np.linspace(0, DREF, 16001)
    hs = np.arange(-4, 12)                                   # 輪は両側に続くとする
    xc = (hs + 0.5) * DL
    ks = kernels()
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 4.1), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.928, bottom=0.108, hspace=0.2, wspace=0.25)
    for c, (title, key, col) in enumerate(COLS):
        St = step(ks[key])
        m = np.array([St(d - h * DL) - St(d - (h + 1) * DL) for h in hs])
        m2 = np.gradient(np.gradient(m, d, axis=-1), d, axis=-1)
        ax0, ax1, ax2 = axs[:, c]
        ax0.set_title(title, fontsize=9.5, pad=4)
        for i, h in enumerate(hs):
            if 0 <= h < 8:
                ax0.plot(d, m[i], color=col, lw=1.6 if h == 3 else 0.7, alpha=1.0 if h == 3 else 0.45)
        f2_lin = (hs[:, None] * m2).sum(0)
        f2_cos = (np.cos(2 * np.pi * xc / A)[:, None] * m2).sum(0)
        ax1.plot(d, f2_lin, color=col, lw=1.0)
        ax2.plot(d, -(2 * np.pi / A) ** 2 * np.cos(2 * np.pi * d / A), color=S.SUB, lw=0.9, ls="--")
        ax2.plot(d, f2_cos, color=col, lw=1.0)
        ax0.set_ylim(-0.05, 1.1); ax0.set_yticks([0, 0.5, 1])
        for ax in (ax1, ax2):
            ax.set_ylim(-42, 42); ax.set_yticks([-30, 0, 30])
        for ax in axs[:, c]:
            ax.axhline(0, color=S.LINE, lw=0.5, zorder=0)
            ax.set_xlim(0, DREF)
        for ax in axs[1:, c]:
            for h in range(1, 8):
                ax.axvline(h * DL, color=S.LINE, lw=0.4, zorder=0)
    axs[0, 0].set_ylabel("窓 $m_h(d)$")
    axs[1, 0].set_ylabel("$f''$（$c_h=h$）")
    axs[2, 0].set_ylabel("$f''$（$c_h=\\cos$）")
    for ax in axs[2]:
        ax.set_xlabel("相手までの距離 $d$")
    fig.savefig(os.path.join(a.out, "fig_ring_kernels.pdf")); plt.close(fig)
    print("wrote fig_ring_kernels.pdf")


if __name__ == "__main__":
    main()
