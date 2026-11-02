# -*- coding: utf-8 -*-
"""擬スピン（2 次元電子ではスピン）で分けた動径分布 g(r)。"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt
from lj_maps import load, radial_g

A_NN = np.sqrt(2 * np.pi / np.sqrt(3))


def main(base, out, ref_c, ref_l, name="rep_gspin", label=""):
    S.use(); plt.rcParams.update({"font.size": 8})
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.19, wspace=0.07)
    cols = {0.33: "#2a78d6", 0.56: "#8a4ab0", 0.72: "#eb6834"}
    for ax, (w, wname) in zip(axs[:2], [("jn", "入れ子窓"), ("jg", "ガウスの輪")]):
        for lam, col in cols.items():
            f = glob.glob(os.path.join(base, f"N36_rs15_lam*_{w}{int(round(lam * 100)):02d}"))[0]
            cfg, z, pos, sp, L, ne, ck = load(f)
            r, gs = radial_g(pos, L, sp, same=True); r, go = radial_g(pos, L, sp, same=False)
            ax.plot(r, gs, "-", color=col, lw=1.2, label="$\\lambda$=%.2f" % lam)
            ax.plot(r, go, "--", color=col, lw=1.2)
        ax.set_title(wname + label, fontsize=8.6, pad=3)
    for f, nm, col in [(ref_l, "液体 $r_s$=5", "#5e6a73"), (ref_c, "結晶 $r_s$=60", "#141412")]:
        cfg, z, pos, sp, L, ne, ck = load(f)
        r, gs = radial_g(pos, L, sp, same=True); r, go = radial_g(pos, L, sp, same=False)
        axs[2].plot(r, gs, "-", color=col, lw=1.2, label=nm); axs[2].plot(r, go, "--", color=col, lw=1.2)
    axs[2].set_title("参考：2 次元電子（run 1929）", fontsize=8.6, pad=3)
    for ax in axs:
        ax.axhline(1, color=S.LINE, lw=0.6, zorder=0); ax.axvline(A_NN, color=S.LINE, lw=0.6, ls=":", zorder=0)
        ax.set_xlim(0, 4.0); ax.set_ylim(0, 2.1); ax.set_xlabel("距離 $r$")
        ax.legend(fontsize=6.8, loc="upper right", handlelength=1.6)
    axs[0].set_ylabel("$g(r)$")
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:5], *(sys.argv[5:7]))
