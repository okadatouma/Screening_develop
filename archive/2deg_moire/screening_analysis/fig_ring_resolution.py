# -*- coding: utf-8 -*-
"""なめらかな窓でも液体と結晶の違いが見えるか（doc/gauss_rings.pdf の図 2）。

(a) 対距離の分布の差（結晶 r_s = 60 − 液体 r_s = 5、0.05 刻み、入れ子窓の run 1929 のウォーカー）
(b) 窓だけで集めたヘッドごとの割合 p_h の差（同じウォーカー）。いまの輪切り tau = 1 と、ガウス核 sigma = 0.5, 0.3 Delta。
    誤差棒は 128 本ずつ 8 組のばらつきから出した標準誤差。

    python screening_analysis/check_ring_readout.py      （先に数値を作る）
    python screening_analysis/fig_ring_resolution.py [--out DIR]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt

DREF, H = 3.2152, 8
DL = DREF / H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "doc"))
    a = ap.parse_args(); S.use()
    plt.rcParams.update({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8.5})
    z = np.load(os.path.join(HERE, "out_gauss_rings", "readout.npz"))
    L, C = "液体 r_s=5", "結晶 r_s=60"
    hl, hc = z[f"hist|{L}"].astype(float), z[f"hist|{C}"].astype(float)
    edges = np.linspace(0, DREF, 65); mid = 0.5 * (edges[1:] + edges[:-1])
    dn = 100 * (hc / hc.sum() - hl / hl.sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.45), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.2, wspace=0.25)
    a1.bar(mid, np.clip(dn, -1.6, 1.6), width=edges[1] - edges[0], color=S.SUB, alpha=0.75)
    big = np.abs(dn) > 1.6                              # セルの端（d_ref 付近）の山は目盛りの外
    a1.text(0.06, 1.62, "$d_{\\rm ref}$ 近くの 2 本は目盛りの外（" + "、".join(f"{v:+.1f}" for v in dn[big]) + "）",
            fontsize=7.2, color=S.SUB, ha="left", va="top")
    a1.set_ylim(-1.75, 1.75)
    a1.set_ylabel("差（%、0.05 刻み）")
    a1.set_title("(a) 対距離の分布：結晶 − 液体", fontsize=9.2, pad=4)
    xc = (np.arange(H) + 0.5) * DL
    sets = [("いまの輪切り $\\tau=1$", "いま tau=1", S.GREEN, -0.1),
            ("ガウス $\\sigma=0.5\\Delta$", "ガウス sigma=0.5D", S.BLUE, 0.0),
            ("ガウス $\\sigma=0.3\\Delta$", "ガウス sigma=0.3D", S.PURPLE, 0.1)]
    for lab, key, col, off in sets:                    # 輪ごとに 3 本の棒（どれも同じ輪 h の値）
        dp = 100 * (z[f"{key}|{C}"] - z[f"{key}|{L}"])
        er = 100 * np.hypot(z[f"err|{key}|{C}"], z[f"err|{key}|{L}"])
        a2.bar(xc + off, dp, width=0.095, color=col, label=lab)
        a2.errorbar(xc + off, dp, yerr=er, fmt="none", ecolor=S.INK, elinewidth=0.7, capsize=1.2)
    a2.set_ylabel("$p_h$ の差（%）")
    a2.set_title("(b) 窓で集めたヘッドごとの割合：結晶 − 液体", fontsize=9.2, pad=4)
    a2.legend(frameon=False, fontsize=7.3, loc="upper left", handlelength=1.2)
    a2.set_ylim(-7, 10.5)
    for ax in (a1, a2):
        ax.axhline(0, color=S.LINE, lw=0.6, zorder=0)
        for h in range(1, H):
            ax.axvline(h * DL, color=S.LINE, lw=0.4, zorder=0)
        ax.set_xlim(0, DREF); ax.set_xlabel("相手までの距離 $d$")
    fig.savefig(os.path.join(a.out, "fig_ring_resolution.pdf")); plt.close(fig)
    print("wrote fig_ring_resolution.pdf")


if __name__ == "__main__":
    main()
