# -*- coding: utf-8 -*-
"""λ-jellium の検算：Coulomb を切った N = 4 の変分最適化が厳密な自由粒子の答えに近づくか。

    python screening_analysis/fig_lj_free.py [--out PATH]

データ：out_lj/free_lam{0,0.72}_train_stats.csv
  （run_lambda_jellium.py 4 10 lam 800、PW_COULOMB=0、2x2 セル、ヘッド 4、行列式 1、ウォーカー 128、CPU）
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt

E0 = 5.4414                       # 3 * (1/2)|b|^2、|b| = a = 1.9046（下のバンドは lam によらず q^2/2）
A2 = 2 * np.pi / np.sqrt(3)       # |b|^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "doc", "fig_lj_free_nocap.png"))
    a = ap.parse_args(); S.use()
    runs = [("0", "$\\lambda=0$", S.BLUE), ("0.72", "$\\lambda=0.72$", S.ORANGE)]
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 2.9))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.89, bottom=0.19, wspace=0.3)
    for lam, lab, col in runs:
        d = pd.read_csv(os.path.join(HERE, "out_lj", f"free_lam{lam}_train_stats.csv"))
        r = lambda v, k=25: pd.Series(v).rolling(k, center=True, min_periods=5).mean()
        axs[0].plot(d.step, r(d.energy), color=col, lw=1.4, label=lab)
        axs[1].plot(d.step, pd.Series(d.locstd).rolling(25, center=True, min_periods=5).median(), color=col, lw=1.4)
        axs[2].plot(d.step, r(d.nminus), color=col, lw=1.4)
    axs[0].axhline(E0, color=S.INK, lw=0.9, ls=(0, (4, 3)))
    axs[0].text(790, E0 - 0.03, "厳密 5.441", ha="right", va="top", fontsize=9, color=S.INK)
    axs[0].set_ylim(5.3, 6.6); axs[0].set_ylabel("$E$"); axs[0].legend(loc="upper right", fontsize=9.5)
    axs[1].set_yscale("log"); axs[1].set_ylim(0.05, 30); axs[1].set_ylabel(r"$\sigma(E_L)$")
    nm = 3 * 0.72 ** 2 * A2 / (1 + 0.72 ** 2 * A2)
    axs[2].axhline(nm, color=S.INK, lw=0.9, ls=(0, (4, 3)))
    axs[2].text(790, nm + 0.08, f"厳密 {nm:.3f}", ha="right", va="bottom", fontsize=9, color=S.INK)
    axs[2].set_ylim(-0.1, 2.6); axs[2].set_ylabel("$\\langle n_-\\rangle$")
    for ax, t in zip(axs, ["(a) エネルギー", "(b) 局所エネルギーのばらつき", "(c) 下成分にいる電子の数"]):
        ax.set_xlim(0, 800); ax.set_xlabel("step"); ax.set_title(t, loc="left", fontsize=10.5)
        ax.grid(True, color=S.LINE, lw=0.5, alpha=0.7)
    fig.savefig(a.out, dpi=200); print("wrote", a.out)


if __name__ == "__main__":
    main()
