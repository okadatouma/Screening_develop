# -*- coding: utf-8 -*-
"""輪切り窓・V 固定の GPU run の図（学習曲線と輪ごとの読み出し）。

    python screening_analysis/fig_rings.py <CustomScreenformer の結果フォルダ> [--out DIR]

先に ring_results.py を回して DIR/readouts.npz を作っておくこと。
"""
import argparse, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S
import ring_results as RR
import matplotlib.pyplot as plt

DREF = 3.2152
SHELLS = [(1.78, "第1近接"), (2.69, "第2"), (2.88, "第3")]
RUNSTY = {
    "s": dict(label="入れ子窓・V 学習（16 ヘッド）", color=S.INK, ls="-", lw=1.5),
    "t": dict(label="殻窓・V 学習（16 ヘッド）", color="#a3a19b", ls=":", lw=1.8),
    "e": dict(label="輪切り τ = 0.5・V 固定", color=S.ORANGE, ls="-", lw=1.5),
    "v": dict(label="輪切り τ = 1・V 固定", color=S.BLUE, ls="-", lw=1.5),
    "w": dict(label="輪切り τ = 0.5・V 学習", color=S.GREEN, ls="--", lw=1.5),
}
PHCOL = {"l05": "#9cc3ee", "b30": "#4a86c8", "c60": "#1b3a6b"}
PHNAME = {"l05": "液体", "b30": "境界", "c60": "結晶"}


def fig_train(root, out, cap=True):
    fig, axs = plt.subplots(2, 3, figsize=(11.0, 6.9 if cap else 5.9), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855 if cap else 0.83,
                        bottom=0.215 if cap else 0.085, hspace=0.12, wspace=0.26)
    for c, (ph, rs, name) in enumerate(RR.PHASES):
        fins = []
        for key in ["s", "t", "e", "v", "w"]:
            a = RR.load_stats(RR.folder(root, key, ph, rs))
            if a is None:
                continue
            a = a[a["step"] < 6000]
            st = pd.Series(a["energy"]).rolling(100, center=True, min_periods=20).mean()
            ss = pd.Series(a["locstd"]).rolling(100, center=True, min_periods=20).median()
            sty = RUNSTY[key]
            axs[0, c].plot(a["step"], st, color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                           label=sty["label"])
            axs[1, c].plot(a["step"], ss, color=sty["color"], ls=sty["ls"], lw=sty["lw"])
            fins.append(RR.final(a)[0])
        lo, hi = min(fins), max(fins); sp = hi - lo
        axs[0, c].set_ylim(lo - 0.25 * sp, hi + 0.9 * sp)
        axs[0, c].set_title(f"{name}（$r_s = {rs}$）")
        axs[1, c].set_yscale("log"); axs[1, c].set_ylim(3, 1500)
        axs[1, c].set_xlabel("step")
        axs[1, c].set_xlim(0, 6000)
        for r in range(2):
            axs[r, c].grid(True, color=S.LINE, lw=0.5, alpha=0.7)
    axs[0, 0].set_ylabel("$E$")
    axs[1, 0].set_ylabel(r"$\sigma(E_L)$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=9.6, bbox_to_anchor=(0.53, 0.995),
               handlelength=2.6, columnspacing=1.6)
    txt = ("$E$：25 電子の全エネルギー（コードのエネルギー単位、100 step の移動平均）。"
           r"$\sigma(E_L)$：局所エネルギー $E_L = H\psi/\psi$ のウォーカー間の標準偏差（100 step の移動中央値）。"
           r"$\psi$ が固有状態なら $\sigma(E_L)=0$。step：変分最適化の反復回数。"
           "入れ子窓の run は 8000 step まで回っているが、6000 step で切って並べた。"
           r"$\tau$：隣の輪へ移る遷移帯が輪の幅に占める割合（$\tau$ が小さいほど縁が鋭い）。")
    if cap:
        S.caption(fig, "図 1　学習曲線", S.wrap(txt, 150), y=0.012, size=8.9)
    fig.savefig(os.path.join(out, "fig_ring_train.png" if cap else "fig_ring_train_nocap.png"), dpi=200)
    plt.close(fig)


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def fig_readout(out, cap=True):
    z = np.load(os.path.join(out, "readouts.npz"))
    edges = np.linspace(0, DREF, 9); mid = 0.5 * (edges[1:] + edges[:-1])
    fig = plt.figure(figsize=(12.4, 5.0 if cap else 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.72], left=0.055, right=0.985,
                          top=0.805 if cap else 0.76, bottom=0.31 if cap else 0.16, wspace=0.28)
    runs = [("e", "輪切り τ = 0.5・V 固定"), ("v", "輪切り τ = 1・V 固定"), ("w", "輪切り τ = 0.5・V 学習")]
    TV = {}
    for i, (key, title) in enumerate(runs):
        ax = fig.add_subplot(gs[0, i])
        for b in edges:
            ax.axvline(b, color=S.LINE, lw=0.6, zorder=0)
        for d0, nm in SHELLS:
            ax.axvline(d0, color=S.SUB, lw=0.9, ls=(0, (2, 2)), zorder=1)
        for j, (ph, off) in enumerate(zip(["l05", "b30", "c60"], [-0.11, 0.0, 0.11])):
            qc = z[f"{key}_{ph}_qc"]; qa = z[f"{key}_{ph}_qa"]
            ax.bar(mid + off, qc, width=0.105, color=PHCOL[ph], label=PHNAME[ph], zorder=2)
            ax.plot(mid + off, qa, ls="none", marker="_", ms=7.5, mew=1.6, color=S.INK, zorder=3,
                    label=("$q_h^{(a)}$（幾何）" if j == 0 else None))
        TV[key] = (tv(z[f"{key}_l05_qa"], z[f"{key}_c60_qa"]), tv(z[f"{key}_l05_qc"], z[f"{key}_c60_qc"]))
        ax.set_xlim(0, DREF); ax.set_ylim(0, 0.47)
        ax.set_xticks(np.round(mid, 1))
        ax.set_xticklabels([f"{m:.1f}" for m in mid], fontsize=8.6)
        ax.set_xlabel("輪の中心の距離 $d$")
        ax.set_title(title, fontsize=10.8)
        for d0, nm in SHELLS:
            ax.text(d0, 0.462, {"第1近接": "1", "第2": "2", "第3": "3"}[nm], ha="center", va="top",
                    fontsize=8.4, color=S.SUB)
        if i == 0:
            ax.set_ylabel("$q_h^{(c)}$")
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.6, bbox_to_anchor=(0.43, 0.985))
    ax = fig.add_subplot(gs[0, 3])
    xs = np.arange(3)
    ga = [TV[k][0] for k, _ in runs]; gc = [TV[k][1] for k, _ in runs]
    ax.bar(xs - 0.19, ga, width=0.36, color="white", edgecolor=S.INK, hatch="///", lw=0.8,
           label="$q^{(a)}$（幾何）")
    ax.bar(xs + 0.19, gc, width=0.36, color=S.INK, label="$q^{(c)}$")
    for x, v in zip(xs - 0.19, ga):
        ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4)
    for x, v in zip(xs + 0.19, gc):
        ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4)
    ax.set_xticks(xs); ax.set_xticklabels(["τ=0.5\nV 固定", "τ=1\nV 固定", "τ=0.5\nV 学習"], fontsize=8.8)
    ax.set_ylim(0, 0.4); ax.set_title("液体と結晶の差", fontsize=10.8)
    ax.set_ylabel("全変動距離")
    ax.legend(loc="upper left", fontsize=8.6)
    txt = (r"$q_h^{(c)}$：ヘッド $h$ が相手の電子から集めて波動関数に足した量の 2 乗の割合。"
           r"横線 $q_h^{(a)}$：窓で集めた直後に測った同じ割合（相手がどの距離にいるかでほぼ決まる）。"
           r"輪 $h$ は $b_h \leq d \leq b_{h+1}$、$b_h = h\,d_{\rm ref}/8$、$d_{\rm ref} = 3.215$（電子どうしの距離 $d$ の最大値、周期ノルム）。"
           "点線 1〜3：三角格子に並んだときの第 1〜3 近接の距離。"
           r"全変動距離 $= \frac{1}{2}\sum_h |q_h^{\rm L} - q_h^{\rm C}|$（L：液体、C：結晶。0 なら同じ分布）。")
    if cap:
        S.caption(fig, "図 2　輪ごとの読み出し", S.wrap(txt, 236), y=0.05, size=8.9)
    fig.savefig(os.path.join(out, "fig_ring_readout.png" if cap else "fig_ring_readout_nocap.png"), dpi=200)
    plt.close(fig)
    return TV


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(HERE, "out_rings"))
    a = ap.parse_args(); S.use()
    fig_train(a.root, a.out); fig_train(a.root, a.out, cap=False)
    TV = fig_readout(a.out); fig_readout(a.out, cap=False)
    for k, (ta, tc) in TV.items():
        print(f"{k}: TV liquid-crystal  a {ta:.3f}  c {tc:.3f}")


if __name__ == "__main__":
    main()
