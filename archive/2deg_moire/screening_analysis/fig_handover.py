# -*- coding: utf-8 -*-
"""輪の境界で、相手の電子が内側のヘッドから外側のヘッドへどう受け渡されるか（τ = 0, 0.5, 1）。

    python screening_analysis/fig_handover.py [--out PATH]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt

DREF = 3.2152
D = DREF / 8
B = [2 * D, 3 * D, 4 * D]                 # 内側の輪 [0.804, 1.206]、外側の輪 [1.206, 1.608]


def g(d):
    return 0.5 * (1 + np.cos(np.pi * (d / DREF) ** 3))


def step(u, dl):
    if dl == 0:
        return (u >= 0).astype(float)
    return 0.5 * (1 + np.sin(0.5 * np.pi * np.clip(u / dl, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out_rings", "fig_handover.png"))
    a = ap.parse_args(); S.use()
    d = np.linspace(0.80, 1.61, 4000)
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 4.7), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.40, wspace=0.10)
    for ax, tau in zip(axs, [0.0, 0.5, 1.0]):
        dl = tau * D / 2
        th1, th2 = step(d - B[1], dl), step(d - B[2], dl)
        inner = g(d) * (step(d - B[0], dl) - th1)
        outer = g(d) * (th1 - th2)
        if tau == 0:                        # 段差は縦線で描かない（値が跳ぶことが見えるように途中で切る）
            m = d < B[1]
            ax.plot(d[m], inner[m], color=S.BLUE, lw=2.2, label="内側の輪のヘッド")
            ax.plot(d[~m], inner[~m], color=S.BLUE, lw=2.2)
            ax.plot(d[m], outer[m], color=S.ORANGE, lw=2.2, label="外側の輪のヘッド")
            ax.plot(d[~m], outer[~m], color=S.ORANGE, lw=2.2)
            ax.plot([B[1]], [g(B[1])], "o", ms=6, mfc="white", mec=S.BLUE, zorder=5)
            ax.plot([B[1]], [0], "o", ms=6, mfc="white", mec=S.ORANGE, zorder=5)
            ax.text(B[1] + 0.02, 0.45, "境界で値が跳ぶ\n→ $\\psi$ に段差", fontsize=9.6, color=S.ENJI)
        else:
            ax.plot(d, inner, color=S.BLUE, lw=2.2, label="内側の輪のヘッド")
            ax.plot(d, outer, color=S.ORANGE, lw=2.2, label="外側の輪のヘッド")
            ax.axvspan(B[1] - dl, B[1] + dl, color="#e8e6df", zorder=0, lw=0)
            ax.annotate("", xy=(B[1] - dl, 0.5), xytext=(B[1] + dl, 0.5), zorder=4,
                        arrowprops=dict(arrowstyle="<->", lw=0.9, color=S.INK, shrinkA=0, shrinkB=0))
            ax.text(B[1], 0.535, f"幅 {2 * dl:.1f} で受け渡す", ha="center", va="bottom", fontsize=9.4,
                    bbox=dict(fc="white", ec="none", pad=1.2), zorder=5)
        ax.plot(d, inner + outer, color=S.INK, lw=1.0, ls=(0, (4, 3)), label="2 つの和（全体窓 $g$）")
        ax.axvline(B[1], color=S.SUB, lw=0.8, ls=":")
        ax.set_title(f"τ = {tau:g}", fontsize=11.5)
        ax.set_xlim(0.80, 1.61); ax.set_ylim(-0.03, 1.1)
        ax.set_xlabel("相手の電子までの距離 $d$")
        ax.set_xticks([1.0, 1.206, 1.4]); ax.set_xticklabels(["1.0", "1.21\n（境界）", "1.4"])
    axs[0].set_ylabel("窓の値")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=9.8, bbox_to_anchor=(0.5, 0.995))
    txt = ("全体窓 $g(d)$ を 2 つの輪（$0.80$–$1.21$ と $1.21$–$1.61$）に切ったところを拡大した。"
           "破線の和はどの $\\tau$ でも $g$ で滑らか。違うのは 1 本 1 本の窓で、$\\tau=0$ では相手の電子が境界を"
           "またいだ瞬間に内側のヘッドから外側のヘッドへ飛び移る。2 つのヘッドは別々の重みで波動関数 $\\psi$ につながって"
           "いるので $\\psi$ が段差状に変わり、運動エネルギーが無限大になる。$\\tau>0$ では灰色の帯の中で少しずつ受け渡すので"
           "段差は消えるが、帯が狭いほど変化が急で、運動エネルギーの代償が大きい。")
    S.caption(fig, "図　輪の境界での受け渡し", S.wrap(txt, 176), y=0.02, size=9.0)
    fig.savefig(a.out, dpi=200); print("wrote", a.out)


if __name__ == "__main__":
    main()
