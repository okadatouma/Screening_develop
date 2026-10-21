# -*- coding: utf-8 -*-
"""窓 m(d) とその 1 階・2 階微分：入れ子窓・輪切り（tau = 0.5）・輪切り（tau = 0）を並べる。
あわせて、縁の曲がり pi^2/(2 l^2) と局所エネルギーのばらつき sigma(E_L) の関係（N = 25 の GPU run）。

    python screening_analysis/fig_window_derivs.py [--out DIR]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt

DREF = 3.2152                    # N = 25 のセルの最大距離（周期ノルム）
A = 1.9046                       # 三角格子の最近接距離（いちばん狭い入れ子窓の幅）
DR = DREF / 8                    # 輪の幅
C_NEST, C_RING, C_STEP = S.BLUE, S.ORANGE, S.ENJI


def curv(l):
    return np.pi ** 2 / (2 * l ** 2)


def nested(d, w):
    return 0.5 * (1 + np.cos(np.pi * np.minimum(d / w, 1.0)))


def theta(u, l):
    """幅 l で 0 から 1 へ上がる階段（l = 0 なら段差）。"""
    if l == 0:
        return (u >= 0).astype(float)
    return 0.5 * (1 + np.sin(np.pi * np.clip(u / l, -0.5, 0.5)))


def ring(d, h, tau):
    g = 0.5 * (1 + np.cos(np.pi * (d / DREF) ** 3))
    return g * (theta(d - h * DR, tau * DR) - theta(d - (h + 1) * DR, tau * DR))


def derivs(y, d):
    y1 = np.gradient(y, d)
    return y1, np.gradient(y1, d)


def span(ax, x0, x1, y, text, col):
    """幅を示す寸法線と、その下のラベル。"""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="|-|, widthA=0.22, widthB=0.22", color=col, lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.text(0.5 * (x0 + x1), y - 0.17, text, fontsize=8, color=col, ha="center", va="center")


def fig_derivs(out):
    d = np.linspace(0, DREF, 40001)
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 4.9), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.935, bottom=0.095, hspace=0.2, wspace=0.28)
    b0, b1 = 4 * DR, 5 * DR
    lr = 0.5 * DR                                 # tau = 0.5 の遷移の幅
    cols = [("入れ子窓（$w=a=1.905$）", nested(d, A), C_NEST),
            ("輪切り $\\tau=0.5$（輪 1.61–2.01）", ring(d, 4, 0.5), C_RING),
            ("輪切り $\\tau=0$（同じ輪）", ring(d, 4, 0.0), C_STEP)]
    for c, (title, m, col) in enumerate(cols):
        m1, m2 = derivs(m, d)
        ax0, ax1, ax2 = axs[:, c]
        ax0.set_title(title, fontsize=9.5, pad=4)
        if c < 2:
            ax0.plot(d, m, color=col, lw=1.5)
            ax1.plot(d, m1, color=col, lw=1.2)
            ax2.plot(d, m2, color=col, lw=1.0)
        else:                                  # 段差：値は跳び、微分は δ 関数、2 階微分は δ'
            for mask in (d < b0, (d >= b0) & (d < b1), d >= b1):
                ax0.plot(d[mask], m[mask], color=col, lw=1.5)
            for b in (b0, b1):
                top = m[np.searchsorted(d, b0)] if b == b0 else m[np.searchsorted(d, b1) - 1]
                ax0.plot([b, b], [0, top], ls=":", color=col, lw=1.0)
            ax1.axhline(0, color=col, lw=1.2)
            for b, sgn in ((b0, 1), (b1, -1)):          # m' = (跳び) x δ(d - b)
                ax1.annotate("", xy=(b, sgn * 6.5), xytext=(b, 0),
                             arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=8))
            ax1.text(b1 + 0.12, 4.5, "$\\delta$ 関数", fontsize=8, color=col)
            ax2.axhline(0, color=col, lw=1.0)
            # m'' = (跳び) x δ'(d - b)：上がる段差は左で +、右で −。下がる段差は逆。
            for b, jump in ((b0, 1), (b1, -1)):
                for side in (-1, 1):
                    ax2.annotate("", xy=(b + 0.025 * side, -side * jump * 115), xytext=(b + 0.025 * side, 0),
                                 arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7))
            ax2.text(b0 - 0.1, 60, "$\\delta'$（$\\pm$無限大）", fontsize=8, color=col, ha="right")
        ax0.set_ylim(-0.5, 1.12); ax0.set_yticks([0, 0.5, 1])
        ax1.set_ylim(-8, 8); ax2.set_ylim(-135, 135)
        for ax in axs[:, c]:
            ax.axhline(0, color=S.LINE, lw=0.5, zorder=0)
            ax.set_xlim(0, DREF)
    # 遷移の幅 l
    span(axs[0, 0], 0, A, -0.2, "$\\ell=w$", C_NEST)
    span(axs[0, 1], b0 - lr / 2, b0 + lr / 2, -0.2, "$\\ell$", C_RING)
    axs[0, 2].text(b0, -0.37, "$\\ell=0$（段差）", fontsize=8, color=C_STEP, ha="center", va="center")
    # 2 階微分の目安 ±pi^2/(2 l^2)
    ax = axs[2, 1]
    for s in (1, -1):
        ax.axhline(s * curv(lr), color=C_RING, lw=0.7, ls="--")
    ax.text(0.08, 96, f"$\\pi^2/(2\\ell^2)={curv(lr):.0f}$", fontsize=7.8, color=C_RING, va="center")
    ax = axs[2, 0]
    ax.text(0.08, -88, f"最大 $\\pi^2/(2w^2)={curv(A):.2f}$\n（この目盛りでは 0 に見える）",
            fontsize=7.5, color=C_NEST, va="center", linespacing=1.3)
    ins = ax.inset_axes([0.42, 0.5, 0.54, 0.34])
    mn2 = derivs(nested(d, A), d)[1]
    ins.plot(d, mn2, color=C_NEST, lw=1.0)
    for s in (1, -1):
        ins.axhline(s * curv(A), color=C_NEST, lw=0.6, ls="--")
    ins.axhline(0, color=S.LINE, lw=0.5, zorder=0)
    ins.set_xlim(0, DREF); ins.set_ylim(-1.9, 1.9)
    ins.set_yticks([-1, 0, 1]); ins.set_xticks([0, 1, 2, 3]); ins.set_xticklabels([])
    ins.tick_params(labelsize=6, length=2, pad=1, width=0.6)
    for sp in ins.spines.values():
        sp.set_linewidth(0.6)
    ins.set_title("縦を拡大", fontsize=6.8, pad=1.5, color=C_NEST)
    axs[0, 0].set_ylabel("$m(d)$"); axs[1, 0].set_ylabel("$m'(d)$"); axs[2, 0].set_ylabel("$m''(d)$")
    for ax in axs[2]:
        ax.set_xlabel("相手までの距離 $d$")
    fig.savefig(os.path.join(out, "fig_window_derivs.pdf")); plt.close(fig)
    print("wrote fig_window_derivs.pdf")


def fig_sigma(out):
    # N = 25、U = 0、6000 step の run（rings-gpu-results の表）。横軸は縁の曲がりの目安 pi^2/(2 l^2)。
    rows = [("入れ子窓", curv(A), 6.4, 7.0, 8.8, C_NEST, (1.0, 12.5, "left")),
            ("輪切り $\\tau=1$", curv(DR), 14.5, 43.9, 44.4, S.GREEN, (curv(DR), 63, "center")),
            ("輪切り $\\tau=0.5$", curv(0.5 * DR), 20.3, 61.6, 88.8, C_RING, (curv(0.5 * DR), 127, "center")),
            ("殻窓", curv(0.114), 27.0, 99.6, 124.1, S.INK, (curv(0.114), 178, "center"))]
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.965, bottom=0.19)
    for name, mm, s5, s30, s60, col, (tx, ty, ha) in rows:
        ax.plot([mm], [s5], "^", ms=5.0, mfc="white", mec=col, mew=1.1)
        ax.plot([mm], [s30], "o", ms=5.3, color=col)
        ax.plot([mm], [s60], "s", ms=5.0, mfc="white", mec=col, mew=1.2)
        ax.text(tx, ty, name, fontsize=7.8, color=col, ha=ha, va="center")
    ax.plot([], [], "^", mfc="white", mec=S.SUB, label="液体（$r_s=5$）")
    ax.plot([], [], "o", color=S.SUB, label="境界（$r_s=30$）")
    ax.plot([], [], "s", mfc="white", mec=S.SUB, label="結晶（$r_s=60$）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, 1500); ax.set_ylim(4, 260)
    ax.set_xlabel("縁の曲がり $\\pi^2/(2\\ell^2)$")
    ax.set_ylabel("$\\sigma(E_L)$")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5, handletextpad=0.3, borderaxespad=0.3)
    fig.savefig(os.path.join(out, "fig_sigma_vs_curv.pdf")); plt.close(fig)
    print("wrote fig_sigma_vs_curv.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "doc"))
    a = ap.parse_args(); S.use()
    plt.rcParams.update({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8.5})
    fig_derivs(a.out); fig_sigma(a.out)


if __name__ == "__main__":
    main()
