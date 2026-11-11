# -*- coding: utf-8 -*-
"""3 種類の窓（入れ子窓・殻窓・全体窓の輪切り）と τ の意味を 1 枚で見せる図。

    python screening_analysis/fig_window_types.py [--out PATH]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt

DREF = 3.2152
d = np.linspace(0, DREF, 3000)


def nested(w):
    return 0.5 * (1 + np.cos(np.pi * np.minimum(d / w, 1.0)))


def shell(c, s):
    return 0.5 * (1 + np.cos(np.pi * np.minimum(np.abs(d - c) / s, 1.0)))


def rings(H=8, tau=0.5, p=3.0):
    b = np.linspace(0, DREF, H + 1); dl = tau * (b[1] - b[0]) / 2
    th = [np.ones_like(d)] + [0.5 * (1 + np.sin(0.5 * np.pi * np.clip((d - bb) / dl, -1, 1))) for bb in b[1:-1]] \
         + [np.zeros_like(d)]
    g = 0.5 * (1 + np.cos(np.pi * (d / DREF) ** p))
    return [g * (th[h] - th[h + 1]) for h in range(H)], g, b, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out_rings", "fig_window_types.png"))
    a = ap.parse_args(); S.use()
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.9))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.215, hspace=0.66, wspace=0.16)
    cmap = plt.get_cmap("viridis")

    # (a) 入れ子窓
    ax = axs[0, 0]
    ws = 1.9046 * (3.2152 / 1.9046) ** (np.arange(16) / 15)
    for k, w in enumerate(ws):
        ax.plot(d, nested(w), color=cmap(k / 15), lw=1.1)
    ax.set_title("(a) 入れ子窓（ジョブ 1929）", loc="left", fontsize=11)
    for y, t, kw in [(0.88, r"$m_h(d)=\frac{1}{2}\,[1+\cos\pi t\,]$", dict(fontsize=10.5)),
                     (0.77, r"$t=\min(d/w_h,\ 1)$", dict(fontsize=10.5)),
                     (0.64, r"幅 $w_h=1.90$〜$3.22$（16 本）", dict(fontsize=9.3, color=S.SUB)),
                     (0.55, r"どれも $d=0$ で 1、$d=w_h$ で 0", dict(fontsize=9.3, color=S.SUB))]:
        ax.text(0.52, y, t, transform=ax.transAxes, va="center", **kw)

    # (b) 殻窓
    ax = axs[0, 1]
    cs = np.linspace(1.5, DREF, 16); sg = cs[1] - cs[0]
    for k, c in enumerate(cs):
        ax.plot(d, shell(c, sg), color=cmap(k / 15), lw=1.1)
    ax.set_title("(b) 殻窓（ジョブ 1931）", loc="left", fontsize=11)
    for y, t, kw in [(0.88, r"$m_h(d)=\frac{1}{2}\,[1+\cos\pi t\,]$", dict(fontsize=10.5)),
                     (0.77, r"$t=\min(|d-c_h|/\sigma,\ 1)$", dict(fontsize=10.5)),
                     (0.64, r"山の中心 $c_h=1.50$〜$3.22$", dict(fontsize=9.3, color=S.SUB)),
                     (0.55, r"（16 本）、半幅 $\sigma=0.114$", dict(fontsize=9.3, color=S.SUB)),
                     (0.40, r"$d<1.39$ の相手は", dict(fontsize=9.3, color=S.SUB)),
                     (0.31, "どのヘッドも見ない", dict(fontsize=9.3, color=S.SUB))]:
        ax.text(0.025, y, t, transform=ax.transAxes, va="center", **kw)

    # (c)(d) 輪切り
    for ax, tau, lab in [(axs[1, 0], 1.0, "(c) 全体窓の輪切り、τ = 1（ジョブ 2072）"),
                         (axs[1, 1], 0.5, "(d) 全体窓の輪切り、τ = 0.5（ジョブ 2071、2073）")]:
        ms, g, b, dl = rings(tau=tau)
        for bb in b:
            ax.axvline(bb, color=S.LINE, lw=0.7, zorder=0)
        for k, m in enumerate(ms):
            ax.plot(d, m, color=cmap(k / 7), lw=1.5)
        ax.plot(d, g, color=S.INK, lw=1.2, ls=(0, (4, 3)))
        ax.set_title(lab, loc="left", fontsize=11, pad=30)
        for bb in b[1:-1]:                       # 遷移帯 [b_h - δ, b_h + δ]
            ax.axvspan(bb - dl, bb + dl, color="#e8e6df", zorder=-1, lw=0)
        top = 1.075
        def arrow(x0, x1, text, color=S.INK):
            ax.annotate("", xy=(x0, top), xytext=(x1, top), annotation_clip=False,
                        arrowprops=dict(arrowstyle="<->", lw=0.9, color=color, shrinkA=0, shrinkB=0))
            ax.text(0.5 * (x0 + x1), top + 0.03, text, ha="center", va="bottom", fontsize=9.0, color=color)
        arrow(b[1] - dl, b[1] + dl, r"遷移帯 $2\delta=\tau\Delta$")
        arrow(b[3], b[4], r"輪の幅 $\Delta$")
        if tau < 1:
            arrow(b[5] + dl, b[6] - dl, r"芯（$m_h=g$）", color=S.ORANGE)
        else:
            ax.text(0.5 * (b[5] + b[6]), top + 0.03, "芯なし", ha="center", va="bottom", fontsize=9.0,
                    color=S.ORANGE)
    for ax in axs.ravel():
        ax.set_xlim(0, DREF); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("電子どうしの距離 $d$")
        ax.set_yticks([0, 0.5, 1])
    for ax in axs[:, 0]:
        ax.set_ylabel("窓 $m_h(d)$")
    txt = (r"$d$：電子どうしの距離（周期ノルム）、$d_{\rm ref}=3.215$ はその最大値（横軸の右端）。色の違う線がそれぞれ 1 つのヘッドの窓。"
           r"(c)(d) の破線は全体窓 $g(d)=\frac{1}{2}[1+\cos\pi(d/d_{\rm ref})^3]$、灰色の縦線が輪の境界 $b_h=h\,\Delta$（$\Delta=d_{\rm ref}/8=0.402$）。"
           r"境界のまわりの灰色の帯（幅 $2\delta$）で隣の輪へ重みが移り、両側の窓を足すといつも $g$ に戻る。"
           r"$\tau=2\delta/\Delta$ は、この遷移帯が輪の幅に占める割合。$\tau=1$ では遷移帯が輪を埋めつくして芯がなく、"
           r"$\tau=0.5$ では輪の真ん中半分が芯になり、そこではヘッドが $g$ をそのまま受け持つ。")
    S.caption(fig, "図　3 種類の窓と τ の意味", S.wrap(txt, 178), y=0.012, size=9.0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200); print("wrote", a.out)


if __name__ == "__main__":
    main()
