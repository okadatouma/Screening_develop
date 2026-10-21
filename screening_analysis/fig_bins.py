# -*- coding: utf-8 -*-
"""窓の 3 つの取り方：入れ子（単調）／帯に切って中を平らにする／帯窓（採用）。"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as ST

AEE = 1.904626


def monotone(d, w):
    return 0.5 * (1 + np.cos(np.pi * np.clip(d[None] / w[:, None], 0, 1)))


def shell(d, c, s):
    return 0.5 * (1 + np.cos(np.pi * np.clip((d[None] - c[:, None]) / s[:, None], -1, 1)))


def step(u, dl):
    """なめらかな階段。Theta(u) + Theta(-u) = 1 を厳密に満たす。"""
    return np.clip(0.5 * (1 + np.sin(np.pi * np.clip(u, -dl, dl) / (2 * dl))), 0, 1)


def binwin(d, edges, dl):
    """m_h(d) = Theta(d - a_h) - Theta(d - a_{h+1})。帯の中で 1、外で厳密に 0。"""
    return np.stack([step(d - edges[h], dl) - step(d - edges[h + 1], dl)
                     for h in range(len(edges) - 1)])


def eff_rank(M):
    A = M - M.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(A, compute_uv=False) ** 2
    return float(sv.sum() ** 2 / (sv ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out/fig_bins.png")
    a = ap.parse_args(); ST.use()
    import matplotlib.pyplot as plt

    H = 16
    dref = 0.643037 * np.sqrt(25.0)
    d = np.linspace(0, dref, 4000)
    w = np.geomspace(AEE, dref, H)                       # ① 入れ子（いま）
    edges = np.linspace(0.0, dref, H + 1)                # ② 平らな帯
    dl = 0.25 * (edges[1] - edges[0])
    D = dref / (H - 1); A = D * np.arange(H)             # ③ 帯窓（採用）

    Mm, Mb, Mc = monotone(d, w), binwin(d, edges, dl), shell(d, A, np.full(H, D))
    rk = [eff_rank(M) for M in (Mm, Mb, Mc)]

    fig = plt.figure(figsize=(13.2, 6.2))
    fig.text(0.035, 0.968, "窓の取り方 3 通り", fontsize=15.0, weight="bold", color=ST.GREEN, va="top")
    fig.text(0.035, 0.922, "$m_h(d)$ = ヘッド $h$ が距離 $d$ の相手に掛ける重み。$d$ = 電子どうしの距離（周期ノルム）、"
             "$H$ = ヘッド数 $=16$、$d_{\\mathrm{ref}}$ = $d$ の最大値 $=3.215$",
             fontsize=10.6, color=ST.SUB, va="top")
    cm = plt.cm.viridis(np.linspace(0.08, 0.92, H))
    axes = [fig.add_axes([0.055 + i * 0.323, 0.455, 0.243, 0.385]) for i in range(3)]
    ttl = ["① 入れ子（いま）　$[0,\\,w_h]$",
           "② 帯に切って中を平らにする",
           "③ 帯窓（採用）：輪の中で $1\\to 0$"]
    for k, (ax, M) in enumerate(zip(axes, (Mm, Mb, Mc))):
        for h in range(H):
            ax.plot(d, np.where(M[h] > 1e-9, M[h], np.nan), color=cm[h], lw=1.15)
        ax.set_title(ttl[k], pad=8)
        ax.set_xlabel("$d$   ($r_s a_B^*$)", labelpad=2)
        ax.set_xlim(0, dref); ax.set_ylim(-0.04, 1.22)
        if k == 0: ax.set_ylabel("$m_h(d)$")
        else: ax.set_yticklabels([])
        ax.text(0.04, 1.15, "実効ランク %.2f" % rk[k], fontsize=10, color=ST.ENJI, weight="bold")
    for e in edges: axes[1].plot([e, e], [-0.04, 0.0], color=ST.INK, lw=0.8)
    for v in A:     axes[2].plot([v, v], [-0.04, 0.0], color=ST.INK, lw=0.8)
    axes[2].annotate("", xy=(A[10], 1.05), xytext=(A[11], 1.05),
                     arrowprops=dict(arrowstyle="<->", color=ST.INK, lw=1.0, shrinkA=0, shrinkB=0))
    axes[2].text(0.5 * (A[10] + A[11]), 1.10, "輪 $[a_h,\\,a_{h+1}]$", fontsize=9,
                 ha="center", va="bottom", color=ST.INK)

    ST.caption(
        fig, "違い",
        "①  遠いヘッドが近いヘッドを丸ごと含む。$q_h$ は累積量。段数を増やしても実効ランクは 2.5 が上限。\n"
        "②  $\\Theta_\\delta(u)=\\frac{1}{2}[1+\\sin(\\pi u/2\\delta)]$ の差で $m_h=\\Theta_\\delta(d-a_h)-\\Theta_\\delta(d-a_{h+1})$。"
        "台は帯ちょうど、$\\sum_h m_h=1$ も厳密。ただし帯の中が平らなので、同じ輪の中では近い相手も遠い相手も同じ重み —— 遮蔽になっていない。\n"
        "③  採用：$m_h(d)=\\frac{1}{2}\\left[1+\\cos\\left(\\pi\\,\\mathrm{clip}((d-a_h)/\\Delta,-1,1)\\right)\\right]$、"
        "$a_h=h\\Delta$、$\\Delta=d_{\\mathrm{ref}}/H$。輪の内側の端で 1、外側の端で 0 へ落ちる（近い相手ほど重い）。\n"
        "     $a_h$ を等間隔に置くと raised cosine は 1 の分割になり $\\sum_h m_h(d)=1$（$0\\leq d\\leq a_{H-1}$ で厳密）。"
        "$h=0$ は $a_0=0$ なので従来の単調窓そのもの —— ① はこの族の 1 本目だけを取り出したもの。\n"
        "     内向きの裾を切って台を $[a_h,a_{h+1}]$ だけにすると $d=a_h$ で $0\\to 1$ の跳びが出る"
        "（減少かつ台がコンパクトなら必ず不連続）ので残す。隣のヘッドの落ち方とちょうど相補的なので情報は重複しない。",
        y=0.050, size=9.2)
    fig.savefig(a.out, dpi=180, facecolor="white")
    print("wrote", a.out)
    print("実効ランク: 入れ子 %.2f / 平らな帯 %.2f / 帯窓 %.2f   （16 ヘッド）" % tuple(rk))
    print("輪の内側の端 a_h =", np.round(A, 3), " 帯幅 %.4f" % D)
    di = np.linspace(0, A[-1], 20000)
    ss = shell(di, A, np.full(H, D)).sum(0)
    print("[0, a_{H-1}] での sum_h m_h : min %.10f  max %.10f" % (ss.min(), ss.max()))
    print("max|m''| = %.1f （平らな帯は %.1f）" % (np.pi**2 / (2 * D**2), np.pi**2 / (8 * dl**2)))


if __name__ == "__main__":
    main()
