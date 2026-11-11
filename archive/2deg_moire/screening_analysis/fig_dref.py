# -*- coding: utf-8 -*-
"""赤い帯（$w_h > d_{\\rm ref}$ の無効な段）を、窓を狭くせずに減らす道を並べる。

    python screening_analysis/fig_dref.py --folder <結晶の run> --out fig_dref.png

d_ref は周期ノルムがセル内で取り得る最大値であり、セルの形と大きさだけで決まる。
したがって梯子の上限を下げる以外に、(1) セルの形を変える、(2) N を増やす、
(3) 窓の形を単調から帯域通過（シェル）に変える、の 3 つの道がある。
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from fig_qwin import _font, _lattice, _pnorm                        # noqa

A_EE = float(np.sqrt(2 * np.pi / np.sqrt(3)))
A1 = np.array([A_EE, 0.0]); A2 = np.array([-A_EE / 2, A_EE * np.sqrt(3) / 2])


def cell(M):
    """三角格子の原始ベクトルを整数行列 M で組み替えたスーパーセル。"""
    L = np.zeros((2, 2))
    L[:, 0] = M[0, 0] * A1 + M[1, 0] * A2
    L[:, 1] = M[0, 1] * A1 + M[1, 1] * A2
    return L


def rhombus(n):
    return cell(np.array([[n, 0], [0, n]])), n * n


def rectangle(n1, n2):
    return cell(np.array([[n1, n2], [0, 2 * n2]])), 2 * n1 * n2


def mono(w, x):
    return 0.5 * (1 + np.cos(np.pi * np.clip(x / w, 0.0, 1.0)))


def band(c, s, x):
    return 0.5 * (1 + np.cos(np.pi * np.clip((x - c) / s, -1.0, 1.0)))


def effrank(M):
    """窓の族が実質いくつの独立な形を持つか。対ごとにヘッド平均を引いて SVD。"""
    X = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float((s.sum() ** 2) / (s ** 2).sum())


def pair_distances(folder, nconf=600):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = _lattice(cfg)
    ne = sum(cfg["system"]["electrons"])
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))
    P = np.load(ck[-1], allow_pickle=True)["data"].item()["positions"]
    P = np.asarray(P).reshape(-1, ne, 2)[:nconf]
    fr = (P[:, :, None, :] - P[:, None, :, :]) @ np.linalg.inv(L).T
    dp = _pnorm(fr, L); iu = np.triu_indices(ne, 1)
    from periodicwave.CustomScreenformer import _max_periodic_distance as DR
    return dp[:, iu[0], iu[1]].ravel(), float(DR(L)), ne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="対の距離を取る run（結晶）")
    ap.add_argument("--wmax", type=float, default=5.0)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--out", default="fig_dref.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ja = _font()
    T = lambda x, y: x if ja else y                                  # noqa: E731

    d, dref, ne = pair_distances(a.folder)
    ds = d[::7]
    H = a.heads
    ENJI, AI, MIDORI, HAI = "#a5262f", "#12406f", "#0f7b6c", "#8a8880"

    fig = plt.figure(figsize=(14.2, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.06, 1.20, 0.94],
                          height_ratios=[1, 1], wspace=0.30, hspace=0.42,
                          left=0.055, right=0.955, top=0.790, bottom=0.300)
    ax1 = fig.add_subplot(gs[:, 0])
    axm = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[1, 1], sharex=axm)
    ax3 = fig.add_subplot(gs[:, 2])

    # --- ① セルの形と大きさ ------------------------------------------------
    ns = np.arange(3, 15)
    Nr = np.array([n * n for n in ns]); Dr = np.array([_dr(rhombus(n)[0]) for n in ns])
    RC = [(3, 2), (4, 3), (5, 3), (6, 3), (7, 4), (8, 5), (9, 5), (11, 6), (12, 7)]
    Nq = np.array([2 * p * q for p, q in RC])
    Dq = np.array([_dr(rectangle(p, q)[0]) for p, q in RC])
    ax1.plot(Nr, Dr, "-o", color=ENJI, lw=2.0, ms=6,
             label=T("120°の菱形（いまの形）", "120° rhombus"))
    ax1.plot(Nq, Dq, "-s", color=AI, lw=2.0, ms=6, mfc="white",
             label=T("長方形（三角格子と整合）", "rectangle"))
    ax1.axhline(a.wmax, color=MIDORI, lw=1.6, ls=(0, (5, 3)))
    ax1.text(Nr[-1], a.wmax, T(" いまの梯子の上限 %.1f" % a.wmax, " $w_{max}$"),
             color=MIDORI, fontsize=9.2, va="bottom", ha="right")
    ax1.plot([25], [dref], marker="*", ms=17, color=ENJI, mec="white", mew=1.2, zorder=5)
    ax1.annotate(T("いま\n$N$=25, $d_{\\rm ref}$=%.2f" % dref, "now"), (25, dref),
                 textcoords="offset points", xytext=(12, -30), fontsize=9.0, color=ENJI)
    for (p, q), N, D in zip(RC, Nq, Dq):
        if N in (24, 36, 56):
            ax1.annotate("%d×%d" % (p, q), (N, D), textcoords="offset points",
                         xytext=(4, 7), fontsize=8.6, color=AI)
    ax1.set_xlabel(T("電子数 $N$", "$N$"), fontsize=10.4)
    ax1.set_ylabel(T("$d_{\\rm ref}$（周期ノルムの最大値, $r_s a_B^{*}$）", "$d_{ref}$"),
                   fontsize=10.4)
    ax1.set_xlim(8, 150); ax1.set_ylim(2.4, 9.2); ax1.grid(alpha=.22, lw=.7)
    ax1.legend(fontsize=9.4, frameon=False, loc="upper left")
    ax1.set_title(T("① セルの形と大きさを変える", "① cell shape and size"),
                  fontsize=11.4, pad=8)

    # --- ② 窓の形 -----------------------------------------------------------
    xs = np.linspace(0, max(a.wmax, dref) * 1.02, 500)
    hist, edges = np.histogram(d, bins=90, range=(0, dref), density=True)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    W = np.geomspace(A_EE, a.wmax, H)
    cc = np.linspace(1.0, dref, H); sg = np.full(H, (dref - 1.0) / (H - 1))
    Mm = np.array([mono(w, ds) for w in W])
    Mb = np.array([band(c, s, ds) for c, s in zip(cc, sg)])
    for ax, fam, ttl in ((axm, [mono(w, xs) for w in W],
                          T("単調な窓（いま）　実質 %.1f 通りの形" % effrank(Mm),
                            "monotone")),
                         (axb, [band(c, s, xs) for c, s in zip(cc, sg)],
                          T("帯域通過（シェル）　実質 %.1f 通りの形" % effrank(Mb),
                            "band-pass"))):
        ax.fill_between(ctr, 0, hist / hist.max(), color=HAI, alpha=.28, lw=0, zorder=0)
        for i, m in enumerate(fam):
            ax.plot(xs, m, lw=1.35, color=plt.cm.viridis(i / (H - 1)), zorder=3)
        ax.axvspan(dref, xs[-1], color=ENJI, alpha=.075, zorder=1)
        ax.axvline(dref, color=ENJI, lw=1.2, ls=(0, (4, 2.5)), alpha=.9, zorder=2)
        ax.set_ylim(0, 1.12); ax.set_xlim(0, xs[-1])
        ax.set_ylabel(T("$m_h(d)$", "$m_h$"), fontsize=10.0)
        ax.set_title(ttl, fontsize=10.4, pad=5)
        ax.grid(alpha=.20, lw=.7)
    plt.setp(axm.get_xticklabels(), visible=False)
    axb.set_xlabel(T("電子どうしの距離 $d$（周期ノルム, $r_s a_B^{*}$）", "$d$"),
                   fontsize=10.4)
    axm.text(dref * 1.02, 1.05, T(" $d_{\\rm ref}$", " $d_{ref}$"), color=ENJI,
             fontsize=9.0, va="top", ha="left")
    axm.text(0.02, 0.94, T("灰色 = 電子どうしの距離の分布", "grey = pair distribution"),
             transform=axm.transAxes, fontsize=8.4, color="#52514e", va="top")

    # --- ③ 有効ランク --------------------------------------------------------
    HH = np.array([4, 8, 12, 16, 24, 32, 48])
    rm, rb = [], []
    for h in HH:
        Wh = np.geomspace(A_EE, a.wmax, h)
        rm.append(effrank(np.array([mono(w, ds) for w in Wh])))
        ch = np.linspace(1.0, dref, h); sh = np.full(h, (dref - 1.0) / (h - 1))
        rb.append(effrank(np.array([band(c, s, ds) for c, s in zip(ch, sh)])))
    ax3.plot(HH, rb, "-s", color=AI, lw=2.2, ms=7, mfc="white",
             label=T("帯域通過（シェル）", "band-pass"))
    ax3.plot(HH, rm, "-o", color=ENJI, lw=2.2, ms=7,
             label=T("単調な窓（いま）", "monotone"))
    ax3.plot(HH, HH, ":", color=HAI, lw=1.4)
    ax3.text(HH[-1], HH[-1], T("ヘッド数と同じ ", "= H "), color=HAI, fontsize=8.8,
             va="top", ha="right", rotation=34)
    ax3.set_xscale("log"); ax3.set_yscale("log"); ax3.minorticks_off()
    ax3.set_xticks(HH); ax3.set_xticklabels(["%d" % h for h in HH], fontsize=9.2)
    ax3.set_yticks([1, 2, 5, 10, 20, 50]); ax3.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax3.set_xlabel(T("ヘッド数 $H$", "heads $H$"), fontsize=10.4)
    ax3.set_ylabel(T("実質いくつの形を持つか（有効ランク）", "effective rank"), fontsize=10.4)
    ax3.grid(alpha=.22, lw=.7, which="both")
    ax3.legend(fontsize=9.4, frameon=False, loc="upper left")
    ax3.set_title(T("③ 窓の形を変える", "③ window shape"), fontsize=11.4, pad=8)

    # --- 表題と説明 ---------------------------------------------------------
    fig.text(0.006, 0.985, T("赤い帯を、窓を狭くせずに減らす",
                             "shrinking the dead band without shrinking $w$"),
             fontsize=13.2, weight="bold", color=MIDORI, va="top")
    fig.text(0.006, 0.932,
             T("$d_{\\rm ref}$ はセルの形と大きさだけで決まる（波動関数にも $r_s$ にも依らない）"
               "　—　クーロンのみ, $N$ = %d, 対の距離は $r_s$ = 60 の run から" % ne,
               "$d_{ref}$ depends only on the cell"),
             fontsize=10.5, color="#0b0b0b", va="top")

    _n36 = _dr(rectangle(6, 3)[0]); _n24 = _dr(rectangle(4, 3)[0])
    lines = [
        "① 120°の菱形は、この距離の測り方にとっていちばん不利な形である。計量の非対角成分が $-1/2$ で、"
        "最大値を与える点（$f=(2/3,1/3)$ の深い穴）で打ち消しが起きるため。",
        "　　三角格子と整合する長方形（$M = [[n_1, n_2],[0, 2n_2]]$, $N = 2n_1n_2$）に替えると"
        "$d_{\\rm ref} = 0.798\\sqrt{N}$ となり、同じ $N$ で 24%% 広がる"
        "（$N$=36 なら %.2f → %.2f）。$N$ を増やすより安い。ただし 6 回対称性は失われる。"
        % (_dr(rhombus(6)[0]), _n36),
        "② しかし本当の問題は赤い帯だけではない。単調な窓は入れ子なので、16 段あっても"
        "実質 %.1f 通りの形しか持たない（③ の赤い線はヘッドを増やしても平らなまま）。"
        % effrank(Mm),
        "　　帯域通過（シェル）$m_h(d) = \\frac{1}{2}\\left[1+\\cos\\left(\\pi\\,{\\rm clip}"
        "((d-c_h)/\\sigma_h,\\,-1,\\,1)\\right)\\right]$ にすると、$d \\leq c_h-\\sigma_h$ で厳密に 0 になる。"
        "選別をするのは外側の端ではなく内側の端なので、",
        "　　窓がセルより広くなっても情報を失わない。制約は $w_h < d_{\\rm ref}$ から"
        "$c_h - \\sigma_h < $ 対の最大距離 に移る。同じ 16 ヘッドで %.1f 通り（$c_h=0$ とすれば単調窓に戻る）。"
        % effrank(Mb),
    ] if ja else ["see caption"]
    fig.text(0.008, 0.225, "\n".join(lines), fontsize=9.4, color="#2b2b2b",
             va="top", linespacing=2.0)
    fig.savefig(a.out, dpi=155)
    print("  wrote %s" % a.out)
    print("  d_ref いま %.4f（菱形 N=%d）" % (dref, ne))
    for p, q in RC:
        L, N = rectangle(p, q)
        print("    長方形 %d×%d  N=%-4d d_ref=%.3f" % (p, q, N, _dr(L)))
    print("  有効ランク（H=%d）: 単調 %.2f  シェル %.2f" % (H, effrank(Mm), effrank(Mb)))


def _dr(L):
    from periodicwave.CustomScreenformer import _max_periodic_distance as DR
    return float(DR(L))


if __name__ == "__main__":
    main()
