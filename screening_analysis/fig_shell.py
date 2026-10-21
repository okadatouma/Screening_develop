# -*- coding: utf-8 -*-
"""単調窓と帯域窓（殻）の違いを 1 枚で示す。

    python screening_analysis/fig_shell.py --out fig_shell.png

読み出し q_h の意味が「累積」から「動径分解」に変わる、というのが要点。
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as ST


def monotone(d, w):
    """m_h(d) = (1 + cos(pi*min(d/w_h, 1)))/2。d >= w_h で厳密に 0。"""
    return 0.5 * (1.0 + np.cos(np.pi * np.clip(d[None] / w[:, None], 0.0, 1.0)))


def shell(d, c, s):
    """m_h(d) = (1 + cos(pi*clip((d-c_h)/sigma_h, -1, 1)))/2。殻の外で厳密に 0。"""
    return 0.5 * (1.0 + np.cos(np.pi * np.clip(
        (d[None] - c[:, None]) / s[:, None], -1.0, 1.0)))


def eff_rank(M):
    """ヘッド平均を抜いた行列の特異値の参加率。1 なら全ヘッドが同じ形。"""
    A = M - M.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(A, compute_uv=False) ** 2
    return float(sv.sum() ** 2 / (sv ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_shell.png")
    ap.add_argument("--nheads", type=int, default=16)
    a = ap.parse_args()
    ST.use()
    import matplotlib.pyplot as plt

    H = a.nheads
    dref = 0.643037 * np.sqrt(25.0)          # N = 25 のとき 3.2152
    aee = 1.904626
    d = np.linspace(0.0, dref, 4000)

    w = np.geomspace(aee, dref, H)                       # 単調窓の目盛り
    c = np.linspace(1.5, dref, H)                        # 殻の中心
    sg = np.full(H, (dref - 1.5) / (H - 1))              # 半値幅 = 中心の間隔

    Mm, Ms = monotone(d, w), shell(d, c, sg)
    rm, rs_ = eff_rank(Mm), eff_rank(Ms)

    fig = plt.figure(figsize=(13.4, 6.6))
    fig.text(0.035, 0.968, "単調窓と帯域窓（殻）", fontsize=15.5, weight="bold",
             color=ST.GREEN, va="top")
    fig.text(0.035, 0.921,
             "距離チャネル $m_h(d)$ の形を変えると、読み出し $q_h$ の意味が変わる",
             fontsize=11.0, color=ST.SUB, va="top")

    cm = plt.cm.viridis(np.linspace(0.08, 0.92, H))
    axes = [fig.add_axes([0.055 + i * 0.323, 0.440, 0.243, 0.395]) for i in range(3)]

    # (1) 単調窓 -------------------------------------------------------------
    ax = axes[0]
    for h in range(H):
        ax.plot(d, Mm[h], color=cm[h], lw=1.15)
    ax.set_title("① 単調窓（今回まで使ったもの）", pad=8)
    ax.set_xlabel("電子どうしの距離 $d$   ($r_s a_B^*$)", labelpad=2)
    ax.set_ylabel("$m_h(d)$")
    ax.set_xlim(0, dref); ax.set_ylim(-0.03, 1.16)
    ax.text(0.47, 0.99,
            "どのヘッドも $d = 0$ で 1。\n遠いヘッドの領域が\n近いヘッドを丸ごと含む",
            transform=ax.transAxes, fontsize=9.3, color=ST.SUB, va="top",
            ha="left", linespacing=1.6)

    # (2) 帯域窓 -------------------------------------------------------------
    ax = axes[1]
    for h in range(H):                       # 殻の外は描かない（軸上に線が残る）
        ax.plot(d, np.where(Ms[h] > 1e-9, Ms[h], np.nan), color=cm[h], lw=0.9, alpha=0.45)
    j = 8                                                # 1 本だけ濃く描いて寸法を示す
    _in = Ms[j] > 1e-9                       # 殻の外は描かない（軸上に線が残る）
    ax.fill_between(d, 0, Ms[j], where=_in, color=ST.BLUE, alpha=0.16)
    ax.plot(d, np.where(_in, Ms[j], np.nan), color=ST.BLUE, lw=1.9)
    ax.set_title("② 帯域窓（殻）", pad=8)
    ax.set_xlabel("電子どうしの距離 $d$   ($r_s a_B^*$)", labelpad=2)
    ax.set_xlim(0, dref); ax.set_ylim(-0.03, 1.16)
    ax.set_yticklabels([])
    ax.text(0.04, 0.99, "各ヘッドは $d \\simeq c_h$ の\n細い輪だけを見る",
            transform=ax.transAxes, fontsize=9.3, color=ST.SUB, va="top",
            linespacing=1.6)
    ax.annotate("", xy=(c[j] - sg[j], 0.5), xytext=(c[j] + sg[j], 0.5),
                arrowprops=dict(arrowstyle="<->", color=ST.BLUE, lw=1.1))
    ax.text(c[j] + 0.10, 0.50, "$2\\sigma_h$", fontsize=10.0, color=ST.BLUE,
            ha="left", va="center")
    ax.plot([c[j], c[j]], [0.0, 1.0], color=ST.BLUE, lw=0.9, ls=":")
    ax.text(c[j], 1.06, "$c_h$", fontsize=10.0, color=ST.BLUE, ha="center", va="bottom")

    # (3) 総和 ---------------------------------------------------------------
    ax = axes[2]
    ax.plot(d, Mm.sum(0), color=ST.ORANGE, lw=2.0, label="単調窓")
    ax.plot(d, Ms.sum(0), color=ST.BLUE, lw=2.0, label="帯域窓（殻）")
    ax.set_title("③ ヘッドについての総和 $\\sum_h m_h(d)$", pad=8)
    ax.set_xlabel("電子どうしの距離 $d$   ($r_s a_B^*$)", labelpad=2)
    ax.set_ylabel("$\\sum_h m_h(d)$", labelpad=2)
    ax.set_xlim(0, dref); ax.set_ylim(-0.6, H + 2.6)
    ax.legend(loc="upper right", fontsize=9.6, borderaxespad=0.5)
    ax.annotate("16 本が同じことを\n言っている", xy=(0.30, 15.2), xytext=(0.20, 9.6),
                fontsize=9.3, color=ST.ORANGE, linespacing=1.6, va="top",
                arrowprops=dict(arrowstyle="->", color=ST.ORANGE, lw=1.0))
    ax.annotate("$\\sum_h m_h = 1$\n（1 の分割）", xy=(2.75, 1.0), xytext=(1.80, 5.2),
                fontsize=9.3, color=ST.BLUE, linespacing=1.7, va="top",
                arrowprops=dict(arrowstyle="->", color=ST.BLUE, lw=1.0))

    ST.caption(
        fig,
        "窓の形と、読み出せるもの",
        "① $m_h(d) = \\frac{1}{2}\\left[1+\\cos\\left(\\pi\\min(d/w_h,\\,1)\\right)\\right]$"
        "     $d \\geq w_h$ で厳密に 0。ヘッドの領域が入れ子になる。\n"
        "② $m_h(d) = \\frac{1}{2}\\left[1+\\cos\\left(\\pi\\,\\mathrm{clip}"
        "((d-c_h)/\\sigma_h,\\,-1,\\,1)\\right)\\right]$"
        "     殻の外で厳密に 0。$c_h = 0$ とおくと ① に戻るので、① は ② の特別な場合。\n"
        "③ 入れ子だと $q_h$ は「$w_h$ より近い相関ぜんぶ」の累積量で、隣のヘッドと中身が重なる。"
        "殻は $\\sigma_h$ を中心の間隔に取ると $\\sum_h m_h = 1$ となり、\n"
        "     $q_h$ が「距離 $c_h$ にどれだけ相関があるか」— 動径分解になって $g(r)$ と直に比べられる。\n"
        "ヘッド平均を抜いた $m_h(d)$ の実効ランクは単調窓 %.2f に対し帯域窓 %.1f。"
        "入れ子は段数をいくら増やしても 2.5 を超えられない。"
        % (rm, rs_),
        y=0.050, size=9.3)

    fig.savefig(a.out, dpi=170, facecolor=fig.get_facecolor())
    print("  wrote", a.out)
    print("  実効ランク: 単調 %.3f  殻 %.3f" % (rm, rs_))


if __name__ == "__main__":
    main()
