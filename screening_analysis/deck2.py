# -*- coding: utf-8 -*-
"""この研究の導入スライド（M1 向け）。物理のセットアップから q・k の解釈まで。

    python screening_analysis/deck_intro.py --folder <結晶 run> --runs "液体=..,境界=..,結晶=.." \
        --out deck_intro.pdf --pngdir out_h/deck
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

INK, SUB, AI, ENJI, MIDORI, HAI, RULE = ("#141412", "#4a4946", "#12406f", "#a5262f",
                                         "#0f7b6c", "#8a8880", "#d6d3cc")
WF, HF = 13.33, 7.5


def _font():
    import matplotlib
    from matplotlib import font_manager as fm
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p:
            fm.fontManager.addfont(p)
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False


class Flow:
    """上から順に置いていくだけの配置器。行数から高さを積むので重ならない。"""

    def __init__(self, fig, x_in, y_in, w_in):
        self.fig, self.x, self.y, self.w = fig, x_in, y_in, w_in

    def fx(self, x_in=None):
        return (self.x if x_in is None else x_in) / WF

    def fy(self, y_in=None):
        return 1.0 - (self.y if y_in is None else y_in) / HF

    def text(self, s, size=10.0, col=INK, wt="normal", ls=1.65, gap=0.10,
             indent=0.0, math_extra=0.0):
        n = s.count("\n") + 1
        self.fig.text(self.fx(self.x + indent), self.fy(), s, fontsize=size, color=col,
                      weight=wt, ha="left", va="top", linespacing=ls)
        self.y += n * size * ls / 72.0 + math_extra + gap
        return self

    def eq(self, s, size=13.0, col=INK, gap=0.14, indent=0.18):
        return self.text(s, size, col, "normal", 1.4, gap, indent, math_extra=0.16)

    def skip(self, inch):
        self.y += inch
        return self

    def axes(self, h_in, w_in=None, x_in=None, gap=0.16, pre=0.0):
        """pre = 軸の上に確保する余白（タイトル用）、gap = 下に確保する余白（軸ラベル用）。"""
        w_in = w_in or self.w
        self.y += pre
        ax = self.fig.add_axes([self.fx(x_in if x_in is not None else self.x),
                                self.fy(self.y + h_in), w_in / WF, h_in / HF])
        self.y += h_in + gap
        return ax


def new_page(pdf, pages, title, lead=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(WF, HF)); fig.patch.set_facecolor("white")
    fig.text(0.027, 1 - 0.40 / HF, title, fontsize=16.5, color=MIDORI,
             weight="bold", va="top")
    y = 0.86
    if lead:
        fig.text(0.027, 1 - 0.78 / HF, lead, fontsize=10.6, color=INK, va="top",
                 linespacing=1.7)
        y = 0.78 + (lead.count("\n") + 1) * 10.6 * 1.7 / 72.0 + 0.16
    fig.add_artist(plt.Line2D([0.027, 0.973], [1 - y / HF] * 2, color=RULE, lw=1.0))
    pages.append(fig)
    return fig, y + 0.22


def foot(fig, s, n=None, total=None):
    fig.text(0.027, 0.030, s, fontsize=8.4, color=SUB, va="bottom")
    if n is not None:
        fig.text(0.973, 0.030, "%d / %d" % (n, total), fontsize=8.4, color=SUB,
                 va="bottom", ha="right")



# ---------------------------------------------------------------- 小さな部品

def psi6_panel(ax):
    """psi6 の定義を絵で示す：中心の電子と最近接 6 個、その角度。"""
    import matplotlib.pyplot as plt
    th = np.array([0.15, 1.05, 2.20, 3.25, 4.20, 5.35])
    r = np.array([1.0, 1.05, .95, 1.02, 1.08, .97])
    P = np.c_[r * np.cos(th), r * np.sin(th)]
    ax.plot([0], [0], "o", ms=11, color=ENJI, mec="white", mew=1.2, zorder=5)
    for k, p in enumerate(P):
        ax.plot([0, p[0]], [0, p[1]], "-", color=HAI, lw=1.0, zorder=1)
        ax.plot([p[0]], [p[1]], "o", ms=9, color=AI, mec="white", mew=1.0, zorder=4)
        ax.text(p[0] * 1.32, p[1] * 1.32, r"$\theta_%d$" % (k + 1), fontsize=8.6,
                color=SUB, ha="center", va="center")
    aa = np.linspace(0, th[0], 40)
    ax.plot(.42 * np.cos(aa), .42 * np.sin(aa), "-", color=ENJI, lw=1.2)
    ax.plot([0, 1.5], [0, 0], ":", color=HAI, lw=1.0)
    ax.set_xlim(-1.6, 1.75); ax.set_ylim(-1.5, 1.55)
    ax.set_aspect("equal"); ax.axis("off")


def moire_panel(ax, twist=7.0, n=13):
    """2 枚の三角格子を少しねじって重ね、うなりが見えるようにする。"""
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3) / 2])
    ij = np.array([[i, j] for i in range(-n, n + 1) for j in range(-n, n + 1)])
    Pt = ij @ np.vstack([a1, a2])
    t = np.radians(twist); Rt = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    Pb = Pt @ Rt.T
    lim = 5.4
    for P, c, m in ((Pt, "#3b3a37", "o"), (Pb, ENJI, "o")):
        k = (np.abs(P[:, 0]) < lim) & (np.abs(P[:, 1]) < lim)
        ax.plot(P[k, 0], P[k, 1], m, ms=3.4, color=c, alpha=.85, mec="none")
    aM = 1.0 / (2 * np.sin(t / 2))
    for s in (0, 1):
        for u in (0, 1):
            ax.plot([s * aM * np.cos(np.radians(twist / 2 + 90))],
                    [s * aM * np.sin(np.radians(twist / 2 + 90))], "")
    circ = plt.Circle((0, 0), 0.55, fill=False, ec=MIDORI, lw=2.0, zorder=6)
    ax.add_patch(circ)
    c2 = np.array([aM * np.cos(np.radians(twist / 2 + 90)),
                   aM * np.sin(np.radians(twist / 2 + 90))])
    ax.add_patch(plt.Circle(tuple(c2), 0.55, fill=False, ec=MIDORI, lw=2.0, zorder=6))
    ax.annotate("", xy=tuple(c2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.8, color=MIDORI), zorder=7)
    ax.text(c2[0] * .5 + .55, c2[1] * .5, "$a_M$", color=MIDORI, fontsize=13,
            ha="left", va="center", zorder=8,
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")


def stream_panel(ax):
    """残差ストリームの図：各ヘッドの寄与 c_h が電子 i の特徴に足し込まれる。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.annotate("", xy=(9.5, 3.4), xytext=(0.4, 3.4),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#3b3a37"))
    ax.text(0.4, 3.75, r"電子 $i$ の特徴 $\mathbf{h}_i$（残差ストリーム）",
            fontsize=10.0, color="#3b3a37")
    ax.text(0.15, 3.4, r"$\mathbf{h}_i^{(0)}$", fontsize=11, ha="right", va="center")
    ax.text(9.62, 3.4, r"$\mathbf{h}_i^{(1)}$", fontsize=11, ha="left", va="center")
    xs = [1.6, 3.3, 5.0, 6.7, 8.4]
    cols = [plt.cm.viridis(v) for v in (0.05, 0.3, 0.55, 0.78, 0.95)]
    for k, (x, c) in enumerate(zip(xs, cols)):
        ax.add_patch(FancyBboxPatch((x - .72, .55), 1.44, 1.25,
                                    boxstyle="round,pad=0.06", fc="white", ec=c, lw=2.0))
        ax.text(x, 1.44, "ヘッド %s" % (["1", "2", "…", "…", "16"][k]), fontsize=9.4,
                ha="center", color=c)
        ax.text(x, 1.02, r"$w_h$ = %s" % (["1.80", "1.94", "…", "…", "3.20"][k]),
                fontsize=8.8, ha="center", color=SUB)
        ax.annotate("", xy=(x, 3.3), xytext=(x, 1.9),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=c))
        ax.text(x + .12, 2.52, r"$c_h(i)$", fontsize=10.2, color=c, ha="left")
        ax.plot([x], [3.4], "+", ms=13, mew=2.2, color=c, zorder=6)
    ax.text(5.0, 0.16, r"各ヘッドは $W^{(h)}_{ij}$ で相手を重みづけして 1 本のベクトル "
                       r"$c_h(i)$ を作り、それを足し込む", fontsize=9.6, ha="center",
            color=SUB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--dens", default="")
    ap.add_argument("--heads", default="5,15")
    ap.add_argument("--nconf", type=int, default=64)
    ap.add_argument("--out", default="deck2.pdf")
    ap.add_argument("--pngdir", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon
    from matplotlib.backends.backend_pdf import PdfPages
    _font()
    globals()["plt"] = plt

    from attn_probe import probe
    from fig_qwin import load as qload, _lattice, _pnorm

    R = probe(a.folder, nconf=a.nconf)
    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    NE, NH, dref = R["NE"], R["NH"], float(R["dref"])
    nup = int((np.asarray(R["spins"][0]) > 0).sum())
    w = R["w"][0]; order = np.argsort(w)
    W1 = np.asarray(R["W"])[0, 0]
    L = _lattice(cfg)
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))
    CM = LinearSegmentedColormap.from_list("blu", ["#ffffff", "#cfe0f2", "#5b93cc", "#123a63"])
    # ---- 結果ページに載せる測定条件を config から自動生成する ----------------
    # ベタ書きにすると「モアレを入れて測ったのか」が図から読めない事故が起きる。
    def _cond_str(runs_spec):
        rs_l, U_l = [], []
        for _part in runs_spec.split(","):
            _f = _part.rsplit("=", 1)[1]
            _c = json.load(open(os.path.join(_f, "config.json")))
            _pk = _c["system"]["make_local_energy_kwargs"]["potential_kwargs"]
            rs_l.append(float(_pk["interaction_energy_scale"]))
            U_l.append(float(_pk.get("moire_potential_strength", 0.0)))
        _fmt = lambda v: ("%g" % v)
        if all(u == 0.0 for u in U_l):
            return ("測定条件：クーロンのみ（$U = 0$、モアレなし）、$N = %d$、"
                    "相を動かしたのは $r_s = %s$。"
                    % (NE, " / ".join(_fmt(v) for v in rs_l)))
        if len(set(rs_l)) == 1:
            return ("測定条件：モアレあり、$r_s = %s$ 固定、$N = %d$、"
                    "相を動かしたのは $U = %s$。"
                    % (_fmt(rs_l[0]), NE, " / ".join(_fmt(v) for v in U_l)))
        return ("測定条件：$N = %d$、$r_s = %s$、$U = %s$。"
                % (NE, " / ".join(_fmt(v) for v in rs_l),
                   " / ".join(_fmt(v) for v in U_l)))

    COND = _cond_str(a.runs) if a.runs else ""

    D = np.load("/tmp/check.npy")            # w, qpL, qpC, qsL, qsC, PhL, PhC
    pages = []
    NP = 20

    # ===================== p1 タイトル =====================================
    fig = plt.figure(figsize=(WF, HF)); fig.patch.set_facecolor("white"); pages.append(fig)
    fig.text(0.06, 0.80, "波動関数が使っている「長さ」を読み出す", fontsize=27,
             color=MIDORI, weight="bold", va="top")
    fig.text(0.06, 0.705, "遮蔽アテンションによるニューラル波動関数", fontsize=15,
             color=INK, va="top")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.665] * 2, color=RULE, lw=1.2))
    fig.text(0.06, 0.605,
             "2 次元電子系は密度を下げると、液体から Wigner 結晶へ移る。\n"
             "そのとき波動関数は「どのくらいの距離まで相手を見ている」のだろうか。",
             fontsize=12.6, color=INK, va="top", linespacing=1.95)
    fig.text(0.06, 0.445,
             "ニューラル波動関数のアテンションに、ヘッドごとに固定した距離の窓を入れる。\n"
             "どのヘッドが効いたかが、そのまま「使っている長さ」の読み出しになる。",
             fontsize=12.6, color=ENJI, va="top", linespacing=1.95)
    fig.text(0.06, 0.275,
             "本編　①$r_s$　②相　③なぜモアレか　④モアレポテンシャル　⑤ハミルトニアン　"
             "⑥有限セル　⑦遮蔽アテンション\n"
             "　　　⑧層とヘッド　⑨$q\\cdot k$ の読み方　⑩液体と結晶　⑪読み出せるもの　"
             "⑫幾何だけで説明できるか\n"
             "付録　A モアレの導出　B 変分モンテカルロと $\\nabla^{2}$　"
             "C 最小イメージの折れ　D 寄与率の定義　E 全ヘッドの数値",
             fontsize=10.6, color=SUB, va="top", linespacing=2.0)

    # ===================== p2 r_s =========================================
    fig, y = new_page(None, pages, "① 2 次元電子系と $r_s$",
                      lead="扱うのは、一様な正電荷の背景の上に置かれた $N$ 個の電子。"
                           "半導体量子井戸や単層 TMD の中の電子がこれに当たる。")
    f = Flow(fig, 0.36, y, 6.0)
    f.text("密度から長さを決める", 11.4, AI, "bold", gap=0.14)
    f.text("面密度を $n_e = N/A$ とする。電子 1 個が占める面積 $1/n_e$ を円と見たときの\n"
           "半径が、その系を代表する長さである。これを $r_s a_B^{*}$ と書く。", 10.0)
    f.eq(r"$\dfrac{1}{n_e} = \pi\,(r_s a_B^{*})^{2}$", 14.0)
    f.text("$a_B^{*} = \\hbar^{2}\\varepsilon/m^{*}e^{2}$（有効ボーア半径）、"
           "${\\rm Ha}^{*} = e^{2}/\\varepsilon a_B^{*}$（有効ハートリー）。\n"
           "物質ごとの事情（有効質量 $m^{*}$ と誘電率 $\\varepsilon$）は全部ここに入る。",
           9.6, SUB, gap=0.26)
    f.text("$r_s$ は無次元数：クーロン / 運動エネルギーの比", 11.4, AI, "bold", gap=0.14)
    f.text("長さを $r_s a_B^{*}$ で、エネルギーを ${\\rm Ha}^{*}/r_s^{2}$ で測ると", 10.0)
    f.eq(r"$\hat H = -\dfrac{1}{2}\sum_i \nabla_i^{2}"
         r" \;+\; r_s \sum_{i<j} r_{ij}^{-1}$", 14.0)
    f.text("運動項の係数が 1/2 に固定され、$r_s$ はクーロン項の前にだけ残る。\n"
           "つまり $r_s$ が大きいほどクーロンが相対的に強い。物質の詳細は消え、\n"
           "$r_s$ ひとつで系が決まる。", 10.0)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("なぜ密度を下げるとクーロンが勝つのか", 11.4, AI, "bold", gap=0.14)
    f2.text("平均距離を $r$ とすると、不確定性から運動エネルギーは $\\sim 1/r^{2}$、\n"
            "クーロンエネルギーは $\\sim 1/r$。$r$ を大きくすると運動項のほうが\n"
            "速く小さくなるので、低密度では必ずクーロンが勝つ。", 10.0)
    axk = f2.axes(2.05, 5.2, gap=0.66)
    rr = np.linspace(0.6, 4.0, 300)
    axk.plot(rr, 1 / rr ** 2, "-", color=AI, lw=2.2, label=r"運動 $\sim 1/r^{2}$")
    axk.plot(rr, 1 / rr, "-", color=ENJI, lw=2.2, label=r"クーロン $\sim 1/r$")
    axk.axvline(1.0, color=HAI, lw=1.0, ls=":")
    axk.set_xlim(0.6, 4.0); axk.set_ylim(0, 2.6)
    axk.set_xlabel("電子どうしの平均距離 $r$（任意単位）", fontsize=9.0, labelpad=1)
    axk.set_ylabel("エネルギー", fontsize=9.0, labelpad=1)
    axk.tick_params(labelsize=8.0); axk.grid(alpha=.2, lw=.6)
    axk.legend(fontsize=9.0, frameon=False)
    f2.text("高密度（左）は運動エネルギーが支配 → 電子は動き回る（液体）。\n"
            "低密度（右）はクーロンが支配 → 電子は互いに離れて固まる（Wigner 結晶）。",
            10.0, INK, gap=0.20)
    f2.text("$r_s$ を上げることは、密度を下げることと同じである。", 10.4, ENJI)
    foot(fig, "① 2 次元電子系と $r_s$", len(pages), NP)

    # ===================== p3 相と秩序変数 ================================
    fig, y = new_page(None, pages, "② 相をどう見分けるか：秩序変数",
                      lead="「液体か結晶か」は目で見ても曖昧なので、数で判定する。"
                           "使うのは並び方の秩序 $\\psi_6$ と、位置のゆらぎの 2 つ。")
    f = Flow(fig, 0.36, y, 4.4)
    axp = f.axes(2.20, 3.1, gap=0.14)
    psi6_panel(axp)
    f.text("電子 $i$ から見た最近接 6 個の方向 $\\theta_1,\\dots,\\theta_6$", 9.4, SUB,
           gap=0.22)
    f.text("① 六方晶秩序 $\\psi_6$", 11.4, AI, "bold", gap=0.12)
    f.eq(r"$\psi_6 = \left|\; \overline{\; e^{\,6 i \theta_k} \;}\; \right|$", 14.0,
         gap=0.10)
    f.text("6 本の方向を 6 倍角にして平均し、その長さを取る。6 倍角にするのは、\n"
           "きれいな三角格子だと 6 本が $60^\\circ$ おきに並び、$6\\theta$ が全部\n"
           "$2\\pi$ の整数倍になって位相が揃うから。", 9.8)
    f.text("$\\psi_6 = 1$　完全な三角格子　　$\\psi_6 = 0$　方向がばらばら", 10.0, ENJI)

    f2 = Flow(fig, 5.15, y, 3.8)
    f2.text("② 位置のゆらぎ（Lindemann 比）", 11.4, AI, "bold", gap=0.12)
    f2.eq(r"$\dfrac{{\rm sd}(d_{\rm NN})}{\langle d_{\rm NN}\rangle}$", 14.0, gap=0.10)
    f2.text("$d_{\\rm NN}$ は各電子から最近接までの距離。その相対的なばらつき。\n"
            "結晶なら最近接距離がそろうので小さくなり、液体なら大きい。\n"
            "2 次元の融解の目安は 0.1 前後（Lindemann 基準）。", 9.8, gap=0.22)
    f2.text("実測", 11.4, AI, "bold", gap=0.12)
    rows = [["", "$r_s$", "$\\psi_6$", "ゆらぎ", "$\\langle d_{\\rm NN}\\rangle$"],
            ["液体", "5", "0.367", "0.266", "1.177"],
            ["境界", "30", "0.408", "0.169", "1.333"],
            ["結晶", "60", "0.461", "0.141", "1.396"]]
    xs = [5.30, 6.05, 6.75, 7.55, 8.25]
    yy = f2.y
    for j, c in enumerate(rows[0]):
        fig.text(xs[j] / WF, 1 - yy / HF, c, fontsize=9.4, color=SUB, weight="bold", va="top")
    fig.add_artist(plt.Line2D([5.30 / WF, 8.95 / WF], [1 - (yy + 0.20) / HF] * 2,
                              color=RULE, lw=0.9))
    for i, r in enumerate(rows[1:]):
        for j, c in enumerate(r):
            fig.text(xs[j] / WF, 1 - (yy + 0.32 + i * 0.235) / HF, c, fontsize=9.8,
                     color=ENJI if r[0] == "結晶" else INK, va="top")
    f2.y = yy + 0.32 + 3 * 0.235 + 0.24
    f2.text("$\\psi_6$ が 1 に届かないのは有限サイズ（$N=25$）とゼロ点振動のため。\n"
            "絶対値ではなく 3 相の間の変化を見るのが正しい使い方。", 9.6, SUB, gap=0.20)
    f2.text("ゆらぎが $0.266 \\to 0.141$ と Lindemann の目安 0.1 に向かって\n"
            "落ちているのが、いちばん素直な判定。", 9.8, ENJI)

    f3 = Flow(fig, 9.35, y, 3.6)
    f3.text("転移はどこか", 11.4, AI, "bold", gap=0.12)
    f3.text("量子モンテカルロによる 2 次元電子ガスの見積もりでは、\n"
            "Wigner 結晶化は $r_s \\approx 31$〜$37$。\n"
            "（Tanatar–Ceperley 1989: $37\\pm5$、\n"
            "　Drummond–Needs 2009: $31\\pm1$）", 9.8, gap=0.22)
    f3.text("この研究の 3 点：$r_s = 5$（液体）、$30$（境界のすぐ手前）、\n"
            "$60$（結晶）。", 9.8, ENJI, gap=0.24)
    f3.text("エネルギーの妥当性", 11.4, AI, "bold", gap=0.12)
    f3.text("$E/N = -1.1061/r_s + 0.814/r_s^{3/2}$ [${\\rm Ha}^{*}$]\n"
            "（2 次元電子ガスの表式）と 0.02〜0.11% で一致。\n"
            "物理としては正しく解けている。", 9.8)
    foot(fig, "② 秩序変数", len(pages), NP)

    # ===================== p3b 3 相の実空間 ===============================
    if a.dens and os.path.exists(a.dens):
        fig, y = new_page(None, pages, "②' 3 つの相の実空間",
                          lead="同じ $N$ = 25、同じネットワークで、$r_s$ だけを変えた 3 つ。"
                               "秩序変数の数字が何を意味しているかはこれを見ると早い。")
        im = mpimg.imread(a.dens)
        h0, w0 = im.shape[:2]
        f = Flow(fig, 0.36, y, 12.6)
        axd = f.axes(3.35, 12.4, gap=0.16)
        axd.imshow(im[int(h0 * 0.115):int(h0 * 0.90)]); axd.axis("off")
        f.text("上段：スーパーセルへ折り返した 1 体密度 $n(\\mathbf{r})$。"
               "$r_s = 5$ では一様、$r_s = 60$ では 25 個の輝点が三角格子を組む。", 10.0)
        f.text("下段：1 個の電子から見た他の電子の分布 $g(x,y)$（対分布）。"
               "こちらは結晶ドメインの選び方に依らない量なので、より客観的な指標になる。\n"
               "$r_s = 60$ では中心の相関ホール（互いに避ける穴）の外に、"
               "配位殻がはっきり立っているのが見える。", 10.0)
        foot(fig, "②' 3 つの相の実空間", len(pages), NP)

    # ===================== p4 なぜモアレか ================================
    fig, y = new_page(None, pages, "③ なぜモアレポテンシャルを入れるのか",
                      lead="純粋なクーロンガスだけでは、読み出した長さが"
                           "「何の長さ」なのかを決められない。それが動機である。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("クーロンだけだと長さが 1 本しかない", 11.4, AI, "bold", gap=0.14)
    f.text("コード単位では電子間隔が $a_{ee} = 1.905$ に固定されていて、$r_s$ を\n"
           "変えても動かない（$r_s$ は結合定数としてしか入らないため）。\n"
           "つまり系の中に長さの候補が実質 1 本しかない。", 10.0)
    f.text("すると、読み出した $\\langle w\\rangle_q$ が動いたときに", 10.0, gap=0.10)
    f.text("　・電子間隔を見ているのか\n"
           "　・相関長を見ているのか\n"
           "　・相互作用の到達距離を見ているのか", 10.2, ENJI, gap=0.14)
    f.text("これらが全部同じスケールに張りついているので、原理的に区別できない。\n"
           "「$r_s$ の関数を測っているだけでは」という批判に答えられない。", 10.0)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("モアレが与えるもの", 11.4, AI, "bold", gap=0.14)
    f2.text("モアレ周期 $a_M$ は基板（ねじれ角と格子不整合）が決めるので、電子の\n"
            "密度とは独立に選べる。長さがもう 1 本増える。", 10.0)
    f2.text("$\\nu = 2/3$ の例", 10.6, AI, "bold", gap=0.10)
    tb = [["$a_M$", "1.555", "基板が決めるモアレ周期"],
          ["$a_{ee}$", "1.905", "密度が決める電子間隔"],
          ["$\\sqrt{3}\\,a_M$", "2.694", "一般化 Wigner 結晶の周期"]]
    yy = f2.y
    for i, r in enumerate(tb):
        for j, (xx, c) in enumerate(zip([7.25, 8.15, 9.05], r)):
            fig.text(xx / WF, 1 - (yy + i * 0.30) / HF, c, fontsize=10.0,
                     color=INK if j < 2 else SUB, va="top")
    f2.y = yy + 3 * 0.30 + 0.20
    f2.text("3 本が離れるので、$\\langle w\\rangle_q$ がどれに追随するかを見れば\n"
            "「窓が何を測っているか」を切り分けられる。", 10.0, ENJI, gap=0.24)
    f2.text("実験との接点", 11.4, AI, "bold", gap=0.12)
    f2.text("TMD のモアレ超格子では、$\\nu = 1$ でモット絶縁体、分数 $\\nu$ で\n"
            "一般化 Wigner 結晶が観測されている。手法を当てる先としても\n"
            "こちらが本命になる。", 10.0, gap=0.22)
    f2.text("注意：これまで回した 3 本は $\\nu = 1$ で、$a_M = a_{ee}$ ちょうど。\n"
            "長さの縮退が解けていないので、この切り分けはまだできていない。",
            10.0, ENJI)
    foot(fig, "③ なぜモアレか", len(pages), NP)

    # ===================== p5 モアレポテンシャル ==========================
    fig, y = new_page(None, pages, "④ モアレ三角ポテンシャルとは何か",
                      lead="2 枚の 2 次元結晶をわずかにずらして重ねると、"
                           "うなりが長周期の超格子を作る。それが電子から見た周期ポテンシャルになる。")
    f = Flow(fig, 0.36, y, 4.3)
    axm = f.axes(3.05, 4.0, gap=0.14)
    moire_panel(axm, twist=7.0)
    f.text("黒と赤はそれぞれの層の原子。$7^\\circ$ ねじってある。緑の丸のところで\n"
           "2 枚がほぼ重なり、その間隔がモアレ周期 $a_M$。", 9.6, SUB, gap=0.18)
    f.eq(r"$a_M \simeq \dfrac{a_0}{\sqrt{\theta^{2}+\delta^{2}}}$", 13.0, gap=0.10)
    f.text("$\\theta$ = ねじれ角、$\\delta = \\Delta a_0/a_0$ = 格子不整合\n"
           "（2 枚の格子定数がわずかに違うこと）。どちらも小さいほど $a_M$ は長い。",
           9.6, SUB)

    f2 = Flow(fig, 5.05, y, 3.7)
    f2.text("なぜポテンシャルになるか", 11.4, AI, "bold", gap=0.12)
    f2.text("モアレセルの中で、場所ごとに「どの原子がどの原子の真上か」が\n"
            "変わる。積層の仕方が変わればバンド端の位置も変わるので、\n"
            "キャリアから見るとモアレ周期の周期ポテンシャルが立つ。\n"
            "最低次の調和（最短の逆格子ベクトル 3 本）だけ残すと", 9.8)
    f2.eq(r"$V(\mathbf{r}) = -2U \sum_{j=1,2,3} \cos(\mathbf{G}_j\cdot\mathbf{r}+\phi)$",
          12.0, gap=0.10)
    f2.text("$|\\mathbf{G}_j| = 4\\pi/(\\sqrt{3}\\,a_M)$、$U$ が深さ、$\\phi$ が谷の形。\n"
            "導出は付録 A。", 9.6, SUB, gap=0.24)
    f2.text("充填率 $\\nu$", 11.4, AI, "bold", gap=0.12)
    f2.text("$\\nu = N/n_{\\rm sites}$　… モアレの谷 1 個あたりの電子数\n"
            "コード単位では $a_M = a_{ee}\\sqrt{\\nu}$。", 10.0)

    f3 = Flow(fig, 9.05, y, 3.9)
    axv = f3.axes(3.05, 3.4, gap=0.14, pre=0.28)
    aM = 1.0
    G = [2 * np.pi / aM * np.array([1, 1 / np.sqrt(3)]),
         2 * np.pi / aM * np.array([0, 2 / np.sqrt(3)]),
         2 * np.pi / aM * np.array([-1, 1 / np.sqrt(3)])]
    gx = np.linspace(-1.6, 1.6, 400); X, Y = np.meshgrid(gx, gx)
    V = -2 * sum(np.cos(g[0] * X + g[1] * Y + np.pi / 4) for g in G)
    axv.pcolormesh(X, Y, V, cmap="magma_r", shading="auto")
    axv.set_aspect("equal"); axv.axis("off")
    axv.set_title("$V(\\mathbf{r})$（明るいほど深い谷）　$\\phi = \\pi/4$",
                  fontsize=9.4, pad=4)
    f3.text("三角形に並んだ谷ができる。電子はここに落ちようとする。", 9.6, SUB, gap=0.22)
    f3.text("$\\nu = 1$：谷 1 個に電子 1 個。バンド理論なら金属だが、電子どうしの\n"
            "　反発で動けなくなる（モット絶縁体）。並びは基板が決めている。", 9.8)
    f3.text("分数 $\\nu$：谷のほうが多いので、電子は「どの谷に座るか」を自分で\n"
            "　選ぶ。その選択をクーロン斥力が決めて秩序が出る\n"
            "　（一般化 Wigner 結晶）。", 9.8, ENJI)
    foot(fig, "④ モアレ三角ポテンシャル", len(pages), NP)

    # ===================== p6 ハミルトニアン ==============================
    fig, y = new_page(None, pages, "⑤ 解きたいハミルトニアン",
                      lead="ここまでの 3 つを足す。すべての run は、"
                           "この 1 本の式のパラメータを振っただけである。")
    f = Flow(fig, 0.36, y, 12.6)
    f.skip(0.30)
    f.eq(r"$\hat H = -\dfrac{1}{2}\sum_i \nabla_i^{2}"
         r" \;+\; r_s \sum_{i<j} r_{ij}^{-1}"
         r" \;+\; \sum_i V(\mathbf{r}_i)$", 20.0, INK, gap=0.16, indent=2.2)
    for xx, lab in ((2.55, "運動"), (5.35, "クーロン"), (8.05, "モアレ")):
        fig.text(xx / WF, 1 - (f.y + 0.02) / HF, lab, fontsize=10.4, color=SUB,
                 ha="center", va="top")
    f.skip(0.42)
    g1 = Flow(fig, 0.55, f.y, 5.8)
    g1.text("相を動かすつまみは 2 つ", 11.4, AI, "bold", gap=0.14)
    g1.text("$r_s$　… 密度。上げると液体 → Wigner 結晶（$U = 0$ でこれだけ）\n"
            "$U$　… モアレの深さ。$r_s$ を固定したまま上げても、電子は谷に\n"
            "　　　落ちて秩序が出る", 10.2)
    g1.text("同じ転移に独立な 2 つの経路があるのが効いてくる（③）。", 10.0, ENJI)
    g2 = Flow(fig, 7.05, f.y, 5.9)
    g2.text("変分モンテカルロで解く", 11.4, AI, "bold", gap=0.14)
    g2.text("試行波動関数 $\\psi_\\vartheta$ をニューラルネットで表し、\n"
            "$E[\\vartheta] = \\langle \\psi_\\vartheta|\\hat H|\\psi_\\vartheta\\rangle"
            "/\\langle\\psi_\\vartheta|\\psi_\\vartheta\\rangle$ を下げる。", 10.2)
    g2.text("その計算に局所エネルギー $E_{\\rm loc} = \\hat H\\psi_\\vartheta/\\psi_\\vartheta$\n"
            "が出てきて、そこに $\\nabla^{2}\\log\\psi_\\vartheta$ が入る。だから波動関数は\n"
            "2 階微分できなければならない。これが後で距離の測り方を縛る（付録 B）。",
            10.0, ENJI)
    foot(fig, "⑤ 解きたいハミルトニアン", len(pages), NP)

    # ===================== p7 有限セル ====================================
    fig, y = new_page(None, pages, "⑥ 有限セルで解く：長さの決まり方",
                      lead="無限系は解けないので、$N$ 電子を入れた箱に周期境界条件を課す。"
                           "ここで系の長さがすべて決まる。")
    f = Flow(fig, 0.36, y, 4.3)
    axc = f.axes(2.45, 4.1, gap=0.12)
    P = np.asarray(np.load(sorted(glob.glob(os.path.join(a.folder, "qmcjax_ckpt_*.npz")))[-1],
                           allow_pickle=True)["data"].item()["positions"])
    P = P.reshape(-1, NE, 2)[0]
    Pc = np.mod(np.linalg.solve(L, P.T).T, 1.0) @ L.T
    corners = np.array([[0, 0], L[:, 0], L[:, 0] + L[:, 1], L[:, 1]])
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            sh = m * L[:, 0] + n * L[:, 1]; mid = (m == 0 and n == 0)
            axc.add_patch(Polygon(corners + sh, closed=True, fill=mid,
                                  fc="#f2f4f7" if mid else "none",
                                  ec=INK if mid else HAI, lw=1.6 if mid else .7,
                                  ls="-" if mid else (0, (3, 3)), alpha=1 if mid else .6))
            axc.plot(Pc[:, 0] + sh[0], Pc[:, 1] + sh[1], "o", ms=4.8 if mid else 2.5,
                     color=AI if mid else HAI, alpha=1 if mid else .35,
                     mec="white", mew=.7 if mid else 0)
    for v, lab in ((L[:, 0], r"$\mathbf{L}_1$"), (L[:, 1], r"$\mathbf{L}_2$")):
        axc.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ENJI,
                                     shrinkA=0, shrinkB=0), zorder=5)
        axc.text(v[0] * .55 + (.8 if v[1] > 1 else 0), v[1] * .55 - (0 if v[1] > 1 else 1.0),
                 lab, color=ENJI, fontsize=12, ha="center", va="center", zorder=6,
                 bbox=dict(fc="white", ec="none", alpha=.85, pad=1.4))
    axc.set_aspect("equal"); axc.axis("off")
    axc.set_xlim(-1.25 * abs(L[0, 0]) + L[0, 1] * .6, 1.25 * abs(L[0, 0]))
    axc.set_ylim(-1.15 * L[1, 1], 1.2 * L[1, 1])
    f.text("濃い箱の $N$ = %d 個だけを解き、外は同じ箱の繰り返しとして扱う。" % NE,
           9.8, SUB)

    f2 = Flow(fig, 4.95, y, 3.9)
    f2.text("面積と格子定数", 11.4, AI, "bold", gap=0.12)
    f2.text("電子 1 個あたりの面積が $\\pi$（コード単位）なので "
            "$A_{\\rm sc} = \\pi N$。", 10.0)
    f2.text("結晶相では電子は三角格子を組む。その格子定数 $a_{ee}$ は\n"
            "「原始セル 1 個に電子 1 個」から決まる：", 10.0)
    f2.eq(r"$\dfrac{\sqrt{3}}{2}a_{ee}^{2} = \pi$"
          r"　$\Rightarrow$　$a_{ee} = \sqrt{2\pi/\sqrt{3}} = 1.905$", 12.0, gap=0.12)
    f2.text("なぜ $|\\mathbf{L}_1| = \\sqrt{N}\\,a_{ee}$ か", 10.8, AI, "bold", gap=0.10)
    f2.text("スーパーセルは、この三角格子の原始セルを $n \\times n$ 並べたもの。\n"
            "だから辺の長さは $n\\,a_{ee}$、中の電子数は $N = n^{2}$。\n"
            "$n = \\sqrt{N}$ を代入して $|\\mathbf{L}_1| = |\\mathbf{L}_2| = \\sqrt{N}\\,a_{ee}$。",
            9.8)
    f2.text("面積でも確かめられる：$A_{\\rm sc} = (\\sqrt{N}a_{ee})^{2}\\sin 60^\\circ\n"
            "= N\\cdot\\frac{\\sqrt{3}}{2}a_{ee}^{2} = \\pi N$。", 9.6, SUB, gap=0.18)
    f2.text("$N = 25$ なら $|\\mathbf{L}_1| = 5 \\times 1.905 = 9.523$。", 9.8, ENJI)

    f3 = Flow(fig, 9.15, y, 3.9)
    f3.text("電子どうしの距離をどう測るか", 11.4, AI, "bold", gap=0.12)
    f3.text("周期境界条件があるので、箱をまたいでも意味を持つ測り方が要る。\n"
            "素直なのは最小イメージ（いちばん近い像までの Euclid 距離）だが、\n"
            "これは箱の半分のところで**折れる**（付録 C）。".replace("**", ""), 9.8)
    f3.text("折れると 2 階微分が発散し、$\\nabla^{2}\\log\\psi$ を含む局所エネルギー\n"
            "が有限にならない。だから使えない。", 9.8, ENJI, gap=0.20)
    f3.text("代わりに、箱を輪に丸めたときの弦の長さを使う（周期ノルム）。", 9.8)
    f3.eq(r"$d^{\rm per} = \dfrac{1}{2\pi}\sqrt{\sum_{mn} G_{mn}"
          r"(a_m a_n + b_m b_n)}$", 12.0, gap=0.10)
    f3.text("$a_m = 1-\\cos 2\\pi f_m$、$b_m = \\sin 2\\pi f_m$、"
            "$G = \\mathbf{L}^{T}\\mathbf{L}$、$f_m$ は隔たりの分数座標。", 9.4, SUB,
            gap=0.20)
    f3.text("弧ではなく弦なので折れ点がない。近距離では Euclid と一致し、\n"
            "遠方では滑らかに頭打ちになる。その上限が", 9.8)
    f3.eq(r"$d_{\rm ref} = \dfrac{3^{3/4}}{2\sqrt{\pi}}\sqrt{N} = 0.643\sqrt{N}$"
          r"　$(N{=}25: 3.215)$", 11.6, ENJI, gap=0.10)
    f3.text("どの 2 電子も $d^{\\rm per} \\leq d_{\\rm ref}$。距離軸の天井になる。", 9.8, SUB)
    foot(fig, "⑥ 有限セルで解く", len(pages), NP)

    # ===================== p8 遮蔽アテンション ============================
    fig, y = new_page(None, pages, "⑦ 遮蔽アテンション：距離の窓を固定する",
                      lead="アテンションの重みを「内容」と「距離」に分け、"
                           "距離側をヘッドごとに固定した窓にする。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("ふつうのアテンション", 11.4, AI, "bold", gap=0.12)
    f.text("電子 $i$ が電子 $j$ をどれだけ見るかを、特徴ベクトルから作った\n"
           "$q_i$ と $k_j$ の内積 $s_{ij} = \\bar q_i\\cdot\\bar k_j$ だけで決める。\n"
           "全対を同じ土俵で見るので、学習後に「どの長さを使ったか」は残らない。", 10.0)
    f.skip(0.10)
    f.text("遮蔽アテンション", 11.4, AI, "bold", gap=0.14)
    f.eq(r"$W^{(h)}_{ij} = \alpha(s_{ij}) \;\cdot\; m_h(d_{ij})$", 16.0, gap=0.18)
    f.text("$\\alpha(s_{ij})$　内容チャネル　… 誰と誰が似ているか（スピンや局所環境）",
           10.2)
    f.text("$m_h(d_{ij})$　距離チャネル　… どれだけ離れているか（ヘッドごとに固定）",
           10.2, ENJI, gap=0.20)
    f.eq(r"$m_h(d) = \dfrac{1}{2}\left[1+\cos\left(\pi\min(d/w_h,\,1)\right)\right]$",
         13.0, gap=0.14)
    f.text("$d \\geq w_h$ で厳密に 0。$w_h$ より遠い相手は、そのヘッドからは完全に\n"
           "見えない。$w_h$ は学習させず固定してある。", 10.0, ENJI)

    f2 = Flow(fig, 7.05, y, 5.9)
    axw = f2.axes(2.25, 5.4, x_in=7.15, gap=0.72, pre=0.30)
    xs = np.linspace(0, dref * 1.02, 300)
    for i, hh in enumerate(order):
        axw.plot(xs, 0.5 * (1 + np.cos(np.pi * np.clip(xs / w[hh], 0, 1))),
                 lw=1.5, color=plt.cm.viridis(i / (NH - 1)))
    axw.set_xlim(0, dref * 1.02); axw.set_ylim(0, 1.06)
    axw.set_xlabel("2 電子の距離 $d$（周期ノルム）", fontsize=9.2, labelpad=1)
    axw.set_ylabel("$m_h(d)$", fontsize=9.2, labelpad=1)
    axw.tick_params(labelsize=8.2); axw.grid(alpha=.2, lw=.6)
    axw.set_title("%d 本の窓を $[%.2f,\\ %.2f]$ に等比で並べる"
                  % (NH, w[order[0]], w[order[-1]]), fontsize=9.6, pad=4)
    f2.text("なぜ固定するか", 11.4, AI, "bold", gap=0.12)
    f2.text("$w_h$ を学習させると、その値は「最適化の結果」であって「波動関数が\n"
            "使っている長さ」とは言えなくなる。固定すれば $w_h$ は動かない目盛りに\n"
            "なり、ヘッド番号がそのまま長さになる。", 10.0)
    f2.text("3 つの相で梯子が完全に共通なので、曲線の違いはそのまま波動関数の\n"
            "違いになる（チェックポイントで $10^{-5}$ まで確認済み）。", 10.0, ENJI,
            gap=0.20)
    f2.text("窓の上限は $d_{\\rm ref}$ で止める。それを超えた窓はどの対も必ず含むので、\n"
            "距離で選ぶ働きを失う。", 9.8, SUB)
    foot(fig, "⑦ 遮蔽アテンション", len(pages), NP)

    # ===================== p9 層とヘッド・どこに出るか =====================
    fig, y = new_page(None, pages, "⑧ 層とヘッド：$W^{(h)}_{ij}$ はどこに出てくるのか",
                      lead="ヘッドごとの寄与を見ることに意味があるのは、"
                           "各ヘッドが独立に 1 本のベクトルを残差ストリームへ足し込むから。")
    f = Flow(fig, 0.36, y, 12.6)
    axs = f.axes(2.35, 8.6, x_in=0.55, gap=0.28)
    stream_panel(axs)
    g1 = Flow(fig, 0.55, f.y, 5.6)
    g1.text("流れ", 11.4, AI, "bold", gap=0.12)
    g1.eq(r"$c_h(i) = \left[\,{\rm gate}\odot{\rm TanhNorm}"
          r"\left(\sum_j W^{(h)}_{ij}\, v_j\right)\right] W_O^{(h)}$", 12.0, gap=0.10)
    g1.eq(r"$\mathbf{h}_i^{(1)} = \mathbf{h}_i^{(0)} + e^{s_O}\sum_h c_h(i)$", 13.0,
          gap=0.12)
    g1.text("更新された特徴 $\\mathbf{h}_i^{(1)}$ が行列式に入り、"
            "$\\psi_\\vartheta$ が決まる。\n"
            "$\\sum_h$ は単なる和なので、**ヘッドの寄与は分離できる**。"
            .replace("**", ""), 10.0, ENJI)

    g2 = Flow(fig, 6.85, f.y, 6.1)
    g2.text("層とは", 11.4, AI, "bold", gap=0.12)
    g2.text("上の図 1 枚が「1 層」。2 層重ねると、電子 $i$ の特徴が $j$ の隣人にも\n"
            "依存しはじめ、窓 $w_h$ と「実際に届く距離」がずれる。\n"
            "この研究では読み出しを濁らせないため **1 層**にしてある。"
            .replace("**", ""), 10.0, gap=0.20)
    g2.text("ヘッドとは", 11.4, AI, "bold", gap=0.12)
    g2.text("1 つの層の中の独立な「見方」のチャネル。ふつうは $q, k$ の射影が違う\n"
            "だけだが、ここでは加えて距離の窓 $w_h$ を固定してある。だからヘッド\n"
            "番号がそのまま長さの目盛りになる。", 10.0)
    foot(fig, "⑧ 層とヘッド", len(pages), NP)

    # ===================== p10 なぜ分けるのか =============================
    fig, y = new_page(None, pages, "⑨ なぜ「窓ごとに分けて $q\\cdot k$ と寄与率を見る」のか",
                      lead="内容と距離を分離したうえで、距離ごとにヘッドを割り当てる。"
                           "そうすると何が測れるようになるのか。\n" + COND)
    f = Flow(fig, 0.36, y, 6.1)
    f.text("2 つのチャネルは別のことを見ている", 11.4, AI, "bold", gap=0.14)
    f.text("$\\alpha(s_{ij})$（$q\\cdot k$）　… **内容**。誰と誰が似ているか。"
           "スピンや\n　　局所環境で決まり、距離の情報は持っていない。"
           .replace("**", ""), 10.0)
    f.text("$m_h(d_{ij})$（窓）　… **距離**。どれだけ離れているか。"
           "内容の情報は\n　　いっさい持っていない。".replace("**", ""), 10.0, gap=0.22)
    f.text("実測でも、$\\alpha$ の分散のうち距離で説明できるのは 1〜2%、スピンで\n"
            "7.5〜10%、残り約 90% はどちらでもない。つまり $\\alpha$ は距離を\n"
            "ほとんど見ていない。役割分担がきれいに立っている。", 9.8, SUB)
    f.text("知りたいこと", 11.4, AI, "bold", gap=0.14)
    f.text("「この波動関数は、どのくらいの距離までの相関を実際に使っているか」", 10.6, ENJI)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("分けるとなぜ測れるのか", 11.4, AI, "bold", gap=0.14)
    f2.text("窓を分けずに 1 つのアテンションで全対を見ると、出力は「全部の距離を\n"
            "混ぜた結果」になり、どの距離が効いたかを後から分離できない。", 10.0)
    f2.text("窓をヘッドごとに固定して分けると：", 10.0, gap=0.12)
    f2.text("　・ヘッド $h$ は $d < w_h$ の相手からしか情報を取れない\n"
            "　・出力への寄与は $\\sum_h c_h$ という和なので、$h$ ごとに切り出せる\n"
            "　・したがって「距離 $w_h$ 以内の情報が出力にどれだけ効いたか」が\n"
            "　　ヘッドごとの寄与率として直接読める", 10.0, ENJI, gap=0.22)
    f2.text("$q\\cdot k$ マトリクスを見る理由", 11.4, AI, "bold", gap=0.12)
    f2.text("寄与率は 1 ヘッドあたり 1 個の数字なので、「なぜその値になったか」\n"
            "までは分からない。マトリクスを見れば、窓の外に落ちた相手の量、\n"
            "自分自身をどれだけ使っているか、スピンブロックの違い、が見える。\n"
            "寄与率の数字の裏付けを取るために見る。", 10.0)
    foot(fig, "⑨ なぜ窓ごとに分けるのか", len(pages), NP)

    # ===================== p11 q・k の読み方 ==============================
    fig, y = new_page(None, pages, "⑩ $q\\cdot k$ マトリクスの読み方",
                      lead="学習後の重み $W^{(h)}_{ij}$ を、ヘッドごとに行列として描いたもの。"
                           "1 つの電子配置について示す。\n" + COND)
    hsel = [int(t) for t in a.heads.split(",")]
    labs = ["窓が狭いヘッド", "窓が広いヘッド"]
    for k, hi in enumerate(hsel):
        hh = order[hi]; M = W1[hh]
        off = ~np.eye(NE, dtype=bool)
        vmax = float(np.quantile(M[off], 0.995)) or 1.0
        axm = fig.add_axes([(0.55 + k * 3.30) / WF, 1 - (y + 3.0) / HF, 2.95 / WF, 2.95 / HF])
        axm.imshow(np.where(M > 1e-12, M, np.nan), cmap=CM, vmin=0, vmax=vmax,
                   interpolation="nearest")
        axm.set_facecolor("#e9e7e2")
        axm.axhline(nup - .5, color="#e07b39", lw=1.2)
        axm.axvline(nup - .5, color="#e07b39", lw=1.2)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_edgecolor(HAI); sp.set_linewidth(.8)
        axm.set_title("%s　$w_h$ = %.2f" % (labs[k], w[hh]), fontsize=10.6, pad=5)
    fig.text(0.55 / WF, 1 - (y + 3.28) / HF,
             "行 ＝ 見る側の電子 $i$、　列 ＝ 見られる側の電子 $j$"
             "（↑%d 個 → ↓%d 個 の順）" % (nup, NE - nup),
             fontsize=10.0, color=INK, va="top")
    g = Flow(fig, 7.45, y, 5.5)
    g.text("見るところ", 11.4, AI, "bold", gap=0.14)
    g.text("橙の線　スピンの境界。左上が ↑↑、右下が ↓↓、非対角が ↑↓ ブロック。", 10.0)
    g.text("灰色　窓の外（$d_{ij}\\geq w_h$ で $W = 0$）。この灰色の量が、そのヘッド\n"
           "　　の「見える範囲」をそのまま表している。", 10.0)
    g.text("対角　$W_{ii} = \\alpha(s_{ii})\\cdot m_h(0) = \\alpha(s_{ii})$。$m_h(0)=1$ なので\n"
           "　　必ず窓の中。その電子が自分自身の情報をどれだけ使っているか。", 10.0)
    g.text("ブロックの濃淡差　同スピンと異スピンの扱いの違い。$m_h$ は距離しか\n"
           "　　見ないので、スピンの構造はすべて $\\alpha$ 側が担っている。", 10.0,
           gap=0.22)
    g.text("色の上限はヘッドごとに、非対角成分の 99.5 パーセンタイルで揃えてある。\n"
           "同じヘッドなら相を跨いで直接比べられる。", 9.6, SUB)
    foot(fig, "⑩ $q\\cdot k$ マトリクスの読み方", len(pages), NP)

    # ===================== p12 液体と結晶 =================================
    wv, qpL, qpC, qsL, qsC, PhL, PhC = D
    fig, y = new_page(None, pages, "⑪ 液体と結晶で何が起きるか：予想と実測",
                      lead="先に物理から予想を立て、そのあと実測を並べる。"
                           "合わなかったところも書く。\n" + COND)
    f = Flow(fig, 0.36, y, 5.9)
    f.text("液体で予想されること", 11.4, AI, "bold", gap=0.12)
    f.text("電子は非局在で、決まった座る場所がない。相関を作っているのは主に\n"
           "短距離のクーロンホール（互いに避ける効果）。近い相手が実際にいるので、\n"
           "狭い窓のヘッドにも 2 体の仕事があるはず。", 10.0)
    f.text("結晶で予想されること", 11.4, AI, "bold", gap=0.12)
    f.text("各電子は格子点のまわりに束縛され、まわりの電子が作る谷の中で振動する\n"
           "（最低次では調和振動子）。すると", 10.0)
    f.text("(a) 最近接距離が $a_{ee}$ 程度にそろい、それより近づかない。窓がそれより\n"
           "　　狭いヘッドには相手が 1 個も入らない → 2 体寄与が消える。\n"
           "(b) 支配的なのは「自分の格子点からのずれ」という 1 体的な情報。\n"
           "　　だから狭いヘッドは 1 体の仕事に振り替わる。", 10.0, ENJI, gap=0.22)
    f.text("実測（全 16 ヘッドのうち代表 5 本。全部は付録 E）", 11.0, AI, "bold", gap=0.12)
    rows = [["$w_h$", "2 体寄与 %", "", "1 体寄与 %（全体で規格化）", ""],
            ["", "液体", "結晶", "液体", "結晶"]]
    idx = [0, 3, 7, 11, 15]
    for i in idx:
        rows.append(["%.2f" % wv[i], "%.2f" % qpL[i], "%.2f" % qpC[i],
                     "%.2f" % qsL[i], "%.2f" % qsC[i]])
    xs = [0.55, 1.65, 2.55, 3.75, 4.90]
    yy = f.y
    for j, c in enumerate(rows[0]):
        fig.text(xs[j] / WF, 1 - yy / HF, c, fontsize=9.4, color=SUB, weight="bold", va="top")
    for j, c in enumerate(rows[1]):
        fig.text(xs[j] / WF, 1 - (yy + 0.24) / HF, c, fontsize=9.2, color=SUB, va="top")
    fig.add_artist(plt.Line2D([0.55 / WF, 5.95 / WF], [1 - (yy + 0.46) / HF] * 2,
                              color=RULE, lw=0.9))
    for i, r in enumerate(rows[2:]):
        for j, c in enumerate(r):
            fig.text(xs[j] / WF, 1 - (yy + 0.58 + i * 0.245) / HF, c, fontsize=9.8,
                     color=ENJI if i in (0, 4) else INK, va="top")

    f2 = Flow(fig, 6.85, y, 6.1)
    f2.text("(a) は成り立っている", 11.4, MIDORI, "bold", gap=0.12)
    f2.text("いちばん狭いヘッド（$w_h = 1.80$）で 2 体寄与は液体 3.09% に対し結晶\n"
            "0.19%。16 倍ちがう。しかもこの傾向は $w_h \\leq 2.27$ の 7 本すべてで\n"
            "同じ向きで、$w_h \\approx 2.3$ で入れ替わり、上では逆転する（付録 E）。\n"
            "選んだ 5 本だけの話ではない。", 10.0)
    f2.text("(b) は成り立っていない", 11.4, ENJI, "bold", gap=0.12)
    f2.text("1 体寄与を全体で規格化して測ると、$w_h = 1.80$ で液体 9.55% に対し\n"
            "結晶 2.49%。結晶のほうが**少ない**。全ヘッドの合計でも液体 55.0% に\n"
            "対し結晶 48.8% で、やはり結晶のほうが少ない。".replace("**", ""), 10.0)
    f2.text("前に「自分自身の割合が 80% → 87% に増える」と書いたが、あれは\n"
            "そのヘッド内での割合で、2 体成分が消えれば自動的に上がる量だった。\n"
            "規格化の見かけであって、1 体の仕事が増えた証拠にはならない。",
            9.8, ENJI, gap=0.20)
    f2.text("正しい言い方", 11.4, AI, "bold", gap=0.12)
    f2.text("結晶では狭いヘッドの仕事が 2 体も 1 体も減る。ただし 2 体の減り方\n"
            "（16 倍）が 1 体（3.8 倍）よりずっと激しいので、残った僅かな仕事は\n"
            "相対的に 1 体寄りになる。「1 体へ振り替わる」ではなく\n"
            "「まず 2 体が先に消える」が正しい。", 10.0, ENJI)
    foot(fig, "⑪ 液体と結晶", len(pages), NP)

    # ===================== p13 読み出せるもの =============================
    fig, y = new_page(None, pages, "⑫ 読み出せるもの：寄与率と $\\langle w\\rangle_q$",
                      lead="ヘッドごとの寄与を窓の長さの上に並べると、"
                           "「波動関数が使っている長さ」が 1 本の数字になる。\n" + COND)
    f = Flow(fig, 0.36, y, 6.4)
    items = []
    for part in a.runs.split(","):
        lab, ff = part.rsplit("=", 1)
        items.append((lab, qload(ff, 48)))
    axq = f.axes(2.85, 5.9, x_in=0.85, gap=0.70)
    for i, (lab, d) in enumerate(items):
        axq.plot(d["w"], d["qp"], "-", marker="os^"[i], ms=5.0, lw=1.9,
                 color=[AI, "#c98a12", ENJI][i], label=lab)
    axq.set_xlabel("固定した窓 $w_h$（周期ノルム, $r_s a_B^{*}$）", fontsize=9.8, labelpad=2)
    axq.set_ylabel("2 体寄与率 $q_h$ [%]", fontsize=9.8, labelpad=2)
    axq.tick_params(labelsize=8.6); axq.grid(alpha=.22, lw=.7)
    axq.legend(fontsize=10.0, frameon=False, loc="upper left")
    wq = [float((d["qp"] * d["w"]).sum() / d["qp"].sum()) for _, d in items]
    dn = [d["d_nn"] for _, d in items]
    f.text("液体 → 境界 → 結晶 で　$\\langle w\\rangle_q$ = "
           + " → ".join("%.2f" % v for v in wq)
           + "　　実測の最近接距離 = " + " → ".join("%.2f" % v for v in dn),
           10.2, ENJI)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("寄与率の定義（詳しくは付録 D）", 11.4, AI, "bold", gap=0.14)
    f2.text("ヘッド $h$ が電子 $i$ の特徴に足し込むベクトルを $c_h(i)$ とする（⑧の図）。\n"
            "これを 2 つに分ける：", 10.0)
    f2.eq(r"$c_h = c_h^{\rm self} + c_h^{\rm pair}$", 13.0, gap=0.10)
    f2.text("$c_h^{\\rm self}$　… 重みを対角だけにした（$W_{ij}\\to W_{ii}\\delta_{ij}$）\n"
            "　　　　ときの寄与。自分自身の情報だけで作られる部分。\n"
            "$c_h^{\\rm pair}$　… 残り。他の電子から来た部分＝相関を運んでいる部分。",
            9.8, gap=0.20)
    f2.eq(r"$q_h = \|c_h^{\rm pair}\|^{2} \,/\, \sum_{h'}\|c_{h'}^{\rm pair}\|^{2}$",
          13.0, gap=0.10)
    f2.text("ノルムは全電子・全配置で平均したもの。2 乗を使うのは、寄与が\n"
            "ベクトルの足し算なのでエネルギー的な重み（大きさの 2 乗）で\n"
            "測るのが自然だから。", 9.8, gap=0.20)
    f2.eq(r"$\langle w\rangle_q = \sum_h q_h\, w_h$", 14.0, gap=0.12)
    f2.text("$q_h$ を重みにした窓の平均。「波動関数が 2 体相関に使っている\n"
            "典型的な長さ」と読める。", 10.0, ENJI)
    foot(fig, "⑫ 読み出せるもの", len(pages), NP)

    # ===================== p14 対照実験 ===================================
    nullL = 100 * PhL / PhL.sum(); nullC = 100 * PhC / PhC.sum()
    wqr = lambda q: float((q * wv).sum() / q.sum())                     # noqa: E731
    tv = lambda x, y_: float(0.5 * np.abs(x - y_).sum() / 100)          # noqa: E731
    fig, y = new_page(None, pages, "⑬ 対照実験：ただの幾何で説明できてしまわないか",
                      lead="結晶では電子が離れているので、狭い窓に相手が入らないのは当たり前とも言える。"
                           "「波動関数を見ている」と言えるのかを確かめる。\n" + COND)
    f = Flow(fig, 0.36, y, 6.3)
    f.text("帰無モデル", 11.4, AI, "bold", gap=0.12)
    f.text("「各ヘッドの寄与は、そのヘッドの窓に入る対の量だけで決まる」\n"
           "という、波動関数の中身をいっさい使わないモデルを立てる：", 10.0)
    f.eq(r"$P_h = \langle m_h(d_{ij})\rangle_{\rm pairs}$,　"
         r"$q_h^{\rm null} = P_h \,/\, \sum_{h'} P_{h'}$", 12.0, gap=0.12)
    f.text("$P_h$ は電子配置（＝密度分布）だけから計算でき、ネットワークの\n"
           "重みを一切使わない。もし実測がこれで再現できてしまうなら、\n"
           "読み出しは幾何を測っているだけということになる。", 9.8, gap=0.20)
    axn = f.axes(2.55, 5.8, x_in=0.70, gap=0.68)
    axn.plot(wv, qpL, "-o", color=AI, lw=1.9, ms=4.6, label="液体 実測")
    axn.plot(wv, nullL, "--", color=AI, lw=1.6, alpha=.75, label="液体 帰無モデル")
    axn.plot(wv, qpC, "-^", color=ENJI, lw=1.9, ms=4.6, label="結晶 実測")
    axn.plot(wv, nullC, "--", color=ENJI, lw=1.6, alpha=.75, label="結晶 帰無モデル")
    axn.set_xlabel("固定した窓 $w_h$", fontsize=9.4, labelpad=1)
    axn.set_ylabel("2 体寄与率 [%]", fontsize=9.4, labelpad=1)
    axn.tick_params(labelsize=8.4); axn.grid(alpha=.22, lw=.7)
    axn.legend(fontsize=8.6, frameon=False, loc="upper left", ncol=2)

    f2 = Flow(fig, 7.15, y, 5.8)
    f2.text("結果", 11.4, MIDORI, "bold", gap=0.14)
    tb = [["", "帰無モデル", "実測", "比"],
          ["$\\langle w\\rangle_q$ の液体→結晶の増分",
           "%+.3f" % (wqr(nullC) - wqr(nullL)), "%+.3f" % (wqr(qpC) - wqr(qpL)),
           "%.1f 倍" % ((wqr(qpC) - wqr(qpL)) / (wqr(nullC) - wqr(nullL)))],
          ["液体と結晶の配分の差（全変動距離）",
           "%.3f" % tv(nullL, nullC), "%.3f" % tv(qpL, qpC),
           "%.1f 倍" % (tv(qpL, qpC) / tv(nullL, nullC))]]
    yy = f2.y
    for j, (xx, c) in enumerate(zip([7.30, 10.05, 11.15, 12.10], tb[0])):
        fig.text(xx / WF, 1 - yy / HF, c, fontsize=9.6, color=SUB, weight="bold", va="top")
    fig.add_artist(plt.Line2D([7.30 / WF, 12.85 / WF], [1 - (yy + 0.22) / HF] * 2,
                              color=RULE, lw=0.9))
    for i, r in enumerate(tb[1:]):
        for j, (xx, c) in enumerate(zip([7.30, 10.05, 11.15, 12.10], r)):
            fig.text(xx / WF, 1 - (yy + 0.36 + i * 0.32) / HF, c, fontsize=9.8,
                     color=ENJI if j == 3 else INK, va="top")
    f2.y = yy + 0.36 + 2 * 0.32 + 0.30
    f2.text("幾何だけでは、実測の変化の 4 分の 1 程度しか説明できない。\n"
            "残りはネットワークが学習した中身から来ている。", 10.4, ENJI, gap=0.24)
    f2.text("いちばんはっきりする点", 11.4, AI, "bold", gap=0.12)
    f2.text("最上段（$w_h = 3.20$）では窓に入る対の量がほぼ同じ\n"
            "（液体 %.1f%% / 結晶 %.1f%%、差 %.1f 点）なのに、実測の寄与率は\n"
            "%.2f%% / %.2f%% と %.2f 倍ちがう。幾何では説明できない。"
            % (PhL[-1], PhC[-1], PhL[-1] - PhC[-1], qpL[-1], qpC[-1], qpC[-1] / qpL[-1]),
            10.0, gap=0.24)
    f2.text("残る限界（正直に）", 11.4, ENJI, "bold", gap=0.12)
    f2.text("・すべて 1 シード。誤差棒がない。\n"
            "・結晶側は step 6000 でまだ寄与配分が動いている（未収束）。\n"
            "・$N = 25$ の 1 サイズのみ。", 9.8, SUB)
    foot(fig, "⑬ 対照実験", len(pages), NP)

    # ===================== 付録 A モアレの導出 =============================
    fig, y = new_page(None, pages, "付録 A　モアレポテンシャルの導出",
                      lead="なぜ $V(\\mathbf{r})$ が 3 本の cos の和で書けるのか。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("1. うなりの周期", 11.4, AI, "bold", gap=0.12)
    f.text("層 1 の逆格子ベクトルを $\\mathbf{g}$、層 2 を $\\mathbf{g}'$ とする。\n"
           "層 2 は層 1 を角度 $\\theta$ 回し、長さを $(1+\\delta)$ 倍したもの:", 10.0)
    f.eq(r"$\mathbf{g}' = (1+\delta)^{-1} R_\theta\, \mathbf{g}$", 12.5, gap=0.10)
    f.text("2 つの層の効果は差ベクトルで効く：", 10.0, gap=0.10)
    f.eq(r"$\mathbf{G} = \mathbf{g}' - \mathbf{g}$,　"
         r"$|\mathbf{G}| \simeq |\mathbf{g}|\sqrt{\theta^{2}+\delta^{2}}$", 12.5, gap=0.10)
    f.text("$|\\mathbf{g}| = 4\\pi/(\\sqrt{3} a_0)$ を入れ、$|\\mathbf{G}| = 4\\pi/(\\sqrt{3} a_M)$\n"
           "と書き直すと", 9.8)
    f.eq(r"$a_M \simeq a_0 \,/\, \sqrt{\theta^{2}+\delta^{2}}$", 13.0, ENJI, gap=0.14)
    f.text("$\\theta$ も $\\delta$ も小さいので $a_M \\gg a_0$。これがモアレ超格子。", 9.8, SUB)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("2. なぜ 3 本の cos か", 11.4, AI, "bold", gap=0.12)
    f2.text("局所の積層のずれ $\\mathbf{d}(\\mathbf{r}) = (\\theta\\,\\hat z\\times + \\delta)"
            "\\mathbf{r}$ はモアレ周期でゆっくり変わる。\n"
            "バンド端はこのずれの関数なので、$V$ もモアレ周期の周期関数になる。\n"
            "Fourier 展開して最短の逆格子ベクトルだけ残すと", 10.0)
    f2.eq(r"$V(\mathbf{r}) = \sum_{j} V_j\, e^{i \mathbf{G}_j\cdot\mathbf{r}}$", 12.5,
          gap=0.10)
    f2.text("三角格子の最短逆格子ベクトルは 6 本あり、$\\pm$ で対になる。\n"
            "$V$ が実数であることから $V_{-j} = V_j^{*}$。$V_j = -U e^{i\\phi}$ と\n"
            "書けば 3 対がまとまって", 10.0)
    f2.eq(r"$V(\mathbf{r}) = -2U \sum_{j=1,2,3} "
          r"\cos(\mathbf{G}_j\cdot\mathbf{r}+\phi)$", 13.0, ENJI, gap=0.14)
    f2.text("$U$ は深さ、$\\phi$ は谷の位置と形を決める。$\\phi = 0$ なら三角格子の\n"
            "点に深い谷、$\\phi$ をずらすとハニカム型に分裂していく。", 9.8, SUB,
            gap=0.20)
    f2.text("TMD ヘテロ二層の第一原理計算では $U \\sim 10$〜$20$ meV、\n"
            "$a_M \\sim 5$〜$10$ nm のオーダー。", 9.6, SUB)
    foot(fig, "付録 A　モアレポテンシャルの導出", len(pages), NP)

    # ===================== 付録 B VMC と ∇^2 ==============================
    fig, y = new_page(None, pages, "付録 B　変分モンテカルロと $\\nabla^{2}\\log\\psi$",
                      lead="なぜ波動関数が 2 階微分できないと困るのか。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("1. 変分原理", 11.4, AI, "bold", gap=0.12)
    f.eq(r"$E[\vartheta] = \dfrac{\langle\psi_\vartheta|\hat H|\psi_\vartheta\rangle}"
         r"{\langle\psi_\vartheta|\psi_\vartheta\rangle} \;\geq\; E_0$", 14.0, gap=0.12)
    f.text("任意の試行関数について基底状態エネルギー以上。だから下げればよい。", 10.0)
    f.text("2. モンテカルロで書き直す", 11.4, AI, "bold", gap=0.12)
    f.text("確率 $p(\\mathbf{R}) = |\\psi_\\vartheta(\\mathbf{R})|^{2}/\\int|\\psi_\\vartheta|^{2}$ "
           "を導入すると", 10.0)
    f.eq(r"$E[\vartheta] = \int p(\mathbf{R})\, E_{\rm loc}(\mathbf{R})\, d\mathbf{R}"
         r" = \langle E_{\rm loc}\rangle_{p}$", 12.5, gap=0.10)
    f.eq(r"$E_{\rm loc}(\mathbf{R}) = \dfrac{\hat H \psi_\vartheta(\mathbf{R})}"
         r"{\psi_\vartheta(\mathbf{R})}$", 13.5, gap=0.12)
    f.text("$p$ からのサンプリング（Metropolis）で期待値を評価する。", 10.0)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("3. ここで 2 階微分が出る", 11.4, AI, "bold", gap=0.12)
    f2.text("運動項を $\\psi$ で割ると", 10.0)
    f2.eq(r"$\dfrac{-\frac{1}{2}\nabla^{2}\psi}{\psi}"
          r" = -\dfrac{1}{2}\left[\nabla^{2}\log\psi"
          r" + \left(\nabla\log\psi\right)^{2}\right]$", 13.0, gap=0.12)
    f2.text("（$\\nabla^{2}\\psi/\\psi = \\nabla^{2}\\log\\psi + (\\nabla\\log\\psi)^{2}$ から）",
            9.6, SUB, gap=0.18)
    f2.text("したがって局所エネルギーは", 10.0, gap=0.08)
    f2.eq(r"$E_{\rm loc} = -\dfrac{1}{2}\sum_i \left[\nabla_i^{2}\log\psi"
          r" + (\nabla_i\log\psi)^{2}\right] + r_s \sum_{i<j} r_{ij}^{-1}"
          r" + \sum_i V(\mathbf{r}_i)$", 11.5, gap=0.14)
    f2.text("4. 何が困るか", 11.4, ENJI, "bold", gap=0.12)
    f2.text("$\\log\\psi$ が距離 $d_{ij}$ を通して座標に依存している以上、$d_{ij}$ 自体が\n"
            "2 階微分可能でないと $\\nabla^{2}\\log\\psi$ が定義できない。折れ点があると\n"
            "そこで 2 階微分が発散し、$E_{\\rm loc}$ が発散する。", 10.0)
    f2.text("モンテカルロでは折れ点の上を有限確率で通るので、$E_{\\rm loc}$ の分散が\n"
            "発散して推定値が定まらない。「たまに巨大な値が出る」形で壊れる。",
            10.0, ENJI)
    foot(fig, "付録 B　変分モンテカルロ", len(pages), NP)

    # ===================== 付録 C 最小イメージの折れ ======================
    fig, y = new_page(None, pages, "付録 C　最小イメージはなぜ「折れる」のか",
                      lead="1 次元で見ると一目で分かる。$x$ は 2 電子の隔たり、"
                           "箱の長さを $\\ell$ とする。")
    f = Flow(fig, 0.36, y, 6.2)
    f.text("最小イメージ距離の定義", 11.4, AI, "bold", gap=0.12)
    f.text("電子 $j$ は $x, x\\pm\\ell, x\\pm2\\ell,\\dots$ に像を持つ。そのうち\n"
           "いちばん近いものまでの距離を取る：", 10.0)
    f.eq(r"$d^{\rm mi}(x) = \min_{n\in\mathbb{Z}} |x + n\ell|"
         r" = \min(x,\; \ell - x)$　$(0\leq x\leq \ell)$", 12.0, gap=0.12)
    f.text("$x < \\ell/2$ では自分自身の像（$n=0$）が最短で $d = x$、\n"
           "$x > \\ell/2$ では隣の像（$n=-1$）が最短で $d = \\ell - x$。\n"
           "$x = \\ell/2$ でどちらが最短かが入れ替わる。", 10.0, gap=0.20)
    f.text("そこで何が起きるか", 11.4, ENJI, "bold", gap=0.12)
    f.text("傾きが $+1$ から $-1$ へ不連続に飛ぶ：", 10.0, gap=0.08)
    f.eq(r"$d\,d^{\rm mi}/dx = +1$　$(x<\ell/2)$,　"
         r"$-1$　$(x>\ell/2)$", 12.5, gap=0.14)
    f.text("1 階微分が跳ぶということは、2 階微分がそこでデルタ関数になる：", 10.0,
           gap=0.08)
    f.eq(r"$\dfrac{d^{2} d^{\rm mi}}{dx^{2}} = -2\,\delta(x-\ell/2)$", 13.0, ENJI,
         gap=0.14)
    f.text("$\\nabla^{2}\\log\\psi$ にこれが入るので、局所エネルギーが発散する。", 10.0)

    f2 = Flow(fig, 7.05, y, 5.9)
    axm2 = f2.axes(2.55, 5.4, x_in=7.20, gap=0.68)
    ell = 1.0
    xx = np.linspace(0, ell, 600)
    axm2.plot(xx, np.minimum(xx, ell - xx), "-", color=HAI, lw=2.6,
              label="最小イメージ $d^{\\rm mi}$")
    axm2.plot(xx, (ell / np.pi) * np.sin(np.pi * xx / ell), "-", color=AI, lw=2.6,
              label="周期ノルム $d^{\\rm per}$")
    for n in (0, -1):
        axm2.plot(xx, np.abs(xx + n * ell), ":", color="#c9a227", lw=1.4)
    axm2.plot([ell / 2], [ell / 2], "o", ms=10, mfc="none", mec=ENJI, mew=2.4, zorder=6)
    axm2.annotate("ここで像が入れ替わり\n傾きが $+1 \\to -1$ に飛ぶ",
                  xy=(ell / 2, ell / 2), xytext=(ell * 0.12, ell * 0.40),
                  fontsize=9.0, color=ENJI, linespacing=1.7,
                  arrowprops=dict(arrowstyle="-", lw=.9, color=ENJI, shrinkA=4, shrinkB=6))
    axm2.set_xlim(0, ell); axm2.set_ylim(0, ell * 0.62)
    axm2.set_xticks([0, ell / 2, ell]); axm2.set_xticklabels(["0", "$\\ell/2$", "$\\ell$"])
    axm2.set_xlabel("2 電子の隔たり $x$", fontsize=9.4, labelpad=1)
    axm2.set_ylabel("測った距離", fontsize=9.4, labelpad=1)
    axm2.tick_params(labelsize=8.4); axm2.grid(alpha=.22, lw=.7)
    axm2.legend(fontsize=9.0, frameon=False, loc="lower center")
    f2.text("点線は各像までの距離 $|x|$ と $|x-\\ell|$。灰色はその下側の包絡線＝\n"
            "最小イメージ。包絡線なので必ず角ができる。", 9.6, SUB, gap=0.20)
    f2.text("周期ノルムはこの角を丸めたもの", 11.4, AI, "bold", gap=0.12)
    f2.text("$d^{\\rm per} = (\\ell/\\pi)\\,|\\sin(\\pi x/\\ell)|$ は同じ周期性を持ちながら\n"
            "全域で滑らか。$x \\to 0$ で傾きが 1 に一致するので、近距離では\n"
            "Euclid 距離と同じ振る舞いをする（カスプ条件が保たれる）。", 10.0)
    foot(fig, "付録 C　最小イメージの折れ", len(pages), NP)

    # ===================== 付録 D 寄与率の定義 ============================
    fig, y = new_page(None, pages, "付録 D　寄与率 $q_h$ の定義",
                      lead="「ヘッド $h$ がどれだけ効いたか」をどう数値にするか。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("1. 出力は和の形をしている", 11.4, AI, "bold", gap=0.12)
    f.eq(r"$\mathbf{h}_i^{(1)} = \mathbf{h}_i^{(0)} + e^{s_O}\sum_h c_h(i)$", 13.0,
         gap=0.12)
    f.text("残差ストリームへの足し込みが単なる和なので、ヘッドごとに切り分けられる。\n"
           "これが「ヘッドごとの寄与」が意味を持つ理由（⑧）。", 10.0, gap=0.20)
    f.text("2. 1 体と 2 体に分ける", 11.4, AI, "bold", gap=0.12)
    f.text("$c_h(i)$ には、相手を見て得た情報と、自分自身の情報が混ざっている。\n"
           "重みを対角だけに落とした量を $c_h^{\\rm self}$ と定義する：", 10.0)
    f.eq(r"$W^{(h)}_{ij} \;\longrightarrow\; W^{(h)}_{ii}\,\delta_{ij}$"
         r"　のときの寄与 $= c_h^{\rm self}$", 11.5, gap=0.12)
    f.eq(r"$c_h^{\rm pair} = c_h - c_h^{\rm self}$", 13.0, ENJI, gap=0.14)
    f.text("$c_h^{\\rm pair}$ が「他の電子から来た情報」＝相関を運んでいる部分。", 10.0)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("3. 大きさで測る", 11.4, AI, "bold", gap=0.12)
    f2.eq(r"$q_h = \dfrac{\langle\|c_h^{\rm pair}(i)\|^{2}\rangle_{i,\mathbf{R}}}"
          r"{\sum_{h'}\langle\|c_{h'}^{\rm pair}(i)\|^{2}\rangle_{i,\mathbf{R}}}$",
          14.0, gap=0.14)
    f2.text("$\\langle\\cdot\\rangle_{i,\\mathbf{R}}$ は全電子と全サンプル配置についての平均。\n"
            "2 乗を使うのは、寄与がベクトルの足し算なので、大きさの 2 乗で測るのが\n"
            "自然だから（内積の非直交性は無視している。そこは近似）。", 9.8, gap=0.20)
    f2.text("4. 注意点", 11.4, ENJI, "bold", gap=0.12)
    f2.text("・$q_h$ は 2 体成分だけで規格化した割合。1 体まで含めた割合とは別物。\n"
            "　混同すると「結晶で自分自身の割合が増えた」のような誤読が起きる（⑪）。\n"
            "・ヘッドどうしの寄与ベクトルが直交しているとは限らないので、$\\sum_h q_h$\n"
            "　が全 2 体寄与に厳密に一致するわけではない。\n"
            "・窓の広いヘッドは足し込む対の数が多いぶん $\\|c_h\\|$ が自動的に大きく\n"
            "　なる。だから ⑬ の帰無モデルとの比較が要る。", 9.8, gap=0.22)
    f2.text("5. 別の測り方でも確かめてある", 11.4, AI, "bold", gap=0.12)
    f2.text("ヘッドを 1 本ずつ止めてエネルギーがどれだけ悪化するかを測ると、\n"
            "その順位は $q_h$ の順位と Spearman 相関 0.926 で一致する。\n"
            "$q_h$ は「そのヘッドを失うと困る度合い」をおおむね捉えている。", 9.8)
    foot(fig, "付録 D　寄与率の定義", len(pages), NP)

    # ===================== 付録 E 全ヘッド ================================
    fig, y = new_page(None, pages, "付録 E　全 16 ヘッドの数値（$N = 25$、窓 $[1.80,\\ 3.20]$）",
                      lead="本編の表は 5 本だけ示したので、選択の影響がないことを確かめる用。\n" + COND)
    hdr = ["$w_h$", "2 体 液体", "2 体 結晶", "比", "1 体 液体", "1 体 結晶",
           "$P_h$ 液体", "$P_h$ 結晶"]
    xs = [0.60, 2.05, 3.45, 4.75, 5.85, 7.25, 8.75, 10.25]
    fig.text(xs[0] / WF, 1 - y / HF, "（％。1 体は全体で規格化、$P_h$ は窓に入る対の割合）",
             fontsize=9.4, color=SUB, va="top")
    y += 0.34
    for j, c in enumerate(hdr):
        fig.text(xs[j] / WF, 1 - y / HF, c, fontsize=9.6, color=SUB, weight="bold", va="top")
    fig.add_artist(plt.Line2D([0.60 / WF, 11.4 / WF], [1 - (y + 0.22) / HF] * 2,
                              color=RULE, lw=0.9))
    for i in range(16):
        rr = ["%.3f" % wv[i], "%.2f" % qpL[i], "%.2f" % qpC[i],
              "%.2f" % (qpC[i] / max(qpL[i], 1e-9)),
              "%.2f" % qsL[i], "%.2f" % qsC[i], "%.1f" % PhL[i], "%.1f" % PhC[i]]
        col = ENJI if qpC[i] / max(qpL[i], 1e-9) > 1 else INK
        for j, c in enumerate(rr):
            fig.text(xs[j] / WF, 1 - (y + 0.38 + i * 0.222) / HF, c, fontsize=9.4,
                     color=col, va="top")
    yb = y + 0.38 + 16 * 0.222 + 0.28
    fig.text(0.60 / WF, 1 - yb / HF,
             "赤 ＝ 結晶のほうが 2 体寄与が大きいヘッド。$w_h \\approx 2.3$ を境に"
             "きれいに入れ替わっており、途中に例外はない。\n"
             "本編で示した 5 本は等間隔に取ったもので、傾向は 16 本すべてで同じ向きである。",
             fontsize=10.0, color=INK, va="top", linespacing=1.85)
    foot(fig, "付録 E　全ヘッドの数値", len(pages), NP)

    # ===================== 出力 ===========================================
    with PdfPages(a.out) as pdf:
        for fg in pages:
            pdf.savefig(fg)
    print("  wrote %s（%d ページ）" % (a.out, len(pages)))
    if a.pngdir:
        os.makedirs(a.pngdir, exist_ok=True)
        for i, fg in enumerate(pages, 1):
            fg.savefig(os.path.join(a.pngdir, "p%02d.png" % i), dpi=150)
        print("  wrote %s/p01..p%02d.png" % (a.pngdir, len(pages)))


if __name__ == "__main__":
    main()
