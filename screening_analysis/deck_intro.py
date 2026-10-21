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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--dens", default="", help="3 相の密度図（上段だけ切って使う）")
    ap.add_argument("--heads", default="5,15")
    ap.add_argument("--nconf", type=int, default=64)
    ap.add_argument("--out", default="deck_intro.pdf")
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

    from attn_probe import probe
    from head_contrib import cached
    from fig_qwin import load as qload, _lattice, _pnorm

    R = probe(a.folder, nconf=a.nconf)
    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    NE, NH, dref = R["NE"], R["NH"], float(R["dref"])
    nup = int((np.asarray(R["spins"][0]) > 0).sum())
    w = R["w"][0]; order = np.argsort(w)
    W1 = np.asarray(R["W"])[0, 0]                     # 1 配置ぶん [NH,NE,NE]
    L = _lattice(cfg)
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))
    CM = LinearSegmentedColormap.from_list("blu", ["#ffffff", "#cfe0f2", "#5b93cc", "#123a63"])
    pages = []

    # ============ p1  タイトル =============================================
    fig = plt.figure(figsize=(WF, HF)); fig.patch.set_facecolor("white"); pages.append(fig)
    fig.text(0.06, 0.78, "波動関数が使っている「長さ」を読み出す", fontsize=27,
             color=MIDORI, weight="bold", va="top")
    fig.text(0.06, 0.685, "遮蔽アテンションによるニューラル波動関数", fontsize=15,
             color=INK, va="top")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.645] * 2, color=RULE, lw=1.2))
    fig.text(0.06, 0.585,
             "2 次元電子系は密度を下げると、液体から Wigner 結晶へ移る。\n"
             "そのとき波動関数は「どのくらいの距離まで相手を見ている」のだろうか。",
             fontsize=12.6, color=INK, va="top", linespacing=1.95)
    fig.text(0.06, 0.415,
             "ニューラル波動関数のアテンションに、ヘッドごとに固定した距離の窓を入れる。\n"
             "どのヘッドが効いたかが、そのまま「使っている長さ」の読み出しになる。",
             fontsize=12.6, color=ENJI, va="top", linespacing=1.95)
    fig.text(0.06, 0.245,
             "① 系（2 次元電子系と $r_s$）　②相図　③モアレ三角ポテンシャル　"
             "④ハミルトニアン　⑤有限セルで解く\n"
             "⑥遮蔽アテンション　⑦$q\\cdot k$ マトリクスの読み方　⑧その解釈　⑨読み出せるもの",
             fontsize=11.0, color=SUB, va="top", linespacing=2.0)

    # ============ p2  2 次元電子系と r_s ===================================
    fig, y = new_page(pdf=None, pages=pages, title="① 2 次元電子系と $r_s$",
                      lead="扱うのは、一様な正電荷の背景の上に置かれた $N$ 個の電子。"
                           "半導体量子井戸や単層 TMD の中の電子がこれに当たる。")
    f = Flow(fig, 0.36, y, 6.0)
    f.text("密度から長さを決める", 11.4, AI, "bold", gap=0.14)
    f.text("面密度を $n_e = N/A$ とする。電子 1 個が占める面積 $1/n_e$ を円と見たときの\n"
           "半径が、その系を代表する長さである。これを $r_s a_B^{*}$ と書く。", 10.0)
    f.eq(r"$\dfrac{1}{n_e} = \pi\,(r_s a_B^{*})^{2}$", 14.0)
    f.text("$a_B^{*} = \\hbar^{2}\\varepsilon/m^{*}e^{2}$（有効ボーア半径）、"
           "$ {\\rm Ha}^{*} = e^{2}/\\varepsilon a_B^{*}$（有効ハートリー）。\n"
           "物質ごとの事情（有効質量 $m^{*}$ と誘電率 $\\varepsilon$）は全部ここに入る。",
           9.6, SUB, gap=0.26)
    f.text("$r_s$ は無次元数：クーロン / 運動エネルギーの比", 11.4, AI, "bold", gap=0.14)
    f.text("長さを $r_s a_B^{*}$ で、エネルギーを ${\\rm Ha}^{*}/r_s^{2}$ で測ると、"
           "ハミルトニアンは", 10.0)
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
    foot(fig, "① 2 次元電子系と $r_s$", 2, 10)

    # ============ p3  相図 =================================================
    fig, y = new_page(None, pages, "② $r_s$ が決める相：液体 ↔ Wigner 結晶",
                      lead="$r_s$ を上げていくと、電子は動き回る液体から、"
                           "互いに離れて三角格子に固まる Wigner 結晶へ移る。")
    f = Flow(fig, 0.36, y, 12.6)
    if a.dens and os.path.exists(a.dens):
        im = mpimg.imread(a.dens)
        h0, w0 = im.shape[:2]
        im = im[int(h0 * 0.115):int(h0 * 0.42)]
        axd = f.axes(2.55, 12.4, gap=0.10)
        axd.imshow(im); axd.axis("off")
    f.text("上：スーパーセルへ折り返した 1 体密度。$r_s = 5$ では一様、"
           "$r_s = 60$ では 25 個の輝点が三角格子を組む。", 9.6, SUB, gap=0.24)
    g1 = Flow(fig, 0.36, f.y, 6.0)
    g1.text("秩序変数", 11.4, AI, "bold", gap=0.12)
    g1.text("$\\psi_6 = \\left|\\langle e^{6i\\theta}\\rangle\\right|$"
            "　… 各電子の最近接 6 個との角度から測る六方晶秩序\n"
            "ゆらぎ/平均 $= {\\rm sd}(d_{\\rm NN})/\\langle d_{\\rm NN}\\rangle$"
            "　… Lindemann 比に相当（2 次元の融解の目安は 0.1 前後）", 9.8)
    rows = [["", "$r_s$", "$\\psi_6$", "ゆらぎ/平均", "$\\langle d_{\\rm NN}\\rangle$"],
            ["液体", "5", "0.367", "0.266", "1.177"],
            ["境界", "30", "0.408", "0.169", "1.333"],
            ["結晶", "60", "0.461", "0.141", "1.396"]]
    xs = [0.50, 1.45, 2.30, 3.20, 4.40]
    yy = g1.y
    for j, c in enumerate(rows[0]):
        fig.text(xs[j] / WF, 1 - yy / HF, c, fontsize=9.4, color=SUB, weight="bold", va="top")
    fig.add_artist(plt.Line2D([0.50 / WF, 5.6 / WF], [1 - (yy + 0.20) / HF] * 2,
                              color=RULE, lw=0.9))
    for i, r in enumerate(rows[1:]):
        for j, c in enumerate(r):
            fig.text(xs[j] / WF, 1 - (yy + 0.30 + i * 0.235) / HF, c, fontsize=9.8,
                     color=ENJI if r[0] == "結晶" else INK, va="top")

    g2 = Flow(fig, 7.05, f.y, 5.9)
    g2.text("転移はどこか", 11.4, AI, "bold", gap=0.12)
    g2.text("量子モンテカルロによる 2 次元電子ガスの見積もりでは、Wigner 結晶化は\n"
            "$r_s \\approx 31$〜$37$（Tanatar–Ceperley 1989 は $37\\pm5$、"
            "Drummond–Needs 2009 は $31\\pm1$）。", 9.8)
    g2.text("この研究で使った 3 点：$r_s = 5$（液体）、$30$（ちょうど境界）、"
            "$60$（結晶）。", 9.8, ENJI, gap=0.22)
    g2.text("エネルギーの妥当性", 11.4, AI, "bold", gap=0.12)
    g2.text("$E/N = -1.1061/r_s + 0.814/r_s^{3/2}$ [${\\rm Ha}^{*}$]（2D 電子ガスの表式）\n"
            "と 0.02〜0.11% で一致している。物理としては正しく解けている。", 9.8)
    foot(fig, "② $r_s$ が決める相", 3, 10)

    # ============ p4  モアレ ===============================================
    fig, y = new_page(None, pages, "③ モアレ三角ポテンシャルとは何か",
                      lead="2 枚の 2 次元結晶をわずかにずらして重ねると、"
                           "うなりが長周期の超格子を作る。それが電子にとっての周期ポテンシャルになる。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("どこから来るか", 11.4, AI, "bold", gap=0.12)
    f.text("格子定数 $a_0$ の 2 枚を、ねじれ角 $\\theta$、格子不整合 $\\delta$ で重ねる。\n"
           "2 つの周期の差がうなりを作り、その周期が", 10.0)
    f.eq(r"$a_M \simeq \dfrac{a_0}{\sqrt{\theta^{2}+\delta^{2}}}$", 13.5)
    f.text("TMD ヘテロ二層では $a_M \\sim 5$〜$10$ nm。$a_0 \\sim 0.33$ nm の"
           "20〜30 倍である。", 9.8, SUB, gap=0.24)
    f.text("なぜポテンシャルになるか", 11.4, AI, "bold", gap=0.12)
    f.text("モアレセルの中で場所ごとに積層の仕方（どの原子がどの原子の上か）が\n"
           "変わる。それに応じてバンド端がずれるので、キャリアから見ると\n"
           "モアレ周期の周期ポテンシャルが立つ。最低次の調和だけ残すと", 10.0)
    f.eq(r"$V(\mathbf{r}) = -2U \sum_{j=1,2,3} \cos(\mathbf{G}_j\cdot\mathbf{r} + \phi)$",
         13.0)
    f.text("$\\mathbf{G}_j$ はモアレ逆格子の最短 3 本、$|\\mathbf{G}_j| = "
           "4\\pi/(\\sqrt{3}\\,a_M)$。$U$ が深さ、$\\phi$ が谷の形を決める。", 9.8, SUB)

    f2 = Flow(fig, 7.05, y, 5.9)
    axv = f2.axes(2.95, 3.15, x_in=7.05, gap=0.14, pre=0.28)
    aM = 1.0
    G = [2 * np.pi / aM * np.array([1, 1 / np.sqrt(3)]),
         2 * np.pi / aM * np.array([0, 2 / np.sqrt(3)]),
         2 * np.pi / aM * np.array([-1, 1 / np.sqrt(3)])]
    gx = np.linspace(-1.6, 1.6, 400); X, Y = np.meshgrid(gx, gx)
    V = -2 * sum(np.cos(g[0] * X + g[1] * Y + np.pi / 4) for g in G)
    axv.pcolormesh(X, Y, V, cmap="magma_r", shading="auto")
    axv.set_aspect("equal"); axv.axis("off")
    axv.set_title("$V(\\mathbf{r})$（明るいほど深い谷。電子はここに落ちる）", fontsize=9.4, pad=4)
    f2.text("三角形に並んだ谷ができる。電子はこの谷に落ちようとする。", 9.8, SUB, gap=0.24)
    f2.text("充填率 $\\nu$", 11.4, AI, "bold", gap=0.12)
    f2.text("$\\nu = N/n_{\\rm sites}$　… モアレの谷 1 個あたりの電子数", 10.4)
    f2.text("$\\nu = 1$：谷 1 個に電子 1 個。バンド理論なら金属だが、電子どうしの\n"
            "　　反発で動けなくなる（モット絶縁体）。並びは基板が決めている。\n"
            "分数 $\\nu$：谷のほうが多いので、電子は「どの谷に座るか」を自分で選ぶ。\n"
            "　　その選択をクーロン斥力が決めて秩序が出る（一般化 Wigner 結晶）。", 9.8)
    f2.text("コード単位では $a_M = a_{ee}\\sqrt{\\nu}$。$\\nu = 1$ だと $a_M = a_{ee}$ で"
            "長さが縮退する。", 9.8, ENJI)
    foot(fig, "③ モアレ三角ポテンシャル", 4, 10)

    # ============ p5  ハミルトニアン =======================================
    fig, y = new_page(None, pages, "④ 解きたいハミルトニアン",
                      lead="ここまでの 3 つを足す。すべての run は、この 1 本の式の"
                           "パラメータを振っただけである。")
    f = Flow(fig, 0.36, y, 12.6)
    f.skip(0.25)
    f.eq(r"$\hat H = -\dfrac{1}{2}\sum_i \nabla_i^{2}"
         r" \;+\; r_s \sum_{i<j} r_{ij}^{-1}"
         r" \;+\; \sum_i V(\mathbf{r}_i)$", 19.0, INK,
         gap=0.45, indent=1.3)
    f.eq(r"$V(\mathbf{r}) = -2U \sum_{j=1,2,3}\cos(\mathbf{G}_j\cdot\mathbf{r}+\phi)$,"
         r"　$|\mathbf{G}_j| = 4\pi/(\sqrt{3}\,a_M)$,　$a_M = a_{ee}\sqrt{\nu}$",
         13.0, INK, gap=0.40, indent=1.3)
    g1 = Flow(fig, 0.55, f.y, 5.8)
    g1.text("相を動かすつまみは 2 つ", 11.4, AI, "bold", gap=0.14)
    g1.text("$r_s$　… 密度。上げると液体 → Wigner 結晶（$U = 0$ でこれだけ）\n"
            "$U$　… モアレの深さ。$r_s$ を固定したまま上げても、電子は谷に\n"
            "　　　落ちて秩序が出る", 10.2)
    g1.text("同じ転移に独立な 2 つの経路があるのが効いてくる。片方だけだと\n"
            "「$r_s$ の関数を測っているだけでは」と言われてしまう。", 10.0, ENJI)
    g2 = Flow(fig, 7.05, f.y, 5.9)
    g2.text("変分モンテカルロで解く", 11.4, AI, "bold", gap=0.14)
    g2.text("試行波動関数 $\\psi_\\vartheta$ をニューラルネットで表し、\n"
            "$E[\\vartheta] = \\langle \\psi_\\vartheta|\\hat H|\\psi_\\vartheta\\rangle /"
            "\\langle\\psi_\\vartheta|\\psi_\\vartheta\\rangle$ を下げる。", 10.2)
    g2.text("局所エネルギー $E_{\\rm loc} = \\hat H\\psi_\\vartheta/\\psi_\\vartheta$ に\n"
            "$\\nabla^{2}\\log\\psi_\\vartheta$ が入るので、波動関数は 2 階微分できる\n"
            "必要がある。これが後で距離の測り方を縛る。", 10.0, ENJI)
    foot(fig, "④ 解きたいハミルトニアン", 5, 10)

    # ============ p6  有限セル =============================================
    fig, y = new_page(None, pages, "⑤ 有限セルで解く：長さの決まり方",
                      lead="無限系は解けないので、$N$ 電子を入れた箱を用意して"
                           "周期境界条件を課す。ここで系の「長さ」がすべて決まる。")
    f = Flow(fig, 0.36, y, 4.3)
    axc = f.axes(2.55, 4.1, gap=0.12)
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
            axc.plot(Pc[:, 0] + sh[0], Pc[:, 1] + sh[1], "o", ms=5.0 if mid else 2.6,
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
    f.text("濃い箱の $N$ = %d 個だけを解き、外は同じ箱の繰り返しとして扱う。\n"
           "電子は端から出ると反対側から入ってくる。" % NE, 9.8, SUB)

    f2 = Flow(fig, 4.95, y, 3.9)
    f2.text("面積と格子定数", 11.4, AI, "bold", gap=0.12)
    f2.text("電子 1 個あたりの面積が $\\pi$（コード単位）なので", 10.0)
    f2.eq(r"$A_{\rm sc} = \pi N$", 13.0, gap=0.12)
    f2.text("結晶相では電子は三角格子を組む。その格子定数 $a_{ee}$ は\n"
            "「原始セル 1 個に電子 1 個」から決まる：", 10.0)
    f2.eq(r"$\dfrac{\sqrt{3}}{2}a_{ee}^{2} = \pi$"
          r"　$\Rightarrow$　$a_{ee} = \sqrt{2\pi/\sqrt{3}} = 1.905$", 12.4, gap=0.12)
    f2.text("これが系の自然な長さ。$r_s$ を変えても動かない\n"
            "（$r_s$ は結合定数としてしか入らないため）。", 9.8, ENJI, gap=0.22)
    f2.text("$|\\mathbf{L}_1| = |\\mathbf{L}_2| = \\sqrt{N}\\,a_{ee}$、なす角 $120^\\circ$。\n"
            "$N = 25$ なら $9.523$。", 9.8, SUB)

    f3 = Flow(fig, 9.15, y, 3.9)
    f3.text("電子どうしの距離をどう測るか", 11.4, AI, "bold", gap=0.12)
    f3.text("周期境界条件があるので、箱をまたいでも意味を持つ測り方が要る。\n"
            "素直な最小イメージ（いちばん近い像までの Euclid 距離）は箱の\n"
            "半分で折れてしまい、そこで 2 階微分が発散する。局所エネルギー\n"
            "に $\\nabla^{2}\\log\\psi$ が入るので、これは使えない。", 9.8)
    f3.text("代わりに、箱を輪に丸めたときの**弦**の長さを使う（周期ノルム）。"
            .replace("**", ""), 9.8, ENJI)
    f3.eq(r"$d^{\rm per} = \dfrac{1}{2\pi}\sqrt{\sum_{mn} G_{mn}"
          r"(a_m a_n + b_m b_n)}$", 12.0, gap=0.10)
    f3.text("$a_m = 1-\\cos 2\\pi f_m$、$b_m = \\sin 2\\pi f_m$、"
            "$G = \\mathbf{L}^{T}\\mathbf{L}$\n"
            "$f_m$ は 2 電子の隔たりの分数座標。", 9.4, SUB, gap=0.20)
    f3.text("弧ではなく弦なので折れ点がない。近距離では Euclid と一致し、\n"
            "遠方では滑らかに頭打ちになる。その上限が", 9.8)
    f3.eq(r"$d_{\rm ref} = \dfrac{3^{3/4}}{2\sqrt{\pi}}\sqrt{N} = 0.643\sqrt{N}$"
          r"　$(N{=}25: 3.215)$", 11.6, ENJI, gap=0.10)
    f3.text("どの 2 電子も $d^{\\rm per} \\leq d_{\\rm ref}$。これが距離軸の天井になる。",
            9.8, SUB)
    foot(fig, "⑤ 有限セルで解く", 6, 10)

    # ============ p7  遮蔽アテンション =====================================
    fig, y = new_page(None, pages, "⑥ 遮蔽アテンション：距離の窓を固定する",
                      lead="ここがこの研究の中身。アテンションの重みを「内容」と「距離」に"
                           "分け、距離側をヘッドごとに固定した窓にする。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("ふつうのアテンション", 11.4, AI, "bold", gap=0.12)
    f.text("電子 $i$ が電子 $j$ をどれだけ見るかを、特徴ベクトルから作った\n"
           "$q_i$ と $k_j$ の内積 $s_{ij} = \\bar q_i \\cdot \\bar k_j$ で決める。\n"
           "全対を同じように見るので、学習後に「どの長さを使ったか」は残らない。", 10.0)
    f.skip(0.10)
    f.text("遮蔽アテンション", 11.4, AI, "bold", gap=0.14)
    f.eq(r"$W^{(h)}_{ij} = \alpha(s_{ij}) \;\cdot\; m_h(d_{ij})$", 16.0, gap=0.18)
    f.text("$\\alpha(s_{ij})$　内容チャネル　… $q\\cdot k$ の類似度から", 10.2)
    f.text("$m_h(d_{ij})$　距離チャネル　… ヘッドごとに固定した窓", 10.2, ENJI, gap=0.20)
    f.eq(r"$m_h(d) = \dfrac{1}{2}\left[1+\cos\left(\pi\min(d/w_h,\,1)\right)\right]$",
         13.0, gap=0.14)
    f.text("$d \\geq w_h$ で厳密に 0。$w_h$ より遠い相手は、そのヘッドからは\n"
           "完全に見えない。$w_h$ は学習させず固定してある。", 10.0, ENJI)

    f2 = Flow(fig, 7.05, y, 5.9)
    axw = f2.axes(2.35, 5.4, x_in=7.15, gap=0.72, pre=0.30)
    xs = np.linspace(0, dref * 1.02, 300)
    for i, hh in enumerate(order):
        axw.plot(xs, 0.5 * (1 + np.cos(np.pi * np.clip(xs / w[hh], 0, 1))),
                 lw=1.5, color=plt.cm.viridis(i / (NH - 1)))
    axw.set_xlim(0, dref * 1.02); axw.set_ylim(0, 1.06)
    axw.set_xlabel("2 電子の距離 $d$（周期ノルム）", fontsize=9.2, labelpad=1)
    axw.set_ylabel("$m_h(d)$", fontsize=9.2, labelpad=1)
    axw.tick_params(labelsize=8.2); axw.grid(alpha=.2, lw=.6)
    axw.set_title("%d 本の窓を $[%.2f,\\ %.2f]$ に等比で並べる" % (NH, w[order[0]], w[order[-1]]),
                  fontsize=9.6, pad=4)
    f2.text("なぜ固定するか", 11.4, AI, "bold", gap=0.12)
    f2.text("$w_h$ を学習させると、その値は「最適化の結果」であって\n"
            "「波動関数が使っている長さ」とは言えなくなる。固定すれば\n"
            "$w_h$ は動かない目盛りになり、ヘッド番号がそのまま長さになる。", 10.0)
    f2.text("3 つの相で梯子が完全に共通なので、曲線の違いはそのまま\n"
            "波動関数の違いになる（チェックポイントで $10^{-5}$ まで確認済み）。",
            10.0, ENJI, gap=0.20)
    f2.text("窓の上限は $d_{\\rm ref}$ で止める。それを超えた窓はどの対も必ず含む\n"
            "ので、距離で選ぶ働きを失う。", 9.8, SUB)
    foot(fig, "⑥ 遮蔽アテンション", 7, 10)

    # ============ p8  q・k の読み方 ========================================
    fig, y = new_page(None, pages, "⑦ $q\\cdot k$ マトリクスの読み方",
                      lead="学習後の重み $W^{(h)}_{ij}$ を、ヘッドごとに行列として描いたもの。"
                           "1 つの電子配置について示す。")
    hsel = [int(t) for t in a.heads.split(",")]
    labs = ["窓が狭いヘッド", "窓が広いヘッド"]
    for k, hi in enumerate(hsel):
        hh = order[hi]; M = W1[hh]
        off = ~np.eye(NE, dtype=bool)
        vmax = float(np.quantile(M[off], 0.995)) or 1.0
        axm = fig.add_axes([(0.55 + k * 3.35) / WF, 1 - (y + 3.05) / HF, 3.0 / WF, 3.0 / HF])
        axm.imshow(np.where(M > 1e-12, M, np.nan), cmap=CM, vmin=0, vmax=vmax,
                   interpolation="nearest")
        axm.set_facecolor("#e9e7e2")
        axm.axhline(nup - .5, color="#e07b39", lw=1.2)
        axm.axvline(nup - .5, color="#e07b39", lw=1.2)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_edgecolor(HAI); sp.set_linewidth(.8)
        axm.set_title("%s　$w_h$ = %.2f" % (labs[k], w[hh]), fontsize=10.6, pad=5)
    fig.text(0.55 / WF, 1 - (y + 3.32) / HF,
             "行 ＝ 見る側の電子 $i$、　列 ＝ 見られる側の電子 $j$（↑%d 個 → ↓%d 個 の順に並べてある）"
             % (nup, NE - nup), fontsize=10.0, color=INK, va="top")
    g = Flow(fig, 7.55, y, 5.4)
    g.text("見るところ", 11.4, AI, "bold", gap=0.14)
    g.text("**橙の線**　スピンの境界。左上が ↑↑、右下が ↓↓、非対角が ↑↓ ブロック。"
           .replace("**", ""), 10.0)
    g.text("**灰色**　窓の外（$d_{ij} \\geq w_h$ で $W = 0$）。狭いヘッドでは"
           "ほとんどの相手が\n　　　窓の外に落ちる。この灰色の量が、そのヘッドの「見える範囲」。"
           .replace("**", ""), 10.0)
    g.text("**対角**　$W_{ii} = \\alpha(s_{ii})\\cdot m_h(0) = \\alpha(s_{ii})$。"
           "$m_h(0) = 1$ なので必ず\n　　　窓の中にある。その電子が自分自身の情報を"
           "どれだけ使っているか。".replace("**", ""), 10.0)
    g.text("**ブロックごとの濃淡差**　同スピンと異スピンの扱いの違い。$m_h$ は距離\n"
           "　　　しか見ないので、スピンの構造はすべて $\\alpha$ 側が担っている。"
           .replace("**", ""), 10.0)
    g.text("色の上限はヘッドごとに、非対角成分の 99.5 パーセンタイルで揃えてある。\n"
           "同じヘッドなら相を跨いで直接比べられる。", 9.6, SUB)
    foot(fig, "⑦ $q\\cdot k$ マトリクスの読み方", 8, 10)

    # ============ p9  解釈 =================================================
    fig, y = new_page(None, pages, "⑧ 何が見えるはずか：層とヘッド、液体と結晶",
                      lead="読む前に、物理から予想を立てておく。予想と実測が合うかどうかが"
                           "この図の使いどころになる。")
    f = Flow(fig, 0.36, y, 6.1)
    f.text("層とヘッドの意味", 11.4, AI, "bold", gap=0.12)
    f.text("**層**　1 回ぶんの「各電子が全電子を見て自分の特徴を更新する」操作。\n"
           "　　2 層重ねると、電子 $i$ の特徴が $j$ の隣人にも依存しはじめる。\n"
           "　　この研究では読み出しを濁らせないため **1 層**にしてある。"
           .replace("**", ""), 10.0)
    f.text("**ヘッド**　1 つの層の中の独立な「見方」のチャネル。ふつうは $q, k$ の\n"
           "　　射影が違うだけだが、ここでは加えて距離の窓 $w_h$ を固定してある。\n"
           "　　だからヘッド番号 ＝ 長さの目盛りになる。".replace("**", ""), 10.0)
    f.text("液体で予想されること", 11.4, AI, "bold", gap=0.12)
    f.text("電子は非局在で、決まった座る場所がない。相関を作っているのは主に\n"
           "短距離のクーロンホール（互いに避ける効果）。近い相手が実際にいるので、\n"
           "狭い窓のヘッドにも 2 体の仕事がある。", 10.0)
    f.text("結晶で予想されること", 11.4, AI, "bold", gap=0.12)
    f.text("各電子は格子点のまわりに束縛され、まわりの電子が作る谷の中で\n"
           "振動する（最低次では調和振動子）。すると 2 つのことが起きる：", 10.0)
    f.text("(a) 最近接距離が $a_{ee}$ 程度にそろい、それより近づかない。\n"
           "　　窓がそれより狭いヘッドには相手が 1 個も入らない。\n"
           "(b) 支配的なのは「自分の格子点からどれだけずれているか」という\n"
           "　　1 体的な情報。だから対角が相対的に重くなる。", 10.0, ENJI)

    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("実測（$N = 25$、窓を $[1.80,\\ 3.20]$ に固定した 16 ヘッド）", 11.4, AI, "bold",
            gap=0.14)
    D = np.load("/tmp/interp.npy") if os.path.exists("/tmp/interp.npy") else None
    rows = [["$w_h$", "自分自身 %", "", "2 体寄与 %", ""],
            ["", "液体", "結晶", "液体", "結晶"]]
    idx = [0, 3, 7, 11, 15]
    if D is not None:
        for i in idx:
            rows.append(["%.2f" % D[0][i], "%.0f" % D[1][i], "%.0f" % D[2][i],
                         "%.2f" % D[3][i], "%.2f" % D[4][i]])
    xs = [7.25, 8.35, 9.30, 10.40, 11.45]
    yy = f2.y
    for j, c in enumerate(rows[0]):
        fig.text(xs[j] / WF, 1 - yy / HF, c, fontsize=9.6, color=SUB, weight="bold", va="top")
    for j, c in enumerate(rows[1]):
        fig.text(xs[j] / WF, 1 - (yy + 0.24) / HF, c, fontsize=9.2, color=SUB, va="top")
    fig.add_artist(plt.Line2D([7.25 / WF, 12.4 / WF], [1 - (yy + 0.46) / HF] * 2,
                              color=RULE, lw=0.9))
    for i, r in enumerate(rows[2:]):
        for j, c in enumerate(r):
            fig.text(xs[j] / WF, 1 - (yy + 0.58 + i * 0.245) / HF, c, fontsize=9.8,
                     color=ENJI if i in (0, 4) else INK, va="top")
    f2.y = yy + 0.58 + len(rows[2:]) * 0.245 + 0.22
    f2.text("いちばん狭いヘッド（$w_h = 1.80$）で、2 体寄与は液体 3.09% に対し\n"
            "結晶 0.19%。16 倍ちがう。予想 (a) のとおり、結晶では相手が入らない。",
            10.0, INK)
    f2.text("同じヘッドで自分自身の割合は液体 80% に対し結晶 87%。狭いヘッドは\n"
            "「消える」のではなく、2 体から 1 体へ役割を変えている。予想 (b)。", 10.0, INK)
    f2.text("いちばん広いヘッド（$w_h = 3.20$）では逆転して、2 体寄与は液体 14.4%\n"
            "に対し結晶 33.6%。相関の重みが広い窓へ移っている。", 10.0, ENJI)
    f2.text("注意：調和振動子の描像はあくまで最低次。実際には隣人の位置ゆらぎと\n"
            "結合したフォノン的な集団運動があり、それが広い窓の 2 体寄与に乗る。",
            9.6, SUB)
    foot(fig, "⑧ 層とヘッド、液体と結晶", 9, 10)

    # ============ p10  読み出せるもの ======================================
    fig, y = new_page(None, pages, "⑨ 読み出せるもの",
                      lead="ヘッドごとの寄与を窓の長さの上に並べると、"
                           "「波動関数が使っている長さ」が 1 本の数字になる。")
    f = Flow(fig, 0.36, y, 6.4)
    if a.runs:
        items = []
        for part in a.runs.split(","):
            lab, ff = part.rsplit("=", 1)
            items.append((lab, qload(ff, 48)))
        axq = f.axes(3.15, 6.0, x_in=0.85, gap=0.70)
        for i, (lab, d) in enumerate(items):
            axq.plot(d["w"], d["qp"], "-", marker="os^"[i], ms=5.0, lw=1.9,
                     color=[AI, "#c98a12", ENJI][i], label=lab)
        axq.set_xlabel("固定した窓 $w_h$（周期ノルム, $r_s a_B^{*}$）", fontsize=9.8, labelpad=2)
        axq.set_ylabel("波動関数への寄与率 $q_h$ [%]", fontsize=9.8, labelpad=2)
        axq.tick_params(labelsize=8.6); axq.grid(alpha=.22, lw=.7)
        axq.legend(fontsize=10.0, frameon=False, loc="upper left")
        wq = [float((d["qp"] * d["w"]).sum() / d["qp"].sum()) for _, d in items]
        dn = [d["d_nn"] for _, d in items]
    f2 = Flow(fig, 7.05, y, 5.9)
    f2.text("寄与率の定義", 11.4, AI, "bold", gap=0.12)
    f2.eq(r"$q_h = \|c_h^{\rm pair}\|^{2} \,/\, \sum_{h'}\|c_{h'}^{\rm pair}\|^{2}$", 13.0,
          gap=0.10)
    f2.text("$c_h$ はヘッド $h$ が出力に足したベクトル、$c_h^{\\rm pair}$ はそこから\n"
            "自分自身のぶんを抜いた 2 体成分。", 9.8, SUB, gap=0.22)
    f2.text("読み出した長さ", 11.4, AI, "bold", gap=0.12)
    f2.eq(r"$\langle w\rangle_q = \sum_h q_h\, w_h$", 14.0, gap=0.12)
    if a.runs:
        f2.text("液体 → 境界 → 結晶 で　" + " → ".join("%.2f" % v for v in wq)
                + "\n実測の最近接距離　　　" + " → ".join("%.2f" % v for v in dn)
                + "\n同じ向きに動いている。", 10.4, ENJI, gap=0.24)
    f2.text("なぜこれが新しいか", 11.4, AI, "bold", gap=0.12)
    f2.text("従来のニューラル波動関数は、エネルギーは出るが「どの長さを使って\n"
            "いるか」は出ない。窓を固定したことで、学習された波動関数の中身から\n"
            "長さが直接読み出せるようになった。", 10.2, INK)
    f2.text("主張は「エネルギーが低い」ことではなく、\n"
            "「これまでの NQS では取り出せなかった情報が取り出せる」ことである。",
            10.4, ENJI)
    foot(fig, "⑨ 読み出せるもの", 10, 10)

    # ============ 出力 =====================================================
    with PdfPages(a.out) as pdf:
        for fg in pages:
            pdf.savefig(fg)
    print("  wrote %s（%d ページ）" % (a.out, len(pages)))
    if a.pngdir:
        os.makedirs(a.pngdir, exist_ok=True)
        for i, fg in enumerate(pages, 1):
            p = os.path.join(a.pngdir, "p%02d.png" % i)
            fg.savefig(p, dpi=150)
        print("  wrote %s/p01..p%02d.png" % (a.pngdir, len(pages)))


if __name__ == "__main__":
    main()
