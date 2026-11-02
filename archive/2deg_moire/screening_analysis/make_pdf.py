# -*- coding: utf-8 -*-
"""図をすべて 1 冊にまとめ、各ページに解説を付けた PDF を作る。

matplotlib で組む（reportlab だと数式が組めないため）。

    python screening_analysis/make_pdf.py [--out screening_moire_report.pdf]
"""
import re
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for p in fm.findSystemFonts():
    if "NotoSansCJK" in p: fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

W, H = 16.54, 11.69                       # A3 横（インチ）
M = 0.040                                 # 余白（図の割合）
INK, SUB, AI, ENJI, RULE = "#141412", "#4a4946", "#155f7a", "#a5262f", "#d6d3cc"

_NO_HEAD = "、。，．）」』】〕〉》%"
_NO_TAIL = "（「『【〔〈《"


def wrap(text, cols):
    """$...$ の数式は途中で折らない。全角は 2 桁として数える。"""
    def w(ch): return 2 if ord(ch) > 0x2E80 else 1
    toks, i = [], 0
    while i < len(text):
        if text[i] == "$":
            j = text.find("$", i + 1)
            if j < 0: j = len(text) - 1
            toks.append(text[i:j + 1]); i = j + 1
        else:
            m = re.match(r"[0-9]+(?:[.,][0-9]+)*%?", text[i:])   # 数字は途中で折らない
            if m:
                toks.append(m.group(0)); i += m.end()
            else:
                toks.append(text[i]); i += 1
    out, line, cur = [], "", 0
    for t in toks:
        tw = sum(w(c) for c in t)
        if cur + tw > cols and line and t[0] not in _NO_HEAD and line[-1] not in _NO_TAIL:
            out.append(line); line, cur = "", 0
        line += t; cur += tw
    if line: out.append(line)
    return out


class Book:
    def __init__(self, path):
        self.pdf = PdfPages(path); self.n = 0

    def _new(self):
        fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        return fig, ax

    def _close(self, fig, ax):
        self.n += 1
        ax.text(M, 0.018, "遮蔽アテンションから何が読めたか — モアレ 2DEG での結果 / 2026-08-26",
                fontsize=8, color=SUB, va="bottom")
        ax.text(1 - M, 0.018, "%d" % self.n, fontsize=8, color=SUB, va="bottom", ha="right")
        self.pdf.savefig(fig); plt.close(fig)

    def page(self, kicker, title, paras, image=None, table=None, size=10.0, cols=None,
             eqs=None, eqsize=12.5):
        fig, ax = self._new()
        y = 1 - M
        if kicker:
            ax.text(M, y, kicker, fontsize=9, color=AI, va="top"); y -= 0.019
        ax.text(M, y, title, fontsize=17, color=INK, va="top", weight="bold"); y -= 0.036
        ax.plot([M, 1 - M], [y, y], color=RULE, lw=1.0); y -= 0.020
        if eqs:
            # 左に短い見出し、右に式。折り返さない（式は 1 行に収める）。
            elh = eqsize * 2.35 / (H * 72)
            xe = M + 0.118
            y -= elh * 0.25
            for lab, mth in eqs:
                ax.text(M, y, lab, fontsize=eqsize - 2.6, color=AI, va="top")
                ax.text(xe, y, mth, fontsize=eqsize, color=INK, va="top")
                tall = any(k in mth for k in ("\\sum", "\\frac", "\\sqrt"))
                y -= elh * (1.34 if tall else 1.0)
            y -= elh * 0.45
        lh = size * 1.98 / (H * 72)      # 数式行は背が高いので余裕を取る
        cols = cols or int((1 - 2 * M) * W * 72 / (size * 0.505))
        for p in paras:
            for ln in wrap(p, cols):
                ax.text(M, y, ln, fontsize=size, color=SUB, va="top"); y -= lh
            y -= lh * 0.42
        if table:
            y -= lh * 0.3
            hdr, rows = table[0], table[1:]
            ncol = len(hdr)
            tw = min(1 - 2 * M, (0.085 * ncol + 0.12))
            xs = [M + tw * j / ncol for j in range(ncol)]
            for j, t in enumerate(hdr):
                ax.text(xs[j], y, t, fontsize=size - 0.3, color=INK, va="top", weight="bold")
            y -= lh * 1.35
            ax.plot([M, M + tw], [y + lh * 0.45, y + lh * 0.45], color=RULE, lw=0.9)
            for r in rows:
                for j, t in enumerate(r):
                    ax.text(xs[j], y, str(t), fontsize=size - 0.3, color=SUB, va="top")
                y -= lh
            y -= lh * 0.5
        if image:
            im = mpimg.imread(image)
            ih, iw = im.shape[0], im.shape[1]
            avail_h = y - 0.055
            avail_w = 1 - 2 * M
            s = min(avail_w * W / iw, avail_h * H / ih)
            dw, dh = iw * s / W, ih * s / H
            axi = fig.add_axes([(1 - dw) / 2, y - dh, dw, dh])
            axi.imshow(im); axi.axis("off")
        self._close(fig, ax)

    def save(self):
        self.pdf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "screening_moire_report.pdf"))
    a = ap.parse_args()
    F = lambda n: os.path.join(HERE, n)                                  # noqa: E731
    b = Book(a.out)

    # ---------------------------------------------------------------- 表紙
    fig, ax = b._new()
    ax.text(M, 0.885, "遮蔽アテンションから何が読めたか", fontsize=32, color=INK,
            va="top", weight="bold")
    ax.text(M, 0.830, "モアレ 2 次元電子系での結果　—　2026-08-26", fontsize=14,
            color=SUB, va="top")
    ax.plot([M, 1 - M], [0.808, 0.808], color=RULE, lw=1.2)
    y = 0.775
    blocks = [
        ("主張", 13, INK, [
            "「より低いエネルギーが出た」ではなく、遮蔽アテンションを使うとこれまでの"
            "ニューラル波動関数では取り出せなかった情報が読める、という主張。"
            "具体的には、波動関数が使っている長さを直接読み出せること。"]),
        ("分かったこと", 13, INK, [
            "① 遮蔽窓は Coulomb の遮蔽長でも外場の周期でもなく、電子どうしの間隔で決まっている。"
            "実効到達距離は $R_{\\mathrm{eff}} = 1.68 \\pm 0.02$（$R_{\\mathrm{eff}}/a_{ee} = 0.88 \\pm 0.01$）で、"
            "モアレ周期 $a_M$ を 2 倍に振っても動かない（5 点）。",
            "② 電子が谷に落ちるほど、相関の仕事が長い窓のヘッドへ一極集中する。"
            "窓 $w_h < 1.0$ のヘッドの分担は 15% から 0.1% まで 150 倍減る。",
            "③ 通常の softmax ではどのヘッドも $\\bar d_h \\simeq 2.9$ に固まり（幅 1.1 倍）、"
            "長さで切り分ける軸がそもそも存在しない。①②の問いは遮蔽でしか測れる問いにならない。"]),
        ("出なかったこと", 13, ENJI, [
            "arXiv:2502.05383 の Fig.6（一般化 Wigner 結晶）は再現できなかった。"
            "三通りの方法で試して全部同じ結論で、この系のこのサイズ・このネットワークでは"
            "谷液体が勝つ。理由と数字は最後のページに書いた。"]),
        ("系", 13, INK, [
            "$N = 18$ 電子（↑9 ↓9）、モアレの谷 27 個、充填 $\\nu = 2/3$。"
            "2 層 × 8 ヘッド、行列式 4。計 20 本の学習を CPU 2 コアで回した。"]),
    ]
    lh = 10.4 * 1.98 / (H * 72)
    cols = int((1 - 2 * M) * W * 72 / (10.4 * 0.505))
    for head, hs, hc, paras in blocks:
        ax.text(M, y, head, fontsize=hs, color=hc, va="top", weight="bold"); y -= lh * 1.35
        for p in paras:
            for ln in wrap(p, cols):
                ax.text(M, y, ln, fontsize=10.4, color=SUB, va="top"); y -= lh
            y -= lh * 0.35
        y -= lh * 0.55
    b._close(fig, ax)

    # ---------------------------------------------------------------- 各ページ
    b.page("定義", "セットアップ — 系と波動関数", [], F("fig_setup.png"))

    b.page("定義", "測り方 — 今回使った量の定義", [], F("fig_methods.png"))

    b.page("結果", "相転移でヘッドの順位が入れ替わる", [],
           F("fig_window.png"), eqsize=11.4,
           eqs=[
               ("遮蔽窓 $w_h$",
                r"$W^{(h)}_{ij} = \alpha(s_{ij})\,m_h(d_{ij}),\quad "
                r"m_h(d) = \frac{1}{2}\left[1+\cos\left(\pi\min(d/w_h,\,1)\right)\right]"
                r"\quad (d \geq w_h \Rightarrow m_h = 0)$"),
               ("ヘッドの重要度",
                r"$W^{(h)}_{ij} \rightarrow W^{(h)}_{ii}\,\delta_{ij}$（1 体成分は残す）"
                r"$,\quad \Delta E_h = E_h^{\mathrm{off}} - E$"),
               ("振るのは $U$ だけ",
                r"$r_s = 6.8313$（論文の $\varepsilon = 5$）, $\nu = 2/3$, "
                r"$a_M = 1.555$, $a_{ee} = 1.905$ を固定。$U = 1.84 \rightarrow 12$"),
               ("秩序変数",
                r"谷ごとの平均占有のばらつき $\mathrm{sd}\{\bar n_s\} : "
                r"0.24 \rightarrow 0.38 \rightarrow 0.47 \rightarrow 0.47$"),
               ("結果",
                r"$w<a_{ee}$ の担当 $41\% \rightarrow 4\%$、中央値 $w : 2.15 \rightarrow 3.62$、"
                r"最大寄与ヘッドの窓 $3.76 \rightarrow 3.96$ で担当 $19\% \rightarrow 65\%$"
                r"（担当と $\mathrm{sd}\{\bar n_s\}$ の相関 $r = -0.85$）"),
               ("解釈",
                r"外場が電子を谷に貼り付けると、近距離の回避は 1 体の軌道が担うようになる"
                r"$\ \Rightarrow\ $ 2 体のアテンションに短い窓は要らなくなり、"
                r"残るのは「どの谷が埋まるか」という遠距離の相関だけ"),
           ])

    b.page("結果 の裏側", "全ヘッドの $q\\cdot k$ 行列（液体 と 1 谷 1 電子）", [],
           F("fig_qk.png"), eqsize=11.4,
           eqs=[
               ("見かた",
                r"$s_{ij} = \hat q_i \cdot \hat k_j$（配置平均）。"
                r"赤いほど「$i$ が $j$ を見たい」、青いほど「見たくない」。"
                r"行が見る側 $i$、列が見られる側 $j$、黒線がスピンの境目"),
               ("並び",
                r"ヘッドは遮蔽窓 $w_h$ の狭い順に左から。枠の上が $w_h$ と、"
                r"そのヘッドを止めたときの代償 $\Delta E_h$ の割合（担当）。"
                r"担当 15% 以上は赤枠"),
               ("読めること",
                r"谷液体では $w_h = 1.70$（担当 16%）と $w_h = 3.51,\,3.76$（21%, 19%）に"
                r"分かれているが、1 谷 1 電子では $w_h = 3.96$ の 1 枚が 65% を占める"),
           ])

    b.page("対照", "同じ入れ替わりを起こすもの・起こさないもの", [],
           F("fig_controls.png"), eqsize=11.4,
           eqs=[
               ("$r_s$ を振る",
                r"$2 \rightarrow 6.83$（3.4 倍）では担当は $37 \sim 41\%$ で動かない。"
                r"$r_s = 30$ まで上げると $0\%$ になり、$U$ と同じ入れ替わりが起きる"),
               ("$a_M$ を振る",
                r"$1.35 \rightarrow 2.69$（2 倍）で担当と $a_M$ は無相関（$r = +0.01$）。"
                r"$\nu$ を変えると状態自体も変わるので綺麗なスキャンではない"),
               ("まとめ",
                r"順位を入れ替えるのは外場の周期 $a_M$ ではなく、"
                r"電子を互いに離して固める働き（深い $U$、あるいは非常に強い $r_s$）"),
           ])

    b.page("結果 ① の裏付け", "届く範囲を絞ったときのエネルギーの代償", [],
           F("fig_ablation.png"), eqsize=11.2,
           eqs=[
               ("絞り方",
                r"学習済みの $\psi$ はそのまま、$W_{ij} \rightarrow W_{ij}\,c_R(d_{ij})$"
                r"   （再学習はしない。$R$ = 外から掛ける到達半径）"),
               ("襟",
                r"$c_R(d) = \frac{1}{2}\left[1+\cos\left(\pi\,\mathrm{clip}"
                r"\left(\frac{d-(1-\beta)R}{\beta R},\,0,\,1\right)\right)\right],"
                r"\ \ \beta = 0.3$   （$d \leq 0.7R$ で 1、$d \geq R$ で 0)"),
               ("代償",
                r"$\Delta E(R) = E(R) - E(\infty)$"),
               ("必要な距離",
                r"$\Delta E(R^{*}) = 0.1 \times \max_R \Delta E(R)$"
                r"   （代償が立ち上がる位置)"),
               ("対照",
                r"$R > $ モアレのスーパーセル $\ \Rightarrow\ \Delta E = 0$（確認済み）"
                r"$\qquad$ 硬い切断 $\Rightarrow$ 前進ラプラシアンが運動項の特異部を落として "
                r"$\Delta E < 0$（偽）"),
               ("結果",
                r"$\nu$ スキャンを含む 9 本で $R^{*} = 1.62 \sim 2.51$"
                r"（前ページの $\diamondsuit$）。$a_M = 1.35 \sim 2.69$ にも "
                r"$\sqrt{3}a_M$ にも合わず、$a_{ee} = 1.905$ の側"),
           ])

    b.page("結果 ③", "通常のアテンションとの差別化点 — 長さで分けて読めるのは遮蔽だけ", [],
           F("fig_softmax.png"),
           eqs=[
               ("測る量",
                r"$\bar d_h = \sum_{i \neq j} W^{(h)}_{ij}\,d_{ij}"
                r"\ /\ \sum_{i \neq j} W^{(h)}_{ij}$"),
               ("softmax",
                r"$\sum_j W_{ij} = 1\ \Rightarrow\ W_{ij} > 0\ (\forall j)$"
                r"$\ \Rightarrow\ $ どのヘッドも全部の対を平均"),
               ("遮蔽",
                r"$d_{ij} \geq w_h\ \Rightarrow\ W^{(h)}_{ij} = 0$"
                r"$\ \Rightarrow\ $ ヘッドが距離を分担"),
               ("状態依存",
                r"softmax の $\bar d_h$ は状態を変えても動かない:"
                r"$\ r_s = 3.42$ で 2.97、$r_s = 6.83$ で 2.95"),
           ],
           table=[["", "$\\bar d_h$ の範囲", "幅", "$\\mathrm{sd}/\\mathrm{mean}$"],
                  ["遮蔽アテンション", "0.43 〜 1.71", "4.0 倍", "0.416"],
                  ["通常の softmax", "2.75 〜 3.10", "1.1 倍", "0.037"]])

    b.page("参考", "各ヘッドが見ている距離", [], F("fig_kernel.png"))

    b.page("目的 ③（部分的）", "基底状態の電荷分布", [],
           F("fig_fig6.png"),
           eqs=[
               ("対応",
                r"arXiv:2502.05383 の Fig.6（$\varepsilon = 10,\ 5$）"),
               ("秩序変数",
                r"$\psi_K = \frac{S(K)/S_{\rm bg} - 1}{N/4 - 1}$"
                r"$\qquad$（完全秩序で 1、乱雑で 0）"),
           ])

    b.page("検証", "論文 arXiv:2502.05383 との突き合わせ",
           [],
           None, eqsize=11.4,
           eqs=[
               ("外場",
                r"論文 $V(\mathbf{r}) = -2V_0\sum_{j=1}^{3}\cos(\mathbf{g}_j\cdot\mathbf{r}+\phi)$, "
                r"$V_0 = 15$ meV, $\phi = \pi/4$, $|\mathbf{g}_j| = 4\pi/\sqrt{3}a_M$　"
                r"— 本計算の $V_M$ と同形・同位相"),
               ("深さの換算",
                r"$\mathrm{Ha}^{*} = (m^{*}/m_e)\varepsilon^{-2}\mathrm{Ha}$, "
                r"$U = V_0[\mathrm{Ha}^{*}]\times r_s^{2}$"
                r"$\ \Rightarrow\ \varepsilon = 5:\ (15/380.96)\times 46.67 = 1.8375$, "
                r"$\ \varepsilon = 10:\ (15/95.24)\times 11.67 = 1.8374$"),
               ("密度と $r_s$",
                r"$n(\nu{=}1) = 1.785\times10^{12}\,\mathrm{cm^{-2}}$, $m^{*} = 0.35\,m_e$, "
                r"$\nu = 2/3$, $\pi(r_s a_B^{*})^{2} = 1/n_e$"
                r"$\ \Rightarrow\ r_s = 3.415\ (\varepsilon{=}10),\ 6.831\ (\varepsilon{=}5)$"),
               ("系",
                r"$N = 18$ 電子・27 サイト・$\nu = 2/3$　— 論文の Table I と同じ"),
               ("差",
                r"論文 NN との差は $+0.95$（$\varepsilon{=}10$）, $+0.85$（$\varepsilon{=}5$）meV/電子。"
                r"$\varepsilon = 5$ では SlaterNet と BP-ED より低い。"
                r"秩序を分けるエネルギーの目安（論文 NN と SlaterNet の差）は $1.12$ で、同じ大きさ"),
               ("計算量",
                r"$4096 \times 150000\,/\,(64 \times 3500) \simeq 2700$ 倍の差。"
                r"設定の誤りではなく、足りないのは計算量"),
           ],
           table=[["手法", "$\\varepsilon$=10 [meV/電子]", "$\\varepsilon$=5 [meV/電子]",
                   "計算量（batch × step）"],
                  ["遮蔽アテンション（本計算）", "−31.12", "−58.28", "64 × 3,500"],
                  ["論文 self-attention NN", "−32.07", "−59.13", "4,096 × 150,000"],
                  ["論文 SlaterNet", "−31.35", "−58.01", "—"],
                  ["論文 BP-ED", "−31.32", "−57.81", "—"],
                  ["通常 softmax（本計算・対照）", "−28.72", "−56.18", "64 × 3,000"]])

    b.page("記録", "出なかったこと、やってはいけなかったこと",
           ["【一般化 Wigner 結晶が作れなかった】三通り試して全部同じ結論だった。",
            "① $r_s$ を 30 まで上げる → 秩序 0.04（乱雑と同じ）。$r_s$ は正しいノブではない。"
            "$r_s$ を上げると電子は間隔 $a_{ee} = 1.905$ の三角格子を組みたがるが、"
            "モアレの谷は $a_M = 1.555$ で整合しない。競合してどちらの秩序も立たない。",
            "② $U$ を 12 まで深くする → 谷の占有は完璧に「1/3 が空、2/3 が 1 個ずつ、"
            "二重占有ゼロ」。電子は完全に谷に落ちている。それでも並びは乱雑（秩序 0.08）。",
            "③ ハニカム配置から walker を出発させる → 学習 320 step で秩序が溶け、"
            "エネルギーも乱れた状態の値へ戻る。$U = 12$ で乱れた状態が $E = -643.4$、"
            "ハニカムに揃えた状態は $-570$ 付近止まり。揃えて得られる Coulomb エネルギーは "
            "8.6 しかないのに対し、揃えると失う「谷から谷へ広がる自由」の方が大きい。",
            "この系のこのサイズ・このネットワークでは谷液体が勝つ。論文との差はネットワークの"
            "容量と最適化予算（2 層 8 ヘッド・行列式 4・数千 step、CPU 2 コア）だと考えている。",
            "",
            "【無効だった手法（記録として残す）】",
            "・モンテカルロをハニカムに拘束して最適化する → 拘束すると $|\\psi|^2$ から"
            "サンプルしていないことになり、VMC の勾配もエネルギーも意味を失う。"
            "局所エネルギーのばらつきが 6 から 80 に跳ね上がったのが症状。結果は破棄した。",
            "・アテンションの重みを硬く切る → 不連続だと前進ラプラシアンが運動エネルギーの特異項を"
            "取りこぼし、$R = 1.48$ で $\\Delta E = -5.9$ というありえない値が出た。"
            "縁を滑らかに落とす襟に替えて解決。",
            "",
            "【設計への示唆】",
            "・窓の梯子の下端は無駄になりやすい。最短の電子間距離より内側まで縮んだヘッドは "
            "$W^{(h)}_{ij} \\equiv 0$ となり勾配も 0 なので二度と戻れない。初期設定 "
            "$w_h \\in [0.30, 3.50]$ では 8 本中 2 本が死に、1 本は $w_h > d_{\\mathrm{ref}}$ で"
            "窓なしだった。$[0.50, 3.35]$ に張り直した。",
            "・モアレを入れると窓は学習で動く。Coulomb だけの系では $w_h$ はほとんど動かなかったが、"
            "モアレを入れると狭い側が 2 倍以上縮む。外場が入って初めて窓に決めるべきものができる、"
            "と読める。"],
           None, size=11.0, cols=112)

    b.save()
    print("->", a.out)


if __name__ == "__main__":
    main()
