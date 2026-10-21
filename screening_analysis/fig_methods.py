"""測り方の図。何をどう測ったかを式で示す（記号はこの図の中で定義する）。"""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

for p in fm.findSystemFonts():
    if "NotoSansCJK" in p: fm.fontManager.addfont(p)
JP = ["Noto Sans CJK JP", "DejaVu Sans"]
MO = ["DejaVu Sans Mono", "Noto Sans CJK JP"]
plt.rcParams["font.family"] = JP
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, MUT, AI, ENJI, RULE = "#191a17", "#6f6f68", "#155f7a", "#a5262f", "#dcd8ce"
BG, CARD = "#f7f6f1", "#fffefa"
FIGW, FIGH = 16.0, 9.2
fig = plt.figure(figsize=(FIGW, FIGH)); fig.patch.set_facecolor(BG)
AX = fig.add_axes([0, 0, 1, 1]); AX.axis("off"); AX.set_xlim(0, 1); AX.set_ylim(0, 1)
LH = lambda s: s * 1.55 / (FIGH * 72)                                   # noqa: E731


def card(x, y, w, h, n, title):
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.005",
                                lw=1.0, ec=RULE, fc=CARD, zorder=1))
    AX.text(x + .010, y + h - .022, n, fontsize=12.5, color=AI, family=MO, weight="bold",
            va="top", zorder=3)
    AX.text(x + .031, y + h - .023, title, fontsize=12.5, color=INK, va="top", zorder=3,
            fontfamily=JP, weight="bold")
    return x + .013, y + h - .066


def T(x, y, txt, size=10.6, color=INK, fam=MO, weight="normal"):
    lh = LH(size)
    for i, ln in enumerate(txt.split("\n")):
        AX.text(x, y - i * lh, ln, fontsize=size, color=color, va="top", zorder=3,
                weight=weight, **({"family": fam} if fam is MO else {"fontfamily": fam}))
    return y - len(txt.split("\n")) * lh


def J(x, y, t, s=10.6, c=INK, w="normal"): return T(x, y, t, s, c, JP, w)


def E(x, y, txt, size=14, color=INK, gap=.052):
    AX.text(x, y, txt, fontsize=size, color=color, va="top", zorder=3, family=MO)
    return y - gap


AX.text(.011, .987, "測り方", fontsize=21, color=INK, fontfamily=JP, weight="bold", va="top")
AX.text(.098, .983, "— 何をどう測ったか。記号はここで定義する。", fontsize=11.5,
        color=MUT, fontfamily=JP, va="top")
AX.plot([.011, .989], [.935, .935], color=INK, lw=1.5)
C1, C2, C3, W = .011, .343, .675, .314

# ================================================== ① ヘッドの寄与を 1 体と 2 体に分ける
x, y = card(C1, .530, W, .385, "①", "ヘッドの寄与を 1 体と 2 体に分ける")
y = J(x, y, "アテンションの出力はヘッドについて厳密に加法的なので、", 10.4, MUT)
y = J(x, y, "ヘッド $h$ が残差ストリームに足すベクトルを取り出せる:", 10.4, MUT)
y = E(x, y - .006, r"$c_h(i) = e^{s_O}\,\left[\mathrm{gate}\odot"
                   r"\mathrm{TanhNorm}\left(\sum_j W^{(h)}_{ij}\hat v^{(h)}_j\right)\right]_i P_O^{(h)}$",
      12.5, INK, .050)
y = J(x, y - .004, "ここで $d_{ii}=0$ なので窓は自分自身に対して必ず開いている。", 10.4, MUT)
y = J(x, y, "つまり $j=i$ の項は相手を見ずに生じる 1 体成分:", 10.4, MUT)
y = E(x, y - .006, r"$c_h^{\rm self} = c_h\ (W_{ij}\to W_{ii}\delta_{ij}),\ \ "
                   r"c_h^{\rm pair} = c_h - c_h^{\rm self}$", 12.5, INK, .050)
y = J(x, y - .002, "$c_h^{\\rm pair}$ が相関を運んでいる部分。寄与率はこれで測る:", 10.4, MUT)
y = E(x, y - .004, r"$q_h = \|c_h^{\rm pair}\|^2 / \sum_{l,h}\|c_h^{\rm pair}\|^2$",
      13, ENJI, .046)
J(x, y - .002, "ヘッドどうしはほぼ直交なので、和を取るときは二乗で効く。", 10.2, MUT)

# ================================================== ② 実効到達距離
x, y = card(C1, .012, W, .500, "②", "アテンションが実際に届いている距離")
y = J(x, y, "ヘッド $h$ が重みを掛けている距離の平均:", 10.4, MUT)
y = E(x, y - .006, r"$\bar d_h = \sum_{i\neq j} W^{(h)}_{ij} d_{ij}\ /\ "
                   r"\sum_{i\neq j} W^{(h)}_{ij}$", 13, INK, .050)
y = J(x, y - .002, "これを寄与率で重み付けした量を、アテンションが使っている", 10.4, MUT)
y = J(x, y, "長さとする:", 10.4, MUT)
y = E(x, y - .006, r"$R_{\rm eff} = \sum_{l,h} q_h\, \bar d_h$", 13.5, ENJI, .050)
y = J(x, y - .004, "補足：$d_{ij}$ は波動関数が使う周期ノルムで、遠方で頭打ち", 10.4, MUT)
y = J(x, y, "になる。$a_M$ などと比べる前に普通の距離へ直す必要がある。", 10.4, MUT)
y = E(x, y - .006, r"$d_{\rm p} = \frac{1}{2\pi}\sqrt{a^\top M a + b^\top M b}$", 12.5, INK, .044)
y = J(x, y, "　$a = 1-\\cos 2\\pi s,\\ b = \\sin 2\\pi s$、$s$ = 分数座標、$M = L^\\top L$", 10.2, MUT)
y = J(x, y - .006, "向きにほとんど依らないので 1 対 1 に変換できる。例:", 10.4, MUT)
y = T(x, y - .004, "  周期ノルム 0.50 0.63 0.79 1.00 1.25 1.58 1.99 2.50\n"
                   "  普通の距離 0.50 0.64 0.81 1.02 1.31 1.70 2.26 3.35", 10.0)

# ================================================== ③ 届く範囲を絞る実験
x, y = card(C2, .512, W, .403, "③", "届く範囲を絞ってエネルギーの代償を測る")
y = J(x, y, "学習済みの波動関数はそのまま、重みに襟を掛けて届く範囲を", 10.4, MUT)
y = J(x, y, "半径 $R$ に縮める（再学習はしない）:", 10.4, MUT)
y = E(x, y - .006, r"$W_{ij} \rightarrow W_{ij}\cdot c_R(d_{ij})$", 13.5, INK, .046)
y = E(x, y - .002, r"$c_R(d) = \frac{1}{2}\left[1+\cos\left(\pi\,"
                   r"\mathrm{clip}\left(\frac{d-(1-\beta)R}{\beta R},0,1\right)\right)\right]$",
      12.5, INK, .052)
y = J(x, y, "$\\beta = 0.3$。$d \\leq 0.7R$ で 1、$d \\geq R$ で 0、境目は滑らか。", 10.2, MUT)
y = J(x, y - .006, "硬く切ってはいけない。不連続だと前進ラプラシアンが運動エネ", 10.4, ENJI)
y = J(x, y, "ルギーの特異項を取りこぼし、偽の低いエネルギーが出る。", 10.4, ENJI)
y = J(x, y - .006, "改めてサンプルして変分エネルギーを測り、代償を見る:", 10.4, MUT)
y = E(x, y - .004, r"$\Delta E(R) = E(R) - E(\infty)$", 13, INK, .044)
J(x, y, "立ち上がり位置 $R^*$ は $\\Delta E$ が最大値の 1 割に落ちる $R$。", 10.2, MUT)

# ================================================== ④ ヘッドを 1 本止める実験
x, y = card(C2, .012, W, .492, "④", "ヘッドを 1 本止めてエネルギーの代償を測る")
y = J(x, y, "窓が広いヘッドはそれだけ多くの対を足し込むので、寄与ベクトル", 10.4, MUT)
y = J(x, y, "の長さ $q_h$ は窓が広いほど自動的に大きくなる。この偏りの無い", 10.4, MUT)
y = J(x, y, "測り方として、ヘッドを 1 本だけ「自分自身しか見ない」状態に", 10.4, MUT)
y = J(x, y, "落とし、エネルギーがどれだけ悪くなるかを見る:", 10.4, MUT)
y = E(x, y - .006, r"$W^{(h)}_{ij} \rightarrow W^{(h)}_{ii}\,\delta_{ij}$", 12.5, INK, .050)
y = E(x, y - .002, r"$\Delta E_h = E_h^{\rm off} - E$", 13, ENJI, .046)
y = J(x, y - .002, "1 体成分は残るので、抜けるのはそのヘッドが運んでいた相関だけ。", 10.4, MUT)
y = J(x, y - .008, "これを窓 $w_h$ の順に並べれば、どの長さの相関が効いているかが", 10.4, MUT)
y = J(x, y, "偏りなく読める。", 10.4, MUT)
y = J(x, y - .008, "※ どちらの実験も、モアレのスーパーセルより広く絞る対照で", 10.2, MUT)
J(x, y, "　 $\\Delta E = 0$ を確認済み。", 10.2, MUT)

# ================================================== ⑤ 秩序変数
x, y = card(C3, .530, W, .385, "⑤", "ハニカム秩序の強さ")
y = J(x, y, "$\\nu = 2/3$ の一般化 Wigner 結晶は、3 つの谷のうち 1 つが空く", 10.4, MUT)
y = J(x, y, "並び。モアレ Brillouin ゾーンの角 $K$ に Bragg ピークが立つ:", 10.4, MUT)
y = E(x, y - .006, r"$S(\mathbf{k}) = \langle\,|\sum_j e^{i\mathbf{k}\cdot\mathbf{r}_j}|^2\,\rangle / N$", 13, INK, .050)
y = E(x, y - .002, r"$\psi_K = \frac{S(K)/S_{\rm bg} - 1}{N/4 - 1}$", 13.5, ENJI, .052)
y = J(x, y, "完全な秩序で 1、乱雑で 0 になるように規格化した（$N/4$ が上限）。", 10.2, MUT)
y = J(x, y - .008, "もう 1 本、谷の占有からも測る。3 つの部分格子ごとに平均を", 10.4, MUT)
y = J(x, y, "取り、そのばらつきを完全秩序の値で割る:", 10.4, MUT)
y = E(x, y - .006, r"$\psi_{\rm sub} = \mathrm{sd}\{\bar n_0,\bar n_1,\bar n_2\} / \sqrt{p(1-p)}$", 12.5, ENJI, .046)
J(x, y, "$p$ = 充填の小数部分。合成配置で較正済み（完全秩序 1.00、乱雑 0.02）。", 10.2, MUT)

# ================================================== ⑥ なぜ ν を振るのか
x, y = card(C3, .012, W, .500, "⑥", "窓が何で決まるかをどう判定するか")
y = J(x, y, "モアレのスーパーセル（電子 $N$ 個と谷 $N_v$ 個が入る箱）の面積は", 10.4, MUT)
y = J(x, y, "$A = \pi N$ に固定される。電子が三角格子を組むときの格子定数は", 10.4, MUT)
y = E(x, y - .006, r"$a_{ee} = \sqrt{2\pi/\sqrt{3}} = 1.905$", 13, INK, .046)
y = J(x, y - .002, "谷の間隔は充填 $\\nu = N/N_v$ だけで決まる:", 10.4, MUT)
y = E(x, y - .006, r"$a_M = \sqrt{2\pi/\sqrt{3}\,\cdot\,\nu} = a_{ee}\sqrt{\nu}$", 13, INK, .046)
y = J(x, y - .002, "$N$ を固定して $N_v$ を変えれば、電子密度はそのままで $a_M$ だけが動く。", 10.4, INK)
y = J(x, y - .006, "スーパーセルの形も $\\nu$ に依らず同じ（面積・辺の長さ・角度がすべて", 10.4, MUT)
y = J(x, y, "一致）なので、窓の梯子も比較対象も変わらない。", 10.4, MUT)
y = J(x, y - .008, "判定：$R_{\\rm eff}$ や $R^{*}$ が", 10.4, INK, "bold")
y = J(x, y - .002, "　$a_M$ に比例して動く　→　外場の周期が決めている", 10.4, MUT)
y = J(x, y, "　$a_{ee}$ のまま動かない　→　電子の間隔が決めている", 10.4, MUT)
J(x, y - .008, "$a_M$ を 2 倍（$\\nu$ = 0.5 → 2）振れば、両者は明確に分かれる。", 10.2, MUT)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_methods.png"),
            dpi=150, facecolor=fig.get_facecolor())
print("saved")
