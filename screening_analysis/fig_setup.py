"""セットアップ図。記号はすべてこの図の中で定義する。"""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Polygon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave.pbc import lattices as LT                                  # noqa: E402

for p in fm.findSystemFonts():
    if "NotoSansCJK" in p: fm.fontManager.addfont(p)
JP = ["Noto Sans CJK JP", "DejaVu Sans"]
MO = ["DejaVu Sans Mono", "Noto Sans CJK JP"]
plt.rcParams["font.family"] = JP
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

INK, MUT, AI, ENJI, RULE = "#191a17", "#6f6f68", "#155f7a", "#a5262f", "#dcd8ce"
BG, CARD = "#f7f6f1", "#fffefa"
FIGW, FIGH = 16.0, 10.0
fig = plt.figure(figsize=(FIGW, FIGH)); fig.patch.set_facecolor(BG)
AX = fig.add_axes([0, 0, 1, 1]); AX.axis("off"); AX.set_xlim(0, 1); AX.set_ylim(0, 1)
LH = lambda s: s * 1.55 / (FIGH * 72)                                        # noqa: E731

NEL, NSITE, U, PHI = 18, 27, 1.8374, np.pi / 4
AM = np.sqrt(2 * np.pi / np.sqrt(3) * NEL / NSITE)
SC, MO_LAT = LT._triangular_lattice_vecs_periodic_potential(AM, NSITE)
_v1, _v2 = SC[:, 0] + SC[:, 1], SC[:, 0]          # 同じ格子、60° をなす基底
_th = -np.arctan2(_v1[1], _v1[0])                 # v1 を水平に回す
_R = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
SC = _R @ np.column_stack([_v1, _v2])
MO_LAT = _R @ MO_LAT


def card(x, y, w, h, n, title):
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.005",
                                lw=1.0, ec=RULE, fc=CARD, zorder=1))
    AX.text(x + .010, y + h - .020, n, fontsize=12.5, color=AI, family=MO, weight="bold",
            va="top", zorder=3)
    AX.text(x + .031, y + h - .021, title, fontsize=12.5, color=INK, va="top", zorder=3,
            fontfamily=JP, weight="bold")
    return x + .013, y + h - .060


def T(x, y, txt, size=10.8, color=INK, fam=MO, weight="normal"):
    lh = LH(size)
    for i, ln in enumerate(txt.split("\n")):
        AX.text(x, y - i * lh, ln, fontsize=size, color=color, va="top", zorder=3, weight=weight,
                **({"family": fam} if fam is MO else {"fontfamily": fam}))
    return y - len(txt.split("\n")) * lh


def J(x, y, t, s=10.8, c=INK, w="normal"):
    return T(x, y, t, s, c, JP, w)


def E(x, y, txt, size=14, color=INK, gap=.048):
    AX.text(x, y, txt, fontsize=size, color=color, va="top", zorder=3, family=MO)
    return y - gap


AX.text(.011, .988, "セットアップ", fontsize=21, color=INK, fontfamily=JP, weight="bold", va="top")
AX.plot([.011, .989], [.938, .938], color=INK, lw=1.5)
C1, C2, C3, W = .011, .343, .675, .314

# ================================================================ ① 系
x0, y0, w0, h0 = C1, .490, W, .405
card(x0, y0, w0, h0, "①", "計算する系")
ax = fig.add_axes([x0 + .030, y0 + .150, .252, .200]); ax.set_facecolor(CARD)
G = 4 * np.pi / (np.sqrt(3) * AM)
_g1 = G * np.array([0.0, 1.0])
_g2 = G * np.array([np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)])
GS = [_R @ g for g in (_g1, _g2, -(_g1 + _g2))]
CO = np.array([[0, 0], SC[:, 0], SC[:, 0] + SC[:, 1], SC[:, 1]])
lo, hi = CO[:, 0].min() - .6, CO[:, 0].max() + .6
ylo, yhi = CO[:, 1].min() - .6, CO[:, 1].max() + .6
gx, gy = np.meshgrid(np.linspace(lo, hi, 660), np.linspace(ylo, yhi, 440))
P = np.stack([gx, gy], -1)
VM = -2 * U * sum(np.cos(P @ g + PHI) for g in GS)
f = np.linalg.solve(SC, P.reshape(-1, 2).T).T.reshape(P.shape)
ins = (f[..., 0] > -1e-9) & (f[..., 0] < 1) & (f[..., 1] > -1e-9) & (f[..., 1] < 1)
ax.imshow(np.ma.masked_where(~ins, VM), origin="lower", cmap="Purples_r",
          extent=[gx.min(), gx.max(), gy.min(), gy.max()], interpolation="bilinear")
# 27 個のモアレの谷（V_M の極小）
best = None
for m in range(-6, 7):
    for n in range(-6, 7):
        r = m * MO_LAT[:, 0] + n * MO_LAT[:, 1]
        v = -2 * U * sum(np.cos(np.dot(r, g) + PHI) for g in GS)
        if best is None or v < best[0]: best = (v, r)
shift = best[1] - (np.round(np.linalg.solve(MO_LAT, best[1])) @ MO_LAT.T)
site = []
for m in range(-2, 12):
    for n in range(-2, 12):
        r = shift + m * MO_LAT[:, 0] + n * MO_LAT[:, 1]
        fr = np.linalg.solve(SC, r)
        if -1e-6 < fr[0] < 1 - 1e-6 and -1e-6 < fr[1] < 1 - 1e-6: site.append(r)
site = np.array(site)
ax.plot(site[:, 0], site[:, 1], "o", ms=4.6, mfc="none", mec="#f0b400", mew=1.4, zorder=5)
ax.add_patch(Polygon(CO, closed=True, fill=False, ec=INK, lw=1.6, zorder=4))
for v, lab, dx, dy in [(SC[:, 0], r"$\mathbf{a}_1$", 0, -1.0), (SC[:, 1], r"$\mathbf{a}_2$", -1.1, 0)]:
    ax.annotate("", xy=v, xytext=(0, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(v[0] / 2 + dx, v[1] / 2 + dy, lab, color=INK, fontsize=12, zorder=6)
ax.set_xlim(lo, hi); ax.set_ylim(ylo - .6, yhi); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
xt, yt = x0 + .013, y0 + .128
yt = J(xt, yt, "モアレの谷が $27$ 個。そこに電子 $N = 18$ 個（↑9 ↓9）と、", 10.6)
yt = J(xt, yt, "一様な正電荷の背景を置く。充填 $\\nu = 18/27 = 2/3$。", 10.6)
yt = J(xt, yt, "周期境界。モアレのスーパーセルの面積は $A_{\\rm sc} = \\pi N$、", 10.6)
yt = J(xt, yt, "つまり 1 電子あたり $\\pi$。", 10.6)
J(xt, yt - .006, "背景の色 = モアレポテンシャル（濃い所ほど谷）、黄丸 = 谷の位置。", 9.8, MUT)

# ================================================================ ② ハミルトニアン
x, y = card(C1, .008, W, .468, "②", "ハミルトニアン")
y = E(x, y, r"$H = -\frac{1}{2}\sum_i \nabla_i^{2} \;+\; r_s\!\!\sum_{i<j}\frac{1}{r_{ij}}"
            r"\;+\; \sum_i V_M(\mathbf{r}_i)$", 14.5, ENJI, .062)
y = J(x, y, "$\\mathbf{r}_i$   電子 $i$ の位置、$r_{ij} = |\\mathbf{r}_i - \\mathbf{r}_j|$", 10.6)
y = J(x, y, "$r_s$    密度パラメータ（③で定義）。相互作用の強さを決める", 10.6)
y = E(x, y - .010, r"$V_M(\mathbf{r}) = -2U[\cos(\mathbf{G}_1\!\cdot\!\mathbf{r}+\phi)"
                   r"+\cos(\mathbf{G}_2\!\cdot\!\mathbf{r}+\phi)+\cos(\mathbf{G}_3\!\cdot\!\mathbf{r}+\phi)]$",
      12, INK, .050)
y = J(x, y - .002, "$\\mathbf{G}_1,\\mathbf{G}_2,\\mathbf{G}_3$   周期 $a_M$ の三角格子の逆格子ベクトル（最短の 3 本）:", 10.6)
y = E(x, y - .006, r"$\mathbf{G}_1 = \frac{4\pi}{\sqrt{3}a_M}(0,\,1),\quad"
                   r"\mathbf{G}_2 = \frac{4\pi}{\sqrt{3}a_M}\left(\frac{\sqrt{3}}{2},\,-\frac{1}{2}\right)$",
      12, INK, .050)
y = E(x, y, r"$\mathbf{G}_3 = -(\mathbf{G}_1+\mathbf{G}_2)$", 12, INK, .034)
y = J(x, y, "　　　　　モアレ格子の基本ベクトルは $\\mathbf{a}^M_1 = a_M(1,0)$、", 10.0, MUT)
y = J(x, y, "　　　　　$\\mathbf{a}^M_2 = a_M(-1/2,\\ \\sqrt{3}/2)$", 10.0, MUT)
y = J(x, y, "$U$      ポテンシャルの強さ", 10.6)
y = J(x, y, "$\\phi$      位相", 10.6)
J(x, y, "$a_M$    モアレの周期", 10.6)

# ================================================================ ③ 単位
x, y = card(C2, .665, W, .230, "③", "単位")
y = J(x, y, "　$a_B^{*} = \\epsilon\\,(m_e/m^{*})\\,a_B$　　有効ボーア半径", 11.2)
y = J(x, y, "　$\\mathrm{Ha}^{*} = (m^{*}/m_e)\\,\\epsilon^{-2}\\,\\mathrm{Ha}$　　有効ハートリー", 11.2)
y = J(x, y - .006, "　$m^{*}$ = 有効質量、$\\epsilon$ = 誘電率", 10.2, MUT)
y = J(x, y - .010, "密度パラメータ $r_s$ の定義（$n_e$ = 電子密度）:", 10.8)
y = E(x, y - .002, r"$\pi\,(r_s a_B^{*})^{2} \;=\; 1/n_e$", 13.5, INK, .040)
J(x, y, "　単位長さ $= r_s a_B^{*}$　　単位エネルギー $= \\mathrm{Ha}^{*}/r_s^{2}$", 11.2)

# ================================================================ ④ 設定
x, y = card(C2, .270, W, .387, "④", "設定")
y = T(x, y, "層          2          ヘッド      8\n"
            "attn_dim    16         value_dim   16\n"
            "mlp_dim     64         行列式      4\n"
            "複素出力    True       Jastrow     なし\n"
            "batch       64         最適化      KFAC (lr 0.1)\n"
            "MCMC        10 step    burn-in     100", 9.9)
y = J(x, y - .010, "コード単位での長さ:", 10.8)
y = T(x, y - .004, "  モアレ周期        a_M      = 1.555\n"
                   "  電荷秩序の周期    √3 a_M   = 2.694\n"
                   "  電子の平均間隔             = 1.905\n"
                   "  谷の局在長        ℓ        = 0.33", 9.9)
y = J(x, y - .009, "窓幅 $w_h$ の初期値（8 ヘッド、幾何級数）:", 10.8)
y = T(x, y - .004, "  0.50  0.64  0.81  1.02  1.31  1.70  2.26  3.35", 10.6)
J(x, y - .006, "Trim 受容幅 $r$ の初期値  1.9（全ヘッド共通、最大 2）", 10.8)

# ================================================================ ⑤ スキャン
x, y = card(C2, .008, W, .254, "⑤", "スキャン")
y = J(x, y, "$U = 1.837$、$\\phi = \\pi/4$ は固定。3 通りに振る:", 10.8)
y = J(x, y, "（対照として同じ系を通常の softmax でも解く）", 10.2, MUT)
y = T(x, y - .006, "  相互作用   r_s   = 2, 3.42, 5, 6.83, 9, 14\n"
                   "  窓の倍率   λ     = 0.5, 0.7, 1.0, 1.4, 2.0\n"
                   "  モアレ周期 a_M   = 1.35, 1.56, 1.76, 2.33, 2.69", 10.6)
y = J(x, y - .006, "$r_s$ … 液体から結晶へ渡らせ、ヘッドの分担の変化を見る", 10.2, MUT)
y = J(x, y, "$\\lambda$ … 窓をずらして、系が要求する窓幅を探る", 10.2, MUT)
y = J(x, y, "$a_M$ … 電子密度は変えずに周期だけ動かし、窓の追随を見る", 10.2, MUT)


# ================================================================ ⑥ 波動関数
x, y = card(C3, .008, W, .887, "⑥", "波動関数 — 遮蔽アテンション")
y = J(x, y, "重みの作り方:", 10.8, INK, "bold")
y = J(x, y - .002, "$x_i$ = 層の入力（電子 $i$ の特徴ベクトル）", 10.4, MUT)
y = E(x, y - .006, r"$q_i = x_i P_q, \quad k_i = x_i P_k, \quad v_i = x_i P_v$", 13, INK, .036)
y = J(x, y, "$P$ は学習される射影行列。$q$ = 見る側（query）、", 10.4, MUT)
y = J(x, y, "$k$ = 見られる側（key）、$v$ = 運ぶ中身（value）", 10.4, MUT)
y = E(x, y - .010, r"$s_{ij} = \hat{q}_i\!\cdot\!\hat{k}_j$", 15, INK, .038)
y = J(x, y, "$\\hat q, \\hat k$ は長さ 1 に正規化したもの（$s_{ij} \\in [-1,1]$）", 10.4, MUT)
y = E(x, y - .010, r"$\alpha(s_{ij}) = \left[\max\left(\frac{s_{ij}-(1-r)}{r},\,0\right)\right]^{2}$",
      14.5, INK, .058)
y = J(x, y, "$r$ = 受容幅。$s_{ij} \\leq 1-r$ の対を 0 に落とす", 10.4, MUT)
y = E(x, y - .010, r"$m(d_{ij}) = \frac{1}{2}\left[1+\cos\left(\pi\min(d_{ij}/w,\,1)\right)\right]$",
      14.5, INK, .054)
y = J(x, y, "$d_{ij}$ = 電子 $i, j$ の距離（周期境界込み）、$w$ = 窓幅", 10.4, MUT)
y = J(x, y, "$d_{ij} \\geq w$ で厳密に 0", 10.4, MUT)
y = E(x, y - .012, r"$W_{ij} = \alpha(s_{ij})\; m(d_{ij})$", 15, ENJI, .042)
y = J(x, y - .010, "層の出力:", 10.8, INK, "bold")
y = E(x, y - .004, r"$\mathrm{out}_i = e^{s_O}\sum_{h}\left[\mathrm{gate}\odot"
                   r"\mathrm{TanhNorm}(\mathrm{attn}_h)\right]_i P_O^{(h)}$", 12, INK, .042)
y = J(x, y, "$h$ = ヘッド番号。$\\odot$ = 成分ごとの積。", 10.4, MUT)
y = E(x, y - .006, r"$(\mathrm{attn}_h)_i = \sum_j W^{(h)}_{ij}\,\hat{v}^{(h)}_j$", 12.5, INK, .042)
y = E(x, y - .006, r"$\mathrm{TanhNorm}(\mathbf{u}) = \mathbf{u}\,\tanh|\mathbf{u}|\,/\,|\mathbf{u}|$",
      12.5, INK, .036)
y = E(x, y - .006, r"$\mathrm{gate}_i = \tanh\left(\mathrm{SiLU}(x_i P_g)\right)$", 12.5, INK, .036)
y = J(x, y - .004, "$P_O^{(h)}$ = 出力行列 $P_O$ のヘッド $h$ のブロック、$e^{s_O}$ = 層ごとの学習スカラー", 10.2, MUT)
y = J(x, y, "$\\mathrm{out}_i$ は $x_i$ に足される（残差接続）", 10.4, MUT)
y = J(x, y - .012, "学習されるパラメータ:", 10.8, INK, "bold")
J(x, y - .002, "  $P_q,\\ P_k,\\ P_v,\\ P_g,\\ P_O$、$s_O$、$w$、$r$", 11)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_setup.png"),
            dpi=150, facecolor=fig.get_facecolor())
print("saved")
