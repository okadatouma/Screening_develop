"""結合を強めるとどのチャネルに重みが移るか — と、窓が学習されていないこと。

    python screening_analysis/fig_channels.py

① チャネル（層×ヘッド）ごとの寄与シェアの r_s 依存
② 狭い窓 / 広い窓の合計シェア
③ 窓 w は初期値から動いていない
④ 収束チェック: 変分エネルギー vs 解析的な調和結晶
"""
import os, sys, glob, json, re
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from head_contrib import cached                                   # noqa: E402
from periodicwave import CustomScreenformer as CS                 # noqa: E402

for p in fm.findSystemFonts():
    if "NotoSansCJK" in p: fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, BLUE, ORANGE, GREEN, RED = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#008300", "#a5262f"
ROOT = os.path.dirname(HERE)
R9 = os.path.join(ROOT, "results/2deg-Coulomb/CustomScreenformer/el5_4_rs%d_tri_det")
G49 = os.path.join(ROOT, "gpu/results/2deg-Coulomb/CustomScreenformer/el25_24_rs5_tri_l2h16")
RS = [1, 5, 20, 30, 35, 40]

sh, W = {}, None
for r in RS:
    C, X, Rn, w = cached(R9 % r, nconf=48)
    sh[r] = 100 * C ** 2 / (C ** 2).sum(); W = w
NL, NH = W.shape

fig = plt.figure(figsize=(15.2, 9.2)); fig.patch.set_facecolor("#fcfcfb")
gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.26, left=.058, right=.985, top=.885, bottom=.185)

# ---------------- ① チャネル別
ax = fig.add_subplot(gs[0, 0])
cm = plt.get_cmap("coolwarm"); lab = []
wn = (np.log(W) - np.log(W.min())) / (np.log(W.max()) - np.log(W.min()))
for l in range(NL):
    for h in range(NH):
        y = [sh[r][l, h] for r in RS]
        ax.plot(RS, y, "o-", color=cm(wn[l, h]), lw=1.7, ms=4.5, alpha=.9)
        lab.append((y[-1], l, h))
lab.sort(reverse=True)
used = []
for yv, l, h in lab[:6]:
    yy = yv
    while any(abs(yy - u) < 1.3 for u in used):
        yy += 1.3
    used.append(yy)
    ax.annotate(" L%dH%d  w/d=%.2f" % (l, h, W[l, h]), (RS[-1], yv), xytext=(48, yy),
                fontsize=8.2, color=cm(wn[l, h]), va="center",
                arrowprops=dict(arrowstyle="-", color=cm(wn[l, h]), lw=.7, alpha=.6))
ax.set_xscale("log"); ax.set_xticks(RS); ax.set_xticklabels([str(r) for r in RS])
ax.set_xlim(0.85, 95)
ax.set_xlabel("$r_s$"); ax.set_ylabel("波動関数への寄与シェア [%]")
ax.set_title("① 結合を強めるとチャネルが組み替わる（色 = 窓の広さ：青が狭い、赤が広い）",
             fontsize=11, loc="left")
ax.grid(alpha=.25)

# ---------------- ② 狭い / 広い
ax = fig.add_subplot(gs[0, 1])
nar = [sum(sh[r][W < 0.7]) for r in RS]
mid = [sum(sh[r][(W >= 0.7) & (W <= 1.0)]) for r in RS]
wid = [sum(sh[r][W > 1.0]) for r in RS]
ax.plot(RS, nar, "o-", color=BLUE, lw=2.6, ms=8, label="狭い窓 $w/d<0.7$")
ax.plot(RS, wid, "s-", color=RED, lw=2.6, ms=7, label="広い窓 $w/d>1.0$")
ax.set_xscale("log"); ax.set_xticks(RS); ax.set_xticklabels([str(r) for r in RS])
ax.set_xlabel("$r_s$"); ax.set_ylabel("合計シェア [%]")
ax.annotate("10.4% → 24.7%\n（+137%、$+6.7$ pt/decade）", (7, 30), color=BLUE, fontsize=10)
ax.annotate("89.6% → 75.3%", (7, 68), color=RED, fontsize=10)
ax.set_title("② 局在が進むと短距離チャネルに重みが移る", fontsize=11, loc="left")
ax.legend(fontsize=9.5, loc="center right"); ax.grid(alpha=.25); ax.set_ylim(0, 100)

# ---------------- ③ 窓は動いていない
ax = fig.add_subplot(gs[1, 0])
init4 = np.geomspace(0.8 / 2.5, 0.8 * 2.5, 4)
init16 = np.geomspace(0.8 / 2.5, 0.8 * 2.5, 16)


def trained(fold, nh):
    ck = sorted(glob.glob(os.path.join(fold, "qmcjax_ckpt_*.npz")))[-1]
    P = jax.tree_util.tree_map(lambda x: np.asarray(x)[0],
                               np.load(ck, allow_pickle=True)["params"].item())
    cfg = json.load(open(os.path.join(fold, "config.json")))
    nl = cfg["network"]["CustomScreenformer"]["num_layers"]
    om = np.stack([P["layers"]["attention"][l]["mask_omega"] for l in range(nl)])
    return np.log1p(np.exp(om))


ax.plot([0.2, 2.4], [0.2, 2.4], "-", color=MUTED, lw=1.2, label="動いていない場合（$y=x$）")
for fold, nh, ini, c, lab, mk in [(R9 % 1, 4, init4, BLUE, "$N$=9 $r_s$=1", "o"),
                                  (R9 % 40, 4, init4, ORANGE, "$N$=9 $r_s$=40", "o"),
                                  (G49, 16, init16, RED, "$N$=49 $r_s$=5 (16 head)", "s")]:
    tw = trained(fold, nh)
    for l in range(tw.shape[0]):
        ax.plot(ini, tw[l], mk, color=c, ms=7, alpha=.75,
                label=lab if l == 0 else None, mew=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("初期値 $w/d_{\\rm ref}$（設定で決まる幾何級数の梯子）")
ax.set_ylabel("学習後 $w/d_{\\rm ref}$")
ax.set_title("③ 窓 $w$ は学習されていない — 全点が対角線上", fontsize=11, loc="left")
ax.annotate("唯一動いたのが最も狭いヘッド（$-3$〜$-12\\%$）", (0.30, 0.215), fontsize=8.6,
            color=INK, ha="left")
ax.legend(fontsize=8.6, loc="lower right"); ax.grid(alpha=.25, which="both")

# ---------------- ④ 収束チェック
ax = fig.add_subplot(gs[1, 1])
V1, SZ, N = -9.954923, 13.655952, 9
nn = {5: -33.264, 20: -166.058, 30: -256.924, 35: -304.997, 40: -350.705}
k = sorted(nn)
dd = [100 * (nn[r] - (r * V1 + 0.5 * np.sqrt(r) * SZ)) / abs(r * V1 + 0.5 * np.sqrt(r) * SZ) for r in k]
ax.axhline(0, color=GREEN, lw=2.4)
ax.annotate(" 解析的な調和 Wigner 結晶（この線）", (5.2, 0.12), color=GREEN, fontsize=9.5, va="bottom")
ax.plot(k, dd, "o-", color=RED, lw=2.4, ms=9)
for r, d in zip(k, dd):
    ax.annotate("  %+.1f%%" % d, (r, d), fontsize=9, color=RED, va="center")
ax.fill_between([4, 50], 0, 4.2, color=RED, alpha=.06)
ax.annotate("NN が解析解より高い\n= 400 step では収束していない", (30, 2.6), color=RED,
            fontsize=9.5, ha="center")
ax.set_xscale("log"); ax.set_xticks(k); ax.set_xticklabels([str(r) for r in k])
ax.set_xlim(4, 50); ax.set_ylim(-0.4, 4.2)
ax.set_xlabel("$r_s$"); ax.set_ylabel("$(E_{\\rm NN}-E_{\\rm harm})/|E_{\\rm harm}|$  [%]")
ax.set_title("④ 収束チェック — NN は解析解に届いていない", fontsize=11, loc="left")
ax.grid(alpha=.25)

fig.suptitle("チャネルは組み替わる。ただし窓は学習されておらず、$N=9$ の状態は結晶ではない",
             fontsize=13.5, x=.006, ha="left", y=.972)
fig.text(.006, .012,
         "①② シェアは $\\|c_h\\|^2/\\sum\\|c\\|^2$（全 16 チャネルで 100%）。$N=9$, 4 層×4 ヘッド、種固定なのでチャネル番号は $r_s$ 間で対応する。"
         "順位相関は 0.92 と高いが、個々のシェアは 2〜4 倍動く（L3H1: +363%、L3H2: $-49\\%$）。\n"
         "③ 窓の初期値は $\\mathrm{geomspace}(0.8/2.5,\\ 0.8\\times2.5,\\ H)$ で設定から決まる。学習後の変化は $N$=49 で 0〜1%、$N$=9 で最も狭いヘッド以外は 2% 未満。"
         "→ 「学習された $w$」として報告していた値は実質的に初期値。\n"
         "④ 調和結晶は変分上界ではないが、大きい $r_s$ では漸近的に正確。NN がその上にある（$r_s$=20〜40 で $+1.0$〜$+1.7\\%$）＝ 400 step では収束していない。"
         "$S(G_1)$/背景 も $r_s$=40 で 2.7（真の結晶なら $\\approx N=9$）。", fontsize=8.8, color=MUTED)
fig.savefig(os.path.join(HERE, "fig_channels.png"), dpi=150, facecolor=fig.get_facecolor())
print("saved fig_channels.png")
