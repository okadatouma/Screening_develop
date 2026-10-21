# -*- coding: utf-8 -*-
"""クーロン 2DEG の液体・境界・結晶を 1 冊にまとめる。"""
import os, sys, json
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

W, H = 16.54, 11.69
M = 0.040
INK, SUB, AI, ENJI, RULE = "#141412", "#4a4946", "#155f7a", "#a5262f", "#d6d3cc"
R = "results/2deg-Coulomb/CustomScreenformer"
RUNS = [("液体", 5, "el13_12_rs5_tri_l2h16_cl05", 0.362, 0.267, 37.5),
        ("境界", 30, "el13_12_rs30_tri_l2h16_cb30", 0.405, 0.171, 47.0),
        ("結晶", 60, "el13_12_rs60_tri_l2h16_cc60", 0.463, 0.141, 50.4)]


def page(pdf, kicker, title, eqs=None, table=None, image=None, eqsize=11.4):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 1 - M
    ax.text(M, y, kicker, fontsize=9, color=AI, va="top"); y -= 0.019
    ax.text(M, y, title, fontsize=17, color=INK, va="top", weight="bold"); y -= 0.036
    ax.plot([M, 1 - M], [y, y], color=RULE, lw=1.0); y -= 0.020
    if eqs:
        elh = eqsize * 2.95 / (H * 72); xe = M + 0.128
        y -= elh * 0.25
        for lab, mth in eqs:
            ax.text(M, y, lab, fontsize=eqsize - 2.6, color=AI, va="top")
            ax.text(xe, y, mth, fontsize=eqsize, color=INK, va="top")
            y -= elh * (1.34 if any(k in mth for k in ("\\sum", "\\frac", "\\sqrt")) else 1.0)
        y -= elh * 0.5
    if table:
        lh = 10.2 * 2.55 / (H * 72)
        hdr, rows = table[0], table[1:]
        ncol = len(hdr); tw = min(1 - 2 * M, 0.115 * ncol + 0.10)
        xs = [M + tw * j / ncol for j in range(ncol)]
        for j, t in enumerate(hdr):
            ax.text(xs[j], y, t, fontsize=10.2, color=INK, va="top", weight="bold")
        y -= lh * 1.4
        ax.plot([M, M + tw], [y + lh * 0.45, y + lh * 0.45], color=RULE, lw=0.9)
        for r in rows:
            for j, t in enumerate(r):
                c = ENJI if (r[0].startswith("結晶")) else SUB
                ax.text(xs[j], y, str(t), fontsize=10.2, color=c, va="top")
            y -= lh
        y -= lh * 0.6
    if image:
        im = mpimg.imread(image)
        ih, iw = im.shape[0], im.shape[1]
        s = min((1 - 2 * M) * W / iw, (y - 0.05) * H / ih)
        dw, dh = iw * s / W, ih * s / H
        axi = fig.add_axes([(1 - dw) / 2, y - dh, dw, dh]); axi.imshow(im); axi.axis("off")
    ax.text(M, 0.018, "クーロン 2DEG — 遮蔽アテンション 2 層 × 8 ヘッド / 2026-08-27",
            fontsize=8, color=SUB, va="bottom")
    pdf.savefig(fig); plt.close(fig)


def main():
    out = os.path.join(HERE, "coulomb_wigner_report.pdf")
    pdf = PdfPages(out)

    tab = [["", "$r_s$", "$E/N$（本計算）", "$E/N$（2D HEG の表式）",
            "$\\psi_6$", "ゆらぎ/平均", "最大窓の自 %"]]
    for lab, rs, t, p6, fl, self_pc in RUNS:
        E = float(open(os.path.join(R, t, "train_stats.csv")).readlines()[-1].split(",")[2])
        th = (-1.1061 / rs + 0.814 / rs ** 1.5) * rs ** 2
        tab.append([lab, "%g" % rs, "%.3f" % (E / 25), "%.3f" % th,
                    "%.3f" % p6, "%.3f" % fl, "%.1f%%" % self_pc])

    page(pdf, "まとめ", "クーロン 2DEG：液体 → 境界 → Wigner 結晶",
         eqs=[("系", r"$N = 25$（↑13 ↓12）, 三角スーパーセル, "
                    r"$H = -\frac{1}{2}\sum_i\nabla_i^2 + r_s\sum_{i<j}r_{ij}^{-1}$"),
              ("ネットワーク", r"遮蔽アテンション 2 層 × 8 ヘッド, attn 16, 行列式 4, "
                           r"batch 2048, 10,000 step, $\eta_0 = 10$, $t_0 = 10^{5}$"),
              ("秩序変数", r"$\psi_6 = \left|\left\langle e^{6i\theta}\right\rangle\right|$"
                        r"（各電子の最近接 6 個の角度）, "
                        r"ゆらぎ/平均 $=\mathrm{sd}(d_{\rm NN})/\langle d_{\rm NN}\rangle$"
                        r"（2 次元の融解の目安 0.1 前後）"),
              ("読み方", r"自 % $=$ 重み全体に占める対角 $W_{ii}$ の割合。"
                       r"大 $=$ 自分の情報を主に使う（局在・結晶側）, "
                       r"小 $=$ 他電子との相関を積極的に使う（液体側）"),
              ("結果", r"$r_s$ を 5 → 60 に上げると密度が一様から三角格子へ変わり、"
                     r"ゆらぎ/平均が $0.27 \rightarrow 0.14$、"
                     r"最大窓ヘッドの自 % が $37.5 \rightarrow 50.4\%$ へ上がる")],
         table=tab)

    page(pdf, "電子密度", "密度と対分布：液体・境界・結晶",
         image=os.path.join(HERE, "fig_coulomb.png"))

    for lab, rs, t, p6, fl, sp in RUNS:
        page(pdf, "$q\\cdot k$ グリッド", "%s　$r_s$ = %g" % (lab, rs),
             image=os.path.join(R, t, t + "_grid.png"))
    for lab, rs, t, p6, fl, sp in RUNS:
        page(pdf, "動径プロファイル", "%s　$r_s$ = %g　— 各ヘッドの重みと窓" % (lab, rs),
             image=os.path.join(R, t, t + "_radial.png"))

    pdf.close()
    print("->", out)


if __name__ == "__main__":
    main()
