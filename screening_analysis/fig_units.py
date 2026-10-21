# -*- coding: utf-8 -*-
"""単位・長さ・d_ref・w_h の関係を 1 枚にまとめた資料。

    python screening_analysis/fig_units.py --out sheet_units.pdf

N, a_M, r_s, A_sc, n_e, w_h の単位, d^per がどうつながっていて、
なぜ N = 25 の三角セルで窓の上限が 3.2152 になるのかを式で示す。
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

INK, SUB, AI, ENJI, MIDORI, RULE = "#141412", "#4a4946", "#155f7a", "#a5262f", "#0f7b6c", "#d6d3cc"


def _font():
    import matplotlib
    from matplotlib import font_manager as fm
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p:
            fm.fontManager.addfont(p)
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sheet_units.pdf")
    ap.add_argument("--png", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()

    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))
    C_TRI = float(3 ** 0.75 / (2 * np.sqrt(np.pi)))       # 菱形の d_ref/√N
    C_SQ = float(np.sqrt(2 / np.pi))                      # 正方形の d_ref/√N

    W, H = 16.54, 11.69
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")

    def txt(x, y, s, size=10.2, col=INK, w="normal", ha="left", va="top", ls=1.75):
        return fig.text(x, y, s, fontsize=size, color=col, weight=w, ha=ha, va=va,
                        linespacing=ls)

    def rule(y, x0=0.030, x1=0.970, c=RULE, lw=1.0):
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color=c, lw=lw))

    # ---- 見出し -----------------------------------------------------------
    txt(0.030, 0.965, "単位・長さ・$d_{\\rm ref}$・窓 $w_h$ の関係", 17.5, MIDORI, "bold")
    txt(0.030, 0.930,
        "なぜ $N = 25$ の三角スーパーセルで窓の上限が 3.2152 になるのか", 11.6, INK)
    rule(0.912)

    L, R = 0.030, 0.560          # 左段・右段の左端
    y = 0.885
    DY = 0.0455                  # 1 ステップの見出し間隔

    def step(y, num, head, eqs, note=None, x=L, size=12.2):
        txt(x, y, "%s  %s" % (num, head), 11.0, AI, "bold")
        yy = y - 0.030
        for e in eqs:
            txt(x + 0.014, yy, e, size, INK)
            yy -= 0.037 if size >= 12 else 0.031
        if note:
            txt(x + 0.014, yy - 0.004, note, 9.5, SUB)
            yy -= 0.030
        return yy

    # ---- 左段：導出の鎖 ----------------------------------------------------
    y = step(y, "①", "単位のとり方",
             [r"長さ $[\,r_s a_B^{*}\,]$,　エネルギー $[\,{\rm Ha}^{*}/r_s^{2}\,]$",
              r"$\hat H = -\dfrac{1}{2}\sum_i \nabla_i^{2}\; + \; r_s \sum_{i<j} r_{ij}^{-1}$"],
             "$r_s$ はハミルトニアンの結合定数としてだけ入る。以下の長さはすべて $r_s$ に依らない。")

    y = step(y - 0.004, "②", "密度 $n_e$ → スーパーセルの面積 $A_{\\rm sc}$",
             [r"$\dfrac{1}{n_e} = \pi\,(r_s a_B^{*})^{2}$"
              r"　（2 次元での $r_s$ の定義）",
              r"$\Rightarrow$　電子 1 個あたりの面積 $= \pi$,　"
              r"$\;A_{\rm sc} = \pi N$"])

    y = step(y - 0.004, "③", "面積 → 三角格子の格子定数 $a_{ee}$",
             [r"$\dfrac{\sqrt{3}}{2}\,a_{ee}^{2} = \pi$"
              r"　$\Rightarrow$　$a_{ee} = \sqrt{2\pi/\sqrt{3}} = 1.904626$",
              r"$|\mathbf{L}_1| = |\mathbf{L}_2| = \sqrt{N}\, a_{ee}$,　なす角 $120^\circ$,　"
              r"$\mathbf{L} = (\mathbf{L}_1\;\mathbf{L}_2)$"])

    y = step(y - 0.004, "④", "2 電子の距離（周期ノルム）",
             [r"$d^{\rm per} = \dfrac{1}{2\pi}\sqrt{\sum_{mn} G_{mn}"
              r"\left(a_m a_n + b_m b_n\right)}$,　$G = \mathbf{L}^{T}\mathbf{L}$",
              r"$a_m = 1-\cos 2\pi f_m$,　$b_m = \sin 2\pi f_m$",
              r"$= \dfrac{1}{\pi}\sqrt{\sum_{mn} G_{mn}\,"
              r"\sin \pi f_m \sin \pi f_n \cos \pi (f_m - f_n)}$"],
             "$f_m$ は 2 電子の隔たりの分数座標。単位は長さと同じ $[\\,r_s a_B^{*}\\,]$。")

    txt(L, y - 0.004, "⑤  その最大値 $d_{\\rm ref}$", 11.0, AI, "bold")
    txt(L + 0.014, y - 0.030,
        r"菱形は $G_{12}/G_{11} = -1/2$。最大は $f = (2/3,\ 1/3)$ で",
        10.2, INK)
    txt(L + 0.014, y - 0.058,
        r"$\sum_{mn} G_{mn}(\cdots) = \left(\sqrt{N} a_{ee}\right)^{2}\cdot\dfrac{9}{8}$",
        12.2, INK)
    txt(L + 0.014, y - 0.099,
        r"$d_{\rm ref} = \dfrac{\sqrt{N}\,a_{ee}}{\pi}\sqrt{\dfrac{9}{8}}"
        r" = \dfrac{3^{3/4}}{2\sqrt{\pi}}\,\sqrt{N} = 0.643037\,\sqrt{N}$",
        13.0, ENJI)
    y = y - 0.148

    y = step(y, "⑥", "窓 $w_h$ の単位と上限",
             [r"$w_h = d_{\rm ref}\cdot{\rm softplus}(\omega_h)$"
              r"　$[\,r_s a_B^{*}\,]$　… 距離と同じ単位",
              r"どの 2 電子も $d^{\rm per} \leq d_{\rm ref}$"
              r"　$\Rightarrow$　$w_h > d_{\rm ref}$ なら全対で $m_h > 0$"],
             "そのヘッドは距離で選ぶ働きを失う。したがって梯子の上限は $d_{\\rm ref}$。")

    # ---- 右段：数値と含意 --------------------------------------------------
    ry = 0.885
    fig.patches.append(plt.Rectangle((R - 0.006, ry - 0.112), 0.418, 0.118,
                                     transform=fig.transFigure, fc="#fdf3f2",
                                     ec=ENJI, lw=1.0, zorder=0))
    txt(R + 0.010, ry - 0.004, "結論", 10.6, ENJI, "bold")
    txt(R + 0.010, ry - 0.038,
        r"$N = 25$：　$d_{\rm ref} = 0.643037 \times \sqrt{25} = 3.2152$",
        13.0, ENJI)
    txt(R + 0.010, ry - 0.082,
        "これが窓の上限が 3.2 である理由。$N$ とセルの形だけで決まる。", 10.2, INK)

    ry -= 0.142
    txt(R, ry, "数値", 11.0, AI, "bold")
    rows = [["$N$", "$n$", r"$|\mathbf{L}_1| = \sqrt{N}a_{ee}$",
             r"$A_{\rm sc}/\pi$", r"$d_{\rm ref}$ 菱形", r"$d_{\rm ref}$ 長方形"]]
    for n in (5, 6, 7, 10, 12):
        N = n * n
        rows.append(["%d" % N, "%d" % n, "%.4f" % (n * a_ee), "%d" % N,
                     "%.4f" % (C_TRI * np.sqrt(N)), "%.4f" % (C_SQ * np.sqrt(N))])
    xs = [R + 0.004, R + 0.042, R + 0.078, R + 0.190, R + 0.240, R + 0.320]
    for j, c in enumerate(rows[0]):
        txt(xs[j], ry - 0.030, c, 9.6, SUB, "bold")
    rule(ry - 0.043, R, R + 0.412, RULE, 0.9)
    for i, r in enumerate(rows[1:]):
        for j, c in enumerate(r):
            txt(xs[j], ry - 0.052 - i * 0.026, c, 9.8,
                ENJI if r[0] == "25" else INK)
    ry -= 0.052 + 5 * 0.026 + 0.016

    txt(R, ry, "$r_s$ 依存性", 11.0, AI, "bold")
    txt(R + 0.010, ry - 0.030,
        "$a_{ee}$,　$A_{\\rm sc}$,　$d_{\\rm ref}$,　$w_h$ —— コード単位ではどれも $r_s$ に依らない。\n"
        "$r_s$ は ① の結合定数としてのみ効く。したがって $r_s$ を 5 → 60 に変えても\n"
        "梯子はまったく同じ目盛りのままで、3 つの相を同一の横軸の上で比べられる。",
        10.0, INK)
    ry -= 0.132

    txt(R, ry, "セルの形", 11.0, AI, "bold")
    txt(R + 0.010, ry - 0.030,
        r"菱形（$120^\circ$）　$d_{\rm ref} = \dfrac{3^{3/4}}{2\sqrt{\pi}}\sqrt{N}"
        r" = 0.643037\sqrt{N}$", 11.4, INK)
    txt(R + 0.010, ry - 0.072,
        r"長方形　　　　　$d_{\rm ref} = \dfrac{\sqrt{L_x^{2}+L_y^{2}}}{\pi}"
        r" \to \sqrt{\dfrac{2}{\pi}}\sqrt{N} = 0.797885\sqrt{N}$", 11.4, INK)
    txt(R + 0.010, ry - 0.112,
        "菱形は非対角成分 $-1/2$ の打ち消しがあるぶん不利で、同じ $N$ で長方形が 24.1% 広い。",
        9.8, SUB)
    ry -= 0.150

    txt(R, ry, "モアレを入れた場合", 11.0, AI, "bold")
    txt(R + 0.010, ry - 0.030,
        r"$a_M = \sqrt{\dfrac{2\pi}{\sqrt{3}}\cdot\dfrac{N}{n_{\rm sites}}}"
        r" = a_{ee}\sqrt{\nu}$,　$\nu = N/n_{\rm sites}$", 11.4, INK)
    txt(R + 0.010, ry - 0.072,
        "$\\nu = 1$ では $a_M = a_{ee} = 1.9046$（これまでの 3 本がこれ）。$\\nu = 2/3$ なら\n"
        "$a_M = 1.5551$、結晶周期 $\\sqrt{3}a_M = 2.6935$ で、$a_{ee}$ と 3 本に分かれる。",
        10.0, INK)

    rule(0.038)
    txt(0.030, 0.028,
        "遮蔽アテンション / クーロン 2DEG　—　"
        "$d_{\\rm ref}$ は周期ノルムがセル内で取り得る最大値。$N$ とセルの形だけで決まり、"
        "波動関数にも $r_s$ にも依らない。", 9.2, SUB)

    fig.savefig(a.out, format="pdf" if a.out.endswith(".pdf") else None)
    print("  wrote %s" % a.out)
    if a.png:
        fig.savefig(a.png, dpi=150)
        print("  wrote %s" % a.png)


if __name__ == "__main__":
    main()
