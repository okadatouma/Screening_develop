# -*- coding: utf-8 -*-
"""順伝播をフェーズに分けて、value と反対称性とスピンがどこに入るかを 1 枚で示す。

    python screening_analysis/fig_pipeline.py --out fig_pipeline.png
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as ST


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="fig_pipeline.png")
    a = ap.parse_args()
    ST.use()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(13.3, 10.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.3); ax.set_ylim(0, 10.6); ax.axis("off")
    fig.text(0.035, 0.972, "順伝播のフェーズ —— value・反対称性・スピンはどこに入るか",
             fontsize=15.0, weight="bold", color=ST.GREEN, va="top")
    fig.text(0.035, 0.935, "1 つの電子配置 $R = \\{(\\mathbf{r}_i, \\sigma_i)\\}_{i=1}^{N}$ が上から下へ流れる。"
             "右の列は、その段で何が入るか。", fontsize=10.6, color=ST.SUB, va="top")

    X0, W = 0.55, 7.55                       # 箱の左端と幅
    def box(y, h, title, lines, col=ST.INK, fill="#f4f3ef", edge=ST.LINE, lw=1.0):
        p = FancyBboxPatch((X0, y), W, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                           fc=fill, ec=edge, lw=lw)
        ax.add_patch(p)
        ax.text(X0 + 0.18, y + h - 0.16, title, fontsize=11.2, weight="bold", color=col, va="top")
        ax.text(X0 + 0.28, y + h - 0.52, lines, fontsize=9.9, color=ST.INK, va="top",
                linespacing=1.55)
        return y

    def arrow(y1, y2):
        ax.annotate("", xy=(X0 + W / 2, y2), xytext=(X0 + W / 2, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ST.SUB, lw=1.3))

    def note(y, text, col=ST.ENJI, weight="bold", size=10.0):
        ax.annotate("", xy=(X0 + W + 0.05, y), xytext=(X0 + W + 0.45, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
        ax.text(X0 + W + 0.55, y, text, fontsize=size, color=col, va="center", weight=weight,
                linespacing=1.5)

    y = 9.55
    # 0 ------------------------------------------------------------------
    h = 1.22; y -= h
    box(y, h, "0　入力特徴（特徴層）",
        "電子 $i$ ごとに  $[\\sin 2\\pi\\mathbf{G}\\cdot\\mathbf{r}_i,\\ \\cos 2\\pi\\mathbf{G}\\cdot\\mathbf{r}_i,\\ \\sigma_i]$　"
        "（1 電子の周期調和成分 + スピン）\n"
        "対の距離 $d^{\\mathrm{per}}_{ij}$ も計算するが、ストリームには入れず窓にだけ渡す")
    note(y + h * 0.62, "スピンはここで入る（$\\sigma_i = \\pm 1$ を 1 成分として）", col=ST.BLUE)
    arrow(y, y - 0.22)
    # 1 ------------------------------------------------------------------
    h = 0.92; y -= h + 0.22
    box(y, h, "1　埋め込み", "$\\mathbf{x}_i = \\mathbf{f}_i\\,W_{\\mathrm{embed}}$　　（64 次元の残差ストリーム）")
    arrow(y, y - 0.22)
    # 2 ------------------------------------------------------------------
    h = 2.85; y -= h + 0.22
    box(y, h, "2　遮蔽アテンション（1 層 × 16 ヘッド）",
        "2a  射影　$\\hat{\\mathbf{q}}_i = \\widehat{\\mathbf{x}_i W_Q}$,　$\\hat{\\mathbf{k}}_j = \\widehat{\\mathbf{x}_j W_K}$,　"
        "$\\hat{\\mathbf{v}}_j = \\widehat{\\mathbf{x}_j W_V}$　（単位長に正規化）\n"
        "2b  内容　$s_{ij} = \\hat{\\mathbf{q}}_i\\cdot\\hat{\\mathbf{k}}_j$,　"
        "$\\alpha(s_{ij}) = \\left[\\mathrm{relu}(s_{ij} - (1-r))/r\\right]^2$　（Trim）\n"
        "2c  距離　$m_h(d^{\\mathrm{per}}_{ij})$　（窓）　　　　2d  重み　$W^{(h)}_{ij} = \\alpha(s_{ij})\\,m_h(d^{\\mathrm{per}}_{ij})$\n"
        "2e  集約　$\\mathbf{a}^{(h)}_i = \\sum_j W^{(h)}_{ij}\\,\\hat{\\mathbf{v}}_j$\n"
        "2f  TanhNorm → ゲート $\\mathbf{g}_i \\odot$ → $W_O^{(h)}$ → $e^{s_O}$\n"
        "2g  残差　$\\mathbf{h}_i = \\mathbf{x}_i + e^{s_O}\\sum_h \\mathbf{c}_h(i)$,　LayerNorm,　MLP,　LayerNorm",
        fill="#eef4fb", edge="#9bb7dd", lw=1.2)
    note(y + h - 1.55, "value はここ（2e）でだけ使う：\n重み $W_{ij}$ が「$j$ をどれだけ見るか」、$\\hat{\\mathbf{v}}_j$ が「$j$ から何を運ぶか」")
    note(y + h - 0.72, "電子の並べ替えに同変（$i \\leftrightarrow j$ で $\\mathbf{h}_i \\leftrightarrow \\mathbf{h}_j$）。\n反対称ではない", col=ST.SUB, weight="normal", size=9.4)
    arrow(y, y - 0.22)
    # 3 ------------------------------------------------------------------
    h = 0.95; y -= h + 0.22
    box(y, h, "3　軌道", "$\\varphi^{(d)}_k(\\mathbf{h}_i) = (\\mathbf{h}_i W_{\\mathrm{orb}} + \\mathbf{b})_{k,d}$,　"
        "$k = 1..N$,　$d = 1..4$　　（包絡なし・Jastrow なし）")
    arrow(y, y - 0.22)
    # 4 ------------------------------------------------------------------
    h = 1.45; y -= h + 0.22
    box(y, h, "4　行列式",
        "$\\Phi^{(d)}_{ki} = \\varphi^{(d)}_k(\\mathbf{h}_i)$　（$N\\times N$、スピンを分けない full determinant）\n"
        "$\\psi(R) = \\sum_{d=1}^{4}\\det\\Phi^{(d)}$",
        fill="#fbeeec", edge="#d9a09a", lw=1.2)
    note(y + h * 0.55, "反対称性はここだけ：\n$i \\leftrightarrow j$ で列が入れ替わり $\\det$ が符号を変える")
    arrow(y, y - 0.22)
    # 5 ------------------------------------------------------------------
    h = 1.05; y -= h + 0.22
    box(y, h, "5　局所エネルギー",
        "$H = -\\frac{1}{2}\\sum_i \\nabla_i^2 + r_s\\sum_{i<j} v_{\\mathrm{Ewald}}(\\mathbf{r}_{ij})\\ (+\\,V_{\\mathrm{moire}})$,　"
        "$E_L = H\\psi/\\psi$")
    note(y + h * 0.5, "$H$ にスピンを含む項はない。\nスピンが効くのは 4 の反対称性（交換）を通してだけ", col=ST.BLUE)

    fig.savefig(a.out, dpi=170, facecolor=fig.get_facecolor())
    print("  wrote", a.out)


if __name__ == "__main__":
    main()
