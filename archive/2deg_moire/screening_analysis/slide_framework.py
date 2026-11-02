# -*- coding: utf-8 -*-
"""この研究の枠組みと q・k マトリクスの読み方を 1 枚のスライドにまとめる。

    python screening_analysis/slide_framework.py --folder <run> --out slide.png
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

INK, SUB, AI, ENJI, MIDORI, HAI, RULE = ("#141412", "#4a4946", "#12406f", "#a5262f",
                                         "#0f7b6c", "#8a8880", "#d6d3cc")


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
    ap.add_argument("--folder", required=True, help="q・k を出す run（結晶）")
    ap.add_argument("--runs", default="", help="下段用 液体=DIR,境界=DIR,結晶=DIR")
    ap.add_argument("--heads", default="2,15", help="大きく見せるヘッド 2 つ（窓昇順の番号）")
    ap.add_argument("--nconf", type=int, default=64)
    ap.add_argument("--out", default="slide_framework.png")
    ap.add_argument("--pdf", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon, Rectangle
    _font()

    from attn_probe import probe
    from fig_qwin import load as qload, _lattice, _pnorm

    R = probe(a.folder, nconf=a.nconf)
    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    NE, NH, dref = R["NE"], R["NH"], R["dref"]
    nup = int((np.asarray(R["spins"][0]) > 0).sum())
    w = R["w"][0]; order = np.argsort(w)
    Wall = np.asarray(R["W"])[:, 0]                       # [nconf, NH, NE, NE]
    # 窓の外を見せたいので配置平均ではなく 1 配置を使う（平均すると電子が動いて灰色が消える）
    Wm = Wall[0]
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))
    L = _lattice(cfg)

    CM = LinearSegmentedColormap.from_list("blu", ["#ffffff", "#cfe0f2", "#5b93cc", "#123a63"])
    W_, H_ = 13.33, 7.5
    fig = plt.figure(figsize=(W_, H_)); fig.patch.set_facecolor("white")
    _y = lambda inch: 1.0 - inch / H_                                   # noqa: E731
    _x = lambda inch: inch / W_                                         # noqa: E731

    def txt(x, y, s, size=9.6, col=INK, wt="normal", ha="left", va="top", ls=1.75):
        return fig.text(x, y, s, fontsize=size, color=col, weight=wt, ha=ha, va=va,
                        linespacing=ls)

    # ================= 見出し ==============================================
    txt(_x(0.36), _y(0.36),
        "遮蔽アテンション：波動関数が使っている「長さ」を読み出す", 17.0, MIDORI, "bold")
    txt(_x(0.36), _y(0.70),
        "ニューラル波動関数のアテンションに、ヘッドごとに固定した距離の窓を入れる。"
        "どのヘッドが効いたかが、そのまま長さの読み出しになる。", 10.6, INK)
    fig.add_artist(plt.Line2D([_x(0.36), _x(12.97)], [_y(0.92)] * 2, color=RULE, lw=1.0))

    # ================= ① 系 ================================================
    X1 = _x(0.36)
    txt(X1, _y(1.22), "① 系", 11.4, AI, "bold")
    txt(X1, _y(1.52),
        r"$\hat H = -\dfrac{1}{2}\sum_i \nabla_i^{2}"
        r" + r_s \sum_{i<j} r_{ij}^{-1} + \sum_i V(\mathbf{r}_i)$", 12.6, INK)
    txt(X1, _y(2.12),
        r"$V(\mathbf{r}) = -2U \sum_{j=1,2,3} \cos(\mathbf{G}_j \cdot \mathbf{r} + \phi)$",
        11.4, INK)
    txt(X1, _y(2.52),
        "モアレの三角ポテンシャル。$U = 0$ でクーロンのみに戻る。\n"
        "$|\\mathbf{G}_j| = 4\\pi/\\sqrt{3}a_M$,　$a_M = a_{ee}\\sqrt{\\nu}$,　"
        "$\\nu = N/n_{\\rm sites}$", 9.2, SUB)
    txt(X1, _y(3.20),
        "周期セルに $N$ 電子。コード単位では長さ $[r_s a_B^{*}]$ で\n"
        "$A_{\\rm sc} = \\pi N$、$a_{ee} = \\sqrt{2\\pi/\\sqrt{3}} = 1.905$（$r_s$ に依らない）。",
        9.4, INK)
    txt(X1, _y(3.86),
        "相を動かすつまみが 2 つ：$r_s$（密度）と $U$（モアレの深さ）。\n"
        "どちらでも液体 → 結晶に行ける。", 9.4, INK)

    # セルの絵
    axc = fig.add_axes([_x(0.45), _y(5.58), _x(2.3), (5.58 - 4.33) / H_])
    P = np.asarray(np.load(sorted(glob.glob(os.path.join(a.folder, "qmcjax_ckpt_*.npz")))[-1],
                           allow_pickle=True)["data"].item()["positions"])
    P = P.reshape(-1, NE, 2)[0]
    fr = np.mod(np.linalg.solve(L, P.T).T, 1.0); Pc = fr @ L.T
    corners = np.array([[0, 0], L[:, 0], L[:, 0] + L[:, 1], L[:, 1]])
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            sh = m * L[:, 0] + n * L[:, 1]; mid = (m == 0 and n == 0)
            axc.add_patch(Polygon(corners + sh, closed=True, fill=mid,
                                  fc="#f2f4f7" if mid else "none",
                                  ec=INK if mid else HAI, lw=1.5 if mid else .7,
                                  ls="-" if mid else (0, (3, 3)), alpha=1 if mid else .6))
            axc.plot(Pc[:, 0] + sh[0], Pc[:, 1] + sh[1], "o", ms=4.2 if mid else 2.4,
                     color=AI if mid else HAI, alpha=1 if mid else .35,
                     mec="white", mew=.6 if mid else 0)
    axc.set_aspect("equal"); axc.axis("off")
    axc.set_xlim(-1.15 * abs(L[0, 0]) + L[0, 1] * .6, 1.15 * abs(L[0, 0]))
    axc.set_ylim(-1.05 * L[1, 1], 1.15 * L[1, 1])
    txt(_x(0.45), _y(5.66), "周期セル（$N$ = %d、外は繰り返し）" % NE, 8.8, SUB)

    # ================= ② ネットワーク =======================================
    X2 = _x(4.45)
    txt(X2, _y(1.22), "② 遮蔽アテンション", 11.4, AI, "bold")
    txt(X2, _y(1.52),
        r"$W^{(h)}_{ij} = \alpha(s_{ij}) \;\cdot\; m_h(d_{ij})$", 14.0, INK)
    txt(X2, _y(2.02),
        r"$\alpha(s_{ij})$　内容チャネル　… $q\cdot k$ の類似度 $s_{ij}$ から", 9.6, INK)
    txt(X2, _y(2.30),
        r"$m_h(d_{ij})$　距離チャネル　… ヘッドごとに固定した窓", 9.6, ENJI)
    txt(X2, _y(2.68),
        r"$m_h(d) = \dfrac{1}{2}\left[1+\cos\left(\pi\min(d/w_h,1)\right)\right]$", 11.2, INK)
    txt(X2, _y(3.16),
        "$d \\geq w_h$ で厳密に 0。$w_h$ は学習させず固定してある。", 9.6, ENJI)
    txt(X2, _y(3.44),
        "$d$ は周期ノルム（トーラス上の弦）。セル内の最大値が\n"
        "$d_{\\rm ref} = 0.643\\sqrt{N}$ で、これが距離軸の天井。", 9.2, SUB)
    txt(X2, _y(3.98),
        "ふつうのアテンションは全対を見るので「どの長さを使ったか」が残らない。\n"
        "窓を固定すると、ヘッド番号がそのまま長さの目盛りになる。", 9.4, INK)

    axw = fig.add_axes([_x(4.72), _y(5.50), _x(2.95), (5.50 - 4.62) / H_])
    xs = np.linspace(0, dref * 1.02, 300)
    for i, hh in enumerate(order):
        axw.plot(xs, 0.5 * (1 + np.cos(np.pi * np.clip(xs / w[hh], 0, 1))),
                 lw=1.3, color=plt.cm.viridis(i / (NH - 1)))
    axw.set_xlim(0, dref * 1.02); axw.set_ylim(0, 1.08)
    axw.set_xlabel("2 電子の距離 $d$", fontsize=8.2, labelpad=1)
    axw.set_ylabel("$m_h(d)$", fontsize=8.6, labelpad=2)
    axw.tick_params(labelsize=7.6); axw.grid(alpha=.2, lw=.6)
    axw.set_title("%d 本の窓（固定した目盛り）" % NH, fontsize=8.6, pad=2)

    # ================= ③ q・k マトリクス ====================================
    X3 = _x(8.55)
    txt(X3, _y(1.22), "③ $q\\cdot k$ マトリクスの読み方", 11.4, AI, "bold")
    hsel = [int(t) for t in a.heads.split(",")]
    labs = ["窓が狭いヘッド", "窓が広いヘッド"]
    for k, hi in enumerate(hsel):
        hh = order[hi]
        M = Wm[hh].copy()
        off = ~np.eye(NE, dtype=bool)
        vmax = float(np.quantile(M[off], 0.995)) or 1.0
        axm = fig.add_axes([_x(8.62 + k * 2.28), _y(4.25), _x(2.0), 2.0 / H_])
        axm.imshow(np.where(M > 1e-12, M, np.nan), cmap=CM, vmin=0, vmax=vmax,
                   interpolation="nearest")
        axm.set_facecolor("#e9e7e2")                      # 窓の外＝灰色
        axm.axhline(nup - .5, color="#e07b39", lw=1.0)
        axm.axvline(nup - .5, color="#e07b39", lw=1.0)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_edgecolor(HAI); sp.set_linewidth(.8)
        axm.set_title("%s　$w_h$ = %.2f（1 配置）" % (labs[k], w[hh]), fontsize=9.0, pad=4)
    txt(_x(8.57), _y(4.44),
        "行 ＝ 見る側の電子 $i$、　列 ＝ 見られる側の電子 $j$、　"
        "橙線 ＝ スピンの境界（↑%d / ↓%d）\n"
        "灰色 ＝ 窓の外（$d \\geq w_h$ で $W = 0$）。狭いヘッドには相手がほとんど入らない。\n"
        "対角の濃さ ＝ その電子が自分自身の情報をどれだけ使っているか。\n"
        "ブロックごとの濃淡差 ＝ 同スピンと異スピンの扱いの違い（$\\alpha$ 側が担う）。\n"
        "濃さの上限は 3 相で共通に取ってあるので、相を跨いで比べられる。"
        % (nup, NE - nup), 8.5, INK, ls=1.95)

    # ================= ④ 読み出し ==========================================
    fig.add_artist(plt.Line2D([_x(0.36), _x(12.97)], [_y(5.84)] * 2, color=RULE, lw=1.0))
    txt(_x(0.36), _y(6.04), "④ そこから読めるもの", 11.4, AI, "bold")
    if a.runs:
        items = []
        for part in a.runs.split(","):
            lab, f = part.rsplit("=", 1)
            items.append((lab, qload(f, 48)))
        axq = fig.add_axes([_x(0.62), _y(7.14), _x(3.5), (7.14 - 6.38) / H_])
        for i, (lab, d) in enumerate(items):
            axq.plot(d["w"], d["qp"], "-", marker="osd^"[i], ms=3.6, lw=1.5,
                     color=[AI, "#c98a12", ENJI][i], label=lab)
        axq.set_xlabel("固定した窓 $w_h$", fontsize=8.2, labelpad=0.5)
        axq.set_ylabel("寄与率 [%]", fontsize=8.4, labelpad=1)
        axq.tick_params(labelsize=7.4); axq.grid(alpha=.2, lw=.6)
        axq.legend(fontsize=7.4, frameon=False, loc="upper left")
        wq = [float((d["qp"] * d["w"]).sum() / d["qp"].sum()) for _, d in items]
        txt(_x(4.45), _y(6.40),
            "相が硬くなるほど、寄与が広い窓のヘッドへ移る。\n"
            "読み出した長さ $\\langle w\\rangle_q = \\sum_h q_h w_h$ は "
            + " → ".join("%.2f" % v for v in wq) + "。",
            9.6, INK)
        txt(_x(4.45), _y(6.98),
            "$q_h = \\|c_h^{\\rm pair}\\|^{2}/\\sum_{h'}\\|c_{h'}^{\\rm pair}\\|^{2}$"
            "　… ヘッド $h$ が出力に足した 2 体成分の割合。", 8.8, SUB)
    txt(_x(8.62), _y(6.40),
        "これが新しいところ：波動関数の中身から「使っている長さ」が\n"
        "直接取り出せる。従来の NQS では出てこない量である。", 9.6, ENJI)

    fig.savefig(a.out, dpi=170)
    print("  wrote %s" % a.out)
    if a.pdf:
        fig.savefig(a.pdf); print("  wrote %s" % a.pdf)


if __name__ == "__main__":
    main()
