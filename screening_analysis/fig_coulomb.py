"""純クーロン（モアレなし）の液体 → 境界 → 結晶。電子密度と対分布を 3 点で並べる。

    python screening_analysis/fig_coulomb.py <液体> <境界> <結晶> [--out ...]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_analysis as MA, sampler as SM        # noqa
import matplotlib.pyplot as plt                          # noqa
from matplotlib.patches import Polygon                   # noqa


def psi6_and_fluct(d, nmax=300):
    L = np.asarray(d["L"]); ne = int(d["ne"])
    X = np.asarray(d["R"])[:nmax]
    p6, nn = [], []
    for RR in X:
        dr = RR[:, None, :] - RR[None, :, :]
        f = dr @ np.linalg.inv(L).T
        dv = (f - np.round(f)) @ L.T
        dd = np.linalg.norm(dv, axis=-1); np.fill_diagonal(dd, 1e9)
        for i in range(ne):
            k = np.argsort(dd[i])[:6]
            th = np.arctan2(dv[i, k, 1], dv[i, k, 0])
            p6.append(np.abs(np.mean(np.exp(6j * th))))
            nn.append(dd[i, k[0]])
    nn = np.array(nn)
    return float(np.mean(p6)), float(nn.std() / nn.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs=3)
    ap.add_argument("--labels", default="液体,境界,結晶")
    ap.add_argument("--nconf", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_coulomb.png"))
    ap.add_argument("--subtitle", default="$N$ = 25（↑13 ↓12）・三角スーパーセル・"
                    "遮蔽アテンション 2 層 × 8 ヘッド・batch 2048・10,000 step")
    a = ap.parse_args()
    style.use()
    labs = a.labels.split(",")

    G = []
    for f in a.folders:
        d = MA.load(f.rstrip("/"))
        d = dict(d); d["R"] = np.asarray(SM.cached(f.rstrip("/"), a.nconf))
        d["R"] = d["R"].reshape(-1, int(d["ne"]), 2)
        n, xe, ye = MA.density(d, nb=80)
        g, gx, gy, lim = MA.pair2d(d, nb=71)
        p6, fl = psi6_and_fluct(d)
        G.append(dict(d=d, n=n, g=g, lim=lim, p6=p6, fl=fl,
                      rs=float(d["rs"]), L=np.asarray(d["L"])))

    Wf, Hh = 13.6, 8.4
    fig = plt.figure(figsize=(Wf, Hh))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.965, top=1 - 1.28 / Hh,
                          bottom=2.05 / Hh, hspace=0.30, wspace=0.16)

    vmax_n = max(np.percentile(g["n"], 99.5) for g in G)
    vmax_g = max(np.percentile(g["g"], 99.5) for g in G)
    for j, (g, lab) in enumerate(zip(G, labs)):
        L = g["L"]
        # ---- 上段：折り返した電子密度（分数座標 -> 実座標の平行四辺形へ）
        ax = fig.add_subplot(gs[0, j])
        nb = g["n"].shape[0]
        fx, fy = np.meshgrid((np.arange(nb) + .5) / nb, (np.arange(nb) + .5) / nb,
                             indexing="ij")
        P = np.stack([fx.ravel(), fy.ravel()], 1) @ L.T
        ax.tripcolor(P[:, 0], P[:, 1], g["n"].ravel(), cmap="magma",
                     vmin=0, vmax=vmax_n, shading="gouraud")
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ L.T
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec="white", lw=1.2))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title("%s　$r_s$ = %g" % (lab, g["rs"]), fontsize=14, pad=9,
                     color=(style.ENJI if j == 2 else style.INK),
                     weight=("bold" if j == 2 else "normal"))
        if j == 0:
            ax.set_ylabel("電子密度 $n(\\mathbf{r})$", fontsize=12.5, labelpad=10)

        # ---- 下段：1 個の電子から見た他の電子の分布
        axb = fig.add_subplot(gs[1, j])
        lim = g["lim"]
        axb.imshow(g["g"].T, origin="lower", cmap="magma", vmin=0, vmax=vmax_g,
                   extent=[-lim, lim, -lim, lim])
        axb.plot(0, 0, "+", color="white", ms=11, mew=2)
        axb.set_xticks([]); axb.set_yticks([])
        for sp in axb.spines.values(): sp.set_visible(False)
        axb.text(0.5, -0.085, "$\\psi_6$ = %.3f　　ゆらぎ/平均 = %.3f" % (g["p6"], g["fl"]),
                 transform=axb.transAxes, ha="center", va="top", fontsize=11.5,
                 color=(style.ENJI if j == 2 else style.SUB),
                 weight=("bold" if j == 2 else "normal"))
        if j == 0:
            axb.set_ylabel("対分布 $g(x,y)$", fontsize=12.5, labelpad=10)

    fig.text(0.030, 1 - 0.26 / Hh, "クーロン 2DEG：液体から Wigner 結晶へ",
             fontsize=17, weight="bold", va="top", ha="left")
    fig.text(0.030, 1 - 0.66 / Hh,
             a.subtitle,
             fontsize=11.5, color=style.SUB, va="top", ha="left")
    body = ("上段はスーパーセルへ折り返した 1 体密度。$r_s$ = 5 では一様、"
            "$r_s$ = 60 では 25 個の輝点が三角格子を組む。"
            "下段は 1 個の電子から見た他の電子の分布で、こちらは結晶ドメインの"
            "選び方に依らない量。$r_s$ = 60 では中心の相関ホールの外に"
            "配位殻がはっきり立つ。$\\psi_6$ は各電子の最近接 6 個の角度から測った"
            "六方晶秩序、ゆらぎ/平均は最近接距離のばらつき（Lindemann 比に相当、"
            "2 次元の融解の目安は 0.1 前後）。")
    style.caption(fig, "図：$r_s$ を上げると密度が一様から三角格子へ変わり、"
                  "最近接距離のばらつきが 0.27 から 0.14 へ落ちる。",
                  style.wrap(body, int(Wf * 12.6)), y=0.026)
    fig.savefig(a.out, dpi=165)
    print("->", a.out)
    for g, lab in zip(G, labs):
        print("  %-6s r_s=%-4g psi6=%.3f  ゆらぎ/平均=%.3f" % (lab, g["rs"], g["p6"], g["fl"]))


if __name__ == "__main__":
    main()
