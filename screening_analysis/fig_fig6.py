"""基底状態の電荷密度と密度‑密度相関（arXiv:2502.05383 Fig.6 に対応する図）。

    python screening_analysis/fig_fig6.py <folder> [<folder> ...] [--out out.png]
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_analysis as MA, sampler as SM      # noqa
from matplotlib.colors import LinearSegmentedColormap  # noqa
import matplotlib.pyplot as plt                        # noqa

CM_N = LinearSegmentedColormap.from_list(
    "n", ["#f7f7f4", "#dbe7ef", "#9dc0dc", "#4f8fc0", "#1d4f80", "#0d2647"])
CM_G = LinearSegmentedColormap.from_list(
    "g", ["#ffffff", "#e6ecdf", "#b7d1a8", "#6aa86e", "#1f6b4a", "#0d3a2c"])


def cell_density(d, R, nb=66):
    L = d["L"]
    f = np.linalg.solve(L, MA.wrap(R, L).reshape(-1, 2).T).T
    H, xe, ye = np.histogram2d(f[:, 0], f[:, 1], bins=nb, range=[[0, 1], [0, 1]])
    U, V = np.meshgrid(xe, ye, indexing="ij")
    return H / H.mean(), U * L[0, 0] + V * L[0, 1], U * L[1, 0] + V * L[1, 1]


def pair_corr(d, R, nb=73):
    L, ne = d["L"], d["ne"]
    dr = R[:, :, None, :] - R[:, None, :, :]
    fr = dr @ np.linalg.inv(L).T
    dv = (fr - np.round(fr)) @ L.T
    xy = dv[:, ~np.eye(ne, dtype=bool)].reshape(-1, 2)
    lim = 0.5 * abs(np.linalg.det(L)) / np.linalg.norm(L[:, 0])
    H, xe, ye = np.histogram2d(xy, bins=nb, range=[[-lim, lim], [-lim, lim]]) \
        if False else np.histogram2d(xy[:, 0], xy[:, 1], bins=nb,
                                     range=[[-lim, lim], [-lim, lim]])
    cellA = (xe[1] - xe[0]) ** 2
    g = H / (len(R) * ne * (ne / abs(np.linalg.det(L))) * cellA)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    m = np.hypot(*np.meshgrid(xc, yc, indexing="ij")) > lim
    return np.ma.array(g, mask=m), xe, ye, lim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nconf", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_fig6.png"))
    a = ap.parse_args()
    style.use()

    D, N, G, O = [], [], [], []
    for f in a.folders:
        d = MA.load(f); R = SM.cached(f, a.nconf)
        dd = dict(d); dd["R"] = R
        D.append(d); N.append(cell_density(d, R)); G.append(pair_corr(d, R))
        O.append((MA.order_parameter(dd), MA.occupation_summary(dd)))
    nc = len(D)
    labs = a.labels.split(",") if a.labels else [r"$r_s = %g$" % d["rs"] for d in D]
    vN = float(max(np.percentile(h, 99.2) for h, _, _ in N))
    vG = float(max(np.percentile(g.compressed(), 99.0) for g, _, _, _ in G))

    W = 3.5 * nc + 1.5
    fig = plt.figure(figsize=(W, 8.25))
    gs = fig.add_gridspec(2, nc + 1, width_ratios=[1] * nc + [0.05],
                          left=0.075, right=0.90, top=0.885, bottom=0.175,
                          hspace=0.10, wspace=0.07)
    for j in range(nc):
        d = D[j]; L = d["L"]
        ax = fig.add_subplot(gs[0, j])
        H, X, Y = N[j]
        im1 = ax.pcolormesh(X, Y, H, cmap=CM_N, shading="flat", vmin=0, vmax=vN)
        P = np.array([[0, 0], L[:, 0], L[:, 0] + L[:, 1], L[:, 1], [0, 0]])
        ax.plot(P[:, 0], P[:, 1], color=style.INK, lw=1.0, zorder=5)
        from matplotlib.patches import Polygon
        clip = Polygon(P[:4], closed=True, transform=ax.transData)
        s = MA.sites_in_cell(d)
        for sh in ([0, 0], L[:, 0], L[:, 1], L[:, 0] + L[:, 1]):
            ln, = ax.plot(s[:, 0] + sh[0], s[:, 1] + sh[1], "o", ms=3.6, mfc="none",
                          mec=style.ENJI, mew=1.0, zorder=6, alpha=0.95)
            ln.set_clip_path(clip)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(min(0, L[0, 1]) - .35, L[0, 0] + max(0, L[0, 1]) + .35)
        ax.set_ylim(min(0, L[1, 1]) - .35, max(L[1, 0], L[1, 1]) + .35)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(labs[j], pad=10, fontsize=14)
        if j == 0: ax.set_ylabel("電荷密度", fontsize=12, color=style.SUB, labelpad=10)

        ax2 = fig.add_subplot(gs[1, j])
        g, xe, ye, lim = G[j]
        im2 = ax2.pcolormesh(xe, ye, g.T, cmap=CM_G, shading="flat", vmin=0, vmax=vG)
        th = np.linspace(0, 2 * np.pi, 240)
        ax2.plot(lim * np.cos(th), lim * np.sin(th), color=style.INK, lw=1.0)
        ax2.plot(0, 0, "+", color=style.ENJI, ms=9, mew=1.7)
        ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlim(-lim * 1.06, lim * 1.06); ax2.set_ylim(-lim * 1.06, lim * 1.06)
        for sp in ax2.spines.values(): sp.set_visible(False)
        if j == 0: ax2.set_ylabel("密度‑密度相関", fontsize=12, color=style.SUB, labelpad=10)

    cax = fig.add_subplot(gs[0, nc]); cb = plt.colorbar(im1, cax=cax)
    _b = cax.get_position(); cax.set_position([_b.x0, _b.y0 + .12 * _b.height,
                                               _b.width, .76 * _b.height])
    cax.set_title(r"$n/\bar{n}$", fontsize=11, pad=8)
    cax2 = fig.add_subplot(gs[1, nc]); cb2 = plt.colorbar(im2, cax=cax2)
    _b = cax2.get_position(); cax2.set_position([_b.x0, _b.y0 + .12 * _b.height,
                                                 _b.width, .76 * _b.height])
    cax2.set_title(r"$g$", fontsize=11, pad=8)

    fig.text(0.075, 0.965, "基底状態の電荷分布", fontsize=16, weight="bold",
             va="top", ha="left")
    body = ("上段は収束した基底状態の電荷密度で、赤丸はモアレポテンシャルの谷の位置。"
            "下段は 1 個の電子（赤十字）から見た他の電子の分布で、こちらはどの秩序ドメイン"
            "を選んだかに依らない。ハニカム秩序（3 つの谷のうち 1 つだけが空く並び）が"
            "立っていれば、下段の最近接の輪が 6 個ではなく 3 個おきに強くなる。")
    # 見出しは測った秩序変数から決める（図に写っていないことを書かない）
    psi = [max(o[0]["psi_K"], o[1]["psi_sub"]) for o in O]
    if max(psi) < 0.35:
        head = "図：どちらも電子は谷に落ちているが、空く谷の並びはまだ揃っていない。"
    elif min(psi) < 0.35:
        head = ("図：相互作用を強めると、全サイト占有の液体からハニカム型の"
                "一般化 Wigner 結晶へ移る。")
    else:
        head = "図：どちらもハニカム秩序が立っている。"
    body2 = body + "　ハニカム秩序の強さ（完全な秩序を 1、乱雑を 0）は " + \
        "、".join("%s で %.2f" % (labs[j].replace("$", ""), psi[j]) for j in range(nc)) + "。"
    style.caption(fig, head, style.wrap(body2, int(W * 11.8)), y=0.035)
    fig.savefig(a.out, dpi=170)
    print("->", a.out)


if __name__ == "__main__":
    main()
