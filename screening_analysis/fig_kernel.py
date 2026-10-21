"""ヘッドが実際に届いている距離を、モアレ周期や電子間隔と並べて見る。

    python screening_analysis/fig_kernel.py <folderA> [<folderB> ...]
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, reach, moire_analysis as MA           # noqa
import matplotlib.pyplot as plt                     # noqa
from matplotlib.colors import Normalize             # noqa


def grab(folder, nconf=48):
    r = reach.compute(folder, nconf)
    from head_contrib import cached as HC
    z = HC(folder, nconf=nconf); Cp = z["Cp"]
    SQ = 100 * Cp ** 2 / (Cp ** 2).sum()          # 2 体の寄与だけで見た割合
    d = MA.load(folder)
    r["SQ"] = SQ
    r["aM"] = float(np.linalg.norm(d["MO"][:, 0]))
    r["rs"] = float(d["rs"])
    r["rho"] = float((SQ * r["dbar"]).sum() / SQ.sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nconf", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_kernel.png"))
    a = ap.parse_args()
    style.use()
    G = [grab(f, a.nconf) for f in a.folders]
    labs = a.labels.split(",") if a.labels else [r"$r_s=%g$" % g["rs"] for g in G]
    nc = len(G)
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))
    XMAX = 3.6

    W = 4.5 * nc + 1.15; Hh = 8.0
    fig = plt.figure(figsize=(W, Hh))
    gs = fig.add_gridspec(2, nc, height_ratios=[1.5, 1], left=0.10,
                          right=1 - 1.05 / W, top=1 - 1.35 / Hh, bottom=2.15 / Hh,
                          hspace=0.30, wspace=0.14)
    cmap = plt.get_cmap("viridis_r")
    nrm = Normalize(0, float(max(g["SQ"].max() for g in G)))
    for j, g in enumerate(G):
        ax = fig.add_subplot(gs[0, j])
        ok = g["cnt"] >= 8            # 対がほとんど無い距離は描かない（統計が無い）
        for l in range(g["NL"]):
            for h in range(g["NH"]):
                p = np.where(ok, g["prof"][l, h], np.nan)
                if not np.isfinite(p).any() or np.nanmax(p) <= 0: continue
                p = p / np.nanmax(p)
                q = float(g["SQ"][l, h])
                ax.plot(g["d"], p, lw=0.8 + 3.4 * np.sqrt(q / 100.0),
                        color=cmap(nrm(q)), alpha=0.45 + 0.55 * min(q / 30.0, 1.0),
                        zorder=2 + q / 20.0,
                        ls="-" if l == 0 else (0, (4, 1.6)))
        for x, t, c in ((g["aM"], "$a_M$", style.ENJI),
                        (a_ee, "$a_{ee}$", style.ORANGE),
                        (np.sqrt(3) * g["aM"], r"$\sqrt{3}a_M$", style.PURPLE)):
            if x > XMAX: continue
            ax.axvline(x, color=c, lw=1.1, ls=":", zorder=1)
            ax.annotate(t, (x, 1.0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(-3, -3), rotation=90,
                        fontsize=9.6, color=c, ha="right", va="top")
        ax.axvline(g["rho"], color=style.BLUE, lw=2.4, alpha=.85, zorder=20)
        ax.annotate("$R_{\\mathrm{eff}}$ = %.2f" % g["rho"], (g["rho"] + 0.06, 0.93),
                    fontsize=10.5, color=style.BLUE, ha="left", weight="bold", zorder=21)
        ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.18)
        ax.set_title(labs[j], fontsize=13.5, pad=18)
        ax.set_xlabel("電子どうしの距離")
        if j == 0: ax.set_ylabel("アテンションの重み（ヘッドごとに最大 1）")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

        ax2 = fig.add_subplot(gs[1, j])
        ax2.fill_between(g["d"], 0, g["gd"], color="#e2e8ea", zorder=1)
        ax2.plot(g["d"], g["gd"], color=style.SUB, lw=1.3, zorder=2)
        for x, c in ((g["aM"], style.ENJI), (a_ee, style.ORANGE),
                     (np.sqrt(3) * g["aM"], style.PURPLE)):
            if x <= XMAX: ax2.axvline(x, color=c, lw=1.1, ls=":")
        ax2.axhline(1.0, color=style.LINE, lw=.8)
        ax2.set_xlim(0, XMAX); ax2.set_xlabel("電子どうしの距離")
        if j == 0: ax2.set_ylabel("動径分布 $g(d)$")
        for sp in ("top", "right"): ax2.spines[sp].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=nrm, cmap=cmap)
    cb = fig.colorbar(sm, ax=fig.axes, fraction=.016, pad=.015)
    cb.set_label("2 体の寄与の割合 [%]", fontsize=10)

    fig.text(0.10, 1 - 0.30 / Hh, "各ヘッドが見ている距離", fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("上段は 1 本が 1 ヘッド（実線 = 第1層、破線 = 第2層）で、色と太さがそのヘッドが運ぶ 2 体の寄与の割合。"
            "遮蔽窓の外では重みが厳密に 0 になるため、線が落ちきる位置がそのヘッドの窓。"
            "太い青線はその割合で重み付けした実効到達距離 $R_{\\mathrm{eff}}$。"
            "下段は同じ状態の動径分布で、どの配位殻までを見ているかの参照。")
    style.caption(fig, "図：ヘッドの到達距離を $a_M$・$a_{ee}$ と並べる。",
                  style.wrap(body, int(W * 11.2)), y=0.028)
    fig.savefig(a.out, dpi=165)
    print("->", a.out)
    for j, g in enumerate(G):
        print("  %-12s 実効到達距離 %.3f | a_M %.3f  √3a_M %.3f  電子間隔 %.3f"
              % (labs[j], g["rho"], g["aM"], np.sqrt(3) * g["aM"], a_ee))


if __name__ == "__main__":
    main()
