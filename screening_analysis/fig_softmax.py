"""遮蔽アテンションと通常 softmax で、ヘッドが「距離を見分けているか」を比べる。

    python screening_analysis/fig_softmax.py <screen_folder> <softmax_folder>
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, reach, moire_analysis as MA        # noqa
from head_contrib import cached as HC            # noqa
import matplotlib.pyplot as plt                  # noqa


def grab(folder, nconf=96):
    r = reach.compute(folder, nconf)
    z = HC(folder, nconf=nconf); Cp = z["Cp"]
    r["SQ"] = 100 * Cp ** 2 / (Cp ** 2).sum()
    r["aM"] = float(np.linalg.norm(MA.load(folder)["MO"][:, 0]))
    r["rho"] = float((r["SQ"] * r["dbar"]).sum() / r["SQ"].sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs=2)
    ap.add_argument("--labels", default="遮蔽アテンション,通常の softmax")
    ap.add_argument("--nconf", type=int, default=96)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_softmax.png"))
    a = ap.parse_args()
    style.use()
    G = [grab(f, a.nconf) for f in a.folders]
    labs = a.labels.split(",")
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))
    COL = [style.GREEN, style.PURPLE]

    Wf, Hh = 12.6, 6.8
    fig = plt.figure(figsize=(Wf, Hh))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], left=0.075, right=0.985,
                          top=1 - 1.05 / Hh, bottom=2.15 / Hh, hspace=0.46, wspace=0.20)

    XM = 3.9
    for i, g in enumerate(G):
        ax = fig.add_subplot(gs[i, 0])
        d = g["dbar"].ravel(); q = g["SQ"].ravel()
        ax.scatter(d, np.zeros_like(d) + 0.5, s=40 + 26 * np.sqrt(np.maximum(q, 0)),
                   color=COL[i], alpha=.75, zorder=3, edgecolor="white", lw=.8)
        lo, hi = d[d > 0].min(), d.max()
        ax.annotate("", (lo, 0.26), (hi, 0.26), arrowprops=dict(
            arrowstyle="<->", color=style.SUB, lw=1.2))
        ax.text((lo + hi) / 2, 0.13, "%.2f 〜 %.2f（%.1f 倍）" % (lo, hi, hi / lo),
                ha="center", fontsize=10, color=style.SUB,
                bbox=dict(fc="white", ec="none", pad=1.5))
        for x, t, c in ((g["aM"], "$a_M$", style.ENJI), (a_ee, "$a_{ee}$", style.ORANGE)):
            ax.axvline(x, color=c, ls=":", lw=1.3)
            if i == 0:
                ax.annotate(t, (x, 1.0), xycoords=("data", "axes fraction"),
                            textcoords="offset points", xytext=(-3, -3), rotation=90,
                            color=c, fontsize=10, ha="right", va="top")
        ax.set_xlim(0, XM); ax.set_ylim(0, 1.0); ax.set_yticks([])
        ax.set_title(labs[i], fontsize=13, color=COL[i], pad=8, loc="left")
        if i == 1: ax.set_xlabel("そのヘッドが重みを掛けている距離の平均")
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)

        axb = fig.add_subplot(gs[i, 1])
        ok = g["cnt"] >= 8
        for l in range(g["NL"]):
            for h in range(g["NH"]):
                p = np.where(ok, g["prof"][l, h], np.nan)
                if not np.isfinite(p).any() or np.nanmax(p) <= 0: continue
                qq = float(g["SQ"][l, h])
                axb.plot(g["d"], p / np.nanmax(p), lw=0.7 + 2.6 * np.sqrt(qq / 100),
                         color=COL[i], alpha=0.30 + 0.60 * min(qq / 25.0, 1.0),
                         ls="-" if l == 0 else (0, (4, 1.6)))
        for x, c in ((g["aM"], style.ENJI), (a_ee, style.ORANGE)):
            axb.axvline(x, color=c, ls=":", lw=1.3)
        axb.set_xlim(0, XM); axb.set_ylim(0, 1.05)
        if i == 0:
            hh = [plt.Line2D([], [], color=COL[i], lw=1.0, ls="-", label="第1層のヘッド"),
                  plt.Line2D([], [], color=COL[i], lw=1.0, ls=(0, (4, 1.6)),
                             label="第2層のヘッド"),
                  plt.Line2D([], [], color=COL[i], lw=3.2, ls="-",
                             label="太いほど 2 体の寄与が大きい"),
                  plt.Line2D([], [], color=style.ENJI, lw=1.3, ls=":", label="$a_M$"),
                  plt.Line2D([], [], color=style.ORANGE, lw=1.3, ls=":", label="$a_{ee}$")]
            axb.legend(handles=hh, fontsize=9.0, loc="upper right", frameon=True,
                       facecolor="white", edgecolor=style.LINE, framealpha=.95,
                       labelspacing=.45, borderpad=.6)
        axb.set_ylabel("重み（ヘッドごとに最大 1）", fontsize=10)
        if i == 1: axb.set_xlabel("電子どうしの距離")
        for sp in ("top", "right"): axb.spines[sp].set_visible(False)

    fig.text(0.075, 1 - 0.28 / Hh, "ヘッドは距離を見分けているか", fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("左は 1 点が 1 ヘッドで、そのヘッドが重みを掛けている距離の平均。点の大きさは"
            "そのヘッドが運ぶ 2 体の寄与の割合。右は同じヘッドの重みを距離の関数として描いた"
            "もの。遮蔽では窓の外で重みが厳密に 0 になるため、ヘッドごとに見る距離が分かれる。"
            "softmax は分母で規格化されるので窓が無く、どのヘッドもモアレのスーパーセル内の対の平均距離を"
            "見ており、長さで分けることができない。")
    style.caption(fig, "図：長さで分けて読めるのは遮蔽アテンションだけ。",
                  style.wrap(body, int(Wf * 12.4)), y=0.028)
    fig.savefig(a.out, dpi=170)
    print("->", a.out)
    for i, g in enumerate(G):
        d = g["dbar"].ravel(); d = d[d > 0]
        print("  %-22s 到達距離 %.2f〜%.2f（%.1f 倍、sd/平均 %.3f）  寄与で重み付け %.3f"
              % (labs[i], d.min(), d.max(), d.max() / d.min(), d.std() / d.mean(), g["rho"]))


if __name__ == "__main__":
    main()
