"""波動関数が必要とする距離は何で決まるか。

    python screening_analysis/fig_length.py <folder> [...] --groups '見出し:ラベル,..|..'

測った距離（窓を人為的に絞って測る R*、および寄与加重の到達距離 R_eff）を、
候補の長さ（電子の平均間隔 a_ee、モアレ周期 a_M、ハニカム秩序の周期 √3 a_M、
Thomas-Fermi 遮蔽長 λ_TF）と同じ長さ軸の上に並べて直接見比べる。
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_summary as MS                 # noqa
from fig_ablation import rstar                    # noqa
import matplotlib.pyplot as plt                   # noqa

RS = r"$R^{*}$"
RE = r"$R_{\mathrm{eff}}$"


def load(folder, frac=0.10):
    s = MS.summarize(folder.rstrip("/"))
    p = os.path.join(folder.rstrip("/"), "window_ablation.json")
    s["rstar"] = float("nan")
    if os.path.exists(p):
        d = json.load(open(p))
        s["rstar"] = float(rstar(d["points"], d["E0"], frac))
    s["lam_tf"] = 1.0 / (2.0 * s["rs"])           # 2D の TF 遮蔽長 a_B*/2（コード単位）
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_length.png"))
    a = ap.parse_args()
    style.use()

    S = [load(f, a.frac) for f in a.folders]
    a_ee = S[0]["a_ee"]

    if a.groups:
        rows, gsep = [], []
        for blk in a.groups.split("|"):
            head, labs = blk.split(":", 1)
            gsep.append((head, len(rows), len(labs.split(","))))
            rows += labs.split(",")
    else:
        rows = [s["tag"] for s in S]
        gsep = [("", 0, len(rows))]
    assert len(rows) == len(S), (len(rows), len(S))

    Wf, Hh = 12.6, 6.8
    fig = plt.figure(figsize=(Wf, Hh))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1], left=0.150, right=0.978,
                          top=1 - 1.16 / Hh, bottom=1.86 / Hh, wspace=0.26)

    # ---------------------------------------- 左：長さ軸の上で直接比べる
    slots = []
    for head, i0, cnt in gsep:
        if head: slots.append(("head", head))
        for j in range(i0, i0 + cnt): slots.append(("row", j))
    ns = len(slots)
    ypos = {k: float(ns - 1 - k) for k in range(ns)}

    ax = fig.add_subplot(gs[0])
    XMAX = 5.2
    rr = np.array([x["rstar"] for x in S if x["rstar"] == x["rstar"]])
    ax.axvspan(rr.min(), rr.max(), color=style.BLUE, alpha=.10, lw=0, zorder=0)
    ax.axvline(a_ee, color=style.ORANGE, lw=2.4, zorder=1)

    ticks, ticklabs = [], []
    for k, sl in enumerate(slots):
        yy = ypos[k]
        if sl[0] == "head":
            ax.text(-0.155, yy, sl[1], transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=11.8, color=style.SUB, weight="bold")
            if k > 0:
                ax.axhline(yy + 0.52, color=style.LINE, lw=.9)
            continue
        j = sl[1]; s_ = S[j]
        ticks.append(yy); ticklabs.append(rows[j])
        ax.plot([s_["lam_tf"]], [yy], "|", color=style.GREEN, ms=13, mew=2.4, zorder=5)
        ax.plot([s_["aM"]], [yy], "v", color=style.ENJI, ms=9, zorder=5)
        ax.plot([np.sqrt(3) * s_["aM"]], [yy], "v", color=style.PURPLE, ms=9, zorder=5)
        if s_["rstar"] == s_["rstar"]:
            ax.plot([s_["rstar"]], [yy], "o", color=style.BLUE, ms=13, zorder=7,
                    markeredgecolor="white", markeredgewidth=1.3)

    ax.set_yticks(ticks); ax.set_yticklabels(ticklabs, fontsize=11)
    ax.set_ylim(-6.3, ns + 0.35)
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("長さ（単位 $r_s a_B^{*}$）", fontsize=12)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=style.LINE, lw=.6, alpha=.55); ax.set_axisbelow(True)
    ax.annotate("$a_{ee} = 1.905$", (a_ee, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-4, -6), rotation=90,
                color=style.ORANGE, fontsize=11.5, va="top", ha="right", weight="bold")
    ax.text(rr.mean(), 1.018, "測った " + RS + " はすべてこの帯の中",
            transform=ax.get_xaxis_transform(), color=style.BLUE,
            fontsize=11.8, va="bottom", ha="center", weight="bold")

    h = [plt.Line2D([], [], ls="none", marker="o", color=style.BLUE, ms=13,
                    label=RS + "　窓を絞って測った、波動関数が必要とする距離"),
         plt.Line2D([], [], ls="none", marker="s", color=style.ORANGE, ms=9,
                    label="$a_{ee}$　電子の平均間隔"),
         plt.Line2D([], [], ls="none", marker="v", color=style.ENJI, ms=9,
                    label="$a_M$　モアレ周期"),
         plt.Line2D([], [], ls="none", marker="v", color=style.PURPLE, ms=9,
                    label=r"$\sqrt{3}\,a_M$　ハニカム秩序の周期"),
         plt.Line2D([], [], ls="none", marker="|", color=style.GREEN, ms=13, mew=2.4,
                    label=r"$\lambda_{\rm TF} = 1/2r_s$　Coulomb 遮蔽長")]
    lg = ax.legend(handles=h, fontsize=10.6, loc="lower left", ncol=2,
                   handletextpad=.5, columnspacing=2.0, borderpad=.8, labelspacing=.6,
                   frameon=True, facecolor="white", edgecolor=style.LINE, framealpha=1.0)

    # ---------------------------------------- 右：どの長さで割ると 1 に揃うか
    axb = fig.add_subplot(gs[1])
    ok = [s for s in S if s["rstar"] == s["rstar"]]
    cands = [("$a_{ee}$", np.array([s["rstar"] / a_ee for s in ok]), style.ORANGE),
             ("$a_M$", np.array([s["rstar"] / s["aM"] for s in ok]), style.ENJI),
             (r"$\sqrt{3}a_M$", np.array([s["rstar"] / (np.sqrt(3) * s["aM"])
                                          for s in ok]), style.PURPLE),
             (r"$\lambda_{\rm TF}$", np.array([s["rstar"] / s["lam_tf"] for s in ok]),
              style.GREEN)]
    axb.axhline(1.0, color=style.INK, lw=1.1, zorder=1)
    rng = np.random.default_rng(0)
    for i, (nm, v, c) in enumerate(cands):
        axb.plot(i + 0.16 * rng.standard_normal(len(v)), v, "o", color=c, ms=8,
                 alpha=.75, mew=0, zorder=3)
        axb.text(i, v.max() * 1.45, "±%.0f%%" % (100 * v.std() / v.mean()),
                 ha="center", fontsize=11, color=c, weight="bold")
    axb.set_yscale("log")
    axb.set_ylim(0.28, 260)
    axb.set_yticks([0.3, 1, 3, 10, 30, 100])
    axb.set_yticklabels(["0.3", "1", "3", "10", "30", "100"])
    axb.set_xticks(range(len(cands)))
    axb.set_xticklabels([c[0] for c in cands], fontsize=13)
    axb.set_xlim(-0.6, len(cands) - 0.4)
    axb.set_ylabel(RS + " / 候補の長さ", fontsize=12)
    axb.set_title("どれで割ると 1 に揃うか", fontsize=12.5, pad=8, loc="left",
                  color=style.INK)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)
    axb.grid(axis="y", color=style.LINE, lw=.6, alpha=.55); axb.set_axisbelow(True)

    # ---------------------------------------- 見出しとキャプション
    fig.text(0.030, 1 - 0.24 / Hh, "波動関数が必要とする距離は何で決まるか", fontsize=15.5,
             weight="bold", va="top", ha="left")
    r = np.array([s["rstar"] for s in ok])
    body = ("左：測った距離と候補の長さを同じ軸に並べたもの。$a_M$ を 2 倍、$r_s$ を 15 倍、"
            "$U$ を 6.5 倍に振っても " + RS + " は $a_{ee}$ の周りから出ない。"
            "右：" + RS + " を各候補で割った値。$a_{ee}$ で割ったときだけ 1 の近くに"
            "±%.0f%% で揃う。$a_M$ は ±%.0f%% で、しかも $a_M$ が伸びるほど比が下がる"
            "（追随していれば一定になるはず）。$\\lambda_{\\rm TF}$ は 2 桁違う。"
            % (100 * (r / a_ee).std() / (r / a_ee).mean(),
               100 * np.std([s["rstar"] / s["aM"] for s in ok])
               / np.mean([s["rstar"] / s["aM"] for s in ok])))
    style.caption(fig, "図：" + RS + " $= %.2f \\pm %.2f\\ a_{ee}$。$a_M$ にも "
                  "$\\lambda_{\\rm TF}$ にも追随しない。"
                  % ((r / a_ee).mean(), (r / a_ee).std()),
                  style.wrap(body, int(Wf * 13.2)), y=0.026)
    fig.savefig(a.out, dpi=165)
    print("->", a.out)
    print("  %-16s %6s %6s %6s | %7s %7s %7s %7s"
          % ("run", "rs", "U", "aM", "R*", "R*/aee", "R*/aM", "R_eff"))
    for lab, s in zip(rows, S):
        print("  %-16s %6.2f %6.2f %6.3f | %7.3f %7.3f %7.3f %7.3f"
              % (lab, s["rs"], s["U"], s["aM"], s["rstar"], s["rstar"] / a_ee,
                 s["rstar"] / s["aM"], s["rho"]))
    for nm, v, _ in cands:
        print("  %-12s 平均 %8.3f  sd/平均 %5.1f%%" % (nm, v.mean(), 100*v.std()/v.mean()))


if __name__ == "__main__":
    main()
