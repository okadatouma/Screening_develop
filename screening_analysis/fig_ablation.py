"""波動関数が必要としている距離：アテンションの届く範囲を縮めたときのエネルギーの代償。

    python screening_analysis/fig_ablation.py <folder> [<folder> ...]
（先に window_ablation.py を各 folder で走らせておく）
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_analysis as MA                # noqa
import matplotlib.pyplot as plt                   # noqa


def rstar(pts, E0, frac=0.10):
    """代償が「一番絞ったときの代償」の frac 以下に落ちる最小の R（線形補間）。

    全エネルギーに対する割合で測ると、モアレの深い系では分母が大きくなって
    基準がきつくなり、系どうしを比べられない。ここでは各系自身の最大の代償を
    ものさしにするので、系のエネルギースケールに依らない。
    """
    R = np.array([p["R"] for p in pts]); dE = np.array([p["dE"] for p in pts])
    th = frac * float(np.max(dE))
    o = np.argsort(R); R, dE = R[o], dE[o]
    for i in range(len(R) - 1):
        if dE[i] > th >= dE[i + 1]:
            t = (dE[i] - th) / (dE[i] - dE[i + 1])
            return float(R[i] + t * (R[i + 1] - R[i]))
    return float(R[-1]) if dE[-1] <= th else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_ablation.png"))
    a = ap.parse_args()
    style.use()
    D = [json.load(open(os.path.join(f.rstrip("/"), "window_ablation.json")))
         for f in a.folders]
    M = [MA.load(f.rstrip("/")) for f in a.folders]
    labs = (a.labels.split(",") if a.labels else
            [r"$r_s=%g$" % d["rs"] for d in M])
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))

    Wf, Hh = 8.2, 6.6
    fig = plt.figure(figsize=(Wf, Hh))
    ax = fig.add_axes([0.115, 2.38 / Hh, 0.845, 1 - 3.43 / Hh])
    PAL = [style.BLUE, style.ENJI, style.GREEN, style.PURPLE, style.ORANGE, "#7a5c2e"]
    cm = lambda t: PAL[int(round(t * (len(D) - 1))) % len(PAL)]        # noqa: E731
    for i, (d, m) in enumerate(zip(D, M)):
        R = np.array([p["R"] for p in d["points"]])
        dE = np.array([p["dE"] for p in d["points"]])
        er = np.array([p["Eerr"] for p in d["points"]])
        c = PAL[i % len(PAL)]
        rs_ = rstar(d["points"], d["E0"], a.frac)
        lab = labs[i] + ("　$R^{*}$=%.2f" % rs_ if rs_ == rs_ else "")
        ax.errorbar(R, 100 * dE / abs(d["E0"]), yerr=100 * er / abs(d["E0"]),
                    fmt="o-", color=c, lw=2, ms=6, capsize=3, label=lab)
        if rs_ == rs_:
            ax.axvline(rs_, color=c, ls="--", lw=1.3, alpha=.7)
    aM = float(np.linalg.norm(M[0]["MO"][:, 0]))
    has_moire = all(m["U"] > 0 for m in M)
    if has_moire and len(set(round(float(np.linalg.norm(m["MO"][:, 0])), 4)
                             for m in M)) == 1:
        for x, t, c in ((aM, "$a_M$", style.ENJI),
                        (np.sqrt(3) * aM, r"$\sqrt{3}a_M$", style.PURPLE),
                        (a_ee, "$a_{ee}$", style.ORANGE)):
            ax.axvline(x, color=c, ls=":", lw=1.3)
            ax.annotate(t, (x, 1.0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(-3, -4), rotation=90,
                        color=c, fontsize=10, ha="right", va="top")
    ax.set_yscale("log"); ax.set_ylim(0.25, None)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
    ax.set_xticklabels(["0.5", "0.7", "1.0", "1.4", "2.0", "2.8", "4.0"])
    ax.set_xlabel("アテンションが届く範囲  $R$")
    ax.set_ylabel("エネルギーの代償  $\\Delta E/|E|$  [%]")
    lg1 = ax.legend(fontsize=10, loc="lower left", title="系（実線＋○ = $\\Delta E(R)$）")
    lg1.get_title().set_fontsize(9.6)
    ax.add_artist(lg1)
    hh = [plt.Line2D([], [], color=style.SUB, ls="--", lw=1.4,
                     label="縦の破線：その系の $R^{*}$"),
          plt.Line2D([], [], color=style.SUB, ls=":", lw=1.4,
                     label="縦の点線：$a_M$・$a_{ee}$・$\\sqrt{3}a_M$")]
    ax.legend(handles=hh, fontsize=9.6, loc="upper right",
              bbox_to_anchor=(1.0, 0.90), frameon=True, facecolor="white",
              edgecolor=style.LINE, framealpha=.95, borderpad=.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.text(0.115, 1 - 0.28 / Hh, "波動関数が必要としている距離", fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("縦の破線が各系の $R^{*}$、縦の点線が $a_M$・$a_{ee}$・$\\sqrt{3}a_M$ の位置。"
            "縦軸は $\\Delta E/|E|$ だが、$R^{*}$ の基準はこの割合ではなく"
            "各系自身の最大の代償の 1 割にしてある。モアレが深い系ほど $|E|$ が大きく、"
            "割合のままでは系どうしを比べられないため。")
    style.caption(fig, "図：どの系も $a_{ee}$ の手前で代償が立ち上がる。",
                  style.wrap(body, int(Wf * 12.4)), y=0.028)
    fig.savefig(a.out, dpi=170)
    print("->", a.out)
    for d, l in zip(D, labs):
        print("  %-12s R* = %.3f  (E0 = %.3f)" % (l, rstar(d["points"], d["E0"], a.frac), d["E0"]))


if __name__ == "__main__":
    main()
