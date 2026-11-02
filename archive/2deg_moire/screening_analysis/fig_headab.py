"""ヘッドを 1 本ずつ「自分自身しか見ない」状態に落としたときのエネルギーの代償。

    python screening_analysis/fig_headab.py <folder> [<folder> ...] --labels "..."
（先に head_ablation.py --mode pair を各 folder で走らせておく）
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_analysis as MA               # noqa
import matplotlib.pyplot as plt                  # noqa

BINS = [(0.0, 1.0, "1.0 未満"), (1.0, 2.0, "1.0–2.0"),
        (2.0, 3.0, "2.0–3.0"), (3.0, 99., "3.0 以上")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_headab.png"))
    a = ap.parse_args()
    style.use()
    D = [json.load(open(os.path.join(f.rstrip("/"), "head_ablation_pair.json")))
         for f in a.folders]
    M = [MA.load(f.rstrip("/")) for f in a.folders]
    labs = a.labels.split(",") if a.labels else [r"$r_s=%g$" % d["rs"] for d in M]
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))

    Wf, Hh = 12.4, 6.4
    fig = plt.figure(figsize=(Wf, Hh))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.075, right=0.985,
                          top=1 - 1.05 / Hh, bottom=2.05 / Hh, wspace=0.22)
    PAL = [style.BLUE, style.ENJI, style.GREEN, style.PURPLE, style.ORANGE]

    ax = fig.add_subplot(gs[0])
    for i, d in enumerate(D):
        w = np.array([h["w"] for h in d["heads"]])
        dE = np.array([h["dE"] for h in d["heads"]])
        tot = dE[dE > 0].sum()
        lay = np.array([h["layer"] for h in d["heads"]])
        o = np.argsort(w)
        ax.plot(w[o], 100 * np.maximum(dE[o], 1e-3) / tot, "-", color=PAL[i % len(PAL)],
                lw=1.6, alpha=.7, label="%s（合計 %.1f）" % (labs[i], tot))
        for L, mk in ((0, "o"), (1, "^")):
            m = lay == L
            ax.plot(w[m], 100 * np.maximum(dE[m], 1e-3) / tot, mk,
                    color=PAL[i % len(PAL)], ms=7, ls="none")
    ax.axvline(a_ee, color=style.ORANGE, ls=":", lw=1.4)
    aM = float(np.linalg.norm(M[0]["MO"][:, 0]))
    ax.axvline(aM, color=style.SUB, ls=":", lw=1.4)
    ax.set_yscale("log"); ax.set_ylim(0.35, 200); ax.set_xlim(0.3, 4.3)
    for xv, tv, cv in ((aM, "$a_M$", style.SUB), (a_ee, "$a_{ee}$", style.ORANGE)):
        ax.annotate(tv, (xv, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(-3, -3), rotation=90,
                    color=cv, fontsize=10, ha="right", va="top")
    ax.set_xlabel("そのヘッドの遮蔽窓  $w_h$")
    ax.set_ylabel("止めたときの代償  $\\Delta E_h / \\sum_h \\Delta E_h$  [%]")
    ax.legend(fontsize=9.6, loc="lower right", framealpha=0.92, frameon=True,
              edgecolor="none", facecolor="white")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    axb = fig.add_subplot(gs[1])
    nb, nd = len(BINS), len(D)
    xs = np.arange(nb)
    for i, d in enumerate(D):
        w = np.array([h["w"] for h in d["heads"]])
        dE = np.clip(np.array([h["dE"] for h in d["heads"]]), 0, None)
        tot = dE.sum()
        vals = [100 * dE[(w >= lo) & (w < hi)].sum() / tot for lo, hi, _ in BINS]
        axb.bar(xs + (i - (nd - 1) / 2) * 0.8 / nd, vals, width=0.8 / nd,
                color=PAL[i % len(PAL)], label=labs[i])
    axb.set_xticks(xs); axb.set_xticklabels([t for _, _, t in BINS], fontsize=10)
    axb.set_xlabel("遮蔽窓の幅"); axb.set_ylabel("代償の内訳 [%]")
    axb.legend(fontsize=9.8)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)

    fig.text(0.075, 1 - 0.28 / Hh, "どの長さのヘッドが効いているか", fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("ヘッドを 1 本だけ「自分自身しか見ない」状態に落とし（1 体成分は残す）、"
            "変分エネルギーがどれだけ悪くなるかを測った。抜けるのはそのヘッドが運んでいた"
            "相関だけなので、窓が広いヘッドほど有利に見えるという偏りがない。"
            "左は 1 点が 1 ヘッド（○ = 第1層、"
            "△ = 第2層）、右は窓の幅でまとめた内訳。合計はその状態で全ヘッドを止めたときの"
            "代償の和で、状態によって桁が違うため、割合に直して比べている。")
    style.caption(fig, "図：状態が変わると、相関を担う長さの分布が変わる。",
                  style.wrap(body, int(Wf * 12.4)), y=0.028)
    fig.savefig(a.out, dpi=170)
    print("->", a.out)
    for i, d in enumerate(D):
        w = np.array([h["w"] for h in d["heads"]])
        dE = np.clip(np.array([h["dE"] for h in d["heads"]]), 0, None)
        tot = dE.sum()
        print("  %-14s 合計 %7.2f  " % (labs[i], tot)
              + "  ".join("%s %4.1f%%" % (t, 100 * dE[(w >= lo) & (w < hi)].sum() / tot)
                          for lo, hi, t in BINS))


if __name__ == "__main__":
    main()
