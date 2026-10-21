"""窓の初期値を変えても同じ到達距離に落ち着くか（初期値依存性の検査）。

    python screening_analysis/fig_lambda.py <folder> [<folder> ...] --lams 0.5,0.7,1.0,1.4,2.0
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_summary as MS, moire_analysis as MA     # noqa
import matplotlib.pyplot as plt                             # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--lams", default=None, help="各 folder の λ をカンマ区切りで")
    ap.add_argument("--nconf", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "fig_lambda.png"))
    a = ap.parse_args()
    style.use()
    S = [MS.summarize(f.rstrip("/"), a.nconf, a.force) for f in a.folders]
    lam = ([float(v) for v in a.lams.split(",")] if a.lams else
           [float(d["tag"].split("_w")[-1].lstrip("w")) for d in S])
    o = np.argsort(lam); lam = np.array(lam)[o]; S = [S[i] for i in o]
    rho = np.array([d["rho"] for d in S])
    E = np.array([d["E"] for d in S]); Ee = np.array([d["Eerr"] for d in S])
    a_ee, aM = S[0]["a_ee"], S[0]["aM"]
    L = np.array(MA.load(S[0]["folder"])["L"])
    lad0 = MA.w_to_euclid(np.geomspace(0.50, 2.50, np.array(S[0]["w_e"]).shape[1]), L)

    Wf, Hh = 11.4, 6.6
    fig = plt.figure(figsize=(Wf, Hh))
    gs = fig.add_gridspec(1, 3, left=0.065, right=0.985, top=1 - 1.0 / Hh,
                          bottom=1.95 / Hh, wspace=0.30)

    ax = fig.add_subplot(gs[0])
    cm = plt.get_cmap("plasma")
    for i, (d, lm) in enumerate(zip(S, lam)):
        w = np.array(d["w_e"])
        ini = np.clip(lad0 * lm, 0, 6)
        for l in range(w.shape[0]):
            ax.plot(ini, np.clip(w[l], 0, 6), "o" if l == 0 else "^",
                    color=cm(i / max(len(S) - 1, 1)), ms=6, alpha=.9,
                    label=("λ=%g" % lm) if l == 0 else None)
    lim = [0, 5.2]
    ax.plot(lim, lim, "-", color=style.LINE, lw=1.2, zorder=0)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("窓の初期値"); ax.set_ylabel("学習後の窓")
    ax.legend(fontsize=8.8, ncol=2)
    ax.set_title("窓は初期値からどれだけ動くか", fontsize=12, pad=8)

    axb = fig.add_subplot(gs[1])
    axb.plot(lam, rho, "o-", color=style.BLUE, lw=2, ms=7)
    axb.axhline(aM, color=style.ENJI, ls=":", lw=1.6); axb.text(lam.max(), aM, " $a_M$",
                color=style.ENJI, va="center", fontsize=10)
    axb.axhline(np.sqrt(3) * aM, color=style.PURPLE, ls=":", lw=1.6)
    axb.text(lam.max(), np.sqrt(3) * aM, r" $\sqrt{3}a_M$", color=style.PURPLE,
             va="center", fontsize=10)
    axb.axhline(a_ee, color=style.ORANGE, ls=":", lw=1.6)
    axb.text(lam.max(), a_ee, " 電子間隔", color=style.ORANGE, va="center", fontsize=10)
    axb.set_xscale("log"); axb.set_xticks(lam); axb.set_xticklabels([("%g" % v) for v in lam])
    axb.set_xlabel("窓の梯子の倍率 λ"); axb.set_ylabel("実効到達距離")
    axb.set_title("初期値を変えても同じ距離に落ち着くか", fontsize=12, pad=8)

    axc = fig.add_subplot(gs[2])
    axc.errorbar(lam, E, yerr=Ee, fmt="o-", color=style.GREEN, lw=2, ms=7, capsize=3)
    axc.set_xscale("log"); axc.set_xticks(lam); axc.set_xticklabels([("%g" % v) for v in lam])
    axc.set_xlabel("窓の梯子の倍率 λ"); axc.set_ylabel("エネルギー")
    axc.set_title("どの初期値がよいか", fontsize=12, pad=8)
    for A in (ax, axb, axc):
        for sp in ("top", "right"): A.spines[sp].set_visible(False)

    fig.text(0.065, 1 - 0.28 / Hh, "窓の初期値依存性", fontsize=16, weight="bold",
             va="top", ha="left")
    body = ("同じ収束済みの波動関数から出発し、窓の梯子だけを λ 倍して同じ予算で再最適化した。"
            "左は各ヘッドの窓が初期値からどこへ動いたか（○ = 第1層、△ = 第2層、灰線は動かない場合）。"
            "中央は寄与率で重み付けした実効到達距離で、λ によらず同じ値に落ち着けば、"
            "その距離は初期化ではなく系が決めていることになる。右は同じ予算で得たエネルギー。")
    style.caption(fig, "図：窓が初期化ではなく系によって決まっているかの検査。",
                  style.wrap(body, int(Wf * 12.4)), y=0.028)
    fig.savefig(a.out, dpi=170)
    print("->", a.out)
    for d, lm, r in zip(S, lam, rho):
        print("  λ=%-4g 到達 %.3f  E = %.3f ± %.3f" % (lm, r, d["E"], d["Eerr"]))


if __name__ == "__main__":
    main()
