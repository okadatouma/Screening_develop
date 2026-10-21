"""主図：秩序の立ち上がりとヘッドの分担の入れ替わりを同じ軸で見る。

    python screening_analysis/fig_order.py <folder> [<folder> ...] [--x rs|aM]
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_summary as MS                   # noqa
import matplotlib.pyplot as plt                     # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--x", default="rs", choices=("rs", "aM"))
    ap.add_argument("--nconf", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "fig_order.png"))
    a = ap.parse_args()
    style.use()

    S = [MS.summarize(f.rstrip("/"), a.nconf, a.force) for f in a.folders]
    S.sort(key=lambda d: d[a.x])
    x = np.array([d[a.x] for d in S])
    xlab = ("相互作用の強さ  $r_s$" if a.x == "rs" else "モアレ周期  $a_M$")
    SK = np.array([d["psi_K"] for d in S])
    ct = np.array([d["psi_sub"] for d in S])
    rho = np.array([d["rho"] for d in S])
    a_ee = S[0]["a_ee"]
    W = np.array([d["w_e"] for d in S])              # (n, L, H) Euclid の窓
    Q = np.array([d["SQ"] for d in S]) * 100         # (n, L, H) 寄与率 [%]
    n, NL, NH = Q.shape
    # ヘッドを窓の広さ順に並べ替えて層をまとめる（同じ段のヘッドは同じ窓で始まる）
    idx = np.argsort(np.nanmean(np.where(np.isfinite(W[0]), W[0], 1e9), axis=0))
    Qs = Q[:, :, idx].sum(1)                         # (n, H) 層を合算
    wref = np.nanmean(np.where(np.isfinite(W), W, np.nan), axis=(0, 1))[idx]

    Wf, Hh = 7.6, 10.2
    fig = plt.figure(figsize=(Wf, Hh))
    gs = fig.add_gridspec(3, 1, left=0.115, right=0.895, top=1 - 1.0 / Hh,
                          bottom=2.0 / Hh, hspace=0.16, height_ratios=[1, 1.25, 1])

    # ---- (a) 秩序 --------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.plot(x, SK, "o-", color=style.ENJI, lw=2.0, ms=6.5,
            label="構造因子から（向きに依らない）")
    ax.plot(x, ct, "s--", color=style.PURPLE, lw=1.6, ms=5.5,
            label="谷の占有から（向きを選んだときだけ立つ）")
    ax.axhline(0, color=style.LINE, lw=.9)
    ax.set_ylabel("ハニカム秩序（完全秩序 = 1）")
    ax.set_ylim(-0.08, 1.05)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=9.4, loc="upper left")
    ax.set_xticklabels([])

    # ---- (b) ヘッドの分担 -------------------------------------------------
    axb = fig.add_subplot(gs[1])
    cmap = plt.get_cmap("Spectral_r")
    base = np.zeros(n)
    for k in range(NH):
        lab = ("窓 %.2f" % wref[k]) if np.isfinite(wref[k]) else "窓 なし"
        axb.fill_between(x, base, base + Qs[:, k], color=cmap(k / max(NH - 1, 1)),
                         lw=.6, ec="white", label=lab)
        base = base + Qs[:, k]
    axb.set_ylabel("波動関数への寄与率 [%]")
    axb.set_ylim(0, 100); axb.set_xlim(x.min(), x.max())
    axb.legend(fontsize=8.4, ncol=2, loc="center left", bbox_to_anchor=(1.005, .5),
               title="ヘッドの窓", title_fontsize=8.8)
    axb.set_xticklabels([])
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)

    # ---- (c) 実効到達距離 -------------------------------------------------
    axc = fig.add_subplot(gs[2])
    axc.plot(x, rho, "o-", color=style.BLUE, lw=2.0, ms=6.5, label="実効到達距離")
    aM = np.array([d["aM"] for d in S])
    axc.plot(x, aM, ":", color=style.ENJI, lw=1.6, label="$a_M$")
    axc.plot(x, np.sqrt(3) * aM, ":", color=style.PURPLE, lw=1.6, label=r"$\sqrt{3}\,a_M$")
    axc.axhline(a_ee, color=style.ORANGE, ls=":", lw=1.6, label="電子の平均間隔")
    axc.set_xlabel(xlab); axc.set_ylabel("距離")
    axc.legend(fontsize=9.2, ncol=2)
    for sp in ("top", "right"): axc.spines[sp].set_visible(False)
    for A in (ax, axb, axc): A.set_xlim(x.min(), x.max())

    fig.text(0.115, 1 - 0.28 / Hh, "秩序の立ち上がりとヘッドの分担", fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("上：3 つの谷のうち 2 つだけが埋まるハニカム秩序の強さ（完全な秩序で 1、"
            "液体で 0 になるよう規格化）。中：各ヘッドが運ぶ 2 体の寄与の割合を、"
            "窓の狭い順に積み上げたもの（全層・全ヘッドで 100%）。"
            "下：その寄与で重み付けした実効到達距離を、モアレ周期・電荷秩序の周期・"
            "電子の平均間隔と比べたもの。")
    style.caption(fig, "図：秩序が立つのと同じところで、寄与を担うヘッドが入れ替わる。",
                  style.wrap(body, int(Wf * 12.8)), y=0.028)
    fig.savefig(a.out, dpi=170)
    print("->", a.out)


if __name__ == "__main__":
    main()
