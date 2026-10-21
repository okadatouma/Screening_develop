"""q·k 行列を 2 つの状態で並べ、各ヘッドの窓と寄与率を添える。

    python screening_analysis/fig_qk.py <folderA> <folderB> [--labels "ε=10,ε=5"]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style                                         # noqa
import matplotlib.pyplot as plt                      # noqa


def grab(folder, nconf=48):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    soft = cfg["network"]["network_type"] == "CustomPsiformer"
    if soft:
        from psi_probe import probe
    else:
        from attn_probe import probe
    r = probe(folder, nconf=nconf)
    from head_contrib import cached as HC
    import moire_analysis as MA
    z = HC(folder, nconf=nconf); C, Cp, w = z["C"], z["Cp"], z["w"]
    SQ = 100 * Cp ** 2 / (Cp ** 2).sum()          # 2 体の寄与だけで見た割合
    we = (np.full_like(C, np.nan) if soft else
          MA.w_to_euclid(np.asarray(w) * r["dref"], np.asarray(r["lattice"])))
    nup = int((np.asarray(r["spins"][0]) > 0).sum())
    dEsh = np.full_like(SQ, np.nan)
    hp = os.path.join(folder, "head_ablation_pair.json")
    if os.path.exists(hp):
        import json as _j
        hd = _j.load(open(hp))["heads"]
        NL_, NH_ = SQ.shape
        d = np.array([max(x["dE"], 0.0) for x in hd]).reshape(NL_, NH_)
        dEsh = 100 * d / d.sum()
    return dict(S=r["S"].mean(0), w=we, SQ=SQ, dEsh=dEsh,
                nup=nup, NL=r["NL"], NH=r["NH"], soft=soft,
                rs=cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
                   ["interaction_energy_scale"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--nconf", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "fig_qk.png"))
    a = ap.parse_args()
    style.use()

    G = [grab(f, a.nconf) for f in a.folders]
    labs = a.labels.split(",") if a.labels else [r"$r_s=%g$" % g["rs"] for g in G]
    NH = G[0]["NH"]; NL = G[0]["NL"]
    rows = [(gi, l) for gi in range(len(G)) for l in range(NL)]
    v = float(max(np.percentile(np.abs(g["S"]), 99.2) for g in G))

    NR, NC = len(rows), NH
    W = 1.62 * NC + 1.6; Hh = 1.62 * NR + 2.9
    fig = plt.figure(figsize=(W, Hh))
    gs = fig.add_gridspec(NR, NC + 1, width_ratios=[1] * NC + [0.045],
                          left=0.085, right=0.935, top=1 - 0.95 / Hh,
                          bottom=2.15 / Hh, hspace=0.52, wspace=0.09)
    for ri, (gi, l) in enumerate(rows):
        g = G[gi]
        order = np.argsort(g["w"][l]) if not g["soft"] else np.arange(NH)
        for ci, h in enumerate(order):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(g["S"][l, h], cmap="RdBu_r", vmin=-v, vmax=v,
                           interpolation="nearest")
            ax.axhline(g["nup"] - .5, color="#1a1a1a", lw=.6)
            ax.axvline(g["nup"] - .5, color="#1a1a1a", lw=.6)
            ax.set_xticks([]); ax.set_yticks([])
            wt = "--" if g["soft"] else ("$w_h$=%.2f" % g["w"][l, h]
                                         if np.isfinite(g["w"][l, h]) else "$w_h$: 制限なし")
            sh = g["dEsh"][l, h]
            big = np.isfinite(sh) and sh >= 15
            ax.set_title("%s\n担当 %s" % (wt, ("%.0f%%" % sh if np.isfinite(sh) else "--")),
                         fontsize=8.6, color=(style.ENJI if big else style.SUB),
                         pad=3.5, linespacing=1.35,
                         fontweight=("bold" if big else "normal"))
            if big:
                for sp in ax.spines.values():
                    sp.set_color(style.ENJI); sp.set_linewidth(2.0)
            if ci == 0:
                ax.set_ylabel("%s\n第%d層" % (labs[gi], l + 1), fontsize=11,
                              color=style.INK, labelpad=9, linespacing=1.6)
            if ri == 0 and ci == 0:
                ax.text(0.25, 0.75, "↑↑", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10.5, color="#1a1a1a", weight="bold")
                ax.text(0.75, 0.25, "↓↓", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10.5, color="#1a1a1a", weight="bold")
                ax.text(0.75, 0.75, "↑↓", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10.5, color="#555", weight="bold")
    cax = fig.add_subplot(gs[:, NC]); plt.colorbar(im, cax=cax)
    cax.set_title(r"$s_{ij}$", fontsize=11, pad=8)

    fig.text(0.085, 1 - 0.26 / Hh, "全ヘッドの $q\\cdot k$ 行列（液体 と 1 谷 1 電子）", fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("各枠が 1 つのヘッド。中身は配置平均した内容の似かた "
            "$s_{ij}=\\hat q_i \\cdot \\hat k_j$ で、赤いほど「$i$ が $j$ を見たい」、"
            "青いほど「見たくない」。行が見る側 $i$、列が見られる側 $j$、"
            "黒線がスピンの境目（左上↑↑、右下↓↓、非対角が↑↓）。"
            "ヘッドは遮蔽窓 $w_h$ の狭い順に左から並べ、枠の上に $w_h$（普通の距離）と、"
            "そのヘッドを止めたときのエネルギーの代償 $\\Delta E_h$ の割合（＝担当）を書いた。"
            "担当 15% 以上の枠は赤枠で囲んである。")
    style.caption(fig, "図：谷液体では担当が窓の狭い側にも分かれているが、"
                  "1 谷 1 電子になると最も広い窓のヘッドに集中する。",
                  style.wrap(body, int(W * 11.2)), y=0.032)
    fig.savefig(a.out, dpi=165)
    print("->", a.out)


if __name__ == "__main__":
    main()
