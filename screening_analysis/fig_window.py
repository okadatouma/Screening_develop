"""遮蔽窓 w の分布が、何で動いて何で動かないか（結果 ①②をまとめた 1 枚）。

    python screening_analysis/fig_window.py --nu <f>,<f>,... --u <f>,<f>,...
        --qk <液体folder>,<結晶folder>

左：横軸が遮蔽窓 w そのもの。丸 1 つが 1 ヘッドで、面積がそのヘッドを止めた
ときのエネルギーの代償 ΔE_h の割合。モアレ周期 a_M を 2 倍に振っても分布は
動かないが、谷を深くして電子を結晶化させると長い窓へ一気に移る。
右：同じ 2 ヘッドの q·k 行列を液体側と結晶側で並べたもの。
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style, moire_summary as MS                       # noqa
import matplotlib.pyplot as plt                         # noqa


def heads(folder):
    p = os.path.join(folder, "head_ablation_pair.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    w = np.array([x["w"] for x in d["heads"]])
    dE = np.array([max(x["dE"], 0.0) for x in d["heads"]])
    lay = np.array([x["layer"] for x in d["heads"]])
    tot = dE.sum()
    o = np.argsort(w)
    med = float(np.interp(0.5, np.cumsum(dE[o]) / tot, w[o]))
    return dict(w=w, dE=dE, share=dE / tot, layer=lay, tot=tot, med=med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", required=True)
    ap.add_argument("--rs", default=None)
    ap.add_argument("--nu", default=None)
    ap.add_argument("--u-labels", default=None)
    ap.add_argument("--rs-labels", default=None)
    ap.add_argument("--nu-labels", default=None)
    ap.add_argument("--title", default="相転移でヘッドの順位が入れ替わる")
    ap.add_argument("--u-title", default="谷を深くする　谷液体 → 1 谷 1 電子")
    ap.add_argument("--cap-title", default=None)
    ap.add_argument("--cap-body", default=None)
    ap.add_argument("--qk", default=None, help="液体folder,結晶folder")
    ap.add_argument("--qk-heads", default="0:5,1:7", help="層:ヘッド を 2 つ")
    ap.add_argument("--out", default=os.path.join(HERE, "fig_window.png"))
    a = ap.parse_args()
    style.use()

    grp = []
    for key, fl, lb, ttl in (
            ("u", a.u, a.u_labels, a.u_title),
            ("rs", a.rs, a.rs_labels, "対照：相互作用 $r_s$ を振る"),
            ("nu", a.nu, a.nu_labels, "対照：モアレ周期 $a_M$ を振る")):
        if not fl: continue
        fs = [x for x in fl.split(",") if x]
        ls = (lb.split(",") if lb else [os.path.basename(x) for x in fs])
        grp.append((key, ttl, fs, ls))

    rows = []                      # (種類, (文字列, 注記) or (label, summary, heads))
    for gi, (key, ttl, fs, ls) in enumerate(grp):
        blk = []
        for f, l in zip(fs, ls):
            h = heads(f)
            if h is None: continue
            blk.append((l, MS.summarize(f), h))
        aee = blk[0][1]["a_ee"]
        fr = np.array([100 * h["share"][h["w"] < aee].sum() for _, _, h in blk])
        key = grp[gi][0]
        if key == "u":
            x = np.array([sm["contrast"] for _, sm, _ in blk])
            note = "担当 と 谷占有のばらつき の相関 $r = %.2f$" % np.corrcoef(fr, x)[0, 1]
        elif key == "rs":
            note = "$r_s$ を 3.4 倍振っても担当は %.0f〜%.0f%%" % (fr.min(), fr.max())
        else:
            x = np.array([sm["aM"] for _, sm, _ in blk])
            note = "担当 と $a_M$ の相関 $r = %+.2f$（無相関）" % np.corrcoef(fr, x)[0, 1]
        rows.append(("head", (ttl, note)))
        rows += [("row", b) for b in blk]
    a_ee = rows[1][1][1]["a_ee"]

    Wf = 15.0 if a.qk else 12.4
    Hh = 6.6 if a.qk else max(6.6, 2.7 + 0.60 * len(rows))
    fig = plt.figure(figsize=(Wf, Hh))
    wr = [1.62, 1] if a.qk else [1, 0.001]
    gs = fig.add_gridspec(1, 2, width_ratios=wr, left=0.118,
                          right=(0.972 if a.qk else 0.760),
                          top=1 - 0.98 / Hh, bottom=1.62 / Hh, wspace=0.34)

    # ================================================== 左：w 軸の上の重要度
    ax = fig.add_subplot(gs[0])
    ns = len(rows)
    ypos = [float(ns - 1 - k) for k in range(ns)]
    XMAX = 4.5
    ax.axvline(a_ee, color=style.ORANGE, lw=2.2, zorder=1)
    ticks, ticklabs = [], []
    for k, (kind, val) in enumerate(rows):
        yy = ypos[k]
        if kind == "head":
            ttl, note = val
            ax.text(-0.128, yy + 0.30, ttl, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=11.6, color=style.INK, weight="bold")
            ax.text(-0.128, yy - 0.30, note, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=10.4, color=style.SUB)
            if k > 0: ax.axhline(yy + 0.72, color=style.LINE, lw=.9)
            continue
        lab, s, h = val
        ticks.append(yy); ticklabs.append(lab)
        ax.plot([s["aM"]], [yy], "v", color=style.ENJI, ms=8, zorder=3)
        ax.scatter(h["w"], np.full_like(h["w"], yy), s=12 + 950 * h["share"],
                   color=style.BLUE, alpha=.55, lw=0, zorder=4)
        ax.plot([h["med"]], [yy], "|", color=style.INK, ms=20, mew=2.4, zorder=6)
        fr = 100 * h["share"][h["w"] < a_ee].sum()
        ax.text(1.035, yy, "%.0f%%" % fr, transform=ax.get_yaxis_transform(),
                fontsize=11.4, color=(style.BLUE if fr > 25 else style.ENJI),
                weight="bold", va="center", ha="center")
        ax.text(1.135, yy, "%.2f" % s["contrast"], transform=ax.get_yaxis_transform(),
                fontsize=10.8, color=style.SUB, va="center", ha="center")
    ax.text(1.035, ypos[0] + 0.30, "$w<a_{ee}$\nの担当", fontsize=10.4,
            transform=ax.get_yaxis_transform(),
            color=style.INK, va="center", ha="center", linespacing=1.35, weight="bold")
    ax.text(1.135, ypos[0] + 0.30, "谷占有の\nばらつき", fontsize=10.4,
            transform=ax.get_yaxis_transform(),
            color=style.SUB, va="center", ha="center", linespacing=1.35)
    ax.set_yticks(ticks); ax.set_yticklabels(ticklabs, fontsize=11)
    ax.set_ylim(-(2.15 if a.qk else 3.1), ns + 0.55); ax.set_xlim(0, XMAX)
    ax.set_xlabel("そのヘッドの遮蔽窓  $w_h$", fontsize=12.5)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=style.LINE, lw=.6, alpha=.5); ax.set_axisbelow(True)
    ax.annotate("$a_{ee}$", (a_ee, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-4, -5), rotation=90,
                color=style.ORANGE, fontsize=11.5, va="top", ha="right", weight="bold")
    h1 = [plt.Line2D([], [], ls="none", marker="o", color=style.BLUE, alpha=.55, ms=13,
                     label=r"1 ヘッド。面積 $=\Delta E_h$ の割合"),
          plt.Line2D([], [], ls="none", marker="|", color=style.INK, ms=17, mew=2.4,
                     label="中央値（$\\Delta E$ の半分がこれより内側）"),
          plt.Line2D([], [], ls="none", marker="v", color=style.ENJI, ms=8,
                     label="$a_M$（モアレ周期）"),
          plt.Line2D([], [], ls="none", marker="s", color=style.ORANGE, ms=9,
                     label="$a_{ee} = 1.905$（電子の平均間隔）")]
    ax.legend(handles=h1, fontsize=10.4, loc="lower left", ncol=2, handletextpad=.5,
              columnspacing=1.6, borderpad=.8, labelspacing=.65, frameon=True,
              facecolor="white", edgecolor=style.LINE, framealpha=1.0)
    ax.set_title("どの長さの窓が相関エネルギーを担っているか", fontsize=13, pad=9,
                 loc="left", color=style.INK)

    # ================================================== 右：q·k 行列（前後比較）
    if a.qk:
        import fig_qk as QK
        fa, fb = a.qk.split(",")
        G = [QK.grab(fa), QK.grab(fb)]
        HA = [heads(fa), heads(fb)]
        hh = [tuple(int(v) for v in t.split(":")) for t in a.qk_heads.split(",")]
        gq = gs[1].subgridspec(2, 3, width_ratios=[1, 1, 0.055], hspace=0.34,
                               wspace=0.10)
        v = float(max(np.percentile(np.abs(g["S"]), 99.2) for g in G))
        NH = G[0]["NH"]
        for r, (l, hd) in enumerate(hh):
            for c, (g, ha_) in enumerate(zip(G, HA)):
                axq = fig.add_subplot(gq[r, c])
                im = axq.imshow(g["S"][l, hd], cmap="RdBu_r", vmin=-v, vmax=v)
                nup = g["nup"]
                for z in (nup - 0.5,):
                    axq.axhline(z, color=style.INK, lw=.8); axq.axvline(z, color=style.INK, lw=.8)
                axq.set_xticks([]); axq.set_yticks([])
                sh = 100 * ha_["share"][l * NH + hd]
                axq.set_title("窓 $w_h$=%.2f　担当 %.0f%%" % (g["w"][l, hd], sh),
                              fontsize=11.2, pad=5,
                              color=(style.ENJI if sh > 30 else style.SUB),
                              weight=("bold" if sh > 30 else "normal"))
                if c == 0:
                    axq.set_ylabel(("短い窓のヘッド" if r == 0 else "長い窓のヘッド"),
                                   fontsize=11.5, labelpad=8)
                if r == 0:
                    axq.text(0.5, 1.32, ("谷液体　$U$=1.84" if c == 0 else "電子が谷に落ちた　$U$=12"),
                             transform=axq.transAxes, ha="center", va="bottom",
                             fontsize=12.5, weight="bold",
                             color=(style.SUB if c == 0 else style.ENJI))
        cax = fig.add_subplot(gq[:, 2])
        cb = fig.colorbar(im, cax=cax); cb.set_label("$s_{ij} = \\hat q_i\\cdot\\hat k_j$",
                                                     fontsize=10.5)

    # ================================================== 見出しとキャプション
    fig.text(0.024, 1 - 0.24 / Hh, a.title, fontsize=16,
             weight="bold", va="top", ha="left")
    body = ("左：丸 1 つが 1 ヘッド、横位置がその窓 $w_h$、面積がそのヘッドを止めたときの"
            "代償 $\\Delta E_h$ の割合。$U$ を上げて電子が 1 谷 1 個ずつに落ちると、"
            "$w<a_{ee}$ を担当していた短い窓のヘッドが仕事を失い、順位が長い窓側へ入れ替わる"
            "（$41\\% \\to 4\\%$）。谷占有のばらつきが跳ぶ場所と一致する。"
            "右：同じ 2 ヘッドの $q\\cdot k$ 行列。行が見る側、列が見られる側、黒線がスピンの境目。")
    ct = a.cap_title or "図：電子が谷に落ちると、短い窓のヘッドが担当を失い長い窓のヘッドが引き受ける。"
    if a.cap_body: body = a.cap_body
    style.caption(fig, ct, style.wrap(body, int(Wf * 12.6)), y=0.026)
    fig.savefig(a.out, dpi=160)
    print("->", a.out)
    for kind, val in rows:
        if kind != "row": continue
        lab, s, h = val
        print("  %-24s a_M=%.3f  中央値 w=%.2f  w<a_ee %3.0f%%  ばらつき %.3f  ΔE合計 %.1f"
              % (lab, s["aM"], h["med"], 100 * h["share"][h["w"] < a_ee].sum(),
                 s["contrast"], h["tot"]))


if __name__ == "__main__":
    main()
