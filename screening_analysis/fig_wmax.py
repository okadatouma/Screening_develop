# -*- coding: utf-8 -*-
"""窓の上限 w_max を変えると読み出しがどう変わるかを、2 本の梯子で並べて比べる。

    python screening_analysis/fig_wmax.py \
        --a "液体=DIR,境界=DIR,結晶=DIR" --alab "梯子 A" \
        --b "液体=DIR,境界=DIR,結晶=DIR" --blab "梯子 B" --out fig_wmax.png

周期ノルムがセル内で取り得る最大値を d_ref と書く。w_h > d_ref の段は、どの対も
必ず窓に入るので「距離で選ぶ」働きを失う。梯子をそこより上へ伸ばすと、寄与が
その段へ移ってしまい、相のちがいを読む分解能が落ちる。それを数値で示す図。
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from fig_qwin import load, _font                                   # noqa

COL = ["#2a78d6", "#c98a12", "#c0392b"]
MRK = ["o", "s", "^"]


def _read(spec, nconf):
    out = []
    for part in spec.split(","):
        lab, folder = part.rsplit("=", 1)
        print("=== %s  %s" % (lab, folder))
        out.append((lab, load(folder, nconf)))
    return out


def _stats(items):
    """(TV 液体→結晶, 参加ヘッド数のリスト, <w>_q のリスト, d_ref 超の寄与のリスト)"""
    p0 = items[0][1]["qp"] / 100.0
    tv = [0.5 * np.abs(d["qp"] / 100.0 - p0).sum() for _, d in items]
    par = [float(1.0 / ((d["qp"] / 100.0) ** 2).sum()) for _, d in items]
    wq = [float((d["qp"] * d["w"]).sum() / d["qp"].sum()) for _, d in items]
    dr = items[0][1]["dref"]
    ov = [float(d["qp"][d["w"] > dr].sum()) for _, d in items]
    return tv, par, wq, ov, dr


def _panel(ax, items, title, ja):
    T = lambda x, y: x if ja else y                                # noqa: E731
    w = items[0][1]["w"]; dr = items[0][1]["dref"]
    cover = w[-1] > dr * 1.001
    if cover:
        ax.axvspan(dr, w[-1] * 1.05, color="#b0392b", alpha=.07, zorder=0)
        ax.axvline(dr, color="#a5262f", lw=1.3, ls=(0, (4, 2.5)), alpha=.9, zorder=2)
    else:
        # 梯子が d_ref のちょうど手前で止まっている場合も、その位置は示しておく
        ax.axvline(dr, color="#a5262f", lw=1.3, ls=(0, (4, 2.5)), alpha=.55, zorder=2)
    for i, (lab, d) in enumerate(items):
        ax.plot(d["w"], d["qp"], "-", marker=MRK[i], ms=5.4, lw=1.9,
                color=COL[i], label=lab, zorder=3)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xlim(1.85, 5.25)
    tk = [2.0, 2.5, 3.0, 4.0, 5.0]
    ax.set_xticks(tk); ax.set_xticklabels(["%.1f" % t for t in tk], fontsize=9.4)
    ax.grid(alpha=.22, lw=.7)
    ax.set_title(title, fontsize=11.2, pad=8)
    ax.set_xlabel(T("固定した窓 $w_h$（周期ノルム, $r_s a_B^{*}$）", "$w_h$"), fontsize=10.2)
    return cover, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--alab", default="A"); ap.add_argument("--blab", default="B")
    ap.add_argument("--out", default="fig_wmax.png")
    ap.add_argument("--nconf", type=int, default=48)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ja = _font()
    T = lambda x, y: x if ja else y                                # noqa: E731

    A = _read(a.a, a.nconf); B = _read(a.b, a.nconf)
    tvA, parA, wqA, ovA, drA = _stats(A)
    tvB, parB, wqB, ovB, drB = _stats(B)

    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.30,
                          left=0.058, right=0.945, top=0.800, bottom=0.315)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2])

    ttl = "① 梯子を %.2f まで（%s）" % (A[0][1]["w"][-1], a.alab)
    cvA, _ = _panel(ax1, A, T(ttl, "① ladder to %.2f" % A[0][1]["w"][-1]), ja)
    ttl2 = "② 梯子を %.2f まで（%s）" % (B[0][1]["w"][-1], a.blab)
    cov, dr = _panel(ax2, B, T(ttl2, "② ladder to %.2f" % B[0][1]["w"][-1]), ja)
    ax1.set_ylabel(T("波動関数への寄与率  $q_h$ [%]", "contribution share [%]"), fontsize=10.4)
    ax1.legend(fontsize=9.8, frameon=False, loc="upper left")
    ymax = max(max(d["qp"].max() for _, d in A), max(d["qp"].max() for _, d in B)) * 1.14
    ax1.set_ylim(0, ymax)
    plt.setp(ax2.get_yticklabels(), visible=False)
    if not cvA:
        ax1.text(0.975, 0.965,
                 T("梯子はちょうど $d_{\\rm ref}$ = %.2f で止めてある\n"
                   "（全ての段が長さを選んでいる）" % drA,
                   "ladder stops at $d_{ref}$"),
                 transform=ax1.transAxes, color="#a5262f", fontsize=8.8, va="top",
                 ha="right", linespacing=1.6, zorder=4,
                 bbox=dict(fc="white", ec="none", alpha=.82, pad=2.0))
    if cov:
        ax2.text(dr * 0.985, ymax * 0.03, T("$d_{\\rm ref}$ = %.2f " % dr, "$d_{ref}$ "),
                 color="#a5262f", fontsize=8.8, va="bottom", ha="right", rotation=90,
                 zorder=4, bbox=dict(fc="white", ec="none", alpha=.85, pad=1.0))
        ax2.text(0.975, 0.965,
                 T("赤い帯：窓がセル全体を覆う\n（全ての対が入り、長さを選べない）",
                   "window covers the whole cell"),
                 transform=ax2.transAxes, color="#a5262f", fontsize=8.8, va="top",
                 ha="right", linespacing=1.6, zorder=4,
                 bbox=dict(fc="white", ec="none", alpha=.82, pad=2.0))

    # --- ③ 読み出しの鋭さ -------------------------------------------------
    xs = np.arange(len(A))
    ax3.plot(xs, parA, "-o", color="#12406f", lw=2.2, ms=8,
             label=T("梯子 %.2f まで" % A[0][1]["w"][-1], "to %.2f" % A[0][1]["w"][-1]))
    ax3.plot(xs, parB, "--s", color="#a5262f", lw=2.0, ms=7.5, mfc="white",
             label=T("梯子 %.2f まで" % B[0][1]["w"][-1], "to %.2f" % B[0][1]["w"][-1]))
    for v, y in ((parA, 8), (parB, -14)):
        for i in xs:
            ax3.annotate("%.1f" % v[i], (xs[i], v[i]), textcoords="offset points",
                         xytext=(0, y), ha="center", fontsize=9.0,
                         color="#12406f" if y > 0 else "#a5262f")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["%s\n$r_s$=%g" % (lab, d["rs"]) for lab, d in A], fontsize=9.8)
    ax3.set_ylabel(T("使うヘッドの数  $1/\\sum_h q_h^2$", "participation ratio"),
                   fontsize=10.4)
    ax3.grid(alpha=.22, lw=.7)
    _y = ax3.get_ylim(); ax3.set_ylim(_y[0] - .55, _y[1] + 1.5)
    ax3.legend(fontsize=9.2, frameon=True, framealpha=.9, edgecolor="none",
               loc="lower left", borderpad=.35)
    ax3.set_title(T("③ 相のちがいが読めているか", "③ contrast"), fontsize=11.2, pad=8)

    fig.text(0.006, 0.985,
             T("窓の上限を伸ばすと、読み出しは鋭くなるか", "does a wider ladder help?"),
             fontsize=13.2, weight="bold", color="#0f7b6c", va="top")
    d0 = A[0][1]
    fig.text(0.006, 0.930,
             T("クーロンのみ, $N=%d$, %d 層 × %d ヘッド, 窓は学習させず固定 — "
               "下限 %.2f は共通で、上限だけを %.2f → %.2f に変えた"
               % (d0["ne"], d0["NL"], d0["NH"], d0["w"][0],
                  A[0][1]["w"][-1], B[0][1]["w"][-1]),
               "Coulomb only, N=%d" % d0["ne"]),
             fontsize=10.5, color="#0b0b0b", va="top")

    dwA = 100 * (wqA[-1] / wqA[0] - 1); dwB = 100 * (wqB[-1] / wqB[0] - 1)
    lines = [
        "$d_{\\rm ref}$ = %.2f は周期ノルムがセル内で取り得る最大値。$w_h > d_{\\rm ref}$ の窓はどの対も必ず含むので、長さを区別できない。" % drA,
        "②の梯子では上の %d 段がこれに当たり、そこへ寄与の %.0f〜%.0f%% が移る（結晶で %.0f%%）。残りの %.0f%% だけが長さを分解している。"
        % (int((B[0][1]["w"] > drA).sum()), min(ovB), max(ovB), ovB[-1], 100 - ovB[-1]),
        "③ 結果として、相のちがいが鈍る：使うヘッドの数の落ち幅 %.1f → %.1f（①）に対し %.1f → %.1f（②）、"
        "配分の移動（全変動距離）液体→結晶 %.3f → %.3f、$\\langle w\\rangle_q$ の伸び %+.1f%% → %+.1f%%。"
        % (parA[0], parA[-1], parB[0], parB[-1], tvA[-1], tvB[-1], dwA, dwB),
        "エネルギーは②のほうがわずかに低い（結晶 %.2f → %.2f, %.2f%%）。変分としては損ではないが、測りたい量の分解能は落ちる。"
        % (A[-1][1]["E"], B[-1][1]["E"],
           100 * (B[-1][1]["E"] - A[-1][1]["E"]) / abs(A[-1][1]["E"])),
    ] if ja else ["ladder comparison"]
    fig.text(0.008, 0.235, "\n".join(lines), fontsize=9.5, color="#2b2b2b",
             va="top", linespacing=2.05)
    fig.savefig(a.out, dpi=155)
    print("  wrote %s" % a.out)
    print("  %-10s 参加 %s  TV(液体→結晶) %.3f  <w>_q %+.1f%%  d_ref超の寄与 %s"
          % (a.alab, " ".join("%.2f" % v for v in parA), tvA[-1], dwA,
             " ".join("%.0f%%" % v for v in ovA)))
    print("  %-10s 参加 %s  TV(液体→結晶) %.3f  <w>_q %+.1f%%  d_ref超の寄与 %s"
          % (a.blab, " ".join("%.2f" % v for v in parB), tvB[-1], dwB,
             " ".join("%.0f%%" % v for v in ovB)))


if __name__ == "__main__":
    main()
