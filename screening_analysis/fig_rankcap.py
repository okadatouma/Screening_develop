# -*- coding: utf-8 -*-
"""入れ子の窓は、どれだけ鋭くしても有効ランクが 2.5 で頭打ちになる。

    python screening_analysis/fig_rankcap.py --folder <結晶の run> --out fig_rankcap.png

窓を鋭くする（物差しを尖らせるのと数学的に同じ操作）と有効ランクは 1.0 → 2.5 まで
上がるが、そこで止まる。入れ子の劣位集合が作る相関行列の固有値が 1/k^2（ブラウン橋）
になり、その参加率が厳密に 90/36 = 2.5 だからである。ヘッドを何本増やしても変わらない。
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from fig_qwin import _font, _lattice, _pnorm                        # noqa

AI, ENJI, MIDORI, HAI, SUMI = "#12406f", "#a5262f", "#0f7b6c", "#8a8880", "#141412"
CAP = 2.5


def er(M):
    X = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float((s.sum() ** 2) / (s ** 2).sum())


def mono(w, b, x):
    """立ち上がり幅 b で切る入れ子窓。b = w が今使っている生の余弦窓。"""
    t = np.clip((x - w + b) / b, 0.0, 1.0); return 0.5 * (1 + np.cos(np.pi * t))


def shell(c, s, x):
    t = np.clip((x - c) / s, -1.0, 1.0); return 0.5 * (1 + np.cos(np.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="fig_rankcap.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ja = _font()
    T = lambda x, y: x if ja else y                                  # noqa: E731

    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    L = _lattice(cfg); ne = sum(cfg["system"]["electrons"])
    P = np.load(sorted(glob.glob(os.path.join(a.folder, "qmcjax_ckpt_*.npz")))[-1],
                allow_pickle=True)["data"].item()["positions"]
    P = np.asarray(P).reshape(-1, ne, 2)[:300]
    fr = (P[:, :, None, :] - P[:, None, :, :]) @ np.linalg.inv(L).T
    iu = np.triu_indices(ne, 1)
    d = _pnorm((fr[:, iu[0], iu[1], :] - np.round(fr[:, iu[0], iu[1], :])
                ).reshape(-1, 1, 2), L).ravel()[::5]

    fig = plt.figure(figsize=(13.2, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.27,
                          left=0.065, right=0.958, top=0.790, bottom=0.335)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])

    # --- ① 鋭さを振る（曲率で横軸を取る）------------------------------------
    bs = np.geomspace(3.0, 0.02, 22)
    curv = np.pi ** 2 / (2 * bs ** 2)
    H = 16
    W = np.quantile(d, np.linspace(1.0 / H, 1.0, H))          # 等質量の閾値（最良配置）
    q = np.quantile(d, np.linspace(0.02, 0.98, H))            # 等質量の中心
    rm = [er(np.array([mono(w, b, d) for w in W])) for b in bs]
    rs_ = [er(np.array([shell(c, b, d) for c in q])) for b in bs]
    ax1.plot(curv, rm, "-o", color=ENJI, lw=2.2, ms=5.2,
             label=T("入れ子の窓（いまの形）", "nested"))
    ax1.plot(curv, rs_, "-s", color=AI, lw=2.2, ms=5.0, mfc="white",
             label=T("シェル窓（帯域通過）", "shell"))
    ax1.axhline(CAP, color=ENJI, ls=(0, (5, 3)), lw=1.5, alpha=.85)
    ax1.text(curv[0] * 1.1, CAP * 1.05, T("入れ子の上限 2.5", "cap 2.5"), color=ENJI,
             fontsize=9.4, va="bottom")
    _i = int(np.argmin(np.abs(bs - 3.0)))
    ax1.plot([curv[_i]], [rm[_i]], "*", ms=17, color=ENJI, mec="white", mew=1.2, zorder=6)
    ax1.annotate(T("いまの窓\n（$b = w_h$）", "now"), (curv[_i], rm[_i]),
                 textcoords="offset points", xytext=(10, 2), fontsize=9.0, color=ENJI)
    ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.minorticks_off()
    ax1.set_xticks([1, 10, 100, 1000, 10000])
    ax1.set_xticklabels(["1", "10", "100", "1000", "$10^4$"])
    ax1.set_yticks([1, 2, 2.5, 5, 10]); ax1.set_yticklabels(["1", "2", "2.5", "5", "10"])
    ax1.set_xlabel(T("窓の鋭さ　$\\max|m_h''|=\\pi^2/2b^2$"
                     "（$b$ = 立ち上がり幅。右へ行くほど鋭い）", "sharpness"), fontsize=10.2)
    ax1.set_ylabel(T("実質いくつの形を持つか（有効ランク）", "effective rank"), fontsize=10.4)
    ax1.grid(alpha=.22, lw=.7, which="both")
    ax1.legend(fontsize=9.6, frameon=False, loc="upper left")
    ax1.set_title(T("① 鋭くすれば上がる。ただし入れ子は 2.5 で止まる",
                    "① sharpening saturates at 2.5"), fontsize=11.4, pad=8)

    # --- ② ヘッドを増やしても変わらない -------------------------------------
    HH = np.array([4, 8, 16, 32, 64, 128])
    for b, col, ls, lab in ((0.4, "#c98a12", "-", T("やや鋭い $b$=0.4", "b=0.4")),
                            (0.02, ENJI, "-", T("ほぼ階段 $b$=0.02", "b=0.02"))):
        v = []
        for h in HH:
            Wh = np.quantile(d, np.linspace(1.0 / h, 1.0, h))
            v.append(er(np.array([mono(w, b, d) for w in Wh])))
        ax2.plot(HH, v, "-o", color=col, lw=2.2, ms=6, ls=ls, label=lab)
    v = []
    for h in HH:
        qh = np.quantile(d, np.linspace(0.02, 0.98, h))
        sp = (qh[-1] - qh[0]) / (h - 1)
        v.append(er(np.array([shell(c, sp, d) for c in qh])))
    ax2.plot(HH, v, "-s", color=AI, lw=2.2, ms=6, mfc="white",
             label=T("シェル窓（$\\sigma$ = 間隔）", "shell"))
    ax2.axhline(CAP, color=ENJI, ls=(0, (5, 3)), lw=1.5, alpha=.85)
    ax2.text(HH[-1], CAP * 0.93, T("入れ子の上限 2.5 ", "cap "), color=ENJI,
             fontsize=9.4, va="top", ha="right")
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.minorticks_off()
    ax2.set_xticks(HH); ax2.set_xticklabels(["%d" % h for h in HH], fontsize=9.4)
    ax2.set_yticks([1, 2, 2.5, 5, 10, 20]); ax2.set_yticklabels(["1", "2", "2.5", "5", "10", "20"])
    ax2.set_xlabel(T("ヘッド数 $H$", "heads"), fontsize=10.4)
    ax2.set_ylabel(T("有効ランク", "effective rank"), fontsize=10.4)
    ax2.grid(alpha=.22, lw=.7, which="both")
    ax2.legend(fontsize=9.4, frameon=False, loc="upper left")
    ax2.set_title(T("② ヘッドを増やしても入れ子は上限を越えない",
                    "② more heads do not help"), fontsize=11.4, pad=8)

    fig.text(0.006, 0.985,
             T("窓を鋭くしても、入れ子である限り 2.5 で止まる",
               "the nesting cap"), fontsize=13.2, weight="bold", color=MIDORI, va="top")
    fig.text(0.006, 0.928,
             T("クーロンのみ, $N$ = %d, $r_s$ = 60（結晶）の対の分布の上で。"
               "閾値・中心はどちらも等質量に置いた（各族にとっての最良配置）" % ne,
               "on the measured pair distribution"),
             fontsize=10.4, color="#0b0b0b", va="top")
    lines = [
        "物差しを尖らせることと窓の形を鋭くすることは、同じ操作である。窓は $m_h(d)=\\varphi(g(d)/v_h)$ "
        "の形をしているので、物差し $g$ を歪めても窓の断面 $\\varphi$ を変えても、"
        "$d$ の関数としての結果は同じ族になる。",
        "鋭くすれば有効ランクは上がる（%.2f → %.2f）。そこはご指摘のとおりで、"
        "前の最小イメージでの比較（1.06 → 1.16）は尖らせ方が甘かった。" % (rm[0], rm[-1]),
        "しかし上限がある。窓が入れ子（$\\{d<w_h\\}$ という劣位集合の列）である限り、"
        "相関行列の固有値は $1/k^2$ に落ち着き（ブラウン橋）、その参加率は厳密に $90/36 = 2.5$。"
        "ヘッドを 4 本から 128 本に増やしても変わらない（②）。",
        "$d_{\\rm ref}$ の数値自体を上げるのは意味がない。物差しを 2 倍すれば $d_{\\rm ref}$ も 2 倍になるが、"
        "$w_h$ も 2 倍になるだけで何も変わらない。効くのは物差しの「形」＝対の分布の並び方であって、大きさではない。",
        "上限を破るのは入れ子をやめること。シェル窓は劣位集合ではなく重なった帯なので、"
        "2.5 を越えて伸びる。なお鋭くするほど $\\max|m_h''| = \\pi^2/2b^2$ が効いてくるので、"
        "$\\nabla^2\\log\\psi$ の側に予算の制約はある。",
    ] if ja else ["see figure"]
    fig.text(0.008, 0.252, "\n".join(lines), fontsize=9.3, color="#2b2b2b",
             va="top", linespacing=1.95)
    fig.savefig(a.out, dpi=155)
    print("  wrote %s" % a.out)
    print("  入れ子: %.2f（いまの窓）→ %.2f（ほぼ階段）  上限 2.5" % (rm[0], rm[-1]))
    print("  シェル: %.2f → %.2f" % (rs_[0], rs_[-1]))


if __name__ == "__main__":
    main()
