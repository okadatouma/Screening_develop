# -*- coding: utf-8 -*-
"""「窓の梯子」とは何か、置き場所を変えると何が変わるかを 1 枚で示す。"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def _font():
    import matplotlib
    from matplotlib import font_manager as fm
    for f in ("Noto Sans CJK JP", "IPAexGothic", "Noto Sans JP"):
        if any(f == x.name for x in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = f
            matplotlib.rcParams["mathtext.fontset"] = "cm"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _lat(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def load(spec):
    from head_contrib import cached
    out = []
    for part in spec.split(","):
        lab, folder = part.rsplit("=", 1)
        cfg = json.load(open(os.path.join(folder, "config.json")))
        L = _lat(cfg); ne = sum(cfg["system"]["electrons"])
        ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
        X = np.asarray(np.load(ck, allow_pickle=True)["data"].item()
                       ["positions"]).reshape(-1, ne, 2)[:600]
        fr = (X[:, :, None, :] - X[:, None, :, :]) @ np.linalg.inv(L).T
        G = L.T @ L
        aa = 1 - np.cos(2 * np.pi * fr); bb = np.sin(2 * np.pi * fr)
        dp = (1 / (2 * np.pi)) * np.sqrt(np.einsum('...m,mn,...n->...', aa, G, aa)
                                         + np.einsum('...m,mn,...n->...', bb, G, bb))
        iu = np.triu_indices(ne, 1)
        z = cached(folder, nconf=48)
        w = z["w"][0] * 3.2152
        q = (z["Cp"] ** 2).sum(0); q = 100 * q / q.sum()
        o = np.argsort(w)
        out.append(dict(lab=lab, d=dp[:, iu[0], iu[1]].ravel(), w=w[o], q=q[o],
                        rs=cfg["system"]["make_local_energy_kwargs"]
                        ["potential_kwargs"]["interaction_energy_scale"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--atitle", default="前の梯子")
    ap.add_argument("--btitle", default="置き直した梯子")
    ap.add_argument("--out", default="fig_ladder.png")
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    ja = _font(); T = lambda x, y: x if ja else y            # noqa: E731
    COL = ["#2a78d6", "#c98a12", "#c0392b"]
    CM = LinearSegmentedColormap.from_list("q", ["#d8d6d0", "#93b8ea", "#2a78d6", "#12406f"])

    sets = [(a.atitle, load(a.a)), (a.btitle, load(a.b))]
    qmax = max(it["q"].max() for _, S in sets for it in S)
    norm = Normalize(0, qmax)

    fig = plt.figure(figsize=(13.6, 7.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.10, wspace=0.14,
                          left=0.058, right=0.918, top=0.800, bottom=0.215)
    dd = np.linspace(0.02, 3.30, 900)

    for c, (title, S) in enumerate(sets):
        ax1 = fig.add_subplot(gs[0, c]); ax2 = fig.add_subplot(gs[1, c], sharex=ax1)
        for i, it in enumerate(S):
            ax1.hist(it["d"], bins=140, range=(0, 3.3), density=True, histtype="step",
                     lw=1.7, color=COL[i], label="%s $r_s$=%g" % (it["lab"], it["rs"]))
        ax1.set_ylabel(T("電子どうしの距離の分布", "pair distance"), fontsize=10.0)
        w_ = S[-1]["w"]
        ax1.set_title(title + T("　（$w_h$ = %.2f … %.2f, 隣どうし %.1f%% 違い）"
                                % (w_[0], w_[-1],
                                   100 * ((w_[-1] / w_[0]) ** (1 / (len(w_) - 1)) - 1)), ""),
                      fontsize=11.6, pad=9)
        ax1.legend(fontsize=8.8, frameon=False, loc="upper left")
        ax1.grid(alpha=.20, lw=.7); ax1.tick_params(labelbottom=False)
        ax1.set_ylim(0, None)

        it = S[-1]
        for h in range(len(it["w"])):
            wv = it["w"][h]
            m = 0.5 * (1 + np.cos(np.pi * np.clip(dd / wv, 0, 1)))
            ax2.plot(dd, m, lw=1.9, color=CM(norm(it["q"][h])), zorder=2 + h * 0.01)
            ax2.plot([wv], [0], marker="|", ms=12, mew=1.7,
                     color=CM(norm(it["q"][h])), clip_on=False, zorder=6)
        ax2.set_xlabel(T("電子どうしの距離 $d_{ij}$（周期ノルム, $r_s a_B^{*}$）", "$d_{ij}$"),
                       fontsize=10.4)
        ax2.set_ylabel(T("窓 $m_h(d)$", "$m_h(d)$"), fontsize=10.4)
        ax2.set_xlim(0, 3.3); ax2.set_ylim(0, 1.03); ax2.grid(alpha=.20, lw=.7)
        ax2.text(0.985, 0.95, T("縦棒 = 16 本の $w_h$", ""), transform=ax2.transAxes,
                 fontsize=8.8, ha="right", va="top", color="#52514e")

    cax = fig.add_axes([0.931, 0.215, 0.014, 0.42])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CM), cax=cax)
    cb.set_label(T("結晶での寄与率 $q_h$ [%]", "$q_h$ [%] (crystal)"), fontsize=9.6)
    cb.ax.tick_params(labelsize=8.6)

    fig.text(0.006, 0.983, T("窓の梯子とは — ヘッド 1 本に 1 個ずつ割り当てた距離のしきい値",
                             "what the window ladder is"),
             fontsize=13.4, weight="bold", color="#0f7b6c", va="top")
    fig.text(0.006, 0.930,
             T("$m_h(d) = \\frac{1}{2}\\left[1+\\cos\\left(\\pi\\min(d/w_h,\\,1)\\right)\\right]$　"
               "— ヘッド $h$ は $d < w_h$ の相手しか見ない（$d \\geq w_h$ で厳密に 0）。"
               "この 16 個の $w_h$ を並べたものを「梯子」と呼んでいる。", ""),
             fontsize=10.4, color="#0b0b0b", va="top")
    fig.text(0.006, 0.135,
             T("窓は入れ子であって、殻の分割ではない。$w$ の大きいヘッドは、小さいヘッドが見る相手を全部含んだ上でさらに遠くまで見る。\n"
               "だから $q_h$ を $w_h$ に対して並べたものは累積量として読むのが正しく、累積が半分に達する窓 $w_{1/2}$ が素直な代表値になる。\n"
               "左では 16 本のうち 12 本が「相手がほとんど入らない」ところに置かれていて、実質 3 本しか働いていなかった。右は使える区間に 16 本を集中させたもの。", ""),
             fontsize=9.3, color="#2b2b2b", va="top", linespacing=1.95)
    fig.savefig(a.out, dpi=155)
    print("  wrote", a.out)


if __name__ == "__main__":
    main()
