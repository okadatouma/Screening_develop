# -*- coding: utf-8 -*-
"""ヘッドごとの動径プロファイルを、窓と同じ物差し（周期ノルム）で相ごとに並べる。

bigN_maps の _radial.png は横軸が Euclid 最小像距離なのに窓が切っているのは
周期ノルムなので、切れ際がぼやけて見える。ここでは両方を周期ノルムに揃える。

上段: 重み <alpha*m>(d)  … ネットワークが実際に配る重み
下段: 内容チャネル <alpha>(d)  … 窓を割り落として、学習された部分だけを見る
      （窓 m は固定で 3 相共通なので、下段の相依存性は純粋に学習の結果）
"""
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


def shells_periodic(L, a_ee):
    """三角格子の殻を周期ノルムに写した位置。"""
    G = L.T @ L
    th = np.linspace(0, 2 * np.pi, 145)[:-1]
    out = []
    for k, nm in ((1, "$a$"), (np.sqrt(3), "$\\sqrt{3}a$"), (2, "$2a$")):
        v = np.stack([k * a_ee * np.cos(th), k * a_ee * np.sin(th)], -1)
        fr = v @ np.linalg.inv(L).T
        aa = 1 - np.cos(2 * np.pi * fr); bb = np.sin(2 * np.pi * fr)
        d = (1 / (2 * np.pi)) * np.sqrt(np.einsum('...m,mn,...n->...', aa, G, aa)
                                        + np.einsum('...m,mn,...n->...', bb, G, bb))
        out.append((float(d.mean()), nm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="fig_radial.png")
    ap.add_argument("--nconf", type=int, default=48)
    ap.add_argument("--nbin", type=int, default=44)
    a = ap.parse_args()

    from attn_probe import probe
    items = []
    for part in a.runs.split(","):
        lab, folder = part.rsplit("=", 1)
        print("=== %s" % lab)
        r = probe(folder, nconf=a.nconf)
        cfg = json.load(open(os.path.join(folder, "config.json")))
        pk = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
        items.append(dict(lab=lab, r=r, L=_lat(cfg),
                          rs=pk["interaction_energy_scale"],
                          U=float(pk.get("moire_potential_strength", 0.0))))

    r0 = items[0]["r"]; NE, NH, dref = r0["NE"], r0["NH"], r0["dref"]
    w = r0["w"][0]                                 # 絶対長
    order = np.argsort(w)
    a_ee = np.sqrt(2 * np.pi / np.sqrt(3))
    byU = len({i["rs"] for i in items}) == 1 and len({i["U"] for i in items}) > 1
    key = (lambda i: "$U$ = %g" % i["U"]) if byU else (lambda i: "$r_s$ = %g" % i["rs"])

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis
    ja = _font(); T = lambda x, y: x if ja else y            # noqa: E731

    edges = np.linspace(0, dref, a.nbin + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    NS = len(items)
    fig = plt.figure(figsize=(4.55 * NS + 0.9, 8.2))
    gs = fig.add_gridspec(2, NS, hspace=0.30, wspace=0.16,
                          left=0.075, right=0.905, top=0.775, bottom=0.175)
    A1 = [fig.add_subplot(gs[0, c]) for c in range(NS)]
    A2 = [fig.add_subplot(gs[1, c], sharex=A1[0]) for c in range(NS)]
    top1 = top2 = 0.0
    prof = []
    for it in items:
        d = np.asarray(it["r"]["d"])               # (B,N,N) 周期ノルム
        W = np.asarray(it["r"]["W"])[:, 0]         # (B,H,N,N) = alpha*m
        off = ~np.eye(NE, dtype=bool)
        dd = d[:, off].ravel()
        idx = np.clip(np.digitize(dd, edges) - 1, 0, a.nbin - 1)
        cnt = np.bincount(idx, minlength=a.nbin).astype(float)
        P, Q = [], []
        for h in order:
            ww = W[:, h][:, off].ravel()
            s = np.bincount(idx, weights=ww, minlength=a.nbin)
            ok = cnt > 0.002 * cnt.sum()          # 対が少なすぎるビンは描かない
            p = np.divide(s, cnt, out=np.full_like(s, np.nan), where=ok)
            m = 0.5 * (1 + np.cos(np.pi * np.clip(ctr / w[h], 0, 1)))
            # 窓の縁は m -> 0 で割り算が暴れるので、m が十分ある範囲だけ
            q = np.divide(p, m, out=np.full_like(p, np.nan), where=ok & (m > 0.15))
            P.append(p); Q.append(q)
        prof.append((np.array(P), np.array(Q), cnt / cnt.sum()))
        top1 = max(top1, np.nanmax(P)); top2 = max(top2, np.nanmax(Q))

    SH = shells_periodic(items[0]["L"], a_ee)
    for c, (it, (P, Q, pd)) in enumerate(zip(items, prof)):
        for k, (ax, Z, tp) in enumerate(((A1[c], P, top1), (A2[c], Q, top2))):
            # 対の分布を背景に。重みが無い区間が「対が無いから」だと分かるように。
            ax.fill_between(ctr, 0, pd / pd.max() * tp * 1.10, color="#e9e6df",
                            zorder=0, lw=0)
            thin = pd < 0.002
            for i0 in np.where(thin)[0]:
                ax.axvspan(edges[i0], edges[i0 + 1], color="#f3ded9", zorder=0, lw=0)
            for i, h in enumerate(order):
                ax.plot(ctr, Z[i], lw=1.7, color=viridis(i / (NH - 1)))
            for sd, nm in SH:
                ax.axvline(sd, color="#008300", ls=(0, (4, 3)), lw=1.0, alpha=.75)
                if k == 0:
                    ax.text(sd, tp * 1.02, nm, color="#008300", fontsize=8.6,
                            ha="center", va="bottom")
            ax.set_xlim(0, dref * 1.01); ax.set_ylim(0, tp * 1.14)
            ax.grid(alpha=.20, lw=.7)
            if c > 0: ax.tick_params(labelleft=False)
        A1[c].set_title("%s　%s" % (it["lab"], key(it)), fontsize=11.6, pad=14)
        A2[c].set_xlabel(T("電子どうしの距離 $d_{ij}$（周期ノルム, $r_s a_B^{*}$）", "$d_{ij}$"),
                         fontsize=10.2)
    A1[0].set_ylabel(T("配る重み  $\\langle \\alpha\\, m \\rangle$", "$\\langle\\alpha m\\rangle$"),
                     fontsize=10.6)
    A2[0].set_ylabel(T("内容チャネルだけ  $\\langle \\alpha \\rangle$", "$\\langle\\alpha\\rangle$"),
                     fontsize=10.6)

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cax = fig.add_axes([0.918, 0.175, 0.014, 0.46])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(w[order][0], w[order][-1]),
                                     cmap=viridis), cax=cax)
    cb.set_label(T("固定した窓 $w_h$", "$w_h$"), fontsize=9.8)
    cb.ax.tick_params(labelsize=8.6)

    fig.text(0.006, 0.985, T("ヘッドごとの重みの届き方 — 窓と同じ物差し（周期ノルム）で",
                             "radial profiles in the periodic norm"),
             fontsize=13.4, weight="bold", color="#0f7b6c", va="top")
    fig.text(0.006, 0.930,
             T("窓 $m_h(d)=\\frac{1}{2}[1+\\cos(\\pi\\min(d/w_h,1))]$ は 3 相で完全に共通・固定。"
               "緑の破線は三角格子の殻を周期ノルムに写した位置。", ""),
             fontsize=10.4, color="#0b0b0b", va="top")
    fig.text(0.006, 0.098,
             T("灰色の影 = 電子どうしの距離の分布。桃色の帯 = 対が全体の 0.2% 未満しか無い区間で、"
               "曲線が途切れているのはデータが無いためであって窓のせいではない。\n"
               "上段は実際に配る重み。窓で切れるので、どの相でも同じ位置でゼロに落ちる。"
               "下段は窓 $m_h$ を割り落として内容チャネル $\\alpha(s_{ij})$ だけにしたもの"
               "（窓は 3 相共通なので、下段の違いは純粋に学習の結果）。", ""),
             fontsize=9.3, color="#2b2b2b", va="top", linespacing=1.95)
    fig.savefig(a.out, dpi=155)
    print("  wrote", a.out)


if __name__ == "__main__":
    main()
