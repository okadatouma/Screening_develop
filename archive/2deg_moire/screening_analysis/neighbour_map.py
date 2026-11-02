"""「誰を見ているか」を電子ラベルに依らない形で描く。

    python screening_analysis/neighbour_map.py results/2deg-Coulomb/CustomScreenformer/el5_4_rs*_tri_det

## なぜ生の i×j 行列ではだめか

電子は区別できないので、行列の 3 行 3 列目に意味はない。MCMC 配置ごとに
「電子3」は違う場所にいる。結晶化しても電子3が特定の格子点に固定されるわけではない。
だから生の行列を r_s で並べても、見えるのは自己（対角）とスピンブロックだけで、
「相互作用する相手が変わった」は読み取れない。

## 代わりに何を描くか

配置ごとに、query 電子 i から見て key 電子 j を**距離順に並べ替える**。
すると列は「自分・第1近接・第2近接・…」という物理的な意味を持つ。
これはラベルの置換に不変で、結晶化に伴う近接殻の変化がそのまま出る。

出力:
  nb_rank.png   x = 近接順位（0 = 自分）, y = r_s, 色 = 平均の重み α·m とスコア s
  nb_shell.png  対距離のヒストグラム（= 結晶化そのもの）と、殻ごとの重み
  nb_map.npy
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def _rs(f):
    m = re.search(r"_rs([0-9.]+)_", os.path.basename(f))
    return float(m.group(1)) if m else 0.0


def true_distance(folder):
    """最小像規約の Euclid 距離。attn_probe が返す d は feature layer の周期ノルムで、
    d_ref 付近で飽和するので順位づけにも g(r) にも使えない。"""
    import glob, json
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    pos = np.asarray(z["data"].item()["positions"][0])
    cfg = json.load(open(os.path.join(folder, "config.json")))
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]
    L = np.array(v[:4]).reshape(2, 2)
    ne = sum(cfg["system"]["electrons"])
    R = pos.reshape(pos.shape[0], ne, 2)
    dr = R[:, :, None, :] - R[:, None, :, :]
    fr = dr @ np.linalg.inv(L).T                       # 格子ベクトルは列 → 転置が要る
    dr = (fr - np.round(fr)) @ L.T
    return np.linalg.norm(dr, axis=-1)                 # (B,i,j)


def collect(folders):
    import jax
    from attn_probe import probe
    out = []
    for f in sorted(folders, key=_rs):
        a = probe(f)
        NE, NL, NH = a["NE"], a["NL"], a["NH"]
        S, W = a["S"], a["W"]                     # (B,L,H,N,N)
        d = true_distance(f)                      # (B,N,N) 最小像 Euclid
        B = S.shape[0]
        # 配置ごと・query ごとに距離で並べ替え。自分は距離 0 なので必ず rank 0。
        dd = d.copy()
        idx = np.argsort(dd, axis=-1)             # (B,i,rank)
        take = lambda X: np.take_along_axis(      # noqa: E731
            X, idx[:, None, None, :, :], axis=-1)
        Sr, Wr = take(S), take(W)                 # (B,L,H,i,rank)
        dr = np.take_along_axis(dd, idx, axis=-1)
        out.append(dict(
            r_s=_rs(f), name=os.path.basename(f), NE=NE, NL=NL, NH=NH,
            S_rank=Sr.mean(axis=(0, 3)),          # (L,H,rank)
            W_rank=Wr.mean(axis=(0, 3)),          # (L,H,rank)
            d_rank=dr.mean(axis=(0, 1)),          # (rank,)
            d_all=d[:, ~np.eye(NE, dtype=bool)].ravel(),
            w=a["w"], dref=a["dref"]))
        wr = out[-1]["W_rank"].mean(0)            # (H,rank) 層平均
        print("  %-24s r_s=%-5g  自己重み比 %s"
              % (out[-1]["name"], out[-1]["r_s"],
                 np.round(wr[:, 0] / np.maximum(wr.sum(1), 1e-9), 3)), flush=True)
        jax.clear_caches()
    return out


def _font():
    import matplotlib.font_manager as fm, matplotlib.pyplot as plt
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p:
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot(rows, outdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ja = _font(); L = lambda a, b: a if ja else b        # noqa: E731
    NH = rows[0]["NH"]; NE = rows[0]["NE"]
    rs = [r["r_s"] for r in rows]
    lab = [L("自分", "self")] + [str(k) for k in range(1, NE)]

    # --- 1. 近接順位マップ -------------------------------------------------
    fig, ax = plt.subplots(2, NH, figsize=(2.9 * NH, 6.4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for h in range(NH):
        Wm = np.stack([r["W_rank"].mean(0)[h] for r in rows])          # (R,rank)
        Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-12)          # 行で規格化
        Sm = np.stack([r["S_rank"].mean(0)[h] for r in rows])
        for k, (M, cm, ttl, fmt) in enumerate([
                (Wm, "magma_r", L("重み α·m（規格化）", "weight"), "%.2f"),
                (Sm, "RdBu_r", L("スコア s = q·k", "score"), "%+.2f")]):
            A = ax[k, h]
            v = np.abs(M).max() if k else None
            im = A.imshow(M, aspect="auto", cmap=cm, origin="lower",
                          vmin=-v if k else 0, vmax=v if k else M.max())
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    A.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=6.2,
                           color="#ffffff" if (not k and M[i, j] > .5 * M.max()) else "#222222")
            A.set_xticks(range(NE)); A.set_xticklabels(lab, fontsize=8)
            A.set_yticks(range(len(rs))); A.set_yticklabels([r"$r_s$=%g" % x for x in rs], fontsize=8)
            A.set_title(("head %d — " % h) + ttl, fontsize=9.5)
            if h == 0:
                A.set_ylabel(L("結合強度 →", "coupling →"), fontsize=9)
            if k:
                A.set_xlabel(L("近接順位", "neighbour rank"), fontsize=9)
    fig.suptitle(L("query 電子から見た「相手」を距離順に並べ替えた図（層平均・ラベル不変）",
                   "attention by neighbour rank"), fontsize=12, x=.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .955])
    p = os.path.join(outdir, "nb_rank.png")
    fig.savefig(p, dpi=155, facecolor=fig.get_facecolor()); print("  wrote", p)

    # --- 2. 距離分布と殻 ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4)); fig.patch.set_facecolor("#fcfcfb")
    import matplotlib.cm as cmm
    col = cmm.viridis(np.linspace(0, .9, len(rows)))
    A = ax[0]
    for i, r in enumerate(rows):
        A.hist(r["d_all"], bins=70, density=True, histtype="step", lw=1.9,
               color=col[i], label=r"$r_s$=%g" % r["r_s"])
    A.set_xlabel(L("対距離 $d_{ij}$（$r_s a_B$ 単位）", "pair distance")); A.set_ylabel(L("密度","density"))
    A.set_title(L("① 結晶化そのもの — 近接殻が立ち上がる（最小像 Euclid 距離）", "pair distance histogram"),
                fontsize=11, loc="left")
    A.legend(fontsize=8.5); A.grid(alpha=.22)

    A = ax[1]
    for i, r in enumerate(rows):
        wr = r["W_rank"].mean(0).mean(0)                 # 層・ヘッド平均 (rank,)
        A.plot(range(NE), wr / wr.sum(), "o-", color=col[i], lw=1.9,
               label=r"$r_s$=%g" % r["r_s"])
    A.set_xticks(range(NE)); A.set_xticklabels(lab)
    A.set_xlabel(L("近接順位", "neighbour rank")); A.set_ylabel(L("重みの割合", "weight share"))
    A.set_title(L("② 相手の分布 — 結合を上げると誰に重みが移るか", "weight by rank"),
                fontsize=11, loc="left")
    A.legend(fontsize=8.5); A.grid(alpha=.22)
    fig.tight_layout()
    p = os.path.join(outdir, "nb_shell.png")
    fig.savefig(p, dpi=155, facecolor=fig.get_facecolor()); print("  wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    rows = collect(a.folders)
    np.save(os.path.join(a.out, "nb_map.npy"), rows, allow_pickle=True)
    plot(rows, a.out)
