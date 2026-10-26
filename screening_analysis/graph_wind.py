# -*- coding: utf-8 -*-
"""関連度グラフがトーラスを巻くのはどの窓か。

窓は距離のしきい値なので、W^(h)_ij > 0 となる対を辺とするグラフは
トーラス上のランダム幾何グラフになる。w を上げていくと、ある閾値で
グラフに「元の電子に戻ってくるのに格子ベクトルぶんずれる」閉路が現れる。
これが可縮でない（＝トポロジカルに非自明な）閉路。

    rank 0 … すべての閉路が可縮（自明）
    rank 1 … 1 方向に巻く
    rank 2 … 2 方向とも巻く（系を張っている）

配置ごとに rank が 1 / 2 になる最小の w を求め、相ごとに分布を見る。
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def _lat(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


class UF:
    """変位つき union-find。root からの分数座標のずれを持ち回る。"""

    def __init__(self, n):
        self.p = np.arange(n)
        self.off = np.zeros((n, 2))          # 分数座標での root からのずれ
        self.wind = []                        # 見つかった巻き付きベクトル

    def find(self, i):
        acc = np.zeros(2)
        while self.p[i] != i:
            acc += self.off[i]
            i = self.p[i]
        return i, acc

    def union(self, i, j, delta):
        """辺 i→j。delta = 最小像での分数変位（f_j - f_i を [-.5,.5) に畳んだもの）"""
        ri, ai = self.find(i)
        rj, aj = self.find(j)
        if ri == rj:
            v = ai + delta - aj              # 閉路の巻き（整数ベクトルのはず）
            if np.abs(v).max() > 0.5:
                self.wind.append(np.round(v).astype(int))
            return
        self.p[rj] = ri
        self.off[rj] = ai + delta - aj

    def rank(self):
        if not self.wind:
            return 0
        W = np.array(self.wind)
        W = W[np.abs(W).max(axis=1) > 0]
        return 0 if len(W) == 0 else int(np.linalg.matrix_rank(W.astype(float), tol=1e-6))


def thresholds(F, dp, ne):
    """1 配置ぶん。rank が 1 / 2 になる最小の w を返す。F = 分数座標"""
    iu = np.triu_indices(ne, 1)
    d = dp[iu]
    o = np.argsort(d)
    dd = np.stack([F[iu[1]] - F[iu[0]]], 0)[0]
    dd = dd - np.round(dd)                    # 最小像の分数変位
    uf = UF(ne)
    w1 = w2 = np.nan
    for k in o:
        uf.union(int(iu[0][k]), int(iu[1][k]), dd[k])
        r = uf.rank()
        if r >= 1 and np.isnan(w1):
            w1 = d[k]
        if r >= 2:
            w2 = d[k]
            break
    return w1, w2


def minimum_image_edges(positions, lattice, tie_tolerance=1e-12):
    """Exact 2D closest lattice image, lattice vectors are columns.

    Returns upper-edge fractional displacements, Euclidean lengths and tie flags.
    An SVD lower bound makes enumeration complete even for skew cells. Equal
    shortest images have no canonical embedded edge: weighted_winding rejects
    active ties instead of reporting representation-dependent winding ranks.
    """
    from itertools import product
    L = np.asarray(lattice,dtype=float); x = np.asarray(positions,dtype=float)
    if L.shape != (2,2) or x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(L).all() or not np.isfinite(x).all():
        raise ValueError('finite [N,2] positions and [2,2] lattice required')
    singular = np.linalg.svd(L,compute_uv=False)
    if singular[-1] <= 0: raise ValueError('singular lattice')
    # Wrapping first avoids huge image coordinates under independent lattice shifts.
    f = np.mod(x @ np.linalg.inv(L).T,1.)
    ij = np.triu_indices(len(x),1); delta = f[ij[1]]-f[ij[0]]
    displacements,distances,ties = [],[],[]
    for d in delta:
        base = np.rint(d).astype(int)
        bound = np.linalg.norm(L @ (d-base)) / singular[-1]
        low=np.ceil(d-bound-1e-12).astype(int); high=np.floor(d+bound+1e-12).astype(int)
        if np.prod(high-low+1)>100000:
            raise ValueError('ill-conditioned lattice: closest-image enumeration too large')
        shifts=np.array(list(product(*(range(a,b+1) for a,b in zip(low,high)))))
        candidates=d-shifts
        lengths=np.linalg.norm(candidates @ L.T,axis=1)
        minimum=lengths.min(); match=np.flatnonzero(np.abs(lengths-minimum)<=tie_tolerance*max(1.,minimum))
        displacements.append(candidates[match[0]]); distances.append(minimum); ties.append(len(match)>1)
    return np.asarray(displacements).reshape(-1,2),np.asarray(distances),np.asarray(ties,dtype=bool)


def weighted_winding(positions, lattice, weights, query_thresholds=None):
    """Filtration B>t; simultaneous equal-weight events, no zero edges.

    critical_rank1/2 are SUPREMA: the rank appears immediately BELOW that value,
    not at equality. Missing events are NaN. Rank is across the union of all
    components. This is an embedded real-space graph, never a many-body Chern number.
    """
    from screening_analysis.topology import symmetrize
    b = symmetrize(weights); x=np.asarray(positions).reshape(-1,2)
    if len(x)!=len(b): raise ValueError('positions and graph size differ')
    dd,dist,ties = minimum_image_edges(x,lattice)
    ij=np.triu_indices(len(b),1); w=b[ij]
    if np.any(ties & (w>0)):
        raise ValueError('active edge has degenerate shortest images; winding embedding is ambiguous')
    events=np.unique(w[w>0])[::-1]; uf=UF(len(b)); ranks=[]
    onset=[np.nan,np.nan]
    for level in events:
        for k in np.flatnonzero(w==level):
            uf.union(int(ij[0][k]),int(ij[1][k]),dd[k])
        rank=uf.rank(); ranks.append(rank)
        for j in range(2):
            if rank>=j+1 and np.isnan(onset[j]): onset[j]=level
    query=np.linspace(0,1,51) if query_thresholds is None else np.asarray(query_thresholds)
    curve=np.array([ranks[np.count_nonzero(events>t)-1] if np.any(events>t) else 0 for t in query])
    return dict(event_weights=events,event_ranks=np.asarray(ranks),thresholds=query,rank=curve,
                critical_rank1=onset[0],critical_rank2=onset[1],minimum_image_distances=dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--nconf", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = []
    for part in a.runs.split(","):
        lab, folder = part.rsplit("=", 1)
        cfg = json.load(open(os.path.join(folder, "config.json")))
        L = _lat(cfg); ne = sum(cfg["system"]["electrons"])
        rs = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"][
            "interaction_energy_scale"]
        ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
        X = np.asarray(np.load(ck, allow_pickle=True)["data"].item()
                       ["positions"]).reshape(-1, ne, 2)[:a.nconf]
        Fr = X @ np.linalg.inv(L).T
        G = L.T @ L
        w1s, w2s = [], []
        for b in range(len(X)):
            f = Fr[b]
            fr = f[:, None, :] - f[None, :, :]
            aa = 1 - np.cos(2 * np.pi * fr); bb = np.sin(2 * np.pi * fr)
            dp = (1 / (2 * np.pi)) * np.sqrt(
                np.einsum('...m,mn,...n->...', aa, G, aa)
                + np.einsum('...m,mn,...n->...', bb, G, bb))
            u, v = thresholds(f, dp, ne)
            w1s.append(u); w2s.append(v)
        w1s = np.array(w1s, float); w2s = np.array(w2s, float)
        res.append((lab, rs, w1s, w2s))
        print("%-6s rs=%-4g  1 方向に巻く w = %.3f ± %.3f   2 方向とも巻く w = %.3f ± %.3f"
              % (lab, rs, np.nanmean(w1s), np.nanstd(w1s),
                 np.nanmean(w2s), np.nanstd(w2s)))
    if a.out:
        np.savez(a.out, **{("%s_w1" % l): u for l, _, u, _ in res},
                 **{("%s_w2" % l): v for l, _, _, v in res})
        print("  wrote", a.out)


if __name__ == "__main__":
    main()
