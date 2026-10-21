"""ヘッドが実際に届いている距離を Euclid 単位で測る。

ネットワークが使う距離は周期ノルム d_p（遠方で d_ref に飽和する）なので、
a_M などの物理量と比べる前に Euclid の最小像距離へ直す必要がある。
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import moire_analysis as MA          # noqa


def compute(folder, nconf=48, nb=72):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    soft = cfg["network"]["network_type"] == "CustomPsiformer"
    probe = (__import__("psi_probe").probe if soft else __import__("attn_probe").probe)
    r = probe(folder, nconf=nconf)
    L = np.asarray(r["lattice"]); NE, NL, NH = r["NE"], r["NL"], r["NH"]
    R = np.asarray(r["positions"]).reshape(-1, NE, 2)
    de = MA.min_image_dist(R[:, :, None, :] - R[:, None, :, :], L)      # (B,N,N)
    off = ~np.eye(NE, dtype=bool)
    d = de[:, off].ravel()
    dmax = float(np.sqrt((L[:, 0] @ L[:, 0])) / np.sqrt(3))              # WS 外接半径
    edges = np.linspace(0, dmax, nb + 1)
    ib = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
    cnt = np.bincount(ib, minlength=nb).astype(float)
    ctr = (edges[:-1] + edges[1:]) / 2

    prof = np.zeros((NL, NH, nb)); dbar = np.zeros((NL, NH))
    for l in range(NL):
        for h in range(NH):
            Wl = np.asarray(r["W"][:, l, h])[:, off].ravel()
            prof[l, h] = np.bincount(ib, weights=Wl, minlength=nb) / np.maximum(cnt, 1)
            dbar[l, h] = float((Wl * d).sum() / max(Wl.sum(), 1e-12))
    gd = cnt / np.maximum(ctr, 1e-9)
    gd = gd / gd[int(0.55 * nb):int(0.85 * nb)].mean()

    w_p = np.asarray(r["w"])                    # 周期ノルム単位の窓（probe は絶対値で返す）
    w_e = (np.full_like(w_p, np.nan) if soft else MA.w_to_euclid(w_p, L))
    return dict(prof=prof, d=ctr, gd=gd, cnt=cnt, dbar=dbar, w_p=w_p, w_e=w_e,
                dref=float(r["dref"]), dmax=dmax, NL=NL, NH=NH, NE=NE,
                soft=soft, lattice=L, folder=folder)


def cached(folder, nconf=48):
    p = os.path.join(folder, "reach.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files}
    out = compute(folder, nconf)
    np.savez(p, **out)
    return out
