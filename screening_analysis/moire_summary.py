"""1 本の run から、図に必要な数字をまとめて取り出してキャッシュする。

    python screening_analysis/moire_summary.py <folder> [<folder> ...] [--nconf 4096]

出すもの: r_s, a_M, ν, エネルギー, ハニカム秩序変数, 谷占有のコントラスト,
各ヘッドの窓 w_h と寄与率, 寄与で重み付けした窓 w*。
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import moire_analysis as MA, sampler as SM          # noqa


def energy(folder, tail=300):
    f = os.path.join(folder, "train_stats.csv")
    d = np.genfromtxt(f, delimiter=",", names=True)
    E = d["energy"][-tail:]
    return dict(E=float(E.mean()), Eerr=float(E.std() / np.sqrt(len(E))),
                steps=int(d["step"][-1]), locstd=float(d["locstd"][-tail:].mean()))


def summarize(folder, nconf=4096, force=False):
    p = os.path.join(folder, "summary.json")
    if os.path.exists(p) and not force:
        return json.load(open(p))
    if force:                                  # 学習が進んでいるのでキャッシュを捨てる
        for f in (glob.glob(os.path.join(folder, "samples_*.npy"))
                  + [os.path.join(folder, "head_contrib.npz"),
                     os.path.join(folder, "reach.npz")]):
            if os.path.exists(f): os.remove(f)
    d = MA.load(folder)
    L, MO, ne = d["L"], d["MO"], d["ne"]
    nsite = int(round(abs(np.linalg.det(L)) / abs(np.linalg.det(MO))))
    aM = float(np.linalg.norm(MO[:, 0]))
    R = SM.cached(folder, nconf)
    d = dict(d); d["R"] = R
    op = MA.order_parameter(d)
    oc = MA.occupation_summary(d)

    from periodicwave import CustomScreenformer as CS
    import jax.numpy as jnp
    dref = float(CS._max_periodic_distance(jnp.asarray(L)))

    from head_contrib import cached as HC
    import reach as RE
    z = HC(folder, nconf=48)
    C, Cd, Cp, X, Rn, w = z["C"], z["Cd"], z["Cp"], z["X"], z["R"], z["w"]
    SQ = Cp ** 2 / (Cp ** 2).sum()               # 2 体の寄与だけで見た割合
    SQall = C ** 2 / (C ** 2).sum()              # 1 体を含めた全体
    softmax = bool(np.all(np.isnan(w)))
    wabs = w * dref                                  # 周期ノルム単位の窓
    we = (np.full_like(wabs, np.nan) if softmax else MA.w_to_euclid(wabs, L))
    rc = RE.compute(folder, nconf=48)
    rho = float((SQ * rc["dbar"]).sum() / SQ.sum())
    wstar = (float(np.nansum(SQ * np.where(np.isfinite(we), we, rc["dmax"])))
             if not softmax else float("nan"))

    out = dict(
        folder=folder, tag=os.path.basename(folder), rs=float(d["rs"]),
        U=float(d["U"]), phi=float(d["phi"]), ne=int(ne), nsite=nsite,
        nu=ne / nsite, aM=aM, dref=dref, softmax=softmax,
        a_ee=float(np.sqrt(2 * np.pi / np.sqrt(3))),     # 液体の平均電子間隔
        S_K=float(op["K_over_bg"]), S_G=float(op["G_over_bg"]),
        psi_K=float(op["psi_K"]), psi=float(oc["psi"]),
        psi_sub=float(oc["psi_sub"]), sublattice=[float(x) for x in oc["sublattice"]],
        contrast=float(oc["contrast"]), ideal_contrast=float(oc["ideal_contrast"]),
        occ_hist=[float(x) for x in oc["hist"]],
        C=C.tolist(), Cd=Cd.tolist(), Cp=Cp.tolist(),
        w=wabs.tolist(), w_e=we.tolist(), SQ=SQ.tolist(), SQall=SQall.tolist(),
        pair_frac=float((Cp ** 2).sum() / (C ** 2).sum()),
        wstar=wstar, rho=rho, dbar=rc["dbar"].tolist(), dmax=float(rc["dmax"]),
        orth=[float(C[l].sum() / Rn[l]) for l in range(C.shape[0])],
        ratio=[float(C[l].sum() / X[l]) for l in range(C.shape[0])],
        **energy(folder))
    json.dump(out, open(p, "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--nconf", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("  %-26s %7s %6s %6s %9s %7s %7s %7s %7s" %
          ("run", "r_s", "nu", "a_M", "E", "S(K)/bg", "占有σ", "w*", "到達"))
    print("  " + "-" * 84)
    for f in a.folders:
        s = summarize(f.rstrip("/"), a.nconf, a.force)
        print("  %-26s %7.4g %6.3f %6.3f %9.3f %7.2f %7.3f %7.3f %7.3f"
              % (s["tag"], s["rs"], s["nu"], s["aM"], s["E"], s["S_K"],
                 s["contrast"], s["wstar"], s["rho"]))
