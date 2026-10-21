# -*- coding: utf-8 -*-
"""12 本の要約：最後の 500 step の E・σ(E_L)・n_-/N と、最後のチェックポイントでの S(G)。"""
import os, sys, glob, re, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lj_maps import load, first_shell, structure_factor, LAMS


def stats(csv, lo, hi):
    z = np.genfromtxt(csv, delimiter=",", names=True)
    sel = (z["step"] >= lo) & (z["step"] < hi)
    E = z["energy"][sel]; nb = len(E) // 50
    Eb = E[:nb * 50].reshape(nb, 50).mean(1)
    return E.mean(), Eb.std(ddof=1) / np.sqrt(nb), z["locstd"][sel].mean(), z["nminus"][sel].mean(), int(z["step"].max())


def main(base, csvname="train_stats.csv", lo=5500, hi=6000):
    rows = []
    for w in ("jn", "jg"):
        for lam in LAMS:
            f = glob.glob(os.path.join(base, f"N36_rs15_lam*_{w}{int(round(lam * 100)):02d}"))[0]
            cfg, z, pos, sp, L, ne, ck = load(f)
            sg = structure_factor(pos, L, first_shell(L, ne))
            E, dE, sd, nm, smax = stats(os.path.join(f, csvname), lo, hi)
            rows.append((w, lam, E, dE, sd, nm / ne, sg, os.path.basename(ck), smax))
            print(f"{w} {lam:.2f}  E={E:9.2f}±{dE:.2f}  sd={sd:6.1f}  n-/N={nm/ne:.3f}  S(G)={sg:.2f}  {os.path.basename(ck)}  last step {smax}")
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(a[1:2] or []), *(int(x) for x in a[2:4]))
