# -*- coding: utf-8 -*-
"""報告の表（table_rows.tex）を作る。6000 step と 8500 step の 2 段（mode="6000" なら 6000 step だけ、E に誤差つき）。"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lj_maps import load, first_shell, structure_factor, LAMS
from lj_summary import stats


def sg_at(folder, ck_name):
    import json
    from lj_maps import lattice_of
    cfg = json.load(open(os.path.join(folder, "config.json")))
    z = np.load(os.path.join(folder, ck_name), allow_pickle=True)
    ne = sum(cfg["system"]["electrons"])
    pos = np.asarray(z["data"].item()["positions"])[0].reshape(-1, ne, 2)
    L = lattice_of(cfg)
    return structure_factor(pos, L, first_shell(L, ne))


def main(base, out, late=True, only6000=False):
    lines = []
    for lam in LAMS:
        cells = ["%.2f" % lam]
        for w in ("jn", "jg"):
            f = glob.glob(os.path.join(base, f"N36_rs15_lam*_{w}{int(round(lam * 100)):02d}"))[0]
            E0, dE0, sd0, nm0, _ = stats(os.path.join(f, "train_stats_step0-5999.csv"), 5500, 6000)
            s0 = sg_at(f, "qmcjax_ckpt_005999.npz")
            if only6000:
                cells += ["$%.1f\\pm%.1f$" % (E0, dE0), "%.0f" % sd0, "%.2f" % (nm0 / 36), "%.2f" % s0]
            elif late and os.path.exists(os.path.join(f, "qmcjax_ckpt_008499.npz")):
                E1, dE1, sd1, nm1, _ = stats(os.path.join(f, "train_stats.csv"), 8000, 8500)
                s1 = sg_at(f, "qmcjax_ckpt_008499.npz")
                cells += ["$%.1f$" % E0, "$%.1f$" % E1, "%.0f" % sd1, "%.2f" % (nm1 / 36), "%.2f / %.2f" % (s0, s1)]
            else:
                cells += ["$%.1f$" % E0, "--", "%.0f" % sd0, "%.2f" % (nm0 / 36), "%.2f / --" % s0]
        lines.append(" & ".join(cells) + r"\\")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[3] if len(sys.argv) > 3 else "late"
    main(sys.argv[1], sys.argv[2], late=(mode == "late"), only6000=(mode == "6000"))
