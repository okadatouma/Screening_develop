# -*- coding: utf-8 -*-
"""報告（A4）用の図：幅 7 インチに収める。full-size 版は fig_lj_maps.py。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt
import numpy as np
from fig_lj_maps import fig_density, fig_qk, fig_gap, WIN

if __name__ == "__main__":
    base, out, ref_c, ref_l = sys.argv[1:5]
    label = sys.argv[5] if len(sys.argv) > 5 else ""
    os.makedirs(out, exist_ok=True)
    S.use(); plt.rcParams.update({"font.size": 8})
    fig_density(base, out, ref_c, ref_l, label, width=7.1, name="rep_pair_density", header=False)
    rows = {}
    rows["jn"] = fig_qk(base, out, "jn", WIN[0][1], label, heads=np.arange(0, 16, 2), name="rep_qk_jn", cell=0.76, header=False)
    rows["jg"] = fig_qk(base, out, "jg", WIN[1][1], label, name="rep_qk_jg", cell=0.76, header=False)
    fig_gap(rows, out, label, header=False, name="rep_qk_gap")
    print("done")
