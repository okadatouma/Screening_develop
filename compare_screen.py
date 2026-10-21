# [SCREEN] Comparison of CustomScreenformer (screening attention) vs
# CustomPsiformer (softmax attention) runs, plus readout of the learned
# screening parameters.
#
# What it does:
#   1. Reads train_stats.csv of both runs and plots energy vs step
#      (raw + exponentially weighted mean), saved as compare_screen.png.
#   2. Loads the newest checkpoint of the Screenformer run and prints the
#      learned Trim acceptance width r_h and screening length w_h for every
#      layer and head. w is reported in simulation-cell length units
#      (for the 2DEG configs: units of r_s * a_B) and as a fraction of the
#      maximal periodic distance d_ref in the cell.
#
# NOTE on distances: both the softmask input d_ij and d_ref use the smooth
# PERIODIC norm (same as the network input features), which agrees with the
# real distance at short range but saturates near the cell boundary.
#
# Usage:
#     python compare_screen.py [<screen_results_folder> [<baseline_results_folder>]]
#
# Example:
#     python compare_screen.py results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri \
#                              results/2deg-Coulomb/CustomPsiformer/el2_0_rs10_tri

import glob
import json
import os
from sys import argv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------- defaults -----------------------------
screen_dir = "results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri"
baseline_dir = "results/2deg-Coulomb/CustomPsiformer/el2_0_rs10_tri"
if len(argv) > 1:
    screen_dir = argv[1]
if len(argv) > 2:
    baseline_dir = argv[2]


def resolve_results_folder(folder_path):
    """Returns an existing results folder, tolerating a missing run tag.

    Runs are written to <base> or <base>_<tag> (see run_2deg_screen.py), so a
    path typed without the tag would otherwise fail. If folder_path does not
    exist, the newest folder whose name starts with it is used instead and the
    substitution is printed.
    """
    if os.path.isdir(folder_path):
        return folder_path
    candidates = [d for d in glob.glob(folder_path + "*") if os.path.isdir(d)]
    if not candidates:
        return folder_path  # let the caller raise a clear FileNotFoundError
    newest = max(candidates, key=os.path.getmtime)
    print(f"'{folder_path}' does not exist; using the newest matching run "
          f"'{newest}' instead.")
    print("Pass the folder explicitly to choose a different one. Available: "
          + ", ".join(sorted(os.path.basename(c) for c in candidates)))
    return newest


# A run tag may have been omitted on the command line; fall back to the newest
# matching folder rather than failing.
screen_dir = resolve_results_folder(screen_dir)
baseline_dir = resolve_results_folder(baseline_dir)


def max_periodic_distance(lattice):
    """Maximal value of the periodic norm over the unit cell (numpy grid scan)."""
    lat = np.asarray(lattice, dtype=np.float64)
    metric = lat.T @ lat
    ndim = lat.shape[0]
    npts = 201 if ndim <= 2 else 41
    axes = [np.linspace(0.0, 1.0, npts, endpoint=False)] * ndim
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    a = 1.0 - np.cos(2.0 * np.pi * grid)
    b = np.sin(2.0 * np.pi * grid)
    cos_term = np.einsum("...m,mn,...n->...", a, metric, a)
    sin_term = np.einsum("...m,mn,...n->...", b, metric, b)
    return float((1.0 / (2.0 * np.pi)) * np.sqrt(cos_term + sin_term).max())


def unreplicate(x):
    """Drop the leading device axis of checkpointed parameters if present."""
    x = np.asarray(x)
    return x[0] if x.ndim >= 2 else x


# ----------------------------- energy curves -----------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
final = {}
for label, folder, color in ((u"Screenformer (screening attention)", screen_dir, "tab:orange"),
                             (u"Psiformer (softmax attention)", baseline_dir, "tab:blue")):
    csv = os.path.join(folder, "train_stats.csv")
    if not os.path.exists(csv):
        print(f"[skip] {csv} not found")
        continue
    df = pd.read_csv(csv)
    n_el = None
    cfg_file = os.path.join(folder, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        n_el = sum(cfg["system"]["electrons"])
    scale = 1.0 / n_el if n_el else 1.0
    unit = "per electron" if n_el else "total"
    ax.plot(df["step"], df["energy"] * scale, color=color, alpha=0.25, lw=0.8)
    ax.plot(df["step"], df["ewmean"] * scale, color=color, lw=1.8, label=label)
    final[label] = (float(df["ewmean"].iloc[-1] * scale),
                    float(np.sqrt(max(df["ewvar"].iloc[-1], 0.0)) * scale),
                    int(df["step"].iloc[-1]), unit)

ax.set_xlabel("optimization step")
ax.set_ylabel("energy " + (u"per electron" if final else "") + " [natural units]")
ax.set_title("VMC energy: screening attention vs softmax attention")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out_png = "compare_screen.png"
fig.savefig(out_png, dpi=160)
print(f"saved plot: {out_png}\n")

for label, (e, err, step, unit) in final.items():
    print(f"{label:45s}: E {unit} = {e:+.6f} +- {err:.6f}  (ewmean/ewstd at step {step})")

# ----------------------------- learned screening knobs -----------------------------
ckpts = sorted(glob.glob(os.path.join(screen_dir, "qmcjax_ckpt_*.npz")))
if not ckpts:
    print(f"\n[skip] no checkpoint found in {screen_dir}")
else:
    ckpt_file = ckpts[-1]
    print(f"\nlearned screening parameters from: {ckpt_file}")

    d_ref = None
    cfg_file = os.path.join(screen_dir, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        lattice = cfg.get("system", {}).get("pbc_lattice", None)
        if isinstance(lattice, str):
            # config.json stores the lattice as
            # "unserializable object: [[ a b ]\n [ c d ]]" -> parse the numbers.
            import re
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.?(?:[eE][-+]?\d+)?",
                                                 lattice.split(":", 1)[-1])]
            n = int(round(len(nums) ** 0.5))
            lattice = np.array(nums).reshape(n, n) if n * n == len(nums) else None
        if lattice is not None:
            d_ref = max_periodic_distance(lattice)
    if d_ref is None:
        print("  (config.json without pbc_lattice; reporting w in units of d_ref)")
        d_ref = 1.0
    else:
        print(f"  d_ref (max periodic distance in cell) = {d_ref:.4f}")

    ckpt = np.load(ckpt_file, allow_pickle=True)
    params = ckpt["params"].tolist()
    attention = params["layers"]["attention"]
    print(f"  {'layer':>5s}  {'head':>4s}  {'trim r':>8s}  {'w':>8s}  {'w/d_ref':>8s}")
    for L, a in enumerate(attention):
        rho = unreplicate(a["trim_rho"])
        r = 2.0 / (1.0 + np.exp(-rho))
        if "mask_omega" in a:
            w = d_ref * np.log1p(np.exp(unreplicate(a["mask_omega"])))
        else:
            w = np.full_like(r, np.nan)  # softmask disabled in this run
        for h in range(len(r)):
            print(f"  {L:5d}  {h:4d}  {r[h]:8.4f}  {w[h]:8.4f}  {w[h]/d_ref:8.4f}")
    print("\n  interpretation: head h attends only within distance w_h (screening")
    print("  length) and only to directions with cos similarity above 1 - r_h.")
    print("  w/d_ref < 1 means the head has learned to screen out far electrons.")
