# Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# [SCREEN] NOTICE: This file is a DUPLICATE of evaluate-energies.py adapted to
# the CustomScreenformer (screening attention) runs. The original file is left
# untouched. Differences:
#   - DEFAULT_FOLDER points at a CustomScreenformer results folder, and a
#     missing run tag falls back to the newest matching folder.
#   - A second panel plots locstd, the standard deviation of the local energy.
#     For an exact eigenstate E_L(R) is independent of R, so locstd -> 0 is a
#     direct measure of how good the wavefunction is, independent of how low
#     the energy happens to sit. It is the more sensitive of the two.
#   - A third panel plots the LEARNED screening lengths w_h over the course of
#     the run, read from every checkpoint in the folder. This is where the
#     multiscreen mechanism becomes visible: each head owns a window and the
#     windows separate (or collapse) as training proceeds.
#   - The learned Trim widths r_h and the output scale e^{s_O} are printed.
# =============================================================================

"""
Plots the energy, the local-energy spread and the learned screening lengths of
a CustomScreenformer run.

The script reads train_stats.csv from a results folder and, when present,
config.json from the same folder to determine the number of electrons and the
type of the Hamiltonian. It handles both homogeneous 2DEG runs ('Coulomb') and
moire runs ('CoulombMoire'). Energies of moire runs are converted to meV;
energies of homogeneous 2DEG runs are reported in the natural units of the
calculation, for which no absolute energy scale is defined.

Usage:
    python evaluate-energies-screen.py [<results_folder> [<burn_in_fraction>]]

Example:
    python evaluate-energies-screen.py results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri_v2 0.5

Running the script without arguments uses the defaults set below. The figure is
written to the results folder as PNG in addition to being shown.
"""

import glob
import json
import os
import re
from sys import argv

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- default settings ---------------------------

# Results folder holding train_stats.csv. Override on the command line.
DEFAULT_FOLDER = "results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri"  # [SCREEN]

# Fraction of the run discarded as equilibration before plotting and averaging.
DEFAULT_BURN_IN_FRACTION = 0.5

# Matplotlib backends that cannot open a window.
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

# Moire system parameters. These are ONLY used for runs with potential_type
# 'CoulombMoire', to convert energies from the natural units of the calculation
# to meV. They must match the parameters the calculation was started with.
ME_EFF_REL = 0.35                    # effective mass in units of the bare electron mass
EPS_INVERSE = 0.1                    # inverse relative dielectric constant
MOIRE_LATTICE_CONSTANT_NM = 8.031    # moire lattice constant, nm
MOIRE_POTENTIAL_STRENGTH_MEV = 15    # moire potential strength, meV


def convert_moire_scales(me_eff_rel=0.35, eps_inverse=0.2, moire_a=8.031,
                         moire_potential_strength=15):
    """Converts moire system parameters from SI to the natural units of the code.

    Default values as used in M Geier, K Nazaryan, T Zaklama, and L Fu,
    Phys. Rev. B 112, 045119.

    Returns:
        energy_scale: conversion factor from code energies to meV
        V: moire potential strength in natural units
        U: interaction energy scale in natural units
    """
    a_0 = 5.29177210544e-2       # Bohr radius, nm
    hbar = 6.582119569509066e-1  # meV * ps
    me = 5.685630060215049e-3    # electron rest mass, meV/[(nm/ps)^2]

    me_eff = me_eff_rel * me
    energy_scale = hbar ** 2 / moire_a ** 2 / me_eff  # meV
    V = moire_potential_strength / energy_scale
    U = (moire_a / a_0) * eps_inverse * (me_eff / me)
    return energy_scale, V, U


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


def load_csv_data(folder_path, file_name="train_stats.csv"):
    """Loads the training statistics written by train.train.

    locstd is optional so that older stats files still load.
    """
    file_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"No {file_name} in {folder_path}. Pass the results folder as the first argument.")
    data = pd.read_csv(file_path)
    required = ["step", "energy", "ewmean"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"{file_path} is missing the columns {missing}.")
    return data


def load_run_config(folder_path, file_name="config.json"):
    """Loads config.json saved next to the results, or returns {} if unavailable."""
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, "r") as config_file:
            return json.load(config_file)
    except (OSError, ValueError) as err:
        print(f"Could not read {file_path} ({err}). Falling back to default settings.")
        return {}


def get_num_electrons(config):
    """Returns the total electron number from a run config, or None if unknown."""
    try:
        return int(sum(config["system"]["electrons"]))
    except (KeyError, TypeError, ValueError):
        return None


def get_potential_type(config, folder_path):
    """Returns the potential type, from the config or from the folder name."""
    try:
        return config["system"]["make_local_energy_kwargs"]["potential_type"]
    except (KeyError, TypeError):
        pass
    # 'CoulombMoire' has to be tested first because 'Coulomb' is a prefix of it.
    normalized_path = folder_path.replace("\\", "/")
    for potential_type in ("CoulombMoire", "Coulomb"):
        if f"2deg-{potential_type}/" in normalized_path + "/":
            return potential_type
    return None


# [SCREEN] ------------------- screening parameters -------------------

def get_lattice_from_config(config, ndim=2):
    """Returns the supercell lattice vectors used by the run, or None.

    config.json stores the lattice as the string representation of the numpy
    array (it is not JSON-serializable), so the numbers are parsed back out.
    """
    raw = config.get("system", {}).get("pbc_lattice")
    if not isinstance(raw, str):
        return None
    numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", raw)
    if len(numbers) != ndim * ndim:
        return None
    return np.array([float(n) for n in numbers]).reshape(ndim, ndim)


def max_periodic_distance(lattice):
    """Maximal value of the periodic norm over the unit cell.

    This is the reference length d_ref that the learned screening length w is
    measured against: w / d_ref >= 1 means the window still spans the whole
    simulation cell, i.e. nothing is actually being screened out.
    """
    metric = lattice.T @ lattice
    npts = 201
    axes = [np.linspace(0.0, 1.0, npts, endpoint=False)] * lattice.shape[0]
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


def screening_history(folder_path):
    """Reads the learned screening knobs from every checkpoint in the folder.

    Returns (steps, w_over_dref, r, out_scale) where w_over_dref and r have
    shape (num_checkpoints, num_layers, num_heads) and out_scale has shape
    (num_checkpoints, num_layers). Any of them is None when the run did not
    have that parameter (e.g. use_softmask = False, or a CustomPsiformer run).
    """
    files = sorted(glob.glob(os.path.join(folder_path, "qmcjax_ckpt_*.npz")))
    steps, w_hist, r_hist, s_hist = [], [], [], []
    for path in files:
        try:
            step = int(os.path.basename(path).split("_")[-1].split(".")[0])
            params = np.load(path, allow_pickle=True)["params"].tolist()
            attention = params["layers"]["attention"]
        except Exception:  # noqa: BLE001 - a partially written checkpoint is fine to skip
            continue
        if not attention or "trim_rho" not in attention[0]:
            return None, None, None, None  # not a screening run
        steps.append(step)
        r_hist.append([2.0 / (1.0 + np.exp(-unreplicate(a["trim_rho"]))) for a in attention])
        if "mask_omega" in attention[0]:
            w_hist.append([np.log1p(np.exp(unreplicate(a["mask_omega"]))) for a in attention])
        if "out_scale" in attention[0]:
            s_hist.append([float(np.asarray(a["out_scale"]).reshape(-1)[0]) for a in attention])
    if not steps:
        return None, None, None, None
    order = np.argsort(steps)
    steps = np.asarray(steps)[order]
    r = np.asarray(r_hist)[order] if r_hist else None
    w = np.asarray(w_hist)[order] if w_hist else None
    s = np.asarray(s_hist)[order] if s_hist else None
    return steps, w, r, s


# --------------------------- read arguments ---------------------------

folder_name = resolve_results_folder(argv[1] if len(argv) > 1 else DEFAULT_FOLDER)
burn_in_fraction = float(argv[2]) if len(argv) > 2 else DEFAULT_BURN_IN_FRACTION
if not 0.0 <= burn_in_fraction < 1.0:
    raise ValueError(f"burn_in_fraction must be in [0, 1). Received: {burn_in_fraction}")

train_data = load_csv_data(folder_name)
config = load_run_config(folder_name)
potential_type = get_potential_type(config, folder_name)
num_electrons = get_num_electrons(config)

# --------------------------- set the energy scale ---------------------------

if potential_type == "CoulombMoire":
    energy_scale, _, _ = convert_moire_scales(
        ME_EFF_REL, EPS_INVERSE, MOIRE_LATTICE_CONSTANT_NM, MOIRE_POTENTIAL_STRENGTH_MEV)
    energy_unit = "meV"
else:
    if potential_type != "Coulomb":
        print(f"Unknown potential type {potential_type!r}; assuming natural units.")
    energy_scale = 1.0
    energy_unit = "natural units"

if num_electrons is None or num_electrons < 1:
    print("Electron number unavailable; plotting the total energy instead of the energy per electron.")
    num_electrons = 1
    energy_label = f"energy ({energy_unit})"
else:
    energy_label = f"energy per electron ({energy_unit})"

# --------------------------- select the plotted window ---------------------------

num_steps = len(train_data)
start_index = min(int(num_steps * burn_in_fraction), max(num_steps - 1, 0))

print(f"Folder:         {folder_name}")
print(f"Potential type: {potential_type}")
print(f"Electrons:      {num_electrons}")
print(f"Steps in file:  {num_steps} "
      f"(step {int(train_data['step'].iloc[0])} to {int(train_data['step'].iloc[-1])})")
print(f"Plotting from index {start_index} onwards (burn_in_fraction = {burn_in_fraction})")

steps = train_data["step"].to_numpy()[start_index:]
energy = train_data["energy"].to_numpy()[start_index:] * energy_scale / num_electrons
ewmean = train_data["ewmean"].to_numpy()[start_index:] * energy_scale / num_electrons
has_locstd = "locstd" in train_data.columns
locstd = train_data["locstd"].to_numpy()[start_index:] if has_locstd else None

print(f"Mean over the plotted window: {np.mean(energy):.6f} +/- "
      f"{np.std(energy) / np.sqrt(len(energy)):.6f} {energy_unit}")
if has_locstd:
    print(f"Local-energy spread over the same window: locstd = {np.mean(locstd):.4f} "
          f"(0 would mean an exact eigenstate)")

# [SCREEN] --------------------- screening knobs ---------------------

lattice = get_lattice_from_config(config, int(config.get("system", {}).get("ndim", 2)))
d_ref = max_periodic_distance(lattice) if lattice is not None else None
ck_steps, w_hist, r_hist, s_hist = screening_history(folder_name)
is_screening_run = ck_steps is not None

if not is_screening_run:
    print("\nNo screening parameters in the checkpoints "
          "(this looks like a CustomPsiformer or SlaterNet run); "
          "plotting energy and local-energy spread only.")
else:
    print(f"\nScreening parameters read from {len(ck_steps)} checkpoint(s): "
          f"steps {[int(v) for v in ck_steps]}")
    if d_ref is not None:
        print(f"  d_ref (max periodic distance in the cell) = {d_ref:.4f}")
    if w_hist is not None:
        w_final = w_hist[-1] * (d_ref if d_ref else 1.0)
        print(f"  final screening lengths w (rows = layers, columns = heads):")
        for L in range(w_final.shape[0]):
            row = "  ".join(f"{v:6.3f}" for v in w_final[L])
            frac = "  ".join(f"{v:5.2f}" for v in (w_final[L] / d_ref)) if d_ref else ""
            print(f"    layer {L}:  w = {row}   w/d_ref = {frac}")
        if d_ref is not None:
            screening_heads = int((w_final / d_ref < 1.0).sum())
            total_heads = w_final.size
            print(f"  {screening_heads} of {total_heads} heads have w < d_ref, "
                  "i.e. actually screen out distant electrons.")
    if r_hist is not None:
        print("  final Trim acceptance widths r (rows = layers, columns = heads):")
        for L in range(r_hist[-1].shape[0]):
            print("    layer " + str(L) + ":  r = "
                  + "  ".join(f"{v:5.3f}" for v in r_hist[-1][L]))
    if s_hist is not None:
        print("  final output scale e^(s_O) per layer: "
              + "  ".join(f"{v:.2f}" for v in np.exp(s_hist[-1])))

# --------------------------- plot ---------------------------

num_panels = 1 + (1 if has_locstd else 0) + (1 if (is_screening_run and w_hist is not None) else 0)
fig, axes = plt.subplots(1, num_panels, figsize=(6.2 * num_panels, 4.6), squeeze=False)
axes = axes[0]
panel = 0

# --- energy ---
ax = axes[panel]; panel += 1
ax.plot(steps, energy, marker='o', linestyle='-', linewidth=0.4, markersize=1,
        alpha=0.4, label="local energy")
ax.plot(steps, ewmean, linestyle='-', linewidth=1.5, color="tab:red",
        label="exponentially weighted mean")
ax.set_xlabel("step")
ax.set_ylabel(energy_label)
ax.set_title("Energy")
ax.legend()
ax.grid(alpha=0.3)

# --- local-energy spread ---
if has_locstd:
    ax = axes[panel]; panel += 1
    ax.plot(steps, locstd, color="tab:purple", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("locstd  (std of the local energy)")
    ax.set_title("Wavefunction quality\n(exact eigenstate would give 0)")
    ax.grid(alpha=0.3, which="both")

# [SCREEN] --- learned screening lengths ---
if is_screening_run and w_hist is not None:
    ax = axes[panel]; panel += 1
    num_layers, num_heads = w_hist.shape[1], w_hist.shape[2]
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, num_heads))
    for h in range(num_heads):
        for L in range(num_layers):
            ax.plot(ck_steps, w_hist[:, L, h], color=colors[h], alpha=0.75,
                    linewidth=1.2, label=f"head {h}" if L == 0 else None)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    ax.text(ck_steps[0], 0.97, "below this line the window is smaller than the cell\n"
            "-> distant electrons are actually screened out",
            fontsize=7.5, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"screening length  $w_h\,/\,d_{\mathrm{ref}}$")
    ax.set_title("Learned screening windows\n(one line per layer, coloured by head)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

fig.suptitle(os.path.basename(os.path.normpath(folder_name)), fontsize=11)
fig.tight_layout()

output_path = os.path.join(folder_name, "energies_screen.png")
fig.savefig(output_path, dpi=150)
print(f"\nSaved plot to {output_path}")

# Only open a window when the backend can actually show one, so that running the
# script headless (MPLBACKEND=Agg, e.g. from a batch file) stays quiet.
if matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS:
    plt.show()
else:
    print(f"Backend {matplotlib.get_backend()!r} is non-interactive; figure saved to disk only.")
