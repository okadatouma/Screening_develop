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
# [SCREEN] NOTICE: This file is a DUPLICATE of evaluate-density.py adapted to
# the CustomScreenformer (screening attention) runs. The original file is left
# untouched. Differences:
#   - DEFAULT_FOLDER points at the CustomScreenformer results folder.
#   - The learned screening parameters (Trim acceptance width r_h and
#     screening length w_h per layer/head) are read from the newest checkpoint
#     and printed as a table next to g(r).
#   - The pair-separation panel additionally shows the softmask window edge
#     d_periodic(dr) = w as dashed contours for the smallest and largest
#     learned w, so the learned screening window can be compared directly with
#     the exchange-correlation hole. If every w exceeds the maximal periodic
#     distance in the cell, no contour exists: the window still covers the
#     whole cell (no spatial screening learned yet).
# =============================================================================

"""
Plots the electron density and the pair separations of a finished
CustomScreenformer run, together with the learned screening windows.

Electron positions are read from the most recent checkpoints in a results
folder. The supercell lattice, the number of electrons and the type of the
Hamiltonian are read from config.json in the same folder.

The left panel shows every electron position folded back into the simulation
cell; for a homogeneous electron gas it should be featureless. The right panel
shows the separations between electrons in the same configuration; the void
around the origin is the exchange-correlation hole, and the dashed contours
mark the learned softmask window edges (smallest and largest w over all
layers and heads).

Usage:
    python evaluate-density-screen.py [<results_folder> [<num_checkpoints> [<max_configurations>]]]

Example:
    python evaluate-density-screen.py results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri 3 256
"""

import glob
import json
import os
import re
from sys import argv

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from periodicwave.pbc import lattices

# --------------------------- default settings ---------------------------

# Results folder holding the checkpoints. Override on the command line.
DEFAULT_FOLDER = "results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri"  # [SCREEN]

# Number of most recent checkpoints to accumulate positions from.
DEFAULT_NUM_CHECKPOINTS = 3

# Configurations drawn for the scatter plots. All loaded configurations are used
# for g(r); this only thins the markers so that individual points stay visible.
DEFAULT_MAX_CONFIGURATIONS = 256

# Target average number of samples per radial bin of g(r).
SAMPLES_PER_BIN = 32

# Matplotlib backends that cannot open a window.
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

POSITION_COLOR = "#2a78d6"
SEPARATION_COLOR = "#e34948"
CELL_COLOR = "#5a5a5a"
WINDOW_COLOR_MIN = "#2e8b57"   # [SCREEN] smallest learned w
WINDOW_COLOR_MAX = "#7b52ab"   # [SCREEN] largest learned w


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


def get_positions_from_latest_npz_files(folder_path, num_checkpoints):
    """Loads electron positions and spins from the newest checkpoints."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"No such results folder: {folder_path}")

    files = [f for f in os.listdir(folder_path)
             if f.startswith("qmcjax_ckpt_") and f.endswith(".npz")]

    numbered = []
    for file in files:
        try:
            numbered.append((int(file.split('_')[-1].split('.')[0]), file))
        except ValueError:
            print(f"Skipping {file}: no step number in the filename.")

    if not numbered:
        raise FileNotFoundError(
            f"No qmcjax_ckpt_######.npz checkpoints in {folder_path}. "
            "Pass the results folder as the first argument.")

    latest = sorted(numbered, reverse=True)[:num_checkpoints]
    print(f"Loading {len(latest)} checkpoint(s) from {folder_path}: "
          f"steps {[step for step, _ in latest]}")

    positions_ckpt, spins_ckpt, filenames = [], [], []
    for _step, filename in latest:
        file_path = os.path.join(folder_path, filename)
        try:
            ckpt_data = np.load(file_path, allow_pickle=True)
            data = ckpt_data['data'].item()
            positions_ckpt.append(np.asarray(data['positions']))
            spins_ckpt.append(np.asarray(data['spins']))
            filenames.append(filename)
        except Exception as err:  # noqa: BLE001 - keep going with the rest
            print(f"Failed to load {filename}: {err}")

    if not positions_ckpt:
        raise RuntimeError("None of the checkpoints could be loaded.")

    # Each checkpoint holds (num_devices, batch_size, num_electrons * ndim).
    positions = np.concatenate([p.reshape(-1, p.shape[-1]) for p in positions_ckpt])
    spins = np.concatenate([s.reshape(-1, s.shape[-1]) for s in spins_ckpt])
    return positions, spins, filenames


# [SCREEN] ------------- learned screening parameters -------------

def unreplicate(x):
    """Drop the leading device axis of checkpointed parameters if present."""
    x = np.asarray(x)
    return x[0] if x.ndim >= 2 else x


def load_screening_params(folder_path):
    """Reads r_h and w_h/d_ref of every layer/head from the newest checkpoint.

    Returns (r, w_over_dref) as (num_layers, num_heads) arrays, or (None, None)
    if the checkpoint has no screening parameters (e.g. a CustomPsiformer run).
    w_over_dref is None (per array) if the run had use_softmask = False.
    """
    files = [f for f in os.listdir(folder_path)
             if f.startswith("qmcjax_ckpt_") and f.endswith(".npz")]
    if not files:
        return None, None
    newest = sorted(files)[-1]
    ckpt = np.load(os.path.join(folder_path, newest), allow_pickle=True)
    try:
        attention = ckpt['params'].tolist()['layers']['attention']
    except (KeyError, TypeError):
        return None, None
    if not attention or 'trim_rho' not in attention[0]:
        return None, None
    r = np.stack([2.0 / (1.0 + np.exp(-unreplicate(a['trim_rho']))) for a in attention])
    if 'mask_omega' in attention[0]:
        w = np.stack([np.log1p(np.exp(unreplicate(a['mask_omega']))) for a in attention])
    else:
        w = None
    return r, w


def max_periodic_distance(lattice):
    """Maximal value of the periodic norm over the unit cell (numpy grid scan)."""
    metric = lattice.T @ lattice
    npts = 201
    axes = [np.linspace(0.0, 1.0, npts, endpoint=False)] * lattice.shape[0]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    a = 1.0 - np.cos(2.0 * np.pi * grid)
    b = np.sin(2.0 * np.pi * grid)
    cos_term = np.einsum("...m,mn,...n->...", a, metric, a)
    sin_term = np.einsum("...m,mn,...n->...", b, metric, b)
    return float((1.0 / (2.0 * np.pi)) * np.sqrt(cos_term + sin_term).max())


def periodic_norm_field(lattice, npts=241, pad=0.55):
    """Periodic norm d_periodic(dr) on a real-space grid around the origin.

    Returns (X, Y, D) suitable for plt.contour: the same distance measure the
    softmask uses, evaluated for separations dr covering the supercell.
    """
    metric = lattice.T @ lattice
    f = np.linspace(-pad, pad, npts)
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    frac = np.stack([F1, F2], axis=-1)                       # fractional coords
    a = 1.0 - np.cos(2.0 * np.pi * frac)
    b = np.sin(2.0 * np.pi * frac)
    cos_term = np.einsum("...m,mn,...n->...", a, metric, a)
    sin_term = np.einsum("...m,mn,...n->...", b, metric, b)
    D = (1.0 / (2.0 * np.pi)) * np.sqrt(cos_term + sin_term)
    # real-space coordinates of the grid: dr = lattice @ frac
    X = lattice[0, 0] * F1 + lattice[0, 1] * F2
    Y = lattice[1, 0] * F1 + lattice[1, 1] * F2
    return X, Y, D


def load_run_config(folder_path, file_name="config.json"):
    """Loads config.json saved next to the results, or returns {} if unavailable."""
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, "r") as config_file:
            return json.load(config_file)
    except (OSError, ValueError) as err:
        print(f"Could not read {file_path} ({err}).")
        return {}


def get_lattice_from_config(config, ndim):
    """Returns the supercell lattice vectors used by the run."""
    raw = config.get("system", {}).get("pbc_lattice")
    if not isinstance(raw, str):
        return None
    numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", raw)
    if len(numbers) != ndim * ndim:
        print(f"Could not parse {ndim}x{ndim} lattice vectors from config.json.")
        return None
    return np.array([float(n) for n in numbers]).reshape(ndim, ndim)


def get_potential_type(config, folder_path):
    """Returns the potential type, from the config or from the folder name."""
    try:
        return config["system"]["make_local_energy_kwargs"]["potential_type"]
    except (KeyError, TypeError):
        pass
    normalized_path = folder_path.replace("\\", "/") + "/"
    for potential_type in ("CoulombMoire", "Coulomb"):
        if f"2deg-{potential_type}/" in normalized_path:
            return potential_type
    return None


def inscribed_radius(lattice):
    """Largest radius around the origin that fits inside the supercell."""
    return 0.5 / np.linalg.norm(np.linalg.inv(lattice), axis=1).max()


def pair_correlation(separations, lattice, num_configurations, num_electrons):
    """Radially averaged pair correlation function g(r)."""
    radii = np.linalg.norm(separations, axis=1)
    r_max = inscribed_radius(lattice)
    radii = radii[radii <= r_max]

    num_bins = int(np.clip(np.sqrt(len(radii) / SAMPLES_PER_BIN), 4, 40))
    counts, edges = np.histogram(radii, bins=num_bins, range=(0.0, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    area = np.abs(np.linalg.det(lattice))
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    num_pairs = num_configurations * num_electrons * (num_electrons - 1)
    expected = num_pairs * shell_area / area

    return centers, counts / expected, np.sqrt(counts) / expected


def draw_supercell(ax, lattice):
    """Outlines the supercell, centred on the origin like the folded positions."""
    a1, a2 = lattice[:, 0], lattice[:, 1]
    corners = np.array([(-a1 - a2) / 2, (a1 - a2) / 2, (a1 + a2) / 2,
                        (-a1 + a2) / 2, (-a1 - a2) / 2])
    ax.plot(corners[:, 0], corners[:, 1], color=CELL_COLOR, linewidth=1.0)
    margin = 0.06 * np.abs(corners).max()
    ax.set_xlim(corners[:, 0].min() - margin, corners[:, 0].max() + margin)
    ax.set_ylim(corners[:, 1].min() - margin, corners[:, 1].max() + margin)
    ax.set_aspect("equal")


def subsample(array, max_rows):
    """Evenly spaced subsample, so the scatter stays readable and reproducible."""
    if len(array) <= max_rows:
        return array
    return array[np.linspace(0, len(array) - 1, max_rows).astype(int)]


# --------------------------- read arguments ---------------------------

folder_name = resolve_results_folder(argv[1] if len(argv) > 1 else DEFAULT_FOLDER)
num_checkpoints = int(argv[2]) if len(argv) > 2 else DEFAULT_NUM_CHECKPOINTS
max_configurations = int(argv[3]) if len(argv) > 3 else DEFAULT_MAX_CONFIGURATIONS
if num_checkpoints < 1:
    raise ValueError(f"num_checkpoints must be positive. Received: {num_checkpoints}")
if max_configurations < 1:
    raise ValueError(f"max_configurations must be positive. Received: {max_configurations}")

config = load_run_config(folder_name)
ndim = int(config.get("system", {}).get("ndim", 2))
if ndim != 2:
    raise NotImplementedError(f"This script plots two-dimensional systems. ndim = {ndim}")

electrons = config.get("system", {}).get("electrons")
num_electrons = int(sum(electrons)) if electrons else None
potential_type = get_potential_type(config, folder_name)

positions_flat, spins, filenames = get_positions_from_latest_npz_files(
    folder_name, num_checkpoints)

if num_electrons is None:
    num_electrons = positions_flat.shape[-1] // ndim
    print(f"Electron number not in config.json; inferred {num_electrons} from the checkpoints.")

lattice = get_lattice_from_config(config, ndim)
if lattice is None:
    raise RuntimeError(
        "Could not read the supercell lattice from config.json. It is needed to fold "
        "the electron positions back into the simulation cell.")
rec = 2 * np.pi * np.linalg.inv(lattice)

configurations = positions_flat.reshape(-1, num_electrons, ndim)
print(f"Folder:         {folder_name}")
print(f"Potential type: {potential_type}")
print(f"Electrons:      {num_electrons}")
print(f"Configurations: {len(configurations)}")
print(f"Supercell area: {np.abs(np.linalg.det(lattice)):.4f}")

# [SCREEN] --------------- learned screening parameters ---------------

d_ref = max_periodic_distance(lattice)
r_table, w_over_dref = load_screening_params(folder_name)
w_table = w_over_dref * d_ref if w_over_dref is not None else None

if r_table is None:
    print("\nNo screening parameters in the checkpoint (not a CustomScreenformer run);")
    print("plotting density and pair separations only.")
else:
    print(f"\nLearned screening parameters (d_ref = max periodic distance = {d_ref:.4f}):")
    print(f"  {'layer':>5s}  {'head':>4s}  {'trim r':>8s}  {'w':>8s}  {'w/d_ref':>8s}")
    for L in range(r_table.shape[0]):
        for h in range(r_table.shape[1]):
            if w_table is not None:
                print(f"  {L:5d}  {h:4d}  {r_table[L, h]:8.4f}  "
                      f"{w_table[L, h]:8.4f}  {w_table[L, h] / d_ref:8.4f}")
            else:
                print(f"  {L:5d}  {h:4d}  {r_table[L, h]:8.4f}  {'(off)':>8s}  {'-':>8s}")
    if w_table is not None:
        w_min, w_max = float(w_table.min()), float(w_table.max())
        print(f"  smallest / largest window: w_min = {w_min:.4f}, w_max = {w_max:.4f}")
        if w_min >= d_ref:
            print("  every window exceeds the maximal periodic distance: the softmask")
            print("  still covers the whole cell (no spatial screening learned yet).")

# --------------------------- electron positions ---------------------------

plotted = subsample(configurations, max_configurations)
if len(plotted) < len(configurations):
    print(f"\nScatter plots show {len(plotted)} of {len(configurations)} configurations "
          f"(evenly spaced); g(r) below uses all of them.")

single_positions = lattices.send_positions_to_first_unit_cell(
    plotted.reshape(-1, ndim), lattice, rec)

# ------------------- pair separations -------------------

i, j = np.triu_indices(num_electrons, k=1)
has_pairs = len(i) > 0
if not has_pairs:
    print("Only one electron: pair separations are undefined.")
else:
    def pair_separations(configs):
        separations = configs[:, j, :] - configs[:, i, :]
        separations = np.concatenate([separations, -separations]).reshape(-1, ndim)
        return lattices.send_positions_to_first_unit_cell(separations, lattice, rec)

    plotted_separations = pair_separations(plotted)

    radii, g_of_r, g_error = pair_correlation(
        pair_separations(configurations), lattice, len(configurations), num_electrons)
    print(f"Pair correlation function g(r), from all {len(configurations)} configurations:")
    for radius, g_value, g_err in zip(radii, g_of_r, g_error):
        bar = "#" * int(round(g_value * 20))
        print(f"  r = {radius:5.3f}   g = {g_value:5.3f} +/- {g_err:5.3f}  {bar}")
    crossing = np.interp(1.0, g_of_r, radii) if g_of_r[-1] >= 1.0 else None
    if crossing is not None:
        print(f"  g(r) reaches 1 at r = {crossing:.3f} "
              f"(inscribed radius {inscribed_radius(lattice):.3f})")

# --------------------------- plot ---------------------------

num_panels = 2 if has_pairs else 1
fig, axes = plt.subplots(1, num_panels, figsize=(6.4 * num_panels, 4.4), squeeze=False)
axes = axes[0]

axes[0].scatter(single_positions[:, 0], single_positions[:, 1], s=20,
                facecolors="none", edgecolors=POSITION_COLOR, linewidths=0.9, alpha=0.6)
draw_supercell(axes[0], lattice)
axes[0].set_title(f"Electron density (folded) — Screenformer\n"
                  f"{len(plotted)} configs × {num_electrons}e — expect ~uniform")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

if has_pairs:
    axes[1].scatter(plotted_separations[:, 0], plotted_separations[:, 1], s=20,
                    facecolors="none", edgecolors=SEPARATION_COLOR, linewidths=0.9, alpha=0.6)
    axes[1].plot(0, 0, marker="+", color="black", markersize=14, markeredgewidth=2.5)
    draw_supercell(axes[1], lattice)

    # [SCREEN] softmask window edges d_periodic(dr) = w for the smallest and
    # largest learned w. Levels above the maximum of the periodic norm draw
    # nothing: those windows still cover the whole cell.
    legend_handles = []
    if w_table is not None:
        X, Y, D = periodic_norm_field(lattice)
        for w_value, color, name in ((float(w_table.min()), WINDOW_COLOR_MIN, "smallest"),
                                     (float(w_table.max()), WINDOW_COLOR_MAX, "largest")):
            if w_value < D.max():
                axes[1].contour(X, Y, D, levels=[w_value], colors=[color],
                                linestyles="dashed", linewidths=1.6)
                label = f"{name} window edge (w = {w_value:.3f})"
            else:
                label = f"{name} w = {w_value:.3f} > cell (no edge)"
            legend_handles.append(Line2D([0], [0], color=color, linestyle="--", label=label))
        axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)

    axes[1].set_title("Pair separation r₂−r₁ with learned screening windows\n"
                      "(correlation hole at origin; dashed: softmask edge d = w)")
    axes[1].set_xlabel("Δx")
    axes[1].set_ylabel("Δy")

fig.tight_layout()

output_path = os.path.join(folder_name, "density_screen.png")
fig.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")

if matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS:
    plt.show()
else:
    print(f"Backend {matplotlib.get_backend()!r} is non-interactive; figure saved to disk only.")
