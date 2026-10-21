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

"""
Plots the electron density and the pair separations of a finished run.

Electron positions are read from the most recent checkpoints in a results
folder. The supercell lattice, the number of electrons and the type of the
Hamiltonian are read from config.json in the same folder, so the script works
for both homogeneous 2DEG runs ('Coulomb') and moire runs ('CoulombMoire').

The left panel shows every electron position folded back into the simulation
cell; for a homogeneous electron gas it should be featureless. The right panel
shows the separations between electrons in the same configuration; the void
around the origin is the exchange-correlation hole.

The radially averaged pair correlation function g(r) is computed from the full
sample and reported as text, so the depth and range of that hole can be read off
without depending on how densely the scatter plot is populated.

Usage:
    python evaluate-density.py [<results_folder> [<num_checkpoints> [<max_configurations>]]]

Example:
    python evaluate-density.py results/2deg-Coulomb/CustomPsiformer/el2_0_rs10_tri 3 256

Running the script without arguments uses the defaults set below. The figure is
written to the results folder as PNG in addition to being shown.
"""

import json
import os
import re
from sys import argv

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from periodicwave.pbc import lattices

# --------------------------- default settings ---------------------------

# Results folder holding the checkpoints. Override on the command line.
DEFAULT_FOLDER = "results/2deg-Coulomb/CustomPsiformer/el2_0_rs10_tri"

# Number of most recent checkpoints to accumulate positions from.
DEFAULT_NUM_CHECKPOINTS = 3

# Configurations drawn for the scatter plots. All loaded configurations are used
# for g(r); this only thins the markers so that individual points stay visible.
DEFAULT_MAX_CONFIGURATIONS = 256

# Target average number of samples per radial bin of g(r).
SAMPLES_PER_BIN = 32

# Matplotlib backends that cannot open a window. Listed explicitly because the
# helper matplotlib used to provide for this was removed in Matplotlib 3.11.
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

POSITION_COLOR = "#2a78d6"
SEPARATION_COLOR = "#e34948"
CELL_COLOR = "#5a5a5a"


def get_positions_from_latest_npz_files(folder_path, num_checkpoints):
    """Loads electron positions and spins from the newest checkpoints.

    Args:
        folder_path: results folder containing qmcjax_ckpt_######.npz files.
        num_checkpoints: how many of the most recent checkpoints to load.

    Returns:
        (positions, spins, filenames), where positions has shape
        (num_configurations, num_electrons * ndim).
    """
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
    """Returns the supercell lattice vectors used by the run.

    config.json stores the lattice as the string representation of the numpy
    array (it is not JSON-serializable), so the numbers are parsed back out.
    Returns None if the lattice cannot be recovered.
    """
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
    # 'CoulombMoire' has to be tested first because 'Coulomb' is a prefix of it.
    normalized_path = folder_path.replace("\\", "/") + "/"
    for potential_type in ("CoulombMoire", "Coulomb"):
        if f"2deg-{potential_type}/" in normalized_path:
            return potential_type
    return None


def inscribed_radius(lattice):
    """Largest radius around the origin that fits inside the supercell.

    Beyond it a circular shell is no longer fully contained in the cell, so the
    uniform-density normalisation of the pair correlation function breaks down.
    """
    # Row i of inv(lattice) maps a position onto its i-th fractional coordinate,
    # so the cell face at fractional coordinate 1/2 sits at distance 1/(2|row|).
    return 0.5 / np.linalg.norm(np.linalg.inv(lattice), axis=1).max()


def pair_correlation(separations, lattice, num_configurations, num_electrons):
    """Radially averaged pair correlation function g(r).

    g(r) = 1 means the electrons are uncorrelated at that separation; the dip
    towards r = 0 is the exchange-correlation hole.

    Returns (radii, g, g_uncertainty), restricted to separations for which a
    circular shell still fits inside the supercell.
    """
    radii = np.linalg.norm(separations, axis=1)
    r_max = inscribed_radius(lattice)
    radii = radii[radii <= r_max]

    num_bins = int(np.clip(np.sqrt(len(radii) / SAMPLES_PER_BIN), 4, 40))
    counts, edges = np.histogram(radii, bins=num_bins, range=(0.0, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Expected count per shell if the electrons were uncorrelated and uniform.
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

folder_name = argv[1] if len(argv) > 1 else DEFAULT_FOLDER
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

# --------------------------- electron positions ---------------------------

plotted = subsample(configurations, max_configurations)
if len(plotted) < len(configurations):
    print(f"Scatter plots show {len(plotted)} of {len(configurations)} configurations "
          f"(evenly spaced); g(r) below uses all of them.")

single_positions = lattices.send_positions_to_first_unit_cell(
    plotted.reshape(-1, ndim), lattice, rec)

# ------------------- pair separations -------------------

# All ordered pairs within each configuration, vectorised over configurations.
# The pair list is symmetric under exchange, so the map comes out symmetric too.
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
axes[0].set_title(f"Electron density (folded)\n"
                  f"{len(plotted)} configs × {num_electrons}e — expect ~uniform")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

if has_pairs:
    axes[1].scatter(plotted_separations[:, 0], plotted_separations[:, 1], s=20,
                    facecolors="none", edgecolors=SEPARATION_COLOR, linewidths=0.9, alpha=0.6)
    axes[1].plot(0, 0, marker="+", color="black", markersize=14, markeredgewidth=2.5)
    draw_supercell(axes[1], lattice)
    axes[1].set_title("Pair separation r₂−r₁\n(correlation hole at origin)")
    axes[1].set_xlabel("Δx")
    axes[1].set_ylabel("Δy")

fig.tight_layout()

output_path = os.path.join(folder_name, "density.png")
fig.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")

# Only open a window when the backend can actually show one, so that running the
# script headless (MPLBACKEND=Agg, e.g. from a batch file) stays quiet.
if matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS:
    plt.show()
else:
    print(f"Backend {matplotlib.get_backend()!r} is non-interactive; figure saved to disk only.")
