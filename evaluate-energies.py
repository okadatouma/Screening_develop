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
Plots the energy per electron as a function of the optimization step.

The script reads train_stats.csv from a results folder and, when present,
config.json from the same folder to determine the number of electrons and the
type of the Hamiltonian. It handles both homogeneous 2DEG runs ('Coulomb') and
moire runs ('CoulombMoire'). Energies of moire runs are converted to meV;
energies of homogeneous 2DEG runs are reported in the natural units of the
calculation, for which no absolute energy scale is defined.

Usage:
    python evaluate-energies.py [<results_folder> [<burn_in_fraction>]]

Example:
    python evaluate-energies.py results/2deg-Coulomb/SlaterNet/el2_0_rs10_tri 0.5

Running the script without arguments uses the defaults set below.
"""

import json
import os
from sys import argv

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- default settings ---------------------------

# Results folder holding train_stats.csv. Override on the command line.
DEFAULT_FOLDER = "results/2deg-Coulomb/SlaterNet/el2_0_rs10_tri"

# Fraction of the run discarded as equilibration before plotting and averaging.
DEFAULT_BURN_IN_FRACTION = 0.5

# Matplotlib backends that cannot open a window. Listed explicitly because the
# helper matplotlib used to provide for this was removed in Matplotlib 3.11.
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

# Moire system parameters. These are ONLY used for runs with potential_type
# 'CoulombMoire', to convert energies from the natural units of the calculation
# to meV. They must match the parameters the calculation was started with.
ME_EFF_REL = 0.35                    # effective mass in units of the bare electron mass
EPS_INVERSE = 0.1                    # inverse relative dielectric constant
MOIRE_LATTICE_CONSTANT_NM = 8.031    # moire lattice constant, nm
MOIRE_POTENTIAL_STRENGTH_MEV = 15    # moire potential strength, meV


def convert_moire_scales(me_eff_rel = 0.35, eps_inverse = 0.2, moire_a = 8.031, moire_potential_strength = 15):
    """
    Converts moire system parameters from SI to natural units used in the code.

    Default values as used in M Geier, K Nazaryan, T Zaklama, and L Fu, Phys. Rev. B 112, 045119

    Arguments:
        me_eff_rel = 0.35             # effective mass in units of bare electron mass
        moire_a = 8.031               # moire lattice constant, nm
        eps_inverse = 0.1             # inverse relative diel. constant of the surrounding dielectric
        moire_potential_strength = 15 # energy scale V of the hexagonal moire potential, meV

    Returns:
        energy_scale: Conversion factor between energies returned from the code and SI units (meV)
        V: moire potential strength in natural units
        U: interaction energy scale in natural units
    """
    # Natural constants
    a_0  = 5.29177210544e-2      # Bohr radius, nm
    hbar = 6.582119569509066e-1  # meV * ps
    me   = 5.685630060215049e-3  # electron rest mass, meV/[(nm/ps)^2]

    # effective mass in SI units (meV/[(nm/ps)^2])
    me_eff = me_eff_rel * me      # effective mass, meV/[(nm/ps)^2]

    # scales converting dimensionless units from the code to SI units
    # Here we present units where distances are measured in terms of the moire lattice constant moire_a
    # and energies are measured in \hbar^2 / moire_a^2 / me_eff,
    energy_scale = hbar**2 / moire_a**2 / me_eff # meV
    # In these units, the kinetic energy term is
    # - 0.5 \sum_j \nabla_j^2 where j runs over all electrons.
    # In these units, the dimensionless moire potential strength is V, determining
    # - 2 V \sum_i \sum_{n = 1}^{3} \cos ( g_n \cdot r_i + \phi )
    V = moire_potential_strength / energy_scale
    # The dimensionless Coulomb interaction energy scale is U, determining
    # 0.5 U \sum_{i \neq j} 1/{|r_j - r_i|}
    U = (moire_a / a_0) * eps_inverse * (me_eff / me)

    return energy_scale, V, U


def load_csv_data(folder_path, file_name="train_stats.csv"):
    """Loads the training statistics written by train.train."""
    file_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"No {file_name} in {folder_path}. Pass the results folder as the first argument.")

    # Load the CSV file, only reading the first 5 columns
    data = pd.read_csv(file_path, usecols=["step", "energy", "ewmean", "ewvar", "pmove"])

    return data


def load_run_config(folder_path, file_name="config.json"):
    """Loads config.json saved next to the results, or returns {} if unavailable.

    The config is optional: without it the script falls back to plotting the
    total energy in natural units.
    """
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
    """Returns the potential type of the run.

    Read from the config where possible, otherwise inferred from the results
    folder name, which follows results/2deg-{potential_type}/...
    """
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


# --------------------------- read arguments ---------------------------

folder_name = argv[1] if len(argv) > 1 else DEFAULT_FOLDER
burn_in_fraction = float(argv[2]) if len(argv) > 2 else DEFAULT_BURN_IN_FRACTION
if not 0.0 <= burn_in_fraction < 1.0:
    raise ValueError(f"burn_in_fraction must be in [0, 1). Received: {burn_in_fraction}")

train_data = load_csv_data(folder_name)
config = load_run_config(folder_name)
potential_type = get_potential_type(config, folder_name)
num_electrons = get_num_electrons(config)

# --------------------------- set the energy scale ---------------------------

if potential_type == "CoulombMoire":
    # Moire calculations carry a physical energy scale, so convert to meV.
    energy_scale, _, _ = convert_moire_scales(
        ME_EFF_REL, EPS_INVERSE, MOIRE_LATTICE_CONSTANT_NM, MOIRE_POTENTIAL_STRENGTH_MEV)
    energy_unit = "meV"
else:
    # The homogeneous 2DEG is expressed in natural units (lengths in units of
    # r_s), for which no conversion to meV is defined.
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
# Discard the leading burn_in_fraction of the run, but always keep at least one point.
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

print(f"Mean over the plotted window: {np.mean(energy):.6f} +/- "
      f"{np.std(energy) / np.sqrt(len(energy)):.6f} {energy_unit}")

# --------------------------- plot ---------------------------

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(steps, energy, marker='o', linestyle='-', linewidth=0.4, markersize=1,
        alpha=0.4, label="local energy")
ax.plot(steps, ewmean, linestyle='-', linewidth=1.5, color="tab:red",
        label="exponentially weighted mean")
ax.set_xlabel("step")
ax.set_ylabel(energy_label)
ax.set_title(os.path.basename(os.path.normpath(folder_name)))
ax.legend()
fig.tight_layout()

output_path = os.path.join(folder_name, "energies.png")
fig.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")

# Only open a window when the backend can actually show one, so that running the
# script headless (MPLBACKEND=Agg, e.g. from a batch file) stays quiet.
if matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS:
    plt.show()
else:
    print(f"Backend {matplotlib.get_backend()!r} is non-interactive; figure saved to disk only.")
