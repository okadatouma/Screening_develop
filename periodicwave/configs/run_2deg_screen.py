# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
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
[SCREEN] DUPLICATE of run_2deg.py that additionally supports the
'CustomScreenformer' network (screening / multiscreen attention).
The original run_2deg.py is left untouched. Differences:
  - imports train_screen instead of train,
  - 'CustomScreenformer' added to ALLOWED_NETWORK_TYPES (and made the default),
  - a cfg.network.CustomScreenformer config block (same sizes as
    CustomPsiformer, plus the screening knobs).

This code runs a calculation of a two-dimensional homogeneous electron gas with Coulomb interaction.

Usage:
    python run_2deg_screen.py [<nup>_<ndown> <r_s> <network_type> [<supercell_shape> [<iterations> [<tag>]]]]

Example:
    python run_2deg_screen.py 2_0 10 CustomScreenformer tri 600
    python run_2deg_screen.py 2_0 10 CustomScreenformer tri 600 v2

The optional <tag> is appended to the results folder name, e.g.
    results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri_v2
so that several settings can be run side by side without overwriting each
other. IMPORTANT: give a new tag whenever the network parameters change
(e.g. after enabling use_output_scale), because a run restarts from any
checkpoint already present in its folder and old checkpoints do not contain
the new parameters.

Running the script without arguments uses the default parameters set below.

Note: the calculation restarts from the newest checkpoint if one is already
present in the results folder. To start a run from scratch, move that folder
aside first.
"""

import os
import sys
from sys import argv
import numpy as np
import ml_collections

from absl import logging
from periodicwave import base_config
from periodicwave import train_screen as train  # [SCREEN]

from periodicwave.utils import custom_logging
from periodicwave.utils import writers
from periodicwave.pbc import lattices

from typing import Tuple

from time import time

import jax
print("Jax Devices:", jax.devices())

# Always use float32 precision
print("Setting jax_default_matmul_precision to float32")
jax.config.update("jax_default_matmul_precision", "float32")
print(f"Matmul precision set to: {jax.default_matmul_precision}")

ALLOWED_NETWORK_TYPES = ("CustomPsiformer", "SlaterNet", "CustomScreenformer")  # [SCREEN]
ALLOWED_SUPERCELL_SHAPES = ("tri", "sq")


def format_for_folder_name(value: float) -> str:
    """Formats a number for use in a results folder name.

    Integer-valued parameters are written without a decimal point so that
    r_s = 10 and r_s = 10.0 map onto the same folder ("rs10" in both cases).
    """
    return str(int(value)) if float(value).is_integer() else str(value)


# Get parameters defining the physical system
print(f"Calculating 2DEG with parameters: {argv}")

# Default parameters, used when the script is called without command line arguments.
nspins = [2, 0]                     # number of spin up, spin down electrons
r_s = 10.0                          # ratio between interaction and kinetic energy scale
network_type = "CustomScreenformer" # "SlaterNet", "CustomPsiformer", "CustomScreenformer" [SCREEN]
supercell_shape = 'tri'             # "tri" (triangular) or "sq" (square) supercell
iterations = 200000                 # number of optimization steps
tag = ""                            # [SCREEN] optional suffix for the results folder

if len(argv) > 1:
    if len(argv) < 4:
        raise ValueError(
            "Expected at least three arguments: <nup>_<ndown> <r_s> <network_type> "
            "[<supercell_shape>]. Received: " + str(argv[1:]))
    program_name = argv[0]
    # number of spin up, spin down electrons
    nspins = [int(n) for n in argv[1].split('_')]
    if len(nspins) != 2:
        raise ValueError(
            "Expected the electron numbers as <nup>_<ndown>, e.g. 2_0. Received: " + str(argv[1]))
    # Ratio between interaction and kinetic energy scale
    r_s = float(argv[2])
    network_type = argv[3]
    if len(argv) > 4:
        supercell_shape = argv[4]
    if len(argv) > 5:
        iterations = int(argv[5])
    if len(argv) > 6:
        tag = argv[6]

if iterations < 1:
    raise ValueError("iterations must be a positive integer. Received: " + str(iterations))
if network_type not in ALLOWED_NETWORK_TYPES:
    raise ValueError(
        "network_type must be one of " + str(ALLOWED_NETWORK_TYPES) + ". Received: " + str(network_type))
if supercell_shape not in ALLOWED_SUPERCELL_SHAPES:
    raise ValueError(
        "supercell_shape must be one of " + str(ALLOWED_SUPERCELL_SHAPES) + ". Received: " + str(supercell_shape))

print(f"Running 2DEG with nspins = {tuple(nspins)}, r_s = {r_s}, "
      f"network_type = {network_type}, supercell_shape = {supercell_shape}, "
      f"iterations = {iterations}")

# --------------------------- Set up config file ---------------------------
cfg = base_config.default()

# Set number of electrons
cfg.system.electrons = (nspins[0], nspins[1])  # (spin up, spin down) electrons

# Create the 2D lattice
cfg.system.ndim = 2
num_electrons = sum(nspins)
if supercell_shape == "tri": # triangular supercell
    supercell_a = np.sqrt(2 * np.pi / np.sqrt(3) * num_electrons)
    supercell_lattice, _ = lattices._triangular_lattice_vecs_periodic_potential(supercell_a, 1)
    area = np.linalg.det(supercell_lattice)
elif supercell_shape == "sq": # square supercell
    supercell_a = np.sqrt(np.pi * num_electrons)
    supercell_lattice = lattices._square_lattice_vecs(supercell_a)
    area = np.linalg.det(supercell_lattice)
else:
    raise NotImplementedError(
        "Only supercell shapes " + str(ALLOWED_SUPERCELL_SHAPES) + " are implemented. Received: " + supercell_shape)

# Set up the Hamiltonian parameters
cfg.system.make_local_energy_fn = "periodicwave.pbc.hamiltonian.local_energy"
cfg.system.make_local_energy_kwargs = {"lattice": supercell_lattice,
    "potential_type": 'Coulomb',
    "potential_kwargs": {'interaction_energy_scale': r_s},
    "kinetic_kwargs": {'laplacian_method': 'folx'}, # use forward laplacian
    }
cfg.system.pbc_lattice = supercell_lattice

# feature layer for pbc
cfg.network.make_feature_layer_fn = ("periodicwave.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = { "lattice": supercell_lattice, "include_r_ae": False }

# envelopes
cfg.network.make_envelope_fn = ( "periodicwave.envelopes.make_null_envelope" ) # use no envelope with pbc

# Set batch size
cfg.batch_size = 1024

# learning rate parameters
cfg.optim.optimizer = 'kfac'
cfg.optim.objective = 'vmc'
cfg.optim.iterations = iterations
cfg.optim.lr.rate  = 0.1
cfg.optim.lr.delay = 20000
cfg.optim.lr.decay = 1.0

# logging parameters
cfg.log.save_frequency = 2.0 # minutes

# mcmc parameters
cfg.mcmc.burn_in = 100
cfg.mcmc.init_width = 1.0
cfg.mcmc.move_width = 0.2
cfg.mcmc.steps = 10
cfg.mcmc.move_width_updater = 'adaptive'

# general network parameters
cfg.network.complex = True # complex wavefunction work better for pbc systems
cfg.network.determinants = 4
cfg.network.jastrow = "NONE"
cfg.network.jastrow_kwargs = {'ndim':cfg.system.ndim}

# network architecture parameters
# Code to run Psiformer
if network_type == 'CustomPsiformer':
    cfg.network.network_type = 'CustomPsiformer'
    cfg.network.CustomPsiformer.num_layers = 4
    cfg.network.CustomPsiformer.num_heads  = 4
    cfg.network.CustomPsiformer.attn_dim   = 16
    cfg.network.CustomPsiformer.value_dim  = 16
    cfg.network.CustomPsiformer.mlp_dim    = 64
    cfg.network.CustomPsiformer.num_perceptrons_per_layer = 2
    cfg.network.CustomPsiformer.use_layer_norm = True
    cfg.network.CustomPsiformer.mlp_activation_fct = "GELU"

# [SCREEN] Code to run Screenformer (screening / multiscreen attention)
elif network_type == 'CustomScreenformer':
    cfg.network.network_type = 'CustomScreenformer'
    # base_config.py has no CustomScreenformer block, so create it here
    # (keeps base_config.py untouched).
    cfg.network.CustomScreenformer = ml_collections.ConfigDict({
        # Same architecture sizes as the CustomPsiformer reference run,
        # so that any difference comes from the attention mechanism itself.
        'num_layers': 4,
        'num_heads': 4,
        'attn_dim': 16,
        'value_dim': 16,
        'mlp_dim': 64,
        'num_perceptrons_per_layer': 2,
        'use_layer_norm': True,
        'mlp_activation_fct': "GELU",
        # Screening knobs:
        'use_softmask': True,   # raised-cosine window on the physical distance r_ee
        'use_gate': True,       # gate tanh(SiLU(W_G x)) on the attention output
        'use_tanh_norm': True,  # bounded aggregation without a softmax denominator
        'use_output_scale': True,       # learnable output scale e^{s_O} on the attention branch
        # Trim acceptance width r in (0, 2). Start SOFT (close to 2): at init the
        # query/key directions are random, so cos similarities sit near 0 and a
        # sharp trim (r = 1) would zero out almost every weight.
        'init_trim_r': 1.9,
        # Initial e^{s_O}. 0 = auto-calibrate to the residual-stream magnitude.
        'init_output_scale': 0.0,
        # Initial screening lengths: geometric centre x/: spread, per head.
        # centre 0.8 puts the window edge inside the cell (where the raised
        # cosine actually has slope), spread 2.5 gives the four heads
        # w/d_ref = 0.32, 0.56, 1.14, 2.0 -> genuine multiscreen from step 0.
        'init_softmask_w_factor': 0.8,
        'init_softmask_w_spread': 2.5,
    })

# Code to run SlaterNet
elif network_type == 'SlaterNet':
    cfg.network.network_type = 'SlaterNet'
    cfg.network.SlaterNet.num_layers = 4
    cfg.network.SlaterNet.mlp_dim    = 64
    cfg.network.SlaterNet.num_perceptrons_per_layer = 2
    cfg.network.SlaterNet.use_layer_norm = True
    cfg.network.SlaterNet.mlp_activation_fct = "GELU"
    cfg.network.determinants = 1 # when using a single determinant, SlaterNet is equivalent to Hartree-Fock

# Get folder name to save the results
folder_name = f"results/2deg-Coulomb/{network_type}/el{nspins[0]}_{nspins[1]}_rs{format_for_folder_name(r_s)}_{supercell_shape}"
if tag:
    folder_name += f"_{tag}"

# save path
cfg.log.save_path = folder_name

# log device info.
# If same folder contains previous runs, rename old log file to keep information.
# The renaming follows the same convention as renaming train_stats.csv
writers.rename_file("device_info", folder_name, file_extension="log")
custom_logging.log_device_info(folder_name + "/device_info.log")
writers.rename_file("config", folder_name, file_extension="json")
custom_logging.save_config_dict_as_json(cfg, cfg.log.save_path + '/config.json')

t_init = time()
# --------------------------- train ---------------------------
train.train(cfg)
logging.info("Training completed after t [s] = " + str(int(time() - t_init)))
