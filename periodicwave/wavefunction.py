# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
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

# NOTICE: make_orbitals was moved from periodicwave/networks.py without changing
# its numerical operations. Original copyright and license are retained above.

"""Ordinary full-electron Slater orbital construction.

make_orbitals returns [matrix] with shape [K*states,N,N], including Envelope and
exp(J/N). The Ansatz then calls network_blocks.logdet_matmul for the determinant
sum and (sign or phase angle, log|psi|). That numerical reducer stays unchanged.
HF orbitals plus NN corrections and the dynamic-spin Jastrow stay in lj_orbitals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from periodicwave import envelopes, jastrows, network_blocks

if TYPE_CHECKING:
  from periodicwave.networks import (
      ApplyLayersFn, BaseNetworkOptions, InitLayersFn, ParamTree)


def make_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn]
) -> ...:
  """Returns init, apply pair for orbitals with spin-channels merged.

  All electrons are treated as a single channel (n_total, 0), matching the
  behavior of the original code path when use_spin_channels=False.
  """
  # Geometry stays in networks; defer this import so either module can be
  # imported first while networks keeps the historical make_orbitals export.
  from periodicwave.networks import (
      construct_input_features, make_periodic_construct_input_features)

  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(
      options.jastrow, options.jastrow_kwargs
  )

  if options.pbc_lattice is not None:
    # periodic features for correct PBC treatment (esp. Jastrow)
    construct_input_features_internal = make_periodic_construct_input_features(options.pbc_lattice)
  else:
    construct_input_features_internal = construct_input_features

  # Merge spin channels: treat all electrons in one channel
  n_total = int(sum(nspins))
  nspins_merged = (n_total, 0)  # keep tuple shape for downstream utilities

  def init(key: chex.PRNGKey) -> ParamTree:
    """Initialise parameters."""
    if n_total == 0:
      raise ValueError('No electrons present!')

    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    # Number of spin-orbitals for the single active channel
    num_states = max(options.states, 1)
    norbitals = n_total * options.determinants * num_states
    if options.complex_output:
      norbitals *= 2  # pack real/imag into even/odd outputs

    # Envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      output_dims = [norbitals // 2] if options.complex_output else [norbitals]
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # Orbital-shaping linear layer (single channel)
    key, subkey = jax.random.split(key)
    params['orbital'] = [
        network_blocks.init_linear_layer(
            subkey,
            in_dim=dims_orbital_in,
            out_dim=norbitals,
            include_bias=options.bias_orbitals,
        )
    ]

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward pass up to (merged) orbitals.

    Returns:
      A single matrix list [orbital_mat] with shape (ndet*states, n_total, n_total).
    """
    # Inputs / equivariant trunk
    ae, ee, r_ae, r_ee = construct_input_features_internal(pos, atoms, ndim=options.ndim)
    h = equivariant_layers_apply(
        params['layers'],
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee,
        spins=spins, charges=charges,
    )

    # PRE_ORBITAL envelope
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h = envelope_factor * h

    # Linear projection to orbitals (single channel)
    orbital = network_blocks.linear_layer(h, **params['orbital'][0])

    # Complex packing
    if options.complex_output:
      orbital = orbital[..., ::2] + 1.0j * orbital[..., 1::2]

    # PRE_DETERMINANT envelope
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # With one channel, just apply params['envelope'][0] to the whole set
      orbital = orbital * options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope'][0]
      )

    # Reshape to (ndet*states, n_total, n_total)
    orbital = jnp.reshape(orbital, (n_total, -1, n_total))  # (elec, blocks, elec)
    orbital = jnp.transpose(orbital, (1, 0, 2))             # (blocks, elec, elec)

    orbitals = [orbital]  # already "concatenated" (only one channel)

    # Jastrow factor (pre-determinant for compatibility with pretraining)
    if jastrow_apply is not None:
      jastrow = jnp.exp(jastrow_apply(r_ee, params['jastrow'], nspins_merged) / n_total)
      orbitals = [o * jastrow for o in orbitals]

    return orbitals

  return init, apply
