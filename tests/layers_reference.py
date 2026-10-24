# Copyright 2023 DeepMind Technologies Limited.
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


"""D前の層factoryの固定参照。旧ループとの同等性検証専用。"""
from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from periodicwave import networks, network_blocks
from periodicwave.CustomScreenformer import (
    ScreenformerOptions, make_screening_attention, make_mlp, _max_periodic_distance)

def make_self_attention_block(num_layers: int,
                              num_heads: int,
                              attn_dim: int,
                              value_dim: int,
                              mlp_dim: int,
                              num_perceptrons_per_layer: int,
                              num_electrons: int,
                              d_ref: float,
                              use_layer_norm: bool = False,
                              mlp_activation_fct: str = "TANH",
                              use_softmask: bool = True,
                              use_gate: bool = True,
                              use_tanh_norm: bool = True,
                              init_trim_r: float = 1.9,
                              init_softmask_w_factor: float = 0.8,
                              init_softmask_w_spread: float = 2.5,
                              init_softmask_centres=None,
                              init_softmask_rings=None,
                               use_fixed_value=False,
                              pair_dim=0,
                              use_output_scale: bool = True,
                              init_output_scale: float = 0.0) ->...:
  """Create a screening self-attention block. [SCREEN]"""
  attention_init, attention_apply = make_screening_attention(
      num_heads, mlp_dim, attn_dim, value_dim, num_electrons, d_ref,
      use_softmask=use_softmask, use_gate=use_gate, use_tanh_norm=use_tanh_norm,
      init_trim_r=init_trim_r, init_softmask_w_factor=init_softmask_w_factor,
      init_softmask_w_spread=init_softmask_w_spread,
      init_softmask_centres=init_softmask_centres,
      init_softmask_rings=init_softmask_rings,
      use_fixed_value=use_fixed_value,
      pair_dim=pair_dim,
      use_output_scale=use_output_scale,
      init_output_scale=init_output_scale)

  if use_layer_norm:
    layer_norm_init, layer_norm_apply = network_blocks.make_layer_norm()

  mlp_init, mlp_apply = make_mlp(num_perceptrons_per_layer, mlp_activation_fct)

  def init(key: chex.PRNGKey, mlp_dim: int) -> networks.ParamTree:
    params = {}
    attn_params = []
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, attn_key, mlp_key = jax.random.split(key, 3)
      attn_params.append(
          attention_init(
              attn_key))
      if use_layer_norm:
        ln_params.append([layer_norm_init(mlp_dim), layer_norm_init(mlp_dim)])
      mlp_params.append(mlp_init(mlp_key, mlp_dim, mlp_dim))

    params['attention'] = attn_params
    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: networks.ParamTree, single_electron_stream: jnp.ndarray,
            d_ee: jnp.ndarray, pair_feat: jnp.ndarray = None) -> jnp.ndarray:
    # [SCREEN] d_ee（と、固定 value を使うときは対の特徴）を各層へ通す。
    x = single_electron_stream
    for layer in range(num_layers):
      attn_output = attention_apply(params['attention'][layer], x, d_ee, pair_feat)
      # Residual + optional LayerNorm.
      x = x + attn_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

      # MLP
      mlp_output = mlp_apply(params['mlp'][layer], x)

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][1], x)

    return x

  return init, apply

def make_screenformer_layers(
    nspins: Tuple[int, ...],
    natoms: int,
    options: ScreenformerOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
  """Creates the permutation-equivariant layers for the Screenformer.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    natoms: number of atoms.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """
  del natoms  # Unused.

  # [SCREEN] Reference scale for the screening length: the largest periodic
  # distance realizable inside the simulation cell.
  d_ref = _max_periodic_distance(options.pbc_lattice)

  # Attention network.
  self_attn_init, self_attn_apply = make_self_attention_block(
      num_layers=options.num_layers,
      num_heads=options.num_heads,
      attn_dim=options.attn_dim,
      value_dim=options.value_dim,
      mlp_dim=options.mlp_dim,
      num_electrons=sum(nspins),
      num_perceptrons_per_layer=options.num_perceptrons_per_layer,
      d_ref=d_ref,
      use_layer_norm=options.use_layer_norm,
      mlp_activation_fct=options.mlp_activation_fct,
      use_softmask=options.use_softmask,
      use_gate=options.use_gate,
      use_tanh_norm=options.use_tanh_norm,
      init_trim_r=options.init_trim_r,
      init_softmask_w_factor=options.init_softmask_w_factor,
      init_softmask_w_spread=options.init_softmask_w_spread,
      init_softmask_centres=getattr(options, 'init_softmask_centres', None),
      init_softmask_rings=getattr(options, 'init_softmask_rings', None),
      use_fixed_value=getattr(options, 'use_fixed_value', False),
      pair_dim=int(options.feature_layer.init()[0][1]),
      use_output_scale=options.use_output_scale,
      init_output_scale=options.init_output_scale,
  )

  def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""
    params = {}
    key, subkey = jax.random.split(key)
    feature_dims, params['input'] = options.feature_layer.init()
    one_electron_feature_dim, _ = feature_dims
    # Concatenate spin of each electron with other one-electron features.
    feature_dim = one_electron_feature_dim + 1

    # Map to Attention dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_linear_layer(
        subkey, in_dim=feature_dim, out_dim=options.mlp_dim, include_bias=False
    )['w']

    # Attention block params.
    key, subkey = jax.random.split(key)
    params.update(self_attn_init(key, options.mlp_dim))

    return options.mlp_dim, params

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the Screenformer interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permuation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """
    del charges  # Unused.

    # Only one-electron features are used by the Screenformer stream.
    ae_features, _ = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    # For the Psiformer, the spin feature is required for correct permutation
    # equivariance.
    ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)

    features = ae_features  # Just 1-electron stream for now.

    # Embed into attention dimension.
    x = jnp.dot(features, params['embed'])

    # [SCREEN] pairwise physical (periodic) distances for the softmask.
    d_ee = r_ee[..., 0]

    # [SCREEN] 固定 value のときは、特徴層が返す 2 電子特徴
    #   e(r_ij) = [d^per_ij, sin 2pi G.r_ij, cos 2pi G.r_ij]
    # をそのまま運ぶ。並進不変で、d^per ~ |r_ij| を含むのでカスプも表現できる。
    # 距離は d_ref で割って他の成分と大きさを揃える。
    pair_feat = None
    if getattr(options, 'use_fixed_value', False):
      _, ee_features = options.feature_layer.apply(
          ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input'])
      pair_feat = jnp.concatenate(
          (ee_features[..., :1] / d_ref, ee_features[..., 1:]), axis=-1)

    return self_attn_apply(params, x, d_ee, pair_feat)

  return init, apply
