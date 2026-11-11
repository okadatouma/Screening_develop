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

# =============================================================================
# NOTICE: This file is a DUPLICATE of CustomPsiformer.py with the softmax
# attention replaced by a "screening attention" (multiscreen) unit.
# The original CustomPsiformer.py is left untouched so both ansatze can be
# compared head-to-head.
#
# Changes relative to CustomPsiformer.py (marked with [SCREEN]):
#   1. make_multi_head_attention -> make_screening_attention:
#      - unit-length normalization of q, k, v  ->  cosine similarity
#        s_ij = q̄_i · k̄_j in [-1, 1]  (no 1/sqrt(d) scaling needed)
#      - Trim: alpha_ij = (relu(s_ij - (1 - r)) / r)^2, learnable acceptance
#        width r per head; s_ij <= 1 - r contributes exactly zero.
#      - Softmask: raised-cosine window m_ij = (1 + cos(pi d_ij / w)) / 2 for
#        d_ij < w, exactly 0 beyond. Learnable screening length w per head.
#        [SCREEN] Here the tokens are ELECTRONS, so the distance d_ij is the
#        PHYSICAL electron-electron distance (periodic norm r_ee), not a token
#        index distance. Each head learns its own screening length w_h:
#        this per-head multiplicity of screening scales is the "multiscreen"
#        mechanism. w is measured in the same units as the simulation cell
#        (for the 2DEG configs: units of r_s * a_B).
#      - Denominator-free aggregation h_i = sum_j alpha_ij m_ij v̄_j
#        (no softmax normalization), followed by TanhNorm
#        o = tanh(|h|) h / |h| which keeps the update bounded.
#      - Optional gate: o <- o * tanh(SiLU(W_G x)) (screening-unit gate).
#   2. Attention receives the pairwise distance matrix d_ij; the apply
#      signatures of the block/layers thread r_ee down to the attention.
#   3. No MiPE / positional rotation: electrons carry no sequence order, and
#      permutation equivariance forbids index-based positional encodings.
#      Positional information enters through the periodic input features.
#
# Smoothness note (relevant for VMC local energy): the raised cosine is C^1
# at d = w with bounded second derivative, and dm/dd -> 0 as d -> 0, so the
# softmask introduces no new cusps at electron coincidences. The squared ReLU
# of Trim is likewise C^1 with bounded second derivative.
# =============================================================================

""" Screening-attention neural network variational wavefunction ansatz. """

from typing import Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from periodicwave import envelopes
from periodicwave import jastrows
from periodicwave import network_blocks
from periodicwave import networks
import jax
import jax.numpy as jnp
import numpy as np


@attr.s(auto_attribs=True, kw_only=True)
class ScreenformerOptions(networks.BaseNetworkOptions):
  """Options controlling the Screenformer part of the network architecture.

  Attributes:
    num_layers: Number of self-attention layers.
    num_heads: Number of self-attention heads per layer.
    attn_dim: Embedding dimension for each self-attention head.
    value_dim: Dimension of the value vector.
    mlp_dim: Dimension of the perceptron layers and the internal token dimension
    num_perceptrons_per_layer: Number of perceptron layers after each self-attention layer.
    use_layer_norm: If true, include a layer norm on both attention and MLP.
    mlp_activation_fct: Non-linear activation function used in the perceptron layers
    use_softmask: [SCREEN] If true, multiply the trimmed similarity by the
      raised-cosine softmask on the physical electron-electron distance.
      If false, only Trim acts (ablation switch).
    use_gate: [SCREEN] If true, apply the screening-unit gate
      tanh(SiLU(W_G x)) to the attention output.
    use_tanh_norm: [SCREEN] If true, apply TanhNorm to the aggregated head
      vectors (bounded output without a softmax denominator).
    init_trim_r: [SCREEN] Initial acceptance width r of Trim (0 < r < 2).
    init_softmask_w_factor: [SCREEN] Geometric CENTRE of the initial screening
      lengths w, as a multiple of the maximal periodic distance in the cell.
      Values around 1 put the window edge inside the cell, where the raised
      cosine has a non-negligible slope, so w actually receives gradient.
    init_softmask_w_spread: [SCREEN] Multiscreen spread. The heads of a layer
      are initialised at geometrically spaced screening lengths from
      centre/spread to centre*spread, so different heads start out screening
      at different scales. spread = 1 makes every head identical (the old
      behaviour, which leaves the heads symmetric and wastes multiscreen).
    use_output_scale: [SCREEN] If true, multiply the attention output by a
      learnable scalar e^{s_O} before the residual is added. TanhNorm and the
      gate both bound the head output from above, so without this scale the
      attention branch cannot reach the magnitude the residual stream needs
      and the layer degenerates towards "MLP only".
    init_output_scale: [SCREEN] Initial value of e^{s_O}. 0 (the default)
      means auto-calibrate: at initialisation the query and key directions are
      random, so the cosine similarity is only of order 1/sqrt(attn_dim) and
      the trimmed weights come out tiny. The scale is then set so that the
      attention branch starts at the same magnitude as the residual stream,
      which is what softmax attention does automatically through its
      denominator. Without this the branch starts ~100x too small and the
      layer behaves like "MLP only" for a long time.
  """

  num_layers: int = 2
  num_heads: int = 4
  attn_dim: int = 32
  value_dim: int = 32
  mlp_dim: int = 128,
  num_perceptrons_per_layer: int = 2,
  use_layer_norm: bool = False
  mlp_activation_fct: str = "TANH"
  use_softmask: bool = True                 # [SCREEN]
  use_gate: bool = True                     # [SCREEN]
  use_tanh_norm: bool = True                # [SCREEN]
  init_trim_r: float = 1.9                  # [SCREEN]
  init_softmask_w_factor: float = 0.8       # [SCREEN]
  init_softmask_w_spread: float = 2.5       # [SCREEN]
  use_output_scale: bool = True             # [SCREEN]
  init_output_scale: float = 0.0            # [SCREEN] 0 = auto-calibrate


TRIM_MAX = 2.0



# [PROBE] 解析用の距離クリップ。CLIP_R（周期ノルム単位）が None なら無効。
# 硬く切ると波動関数が不連続になり、前進ラプラシアンが運動エネルギーの特異項を
# 取りこぼして偽の低いエネルギーを出す。そこで縁だけ滑らかに落とす襟を掛ける:
#   c(d) = 1                              (d <= (1-β)R)
#          ½[1+cos(π (d-(1-β)R)/(βR))]    ((1-β)R < d < R)
#          0                              (d >= R)
CLIP_R = None
CLIP_BETA = 0.30

def make_layer_norm() ->...:
  """Implementation of LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(params: networks.ParamTree,
            inputs: jnp.ndarray,
            axis: int = -1) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply


# [SCREEN] Helper: unit-length normalization along the last axis.
# The small eps keeps the map twice differentiable at the origin.
def _unit_length(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps * eps)
  return x / norm


# [SCREEN] Helper: maximal value of the periodic norm over the unit cell,
# used as the reference scale for the learnable screening length w.
def _max_periodic_distance(lattice) -> float:
  lat = np.asarray(lattice, dtype=np.float64)
  if lat.size == 0:
    return 1.0
  ndim = lat.shape[0]
  metric = lat.T @ lat
  npts = 201 if ndim <= 2 else 41
  axes = [np.linspace(0.0, 1.0, npts, endpoint=False)] * ndim
  grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
  a = 1.0 - np.cos(2.0 * np.pi * grid)
  b = np.sin(2.0 * np.pi * grid)
  cos_term = np.einsum('...m,mn,...n->...', a, metric, a)
  sin_term = np.einsum('...m,mn,...n->...', b, metric, b)
  return float((1.0 / (2.0 * np.pi)) * np.sqrt(cos_term + sin_term).max())


def make_screening_attention(num_heads: int, embed_dim: int, attn_dim: int,
                             value_dim: int, num_electrons: int,
                             d_ref: float,
                             use_softmask: bool = True,
                             use_gate: bool = True,
                             use_tanh_norm: bool = True,
                             init_trim_r: float = 1.9,
                             init_softmask_w_factor: float = 0.8,
                             init_softmask_w_spread: float = 2.5,
                             use_output_scale: bool = True,
                             init_output_scale: float = 0.0) ->...:
  """[SCREEN] Screening (multiscreen) replacement of MultiHeadAttention.

  Pipeline per head h:
    q, k, v (, g)  ->  unit length q̄, k̄, v̄
    s_ij = q̄_i · k̄_j                       (cosine similarity, bounded)
    alpha_ij = (relu(s_ij - (1 - r_h)) / r_h)^2          (Trim)
    m_ij = (1 + cos(pi * min(d_ij / w_h, 1))) / 2        (Softmask, physical d)
    h_i = sum_j alpha_ij m_ij v̄_j          (denominator-free aggregation)
    o_i = tanh(|h_i|) h_i / |h_i|           (TanhNorm)
    o_i <- o_i * tanh(SiLU(g_i))            (gate, optional)
    Delta x_i = e^{s_O} W_O o_i             (learnable output scale)
  r_h, w_h are learnable per head and s_O is learnable per layer:
    r_h = 2 * sigmoid(trim_rho_h)        in (0, 2)
    w_h = d_ref * softplus(mask_omega_h) > 0.

  Multiscreen initialisation: the heads do NOT start at the same screening
  length. They are spread geometrically around init_softmask_w_factor by a
  factor init_softmask_w_spread, so that within one layer some heads look
  only at close neighbours while others still see the whole cell. Identical
  initial windows leave the heads symmetric and there is nothing to break
  that symmetry except noise.
  """

  # Inverse parameterizations for the requested initial values.
  # [FIX] paper (arXiv:2604.01178) defines r = sigmoid(s_r) in (0,1).
  init_trim_r = float(np.clip(init_trim_r, 1e-3, TRIM_MAX - 1e-3))
  rho_init = float(np.log(init_trim_r / (TRIM_MAX - init_trim_r)))  # logit(r/TRIM_MAX)

  # [SCREEN] geometrically spaced initial screening lengths (multiscreen).
  centre = max(float(init_softmask_w_factor), 1e-3)
  spread = max(float(init_softmask_w_spread), 1.0)
  if num_heads == 1 or spread == 1.0:
    w_factors = np.full((num_heads,), centre)
  else:
    w_factors = np.geomspace(centre / spread, centre * spread, num_heads)
  omega_init = np.log(np.expm1(np.clip(w_factors, 1e-3, None)))

  # [SCREEN] Auto-calibration of the output scale e^{s_O}.
  # At init, q and k are random directions, so s_ij = cos(q,k) is spread like a
  # Gaussian of width 1/sqrt(attn_dim) around 0. The Trim weight
  # alpha = (relu(s - (1-r))/r)^2 therefore has a small mean, and the
  # denominator-free sum h_i = sum_j alpha_ij m_ij vbar_j comes out ~100x
  # smaller than the residual stream. Softmax attention avoids this because its
  # denominator renormalises the weights to sum to one. Here we restore the
  # scale explicitly: the expected weight is computed by quadrature and
  # e^{s_O} is set so the branch starts at the residual-stream magnitude.
  _sig = 1.0 / np.sqrt(float(attn_dim))
  _s_grid = np.linspace(-5.0 * _sig, 5.0 * _sig, 4001)
  _pdf = np.exp(-0.5 * (_s_grid / _sig) ** 2)
  _pdf = _pdf / _pdf.sum()
  _alpha_mean = float(
      (np.square(np.maximum(_s_grid - (1.0 - init_trim_r), 0.0) / init_trim_r) * _pdf).sum())
  # Empirical gain of the remaining stages (TanhNorm, gate, W_O) relative to
  # the naive estimate. Calibrated by measuring |attention output| at
  # initialisation against softmax attention on the same inputs, so that the
  # screening branch and the softmax branch start at the same magnitude and
  # the two ansatze are compared on equal footing.
  _branch_gain = 0.85 if use_gate else 2.4
  # The branch should start noticeably below the residual stream, not at parity:
  # the trimmed / masked weights carry sharp features (the ReLU^2 kink of Trim
  # and the window edge of the softmask) and amplifying them to full stream
  # magnitude raises the curvature of log|psi|, which shows up directly as a
  # larger spread of the local energy. A fraction of ~0.3 measured best.
  _branch_target_fraction = 0.3
  if init_output_scale and init_output_scale > 0.0:
    _scale0 = float(init_output_scale)
  else:
    _scale0 = _branch_target_fraction * np.sqrt(float(embed_dim)) / (
        max(_alpha_mean, 1e-8) * np.sqrt(float(num_electrons)) * _branch_gain)
  _scale0 = float(np.clip(_scale0, 1e-3, 1e4))
  out_scale_init = float(np.log(_scale0))

  def init(key: chex.PRNGKey) -> Mapping[str, jnp.ndarray]:

    key, *subkeys = jax.random.split(key, num=6)
    params = {}
    params['q_w'] = network_blocks.init_linear_layer(
        subkeys[0], in_dim=embed_dim, out_dim=num_heads * attn_dim, include_bias=False)['w']
    params['k_w'] = network_blocks.init_linear_layer(
        subkeys[1], in_dim=embed_dim, out_dim=num_heads * attn_dim, include_bias=False)['w']
    params['v_w'] = network_blocks.init_linear_layer(
        subkeys[2], in_dim=embed_dim, out_dim=num_heads * value_dim, include_bias=False)['w']
    params['attn_output'] = network_blocks.init_linear_layer(
        subkeys[3], in_dim=num_heads * value_dim, out_dim=embed_dim,
        include_bias=False)['w']
    if use_gate:
      # [SCREEN] gate projection W_G (screening unit's replacement for a heavy FFN)
      params['g_w'] = network_blocks.init_linear_layer(
          subkeys[4], in_dim=embed_dim, out_dim=num_heads * value_dim, include_bias=False)['w']
    # [SCREEN] learnable knobs, one per head: acceptance width r, screening length w
    params['trim_rho'] = rho_init * jnp.ones((num_heads,))
    if use_softmask:
      # per-head, geometrically spaced -> multiscreen from the very first step
      params['mask_omega'] = jnp.asarray(omega_init, dtype=jnp.float32)
    if use_output_scale:
      # [SCREEN] s_O: learnable output scale, auto-calibrated at init
      params['out_scale'] = jnp.asarray(out_scale_init, dtype=jnp.float32)

    return params

  def apply(params: networks.ParamTree, single_electron_stream: jnp.ndarray,
            d_ee: jnp.ndarray) -> jnp.ndarray:
    """Computes screening attention.

    Args:
      params: Parameters for attention embeddings and screening knobs.
      single_electron_stream: Shape [num_electrons, embed_dim].
      d_ee: [SCREEN] pairwise physical electron-electron distances (periodic
        norm), shape [num_electrons, num_electrons]. Diagonal is 0.

    Returns:
      A projection of screening-weighted value projections.
      Shape [num_electrons, embed_dim]
    """

    # Projections for q, k, v.
    # Output shape: [..., index_dim, num_heads, attn_dim].
    q = jnp.dot(single_electron_stream, params['q_w']).reshape(num_electrons, num_heads, attn_dim)
    k = jnp.dot(single_electron_stream, params['k_w']).reshape(num_electrons, num_heads, attn_dim)
    v = jnp.dot(single_electron_stream, params['v_w']).reshape(num_electrons, num_heads, value_dim)

    # [SCREEN] 1. unit-length normalization -> bounded cosine similarity.
    qbar = _unit_length(q)
    kbar = _unit_length(k)
    vbar = _unit_length(v)
    # s: [num_heads, i, j]
    s = jnp.einsum('ihd,jhd->hij', qbar, kbar)

    # [SCREEN] 2. Trim with learnable acceptance width r_h in (0, 2).
    r = TRIM_MAX * jax.nn.sigmoid(params['trim_rho'])      # [FIX] r in (0, TRIM_MAX)
    r = r[:, None, None]
    alpha = jnp.square(jax.nn.relu(s - (1.0 - r)) / r)      # exactly 0 for s <= 1-r

    # [SCREEN] 3. Softmask on the PHYSICAL distance with learnable screening
    # length w_h (raised cosine; exactly 0 for d >= w; C^1 at the edge).
    if use_softmask:
      w = d_ref * jax.nn.softplus(params['mask_omega'])     # [num_heads]
      t = jnp.clip(d_ee[None, :, :] / w[:, None, None], 0.0, 1.0)
      m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
      weights = alpha * m
    else:
      weights = alpha

    # [PROBE] 解析用: params に '_diag_only'（ヘッドごとの 0/1）が入っていれば、
    # そのヘッドの重みを対角（自分自身）だけに落とす。1 体成分は残したまま
    # 2 体成分だけを抜けるので、「そのヘッドが運ぶ相関」の効きを測れる。
    _do = params.get('_diag_only', None)
    if _do is not None:
      _eye = jnp.eye(d_ee.shape[0])[None]
      _g = _do[:, None, None]
      weights = weights * (_g * _eye + (1.0 - _g))

    # [PROBE] 解析用のフック。学習では None のまま（何も起きない）。
    # 学習した窓の形はそのままに、届く範囲だけを R に縮めて
    # 「どこまでの距離が要るのか」を測るために使う。
    if CLIP_R is not None:
      _t = jnp.clip((d_ee[None, :, :] - (1.0 - CLIP_BETA) * CLIP_R)
                    / (CLIP_BETA * CLIP_R), 0.0, 1.0)
      weights = weights * 0.5 * (1.0 + jnp.cos(jnp.pi * _t))

    # [SCREEN] 4. denominator-free aggregation (no softmax).
    attn = jnp.einsum('hij,jhd->ihd', weights, vbar)        # [i, num_heads, value_dim]

    # [SCREEN] 5. TanhNorm: bounded output magnitude per head.
    if use_tanh_norm:
      eps = 1e-6
      n = jnp.sqrt(jnp.sum(attn * attn, axis=-1, keepdims=True) + eps * eps)
      attn = attn * jnp.tanh(n) / n

    # Concatenate attention matrix of all heads into a single vector.
    attn = attn.reshape(*single_electron_stream.shape[:-1], -1)   # [length, heads*value_dim]

    # [SCREEN] 6. gate (elementwise, per electron -> permutation equivariant).
    if use_gate:
      g = jnp.dot(single_electron_stream, params['g_w'])
      attn = attn * jnp.tanh(jax.nn.silu(g))

    # Apply a final projection to get the final embeddings.
    out = network_blocks.linear_layer(attn, params['attn_output'])

    # [SCREEN] 7. learnable output scale e^{s_O}. TanhNorm bounds |o| < 1 and
    # the gate shrinks it further, so without this the attention branch cannot
    # contribute at the magnitude the residual stream works at.
    if use_output_scale:
      out = jnp.exp(params['out_scale']) * out

    return out

  return init, apply


def make_mlp(num_perceptron_per_layer: int = 1, activation_fct_name: str = "TANH") ->...:
  """Construct MLP, with final linear projection to embedding size."""
  if activation_fct_name.upper() == "TANH":
    activation_fct = jnp.tanh
  elif activation_fct_name.upper() == "ELU":
    activation_fct = jax.nn.elu
  elif activation_fct_name.upper() == "GELU":
    activation_fct = jax.nn.gelu

  if num_perceptron_per_layer == 1:
    def init(key: chex.PRNGKey, mlp_dim: int, embed_dim: int,
            ) -> Sequence[networks.Param]:
      key1, key2 = jax.random.split(key)
      weight = (
          jax.random.normal(key1, shape=(embed_dim, mlp_dim)) /
          jnp.sqrt(float(embed_dim)))
      bias = jax.random.normal(key2, shape=(mlp_dim,))
      params = [{'w': weight, 'b': bias}]
      return params

    def apply(params: Sequence[networks.Param],
              inputs: jnp.ndarray) -> jnp.ndarray:
      return activation_fct(jnp.dot(inputs, params[0]['w']) + params[0]['b'])

    return init, apply
  else:
    def init(key: chex.PRNGKey, mlp_dim: int, embed_dim: int,
            ) -> Sequence[networks.Param]:
      params = []
      dims_one_in = [mlp_dim for _ in range(num_perceptron_per_layer)]
      dims_one_out = [mlp_dim for _ in range(num_perceptron_per_layer)]
      dims_one_in[0] = embed_dim
      # dims_one_out[-1] = embed_dim
      for i in range(len(dims_one_in)):
        key, subkey = jax.random.split(key)
        params.append(
            network_blocks.init_linear_layer(
                subkey,
                in_dim=dims_one_in[i],
                out_dim=dims_one_out[i],
                include_bias=True))
      return params

    def apply(params: Sequence[networks.Param],
              inputs: jnp.ndarray) -> jnp.ndarray:
      x = inputs
      for i in range(len(params)):
        x = activation_fct(jnp.dot(x, params[i]['w']) + params[i]['b'])
      return x

    return init, apply


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
                              use_output_scale: bool = True,
                              init_output_scale: float = 0.0) ->...:
  """Create a screening self-attention block. [SCREEN]"""
  attention_init, attention_apply = make_screening_attention(
      num_heads, mlp_dim, attn_dim, value_dim, num_electrons, d_ref,
      use_softmask=use_softmask, use_gate=use_gate, use_tanh_norm=use_tanh_norm,
      init_trim_r=init_trim_r, init_softmask_w_factor=init_softmask_w_factor,
      init_softmask_w_spread=init_softmask_w_spread,
      use_output_scale=use_output_scale,
      init_output_scale=init_output_scale)

  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()

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
            d_ee: jnp.ndarray) -> jnp.ndarray:
    # [SCREEN] d_ee is threaded through to every attention layer.
    x = single_electron_stream
    for layer in range(num_layers):
      attn_output = attention_apply(params['attention'][layer], x, d_ee)
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

    return self_attn_apply(params, x, d_ee)

  return init, apply


def make_fermi_net(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 16,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    # Screenformer-specific kwargs below.
    num_layers: int,
    num_heads: int,
    attn_dim: int,
    value_dim: int,
    mlp_dim: int,
    num_perceptrons_per_layer: int,
    use_layer_norm: bool,
    mlp_activation_fct: str,
    pbc_lattice: jnp.ndarray,
    use_softmask: bool = True,
    use_gate: bool = True,
    use_tanh_norm: bool = True,
    init_trim_r: float = 1.9,
    init_softmask_w_factor: float = 0.8,
    init_softmask_w_spread: float = 2.5,
    use_output_scale: bool = True,
    init_output_scale: float = 0.0,
) -> networks.Network:
  """Screenformer: stacked screening self-attention layers. [SCREEN]

  Includes standard envelope and determinants. Identical to
  CustomPsiformer.make_fermi_net except that the attention operator is the
  screening (multiscreen) unit and the extra [SCREEN] options below.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'none' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    num_layers: Number of stacked self-attention layers.
    num_heads: Number of self-attention heads.
    attn_dim: Embedding dimension per-head for self-attention.
    value_dim: Dimension of the value vectors of the self-attention mechanism
    mlp_dim: Hidden dimensions of the MLP.
    use_layer_norm: If true, use layer_norm on both attention and MLP.
    mlp_activation_fct: Which non-linearity to use in the MLP, one of TANH, ELU, GELU
    use_softmask: [SCREEN] enable the distance softmask.
    use_gate: [SCREEN] enable the screening-unit gate.
    use_tanh_norm: [SCREEN] enable TanhNorm on aggregated head vectors.
    init_trim_r: [SCREEN] initial Trim acceptance width.
    init_softmask_w_factor: [SCREEN] centre of the initial screening lengths,
      in units of the maximal periodic distance in the cell.
    init_softmask_w_spread: [SCREEN] multiscreen spread of the initial
      screening lengths across the heads of a layer.
    use_output_scale: [SCREEN] learnable output scale e^{s_O}.
    init_output_scale: [SCREEN] initial e^{s_O}; 0 means auto-calibrate.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, )

  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.NONE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = ScreenformerOptions(
      ndim=ndim,
      determinants=determinants,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      num_layers=num_layers,
      num_heads=num_heads,
      attn_dim=attn_dim,
      value_dim=value_dim,
      mlp_dim=mlp_dim,
      num_perceptrons_per_layer=num_perceptrons_per_layer,
      use_layer_norm=use_layer_norm,
      mlp_activation_fct=mlp_activation_fct,
      pbc_lattice=pbc_lattice,
      use_softmask=use_softmask,
      use_gate=use_gate,
      use_tanh_norm=use_tanh_norm,
      init_trim_r=init_trim_r,
      init_softmask_w_factor=init_softmask_w_factor,
      init_softmask_w_spread=init_softmask_w_spread,
      use_output_scale=use_output_scale,
      init_output_scale=init_output_scale,
  )  # pytype: disable=wrong-keyword-args

  screenformer_layers = make_screenformer_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=screenformer_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Screenformer.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    result = network_blocks.logdet_matmul(orbitals)
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )
