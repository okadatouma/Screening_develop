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

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import os

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
  init_softmask_centres: Any = None         # [SCREEN] (centres, sigmas) / d_ref。帯域窓
  init_softmask_rings: Any = None           # [SCREEN] (edges, delta, p) / d_ref。全体窓 g の輪切り
  use_fixed_value: bool = False             # [SCREEN] value を対の固定特徴 e(r_ij) にする
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


# [SCREEN] PW_FREEZE_W=1 にすると窓 w_h を学習させず、初期値のまま固定する。
# 窓を「目盛り」として据え置き、ヘッドごとの寄与率の移り変わりだけを読む使い方。
FREEZE_W = os.environ.get("PW_FREEZE_W", "0").lower() not in ("0", "", "false")

# [λ-jellium] 窓と Trim のなめらかさ。既定は従来どおり（余弦の窓、Trim は relu の 2 乗）。
#   PW_WIN_SHAPE=smooth7 : 入れ子窓を m = 1 - S_7(d/w)、S_7(x) = 35x^4 - 84x^5 + 70x^6 - 20x^7 にする
#                          （d = w で 3 階微分まで連続、max|m''| = 7.5/w^2。余弦の窓は 2 階微分が pi^2/2w^2 跳ぶ）
#   PW_TRIM_POW=4        : Trim を relu の 4 乗にする（s = 1 - r で 3 階微分まで連続）
WIN_SHAPE = os.environ.get("PW_WIN_SHAPE", "cos").lower()
TRIM_POW = int(os.environ.get("PW_TRIM_POW", "2"))
if WIN_SHAPE not in ("cos", "smooth7"):
  raise ValueError("PW_WIN_SHAPE は cos か smooth7")
if TRIM_POW not in (2, 4):
  raise ValueError("PW_TRIM_POW は 2 か 4")


def _smooth7(x):
  """S_7(x) = 35x^4 - 84x^5 + 70x^6 - 20x^7（x は [0, 1] に clip 済み）。S_7(0) = 0、S_7(1) = 1、両端で 3 階微分まで 0。"""
  x2 = x * x
  return x2 * x2 * (35.0 + x * (-84.0 + x * (70.0 - 20.0 * x)))


# [SCREEN] ガウスでぼかした輪切りは erf を使う。folx の前向きラプラシアンに erf を要素ごとの関数として
# 登録しておく（未登録だと folx は erf の入力全体についてヘッセ行列を作ってしまう）。
try:
  import folx as _folx
  _folx.register_function(jax.lax.erf_p, _folx.wrap_forward_laplacian(jax.lax.erf, in_axes=()))
except Exception:  # folx の無い環境（解析だけなど）ではそのまま
  pass


def _ring_steps(d_ee, b, dl, gauss=False):
  """輪切りの階段 T_k(d)（k = 0..H、T_0 = 1、T_H = 0）。窓は m_h = g (T_h - T_{h+1})。

  gauss=False：半周期の sin の階段（遷移の半幅 dl）。
  gauss=True ：輪の指示関数をガウス核（標準偏差 dl）でぼかしたもの。
      T_k(d) = Phi((d - b_k)/dl) + Phi((-d - b_k)/dl)
  第 2 項は d = 0 での折り返しで、T_k は d の偶関数（m_h'(0) = 0）。どの階数の微分も連続。
  どちらも差が telescoping するので sum_h m_h = g は厳密。
  """
  if gauss:
    s2 = dl * jnp.sqrt(2.0)
    bb = b[:, None, None]
    th = 1.0 + 0.5 * (jax.lax.erf((d_ee[None, :, :] - bb) / s2)
                      + jax.lax.erf((-d_ee[None, :, :] - bb) / s2))
  else:
    u = (d_ee[None, :, :] - b[:, None, None]) / dl
    th = 0.5 * (1.0 + jnp.sin(0.5 * jnp.pi * jnp.clip(u, -1.0, 1.0)))
  return th.at[0].set(1.0).at[-1].set(0.0)             # 端の輪は片側だけ切る


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
                             init_softmask_centres=None,
                             init_softmask_rings=None,
                             use_fixed_value=False,
                             pair_dim=0,
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

  # [SCREEN] 帯域窓を使うときは、中心 c_h / d_ref と半値幅 sigma_h / d_ref を
  # 呼び出し側から (centres, sigmas) で渡す。sigma を中心の間隔に取ると
  # raised cosine は 1 の分割になる。
  centre_init = None
  if init_softmask_centres is not None:
    if isinstance(init_softmask_centres, str):
      # config.json に保存すると numpy 配列は "unserializable object: [...]" という
      # 文字列になる。そこから復元はできないので、黙って単調窓に落とさず止める。
      # 学習済みモデルを組み直すときは checkpoint の mask_centre を使うこと。
      raise ValueError(
          "init_softmask_centres が文字列です（config.json 由来）。"
          "配列の組 (centres, sigmas)/d_ref を渡すか、checkpoint の "
          "'mask_centre' から復元してください。")
    _c, _sg = init_softmask_centres
    centre_init = np.asarray(_c, dtype=np.float64)
    omega_init = np.log(np.expm1(np.clip(np.asarray(_sg, dtype=np.float64), 1e-3, None)))

  # [SCREEN] 全体窓の輪切り。g(d) = (1 + cos(pi (d/d_ref)^p))/2 を 1 本だけ作り
  # （g(0) = 1、g(d_ref) = 0）、境界 b_0 = 0 < b_1 < ... < b_H の輪に切って配る。
  #   m_h(d) = g(d) [Theta(d - b_h) - Theta(d - b_{h+1})]
  # Theta は幅 2 delta のなめらかな階段。差が telescoping するので
  #   sum_h m_h(d) = g(d)   （境界の置き方によらず厳密）
  # 輪の芯（遷移帯の外）では m_h = g そのもの = 近い相手ほど重い。
  rings_init = None
  rings_kernel = 'sin'
  if init_softmask_rings is not None:
    if isinstance(init_softmask_rings, str):
      raise ValueError("init_softmask_rings が文字列です（config.json 由来）。"
                       "(edges, delta, p) を渡すこと。")
    _b, _dl, _pp = init_softmask_rings[:3]
    # 4 番目の要素（任意）は階段の形：'sin'（既定、半周期の sin）か 'gauss'（ガウス核でぼかす）。
    # gauss のとき _dl はガウス核の標準偏差 sigma / d_ref。
    rings_kernel = str(init_softmask_rings[3]) if len(init_softmask_rings) > 3 else 'sin'
    if rings_kernel not in ('sin', 'gauss'):
      raise ValueError("輪切りの核は 'sin' か 'gauss'（いま %r）" % rings_kernel)
    rings_init = np.asarray(_b, dtype=np.float64)
    if rings_init.shape[0] != num_heads + 1:
      raise ValueError("edges は num_heads + 1 個必要（いま %d 個、heads=%d）"
                       % (rings_init.shape[0], num_heads))
    envp_init = float(_pp)
    omega_init = np.full(num_heads, np.log(np.expm1(max(float(_dl), 1e-4))))

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
      (np.power(np.maximum(_s_grid - (1.0 - init_trim_r), 0.0) / init_trim_r, TRIM_POW) * _pdf).sum())
  # Empirical gain of the remaining stages (TanhNorm, gate, W_O) relative to
  # the naive estimate. Calibrated by measuring |attention output| at
  # initialisation against softmax attention on the same inputs, so that the
  # screening branch and the softmax branch start at the same magnitude and
  # the two ansatze are compared on equal footing.
  _branch_gain = 0.85 if use_gate else 2.4
  if use_fixed_value:
    # 固定 value は単位長ではないので、TanhNorm 前の大きさが違う。TanhNorm が
    # 上から抑えるため過大にはならないが、初期の枝が小さすぎないよう合わせる。
    _branch_gain = _branch_gain * np.sqrt(max(float(pair_dim), 1.0) / float(value_dim))
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
    # [SCREEN] use_fixed_value のときヘッド出力の次元は pair_dim（value は作らない）
    _vd = pair_dim if use_fixed_value else value_dim
    if not use_fixed_value:
      params['v_w'] = network_blocks.init_linear_layer(
          subkeys[2], in_dim=embed_dim, out_dim=num_heads * value_dim, include_bias=False)['w']
    params['attn_output'] = network_blocks.init_linear_layer(
        subkeys[3], in_dim=num_heads * _vd, out_dim=embed_dim,
        include_bias=False)['w']
    if use_gate:
      # [SCREEN] gate projection W_G (screening unit's replacement for a heavy FFN)
      params['g_w'] = network_blocks.init_linear_layer(
          subkeys[4], in_dim=embed_dim, out_dim=num_heads * _vd, include_bias=False)['w']
    # [SCREEN] learnable knobs, one per head: acceptance width r, screening length w
    params['trim_rho'] = rho_init * jnp.ones((num_heads,))
    if use_softmask:
      # per-head, geometrically spaced -> multiscreen from the very first step
      params['mask_omega'] = jnp.asarray(omega_init, dtype=jnp.float32)
      if centre_init is not None:
        # [SCREEN] 帯域窓の中心 c_h / d_ref。生のまま持つので 0 も表現できる
        # （0 = 従来の単調減少窓）。
        params['mask_centre'] = jnp.asarray(centre_init, dtype=jnp.float32)
      if rings_init is not None:
        params['mask_rings'] = jnp.asarray(rings_init, dtype=jnp.float32)  # b_h / d_ref
        params['mask_envp'] = jnp.asarray(envp_init, dtype=jnp.float32)    # 指数 p
        if rings_kernel == 'gauss':
          params['mask_gauss'] = jnp.asarray(1.0, dtype=jnp.float32)       # ガウス核の目印
    if use_output_scale:
      # [SCREEN] s_O: learnable output scale, auto-calibrated at init
      params['out_scale'] = jnp.asarray(out_scale_init, dtype=jnp.float32)

    return params

  def apply(params: networks.ParamTree, single_electron_stream: jnp.ndarray,
            d_ee: jnp.ndarray, pair_feat: jnp.ndarray = None) -> jnp.ndarray:
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
    v = (None if use_fixed_value
         else jnp.dot(single_electron_stream, params['v_w']).reshape(
             num_electrons, num_heads, value_dim))

    # [SCREEN] 1. unit-length normalization -> bounded cosine similarity.
    qbar = _unit_length(q)
    kbar = _unit_length(k)
    vbar = None if v is None else _unit_length(v)
    # [PROBE] 解析用: params に '_v_mode'（ヘッドごとの 0/1）があれば、そのヘッドの
    # value を電子平均で置き換える。value が電子ごとに違うことに意味があるか
    # （＝ヘッドが「誰を見るか」だけでなく「何を運ぶか」も使っているか）を測る。
    # 学習では None のまま（何も起きない）。
    _vm = params.get('_v_mode', None)
    if _vm is not None and vbar is not None:
      _vmean = jnp.mean(vbar, axis=0, keepdims=True)              # [1, H, VD]
      vbar = vbar * (1.0 - _vm)[None, :, None] + _vmean * _vm[None, :, None]
    # s: [num_heads, i, j]
    s = jnp.einsum('ihd,jhd->hij', qbar, kbar)

    # [SCREEN] 2. Trim with learnable acceptance width r_h in (0, 2).
    r = TRIM_MAX * jax.nn.sigmoid(params['trim_rho'])      # [FIX] r in (0, TRIM_MAX)
    r = r[:, None, None]
    alpha = jax.nn.relu(s - (1.0 - r)) / r                   # exactly 0 for s <= 1-r
    alpha = jnp.square(alpha) if TRIM_POW == 2 else jnp.square(jnp.square(alpha))

    # [SCREEN] 3. Softmask on the PHYSICAL distance with learnable screening
    # length w_h (raised cosine; exactly 0 for d >= w; C^1 at the edge).
    if use_softmask:
      _om = params['mask_omega']
      if FREEZE_W:
        _om = jax.lax.stop_gradient(_om)   # 窓を固定（目盛りとして使う）
      w = d_ref * jax.nn.softplus(_om)                     # [num_heads] 半値幅 sigma_h
      # [SCREEN] 帯窓。内側の端 a_h を入れると、ヘッド h は自分の輪の中で
      # 重みが 1 から 0 へ落ちる（近い相手ほど重い）形になる。
      #   m_h(d) = (1 + cos(pi * clip((d - a_h)/Delta_h, -1, 1))) / 2
      #   台は [a_h - Delta_h, a_h + Delta_h]、外は厳密に 0、両端で m' = 0（C^1）。
      # a_h = 0 のとき clip の下側は効かず、従来の単調減少窓と厳密に一致する
      # （つまり単調窓はこの族の h = 0 そのもの）。
      # a_h = h * Delta と等間隔に置くと raised cosine は 1 の分割になり、
      #   sum_h m_h(d) = 1   ([a_0, a_{H-1}] で厳密)
      # Delta = d_ref / (H - 1) に取れば a_{H-1} = d_ref なので、届く範囲すべてで 1。
      # となるので q_h がそのまま動径分布になる。内側の裾は隣のヘッドの
      # 落ち方とちょうど相補的なので、情報としては重複しない。
      # 内向きの裾を切って [a_h, a_{h+1}] だけに台を持たせると d = a_h で
      # 0 -> 1 の跳びが出る（decreasing かつ compact support なら必ず不連続）。
      # log psi の 2 階微分が発散するのでそれは採らない。
      if 'mask_rings' in params:
        _b = params['mask_rings']
        if FREEZE_W:
          _b = jax.lax.stop_gradient(_b)
        _pp = jax.lax.stop_gradient(params['mask_envp'])   # 形の指数は常に固定
        b = d_ref * _b                                     # [num_heads + 1]
        dl = w[0]                                          # 遷移の半幅 delta（gauss なら標準偏差 sigma）
        if 'mask_gauss' in params:
          th = _ring_steps(d_ee, b, dl * jax.lax.stop_gradient(params['mask_gauss']), gauss=True)
        else:
          th = _ring_steps(d_ee, b, dl)
        P = th[:-1] - th[1:]                               # 1 の分割（telescoping）
        x = jnp.clip(d_ee / d_ref, 0.0, 1.0)
        # x^p を素直に書くと x = 0（対角 d_ii = 0）で 2 階微分が 0 * inf = NaN になる
        # （p = 1 で実際に全配置 NaN）。両側 where で安全側の枝からだけ微分を取る。
        _pos = x > 0.0
        xp = jnp.where(_pos, jnp.power(jnp.where(_pos, x, 1.0), _pp), 0.0)
        g = 0.5 * (1.0 + jnp.cos(jnp.pi * xp))             # 全体で 1 本の窓
        m = P * g[None, :, :]
      elif 'mask_centre' in params:
        _ct = params['mask_centre']
        if FREEZE_W:
          _ct = jax.lax.stop_gradient(_ct)
        c = d_ref * _ct                                    # [num_heads] 生の値（0 も負も可）
        t = jnp.clip((d_ee[None, :, :] - c[:, None, None]) / w[:, None, None], -1.0, 1.0)
        m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
      else:
        t = jnp.clip(d_ee[None, :, :] / w[:, None, None], 0.0, 1.0)
        if WIN_SHAPE == "smooth7":
          m = 1.0 - _smooth7(t)
        else:
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
    if use_fixed_value:
      # [SCREEN] value を学習させず、対の相対座標の固定特徴 e(r_ij) を運ぶ。
      # 重み W_ij が「誰をどれだけ見るか」、e(r_ij) が「何を運ぶか」。
      # 全ヘッドで同じ e を使うので、ヘッドを分けているのは窓だけになる。
      # j = i は落とす。e(r_ii) = [0, 0, 1] は定数（1 体項）で、しかも帯窓では
      # a_0 = 0 のヘッドだけがそれを拾うので、q_h の動径分布としての意味が濁る。
      _w = weights * (1.0 - jnp.eye(d_ee.shape[0]))[None]
      attn = jnp.einsum('hij,ijd->ihd', _w, pair_feat)      # [i, num_heads, pair_dim]
    else:
      attn = jnp.einsum('hij,jhd->ihd', weights, vbar)      # [i, num_heads, value_dim]

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
    init_softmask_centres=None,
    init_softmask_rings=None,
    use_fixed_value: bool = False,
    use_output_scale: bool = True,
    init_output_scale: float = 0.0,
    lj_hf_path: str = '',
    lj_lam: float = 0.0,
    lj_cusps=None,
    lj_net_scale: float = 0.05,
    lj_jastrow: bool = True,
) -> networks.Network:
  """Screenformer: stacked screening self-attention layers. [SCREEN]

  [λ-jellium] lj_hf_path が空でなければ、軌道を lj_orbitals.make_lj_orbitals（HF の Bloch スピノル
  + 擬スピンで重みを分けた網の補正、擬スピンの組ごとの Jastrow）にする。

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
      init_softmask_centres=init_softmask_centres,
      init_softmask_rings=init_softmask_rings,
      use_fixed_value=use_fixed_value,
      use_output_scale=use_output_scale,
      init_output_scale=init_output_scale,
  )  # pytype: disable=wrong-keyword-args

  screenformer_layers = make_screenformer_layers(nspins, charges.shape[0], options)

  if lj_hf_path:
    from periodicwave import lj_orbitals
    _k, _G, _c = lj_orbitals.load_hf_table(lj_hf_path)
    print(f"[λJ] HF の軌道 {lj_hf_path}：k {_k.shape[0]} 点、G {_G.shape[0]} 本、lam = {lj_lam}、"
          f"カスプ (++, --, +-) = {tuple(lj_cusps) if lj_cusps is not None else None}、網の初期の大きさ {lj_net_scale}、"
          f"Jastrow {'あり' if lj_jastrow else 'なし'}")
    orbitals_init, orbitals_apply = lj_orbitals.make_lj_orbitals(
        nspins, charges, options, screenformer_layers, _k, _G, _c, float(lj_lam),
        tuple(lj_cusps) if lj_cusps is not None else (0.0, 0.0, 0.0),
        net_scale=float(lj_net_scale), use_jastrow=bool(lj_jastrow))
  else:
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
