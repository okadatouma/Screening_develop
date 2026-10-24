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
from periodicwave import attention
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

# Keep the original public name for analysis scripts and external callers.
make_layer_norm = network_blocks.make_layer_norm


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


# Keep helper names used by existing analysis scripts.
_smooth7 = attention._smooth7


# [SCREEN] ガウスでぼかした輪切りは erf を使う。folx の前向きラプラシアンに erf を要素ごとの関数として
# 登録しておく（未登録だと folx は erf の入力全体についてヘッセ行列を作ってしまう）。
try:
  import folx as _folx
  _folx.register_function(jax.lax.erf_p, _folx.wrap_forward_laplacian(jax.lax.erf, in_axes=()))
except Exception:  # folx の無い環境（解析だけなど）ではそのまま
  pass


_ring_steps = attention._ring_steps


# [SCREEN] Helper: unit-length normalization along the last axis.
# The small eps keeps the map twice differentiable at the origin.
_unit_length = attention._unit_length


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

  # 既存factoryの静的な構成。学習パラメータやcheckpointには追加しない。
  attention_options = dict(
      num_electrons=num_electrons,
      num_heads=num_heads,
      attn_dim=attn_dim,
      value_dim=value_dim,
      d_ref=d_ref,
      use_softmask=use_softmask,
      use_fixed_value=use_fixed_value,
      use_tanh_norm=use_tanh_norm,
      use_gate=use_gate,
      use_output_scale=use_output_scale,
  )

  def apply(params: networks.ParamTree, single_electron_stream: jnp.ndarray,
            d_ee: jnp.ndarray, pair_feat: jnp.ndarray = None, *,
            return_aux: bool = False):
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
    if return_aux:
      return attention.screening_attention(
          params, single_electron_stream, d_ee, pair_feat, options=attention_options,
          trim_max=TRIM_MAX, trim_pow=TRIM_POW, win_shape=WIN_SHAPE,
          freeze_w=FREEZE_W, clip_r=CLIP_R, clip_beta=CLIP_BETA, return_aux=True)
    return attention.screening_attention(
        params, single_electron_stream, d_ee, pair_feat, options=attention_options,
        trim_max=TRIM_MAX, trim_pow=TRIM_POW, win_shape=WIN_SHAPE,
        freeze_w=FREEZE_W, clip_r=CLIP_R, clip_beta=CLIP_BETA)

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
            d_ee: jnp.ndarray, pair_feat: jnp.ndarray = None, *,
            return_aux: bool = False):
    """通常は最終X、return_aux=Trueなら(final_X, layers)を返す。

    layersは層順のtuple。各辞書のinput/outputと残差・LN・MLP出力は[N,E]。
    attentionにはB2の辞書、attention_outputには実際に残差へ加えた[N,E]を保持。
    LNなしではln1_output/ln2_outputは恒等。次層は常に本体のxを使う。
    return_auxはJITで静的指定。通常モードは辞書・リストを構築しない。
    """
    if return_aux:
      layers = []
    # [SCREEN] d_ee（と、固定 value を使うときは対の特徴）を各層へ通す。
    x = single_electron_stream
    for layer in range(num_layers):
      if return_aux:
        record = dict(input=x)
        attn_output, attn_aux = attention_apply(
            params['attention'][layer], x, d_ee, pair_feat, return_aux=True)
        record.update(attention=attn_aux, attention_output=attn_output)
      else:
        attn_output = attention_apply(params['attention'][layer], x, d_ee, pair_feat)
      # Residual + optional LayerNorm.
      x = x + attn_output
      if return_aux:
        record['attention_residual'] = x
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)
      if return_aux:
        record['ln1_output'] = x

      # MLP
      mlp_output = mlp_apply(params['mlp'][layer], x)
      if return_aux:
        record['mlp_output'] = mlp_output

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if return_aux:
        record['mlp_residual'] = x
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][1], x)
      if return_aux:
        record.update(ln2_output=x, output=x)
        layers.append(record)

    if return_aux:
      return x, tuple(layers)
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

    # Map to Attention dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_input_embedding(
        subkey, one_electron_feature_dim, options.mlp_dim
    )

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
      return_aux: bool = False,
  ):
    """Applies the Screenformer interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permuation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.
      return_aux: 静的Trueなら(final_X, layers)を返す。内容はブロックapply参照。

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

    # Spin is part of the equivariant one-electron embedding.
    x = network_blocks.apply_input_embedding(params['embed'], ae_features, spins)

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

    if return_aux:
      return self_attn_apply(params, x, d_ee, pair_feat, return_aux=True)
    return self_attn_apply(params, x, d_ee, pair_feat)

  return init, apply


def make_screenformer_diagnostic(nspins, options):
  """Build a diagnostic trunk, independently JIT-compiled from Network.apply.

  The returned evaluate(params, pos, spins, atoms, charges) accepts the unchanged
  full network PyTree and returns (final_X, layers). X and layer boundaries have
  shape [N, E]; layers[i]['attention'] contains the B2 quantities. In each layer,
  attention_residual / ln1_output are before / after the first LayerNorm;
  ln1_output / mlp_output are MLP input / output; mlp_residual / ln2_output are
  before / after the second LayerNorm. With no LayerNorm these pairs coincide.

  Uses the same embedding and Transformer loop as Production. Extra outputs can
  change XLA fusion: these are diagnostic reevaluations, NOT Production-internal
  bit patterns. Do not use them for VMC/optimization, or compile this together
  with Network.apply. Globals are read at trace time as in the ordinary trunk.
  This interface stops before orbitals; it also works with an HF-connected trunk
  without trying to reconstruct that wavefunction.
  """
  _, apply = make_screenformer_layers(nspins, 0, options)
  geometry = networks.make_periodic_construct_input_features(options.pbc_lattice)

  def evaluate(params, pos, spins, atoms, charges):
    ae, ee, r_ae, r_ee = geometry(pos, atoms, ndim=options.ndim)
    return apply(params['layers'], ae=ae, ee=ee, r_ae=r_ae, r_ee=r_ee,
                 spins=spins, charges=charges, return_aux=True)

  return evaluate


def screening_diagnostic_status(final_x, *, orbitals=None, phase_logabs=None):
  """Check numerical rank separately from the diagnostic forward/JIT.

  Rank threshold is max(matrix.shape[-2:]) * eps of that array's real dtype.
  A small feature singular value is a warning; a nonuniform envelope can change
  orbital rank. Pass orbital matrices [ndet, Nspin, Nspin] (one per spin channel)
  for a wavefunction check. At N=E=4, initial final LayerNorm with zero offset,
  null envelope and no orbital bias, all determinants are rank deficient.

  Flags identify ill-conditioning, not a proof of stability when False. In
  particular determinant cancellation can occur even with full-rank orbitals.
  Missing wavefunction inputs are explicitly marked as unchecked (None), never
  silently certified. Does not replace, clip, detach or repair any values.
  """
  singular_values = jnp.linalg.svd(final_x, compute_uv=False)
  threshold = max(final_x.shape) * jnp.finfo(final_x.dtype).eps
  ratio = singular_values[-1] / jnp.maximum(singular_values[0], jnp.finfo(final_x.dtype).tiny)
  feature_deficient = (final_x.shape[0] > final_x.shape[1]) | (ratio <= threshold)
  state = dict(feature_singular_values=singular_values, feature_rcond=ratio,
               feature_rank_deficient=feature_deficient,
               finite_features=jnp.all(jnp.isfinite(final_x)),
               orbital_rcond=None, singular_determinants=None,
               all_determinants_rank_deficient=None, finite_phase_logabs=None,
               orbital_check_performed=orbitals is not None,
               wavefunction_unstable_or_undefined=None)
  unstable = None
  if orbitals is not None:
    ratios, bad = [], []
    for matrix in orbitals:
      s = jnp.linalg.svd(matrix, compute_uv=False)
      eps = jnp.finfo(matrix.real.dtype).eps
      r = s[..., -1] / jnp.maximum(s[..., 0], jnp.finfo(matrix.real.dtype).tiny)
      ratios.append(r)
      bad.append((r <= max(matrix.shape[-2:]) * eps) | ~jnp.all(jnp.isfinite(matrix), axis=(-2, -1)))
    singular = jnp.any(jnp.stack(bad), axis=0)
    state.update(orbital_rcond=tuple(ratios), singular_determinants=singular,
                 all_determinants_rank_deficient=jnp.all(singular))
    unstable = jnp.any(singular)
  if phase_logabs is not None:
    finite = jnp.all(jnp.isfinite(jnp.asarray(phase_logabs)))
    state['finite_phase_logabs'] = finite
    unstable = ~finite if unstable is None else unstable | ~finite
  state['wavefunction_unstable_or_undefined'] = unstable
  state['local_energy_unstable_or_undefined'] = unstable
  return state


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
