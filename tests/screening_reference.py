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


"""B1前のforward演算の固定参照（本体の評価関数は利用しない）。"""
import jax
import jax.numpy as jnp
from periodicwave import network_blocks
from tests.window_reference import screening_window

def _unit_length(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps * eps)
  return x / norm

def screening_attention(params, single_electron_stream, d_ee, pair_feat=None, *,
                        options, trim_max, trim_pow, win_shape, freeze_w,
                        clip_r, clip_beta):
  num_electrons = options['num_electrons']
  num_heads = options['num_heads']
  attn_dim = options['attn_dim']
  value_dim = options['value_dim']
  d_ref = options['d_ref']
  use_softmask = options['use_softmask']
  use_fixed_value = options['use_fixed_value']
  use_tanh_norm = options['use_tanh_norm']
  use_gate = options['use_gate']
  use_output_scale = options['use_output_scale']
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
  r = trim_max * jax.nn.sigmoid(params['trim_rho'])      # [FIX] r in (0, trim_max)
  r = r[:, None, None]
  alpha = jax.nn.relu(s - (1.0 - r)) / r                   # exactly 0 for s <= 1-r
  alpha = jnp.square(alpha) if trim_pow == 2 else jnp.square(jnp.square(alpha))

  # [SCREEN] 3. Softmask on the PHYSICAL distance with learnable screening
  # length w_h (raised cosine; exactly 0 for d >= w; C^1 at the edge).
  if use_softmask:
    m = screening_window(
        params, d_ee, d_ref, win_shape=win_shape, freeze_w=freeze_w)
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
  if clip_r is not None:
    _t = jnp.clip((d_ee[None, :, :] - (1.0 - clip_beta) * clip_r)
                  / (clip_beta * clip_r), 0.0, 1.0)
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
