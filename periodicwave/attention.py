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


# WindowとScreening評価をCustomScreenformer.pyから移動。元の著作権・ライセンスを継承。
"""Screening Attentionの数式。初期化・幾何・Transformerは呼出元の責務。"""

from periodicwave import network_blocks

import jax
import jax.numpy as jnp


def _smooth7(x):
  """S_7(x) = 35x^4 - 84x^5 + 70x^6 - 20x^7（x は [0, 1] に clip 済み）。S_7(0) = 0、S_7(1) = 1、両端で 3 階微分まで 0。"""
  x2 = x * x
  return x2 * x2 * (35.0 + x * (-84.0 + x * (70.0 - 20.0 * x)))


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


def screening_window(params, d_ee, d_ref, *, win_shape, freeze_w):
  """Screeningの距離窓 m_h(d_ij) を返す（shape [H, N, N]）。

  params: mask_omega [H] は w=d_ref*softplus(omega) の幅。
    mask_centre [H] があれば中心 c=d_ref*centre の帯窓。
    mask_rings [H+1] があれば境界 b=d_ref*rings の輪切り窓。
    mask_envp（scalar）は全体窓の指数、mask_gauss（scalar）があればGaussian階段。
  d_ee: 周期ノルムの距離 [N,N]。d_ref: 既存の基準距離（同じ長さ単位）。
  win_shape: 単調窓だけに適用する 'cos' または 'smooth7'。
  freeze_w: 幅・中心・輪境界への勾配を止める。座標への勾配は止めない。
    全体窓の指数とGaussian目印への勾配は常に停止する。

  単調窓: m=(1+cos(pi*clip(d/w,0,1)))/2 または 1-S7(clip(d/w,0,1))。
  帯窓: m=(1+cos(pi*clip((d-c)/w,-1,1)))/2。
  輪切り: m_h=(T_h-T_{h+1})*g, g=(1+cos(pi*clip(d/d_ref,0,1)^p))/2。
  余弦の接続はC1、smooth7はC3。境界でのclipの微分規約と、
  原点の累乗保護を維持する。Windowなしは呼出元でこの関数を通らない。
  JITではwin_shapeとfreeze_wを静的設定として渡す。
  """
  _om = params['mask_omega']
  if freeze_w:
    _om = jax.lax.stop_gradient(_om)   # 窓を固定（目盛りとして使う）
  w = d_ref * jax.nn.softplus(_om)                     # [num_heads] 半値幅 sigma_h
  if 'mask_rings' in params:
    _b = params['mask_rings']
    if freeze_w:
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
    if freeze_w:
      _ct = jax.lax.stop_gradient(_ct)
    c = d_ref * _ct                                    # [num_heads] 生の値（0 も負も可）
    t = jnp.clip((d_ee[None, :, :] - c[:, None, None]) / w[:, None, None], -1.0, 1.0)
    m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
  else:
    t = jnp.clip(d_ee[None, :, :] / w[:, None, None], 0.0, 1.0)
    if win_shape == "smooth7":
      m = 1.0 - _smooth7(t)
    else:
      m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
  return m


def _unit_length(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps * eps)
  return x / norm


def screening_attention(params, single_electron_stream, d_ee, pair_feat=None, *,
                        options, trim_max, trim_pow, win_shape, freeze_w,
                        clip_r, clip_beta, return_aux=False,
                        window=None, qk_stream=None, gate_stream=None):
  """Screening Attentionの出力 [N, embed_dim] を評価する。

  x=single_electron_stream は [N,embed_dim]、d_eeは周期距離 [N,N]。
  pair_feat [N,N,pair_dim] は固定Value時だけ使用する。
  optionsは既存factoryの通常の辞書。num_electrons/num_heads/attn_dim/
  value_dimはreshapeの寸法、d_refはWindow用の基準距離。
  use_softmask/use_fixed_value/use_tanh_norm/use_gate/use_output_scaleは既存フラグ。
  この関数では初期化や設定の取得を行わない。
  実験用の任意指定: window[H,N,N]、qk_stream/gate_stream[N,E]。
  Noneなら従来のWindowとXを使う。Valueは常に元のXから生成する。
  trim_max/trim_pow、win_shape/freeze_w、clip_r/clip_betaは呼出元が
  apply実行時（JITならtrace時）に読む既存設定。静的なPython値として渡す。

  qbar,kbar,vbar → s=qbar・kbar → alpha=Trim(s) → weights=alpha*m(d)。
  行和で割らずValueを集約し、TanhNorm → Gate → W_O → exp(out_scale)。
  通常Valueは自己項を含み、固定Valueは対角を除く。
  return_aux=Falseは従来の出力配列のみ。Trueは(out, aux)を返す。
  JITではreturn_auxを静的に指定する。通常のNetwork APIは変更しない。

  auxは計算済み配列への参照（再計算・stop_gradientなし）:
    q/k [N,H,attn_dim], v [N,H,value_dim]、各 *_normalized は長さ正規化後。
    v_effective は_v_mode介入後、pair_features [N,N,pair_dim]は固定Value入力。
    similarity/alpha/window/weights_window/weights/aggregation_weights [H,N,N]。
    weights_window=alpha*m（窓なしならalpha）、weightsは_diag_only/clip介入後、
    aggregation_weightsはさらに固定Valueの対角除去を反映した集約用重み。
    aggregate/tanh_output [N,H,D] は集約直後/TanhNorm後（DはValue次元）。
    gate_output [N,H*D] は連結・Gate後、projected_output [N,embed_dim]はW_O後。
  最終出力outは出力倍率適用後。無効な段階は恒等、存在しないv/window/
  pair_featuresはNone。v_normalizedは介入前、v_effectiveは集約に使う値。
  内部量を返すモードは配列を保持するため、通常よりメモリを使用する。
  """
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

  # 1. Q/K/V: XWを[N,H,D]に分ける。固定Value時はVを作らない。
  qk_input = single_electron_stream if qk_stream is None else qk_stream
  q = jnp.dot(qk_input, params['q_w']).reshape(num_electrons, num_heads, attn_dim)
  k = jnp.dot(qk_input, params['k_w']).reshape(num_electrons, num_heads, attn_dim)
  v = (None if use_fixed_value
       else jnp.dot(single_electron_stream, params['v_w']).reshape(
           num_electrons, num_heads, value_dim))

  # 2. Q/K/Vをeps付きの長さで正規化する。
  qbar = _unit_length(q)
  kbar = _unit_length(k)
  vbar = None if v is None else _unit_length(v)
  if return_aux:
    aux = dict(q=q, k=k, v=v, q_normalized=qbar, k_normalized=kbar,
               v_normalized=vbar, pair_features=pair_feat if use_fixed_value else None)
  # [PROBE] 解析用: params に '_v_mode'（ヘッドごとの 0/1）があれば、そのヘッドの
  # value を電子平均で置き換える。value が電子ごとに違うことに意味があるか
  # （＝ヘッドが「誰を見るか」だけでなく「何を運ぶか」も使っているか）を測る。
  # 学習では None のまま（何も起きない）。
  _vm = params.get('_v_mode', None)
  if _vm is not None and vbar is not None:
    _vmean = jnp.mean(vbar, axis=0, keepdims=True)              # [1, H, VD]
    vbar = vbar * (1.0 - _vm)[None, :, None] + _vmean * _vm[None, :, None]
  # 3. Similarity s_ij = qbar_i・kbar_j、shape [H,N,N]。
  s = jnp.einsum('ihd,jhd->hij', qbar, kbar)

  # 4. Trim: alpha = (relu(s - (1-r))/r)^p、r in (0,trim_max)。
  r = trim_max * jax.nn.sigmoid(params['trim_rho'])      # [FIX] r in (0, trim_max)
  r = r[:, None, None]
  alpha = jax.nn.relu(s - (1.0 - r)) / r                   # exactly 0 for s <= 1-r
  alpha = jnp.square(alpha) if trim_pow == 2 else jnp.square(jnp.square(alpha))

  # 5–6. 周期距離のWindowと重み weights=alpha*m（行和で割らない）。
  if use_softmask:
    m = (screening_window(params, d_ee, d_ref, win_shape=win_shape, freeze_w=freeze_w)
         if window is None else window)
    weights = alpha * m
    if return_aux:
      aux['window'] = m
  else:
    weights = alpha
    if return_aux:
      aux['window'] = None
  if return_aux:
    aux.update(similarity=s, alpha=alpha, weights_window=weights, v_effective=vbar)

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

  if return_aux:
    aux['weights'] = weights

  # 7. Valueの重み付き集約（分母なし）。固定Valueのみ自己項を除く。
  if use_fixed_value:
    # [SCREEN] value を学習させず、対の相対座標の固定特徴 e(r_ij) を運ぶ。
    # 重み W_ij が「誰をどれだけ見るか」、e(r_ij) が「何を運ぶか」。
    # 全ヘッドで同じ e を使うので、ヘッドを分けているのは窓だけになる。
    # j = i は落とす。e(r_ii) = [0, 0, 1] は定数（1 体項）で、しかも帯窓では
    # a_0 = 0 のヘッドだけがそれを拾うので、q_h の動径分布としての意味が濁る。
    _w = weights * (1.0 - jnp.eye(d_ee.shape[0]))[None]
    attn = jnp.einsum('hij,ijd->ihd', _w, pair_feat)      # [i, num_heads, pair_dim]
    if return_aux:
      aux['aggregation_weights'] = _w
  else:
    attn = jnp.einsum('hij,jhd->ihd', weights, vbar)      # [i, num_heads, value_dim]
    if return_aux:
      aux['aggregation_weights'] = weights
  if return_aux:
    aux['aggregate'] = attn

  # 8. TanhNorm: headごとの集約ベクトルの長さを抑える。
  if use_tanh_norm:
    eps = 1e-6
    n = jnp.sqrt(jnp.sum(attn * attn, axis=-1, keepdims=True) + eps * eps)
    attn = attn * jnp.tanh(n) / n

  if return_aux:
    aux['tanh_output'] = attn

  # Concatenate attention matrix of all heads into a single vector.
  attn = attn.reshape(*single_electron_stream.shape[:-1], -1)   # [length, heads*value_dim]

  # 9. Gate: tanh(SiLU(XW_G))を要素ごとに掛ける。
  if use_gate:
    g = jnp.dot(single_electron_stream if gate_stream is None else gate_stream, params['g_w'])
    attn = attn * jnp.tanh(jax.nn.silu(g))

  if return_aux:
    aux['gate_output'] = attn

  # 10. 連結したheadを一度の線形射影W_Oで統合する。
  out = network_blocks.linear_layer(attn, params['attn_output'])
  if return_aux:
    aux['projected_output'] = out

  # Learnable output scale e^{s_O}. TanhNorm bounds |o| < 1 and
  # the gate shrinks it further, so without this the attention branch cannot
  # contribute at the magnitude the residual stream works at.
  if use_output_scale:
    out = jnp.exp(params['out_scale']) * out

  if return_aux:
    return out, aux
  return out
