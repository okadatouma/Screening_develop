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


"""段階A前のWindowの固定参照。元の演算ASTを保存し、本体から独立して検証する。"""
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
    _om = params['mask_omega']
    if freeze_w:
        _om = jax.lax.stop_gradient(_om)
    w = d_ref * jax.nn.softplus(_om)
    if 'mask_rings' in params:
        _b = params['mask_rings']
        if freeze_w:
            _b = jax.lax.stop_gradient(_b)
        _pp = jax.lax.stop_gradient(params['mask_envp'])
        b = d_ref * _b
        dl = w[0]
        if 'mask_gauss' in params:
            th = _ring_steps(d_ee, b, dl * jax.lax.stop_gradient(params['mask_gauss']), gauss=True)
        else:
            th = _ring_steps(d_ee, b, dl)
        P = th[:-1] - th[1:]
        x = jnp.clip(d_ee / d_ref, 0.0, 1.0)
        _pos = x > 0.0
        xp = jnp.where(_pos, jnp.power(jnp.where(_pos, x, 1.0), _pp), 0.0)
        g = 0.5 * (1.0 + jnp.cos(jnp.pi * xp))
        m = P * g[None, :, :]
    elif 'mask_centre' in params:
        _ct = params['mask_centre']
        if freeze_w:
            _ct = jax.lax.stop_gradient(_ct)
        c = d_ref * _ct
        t = jnp.clip((d_ee[None, :, :] - c[:, None, None]) / w[:, None, None], -1.0, 1.0)
        m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    else:
        t = jnp.clip(d_ee[None, :, :] / w[:, None, None], 0.0, 1.0)
        if win_shape == 'smooth7':
            m = 1.0 - _smooth7(t)
        else:
            m = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    return m
