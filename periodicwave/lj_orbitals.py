# -*- coding: utf-8 -*-
"""[λ-jellium] 擬スピンを成分にもつ軌道（スピノル）と、擬スピンの組ごとの Jastrow。

  Psi(R, S) = e^{J(R,S)} sum_k det[ phi^k_j(r_i, s_i) ]
  phi^k_j(r_i, s) = phi^HF_j(r_i, s) + eps [W^k_s h_i]_j            （s = +1, -1 で別の重み W_+, W_-）
  phi^HF_j(r, s)  = e^{i k_j.r} sum_G c_j(G) u_{k_j+G}(s) e^{iG.r}   （下のバンドに射影した HF の Bloch 軌道、固定）
  J = sum_{i<j} -gamma_{s_i s_j} alpha_{s_i s_j} exp(-d_ij / alpha_{s_i s_j})    （d は周期ノルム）
      d -> 0 で dJ/dd = gamma（カスプ）、遠くで 0。-gamma alpha^2/(alpha + d) の形（Psiformer）は 1/d で遠くまで残り、
      擬スピンの組で強さが違うので擬スピン -1 の数そのものに偏りをかける（N = 9 で 1 本反転ごとに log|Psi| が ~5 下がった）。
      指数の形なら擬スピンに依る部分は alpha の距離（初期 0.3）の内側だけ。alpha = softplus(theta) で正に保つ。

h_i は CustomScreenformer の出力（擬スピン s_i は入力の 1 成分として入っている）。eps は網の部分の初期の大きさ。
カスプ（d -> 0 で d log Psi / d d = gamma）：2 体の相対運動の運動エネルギーの係数は c_{s_i} + c_{s_j}、
c_+ = 1/2 + Delta lam^2、c_- = 1/2（lambda_jellium.py の c_i）。Coulomb の強さ r_s に対し、2 次元で
  逆向き（節なし）gamma_{+-} = r_s / (c_+ + c_-)、同じ向き（行列式の節つき、l = 1）gamma_{ss} = r_s / (3 * 2 c_s)。
"""

from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

from periodicwave import network_blocks
from periodicwave import networks


ALPHA0 = 0.3   # Jastrow の到達距離の初期値（r_s a_B 単位。最近接距離 a = 1.905 の 1/6 ほど）


def load_hf_table(path):
  """lj_hf.LJHF.orbitals_text の表（k_x k_y G_x G_y Re c Im c）を読む。返り値 k (Nk,2), G (NG,2), c (Nk,NG)。"""
  t = np.loadtxt(path)
  kxy = t[:, 0:2]
  gxy = t[:, 2:4]
  c = t[:, 4] + 1j * t[:, 5]
  # k の並び（連続する同じ k の塊）
  ks, idx = [], []
  for row in kxy:
    if not ks or np.max(np.abs(ks[-1] - row)) > 1e-9:
      ks.append(row.copy())
  nk = len(ks)
  ng = t.shape[0] // nk
  G = gxy[:ng]
  return np.array(ks), G, c.reshape(nk, ng)


def cusp_values(r_s, lam, delta):
  cp = 0.5 + delta * lam ** 2
  cm = 0.5
  return (r_s / (3.0 * 2 * cp), r_s / (3.0 * 2 * cm), r_s / (cp + cm))   # (++, --, +-)


def make_lj_orbitals(
    nspins: Sequence[int],
    charges: jnp.ndarray,
    options,
    equivariant_layers,
    hf_k: np.ndarray,
    hf_G: np.ndarray,
    hf_c: np.ndarray,
    lam: float,
    cusps: Sequence[float],
    net_scale: float = 0.05,
    use_jastrow: bool = True,
):
  equivariant_layers_init, equivariant_layers_apply = equivariant_layers
  n = int(sum(nspins))
  nk = hf_k.shape[0]
  if nk != n:
    raise ValueError(f"HF の k の数 {nk} が電子数 {n} と違う")
  ndet = int(options.determinants)
  construct = networks.make_periodic_construct_input_features(options.pbc_lattice)

  kk = jnp.asarray(hf_k)                       # (N, 2)
  GG = jnp.asarray(hf_G)                       # (NG, 2)
  q = hf_k[:, None, :] + hf_G[None, :, :]      # (N, NG, 2)
  qp = q[..., 0] + 1j * q[..., 1]
  nrm = 1.0 / np.sqrt(1.0 + lam ** 2 * np.sum(q ** 2, axis=-1))
  coef_p = jnp.asarray(hf_c * nrm)             # c_j(G) u_{k_j+G}(+1)
  coef_m = jnp.asarray(hf_c * lam * qp * nrm)  # c_j(G) u_{k_j+G}(-1)
  g_pp, g_mm, g_pm = [float(v) for v in cusps]
  iu = np.triu_indices(n, k=1)

  def init(key: chex.PRNGKey):
    key, subkey = jax.random.split(key)
    params = {}
    dims_in, params['layers'] = equivariant_layers_init(subkey)
    norb = 2 * 2 * n * ndet                     # (s = +1, -1) x (Re, Im) x 軌道 x 行列式
    key, subkey = jax.random.split(key)
    params['orbital'] = [network_blocks.init_linear_layer(
        subkey, in_dim=dims_in, out_dim=norb, include_bias=options.bias_orbitals)]
    if use_jastrow:
      # alpha = softplus(theta) > 0（長い学習で alpha が 0 を越えて負になると exp(-d/alpha) が発散するので、正に保つ）
      th0 = float(np.log(np.expm1(ALPHA0)))
      params['jastrow_ss'] = {'alpha_pp': th0 * jnp.ones((1,)), 'alpha_mm': th0 * jnp.ones((1,)),
                              'alpha_pm': th0 * jnp.ones((1,))}
    return params

  def hf_orbitals(pos, spins):
    r = jnp.reshape(pos, (n, 2))
    eG = jnp.exp(1j * (r @ GG.T))                                   # (N_e, NG)
    per = jnp.where(spins[:, None] > 0, eG @ coef_p.T, eG @ coef_m.T)  # (N_e, N_orb)
    return jnp.exp(1j * (r @ kk.T)) * per

  def jastrow(r_ee, spins):
    d = r_ee[..., 0][iu]
    si, sj = spins[iu[0]], spins[iu[1]]
    return d, si, sj

  def apply(params, pos, spins, atoms, charges):
    ae, ee, r_ae, r_ee = construct(pos, atoms, ndim=options.ndim)
    h = equivariant_layers_apply(params['layers'], ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee,
                                 spins=spins, charges=charges)
    out = network_blocks.linear_layer(h, **params['orbital'][0])    # (N_e, 4 n ndet)
    out = out[..., ::2] + 1.0j * out[..., 1::2]                     # (N_e, 2 n ndet)
    out = jnp.reshape(out, (n, 2, ndet, n))
    net = jnp.where(spins[:, None, None] > 0, out[:, 0], out[:, 1])  # (N_e, ndet, N_orb)
    phi = hf_orbitals(pos, spins)[:, None, :] + net_scale * net       # (N_e, ndet, N_orb)
    orbital = jnp.transpose(phi, (1, 0, 2))                           # (ndet, N_e, N_orb)
    if use_jastrow:
      d, si, sj = jastrow(r_ee, spins)
      pp = (si > 0) & (sj > 0)
      mm = (si < 0) & (sj < 0)
      a_pp, a_mm, a_pm = (jax.nn.softplus(params['jastrow_ss'][k][0]) for k in ('alpha_pp', 'alpha_mm', 'alpha_pm'))
      a = jnp.where(pp, a_pp, jnp.where(mm, a_mm, a_pm))
      g = jnp.where(pp, g_pp, jnp.where(mm, g_mm, g_pm))
      J = jnp.sum(-g * a * jnp.exp(-d / a))
      orbital = orbital * jnp.exp(J / n)
    return [orbital]

  return init, apply
