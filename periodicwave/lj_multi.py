# -*- coding: utf-8 -*-
"""[λ-jellium 多点] 1 本のネットワークで多くの (r_s, lam) を学習するための道具（HF なし、2026-11）。

方針（ユーザーと決めたこと）：点ごとに変えるのは (r_s, lam) だけで、相の答え（HF、相ごとの出発）は教えない。
  * 各ウォーカーは 1 つの点 p に属し、data.charges = (r_s, lam, p) を持つ（チェックポイントにも残る）。
    - ネットワークは (r_s, lam) を 1 電子の入力の最後の 2 成分として受け取る（CustomScreenformer.condition_inputs）。
    - 局所エネルギーはウォーカーの (r_s, lam) で計算する（lambda_jellium.local_energy の per_walker）。
    - Jastrow のカスプもウォーカーの (r_s, lam) から（lj_orbitals の per_walker）。
    - 勾配で E_L から引く平均は点ごと（loss.make_loss の num_groups）。
  * 軌道は HF なしのネットワークの出力（lj_orbitals の hf_k = None）。
  * 出発：ネットワークの軌道を、相互作用のない λ-jellium の基底状態の軌道
        phi0_j(r, s) = e^{i k_j·r} u_{k_j}(s; lam),   u_k(+1) = 1/sqrt(1 + lam^2 k^2),  u_k(-1) = lam k_+/sqrt(1 + lam^2 k^2)
    に合わせる（事前学習）。k_j は結晶の第 1 ブリルアン域の L x L の k 点（lj_hf.kgrid、N = L^2 個）。
    下のバンド（Delta h の固有値 0）だけを使うので、相互作用を切った H の基底状態の 1 つ（N = 36 では最後の殻が
    縮退しているので、C3 対称な第 1 ブリルアン域の詰め方を選ぶ）。どの点でも同じ規則で、相の情報は入っていない。
  * 各デバイスに全部の点のウォーカーを同じ数ずつ置く（デバイスあたりのウォーカー数は点の数の倍数）。

単位はコード単位（長さ r_s a_B、エネルギー 2/r_s^2 Ry）。セルは列が並進ベクトルの行列。
"""
import os
import time

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np
import optax

from periodicwave import constants
from periodicwave import networks
from periodicwave.pbc import lj_hf


def load_points(path):
  """点の表を読む。各行 'r_s lam'（3 列目以降と # から後ろは読まない）。返り値 (P, 2)。"""
  pts = []
  with open(path, encoding='utf-8') as fh:
    for line in fh:
      line = line.split('#', 1)[0].strip()
      if not line:
        continue
      v = line.replace(',', ' ').split()
      pts.append((float(v[0]), float(v[1])))
  if not pts:
    raise ValueError(f"点の表 {path} が空")
  return np.asarray(pts, dtype=np.float64)


def walker_params(points, n_per_device):
  """デバイスあたり n_per_device 本のウォーカーの (r_s, lam, 点の番号)、(n_per_device, 3)。点ごとに同じ本数ずつ、順に並べる。"""
  points = np.asarray(points, dtype=np.float64)
  P = points.shape[0]
  if n_per_device % P:
    raise ValueError(f"デバイスあたりのウォーカー数 {n_per_device} が点の数 {P} の倍数でない"
                     "（PW_BATCH は デバイスの数 x 点の数 の倍数にする）")
  per = n_per_device // P
  idx = np.repeat(np.arange(P), per)
  return np.concatenate((points[idx], idx[:, None].astype(np.float64)), axis=1)


def cond_scaling(points):
  """ネットワークの入力 ((r_s - 中心)/幅, (lam - 中心)/幅) の中心と幅。点の範囲の中点と半幅（範囲が 0 なら幅 1）。"""
  points = np.asarray(points, dtype=np.float64)
  lo, hi = points.min(axis=0), points.max(axis=0)
  center = 0.5 * (lo + hi)
  scale = np.where(hi - lo > 1e-12, 0.5 * (hi - lo), 1.0)
  return [float(v) for v in center], [float(v) for v in scale]


def free_band_kpoints(L):
  """L x L のセルの k 点（結晶の第 1 ブリルアン域、lj_hf.kgrid）、(L^2, 2)。セルの向きは run_lambda_jellium.py と同じ。"""
  _, _, B = lj_hf.geometry(L)
  _, _, k = lj_hf.kgrid(B, L)
  return np.asarray(k, dtype=np.float64)


def free_band_orbitals(pos, spins, lam, kpts):
  """相互作用のない下のバンドの軌道 phi0_j(r_i, s_i) = e^{i k_j·r_i} u_{k_j}(s_i; lam)、(N_e, N_orb)、複素。"""
  n = spins.shape[0]
  r = jnp.reshape(pos, (n, 2))
  k = jnp.asarray(kpts, dtype=r.dtype)
  nrm = 1.0 / jnp.sqrt(1.0 + lam ** 2 * jnp.sum(k * k, axis=-1))           # (N_orb,)
  u_m = lam * (k[:, 0] + 1j * k[:, 1]) * nrm                                # u_k(-1)
  u_p = jax.lax.complex(nrm, jnp.zeros_like(nrm))                           # u_k(+1)
  amp = jnp.where(spins[:, None] > 0, u_p[None, :], u_m[None, :])          # (N_e, N_orb)
  return jnp.exp(1j * (r @ k.T)) * amp


def group_means(values, charges, num_points):
  """記録用：ウォーカーの値を点ごとに平均する（numpy、全デバイス）。values (..., B)、charges (..., B, 3)。"""
  v = np.asarray(values).reshape(-1)
  g = np.asarray(charges).reshape(-1, np.asarray(charges).shape[-1])[:, 2].astype(int)
  cnt = np.bincount(g, minlength=num_points)
  if np.iscomplexobj(v):
    s = np.bincount(g, weights=v.real, minlength=num_points) + 1j * np.bincount(g, weights=v.imag, minlength=num_points)
  else:
    s = np.bincount(g, weights=v, minlength=num_points)
  return s / np.maximum(cnt, 1)


def pretrain(params, data, network, lattice, L, steps, lr, key, log_every=100, save_path=None):
  """ネットワークの軌道（Jastrow の e^{J/N} を掛ける前、行列式 ndet 枚とも）を自由な下のバンドの軌道に合わせる。

    loss = < |Phi_net(R, S) - Phi0(R, S; lam)|^2 >      （成分と行列式とウォーカーで平均）
  R は毎回セルの中で一様に、S は ±1 を等確率で選び直す（データの charges の (r_s, lam) はそのまま）。
  Adam（学習率 lr）。Jastrow の長さ（jastrow_ss）は動かさない。params・data は pmap 用（先頭がデバイスの軸）。
  save_path があれば pretrain_stats.csv に loss を残す。返り値は合わせた params。
  """
  kpts = free_band_kpoints(L)
  lat = jnp.asarray(np.asarray(lattice, dtype=np.float64))
  optimizer = optax.adam(lr)

  def loss_one(p, pos, spins, atoms, charges):
    phi = network.orbitals(p, pos, spins, atoms, charges, with_jastrow=False)[0]   # (ndet, N, N)
    target = free_band_orbitals(pos, spins, charges[1], kpts)                     # (N, N)
    return jnp.mean(jnp.abs(phi - target[None]) ** 2)

  def batch_loss(p, d):
    return jnp.mean(jax.vmap(loss_one, in_axes=(None, 0, 0, 0, 0))(
        p, d.positions, d.spins, d.atoms, d.charges))

  def step(p, opt_state, d):
    val, grad = jax.value_and_grad(batch_loss)(p, d)
    grad = constants.pmean(grad)
    if 'jastrow_ss' in grad:
      grad = dict(grad)
      grad['jastrow_ss'] = jax.tree_util.tree_map(jnp.zeros_like, grad['jastrow_ss'])
    updates, opt_state = optimizer.update(grad, opt_state, p)
    return optax.apply_updates(p, updates), opt_state, constants.pmean(val)

  pstep = constants.pmap(step)
  opt_state = jax.pmap(optimizer.init)(params)
  shape_r = data.positions.shape                     # (devices, B, 2N)
  n = data.spins.shape[-1]
  dt = data.positions.dtype
  fh = None
  if save_path:
    fh = open(os.path.join(save_path, 'pretrain_stats.csv'), 'w', encoding='utf-8')
    fh.write('step,loss\n')
  t0 = time.time()
  for t in range(int(steps)):
    key, k1, k2 = jax.random.split(key, 3)
    u = jax.random.uniform(k1, shape_r[:-1] + (n, 2), dtype=dt)
    pos = jnp.einsum('...nd,ed->...ne', u, lat.astype(dt)).reshape(shape_r)
    spins = jnp.where(jax.random.bernoulli(k2, 0.5, data.spins.shape), 1.0, -1.0).astype(data.spins.dtype)
    d = networks.FermiNetData(**(dict(data) | {
        'positions': kfac_jax.utils.broadcast_all_local_devices(pos.reshape(shape_r)),
        'spins': kfac_jax.utils.broadcast_all_local_devices(spins)}))
    params, opt_state, val = pstep(params, opt_state, d)
    if t % log_every == 0 or t == int(steps) - 1:
      v = float(np.asarray(val)[0])
      print(f"[多点] 事前学習 step {t:5d}  loss {v:.4e}  ({time.time() - t0:.0f} s)", flush=True)
      if fh:
        fh.write(f'{t},{v:.8e}\n')
        fh.flush()
  if fh:
    fh.close()
  return params
