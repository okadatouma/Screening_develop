# -*- coding: utf-8 -*-
"""λ-jellium の局所エネルギー（擬スピン付き 2 成分電子）。

模型：Soejima et al., arXiv:2503.12704（PRL 135, 186505）。NN-VMC の先行例は
Valenti, Vituri et al., arXiv:2512.07947（Delta = 2/r_s Ry、N = 36）。

電子の実スピンは偏極していて、各電子が 2 成分の擬スピン s_i = +1（上成分）/ -1（下成分）を持つ。
波動関数 Psi(R, S) は位置 R と擬スピン S の関数で、組 (r_i, s_i) の入れ替えで反対称。
CustomScreenformer は s_i を 1 電子特徴に足している（CustomScreenformer.py の spins 連結）ので、
nspins = (N, 0) の 1 枚の行列式で、そのまま (r, s) の関数になる。擬スピンはモンテカルロで
位置と一緒に標本化する（mcmc.mh_spin_flip）。

ハミルトニアン（コード単位：長さ r_s a_B、エネルギー 1 = Ry r_s^2 / 2）
    H = sum_i [ -1/2 lap_i + Delta h(i) ] + r_s sum_{i<j} 1/r_ij          （Coulomb は Ewald）
    h = [[ -lam^2 lap , i lam d    ],
         [ i lam dbar , 1          ]],    d = d_x - i d_y,  dbar = d_x + i d_y
論文（Ry、長さ r_s a_B）の Delta = 2/r_s は、コード単位で Delta = r_s。lam は長さ r_s a_B 単位。
h の下のバンドは平ら（固有値 0、スピノル (1, lam q_+) / sqrt(1 + lam^2 q^2)、q_+ = q_x + i q_y）、
上のバンドは Delta (1 + lam^2 q^2)。

局所エネルギー E_L = [H Psi](R, S) / Psi(R, S)（複素数）
    E_L = - sum_i c_i lap_i Psi / Psi,          c_i = 1/2 + Delta lam^2 [s_i = +1]
          + Delta n_-,                           n_- = #{i : s_i = -1}
          + i Delta lam sum_i [Psi(S^(i)) / Psi(S)] D_i log Psi(S^(i))
          + V_Ewald
    S^(i)：電子 i の擬スピンだけを反転した配置。D_i = d_{x_i} - i s_i d_{y_i}
    （いまの s_i が +1 なら d、-1 なら dbar）。
Im E_L は |Psi|^2 平均で 0 になるが、位相を動かすパラメータの勾配 2 Re<(E_L - E) O*> に要るので
捨てない（pbc.hamiltonian.local_energy の folx・複素の枝は Re だけを返す点に注意）。
"""

from typing import Optional, Sequence

import folx
import jax
import jax.numpy as jnp
import numpy as np

from periodicwave import networks
from periodicwave.pbc import hamiltonian as pbc_hamiltonian


def make_weighted_laplacian(f, method: str = 'folx', ndim: int = 2):
  """(params, data, c) -> sum_i c_i lap_i Psi / Psi（複素）を返す関数を作る。

  c は電子ごとの重み（長さ N）。
    folx    : 変数を x = sqrt(c) * y と伸ばすと lap_y g = sum_k c_k d_k^2 f になるので、
              前進ラプラシアン 1 回で重み付きの和が出る。学習用。
    default : ヘッシアンの対角を 1 座標ずつ（grad の jvp）。検算・解析用。
  lap Psi / Psi = lap log Psi + (grad log Psi)^2（複素のまま 2 乗、共役は取らない）。
  """
  if method == 'folx':
    def wlap(params, data, c):
      sc = jnp.repeat(jnp.sqrt(c), ndim).astype(data.positions.dtype)

      def g(y):
        return f(params, sc * y, data.spins, data.atoms, data.charges)

      ph, la = folx.forward_laplacian(g, sparsity_threshold=0)(data.positions / sc)
      lap = la.laplacian + 1j * ph.laplacian
      jac = la.jacobian.dense_array + 1j * ph.jacobian.dense_array
      return lap + jnp.sum(jac * jac)
    return wlap

  if method == 'default':
    def wlap(params, data, c):
      lap_i = per_electron_laplacian(f, params, data, ndim)
      return jnp.sum(c * lap_i)
    return wlap

  raise NotImplementedError(f'laplacian_method {method} is not implemented')


def per_electron_laplacian(f, params, data, ndim: int = 2):
  """電子ごとの lap_i Psi / Psi（複素、長さ N）。ヘッシアンの対角を jvp で 1 列ずつ取る。"""
  x = data.positions

  def grad_logpsi(xx):
    gp = jax.grad(lambda z: f(params, z, data.spins, data.atoms, data.charges)[0])(xx)
    gl = jax.grad(lambda z: f(params, z, data.spins, data.atoms, data.charges)[1])(xx)
    return gl + 1j * gp

  g0, dg = jax.linearize(grad_logpsi, x)
  n = x.shape[0]
  eye = jnp.eye(n, dtype=x.dtype)
  hdiag = jax.lax.map(lambda k: dg(eye[k])[k], jnp.arange(n))
  per_coord = hdiag + g0 * g0
  return jnp.sum(per_coord.reshape(-1, ndim), axis=-1)


def make_flip_terms(f, ndim: int = 2, flip_batch: int = 0):
  """(params, data, phase0, logabs0) -> 電子ごとの [Psi(S^(i))/Psi(S)] D_i log Psi(S^(i))（複素、長さ N）。

  D_i log Psi(S^(i)) は前進モード微分（x_i, y_i の 2 方向）で取る。flip_batch > 0 なら
  電子を flip_batch 個ずつ順に処理してメモリを抑える（0 なら N 個まとめて vmap）。
  """
  def terms(params, data, phase0, logabs0):
    x, s = data.positions, data.spins
    n = s.shape[0]
    nx = x.shape[0]

    def one(i):
      si = s * (1.0 - 2.0 * jax.nn.one_hot(i, n, dtype=s.dtype))

      def fi(xx):
        return f(params, xx, si, data.atoms, data.charges)

      ex = jax.nn.one_hot(ndim * i, nx, dtype=x.dtype)
      ey = jax.nn.one_hot(ndim * i + 1, nx, dtype=x.dtype)
      (ph, la), lin = jax.linearize(fi, x)
      dph_x, dla_x = lin(ex)
      dph_y, dla_y = lin(ey)
      d = (dla_x + 1j * dph_x) - 1j * s[i] * (dla_y + 1j * dph_y)
      ratio = jnp.exp((la - logabs0) + 1j * (ph - phase0))
      return ratio * d

    idx = jnp.arange(n)
    if flip_batch and 0 < flip_batch < n:
      return jax.lax.map(one, idx, batch_size=flip_batch)
    return jax.vmap(one)(idx)

  return terms


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = True,
    lattice: Optional[jnp.ndarray] = None,
    convergence_radius: int = 5,
    lam: float = 0.0,
    delta: Optional[float] = None,
    potential_type: str = 'Coulomb',
    potential_kwargs: Optional[dict] = None,
    kinetic_kwargs: Optional[dict] = None,
    flip_batch: int = 0,
    return_components: bool = True,
):
  """λ-jellium の局所エネルギー e_l(params, key, data) -> (E_L, 付随量) を作る。

  Args:
    f: (phase, log|psi|) を返すネットワーク（complex_output が必要）。
    lattice: スーパーセルの格子ベクトル（列）。
    lam: λ（長さ r_s a_B 単位）。
    delta: Δ（コード単位）。None なら interaction_energy_scale（= r_s）、すなわち論文の 2/r_s Ry。
    potential_type: 'Coulomb'（Ewald、強さ r_s）または 'none'（検算用）。
    potential_kwargs: {'interaction_energy_scale': r_s}。
    kinetic_kwargs: {'laplacian_method': 'folx' | 'default'}。
    flip_batch: 擬スピン反転項を何電子ずつ処理するか（0 = 全部まとめて）。
    return_components: True なら付随量 [V, n_-] をウォーカーごとに返す
      （loss の local_energy_mat に入り、学習ログに出る）。
  """
  del nspins, use_scan
  if not complex_output:
    raise ValueError('λ-jellium には複素の波動関数が要る（cfg.network.complex = True）。')
  if lattice is None:
    raise ValueError('lambda_jellium.local_energy には lattice が要る。')
  potential_kwargs = dict(potential_kwargs or {})
  kinetic_kwargs = dict(kinetic_kwargs or {})
  r_s = float(potential_kwargs.get('interaction_energy_scale', 1.0))
  delta = r_s if delta is None else float(delta)
  lam = float(lam)
  method = kinetic_kwargs.get('laplacian_method', 'folx')
  ndim = int(np.asarray(lattice).shape[0])

  wlap = make_weighted_laplacian(f, method, ndim)
  flips = make_flip_terms(f, ndim, int(flip_batch))

  if potential_type == 'Coulomb':
    potential = pbc_hamiltonian.make_2DCoulomb_potential(
        lattice, jnp.array([0.0]), charges, convergence_radius, r_s)
  elif potential_type in ('none', 'None', None):
    potential = None
  else:
    raise NotImplementedError(f'lambda_jellium: potential_type {potential_type}')

  print(f"[λ-jellium] lam = {lam}, Delta = {delta} (code units; Delta/r_s = {delta / r_s:.4g}), "
        f"r_s = {r_s}, laplacian = {method}, flip_batch = {flip_batch}, potential = {potential_type}")

  def _e_l(params, key, data):
    del key
    s = data.spins
    dtype = data.positions.dtype
    plus = (s > 0).astype(dtype)
    n_minus = jnp.sum(1.0 - plus)
    c = 0.5 + delta * lam ** 2 * plus
    e_l = -wlap(params, data, c) + delta * n_minus
    if lam != 0.0:
      phase0, logabs0 = f(params, data.positions, s, data.atoms, data.charges)
      e_l = e_l + 1j * delta * lam * jnp.sum(flips(params, data, phase0, logabs0))
    if potential is not None:
      ae, ee, _, _ = networks.construct_input_features(data.positions, data.atoms, ndim=ndim)
      v = potential(ae, ee)
    else:
      v = jnp.zeros((), dtype)
    e_l = e_l + v
    if return_components:
      return e_l, jnp.stack([jnp.asarray(v, dtype), n_minus])
    return e_l, None

  return _e_l


def make_energy_components(
    f,
    lattice,
    lam: float,
    delta: float,
    r_s: float,
    convergence_radius: int = 5,
    coulomb: bool = True,
):
  """解析用：局所エネルギーを項ごとに分けて返す関数 (params, data) -> 複素 [T, K, C, F, V] を作る。

    T = -1/2 sum_i lap_i Psi / Psi                      （自由粒子の運動エネルギー）
    K = -Delta lam^2 sum_{i: s_i=+1} lap_i Psi / Psi
    C = Delta n_-
    F = i Delta lam sum_i [Psi(S^(i))/Psi(S)] D_i log Psi(S^(i))
    V = Ewald Coulomb
  H_Delta = K + C + F は h(i) の和の局所値で、その期待値は >= 0（下のバンドだけなら 0）。
  <H_Delta> / (N Delta) を上のバンドへの混ざりの目安に使う。
  """
  ndim = int(np.asarray(lattice).shape[0])
  flips = make_flip_terms(f, ndim, 0)
  potential = (pbc_hamiltonian.make_2DCoulomb_potential(
      lattice, jnp.array([0.0]), jnp.array([0.0]), convergence_radius, r_s) if coulomb else None)

  def comps(params, data):
    s = data.spins
    plus = (s > 0).astype(data.positions.dtype)
    lap_i = per_electron_laplacian(f, params, data, ndim)
    t = -0.5 * jnp.sum(lap_i)
    k = -delta * lam ** 2 * jnp.sum(plus * lap_i)
    cst = delta * jnp.sum(1.0 - plus) + 0j
    if lam != 0.0:
      phase0, logabs0 = f(params, data.positions, s, data.atoms, data.charges)
      fl = 1j * delta * lam * jnp.sum(flips(params, data, phase0, logabs0))
    else:
      fl = 0j
    if potential is not None:
      ae, ee, _, _ = networks.construct_input_features(data.positions, data.atoms, ndim=ndim)
      v = potential(ae, ee) + 0j
    else:
      v = 0j
    return jnp.stack([t, k, cst, fl, v])

  return comps
