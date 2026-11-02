# -*- coding: utf-8 -*-
"""参照用コード ④-2：局所エネルギー E_L = [Hψ](R, S) / ψ(R, S)（複素数）。

  ハミルトニアン（コード単位：長さ r_s a_B、エネルギーは 2/r_s² Ry を 1 とする。Δ = r_s が論文の 2/r_s Ry）
    H = Σ_i [ -½∇_i² + Δ h(i) ] + r_s Σ_{i<j} 1/|r_i - r_j|          （クーロンはセルの周期像も足す Ewald 和）
    h = [[ -λ²∇², iλ∂ ], [ iλ∂̄, 1 ]]   （擬スピン +1 が上の行）、∂ = ∂_x - i∂_y、∂̄ = ∂_x + i∂_y
  これを ψ(R, S) に作用させて ψ で割ると

    E_L = - Σ_i c_i ∇_i²ψ/ψ                       c_i = ½ + Δλ² [s_i = +1]
          + Δ n_-                                  n_- = 擬スピン -1 の電子の数
          + iΔλ Σ_i [ψ(S^(i))/ψ(S)] D_i ln ψ(S^(i))  S^(i)：電子 i の擬スピンだけ反転、D_i = ∂_{x_i} - i s_i ∂_{y_i}
          + V_Ewald
  ∇²ψ/ψ = ∇²ln ψ + (∇ln ψ)²（複素のまま 2 乗）。

本体：periodicwave/pbc/lambda_jellium.py の local_energy（ラプラシアンは folx の前進ラプラシアンで速く計算する。
ここではわかりやすさのため 2 階微分の行列（jax.hessian）の対角を使う）、
クーロンは periodicwave/pbc/hamiltonian.py の make_2DCoulomb_potential。
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from lj_reference import wavefunction


def coulomb_energy(model, truncation=5):
    """V_Ewald(x) を返す関数を作る。本体と同じ分け方（γ = (2.8/√面積)²、格子 ±5 まで）。

      V = r_s [ Σ_{i<j} φ(r_ij) + ½ N ξ - ½ N² φ_0 ]
      φ(r) = Σ_L erfc(√γ|r - L|)/|r - L| + (2π/面積) Σ_{G≠0} cos(G·r) erfc(|G|/2√γ)/|G|     （実空間と逆空間の和）
      ξ   = Σ_{L≠0} erfc(√γ|L|)/|L| + (2π/面積) Σ_{G≠0} erfc(|G|/2√γ)/|G| - 2√γ/√π        （自分の周期像との相互作用）
      φ_0 = 2π/(面積 √γ √π)                                                              （一様な正の背景）
    r_ij は最小像（セルの分数座標を [-½, ½) に戻す）。
    """
    A = model['A']
    area = abs(np.linalg.det(A))
    gamma = (2.8 / np.sqrt(area)) ** 2
    ords = sorted(range(-truncation, truncation + 1), key=abs)
    n = np.array(list(itertools.product(ords, repeat=2)), dtype=np.float64)   # 最初が (0, 0)
    L = jnp.asarray(n @ A.T)                                 # 実空間の格子ベクトル（0 を含む）
    G = jnp.asarray(n[1:] @ (2 * np.pi * np.linalg.inv(A)))  # 逆格子ベクトル（0 を除く）
    Gn = jnp.linalg.norm(G, axis=-1)
    Ln = jnp.linalg.norm(L[1:], axis=-1)
    sg = np.sqrt(gamma)
    xi = (jnp.sum(erfc(sg * Ln) / Ln) + (2 * np.pi / area) * jnp.sum(erfc(Gn / (2 * sg)) / Gn)
          - 2 * sg / np.sqrt(np.pi))
    phi0 = 2 * np.pi / area / sg / np.sqrt(np.pi)

    def phi(r):
        dl = jnp.linalg.norm(r - L, axis=-1)
        return (jnp.sum(erfc(sg * dl) / dl)
                + (2 * np.pi / area) * jnp.sum(jnp.cos(G @ r) * erfc(Gn / (2 * sg)) / Gn))

    def V(x):
        r = x.reshape(-1, 2)
        N = r.shape[0]
        i, j = np.triu_indices(N, k=1)
        f = (r[j] - r[i]) @ jnp.asarray(model['Ainv']).T
        rij = ((f + 0.5) % 1.0 - 0.5) @ jnp.asarray(A).T          # 最小像
        return model['r_s'] * (jnp.sum(jax.vmap(phi)(rij)) + 0.5 * N * xi - 0.5 * N ** 2 * phi0)

    return V


def make_local_energy(model):
    """e_l(params, x, s, atom) -> (E_L, 項ごとの値の辞書) を作る。1 つの配置について計算する。"""
    lam, delta = model['lam'], model['delta']
    V = coulomb_energy(model)

    def e_l(params, x, s, atom):
        def lp(xx, ss):                                          # (φ, ln|ψ|) を 1 本の配列に
            return jnp.stack(wavefunction.log_psi(params, model, xx, ss, atom))

        # (1) 運動エネルギー（λ² の項を含む）：- Σ_i c_i ∇_i²ψ/ψ
        J = jax.jacrev(lp)(x, s)                                 # [2, 2N]：(∂φ, ∂ln|ψ|)
        Hs = jax.hessian(lp)(x, s)                               # [2, 2N, 2N]
        g = J[1] + 1j * J[0]                                     # ∂ ln ψ
        hd = jnp.diagonal(Hs[1]) + 1j * jnp.diagonal(Hs[0])      # ∂² ln ψ（座標ごと）
        lap = (hd + g * g).reshape(-1, 2).sum(axis=-1)           # ∇_i²ψ/ψ（電子ごと）
        c = 0.5 + delta * lam ** 2 * (s > 0)
        kinetic = -jnp.sum(c * lap)

        # (2) 擬スピンを 1 本反転させる項：iΔλ Σ_i [ψ(S^(i))/ψ(S)] D_i ln ψ(S^(i))
        ph0, la0 = wavefunction.log_psi(params, model, x, s, atom)

        def flip(i):
            si = s * (1.0 - 2.0 * jax.nn.one_hot(i, s.shape[0], dtype=s.dtype))
            ph, la = wavefunction.log_psi(params, model, x, si, atom)
            Ji = jax.jacrev(lp)(x, si)
            dx = Ji[1, 2 * i] + 1j * Ji[0, 2 * i]
            dy = Ji[1, 2 * i + 1] + 1j * Ji[0, 2 * i + 1]
            return jnp.exp((la - la0) + 1j * (ph - ph0)) * (dx - 1j * s[i] * dy)

        flips = 1j * delta * lam * jnp.sum(jax.vmap(flip)(jnp.arange(s.shape[0])))

        # (3) 上のバンドの分 Δ n_-、(4) クーロン
        upper = delta * jnp.sum(s < 0)
        v = V(x)
        return kinetic + flips + upper + v, dict(kinetic=kinetic, flips=flips, upper=upper, coulomb=v)

    return e_l
