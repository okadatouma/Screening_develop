# -*- coding: utf-8 -*-
"""参照用コード ⑤-3：軌道（HF の軌道 ＋ ネットワークの補正）と、擬スピンの組ごとの Jastrow。

  φ^k_j(r_i, s_i) = φ^HF_j(r_i, s_i) + ε [W^k_{s_i} h_i]_j      （k = 1..4 は行列式の番号、j = 1..N は軌道の番号）

  φ^HF_j(r, s) = e^{i k_j·r} Σ_G c_j(G) u_{k_j+G}(s) e^{iG·r}     ① の HF の表（下のバンドに射影した Bloch 軌道、固定）
    u_q(+1) = 1/√(1 + λ²|q|²)、u_q(-1) = λ q_+/√(1 + λ²|q|²)、q_+ = q_x + i q_y   （下のバンドのスピノル）
  W^k_± h_i：⑤-2 の特徴 h_i を複素の軌道の値に写す線形写像（擬スピン +1 と -1 で別の重み）。ε = 0.05。

  J = Σ_{i<j} -γ_{s_i s_j} a_{s_i s_j} exp(-d_ij / a_{s_i s_j})，  a = softplus(θ_{s_i s_j})（++、--、+- の 3 つを学習）
  d → 0 で ∂J/∂d = γ（カスプ）。行列式の各成分に e^{J/N} を掛けるので、行列式全体では e^{J} が掛かる。

本体：periodicwave/lj_orbitals.py の make_lj_orbitals（hf_orbitals、apply）。
使うパラメータ：params['orbital'][0]['w']（64 × 4N·ndet。列の並びは外から順に 擬スピンの別 2、行列式 ndet、軌道 N、(実部, 虚部)）、
params['jastrow_ss']（alpha_pp, alpha_mm, alpha_pm）。
"""
import jax
import jax.numpy as jnp
import numpy as np


def hf_orbitals(model, x, s):
    """φ^HF_j(r_i, s_i)（[N_電子, N_軌道]、複素）。"""
    k, G, c, lam = model['hf_k'], model['hf_G'], model['hf_c'], model['lam']
    q = k[:, None, :] + G[None, :, :]                            # k_j + G
    nrm = 1.0 / np.sqrt(1.0 + lam ** 2 * np.sum(q ** 2, axis=-1))
    coef_p = jnp.asarray(c * nrm)                                # c_j(G) u(+1)
    coef_m = jnp.asarray(c * lam * (q[..., 0] + 1j * q[..., 1]) * nrm)   # c_j(G) u(-1)
    r = x.reshape(-1, 2)
    eG = jnp.exp(1j * (r @ jnp.asarray(G).T))                    # e^{iG·r_i}
    per = jnp.where(s[:, None] > 0, eG @ coef_p.T, eG @ coef_m.T)
    return jnp.exp(1j * (r @ jnp.asarray(k).T)) * per


def jastrow(params, model, d, s):
    """J（実数）。d は対の距離 [N, N]、s は擬スピン。"""
    n = s.shape[0]
    iu = np.triu_indices(n, k=1)
    dij, si, sj = d[iu], s[iu[0]], s[iu[1]]
    pp, mm = (si > 0) & (sj > 0), (si < 0) & (sj < 0)
    g_pp, g_mm, g_pm = model['cusps']
    p = params['jastrow_ss']
    a_pp, a_mm, a_pm = (jax.nn.softplus(p[key][0]) for key in ('alpha_pp', 'alpha_mm', 'alpha_pm'))
    a = jnp.where(pp, a_pp, jnp.where(mm, a_mm, a_pm))
    g = jnp.where(pp, g_pp, jnp.where(mm, g_mm, g_pm))
    return jnp.sum(-g * a * jnp.exp(-dij / a))


def orbital_matrices(params, model, x, s, h, d):
    """行列式に入れる行列 Φ^k_ij = φ^k_j(r_i, s_i) e^{J/N}（[ndet, N, N]、複素）。h は ⑤-2 の特徴。"""
    n, ndet = s.shape[0], model['ndet']
    out = h @ params['orbital'][0]['w']                          # [N, 4 N ndet]
    out = out[..., ::2] + 1j * out[..., 1::2]                    # 実部と虚部の組 → 複素 [N, 2 N ndet]
    out = out.reshape(n, 2, ndet, n)                             # [電子, 擬スピンの別, 行列式, 軌道]
    net = jnp.where(s[:, None, None] > 0, out[:, 0], out[:, 1])  # 電子 i の擬スピンの重み W_{s_i}
    phi = hf_orbitals(model, x, s)[:, None, :] + model['eps'] * net
    return jnp.transpose(phi, (1, 0, 2)) * jnp.exp(jastrow(params, model, d, s) / n)
