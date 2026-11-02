# -*- coding: utf-8 -*-
"""参照用コード ④-3、④-4：エネルギーとその勾配、パラメータの更新。

  E(θ) = <E_L>（|ψ_θ|² の平均。W 本のウォーカーの平均で見積もる）
  ∂E/∂θ = 2 Re <(E_L - E) O*>、O = ∂ ln ψ_θ / ∂θ = ∂ ln|ψ| + i ∂φ
         = (2/W) Σ_b [ Re(E_L,b - E) ∂ ln|ψ_b| + Im(E_L,b) ∂φ_b ]

  外れ値を抑えるため、勾配に使う E_L は実部と虚部を別々に [中心 ± 5 × 平均絶対偏差] に切り詰める（中心は平均）。
  切り詰めた値の平均の実部を E の代わりに引く（本体の clip_local_energy = 5、center_at_clip = True）。
  lj3 の窓の幅 ω は学習しない（attention.window で勾配を止めている）。

  更新：本体は KFAC（自然勾配法の近似。kfac_jax、学習率 10、減衰つき）。ここでは仕組みを示すため単純な勾配降下 θ ← θ - η ∂E/∂θ。

本体：periodicwave/loss.py の make_loss（total_energy_jvp）と clip_local_values、periodicwave/train_screen.py の学習のループ。
"""
import jax
import jax.numpy as jnp

from lj_reference import hamiltonian, sampling, wavefunction


def clip_local_energy(e_l, scale=5.0):
    """E_L の実部と虚部を、それぞれ [平均 ± scale × 平均絶対偏差] に切り詰める。"""
    center = jnp.mean(e_l)

    def clip(v, c):
        tv = jnp.mean(jnp.abs(v - c))
        return jnp.clip(v, c - scale * tv, c + scale * tv)

    return clip(e_l.real, center.real) + 1j * clip(e_l.imag, center.imag)


def energy_and_gradient(params, model, x, s, atom, e_l, scale=5.0):
    """返り値 (E, ∂E/∂θ)。x, s, e_l は W 本のウォーカーの値（e_l は local_energy で計算したもの）。"""
    W = s.shape[0]
    e_c = clip_local_energy(e_l, scale)
    a = e_c - jnp.mean(e_c).real                                 # 勾配に掛ける重み（定数として扱う）

    def weighted(p):
        ph, la = jax.vmap(lambda xx, ss: wavefunction.log_psi(p, model, xx, ss, atom))(x, s)
        return (2.0 / W) * jnp.sum(a.real * la + a.imag * ph)

    return jnp.mean(e_l).real, jax.grad(weighted)(params)


def vmc_step(params, model, x, s, atom, key, width, lr=1e-3):
    """学習の 1 step の流れ：④-1 サンプル → ④-2 局所エネルギー → ④-3 勾配 → ④-4 更新（ここでは勾配降下）。"""
    x, s, p_move, p_flip = sampling.mcmc_step(params, model, x, s, atom, key, width)
    e_l_fn = hamiltonian.make_local_energy(model)
    e_l = jax.vmap(lambda xx, ss: e_l_fn(params, xx, ss, atom)[0])(x, s)
    E, grad = energy_and_gradient(params, model, x, s, atom, e_l)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grad)
    return params, x, s, E, (p_move, p_flip)
