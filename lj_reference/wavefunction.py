# -*- coding: utf-8 -*-
"""参照用コード ⑤-4：波動関数 ψ_θ(R, S) = Σ_k det Φ^k の対数を返す。

  R = (r_1..r_N)、S = (s_1..s_N)。ψ は組 (r_i, s_i) の入れ替えで符号を変える（行列式の行の入れ替え）。
  返り値は (位相 φ, ln|ψ|)：ψ = |ψ| e^{iφ}。⑤-1〜⑤-3 をつなぐだけのファイル。

  行列式 4 枚の和は、桁あふれを避けるため最大の |det| で割ってから足す（log-sum-exp）。

本体：periodicwave/CustomScreenformer.py の make_fermi_net（network_apply）と
periodicwave/network_blocks.py の logdet_matmul。
"""
import jax.numpy as jnp

from lj_reference import attention, inputs, orbitals


def log_psi(params, model, x, s, atom):
    """(φ, ln|ψ|)。x：位置を並べた長さ 2N の配列、s：擬スピン ±1（長さ N）、atom：原点（ダミーの原子の位置）。"""
    h = attention.features(params, model, x, s, atom)            # ⑤-1, ⑤-2
    d = inputs.pair_distances(model, x)
    Phi = orbitals.orbital_matrices(params, model, x, s, h, d)   # ⑤-3
    sign, logdet = jnp.linalg.slogdet(Phi)                       # 4 枚それぞれ：det = sign · e^{logdet}
    m = jnp.max(logdet)
    total = jnp.sum(sign * jnp.exp(logdet - m))
    return jnp.angle(total), jnp.log(jnp.abs(total)) + m


def log_psi_complex(params, model, x, s, atom):
    """ln ψ = ln|ψ| + iφ（複素）。④ の局所エネルギーと勾配で使う。"""
    phase, logabs = log_psi(params, model, x, s, atom)
    return logabs + 1j * phase
