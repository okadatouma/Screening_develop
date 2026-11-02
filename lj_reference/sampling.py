# -*- coding: utf-8 -*-
"""参照用コード ④-1：|ψ(R, S)|² に従って配置 (R, S) を取り直す（Metropolis 法）。

  1 回の呼び出しで、次を steps = 10 回くりかえす（ウォーカー W 本を同時に）：
    (a) 位置の提案：全電子を R' = R + w ξ（ξ は標準正規分布、w は幅）。受理の確率 min(1, |ψ(R', S)|²/|ψ(R, S)|²)
    (b) 擬スピンの提案を 3 回：電子 i を一様に選んで s_i → -s_i。受理の確率 min(1, |ψ(R, S')|²/|ψ(R, S)|²)
  提案は対称なので、受理の確率は |ψ|² の比だけで決まる。受理の判定は ln u < ln|ψ'|² - ln|ψ|²（u は一様乱数）。
  幅 w は本体が 100 step ごとに調整する：位置の受理率が 0.55 を超えれば 1.1 倍、0.50 を下回れば 0.9 倍（train_screen.py）。

本体：periodicwave/mcmc.py の _make_mcmc_step_with_flips、mh_update、mh_spin_flip、mh_accept。
乱数の使い方も本体と同じ順にしてあるので、同じ乱数の鍵からは同じ配置が出る（check.py で確かめる）。
"""
import jax
import jax.numpy as jnp

from lj_reference import wavefunction


def mcmc_step(params, model, x, s, atom, key, width, steps=10, flips=3):
    """x：[W, 2N] 位置、s：[W, N] 擬スピン。返り値 (x, s, 位置の受理率, 反転の受理率)。"""
    W, N = s.shape

    def log_prob(xx, ss):                                        # ln|ψ|²（ウォーカーごと）
        f = lambda a, b: wavefunction.log_psi(params, model, a, b, atom)[1]
        return 2.0 * jax.vmap(f)(xx, ss)

    def accept(lp_old, lp_new, key):
        key, sub = jax.random.split(key)
        return lp_new - lp_old > jnp.log(jax.random.uniform(sub, shape=lp_old.shape)), key

    lp = log_prob(x, s)
    n_move = n_flip = 0.0
    for _ in range(steps):
        key, sub = jax.random.split(key)                         # (a) 位置
        x2 = x + width * jax.random.normal(sub, shape=x.shape)
        lp2 = log_prob(x2, s)
        ok, key = accept(lp, lp2, key)
        x, lp, n_move = jnp.where(ok[:, None], x2, x), jnp.where(ok, lp2, lp), n_move + jnp.sum(ok)
        for _ in range(flips):                                   # (b) 擬スピン
            key, sub = jax.random.split(key)
            i = jax.random.randint(sub, (W,), 0, N)
            s2 = s * (1.0 - 2.0 * jax.nn.one_hot(i, N, dtype=s.dtype))
            lp2 = log_prob(x, s2)
            ok, key = accept(lp, lp2, key)
            s, lp, n_flip = jnp.where(ok[:, None], s2, s), jnp.where(ok, lp2, lp), n_flip + jnp.sum(ok)
    return x, s, n_move / (steps * W), n_flip / (steps * flips * W)
