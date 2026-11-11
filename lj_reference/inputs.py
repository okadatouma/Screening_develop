# -*- coding: utf-8 -*-
"""参照用コード ⑤-1：電子の配置から、ネットワークに入れる量を作る。

  位置 r_i（i = 1..N）と擬スピン s_i = ±1 から
    (a) 1 電子の入力 12 成分（埋め込みの前）
    (b) 対の距離 d_ij（周期ノルム。遮蔽アテンションの窓と Jastrow が使う）
を作る。セルの分数座標 f = A^{-1} r、θ = 2π f。

本体：
  (a) periodicwave/pbc/feature_layer.py の make_pbc_feature_layer（sin θ, cos θ の 4 成分）
      ＋ periodicwave/CustomScreenformer.py の one_electron_inputs（高調波 6 成分と one-hot 2 成分）
  (b) periodicwave/networks.py の make_periodic_construct_input_features（r_ee）
"""
import jax.numpy as jnp


def periodic_norm(model, f):
    """分数座標の差 f（[..., 2]）の周期ノルム。

      d = (1/2π) sqrt( a^T M a + b^T M b )、a = 1 - cos 2πf、b = sin 2πf、M = A^T A
    短い距離では |A f| と一致し、セルの周期をもつ（f → f + 整数で変わらない）。
    """
    M = jnp.asarray(model['metric'])
    a = 1.0 - jnp.cos(2 * jnp.pi * f)
    b = jnp.sin(2 * jnp.pi * f)
    return jnp.sqrt(jnp.einsum('...m,mn,...n->...', a, M, a)
                    + jnp.einsum('...m,mn,...n->...', b, M, b)) / (2 * jnp.pi)


def pair_distances(model, x):
    """対の距離 d_ij（[N, N]、対角は 0）。x は位置を並べた長さ 2N の配列。

    対角は 0 だが、そのまま 0 のノルムを微分すると NaN になるので、本体と同じく対角に 1 を足してから 0 を掛ける。
    """
    r = x.reshape(-1, 2)
    n = r.shape[0]
    ee = r[None, :, :] - r[:, None, :]                       # ee[i, j] = r_j - r_i
    f = ee @ jnp.asarray(model['Ainv']).T                     # 分数座標
    eye = jnp.eye(n)
    return periodic_norm(model, f + eye[..., None]) * (1.0 - eye)


def one_electron_inputs(model, x, s, atom):
    """1 電子の入力 [N, 12]。並びは本体と同じ：

      sin θ_1, sin θ_2, cos θ_1, cos θ_2,                        θ = 2π f、f = A^{-1}(r - 原点)
      cos G_1·r, sin G_1·r, cos G_2·r, sin G_2·r, cos G_3·r, sin G_3·r,   G_k·r = 2π n_k·f（n_k = (6,0), (0,6), (6,-6)）
      n_+ = (1 + s)/2, n_- = (1 - s)/2                           （擬スピンの one-hot）
    G_k は結晶（電子 1 個の単位胞）の最短の逆格子ベクトルで、6×6 のセルでは分数座標の 6 倍の波になる。
    """
    r = x.reshape(-1, 2) - jnp.asarray(atom).reshape(-1)[:2]
    f = r @ jnp.asarray(model['Ainv']).T
    theta = 2 * jnp.pi * f
    cell = jnp.concatenate((jnp.sin(theta), jnp.cos(theta)), axis=-1)
    ph = 2 * jnp.pi * f @ jnp.asarray(model['harmonics']).T                  # [N, 3]
    harm = jnp.stack((jnp.cos(ph), jnp.sin(ph)), axis=-1).reshape(r.shape[0], -1)
    onehot = jnp.stack((0.5 * (1.0 + s), 0.5 * (1.0 - s)), axis=-1)
    return jnp.concatenate((cell, harm, onehot.astype(cell.dtype)), axis=-1)
