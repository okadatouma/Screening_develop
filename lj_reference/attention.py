# -*- coding: utf-8 -*-
"""参照用コード ⑤-2：埋め込み → 遮蔽アテンション → MLP。各電子の特徴 h_i を作る。

  x_i = W_emb^T（12 成分の入力）                                          埋め込み（64 次元）
  x_i ← LN( x_i + e^{s_O} W_O [ gate_i ⊙ TanhNorm( Σ_j W^h_ij v̂^h_j ) ]_h )   遮蔽アテンション（ヘッド h = 1..16）
  x_i ← LN( x_i + MLP(x_i) )                                             MLP（GELU の 2 層）
  h_i = x_i

  ヘッド h の重み W^h_ij = α(s^h_ij) m_h(d_ij)
    s^h_ij = q̂^h_i · k̂^h_j          q = W_q^T x、k = W_k^T x を単位長にしたもののコサイン類似度（-1 ≤ s ≤ 1）
    α(s)  = ((1 + s)/2)^4          Trim（幅 r = 2 で固定。q と k の向きが近い相手ほど重い）
    m_h(d) = 1 - S_7(min(d/w_h, 1))  窓（S_7(t) = 35t^4 - 84t^5 + 70t^6 - 20t^7。d = 0 で 1、d ≥ w_h で 0）
    w_h = d_ref · softplus(ω_h)    窓の幅（ヘッドごと。lj3 では学習しない）
  softmax の分母はない（重みの和で割らない）。TanhNorm は |o| が 1 を超えないようにする o → tanh(|o|) o/|o|。

本体：periodicwave/CustomScreenformer.py の make_screenformer_layers（埋め込み）、make_screening_attention（遮蔽アテンション）、
make_self_attention_block（残差と LayerNorm）、make_mlp（MLP）、make_layer_norm。
使うパラメータ：params['layers'] の 'embed'、'attention'[0]（q_w, k_w, v_w, g_w, attn_output, mask_omega, out_scale）、
'ln'[0]（2 つの LayerNorm）、'mlp'[0]（2 層の w, b）。
"""
import jax
import jax.numpy as jnp

from lj_reference import inputs


def unit(v, eps=1e-6):
    """最後の軸で長さ 1 にする（0 の近くでも 2 回微分できるよう eps を足す）。"""
    return v / jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + eps * eps)


def layer_norm(p, x, eps=1e-5):
    """LayerNorm：各電子の 64 成分を平均 0・分散 1 にそろえてから scale 倍して offset を足す。"""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return p['scale'] * (x - mean) / jnp.sqrt(var + eps) + p['offset']


def window(model, d, omega):
    """窓 m_h(d_ij)（[H, N, N]）。lj3 では窓の幅を学習しない（本体の PW_FREEZE_W=1）ので勾配を止める。"""
    w = model['d_ref'] * jax.nn.softplus(jax.lax.stop_gradient(omega))   # [H]
    t = jnp.clip(d[None, :, :] / w[:, None, None], 0.0, 1.0)
    s7 = t ** 4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))
    return 1.0 - s7


def screening_attention(model, p, x, d):
    """遮蔽アテンションの出力（残差に足す量）[N, 64]。p = params['layers']['attention'][0]、d = 対の距離。"""
    n, H, D, Dv = x.shape[0], model['heads'], model['attn_dim'], model['value_dim']
    q = unit((x @ p['q_w']).reshape(n, H, D))
    k = unit((x @ p['k_w']).reshape(n, H, D))
    v = unit((x @ p['v_w']).reshape(n, H, Dv))
    s = jnp.einsum('ihd,jhd->hij', q, k)                               # コサイン類似度 s^h_ij
    alpha = (0.5 * (1.0 + s)) ** 4                                    # Trim（r = 2）
    W = alpha * window(model, d, p['mask_omega'])                     # 重み W^h_ij
    o = jnp.einsum('hij,jhd->ihd', W, v)                              # Σ_j W^h_ij v̂^h_j（分母なし）
    norm = jnp.sqrt(jnp.sum(o * o, axis=-1, keepdims=True) + 1e-12)
    o = (o * jnp.tanh(norm) / norm).reshape(n, H * Dv)                # TanhNorm
    o = o * jnp.tanh(jax.nn.silu(x @ p['g_w']))                       # ゲート
    return jnp.exp(p['out_scale']) * (o @ p['attn_output'])           # W_O でヘッドを足し、e^{s_O} 倍


def mlp(p, x):
    """GELU の 2 層（64 → 64 → 64）。p = params['layers']['mlp'][0]。"""
    for layer in p:
        x = jax.nn.gelu(x @ layer['w'] + layer['b'])
    return x


def features(params, model, x, s, atom):
    """各電子の特徴 h_i [N, 64]（⑤-3 の軌道に入る）。"""
    L = params['layers']
    d = inputs.pair_distances(model, x)
    h = inputs.one_electron_inputs(model, x, s, atom) @ L['embed']   # 埋め込み
    h = layer_norm(L['ln'][0][0], h + screening_attention(model, L['attention'][0], h, d))
    h = layer_norm(L['ln'][0][1], h + mlp(L['mlp'][0], h))
    return h
