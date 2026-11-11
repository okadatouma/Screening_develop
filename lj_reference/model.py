# -*- coding: utf-8 -*-
"""参照用コード ③：計算の設定と、学習したパラメータを読む。

lj3 の run のフォルダ（config.json とチェックポイント qmcjax_ckpt_*.npz）から、
① の HF の表、⑤ のネットワークの形、④ のハミルトニアンの定数を取り出して 1 つの辞書 model にまとめる。
本体では periodicwave/configs/run_lambda_jellium.py が設定を作り、train_screen.py がパラメータを持つ。

単位はコード単位：長さは r_s a_B、エネルギーは 2/r_s² Ry を 1 とする（E[Ry] = (2/r_s²) E）。セルは列が並進ベクトルの行列 A。
窓の形（1 - S_7）と Trim の 4 乗は config.json に残らない（本体では環境変数 PW_WIN_SHAPE、PW_TRIM_POW で選ぶ）ので、
lj3 の run の値を attention.py に書き込んである。
"""
import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np


def read_hf_table(path):
    """① の HF の表（k_x k_y G_x G_y Re c Im c の行）を読む。返り値 k (N, 2)、G (NG, 2)、c (N, NG)。

    本体：periodicwave/lj_orbitals.py の load_hf_table。表は periodicwave/pbc/lj_hf.py の LJHF.orbitals_text が書く。
    """
    t = np.loadtxt(path)
    ks = []
    for row in t[:, 0:2]:                          # 同じ k の行が続く
        if not ks or np.max(np.abs(ks[-1] - row)) > 1e-9:
            ks.append(row.copy())
    nk = len(ks)
    ng = t.shape[0] // nk
    c = (t[:, 4] + 1j * t[:, 5]).reshape(nk, ng)
    return np.array(ks), t[:ng, 2:4], c


def max_periodic_distance(A):
    """セルの中での周期ノルムの最大値 d_ref（窓の幅 w の物差し）。本体：CustomScreenformer._max_periodic_distance。"""
    f = np.linspace(0.0, 1.0, 201, endpoint=False)
    grid = np.stack(np.meshgrid(f, f, indexing='ij'), axis=-1)
    a, b = 1.0 - np.cos(2 * np.pi * grid), np.sin(2 * np.pi * grid)
    M = A.T @ A
    d2 = np.einsum('...m,mn,...n->...', a, M, a) + np.einsum('...m,mn,...n->...', b, M, b)
    return float(np.sqrt(d2).max() / (2 * np.pi))


def cusp_values(r_s, lam, delta):
    """Jastrow のカスプ γ（++、--、+-）。本体：lj_orbitals.cusp_values。

    2 体の相対運動の運動エネルギーの係数は c_{s_i} + c_{s_j}、c_+ = 1/2 + Δλ²、c_- = 1/2。
    逆向き（節なし）γ_{+-} = r_s/(c_+ + c_-)、同じ向き（行列式の節つき）γ_{ss} = r_s/(3·2c_s)。
    """
    cp, cm = 0.5 + delta * lam ** 2, 0.5
    return r_s / (3.0 * 2 * cp), r_s / (3.0 * 2 * cm), r_s / (cp + cm)


def _lattice(cfg):
    v = cfg['system'].get('pbc_lattice', cfg['system']['make_local_energy_kwargs']['lattice'])
    if isinstance(v, list):
        return np.asarray(v, dtype=np.float64)
    nums = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', str(v))][:4]
    return np.asarray(nums, dtype=np.float64).reshape(2, 2)


def check_lj3(cfg, folder=''):
    """このコードが写しているのは lj3 の設定だけ。config.json がそれと違えば止める。"""
    net = cfg['network']
    cs = net.get('CustomScreenformer', {})
    want = {
        ('network', 'network_type'): (net.get('network_type'), 'CustomScreenformer'),
        ('CustomScreenformer', 'num_layers'): (cs.get('num_layers'), 1),
        ('CustomScreenformer', 'input_mode'): (cs.get('input_mode', 'sincos_s'), 'lj12'),
        ('CustomScreenformer', 'trim_fixed'): (cs.get('trim_fixed', False), True),
        ('CustomScreenformer', 'use_layer_norm'): (cs.get('use_layer_norm'), True),
        ('CustomScreenformer', 'mlp_activation_fct'): (cs.get('mlp_activation_fct'), 'GELU'),
        ('CustomScreenformer', 'use_softmask'): (cs.get('use_softmask'), True),
        ('CustomScreenformer', 'use_gate'): (cs.get('use_gate'), True),
        ('CustomScreenformer', 'use_tanh_norm'): (cs.get('use_tanh_norm'), True),
        ('CustomScreenformer', 'use_output_scale'): (cs.get('use_output_scale'), True),
        ('CustomScreenformer', 'use_fixed_value'): (cs.get('use_fixed_value', False), False),
        ('CustomScreenformer', 'init_softmask_centres'): (cs.get('init_softmask_centres'), None),
        ('CustomScreenformer', 'init_softmask_rings'): (cs.get('init_softmask_rings'), None),
        ('CustomScreenformer', 'lj_jastrow'): (cs.get('lj_jastrow'), True),
        ('network', 'bias_orbitals'): (net.get('bias_orbitals'), False),
        ('network', 'jastrow'): (str(net.get('jastrow')).upper() in ('NONE', 'DEFAULT'), True),
        ('network', 'make_envelope_fn'): (str(net.get('make_envelope_fn')).endswith('make_null_envelope'), True),
        ('network', 'include_r_ae'): (net.get('make_feature_layer_kwargs', {}).get('include_r_ae'), False),
        ('optim', 'clip_median'): (cfg['optim'].get('clip_median'), False),
        ('optim', 'center_at_clip'): (cfg['optim'].get('center_at_clip'), True),
    }
    bad = [f"{a}.{b} = {v!r}（lj3 は {w!r}）" for (a, b), (v, w) in want.items() if v != w]
    if bad:
        raise ValueError(f"lj3 の設定の run ではない {folder}：" + "、".join(bad))


def load_run(folder, hf_dir=None, checkpoint=None):
    """run のフォルダを読む。返り値 (model, params, walkers)。

    model    ：定数の辞書（中身は下の dict(...) の各行のコメント）。
    params   ：学習したパラメータ（本体と同じ入れ子の辞書。⑤ の各ファイルがどの葉を使うかを書いている）。
    walkers  ：チェックポイントのウォーカー {'x': (W, 2N) 位置, 's': (W, N) 擬スピン ±1, 'atom': (2,) 原点}。
    hf_dir   ：HF の表を置いたフォルダ（config の lj_hf_path は計算サーバでの相対パスなので、ファイル名だけ使う）。
    """
    cfg = json.load(open(os.path.join(folder, 'config.json')))
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, 'qmcjax_ckpt_*.npz')))[-1]
    z = np.load(ck, allow_pickle=True)
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a)[0]), z['params'].item())
    data = z['data'].item()
    walkers = {'x': np.asarray(data['positions'])[0], 's': np.asarray(data['spins'])[0],
               'atom': np.asarray(data['atoms'])[0][0][0]}

    check_lj3(cfg, folder)
    net, cs = cfg['network'], cfg['network']['CustomScreenformer']
    ek = cfg['system']['make_local_energy_kwargs']
    A = _lattice(cfg)
    r_s = float(ek['potential_kwargs']['interaction_energy_scale'])
    lam, delta = float(ek['lam']), float(ek['delta'])
    hf_path = cs['lj_hf_path'] if hf_dir is None else os.path.join(hf_dir, os.path.basename(cs['lj_hf_path']))
    k, G, c = read_hf_table(hf_path)
    # 本体のネットワークは config のカスプの値と λ（lj_cusps、lj_lam）を使う。式（cusp_values）と同じであることを確かめる。
    cusps = tuple(float(v) for v in cs['lj_cusps'])
    assert np.allclose(cusps, cusp_values(r_s, lam, delta), rtol=1e-12, atol=0.0)
    assert float(cs['lj_lam']) == lam
    model = dict(
        N=int(cfg['system']['electrons'][0]),       # 電子の数（擬スピンは成分なので nspins = (N, 0)）
        A=A, Ainv=np.linalg.inv(A), metric=A.T @ A,  # セルの並進ベクトル（列）、その逆、計量 A^T A
        d_ref=max_periodic_distance(A),             # 窓の物差し
        harmonics=np.asarray(cs['input_harmonics'], dtype=np.float64),   # 結晶の G を G·r = 2π n·f の整数の組 n で
        heads=int(cs['num_heads']), attn_dim=int(cs['attn_dim']), value_dim=int(cs['value_dim']),
        ndet=int(net['determinants']),
        lam=lam, delta=delta, r_s=r_s,              # λ、上のバンドを抑えるエネルギー Δ、クーロンの強さ r_s
        eps=float(cs['lj_net_scale']),              # 軌道のうちネットワークの部分の倍率 ε
        cusps=cusps,                                # Jastrow のカスプ γ（++、--、+-）
        hf_k=k, hf_G=G, hf_c=c,                     # ① の HF の表
        hf_path=hf_path,
        checkpoint=ck,
        mcmc_width=float(np.asarray(z['mcmc_width']).reshape(-1)[0]),   # チェックポイントの時点の MCMC の幅 w
    )
    return model, params, walkers
