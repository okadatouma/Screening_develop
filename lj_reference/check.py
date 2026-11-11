# -*- coding: utf-8 -*-
"""参照用コードと本体の突き合わせ（lj3 の run のチェックポイントで）。

  cd <リポジトリの根>
  JAX_PLATFORMS=cpu python -m lj_reference.check <run のフォルダ> [<run のフォルダ> ...] --hf-dir <HF の表のフォルダ>

同じパラメータ・同じウォーカーで、参照用コード（lj_reference/）と本体（periodicwave/）の両方を計算して差を出す。
倍精度（x64）で計算する。本体の学習は単精度だが、ここでは式が同じかどうかを見たいので、丸めの差を小さくしておく。

  段    比べる量                 本体の関数
  ⑤-1  1 電子の入力 12 成分      networks.make_periodic_construct_input_features、pbc/feature_layer.py、
                                CustomScreenformer.one_electron_inputs
        対の距離 d_ij             同上（r_ee）
  ⑤-2  特徴 h_i                 CustomScreenformer.make_screenformer_layers の apply
  ⑤-3  軌道の行列 Φ^k           network.orbitals（lj_orbitals.make_lj_orbitals の apply）
  ⑤-4  ln|ψ| と位相 φ           network.apply（network_blocks.logdet_matmul）
  ④-1  MCMC の 1 回の呼び出し     mcmc.make_mcmc_step（同じ乱数の鍵、同じ幅）
  ④-2  E_L                      pbc/lambda_jellium.local_energy（学習と同じ folx のラプラシアン）
        E_L の項                 pbc/lambda_jellium.make_energy_components
  ④-3  E と勾配 ∂E/∂θ           loss.make_loss を jax.value_and_grad(has_aux=True) したもの（学習のコードと同じ呼び方）
"""
import os

# 本体の窓と Trim の形、窓の固定は import のときに環境変数で決まる。lj3 の run（ws_pw_kit/pw_lj3.sh）にそろえる。
os.environ['PW_WIN_SHAPE'] = 'smooth7'
os.environ['PW_TRIM_POW'] = '4'
os.environ['PW_FREEZE_W'] = '1'

import argparse
import json
import time

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from periodicwave import CustomScreenformer as CS
from periodicwave import envelopes, mcmc, networks
from periodicwave import loss as qmc_loss
from periodicwave.pbc import feature_layer as FL
from periodicwave.pbc import lambda_jellium

from lj_reference import attention, gradient, hamiltonian, inputs, orbitals, sampling, wavefunction
from lj_reference import model as M

assert CS.WIN_SHAPE == 'smooth7' and CS.TRIM_POW == 4 and CS.FREEZE_W

BATCH_AXES = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)


def build_production(folder, hf_dir, model):
    """本体の部品を、学習（train_screen.train）と同じ引数で作る。"""
    cfg = json.load(open(os.path.join(folder, 'config.json')))
    A = jnp.asarray(model['A'])
    nspins = tuple(int(v) for v in cfg['system']['electrons'])
    charges = jnp.zeros((1,))
    nc = cfg['network']
    kw = dict(nc['CustomScreenformer'])
    kw['lj_hf_path'] = model['hf_path']
    fl = FL.make_pbc_feature_layer(natoms=1, nspins=nspins, ndim=2, lattice=A,
                                   include_r_ae=nc['make_feature_layer_kwargs']['include_r_ae'])
    net = CS.make_fermi_net(nspins, charges, ndim=2, determinants=nc['determinants'],
                            envelope=envelopes.make_null_envelope(), feature_layer=fl,
                            jastrow=nc['jastrow'], jastrow_kwargs=nc['jastrow_kwargs'],
                            bias_orbitals=nc['bias_orbitals'], complex_output=True, pbc_lattice=A, **kw)
    ek = dict(cfg['system']['make_local_energy_kwargs'], lattice=A)
    local_energy = lambda_jellium.local_energy(f=net.apply, charges=charges, nspins=nspins,
                                               complex_output=True, use_scan=False, **ek)
    components = lambda_jellium.make_energy_components(net.apply, A, model['lam'], model['delta'], model['r_s'])

    def log_network(*a):
        phase, mag = net.apply(*a)
        return mag + 1j * phase

    op = cfg['optim']
    total_energy = qmc_loss.make_loss(log_network, local_energy, clip_local_energy=op['clip_local_energy'],
                                      clip_from_median=op['clip_median'],
                                      center_at_clipped_energy=op['center_at_clip'], complex_output=True)
    batch_logabs = jax.vmap(lambda *a: net.apply(*a)[1], in_axes=(None, 0, 0, 0, 0))

    def make_mcmc(W):
        return mcmc.make_mcmc_step(batch_logabs, W, steps=cfg['mcmc']['steps'], ndim=2,
                                   blocks=cfg['mcmc']['blocks'], spin_flips=cfg['mcmc']['spin_flips'])

    construct = networks.make_periodic_construct_input_features(A)
    layers_apply = CS.make_screenformer_layers(nspins, 1, net.options)[1]

    def prod_inputs(x, s, atoms):                       # ⑤-1：入力 12 成分と d_ij
        ae, ee, r_ae, r_ee = construct(x, atoms, ndim=2)
        feat, _ = net.options.feature_layer.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        return CS.one_electron_inputs(net.options, feat, ae, s), r_ee[..., 0]

    def prod_features(params, x, s, atoms, charges):    # ⑤-2：特徴 h_i
        ae, ee, r_ae, r_ee = construct(x, atoms, ndim=2)
        return layers_apply(params['layers'], ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, spins=s, charges=charges)

    return dict(cfg=cfg, net=net, local_energy=local_energy, components=components,
                total_energy=total_energy, make_mcmc=make_mcmc, inputs=prod_inputs, features=prod_features)


def batch(x, s, atom):
    W = x.shape[0]
    return networks.FermiNetData(positions=x, spins=s, atoms=jnp.tile(atom[None, None, :], (W, 1, 1)),
                                 charges=jnp.zeros((W, 1)))


def diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(b)))


def check_run(folder, hf_dir=None, n_psi=64, n_el=4, n_mc=16, n_grad=8, seed=0, skip=()):
    model, params, walkers = M.load_run(folder, hf_dir=hf_dir)
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), params)
    X = jnp.asarray(walkers['x'], jnp.float64)
    S = jnp.asarray(walkers['s'], jnp.float64)
    atom = jnp.asarray(walkers['atom'], jnp.float64)
    P = build_production(folder, hf_dir, model)
    net = P['net']
    rows = []

    def add(step, name, n, d, unit=''):
        rows.append(dict(step=step, name=name, n=int(n), maxdiff=d[0], scale=d[1], unit=unit))
        print(f"  {step}  {name}：{n} 本、差の最大 {d[0]:.2e}（値の大きさ {d[1]:.3g}{unit}）", flush=True)

    over = lambda f: jax.jit(jax.vmap(f, in_axes=(None, 0, 0)))                    # 参照用（params, x, s）
    over_p = lambda f: jax.jit(jax.vmap(f, in_axes=(None, 0, 0, 0, 0)))            # 本体（params, x, s, atoms, charges）

    # ⑤ 1 回の ψ の計算
    x, s = X[:n_psi], S[:n_psi]
    d = batch(x, s, atom)
    p_in, p_d = jax.jit(jax.vmap(P['inputs']))(x, s, d.atoms)
    r_in = jax.vmap(lambda xx, ss: inputs.one_electron_inputs(model, xx, ss, atom))(x, s)
    r_d = jax.vmap(lambda xx: inputs.pair_distances(model, xx))(x)
    add('⑤-1', '1 電子の入力 12 成分', n_psi, diff(r_in, p_in))
    add('⑤-1', '対の距離 d_ij', n_psi, diff(r_d, p_d))
    p_h = over_p(P['features'])(params, x, s, d.atoms, d.charges)
    r_h = over(lambda p, xx, ss: attention.features(p, model, xx, ss, atom))(params, x, s)
    add('⑤-2', '特徴 h_i', n_psi, diff(r_h, p_h))
    p_phi = over_p(lambda *a: net.orbitals(*a)[0])(params, x, s, d.atoms, d.charges)
    r_phi = over(lambda p, xx, ss: orbitals.orbital_matrices(
        p, model, xx, ss, attention.features(p, model, xx, ss, atom), inputs.pair_distances(model, xx)))(params, x, s)
    add('⑤-3', '軌道の行列 Φ^k_ij', n_psi, diff(r_phi, p_phi))
    p_ph, p_la = over_p(net.apply)(params, x, s, d.atoms, d.charges)
    r_ph, r_la = over(lambda p, xx, ss: wavefunction.log_psi(p, model, xx, ss, atom))(params, x, s)
    add('⑤-4', 'ln|ψ|', n_psi, diff(r_la, p_la))
    dph = np.abs(np.angle(np.exp(1j * (np.asarray(r_ph) - np.asarray(p_ph)))))
    add('⑤-4', '位相 φ（2π を法として）', n_psi, (float(dph.max()), float(np.pi)), unit='、π')

    # ④-1 MCMC：同じ鍵と幅で 1 回呼ぶ（10 step、各 step で位置 1 回と擬スピン 3 回の提案）
    if 'mc' not in skip:
        x, s = X[:n_mc], S[:n_mc]
        key = jax.random.PRNGKey(seed + 1)
        w = model['mcmc_width']
        p_data, p_acc = jax.jit(P['make_mcmc'](n_mc))(params, batch(x, s, atom), key, jnp.asarray(w))
        r_x, r_s, r_pm, r_pf = jax.jit(
            lambda p, xx, ss, k: sampling.mcmc_step(p, model, xx, ss, atom, k, w))(params, x, s, key)
        add('④-1', 'MCMC 後の位置', n_mc, diff(r_x, p_data.positions))
        nbad = int(np.sum(np.asarray(r_s) != np.asarray(p_data.spins)))
        add('④-1', 'MCMC 後の擬スピンが違う電子の数', n_mc, (float(nbad), float(np.asarray(s).size)), unit=' 個中')
        add('④-1', '受理率（位置、反転）', n_mc, diff(jnp.stack([r_pm, r_pf]), p_acc))
        print(f"        受理率 本体 {np.asarray(p_acc).round(4).tolist()}、参照 {[round(float(r_pm), 4), round(float(r_pf), 4)]}"
              f"、幅 {w:.4g}、反転した擬スピン {int(np.sum(np.asarray(p_data.spins) != np.asarray(s)))} 個")

    e_l_fn = hamiltonian.make_local_energy(model)
    ref_el = over(lambda p, xx, ss: e_l_fn(p, xx, ss, atom))

    # ④-2 局所エネルギー
    if 'el' not in skip:
        x, s = X[:n_el], S[:n_el]
        d = batch(x, s, atom)
        keys = jax.random.split(jax.random.PRNGKey(seed), n_el)
        p_el, p_aux = jax.jit(jax.vmap(P['local_energy'], in_axes=(None, 0, BATCH_AXES)))(params, keys, d)
        p_c = jax.jit(jax.vmap(P['components'], in_axes=(None, BATCH_AXES)))(params, d)   # [T, K, C, F, V]
        r_el, r_t = ref_el(params, x, s)
        add('④-2', 'E_L（本体は学習と同じ folx）', n_el, diff(r_el, p_el))
        add('④-2', '運動 -Σ c_i ∇²ψ/ψ', n_el, diff(r_t['kinetic'], p_c[:, 0] + p_c[:, 1]))
        add('④-2', '擬スピン反転 iΔλ Σ_i …', n_el, diff(r_t['flips'], p_c[:, 3]))
        add('④-2', 'Δ n_-', n_el, diff(r_t['upper'], p_c[:, 2].real))
        add('④-2', 'クーロン（Ewald）', n_el, diff(r_t['coulomb'], p_aux[:, 0]))
        print(f"        E_L/N（参照）{np.round(np.asarray(r_el) / model['N'], 4).tolist()}")

    # ④-3 エネルギーと勾配
    if 'grad' not in skip:
        x, s = X[:n_grad], S[:n_grad]
        key = jax.random.PRNGKey(seed + 2)
        (p_E, _), p_g = jax.jit(jax.value_and_grad(P['total_energy'], argnums=0, has_aux=True))(
            params, key, batch(x, s, atom))
        r_el = ref_el(params, x, s)[0]
        r_E, r_g = jax.jit(lambda p, xx, ss, ee: gradient.energy_and_gradient(p, model, xx, ss, atom, ee))(
            params, x, s, r_el)
        add('④-3', 'E = <E_L>', n_grad, diff(r_E, p_E))
        lr = jax.tree_util.tree_leaves(r_g)
        lp = jax.tree_util.tree_leaves(p_g)
        gd = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(lr, lp))
        gs = max(float(np.max(np.abs(np.asarray(b)))) for b in lp)
        add('④-3', '勾配 ∂E/∂θ（全パラメータ）', n_grad, (gd, gs))
        rel = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(np.asarray(b))), 1e-300))
                  for a, b in zip(lr, lp) if np.max(np.abs(np.asarray(b))) > 0)
        add('④-3', '勾配（パラメータの組ごとの相対差の最大）', n_grad, (rel, 1.0), unit='、相対')
        om = (r_g['layers']['attention'][0]['mask_omega'], p_g['layers']['attention'][0]['mask_omega'])
        add('④-3', '窓の幅 ω の勾配（固定なので 0）', n_grad,
            (float(max(np.max(np.abs(np.asarray(om[0]))), np.max(np.abs(np.asarray(om[1]))))), 0.0))

    info = dict(folder=os.path.basename(os.path.normpath(folder)), checkpoint=os.path.basename(model['checkpoint']),
                hf=os.path.basename(model['hf_path']), lam=model['lam'])
    return info, rows


def main():
    ap = argparse.ArgumentParser(description='参照用コード（lj_reference）と本体（periodicwave）の突き合わせ')
    ap.add_argument('folders', nargs='+', help='lj3 の run のフォルダ（config.json と qmcjax_ckpt_*.npz がある所）')
    ap.add_argument('--hf-dir', default=None, help='HF の表（hf_N36_lam*_{wc,ahc}.txt）のフォルダ')
    ap.add_argument('--n-psi', type=int, default=64, help='⑤ を比べるウォーカーの数')
    ap.add_argument('--n-el', type=int, default=4, help='④-2 を比べるウォーカーの数')
    ap.add_argument('--n-mc', type=int, default=16, help='④-1 を比べるウォーカーの数')
    ap.add_argument('--n-grad', type=int, default=8, help='④-3 を比べるウォーカーの数')
    ap.add_argument('--skip', default='', help='飛ばす段をカンマで（mc, el, grad）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='結果を JSON で書く')
    args = ap.parse_args()
    skip = tuple(v.strip() for v in args.skip.split(',') if v.strip())
    out = []
    for f in args.folders:
        t0 = time.time()
        print(f"[{os.path.basename(os.path.normpath(f))}]", flush=True)
        info, rows = check_run(f, args.hf_dir, args.n_psi, args.n_el, args.n_mc, args.n_grad, args.seed, skip)
        info['seconds'] = round(time.time() - t0, 1)
        print(f"  （{info['checkpoint']}、HF {info['hf']}、{info['seconds']} 秒）", flush=True)
        out.append(dict(info, rows=rows))
    if args.out:
        json.dump(out, open(args.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
