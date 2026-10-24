"""窓研究の単一GPU/CPU入口。物理・MCMC・VMC勾配は既存の実装を使用する。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from periodicwave import checkpoint, loss, mcmc, networks, train_screen
from periodicwave.pbc import lambda_jellium, lattices
from .network import build

SCHEMA = 'screening-window-v1'
PRESETS = {'A': ('E2', 'W1', 'learned', 'learned', False),
           'B': ('E3', 'W2', 'pair', 'position', False),
           'C': ('E2', 'W3', 'learned', 'learned', False),
           'D': ('E3', 'W2', 'pair', 'position', True)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='N4・3更新。HFなしの配線検証用、物理的な相の検証ではない')
    p.add_argument('--preset', choices=PRESETS, default='A')
    p.add_argument('--input', choices=['E0', 'E1', 'E2', 'E3'])
    p.add_argument('--window', choices=['W0', 'W1', 'W2', 'W3'])
    p.add_argument('--value', choices=['learned', 'pair'])
    p.add_argument('--gate', choices=['learned', 'position', 'off'])
    p.add_argument('--pair-windows', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--layer-norm', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--jastrow', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--freeze-window', action='store_true')
    p.add_argument('--coulomb', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--hf', type=Path)
    p.add_argument('--hf-meta', type=Path, help='HF表の作成条件JSON（lam,rs,delta,n,lattice,label,sha256）')
    p.add_argument('--label', choices=['WC', 'AHC'], help='HF初期状態のラベル。最終状態の相を意味しない')
    p.add_argument('--init-spins', choices=['up', 'random'], default='up')
    p.add_argument('--lam', type=float)
    p.add_argument('--delta', type=float)
    p.add_argument('--rs', type=float, default=15.)
    p.add_argument('--n', type=int, default=36)
    for name, default in [('layers', 1), ('heads', 16), ('width', 64), ('attn-dim', 16),
                          ('value-dim', 16), ('determinants', 4), ('harmonic', 6),
                          ('batch', 128), ('iterations', 20000), ('warmup', 1000),
                          ('window-ramp', 100), ('burn', 100), ('mcmc-steps', 10),
                          ('spin-flips', 3), ('flip-batch', 4), ('chunk', 4), ('save-every', 1000),
                          ('seed', 0), ('ewald-radius', 5)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('w-min', .25), ('w-first', .45), ('w-max', None),
                          ('min-gap-fraction', 0.), ('transition', .1),
                          ('pair-max-log-shift', .15), ('net-scale', .05),
                          ('output-scale', 0.),
                          ('lr', .01), ('lr-delay', 20000.), ('lr-decay', 1.),
                          ('window-ratio', .05), ('move-width', .2), ('clip', 0.)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--basis-edges', help='0,a,sqrt(3)a,2a,dmax等の数値をカンマ区切り。単位r_s a_B')
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--laplacian', choices=['default', 'folx'], default='folx')
    p.add_argument('--trim-pow', type=int, choices=[2, 4], default=2)
    p.add_argument('--require-gpu', action='store_true', help='大学でCPUへ意図せずfallbackした場合に停止')
    p.add_argument('--validate-only', action='store_true', help='HF/設定/初期化だけ検査。学習・出力作成なし')
    return p


def configuration(args):
    # CLIから実効設定を確定する段階。HF表のλは表自体から読めないため、作成JSONと照合する。
    # smokeの小規模条件は配線検証用であり、本計算の物理的な収束条件ではない。
    c = {k: v for k, v in vars(args).items() if k not in ('out', 'resume', 'require_gpu', 'validate_only')}
    for name, v in zip(('input', 'window', 'value', 'gate', 'pair_windows'), PRESETS[args.preset]):
        if c[name] is None: c[name] = v
    if c['smoke']:
        c.update(n=4, heads=2, width=8, attn_dim=2, value_dim=2, determinants=1,
                 batch=4, iterations=3 if c['iterations'] == 20000 else c['iterations'], warmup=1, window_ramp=1, burn=2,
                 mcmc_steps=1, spin_flips=1, chunk=2, save_every=1, rs=1.,
                 delta=1., w_min=.1, w_first=.2, coulomb=False, layer_norm=False)
        c['init_spins'] = 'random'
        if c['lam'] is None: c['lam'] = .3
    if c['lam'] is None: raise ValueError('--lamを明示してください（HF作成時と同じ値）')
    if c['delta'] is None: c['delta'] = c['rs']
    if not c['smoke'] and not c['hf']: raise ValueError('本計算には--hfが必要です。HFなしの動作確認は--smokeのみ')
    if int(np.sqrt(c['n']))**2 != c['n']: raise ValueError('この入口の三角セルではnは平方数です')
    if min(c[k] for k in ('layers', 'heads', 'width', 'batch', 'iterations', 'save_every', 'mcmc_steps')) < 1:
        raise ValueError('層数・次元・batch・iterations・save-every・mcmc-stepsは正にしてください')
    if c['rs'] <= 0 or c['transition'] <= 0 or c['window_ratio'] < 0 or c['warmup'] < 0:
        raise ValueError('rs/transitionは正、window-ratio/warmupは非負です')
    a = np.sqrt(2*np.pi/np.sqrt(3))
    L, crystal = lattices._triangular_lattice_vecs_periodic_potential(a, c['n'])
    c['lattice'] = np.asarray(L).tolist(); c['crystal_lattice'] = np.asarray(crystal).tolist()
    from periodicwave.CustomScreenformer import _max_periodic_distance
    dref = _max_periodic_distance(np.asarray(L)); hi = c['w_max'] or dref
    if c['basis_edges']:
        c['basis_edges'] = [float(x) for x in c['basis_edges'].split(',')]
    elif c['smoke']:
        c['basis_edges'] = np.linspace(0, hi, 5).tolist()
    else:
        c['basis_edges'] = [0, a, np.sqrt(3)*a, 2*a, hi]
        if not np.all(np.diff(c['basis_edges']) > 0) and c['window'] == 'W3':
            raise ValueError('セルが小さいため物理基底が入りません。--basis-edgesを明示してください')
    c['trim_max'] = 2.; c['cut_radius'] = None; c['cut_beta'] = .3
    if c['hf']:
        c['hf'] = str(c['hf'].resolve())
        c['hf_sha256'] = hashlib.sha256(Path(c['hf']).read_bytes()).hexdigest()
        table = np.loadtxt(c['hf'])
        if table.ndim != 2 or table.shape[1] != 6 or not np.isfinite(table).all():
            raise ValueError('HF表は有限な6列 kx ky Gx Gy Re(c) Im(c) が必要です')
        if not c['hf_meta'] or not c['label']:
            raise ValueError('--hf-metaと--labelが必要です。6列係数だけからλや相ラベルは判定できません')
        c['hf_meta'] = str(c['hf_meta'].resolve())
        meta = json.loads(Path(c['hf_meta']).read_text())
        for key in ('lam', 'rs', 'delta', 'n', 'label'):
            if meta.get(key) != c[key]: raise ValueError(f'HF作成条件の不一致/不足: {key}')
        if meta.get('sha256') != c['hf_sha256']:
            raise ValueError('HF表のsha256がメタデータと不一致です')
        if not np.allclose(np.asarray(meta.get('lattice', [])), L, rtol=0, atol=1e-12):
            raise ValueError('HF作成時の格子（列ベクトル）が異なります')
        from periodicwave.lj_orbitals import load_hf_table
        k, G, coeff = load_hf_table(c['hf'])
        if len(k) != c['n']: raise ValueError('HF軌道数とnが不一致です')
        for block in table.reshape(len(k), len(G), 6):
            if not np.allclose(block[:, 2:4], G, atol=1e-10, rtol=0):
                raise ValueError('HF表で各kブロックのG列が不一致です')
        c['hf_metadata'] = meta
    else: c['hf_sha256'] = None
    return c


def optimizer(config, params):
    """窓だけwarmup中は状態も凍結。解放後、別Adam状態で小さい学習率を使う。"""
    c = config
    def lr(step): return c['lr']*(1+step/c['lr_delay'])**(-c['lr_decay'])
    def window_lr(step):
        ramp = jnp.minimum(1., (step+1)/max(c['window_ramp'], 1))
        return lr(step+c['warmup'])*c['window_ratio']*ramp
    labels = jax.tree_util.tree_map_with_path(
        lambda path, _: 'window' if any(getattr(k, 'key', None) == 'window' for k in path) else 'network', params)
    window = optax.conditionally_mask(optax.adam(window_lr),
        lambda step: (step >= c['warmup']) & (not c['freeze_window']))
    return optax.partition({'network': optax.adam(lr), 'window': window}, labels)


def kernels(config, net):
    # ここで実験モデルを既存の位置/spin MCMC、λ専用Hamiltonian、G1修正済みVMC勾配へ接続する。
    c = config
    def logpsi(p, x, s, a, z):
        phase, logabs = net.apply(p, x, s, a, z)
        return logabs + 1j*phase
    batch_logabs = jax.vmap(lambda p, x, s, a, z: net.apply(p, x, s, a, z)[1], in_axes=(None, 0, 0, 0, 0))
    sample = jax.jit(mcmc.make_mcmc_step(batch_logabs, c['batch'], steps=c['mcmc_steps'],
                        ndim=2, blocks=1, spin_flips=c['spin_flips']))
    energy = lambda_jellium.local_energy(net.apply, jnp.zeros(1), (c['n'], 0),
        lattice=jnp.asarray(c['lattice']), lam=c['lam'], delta=c['delta'],
        potential_type='Coulomb' if c['coulomb'] else 'none',
        convergence_radius=c['ewald_radius'], potential_kwargs={'interaction_energy_scale': c['rs']},
        flip_batch=c['flip_batch'],
        kinetic_kwargs={'laplacian_method': c['laplacian']})
    evaluate = loss.make_loss(logpsi, energy, complex_output=True,
        clip_local_energy=c['clip'], local_energy_chunk=c['chunk'])
    return sample, evaluate


def initial_data(c, key):
    # 一様配置から位置と擬スピンを共に平衡化する。結晶/相を初期配置へ仮定しない。
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (c['batch'], c['n'], 2)) @ jnp.asarray(c['lattice']).T
    s = jnp.where(jax.random.bernoulli(k2, shape=(c['batch'], c['n'])), 1., -1.)
    if c['init_spins'] == 'up': s = jnp.ones_like(s)
    return networks.FermiNetData(positions=x.reshape(c['batch'], -1), spins=s,
        atoms=jnp.zeros((c['batch'], 1, 2)), charges=jnp.zeros((c['batch'], 1)))


def save_state(folder, t, data, params, state, width, key):
    # 既存checkpoint形式＋独立RNG sidecar。両方存在する時だけ再開可能。
    path = folder / f'qmcjax_ckpt_{t:06d}.npz'
    if path.exists(): raise FileExistsError(path)
    expand = lambda tree: jax.tree.map(lambda x: np.asarray(x)[None], tree)
    # partitionのNamedTuple型をnp.asarrayでlist化させない。既存loaderはobject scalarを復元できる。
    state_object = np.empty((), dtype=object)
    state_object[()] = expand(state)
    checkpoint.save(str(folder), t, expand(data), expand(params), state_object, jnp.asarray([width]))
    with (folder / f'rng_{t:06d}.npy').open('xb') as f: np.save(f, np.asarray(key))


def restore_state(folder, batch):
    paths = sorted(folder.glob('qmcjax_ckpt_*.npz'))
    complete = [p for p in paths if (folder/f'rng_{int(p.stem.split("_")[-1]):06d}.npy').exists()]
    if not complete: raise ValueError('RNG sidecarを伴う完了checkpointがありません')
    t, data, p, state, width = checkpoint.restore(str(complete[-1]), batch)
    strip = lambda tree: jax.tree.map(lambda x: jnp.asarray(x)[0], tree)
    key = jnp.asarray(np.load(folder/f'rng_{t-1:06d}.npy'))
    return t, strip(data), strip(p), strip(state), float(width[0]), key


def metadata(config):
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root/'periodicwave').rglob('*.py'))
    versions = {}
    for name in ('jax', 'jaxlib', 'folx', 'optax', 'kfac-jax', 'numpy'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'unavailable'
    return dict(schema=SCHEMA, config=config, versions=versions, python=platform.python_version(),
        environment={k: v for k, v in os.environ.items() if k.startswith(('JAX_', 'XLA_', 'PW_'))
                     or k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')},
        matmul_precision=str(jax.config.jax_default_matmul_precision), devices=[str(d) for d in jax.devices()],
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        units='length=r_s*a_B; energy=total code units, multiply by 1/r_s^2 for total Ha',
        initialization=f'uniform positions, {config["init_spins"]} pseudospins, followed by burn-in',
        diagnostic='independent JIT; may differ from forward by XLA rounding')


def run(args):
    c = configuration(args)
    jax.config.update('jax_enable_x64', c['dtype'] == 'float64')
    if args.require_gpu and jax.default_backend() != 'gpu': raise RuntimeError('--require-gpu: JAXがGPUを認識していません')
    if jax.device_count() != 1: raise ValueError('この入口は単一deviceです。CUDA_VISIBLE_DEVICES等で1GPUを選択してください')
    if args.validate_only:
        net, _, dref = build(c)
        params = net.init(jax.random.PRNGKey(c['seed']))
        print(json.dumps(dict(config=c, d_ref=dref, parameters=sum(x.size for x in jax.tree.leaves(params))), indent=2))
        return
    folder = args.out.resolve()
    if args.resume:
        old = json.loads((folder/'manifest.json').read_text())
        for key, value in old['config'].items():
            if key != 'iterations' and c[key] != value: raise ValueError(f'再開設定が違います: {key}')
        current = metadata(c)
        # コメントだけの編集もソースハッシュは変わる。既存runの再開条件を黙って緩めない。
        if current['source_sha256'] != old['source_sha256'] or current['versions'] != old['versions']:
            raise ValueError('再開時のソース/依存バージョンが異なります。旧runを変更せず新規実験を作成してください')
        if current['matmul_precision'] != old['matmul_precision']:
            raise ValueError('再開時のmatmul precisionが異なります')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        with (folder/'manifest.json').open('x') as f: json.dump(metadata(c), f, indent=2)
    net, _, _ = build(c)
    key, kp, kd = jax.random.split(jax.random.PRNGKey(c['seed']), 3)
    params = net.init(kp); opt = optimizer(c, params)
    state = opt.init(params); data = initial_data(c, kd); start = 0; width = c['move_width']
    if args.resume: start, data, params, state, width, key = restore_state(folder, c['batch'])
    sample, evaluate = kernels(c, net)
    update = jax.jit(train_screen.make_opt_update_step(evaluate, opt))
    if not args.resume:
        for _ in range(c['burn']):
            key, ks = jax.random.split(key); data, _ = sample(params, data, ks, width)
    name = f'stats_resume_{start:06d}.csv' if args.resume else 'stats.csv'
    fields = ['step', 'energy_total_code', 'energy_per_electron_Ha', 'imag_mean',
              'variance_abs_complex', 'variance_real', 'position_acceptance', 'spin_acceptance', 'seconds']
    with (folder/name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for t in range(start, c['iterations']):
            before = time.perf_counter(); key, ks, ke = jax.random.split(key, 3)
            data, accept = sample(params, data, ks, width)
            params, state, energy, aux = update(params, data, state, ke)
            el = np.asarray(aux.local_energy); accept = np.atleast_1d(np.asarray(accept))
            if not np.isfinite(el).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
                raise FloatingPointError(f'非有限値: step={t}。直前のcheckpointは保持しました')
            row = dict(zip(fields, [t, float(np.real(energy)), float(np.real(energy))/c['n']/c['rs']**2,
                float(el.imag.mean()), float(np.mean(abs(el-el.mean())**2)), float(np.var(el.real)),
                float(accept[0]), float(accept[1]) if len(accept)>1 else None, time.perf_counter()-before]))
            writer.writerow(row); f.flush(); print(json.dumps(row), flush=True)
            if (t+1) % c['save_every'] == 0 or t+1 == c['iterations']:
                save_state(folder, t, data, params, state, width, key)
    return folder


if __name__ == '__main__':
    run(parser().parse_args())
