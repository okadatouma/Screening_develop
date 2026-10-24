"""実験checkpointの独立JIT診断。旧解析・キャッシュへ書き込まない。"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import networks
from screening_analysis.head_contrib import projected_heads
from .network import build
from .run import kernels, restore_state


def minimum_image_distance(points, centers, lattice):
    """min_{a,n∈Z²}|r−center_a−L n|。最小像Euclidean、Attention周期ノルムとは別。

    斜交セルではfractional成分のroundだけでは最短でない。特異値から探索上限を得る。
    入出力[N,2],[K,2],[2,2] → [N]。長さはr_s a_B。
    """
    L = np.asarray(lattice, float)
    if L.shape != (2, 2) or not np.isfinite(L).all() or abs(np.linalg.det(L)) < 1e-12:
        raise ValueError('coreの格子は非特異な2×2行列が必要です')
    dx = np.asarray(points)[:, None, :] - np.asarray(centers)[None, :, :]
    frac = dx @ np.linalg.inv(L).T
    residual = frac - np.rint(frac)
    best = np.linalg.norm(residual @ L.T, axis=-1)
    # 改善候補は|residual−n| <= best/sigma_min内に限られる。
    extent = int(np.ceil(best.max()/np.linalg.svd(L, compute_uv=False)[-1]+.5))
    for n in itertools.product(range(-extent, extent+1), repeat=2):
        best = np.minimum(best, np.linalg.norm((residual-n) @ L.T, axis=-1))
    return best.min(axis=1)


def core_statistics(similarity, alpha, weights, window, positions, spins, lattice, centers, radius):
    """Query iのみcore内、j≠i、同符号対。全標本で分子/分母を合算する。

    C_h^{core,ss}=Σ_bij core_i 1[s_i=s_j=s] 1[i≠j] m_hij C_hij / Σ_bij mask m_hij。
    ΔC=C^{--}−C^{++}。αと実集約Wも同じWindow重みの別統計として返す。
    有効対のないHeadはNaN（ゼロではない）。Head出力寄与を含めない。
    配列は[B,H,N,N]、positions[B,N,2]、spins[B,N]。
    """
    if radius <= 0 or not np.isfinite(radius): raise ValueError('core半径は有限かつ正です')
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('core中心は有限の[K,2] Cartesian座標です')
    inside = np.stack([minimum_image_distance(p, centers, lattice) <= radius for p in positions])
    n = spins.shape[1]; result = {'core_query_count': inside.sum()}
    for sign, label in ((1, 'pp'), (-1, 'mm')):
        mask = inside[:, :, None] & (spins[:, :, None] == sign) & (spins[:, None, :] == sign)
        mask &= ~np.eye(n, dtype=bool)[None]
        wm = window * mask[:, None, :, :]
        denominator = wm.sum(axis=(0, 2, 3))
        result['core_window_mass_'+label] = denominator
        result['core_pair_count_'+label] = mask.sum()
        for name, value in (('C', similarity), ('alpha', alpha), ('W_aggregation', weights)):
            numerator = (wm*value).sum(axis=(0, 2, 3))
            result['core_'+name+'_'+label] = np.divide(numerator, denominator,
                out=np.full_like(numerator, np.nan), where=denominator > 0)
    for name in ('C', 'alpha', 'W_aggregation'):
        result['core_delta_'+name] = result['core_'+name+'_mm']-result['core_'+name+'_pp']
    return result


def blocking(values):
    """[time,walker]からwalker別block平均のSEM。plateau未確認は明示する。"""
    x = np.asarray(values, float); sizes = []; errors = []
    b = 1
    while x.shape[0]//b >= 8:
        blocks = x[:(len(x)//b)*b].reshape(-1, b, x.shape[1]).mean(axis=1)
        errors.append(float(blocks.std(ddof=1)/np.sqrt(blocks.size))); sizes.append(b); b *= 2
    sem = max(errors[-3:]) if errors else None
    plateau = len(errors) >= 3 and min(errors[-3:]) > 0 and max(errors[-3:])/min(errors[-3:]) < 1.25
    return dict(mean=float(x.mean()), sem_blocking=sem, block_sizes=sizes, sem_by_block=errors,
                plateau=plateau, samples=int(x.size), warning=None if plateau else 'blocking plateau未確認')


def load(folder):
    manifest = json.loads((folder/'manifest.json').read_text()); c = manifest['config']
    if manifest['schema'] != 'screening-window-v1': raise ValueError('未対応の実験schema')
    if c['hf'] and hashlib.sha256(Path(c['hf']).read_bytes()).hexdigest() != c['hf_sha256']:
        raise ValueError('HF表が実験時から変化しています')
    jax.config.update('jax_enable_x64', c['dtype'] == 'float64')
    _, data, params, _, width, key = restore_state(folder, c['batch'])
    return manifest, c, data, params, width, key


def analyze(args):
    manifest, c, data, params, width, key = load(args.run)
    net, diagnose, dref = build(c)
    if (args.core_centers is None) != (args.core_radius is None):
        raise ValueError('--core-centersと--core-radiusは両方必要です')
    centers = None if args.core_centers is None else json.loads(args.core_centers)
    sample, evaluate = kernels(c, net)
    evaluate = jax.jit(evaluate)
    for _ in range(args.burn):
        key, k = jax.random.split(key); data, _ = sample(params, data, k, width)
    diagnostic = jax.jit(diagnose)
    # 独立標本数を装わない。各測定でMCMCを進め、時系列のまま記録する。
    values = {}; measured_energies = []; coordinates = []; spin_samples = []
    for t in range(args.measure):
        key, k, ke = jax.random.split(key, 3); data, _ = sample(params, data, k, width)
        _, aux = evaluate(params, ke, data); measured_energies.append(np.asarray(aux.local_energy))
        for b in range(min(args.walkers, c['batch'])):
            _, records, features = diagnostic(params, data.positions[b], data.spins[b], data.atoms[b], data.charges[b])
            coordinates.append(np.asarray(data.positions[b]).reshape(c['n'], 2)); spin_samples.append(np.asarray(data.spins[b]))
            for l, rec in enumerate(records):
                ap = params['layers']['attention'][l]; a = rec['attention']; prefix = f'layer{l}_'
                heads, bias = projected_heads(a['gate_output'], ap['attn_output'], c['heads'], jnp.exp(ap['out_scale']))
                for name, v in dict(C=a['similarity'], alpha=a['alpha'], window=a['window'],
                        W_window=a['weights_window'], W_aggregation=a['aggregation_weights'],
                        head_contribution=heads, shared_bias=bias, attention_output=rec['attention_output'],
                        **rec['window_details']).items():
                    values.setdefault(prefix+name, []).append(np.asarray(v))
    arrays = {name: np.stack(v) for name, v in values.items()}
    arrays.update(positions=np.stack(coordinates), spins=np.stack(spin_samples), local_energy=np.stack(measured_energies))
    from periodicwave import networks as nw
    geometry = nw.make_periodic_construct_input_features(jnp.asarray(c['lattice']))
    distance = np.stack([np.asarray(geometry(p.reshape(-1), jnp.zeros((1, 2)), ndim=2)[3][..., 0]) for p in arrays['positions']])
    for l in range(c['layers']):
        p = f'layer{l}_'; w = arrays[p+'W_aggregation']
        denom = w.sum(axis=(0, 2, 3))
        arrays[p+'dbar'] = np.divide((w*distance[:, None]).sum(axis=(0, 2, 3)), denom,
            out=np.full_like(denom, np.nan), where=denom > 0)
        off = w*(1-np.eye(c['n']))[None, None]
        denoff = off.sum(axis=(0, 2, 3))
        arrays[p+'dbar_offdiag'] = np.divide((off*distance[:, None]).sum(axis=(0, 2, 3)), denoff,
            out=np.full_like(denoff, np.nan), where=denoff > 0)
        power = np.mean(np.sum(arrays[p+'head_contribution']**2, axis=-1), axis=(0, 1))
        arrays[p+'q_head_mean_squared_norm'] = power/power.sum() if power.sum() else np.full_like(power, np.nan)
        if p+'scales_pair' in arrays:
            arrays[p+'delta_w_mm_minus_pp'] = arrays[p+'scales_pair'][0, 1]-arrays[p+'scales_pair'][0, 0]
        if centers is not None:
            core = core_statistics(arrays[p+'C'], arrays[p+'alpha'], w, arrays[p+'window'],
                    arrays['positions'], arrays['spins'], c['lattice'], centers, args.core_radius)
            arrays.update({p+k: v for k, v in core.items()})
    report = dict(version='window-analysis-v1', manifest=manifest, burn=args.burn, measure=args.measure,
        walkers=min(args.walkers, c['batch']), energy_real=blocking(arrays['local_energy'].real),
        energy_imag=blocking(arrays['local_energy'].imag), d_ref=dref,
        core=None if centers is None else dict(centers_cartesian=centers, radius=args.core_radius,
            distance='minimum-image Euclidean, supercell L, length r_s*a_B', query_only=True,
            exclude_self=True, weighting='Window m; pooled numerator/denominator', delta='mm minus pp'),
        head_power='mean ||c_h||^2 normalized across heads; shared bias excluded',
        warning='Diagnostic uses independent JIT. C, alpha, W_aggregation are distinct. No topology inference.')
    if args.radii:
        report['cutoff_scan'] = cutoff_scan(c, params, data, width, key, args)
    if args.compare:
        other_report = json.loads(args.compare.with_suffix('.json').read_text())
        if report['core'] is None or other_report['core'] != report['core']:
            raise ValueError('WC/AHC core比較には同じ明示core定義が必要です')
        other_config = other_report['manifest']['config']
        keys = ('lam', 'rs', 'delta', 'lattice', 'n', 'input', 'harmonic', 'window', 'heads',
                'w_min', 'w_max', 'w_first', 'min_gap_fraction', 'transition', 'basis_edges',
                'pair_windows', 'pair_max_log_shift', 'trim_pow', 'value', 'gate', 'layers', 'dtype',
                'layer_norm', 'width', 'attn_dim', 'value_dim', 'determinants', 'output_scale',
                'net_scale', 'jastrow', 'coulomb', 'freeze_window')
        for k in keys:
            if c[k] != other_config[k]: raise ValueError(f'比較条件が異なります: {k}')
        with np.load(args.compare) as old:
            for l in range(c['layers']):
                key_name = f'layer{l}_core_delta_C'
                arrays[key_name+'_this_minus_other'] = arrays[key_name]-old[key_name]
        report['comparison'] = dict(other=str(args.compare.resolve()),
            this_label=c['label'], other_label=other_config['label'],
            definition='this ΔC_core minus other ΔC_core; matched λ/core/window conditions; learned widths may differ')
    # 両ファイルの存在を先に確認。旧結果を上書きしない。
    out = args.out or args.run/'window_analysis_v1.npz'; sidecar = out.with_suffix('.json')
    if out.exists() or sidecar.exists(): raise FileExistsError(f'別の--outを指定してください: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as f: np.savez_compressed(f, **arrays)
    with sidecar.open('x') as f: json.dump(report, f, indent=2, allow_nan=False)
    print(out); return out


def cutoff_scan(c, params, initial, width, key, args):
    """ψを変更するため半径ごとに再平衡化。元標本のEL再評価だけでは代用しない。"""
    radii = [float(r) for r in args.radii.split(',')]
    if any(r <= 0 or not np.isfinite(r) for r in radii) or not 0 < args.cut_beta < 1:
        raise ValueError('cut半径は正、cut-betaは(0,1)')
    if args.cut_layer is not None and not 0 <= args.cut_layer < c['layers']: raise ValueError('cut-layer範囲外')
    if args.cut_head is not None and not 0 <= args.cut_head < c['heads']: raise ValueError('cut-head範囲外')
    result = []
    for index, radius in enumerate([None]+sorted(set(radii))):
        changed = dict(c, cut_radius=radius, cut_beta=args.cut_beta, cut_layer=args.cut_layer, cut_head=args.cut_head)
        net, _, _ = build(changed); sample, evaluate = kernels(changed, net)
        evaluate = jax.jit(evaluate); data = initial; rng = jax.random.fold_in(key, 5000+index)
        for _ in range(args.cut_burn):
            rng, k = jax.random.split(rng); data, _ = sample(params, data, k, width)
        energies = []
        for _ in range(args.cut_measure):
            rng, k, ke = jax.random.split(rng, 3); data, _ = sample(params, data, k, width)
            _, aux = evaluate(params, ke, data); energies.append(np.asarray(aux.local_energy))
        el = np.stack(energies)
        item = dict(radius=radius, real=blocking(el.real), imag=blocking(el.imag))
        result.append(item)
    base = result[0]['real']
    for item in result[1:]:
        item['energy_cost'] = item['real']['mean']-base['mean']
        item['cost_sem'] = (None if base['sem_blocking'] is None or item['real']['sem_blocking'] is None else
            float(np.hypot(base['sem_blocking'], item['real']['sem_blocking'])))
    # 有意な悪化がない場合やplateau未確認時、R*を捏造しない。
    resolved = [v for v in result[1:] if v['cost_sem'] is not None and v['energy_cost'] > 3*v['cost_sem']
                and v['real']['plateau'] and base['plateau']]
    threshold = .1*max((v['energy_cost'] for v in resolved), default=0.)
    acceptable = [v['radius'] for v in result[1:] if resolved and v['cost_sem'] is not None and v['energy_cost']+3*v['cost_sem'] <= threshold
                  and v['real']['plateau']]
    return dict(results=result, cut_layer=args.cut_layer, cut_head=args.cut_head, beta=args.cut_beta,
                burn=args.cut_burn, measure=args.cut_measure, Rstar=min(acceptable) if acceptable else None,
                definition='smallest measured R with cost+3SEM <= 10% max resolved positive cost; no interpolation',
                warning='finite scan, variational cutoff response, not a screening length or energy decomposition')


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path); p.add_argument('--burn', type=int, default=100)
    p.add_argument('--compare', type=Path, help='他方の解析npz。同じλ/core/window条件を検査して差を保存')
    p.add_argument('--measure', type=int, default=32); p.add_argument('--walkers', type=int, default=8)
    p.add_argument('--core-centers', help='Cartesian座標のJSON [[x,y],...]。単位r_s*a_B')
    p.add_argument('--core-radius', type=float)
    p.add_argument('--radii', help='R*用のcut半径リスト。各半径で別途MCMC、計算負荷大')
    p.add_argument('--cut-layer', type=int); p.add_argument('--cut-head', type=int)
    p.add_argument('--cut-beta', type=float, default=.3)
    p.add_argument('--cut-burn', type=int, default=200); p.add_argument('--cut-measure', type=int, default=256)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.measure, args.walkers, args.cut_measure) <= 0 or min(args.burn, args.cut_burn) < 0:
        raise ValueError('測定数/ウォーカー数は正、burnは非負です')
    analyze(args)
