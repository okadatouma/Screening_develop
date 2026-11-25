"""Read-only checkpoint diagnostics, caches, tables and Japanese reports.

python -m screening_analysis.physical_phase_diagnostics --config CONFIG --mode all
--mode report regenerates figures/text without JAX or checkpoints. The numerical
worker uses official diagnostic/load_model/phase_sensitivity APIs exclusively.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATCH_KEYS = ('positions', 'spins', 'atoms', 'charges')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(json_safe(rows))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files if k != 'metadata_json'}
        if 'metadata_json' in z:
            result['metadata'] = json.loads(str(z['metadata_json']))
        return result


def resolve(path, parent=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (parent/path).resolve()


def read_config(path):
    path = Path(path)
    if path.suffix == '.json':
        config = json.loads(path.read_text())
    else:
        import yaml
        config = yaml.safe_load(path.read_text())
    defaults = dict(samples=32, phase_samples=4, distance_bins=12, qmax=8, min_bin_count=20,
                    seed=20261102, coordinate_epsilon=1e-4, coordinate_samples=2,
                    ablation_samples=2, plot_python=sys.executable)
    unknown = set(config)-set(defaults)-{'experiment', 'output', 'runs'}
    if unknown:
        raise ValueError('Unknown analysis settings: '+str(sorted(unknown)))
    config = {**defaults, **config}
    for key in ('samples', 'distance_bins', 'qmax', 'min_bin_count'):
        if type(config[key]) is not int or config[key] < (2 if key=='min_bin_count' else 1):
            raise ValueError('positive integer '+key+' required')
    for key in ('phase_samples', 'coordinate_samples', 'ablation_samples'):
        if type(config[key]) is not int or not 0 <= config[key] <= config['samples']:
            raise ValueError(key+' must be in [0,samples]')
    if not np.isfinite(config['coordinate_epsilon']) or config['coordinate_epsilon']<=0:
        raise ValueError('positive coordinate_epsilon required')
    config['output'] = str(resolve(config['output']))
    return config


def discover(config):
    """The experiment manifest is the authority; no guessing from folder names."""
    if 'runs' in config:
        targets = config['runs']
    else:
        path = resolve(config['experiment']); experiment = json.loads(path.read_text())
        baseline = json.loads(Path(experiment['baseline_plan']).read_text())
        targets = [dict(heads=2, case=c, training=str(Path(baseline['root'])/c['id']/'training'),
                        study=str(Path(experiment['baseline_analysis'])/c['id']))
                   for c in baseline['cases'] if c['rs'] in (4, 25)]
        targets += [dict(t, study=str(path.parent/f"heads-{t['heads']}"/'connection-study'/t['case']['id']))
                    for t in experiment['tasks']]
    ids = [t['case']['id'] for t in targets]
    if len(ids) != len(set(ids)) or any(not identifier or Path(identifier).name!=identifier or identifier in ('.','..') for identifier in ids):
        raise ValueError('unique simple run IDs required')
    return targets


def input_id(batch, i):
    # Reuse the existing canonical exact-input fingerprint/alignment contract.
    from screening_analysis.ph_research import fingerprint
    return fingerprint(batch, i)


def selected_batch(data, count):
    count = min(count, len(data['positions']))
    ids = np.linspace(0, len(data['positions'])-1, count).astype(int)
    return {k: data[k][ids] for k in BATCH_KEYS}, ids


def source_hashes():
    paths = list((ROOT/'screening_analysis').glob('physical_phase_*.py'))
    paths += [ROOT/'periodicwave'/f for f in ('attention.py', 'CustomScreenformer.py', 'networks.py')]
    paths += [ROOT/'screening_analysis'/f for f in ('attn_probe.py', 'checkpoint_model.py', 'connection_study.py',
              'connection_audit.py', 'directed_loops.py', 'graph_wind.py', 'rs_observables.py', 'ph_research.py', 'lj_maps.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def validate_probe(p):
    """Independent NumPy definitions for this no-intervention checkpoint bank."""
    settings = p['metadata']['settings']
    r = p['r'][None, :, :, None, None]
    alpha = (np.maximum(p['S']-(1-r), 0)/r)**settings['TRIM_POW']
    error = float(np.max(abs(alpha-p['Alpha'])))
    window_product = p['Alpha']*p['Window']
    product_error = float(np.max(abs(window_product-p['W_window'])))
    aggregation = window_product.copy()
    if p['metadata']['fixed_value']:
        aggregation[..., np.arange(int(p['NE'])), np.arange(int(p['NE']))] = 0.
    aggregation_error = float(np.max(abs(aggregation-p['W'])))
    if max(error, product_error, aggregation_error)>1e-10:
        raise AssertionError('official Alpha/Window/aggregation definition mismatch')
    return dict(alpha_formula_max_error=error, window_product_max_error=product_error,
                aggregation_max_error=aggregation_error, pass_definitions=True)


def transformation_diagnostics(net, params, p, nspins):
    """Actual model permutation/PBC checks; reflection is an active diagnostic.

    Passive reversal of the geometric orientation is tested independently by
    unit tests. An actively reflected input need not preserve this Hamiltonian.
    """
    import jax
    import jax.numpy as jnp
    from periodicwave import CustomScreenformer as CS
    from screening_analysis.physical_phase_metrics import product_circulation, finite_mean
    from screening_analysis.directed_loops import oriented_triangles
    fn = jax.jit(CS.make_screenformer_diagnostic(tuple(nspins), net.options))
    def evaluate(x, s):
        _, layers = fn(params, jnp.asarray(x), jnp.asarray(s), jnp.asarray(p['atoms'][0]), jnp.asarray(p['charges'][0]))
        return np.stack([np.asarray(l['attention']['similarity']) for l in layers])
    n = int(p['NE']); x = p['positions'][0].reshape(n, 2); s = p['spins'][0]
    original = evaluate(x.ravel(), s)
    order = np.roll(np.arange(n)[::-1], 1)
    permuted = evaluate(x[order].ravel(), s[order])
    shifted = x.copy(); shifted[0] += p['lattice'][:, 0]
    translated = evaluate(shifted.ravel(), s)
    perr = float(np.max(abs(permuted-original[..., order, :][..., :, order])))
    terr = float(np.max(abs(translated-original)))
    if max(perr, terr)>1e-9:
        raise AssertionError('actual QK permutation/PBC check failed')
    reflected = x*np.array([-1., 1.]); mirror = evaluate(reflected.ravel(), s)
    t = oriented_triangles(x, p['lattice'])['triangles']
    tm = oriented_triangles(reflected, p['lattice'])['triangles']
    return dict(qk_permutation_max_error=perr, qk_pbc_max_error=terr,
        active_reflection_original_mean=finite_mean(product_circulation(original, t)),
        active_reflection_recomputed_mean=finite_mean(product_circulation(mirror, tm)),
        reflection_note='x->(-x,y) at fixed lattice/checkpoint; no sign or symmetry asserted; synthetic passive transformations tested separately')


def capture_probe(target, info, batch, directory):
    from screening_analysis.attn_probe import probe, save_probe
    cache = directory/'probe.npz'
    fingerprint = hashlib.sha256(b''.join(np.ascontiguousarray(batch[k]).tobytes() for k in BATCH_KEYS)).hexdigest()
    expected = dict(checkpoint_sha256=info['checkpoint_sha256'], config_sha256=sha256(info['config_path']),
                    evaluation_input_sha256=fingerprint, collect_internal=True)
    if cache.exists():
        p = load_npz(cache)
        if any(p['metadata'].get(k)!=v for k, v in expected.items()):
            raise ValueError('cache provenance differs; choose a new output directory: '+str(cache))
        for path, expected in p['metadata']['source_sha256'].items():
            if sha256(path) != expected:
                raise ValueError('model source changed; use a new output directory')
        return p
    p = probe(target['training'], checkpoint=info['checkpoint'], config_path=info['config_path'],
              lattice=info['lattice'], evaluation_data=batch, nconf=len(batch['positions']), collect_internal=True)
    save_probe(p, cache)
    return p


def aligned_sensitivity(target, subset, p, net, params, output, count):
    from screening_analysis.ph_research import align_gradients
    from screening_analysis.rs_observables import phase_sensitivity
    if not count:
        return None, dict(status='未実施: phase_samples=0')
    study = Path(target['study'])
    # Common bank is shared with the prior validated edge derivatives. Native
    # prior derivatives are saved-training walkers, not these fresh MC inputs.
    existing = study/'common-pairs' if subset=='common' else None
    if existing is not None and (existing/'connection-gradients.npz').exists():
        inputs = load_npz(study/'common-measurements.npz')
        result = load_npz(existing/'connection-gradients.npz')
        try:
            pairs = align_gradients(p, inputs, result['gradients'], result['valid'])
        except ValueError:
            pairs = []
        if pairs:
            # Exact matching, not array-order assumptions. Keep all reusable pairs.
            pi, gi = np.asarray(pairs[:count], int).T
            checks = list(csv.DictReader((existing/'finite-difference.csv').open()))
            fd = finite_difference_checks(checks, result['gradients'])
            passed = fd['passed']
            if not passed:
                raise ValueError('saved common sensitivity finite-difference validation failed')
            return (pi, gi, result['gradients']), dict(status='既存実装・保存微分を再利用', matched_samples=len(pi),
                path=str(existing/'connection-gradients.npz'), sha256=sha256(existing/'connection-gradients.npz'),
                finite_difference_pass=passed, amplitude_phase_fd=fd)
    dest = output/'phase'; dest.mkdir(exist_ok=True)
    batch, ids = selected_batch(p, count)
    if (dest/'connection-gradients.npz').exists():
        result = load_npz(dest/'connection-gradients.npz'); inputs = load_npz(dest/'inputs.npz')
        status = json.loads((dest/'validation.json').read_text())
    else:
        status = phase_sensitivity(net, params, batch, dest, int(p['NL']), int(p['NH']))
        np.savez_compressed(dest/'inputs.npz', **batch)
        write_json(dest/'validation.json', status)
        result = load_npz(dest/'connection-gradients.npz'); inputs = batch
    if not status.get('finite_difference_pass', False):
        raise ValueError('new sensitivity finite-difference validation failed')
    checks = list(csv.DictReader((dest/'finite-difference.csv').open()))
    fd = finite_difference_checks(checks, result['gradients'])
    if not fd['passed']:
        raise ValueError('amplitude/phase finite-difference validation failed')
    status = dict(status, amplitude_phase_fd=fd)
    pairs = align_gradients(p, inputs, result['gradients'], result['valid'])
    if not pairs:
        return None, dict(status='データ不足: finite aligned derivatives absent')
    pi, gi = np.asarray(pairs, int).T
    return (pi, gi, result['gradients']), dict(status='既存phase_sensitivityを実行', matched_samples=len(pi), **status)


def wavefunction_diagnostics(net, params, p, config):
    """Physical coordinate derivatives versus internal head-removal responses.

    Coordinates recompute the entire antisymmetric wavefunction. Derivatives of
    phi are angle derivatives in the fixed implemented gauge, not Hall response.
    Head removal zeros the corresponding W_O rows (including self contribution).
    """
    import jax
    import jax.numpy as jnp
    from screening_analysis.connection_audit import angle_difference
    evaluate = jax.jit(net.apply)
    def observables(x, s, a, c):
        phase, amplitude = net.apply(params, x, s, a, c)
        return jnp.stack((amplitude, phase))
    derivative = jax.jit(jax.jacrev(observables, 0))
    rows = []; coordinate_gradients = []
    eps = config['coordinate_epsilon']
    for sample in range(min(config['coordinate_samples'], len(p['positions']))):
        args = [jnp.asarray(p[k][sample]) for k in BATCH_KEYS]
        phase, logabs = evaluate(params, *args)
        g = np.asarray(derivative(*args))
        if not np.isfinite([phase, logabs]).all() or not np.isfinite(g).all():
            rows.append(dict(kind='coordinate', sample=sample, status='nonfinite wavefunction/derivative'))
            continue
        direction = np.zeros_like(args[0]); direction[np.argmax(np.linalg.norm(g, axis=0))] = 1.
        errors = []
        for step in (eps, eps/2):
            pp, ap = evaluate(params, args[0]+step*direction, *args[1:])
            pm, am = evaluate(params, args[0]-step*direction, *args[1:])
            fd = np.array([(ap-am)/(2*step), angle_difference(pp, pm)/(2*step)], float)
            actual = g@direction; errors.append(float(np.max(abs(fd-actual)/(1+abs(actual)))))
        rows.append(dict(kind='coordinate', sample=sample, status='ok' if errors[-1]<1e-4 else 'unstable finite difference',
            phase_gradient_rms=float(np.sqrt(np.mean(g[1]**2))), amplitude_gradient_rms=float(np.sqrt(np.mean(g[0]**2))),
            fd_scaled_error=errors[-1], epsilon=eps/2))
        coordinate_gradients.append((sample, g))
    for layer, ap in enumerate(params['layers']['attention']):
        for head in range(int(p['NH'])):
            changed = jax.tree.map(lambda a: a, params)
            projection = ap['attn_output']; width = net.options.value_dim
            # W_O has rows [head, value]. Shared output bias is not a head.
            if isinstance(projection, dict):
                replacement = dict(projection)
                replacement['w'] = projection['w'].at[head*width:(head+1)*width].set(0.)
            else:
                replacement = projection.at[head*width:(head+1)*width].set(0.)
            changed['layers']['attention'][layer]['attn_output'] = replacement
            for sample in range(min(config['ablation_samples'], len(p['positions']))):
                args = [jnp.asarray(p[k][sample]) for k in BATCH_KEYS]
                phase, amp = evaluate(params, *args); phase0, amp0 = evaluate(changed, *args)
                valid = np.isfinite([phase, amp, phase0, amp0]).all()
                rows.append(dict(kind='head_ablation', sample=sample, layer=layer, head=head,
                    status='ok' if valid else 'nonfinite', dlogabs=float(amp0-amp) if valid else np.nan,
                    dphase=float(angle_difference(phase0, phase)) if valid else np.nan))
    return rows, coordinate_gradients


def run_target(target, config, root):
    import jax
    from screening_analysis.checkpoint_model import load_model
    from screening_analysis.ph_research import configuration_physics
    from screening_analysis.physical_phase_metrics import analyze_probe, finite_mean, block_summary, corr, sources
    from screening_analysis.connection_audit import symmetric_edges
    case = target['case']; study = Path(target['study']); out = root/'cache'/case['id']
    out.mkdir(parents=True, exist_ok=True)
    conditions = json.loads((study/'conditions.json').read_text())
    checkpoint = conditions['checkpoint']
    all_rows = {}; metrics = []; manifest = []; validations = []
    with load_model(target['training'], checkpoint=checkpoint) as (net, params, _, info):
        cfg = info['config']; actual = cfg['system']['make_local_energy_kwargs']
        if (info['step']!=79999 or sum(cfg['system']['electrons'])!=case['n'] or
            net.options.num_heads!=target['heads'] or net.options.num_layers!=1 or cfg['seed']!=case['seed'] or
            not np.isclose(actual['lam'], case['lam']) or not np.isclose(actual['potential_kwargs']['interaction_energy_scale'], case['rs'])):
            raise ValueError('checkpoint does not match declared 80000-update experiment')
        if info['checkpoint_sha256'] != conditions['checkpoint_sha256']:
            raise ValueError('prior measurement/checkpoint hash mismatch')
        native = load_npz(study/'equilibrium-measurements.npz')
        common = load_npz(Path(json.loads(resolve(config['experiment']).read_text())['baseline_plan']).parent/'common-inputs.npz') if 'experiment' in config else load_npz(resolve(target['common_inputs']))
        if 'lattice' in common and not np.allclose(common['lattice'], info['lattice'], atol=1e-12, rtol=0):
            raise ValueError('common lattice mismatch')
        summary = json.loads((study/'summary.json').read_text())
        for subset, bank in [('native', native), ('common', common)]:
            dest = out/subset; dest.mkdir(exist_ok=True)
            batch, bank_ids = selected_batch(bank, config['samples'])
            # Preserve existing phase-input identities by using first common samples.
            if subset=='common':
                bank_ids = np.arange(min(config['samples'], len(bank['positions'])))
                batch = {k: bank[k][bank_ids] for k in BATCH_KEYS}
            p = capture_probe(target, info, batch, dest)
            definition_checks = validate_probe(p)
            transform_checks = transformation_diagnostics(net, params, p, cfg['system']['electrons'])
            gradient, sensitivity_status = aligned_sensitivity(target, subset, p, net, params, dest, config['phase_samples'])
            tables, arrays = analyze_probe(p, dict(config, rs=case['rs']), gradient)
            # Physical S(q), g(r) reuse the established implementation on ALL native
            # saved MC measurements, with ordered time/chain provenance retained.
            physics_probe = dict(p, **{k: bank[k] for k in BATCH_KEYS})
            physical_rows, physical_arrays = configuration_physics(physics_probe, np.arange(len(bank['positions'])), 0)
            blocks = summary['sampling']['blocks'] if subset=='native' else None
            sf = np.array([r['structure_factor_shell'] for r in physical_rows]); sf_stats = block_summary(sf, blocks)
            tables['configuration'] = []
            for sample, row in enumerate(physical_rows):
                row.update(configuration_id=input_id(bank, sample), role='native_MCMC' if subset=='native' else 'common_synthetic',
                           local_energy_real=float(native['local_energy'][sample].real) if subset=='native' else None,
                           local_energy_imag=float(native['local_energy'][sample].imag) if subset=='native' else None)
                tables['configuration'].append(row)
            for table in tables:
                for row in tables[table]:
                    if 'sample' in row and table!='configuration':
                        row['configuration_id'] = input_id(p, row['sample'])
            wf_rows, coordinate_gradients = wavefunction_diagnostics(net, params, p, config)
            # Coordinate gradients describe particle response, not a unique pair
            # perturbation. Compare to incident edge strength and label accordingly.
            for sample, g in coordinate_gradients:
                arrays[f'coordinate_gradients_b{sample}'] = g
                for name, weights in sources(p).items():
                    for head in range(int(p['NH'])):
                        w = weights[sample, 0, head].copy(); np.fill_diagonal(w, 0.)
                        strength = (w.sum(0)+w.sum(1))/2
                        for ci, channel in [(0, 'amplitude'), (1, 'phase')]:
                            magnitude = np.linalg.norm(g[ci].reshape(-1, 2), axis=1)
                            wf_rows.append(dict(kind='coordinate_strength_correlation', sample=sample,
                                source=name, head=head, channel=channel, pearson=corr(strength, magnitude), status='fixed gauge; particle response'))
            for row in wf_rows:
                row['configuration_id'] = input_id(p, row['sample'])
            tables['wavefunction'] = wf_rows
            np.savez_compressed(dest/'analysis-arrays.npz', **arrays, **{f'physics_{k}': v for k, v in physical_arrays.items()})
            for name, rows in tables.items():
                tagged = [dict(model_id=case['id'], subset=subset, rs=case['rs'], heads=target['heads'],
                               seed=case['seed'], **r) for r in rows]
                write_csv(dest/(name+'.csv'), tagged)
                all_rows.setdefault(name, []).extend(tagged)
            head = [r for r in tables['head'] if r['source']=='screening']
            phase = [r for r in tables['phase'] if r['source']=='screening' and r['channel']=='phase']
            eigen = [r for r in tables['eigen'] if r['matrix']=='normalized']
            circulation = [r for r in tables['circulation'] if r['source']=='qk_signed' and r['spin']=='all']
            head_summary = arrays['head_summary_l0']
            mc_energy = block_summary(native['local_energy'].real, summary['sampling']['blocks'])
            metrics.append(dict(model_id=case['id'], subset=subset, rs=case['rs'], heads=target['heads'], seed=case['seed'],
                initialization=case['initialization'], parameters=sum(np.asarray(v).size for v in jax.tree.leaves(params)),
                samples=len(p['positions']), physical_samples=len(bank['positions']), structure_factor=sf_stats['mean'],
                structure_factor_configuration_sd=sf_stats['configuration_sd'], structure_factor_block_sem=sf_stats['block_sem'],
                energy_code=mc_energy['mean'] if subset=='native' else None,
                energy_block_sem=mc_energy['block_sem'] if subset=='native' else None,
                energy_per_electron_Ha=mc_energy['mean']/(case['n']*case['rs']**2) if subset=='native' else None,
                r90=finite_mean([r['r90'] for r in head]), residual_variance=finite_mean([r['residual_variance'] for r in head]),
                residual_coverage=finite_mean([r['residual_coverage'] for r in head]),
                residual_peak=finite_mean([r['residual_peak'] for r in head]), crystal_peak=finite_mean([r['crystal_peak'] for r in head]),
                crystal_increment=finite_mean([r['crystal_increment'] for r in head]),
                edge_effective_heads=head_summary[0], output_effective_heads=head_summary[1], cancellation_ratio=head_summary[2],
                phase_pearson=finite_mean([r['pearson'] for r in phase]),
                spin_rayleigh=finite_mean([r['spin_rayleigh'] for r in eigen]),
                crystal_rayleigh=finite_mean([r['crystal_rayleigh'] for r in eigen]),
                circulation_rms=finite_mean([r['rms'] for r in circulation]),
                phase_status='unknown', mc_convergence='unverified', error_definition='contiguous time-block SEM; no independent-run uncertainty'))
            invalid_coordinates = sum(r.get('status') not in ('ok',) for r in wf_rows if r['kind']=='coordinate')
            status = dict(model_id=case['id'], subset=subset, samples=len(p['positions']), sensitivity=sensitivity_status,
                definitions=definition_checks, transformations=transform_checks,
                rayleigh_ritz_pass=all(r['rayleigh_ritz_pass'] for r in tables['eigen']),
                coordinate_unstable_samples=invalid_coordinates,
                head_ablation='fixed input amplitude/phase only; energy and reweighted S(q) not evaluated',
                residual_crossfit='alternating whole configurations; MC folds may remain correlated',
                configuration_uncertainty='configuration SD; MC block SEM only for full ordered native S(q)/energy; seed variance unavailable')
            validations.append(status)
            manifest.append(dict(model_id=case['id'], subset=subset, probe_metadata=p['metadata'],
                selected_bank_indices=bank_ids.tolist(), bank=str(study/'equilibrium-measurements.npz') if subset=='native' else 'common manifest bank',
                bank_input_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(bank[k]).tobytes() for k in BATCH_KEYS)).hexdigest(),
                config_path=info['config_path'], hf_path=cfg['network']['CustomScreenformer'].get('lj_hf_path'),
                hf_sha256=sha256(cfg['network']['CustomScreenformer']['lj_hf_path']) if cfg['network']['CustomScreenformer'].get('lj_hf_path') else None,
                sampling=summary['sampling'] if subset=='native' else dict(origin='synthetic uniform positions; alternating polarized/random spins; not equilibrium'),
                study_conditions_sha256=sha256(study/'conditions.json')))
            print(case['id'], subset, 'complete', flush=True)
    return all_rows, metrics, manifest, validations


def compare(rows):
    result = []
    for subset in ('native', 'common'):
        selected = [r for r in rows if r['subset']==subset]
        for h in sorted({r['heads'] for r in selected}):
            a = next((r for r in selected if r['heads']==h and r['rs']==4), None)
            b = next((r for r in selected if r['heads']==h and r['rs']==25), None)
            if a is None or b is None:
                continue
            for key in ('structure_factor', 'r90', 'residual_variance', 'residual_peak', 'crystal_increment',
                        'edge_effective_heads', 'output_effective_heads', 'cancellation_ratio', 'phase_pearson',
                        'spin_rayleigh', 'crystal_rayleigh', 'circulation_rms'):
                result.append(dict(subset=subset, heads=h, metric=key, rs4=a[key], rs25=b[key],
                    difference=b[key]-a[key], inference='descriptive single-run contrast; no significance test'))
    return result


def analyze(config, resume=False):
    root = Path(config['output']); root.mkdir(parents=True, exist_ok=resume)
    for name in ('data', 'diagnostics', 'cache', 'logs', 'figures'):
        (root/name).mkdir(exist_ok=True)
    recorded = root/'diagnostics/analysis_config.json'
    if recorded.exists() and json.loads(recorded.read_text()) != config:
        raise ValueError('resume configuration changed; choose a new output')
    write_json(recorded, config)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--short'], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    tables = {}; metrics = []; manifest = []; validations = []; failures = []
    targets = discover(config)
    for target in targets:
        try:
            rows, model_metrics, inputs, checks = run_target(target, config, root)
            for key, value in rows.items():
                tables.setdefault(key, []).extend(value)
            metrics.extend(model_metrics); manifest.extend(inputs); validations.extend(checks)
        except Exception as error:
            identifier = target['case']['id']
            failures.append(dict(model_id=identifier, error=str(error), status='未実施: analysis failed; see logs'))
            (root/'logs'/(identifier+'.txt')).write_text(traceback.format_exc())
            print(identifier, 'FAILED:', str(error), flush=True)
        # Persist completed cases even if later cases fail.
        for key, value in tables.items():
            filename = {'head':'head_metrics', 'configuration':'configuration_metrics'}.get(key, key)
            write_csv(root/'data'/(filename+'.csv'), value)
        write_csv(root/'data/metrics.csv', metrics)
        write_csv(root/'data/phase_comparison.csv', compare(metrics))
        write_json(root/'diagnostics/input_manifest.json', dict(runs=manifest, revision=revision, dirty_status=dirty,
            source_sha256=source_hashes(), seed=config['seed']))
        write_json(root/'diagnostics/validation.json', dict(expected_models=len(targets), completed_subsets=len(metrics),
            checks=validations, failures=failures, phase_verified=False,
            unavailable=['independent training seed variation', 'MC mixing/ESS convergence',
                         'head-ablation energy and S(q)', 'trained twist-family Hall/Chern evidence']))
        import jax
        jax.clear_caches()
    return not failures and len(metrics)==2*len(targets)


def finite_difference_checks(rows, gradients):
    """Validate both amplitude and wrapped phase, including saved common data."""
    selected = [r for r in rows if float(r['epsilon'])==5e-4]
    amplitude = []; phase = []; identity = []
    for row in selected:
        index = tuple(int(row[k]) for k in ('sample', 'layer', 'head', 'edge'))
        g = gradients[index[0], 0, index[1], index[2], index[3]]
        amplitude.append(float(row['amplitude_error'])/(1+abs(g)))
        phase.append(float(row['phase_scaled_error']))
        identity.append(float(row['identity_error']))
    values = np.r_[amplitude, phase, identity]
    passed = bool(selected) and np.isfinite(values).all() and max(amplitude)<1e-4 and max(phase)<1e-4 and max(identity)<1e-9
    return dict(passed=bool(passed), checked_edges=len(selected),
        amplitude_max_scaled_error=max(amplitude, default=np.nan), phase_max_scaled_error=max(phase, default=np.nan),
        identity_max_error=max(identity, default=np.nan))


def verify_outputs(config):
    """Offline provenance and amplitude/phase checks, with no JAX dependency."""
    root = Path(config['output']); manifest = json.loads((root/'diagnostics/input_manifest.json').read_text())
    status = json.loads((root/'diagnostics/validation.json').read_text()); checks = []
    for record in manifest['runs']:
        metadata = record['probe_metadata']; checksums = {}
        for key in ('checkpoint', 'config_path'):
            path = metadata[key]; expected = metadata['checkpoint_sha256' if key=='checkpoint' else 'config_sha256']
            actual = sha256(path); checksums[key] = dict(path=path, sha256=actual, unchanged=actual==expected)
        for path, expected in metadata['source_sha256'].items():
            checksums[path] = dict(unchanged=sha256(path)==expected)
        if record.get('hf_path'):
            checksums['hf'] = dict(path=record['hf_path'], sha256=sha256(record['hf_path']),
                                   unchanged=sha256(record['hf_path'])==record['hf_sha256'])
        subset = record['subset']; model = record['model_id']
        check = next(c for c in status['checks'] if c['model_id']==model and c['subset']==subset)
        sensitivity = check['sensitivity']
        gradient_path = Path(sensitivity.get('path', root/'cache'/model/subset/'phase/connection-gradients.npz'))
        fd = dict(passed=None, status='未実施: phase_samples=0')
        if config['phase_samples']:
            rows = list(csv.DictReader((gradient_path.parent/'finite-difference.csv').open()))
            gradient = load_npz(gradient_path)
            fd = dict(**finite_difference_checks(rows, gradient['gradients']), path=str(gradient_path),
                sha256=sha256(gradient_path), valid_samples=int(gradient['valid'].sum()))
        bank = Path(record['bank'])
        if subset=='common':
            if 'experiment' in config:
                experiment = json.loads(resolve(config['experiment']).read_text())
                baseline = json.loads(Path(experiment['baseline_plan']).read_text())
                bank = Path(baseline['common_inputs'])
            else:
                target = next(t for t in config['runs'] if t['case']['id']==model)
                bank = resolve(target['common_inputs'])
        checks.append(dict(model_id=model, subset=subset, checksums=checksums,
            bank_path=str(bank), bank_file_sha256=sha256(bank), sensitivity=fd,
            passed=all(c['unchanged'] for c in checksums.values()) and fd['passed'] is not False
                and check['rayleigh_ritz_pass'] and not check['coordinate_unstable_samples']))
    complete = not status['failures'] and len(checks)==2*status['expected_models']
    passed = complete and all(c['passed'] for c in checks)
    write_json(root/'diagnostics/verification.json', dict(passed=passed, complete=complete, checks=checks,
        verification_source_sha256=sha256(__file__), source_sha256=source_hashes(),
        note='Post-analysis checks of unchanged checkpoint/config/HF/model source; both amplitude and phase FD; bank file hashes recorded at verification time.'))
    print('Offline verification:', passed, flush=True)
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--mode', choices=('all', 'analyze', 'report', 'verify'), default='all')
    p.add_argument('--resume', action='store_true', help='reuse matching cached probes/derivatives')
    args = p.parse_args(); config = read_config(args.config)
    success = True
    if args.mode in ('all', 'analyze'):
        success = analyze(config, args.resume)
    if args.mode in ('all', 'verify'):
        success = verify_outputs(config) and success
    if args.mode in ('all', 'report'):
        subprocess.run([config['plot_python'], '-m', 'screening_analysis.physical_phase_report', config['output']],
                       cwd=ROOT, check=True)
    if not success:
        raise SystemExit(1)


if __name__=='__main__':
    main()
