"""合成HF表でA〜Dの短期学習・RNG付き再開・core診断を接続検証。物理的な相判定ではない。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('output', type=Path)
    args = p.parse_args(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    from periodicwave.window_experiments.run import parser, configuration
    c = configuration(parser().parse_args(['--smoke', '--out', str(root), '--lam', '.64']))
    reciprocal = 2*np.pi*np.linalg.inv(np.asarray(c['lattice']))
    k = np.array([[0, 0], reciprocal[0], reciprocal[1], reciprocal[0]+reciprocal[1]])
    hf = root/'synthetic_hf.txt'; np.savetxt(hf, [[*q, 0, 0, 1, 0] for q in k], fmt='%.17g')
    meta = {key: c[key] for key in ('lam', 'rs', 'delta', 'n', 'lattice')}
    meta.update(label='WC', sha256=hashlib.sha256(hf.read_bytes()).hexdigest(), purpose='synthetic wiring test; not a WC solution')
    meta_path = root/'synthetic_hf.json'; meta_path.write_text(json.dumps(meta, indent=2))
    common = ['--smoke', '--lam', '.64', '--hf', str(hf), '--hf-meta', str(meta_path), '--label', 'WC']
    results = []
    def execute(name, module, options):
        command = [sys.executable, '-m', module] + options
        env = dict(os.environ, JAX_PLATFORMS='cpu')
        with (root/(name+'.txt')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        results.append(dict(name=name, command=command, exit_code=result.returncode))
        if result.returncode: raise RuntimeError(f'{name} failed; see {root/name}.txt')
    for precision in ('float32', 'float64'):
        for preset in 'ABCD':
            name = preset+'-'+precision
            execute(name, 'periodicwave.window_experiments.run', common+['--out', str(root/name),
                '--preset', preset, '--dtype', precision, '--laplacian', 'folx'])
    base = root/'A-float64'
    execute('resume', 'periodicwave.window_experiments.run', common+['--out', str(base), '--resume', '--iterations', '4', '--dtype', 'float64'])
    full = root/'A-float64-uninterrupted'
    execute('uninterrupted', 'periodicwave.window_experiments.run', common+['--out', str(full), '--iterations', '4', '--dtype', 'float64'])
    import jax
    from periodicwave.window_experiments.run import restore_state
    jax.config.update('jax_enable_x64', True)
    a = restore_state(base, 4); b = restore_state(full, 4)
    if jax.tree.structure(a) != jax.tree.structure(b): raise AssertionError('checkpoint tree differs')
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)): np.testing.assert_array_equal(x, y)
    execute('analysis', 'periodicwave.window_experiments.analyze', [str(base), '--burn', '1', '--measure', '8',
        '--walkers', '2', '--core-centers', '[[0,0]]', '--core-radius', '1',
        '--radii', '.8,1.2', '--cut-burn', '2', '--cut-measure', '8'])
    report = json.loads((base/'window_analysis_v1.json').read_text())
    assert report['cutoff_scan']['Rstar'] is None  # 短すぎる系列から物理長を捏造しない。
    with np.load(base/'window_analysis_v1.npz') as z:
        assert 'layer0_core_delta_C' in z and 'layer0_core_alpha_mm' in z
        np.testing.assert_allclose(z['layer0_q_head_mean_squared_norm'].sum(), 1)
    (root/'summary.json').write_text(json.dumps(dict(results=results, resumed_tree_and_rng_exact=True,
        note='Synthetic N4 HF, no Coulomb, 3/4 Adam updates. Not university N36 or phase validation.'), indent=2))
    print(root/'summary.json')


if __name__ == '__main__': main()
