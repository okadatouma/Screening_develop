"""Post-training measurements for vmc.job; write only into a new output directory.

Uses repository checkpoint reconstruction and pseudospin Metropolis sampling.
Training energies are diagnostics, not an independent energy measurement.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def structure_measurements(positions, lattice, bins=48):
    """Canonical finite-N g(r), uniform=1; S(k) includes disconnected density."""
    from itertools import product
    from screening_analysis.graph_wind import minimum_image_edges
    from screening_analysis.lj_maps import first_shell, sk, structure_factor

    positions = np.asarray(positions)
    samples, n, _ = positions.shape
    area = abs(np.linalg.det(lattice))
    frac = np.mod(positions @ np.linalg.inv(lattice).T, 1)
    density, x, y = np.histogram2d(
        *frac.reshape(-1, 2).T, bins=bins, range=((0, 1), (0, 1)))
    density /= samples * area / bins**2
    translations = np.array([v for v in product(range(-3, 4), repeat=2)
                             if v != (0, 0)]) @ lattice.T
    rmax = .45 * np.linalg.norm(translations, axis=1).min()
    edges = np.linspace(0, rmax, bins + 1)
    hist = np.zeros(bins)
    for r in positions:
        hist += np.histogram(minimum_image_edges(r, lattice)[1], bins=edges)[0]
    norm = samples * n * (n - 1) / (2 * area) * np.pi * np.diff(edges**2)
    gr = hist / norm
    sf = sk(positions, lattice, nmax=12)
    modes = np.array(list(sf))
    values = np.array(list(sf.values()))
    vectors = modes @ (2 * np.pi * np.linalg.inv(lattice))
    shell = first_shell(lattice, n)
    return dict(density=density, density_fractional_x_edges=x,
                density_fractional_y_edges=y, r_edges=edges, gr=gr,
                integer_modes=modes, k_vectors=vectors, structure_factor=values,
                first_shell=np.asarray(shell),
                first_shell_structure_factor=structure_factor(positions, lattice, shell))


def energy_summary(folder, output, block_size=500):
    with (folder / 'train_stats.csv').open() as f:
        rows = list(csv.DictReader(f))
    energy = np.array([float(r['energy']) for r in rows])
    if not len(energy) or not np.isfinite(energy).all():
        raise ValueError('Training energies must be nonempty and finite')
    tail = energy[-20000:]
    count = len(tail) // block_size
    blocks = (tail[-count * block_size:].reshape(count, block_size).mean(1)
              if count else np.array([]))
    sem = float(blocks.std(ddof=1) / np.sqrt(count)) if count > 1 else None
    result = dict(updates=len(energy), tail_updates=len(tail),
                  tail_mean=float(tail.mean()), block_size=block_size,
                  blocks=count, block_sem=sem,
                  note='Training-batch diagnostic; autocorrelation/block-size convergence unverified; not independent energy measurement')
    (output / 'energy-summary.json').write_text(json.dumps(result, indent=2))
    # Standalone trace, no plotting dependency; full numerical data stay in CSV.
    ids = np.unique(np.linspace(0, len(energy)-1, min(2000, len(energy))).astype(int))
    low, high = energy.min(), energy.max()
    span = max(float(high-low), 1e-12)
    points = ' '.join(f'{40+920*i/max(len(energy)-1,1):.2f},{260-220*(energy[i]-low)/span:.2f}' for i in ids)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="300">'
           '<rect width="100%" height="100%" fill="white"/>'
           '<text x="40" y="20">Training energy (code units); horizontal axis: update</text>'
           f'<text x="40" y="285">range {low:.6g} to {high:.6g}; updates {len(energy)}</text>'
           f'<polyline points="{points}" fill="none" stroke="#286c9c"/></svg>')
    (output / 'energy-trace.svg').write_text(svg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-hf', choices=('wc', 'ahc'), required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--sweeps', type=int, default=512)
    a = p.parse_args()
    if a.chains < 2 or a.sweeps < 2:
        p.error('At least two chains and sweeps are required')
    folder = a.folder.resolve()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    energy_summary(folder, output)

    from screening_analysis.checkpoint_model import load_model
    from screening_analysis.physical_loops import make_network_evaluator, sample_wavefunction
    with load_model(folder) as (net, params, data, info):
        lattice = np.asarray(info['lattice'])
        n = sum(info['config']['system']['electrons'])
        evaluate = make_network_evaluator(net, params, data['atoms'][0], data['charges'][0])
        sampled = sample_wavefunction(
            evaluate, lattice, n, seed=a.seed + 1000, chains=a.chains,
            sweeps=a.sweeps, burn=256, steps_per_sample=4, width=.02,
            pseudospin=True)
        np.savez_compressed(output / 'fresh-samples.npz', **sampled)
        # Keep chain/time axes in samples; pooled means below do not assume independence.
        positions = sampled['positions'].reshape(-1, n, 2)
        measured = structure_measurements(positions, lattice)
        np.savez_compressed(output / 'density-correlations.npz', lattice=lattice, **measured)
        np.savetxt(output / 'gr.csv',
                   np.column_stack(((measured['r_edges'][1:] + measured['r_edges'][:-1])/2,
                                    measured['gr'])), delimiter=',', header='r,g_canonical', comments='')
        np.savetxt(output / 'structure-factor.csv',
                   np.column_stack((measured['integer_modes'], measured['k_vectors'],
                                    measured['structure_factor'])), delimiter=',',
                   header='m1,m2,kx,ky,S_total', comments='')
        metadata = dict(checkpoint=info['checkpoint'], checkpoint_sha256=info['checkpoint_sha256'],
                        initial_hf_kind=a.initial_hf, phase='unknown', phase_verified=False,
                        samples=len(positions), chains=a.chains, sweeps=a.sweeps,
                        acceptance=sampled['acceptance'].tolist(),
                        spin_acceptance=sampled['spin_acceptance'].tolist(),
                        first_shell_structure_factor=measured['first_shell_structure_factor'],
                        density_definition='particle number per physical code-unit area, integrates to N',
                        gr_definition='canonical finite-N radial pair correlation; uniform=1; self pairs excluded',
                        structure_factor_definition='S(k)=mean(abs(sum_j exp(i k.r_j))**2)/N; k!=0; disconnected part included',
                        limitations='Finite fresh MCMC; mixing, effective sample size and errors unverified; no phase certification')
        (output / 'measurement-summary.json').write_text(json.dumps(metadata, indent=2))

    # Pin the checkpoint used above, and run the existing PH/winding entry on CPU.
    manifest = dict(seed=a.seed, null_repeats=1, runs=[dict(
        id='final', folder=str(folder), checkpoint=info['checkpoint'],
        config_path=str(folder / 'runtime_config.json'), nconf=8,
        initial_hf_kind=a.initial_hf, phase='unknown', phase_verified=False)])
    path = output / 'topology-manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    env = dict(os.environ, JAX_PLATFORMS='cpu')
    subprocess.run([sys.executable, '-B', '-u', '-m', 'screening_analysis.topology_run',
                    str(path), '--output', str(output / 'topology')], env=env, check=True)
    (output / 'finished.json').write_text(json.dumps(dict(analysis_complete=True)))


if __name__ == '__main__':
    main()
