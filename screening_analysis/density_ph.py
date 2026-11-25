"""Matched-edge-count flag PH, using saved probes without model evaluation.

python -m screening_analysis.density_ph ROOT --output NEW_DIR --null-repeats 8
Only positive edges are admitted. Ties use a reproducible shared random priority;
tie-splitting boundaries are flagged and excluded from primary summaries.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from screening_analysis.topology import persistence as persistence_fn
from screening_analysis.topology import symmetrize
from screening_analysis.topology_run import graph_sources


def bounded_weights(weights, tolerance=1e-12):
    """Clip only roundoff excursions, never silently rescale invalid weights."""
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or w.min() < -tolerance or w.max() > 1+tolerance:
        raise ValueError('weights outside [0,1] beyond roundoff tolerance')
    return np.clip(w, 0, 1)


def density_curve(weights, max_k=None, priorities=None):
    """Exact k-edge filtration; returns Betti curves, tie flags, and PH diagrams.

    Ordinal birth times are edge counts. Triangles enter at their last edge.
    Essential intervals refer to the graph truncated at max_k, not full support.
    """
    b = symmetrize(weights)
    ij = np.triu_indices(len(b), 1)
    values = b[ij]
    positive = np.flatnonzero(values > 0)
    support = len(positive)
    max_k = support if max_k is None else int(max_k)
    if not 0 <= max_k <= support:
        raise ValueError('max_k exceeds positive support')
    if priorities is None:
        priorities = np.arange(len(values))
    order = positive[np.lexsort((np.asarray(priorities)[positive], -values[positive]))]
    m = len(values)
    ranked = np.zeros_like(b)
    edges = order[:max_k]
    ranked[ij[0][edges], ij[1][edges]] = 1 - np.arange(1, max_k + 1) / (m + 1)
    ranked += ranked.T
    diagrams = persistence_fn(ranked)
    k = np.arange(max_k + 1)
    # Query between consecutive ordinal edge events, never at a rounded boundary.
    times = (k + .5) / (m + 1)
    curves = np.stack([((d[:, 0, None] < times) & (times < d[:, 1, None])).sum(axis=0)
                       for d in diagrams])
    split = np.zeros(max_k + 1, dtype=bool)
    for count in range(1, max_k + 1):
        split[count] = count < support and values[order[count-1]] == values[order[count]]
    return curves, split, tuple(d * (m + 1) for d in diagrams)


def distance_shuffle(weights, distances, rng, bins=10):
    """Permute all edge values (including zeros) within distance quantile bins.

    Preserves exact edge-weight multiset and positive count in every bin;
    preserves binned edge-length distribution, not individual distances/degrees.
    """
    b = symmetrize(weights)
    ij = np.triu_indices(len(b), 1)
    d = symmetrize_distance(distances)[ij]
    cuts = np.unique(np.quantile(d, np.linspace(0, 1, bins + 1)))
    labels = np.searchsorted(cuts[1:-1], d, side='right')
    values = b[ij].copy()
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        values[ids] = rng.permutation(values[ids])
    b[ij] = values
    b[ij[1], ij[0]] = values
    return b, cuts


def symmetrize_distance(distances):
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('finite nonnegative square distances required')
    return (d + d.T) / 2


def analyze(probe_path, out, repeats=8, seed=2475):
    with np.load(probe_path, allow_pickle=False) as z:
        probe = {k: z[k] for k in z.files if k != 'metadata_json'}
        metadata = json.loads(str(z['metadata_json']))
    sources = graph_sources(probe)
    bcount, layers, heads, n, _ = probe['S'].shape
    m = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    rows, records, diagrams, overlaps = [], [], {}, []
    roundoff = {s: int(np.count_nonzero((w < 0) | (w > 1))) for s,w in sources.items()}
    sources = {s: bounded_weights(w) for s,w in sources.items()}
    for sample in range(bcount):
        dist = symmetrize_distance(probe['d'][sample])
        for layer in range(layers):
            for head in range(heads):
                matrices = {s: symmetrize(w[sample, layer, head]) for s, w in sources.items()}
                matrices['distance'] = np.exp(-dist / float(probe['dref']))
                supports = {s: int(np.count_nonzero(w[np.triu_indices(n, 1)] > 0)) for s, w in matrices.items()}
                max_k = min(supports.values())
                priority = rng.permutation(m)
                ij = np.triu_indices(n, 1)
                orders = {s: np.lexsort((priority, -w[ij])) for s,w in matrices.items()}
                for k in (30, 60, 90):
                    if k > max_k:
                        continue
                    for a,b in (('qk','trim'),('window','screening'),('window','distance')):
                        left,right = set(orders[a][:k]),set(orders[b][:k])
                        overlaps.append(dict(sample=sample,layer=layer,head=head,k=k,
                                             left=a,right=b,jaccard=len(left & right)/len(left | right)))
                results = {}
                for source, matrix in matrices.items():
                    betti, split, ds = density_curve(matrix, max_k, priority)
                    results[source] = betti, split
                    for dim, diagram in enumerate(ds):
                        diagrams[f'b{sample}_l{layer}_h{head}_{source}_H{dim}'] = diagram
                valid = ~np.logical_or.reduce([v[1] for v in results.values()])
                _, distance_cuts = distance_shuffle(matrices['distance'], dist, np.random.default_rng(0))
                records.append(dict(sample=sample, layer=layer, head=head, max_k=max_k,
                                    distance_bin_edges=distance_cuts.tolist(),
                                    supports=supports, untied_k_count=int(valid.sum())))
                for source, matrix in matrices.items():
                    for control in ('raw', 'distance_shuffle'):
                        for repeat in range(1 if control == 'raw' else repeats):
                            if control == 'raw':
                                betti, split = results[source]
                            else:
                                shuffled, cuts = distance_shuffle(matrix, dist, rng)
                                betti, split, _ = density_curve(shuffled, max_k, priority)
                            for k in range(max_k + 1):
                                rows.append(dict(sample=sample, layer=layer, head=head, source=source,
                                                 control=control, repeat=repeat, k=k, p=k/m,
                                                 beta0=int(betti[0, k]), beta1=int(betti[1, k]),
                                                 tie_split=bool(split[k]), common_untied=bool(valid[k] and not split[k])))
    out.mkdir(parents=True, exist_ok=False)
    with (out/'curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if overlaps:
        with (out/'edge-overlap.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(overlaps[0]));writer.writeheader();writer.writerows(overlaps)
    np.savez_compressed(out/'diagrams-edge-count.npz', **diagrams)
    # One paired statistic per configuration/head over its shared admissible k.
    paired = []
    for record in records:
        selected = [r for r in rows if (r['sample'],r['layer'],r['head']) ==
                    (record['sample'],record['layer'],record['head']) and r['control']=='raw' and r['common_untied']]
        means = {s: float(np.mean([r['beta1'] for r in selected if r['source']==s])) if selected else None
                 for s in matrices}
        paired.append({**record, 'mean_beta1_common_untied': means})
    aggregates = {s: float(np.mean([r['mean_beta1_common_untied'][s] for r in paired
                                    if r['mean_beta1_common_untied'][s] is not None])) for s in matrices}
    # Aggregate only where all configuration/head graphs have support and no ties.
    common_max = min(r['max_k'] for r in records)
    primary = {s: float(np.mean([r['beta1'] for r in rows if r['source']==s and
                               r['control']=='raw' and r['k']<=common_max and r['common_untied']]))
               for s in matrices}
    summary = dict(probe=str(probe_path.resolve()), probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
                   metadata=metadata, seed=seed, null_repeats=repeats, n=n, possible_edges=m,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   roundoff_clipped_entries=roundoff, roundoff_tolerance=1e-12,
                   common_max_k=common_max, common_max_p=common_max/m, groups=paired,
                   primary_mean_beta1_global_common_k=primary,
                   mean_beta1_per_graph_shared_untied_k=aggregates,
                   tie_policy='Exact top-k with shared seeded random priorities; boundaries splitting exact ties excluded from primary summaries and plotted means.',
                   null='10 distance-quantile bins, permutation of all edge values including zeros within each bin; exact binned support/weight multisets preserved. Not degree-preserving.',
                   statistical_unit='Independent training run. Configurations/heads and null repeats are descriptive, not independent training replicates.',
                   diagrams='Birth/death in edge-count units, positive edges only, truncated at each graph common max_k; infinite deaths are right-censored by truncation.',
                   geometry='Distance baseline exp(-probe d/dref), using the existing periodic-norm probe distance, not a new Euclidean/minimum-image metric.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    plot(rows, common_max, m, out/'matched-density.svg')
    return summary


def plot(rows, max_k, m, path):
    colors = dict(qk='#8e44ad', trim='#d47c11', window='#2675b6', screening='#18845a', distance='#555555')
    grouped = {}
    for r in rows:
        if r['k'] <= max_k and r['common_untied']:
            grouped.setdefault((r['source'], r['control'], r['k']), []).append(r['beta1'])
    values = {key: float(np.mean(v)) for key,v in grouped.items()}
    ymax = max(values.values(), default=1) * 1.1 or 1
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600" viewBox="0 0 1000 600">',
           '<rect width="1000" height="600" fill="white"/><g font-family="sans-serif" font-size="14">',
           '<text x="70" y="28">Matched edge density: mean beta1 (untied boundaries only)</text>']
    for j in range(6):
        y = 510-440*j/5
        svg.append(f'<path d="M70 {y}H790" stroke="#ddd"/><text x="12" y="{y+5}">{ymax*j/5:.1f}</text>')
        x = 70+720*j/5
        svg.append(f'<text x="{x-14}" y="535">{max_k/m*j/5:.3f}</text>')
    for index, (source,color) in enumerate(colors.items()):
        for control in ('raw', 'distance_shuffle'):
            pts = [(k,v) for (s,c,k),v in values.items() if s==source and c==control]
            pts.sort()
            points = ' '.join(f'{70+720*k/max(1,max_k):.2f},{510-440*v/ymax:.2f}' for k,v in pts)
            dash = ' stroke-dasharray="5 4"' if control!='raw' else ''
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        svg.append(f'<text x="815" y="{90+30*index}" fill="{color}">{source}</text>')
    svg.extend(['<text x="815" y="280">solid: original</text><text x="815" y="305">dashed: distance null</text>',
                '<text x="350" y="565">p = k / possible edges</text></g></svg>'])
    path.write_text('\n'.join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--null-repeats', type=int, default=8)
    args = parser.parse_args()
    if args.null_repeats < 1:
        parser.error('positive null-repeats required')
    probes = sorted(args.root.glob('*/analysis/topology/final/probe.npz'))
    if not probes:
        parser.error('no final probes found')
    args.output.mkdir(parents=True, exist_ok=False)
    for path in probes:
        name = path.parents[3].name
        summary = analyze(path, args.output/name, args.null_repeats)
        print(name, json.dumps(summary['mean_beta1_per_graph_shared_untied_k']), flush=True)


if __name__ == '__main__':
    main()
