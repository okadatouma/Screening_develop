"""Non-PH statistics on the official probe arrays (no model forward here).

Lengths are code lengths (= rs*aB). Lattice vectors are COLUMNS. Fourier
weights exclude self edges and are symmetrized only for unoriented quantities.
Signed QK is never interpreted as a probability. Missing statistics are NaN.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata

from screening_analysis.connection_audit import correlation, symmetric_edges
from screening_analysis.directed_loops import oriented_triangles
from screening_analysis.graph_wind import minimum_image_edges
from screening_analysis.rs_observables import attention_modes, row_entropy


def finite_mean(x):
    x = np.asarray(x, dtype=float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else np.nan


def corr(x, y, rank=False):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if rank:
        x, y = rankdata(x), rankdata(y)
    value = correlation(x, y)
    return np.nan if value is None else value


def positive_weights(w):
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all() or np.any(w < -1e-12):
        raise ValueError('finite nonnegative connection weights required')
    return np.maximum(w, 0.)


def sources(p):
    """qk_affine is a display control, NOT Alpha or a softmax probability."""
    return dict(qk_affine=np.clip((p['S'] + 1.) / 2., 0., 1.),
                alpha=positive_weights(p['Alpha']),
                window=positive_weights(p.get('Window', np.ones_like(p['Alpha']))),
                screening=positive_weights(p['W']))


def reciprocal_grid(lattice, nmax):
    modes = np.array([(i, j) for i in range(-nmax, nmax + 1)
                      for j in range(-nmax, nmax + 1)], dtype=int)
    return modes, modes @ (2 * np.pi * np.linalg.inv(lattice))


def pair_cosines(positions, lattice, modes):
    x = np.asarray(positions).reshape(-1, 2)
    i, j = np.triu_indices(len(x), 1)
    fractional = x @ np.linalg.inv(lattice).T
    return np.cos(2 * np.pi * (fractional[i] - fractional[j]) @ modes.T)


def pair_fourier(weights, cosines, signed=False):
    """S_A=sum_pairs A*cos/sum_pairs A; signed QK uses sum(abs(A)).

    A can end in [pairs], cosines=[pairs,q]. Positive nonempty S_A(0)=1.
    Zero mass is undefined, rather than an artificial zero spectrum.
    """
    w = np.asarray(weights, dtype=float)
    if not signed:
        w = positive_weights(w)
    mass = np.sum(abs(w) if signed else w, axis=-1)
    return np.divide(w @ cosines, mass[..., None],
                     out=np.full(w.shape[:-1] + (cosines.shape[-1],), np.nan),
                     where=mass[..., None] > 0)


def distance_statistics(distances, weights, bin_edges):
    """Pooled edge mass, CDF and exact weighted quantiles; no density ambiguity."""
    d, w = np.asarray(distances).ravel(), positive_weights(weights).ravel()
    if d.shape != w.shape or not np.isfinite(d).all():
        raise ValueError('aligned finite distances/weights required')
    mass = w.sum()
    hist = np.histogram(d, bins=bin_edges, weights=w)[0]
    if mass <= 0:
        return dict(mass=0., r_eff=np.nan, r50=np.nan, r90=np.nan, r95=np.nan,
                    probability=np.full(len(hist), np.nan), cdf=np.full(len(hist), np.nan))
    order = np.argsort(d, kind='stable')
    cumulative = np.cumsum(w[order])
    quantiles = {f'r{int(100*q)}': float(d[order[min(np.searchsorted(cumulative, q*mass), len(d)-1)]])
                 for q in (.5, .9, .95)}
    return dict(mass=float(mass), r_eff=float(d @ w / mass), **quantiles,
                probability=hist / mass, cdf=np.cumsum(hist) / mass)


def geometry(p, bins):
    n = int(p['NE']); ij = np.triu_indices(n, 1)
    physical = np.array([minimum_image_edges(x.reshape(n, 2), p['lattice'])[1]
                         for x in p['positions']])
    network = p['d'][:, ij[0], ij[1]]
    edges = np.linspace(0, float(p['dref'])*(1+1e-10), bins+1)
    labels = np.clip(np.searchsorted(edges, network, side='right')-1, 0, bins-1)
    # Undirected statistics retain ++ and --, and merge the mixed-spin class.
    spin_class = ((p['spins'][:, ij[0]] > 0).astype(int)
                  + (p['spins'][:, ij[1]] > 0).astype(int))
    return dict(physical=physical, network=network, edges=edges, bins=labels,
                spin_class=spin_class, labels=3*labels+spin_class,
                same=spin_class != 1)


def crossfit_residual(values, labels, min_count=20, epsilon=1e-8):
    """Two held-out configuration folds; no edge from the evaluation fold fits mu/s.

    Alternating configurations are folds, not independent MC chains. A zero
    conditional SD has undefined standardized residual. Return raw residual too.
    """
    y, labels = np.asarray(values, float), np.asarray(labels, int)
    if y.shape != labels.shape or y.ndim != 2 or min_count < 2 or epsilon <= 0:
        raise ValueError('aligned [configuration,pair] arrays, min_count>=2 required')
    prediction = np.full_like(y, np.nan); z = prediction.copy(); raw = prediction.copy()
    counts = np.zeros_like(labels)
    fold = np.arange(len(y)) % 2
    for held_out in (0, 1):
        train, test = fold != held_out, fold == held_out
        for label in np.unique(labels[test]):
            selected = train[:, None] & (labels == label) & np.isfinite(y)
            data = y[selected]; target = test[:, None] & (labels == label)
            counts[target] = len(data)
            if len(data) < min_count:
                continue
            mu, sd = data.mean(), data.std(ddof=1)
            prediction[target] = mu; raw[target] = y[target]-mu
            if sd > epsilon:
                z[target] = raw[target] / (sd + epsilon)
    return dict(prediction=prediction, raw=raw, z=z, counts=counts, fold=fold)


def conditional_correlations(x, y, labels, min_count=10):
    rows = []
    for label in np.unique(labels):
        mask = (labels == label) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= min_count:
            rows.append(dict(label=int(label), count=int(mask.sum()),
                             pearson=corr(x[mask], y[mask]), spearman=corr(x[mask], y[mask], True)))
    return rows


def top_overlap(x, y, fraction=.1):
    """Fraction of top sensitivity edges also in top weight edges; ties => NA."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all():
        return np.nan
    k = max(1, int(np.ceil(fraction*len(x))))
    a, b = np.argsort(-x), np.argsort(-y)
    if k < len(x) and (x[a[k-1]] == x[a[k]] or y[b[k-1]] == y[b[k]]):
        return np.nan
    return len(set(a[:k]) & set(b[:k]))/k


def head_specialization(weights, head_vectors=None):
    """Edge Pearson and JS (nats); output Gram includes signed cancellation."""
    w = positive_weights(weights); e = symmetric_edges(w)  # [B,H,E]
    h = w.shape[1]; pairs = []
    for i in range(h):
        for j in range(i, h):
            a, b = e[:, i].ravel(), e[:, j].ravel()
            js = float(jensenshannon(a/a.sum(), b/b.sum())**2) if min(a.sum(), b.sum()) > 0 else np.nan
            output_corr = (corr(head_vectors[:, :, i], head_vectors[:, :, j])
                           if head_vectors is not None else np.nan)
            pairs.append(dict(head_a=i, head_b=j, edge_pearson=corr(a, b), js_nats=js,
                              output_pearson=output_corr))
    edge_gram = np.einsum('bhe,bke->hk', e, e)
    def effective(g):
        denominator = np.sum(g*g)
        return float(np.trace(g)**2/denominator) if denominator > 0 else np.nan
    result = dict(edge_effective_heads=effective(edge_gram), output_effective_heads=np.nan,
                  cancellation_ratio=np.nan)
    if head_vectors is not None:
        gram = np.einsum('bnhe,bnke->hk', head_vectors, head_vectors)
        result.update(output_effective_heads=effective(gram),
                      cancellation_ratio=float(gram.sum()/np.trace(gram)) if np.trace(gram)>0 else np.nan)
    return pairs, result


def product_circulation(matrix, triangles):
    """Signed forward-minus-reverse PRODUCTS, not the existing sum circulation."""
    a = np.asarray(matrix); t = np.asarray(triangles, int).reshape(-1, 3)
    i, j, k = t.T
    return a[..., i, j]*a[..., j, k]*a[..., k, i] - a[..., i, k]*a[..., k, j]*a[..., j, i]


def mode_projections(weights, spins, positions, lattice, shell):
    """Both Rayleigh values and bounds use the exact SAME matrix/configuration."""
    w = positive_weights(weights); n = len(w)
    modes = attention_modes(w, spins)
    a = (w+w.T)/2; np.fill_diagonal(a, 0.)
    inv = np.divide(1., np.sqrt(modes['degree']), out=np.zeros(n), where=modes['degree']>0)
    m = inv[:, None]*a*inv[None, :]
    s = np.asarray(spins, float)-np.mean(spins)
    v = np.exp(2j*np.pi*(np.asarray(positions).reshape(n, 2) @ np.linalg.inv(lattice).T) @ shell.T)
    rows = []
    for name, matrix, eigenvalues in [('raw', a, modes['eigenvalues']),
                                       ('normalized', m, modes['normalized_eigenvalues'])]:
        norms = np.sum(abs(v)**2, axis=0)
        crystal = np.einsum('iq,ij,jq->q', v.conj(), matrix, v).real/norms
        spin = float(s@matrix@s/(s@s)) if s@s > 0 else np.nan
        lo, hi = float(eigenvalues.min()), float(eigenvalues.max())
        rayleigh = np.r_[crystal, spin]; finite = rayleigh[np.isfinite(rayleigh)]
        passed = bool(np.all((finite >= lo-1e-9) & (finite <= hi+1e-9)))
        if not passed:
            raise AssertionError('Rayleigh-Ritz failed for the same adjacency and vector')
        rows.append(dict(matrix=name, lambda_min=lo, lambda_max=hi, spin_rayleigh=spin,
                         crystal_rayleigh=float(crystal.mean()), rayleigh_ritz_pass=passed))
    # Degenerate modes are represented by projectors onto eigenspaces; a single
    # eigenvector in a degenerate subspace has no unique physical interpretation.
    ev, vec = modes['normalized_eigenvalues'], modes['normalized_eigenvectors']
    overlaps = abs(vec.T.conj() @ v)**2/n
    spin_overlap = abs(s @ vec)**2/(s@s) if s@s > 0 else np.full(n, np.nan)
    return rows, dict(eigenvalues=ev, eigenvectors=vec, crystal_mode_overlap=overlaps,
                      spin_overlap=spin_overlap, matrix=m)


def block_summary(values, blocks=None):
    """SEM across supplied contiguous time blocks, not across particles/heads."""
    x = np.asarray(values, float)
    finite = np.isfinite(x)
    result = dict(mean=finite_mean(x), configuration_sd=float(x[finite].std(ddof=1)) if finite.sum()>1 else np.nan,
                  block_sem=np.nan, blocks=0)
    if blocks and len(x) % blocks == 0 and blocks >= 4 and finite.all():
        b = x.reshape(blocks, -1).mean(axis=1)
        result.update(block_sem=float(b.std(ddof=1)/np.sqrt(blocks)), blocks=blocks)
    return result


def analyze_probe(p, settings, gradients=None):
    """Analyze one model/subset. Return small tables + regenerable plot arrays."""
    from screening_analysis.connection_study import analyze_connections
    from screening_analysis.lj_maps import first_shell
    b, nl, nh, n, _ = p['S'].shape
    if not np.isfinite(p['S']).all() or not np.isfinite(p['d']).all():
        raise ValueError('nonfinite probe similarities/distances')
    g = geometry(p, settings['distance_bins']); matrices = sources(p)
    modes, q = reciprocal_grid(p['lattice'], settings['qmax'])
    shell = np.asarray(first_shell(p['lattice'], n))
    peak = np.any(np.all(modes[:, None] == shell[None, :], axis=-1), axis=1)
    background = (np.linalg.norm(modes, axis=1)>0) & ~peak
    cosine = np.array([pair_cosines(x, p['lattice'], modes) for x in p['positions']])
    tables = {k: [] for k in ('head', 'configuration', 'fourier', 'distance', 'distance_summary',
                              'residual', 'head_pairs', 'circulation', 'phase', 'conditional', 'eigen')}
    if not peak.any():
        raise ValueError('qmax must include the crystal reference shell')
    arrays = dict(integer_modes=modes, q_vectors=q, shell=shell, positions=p['positions'], spins=p['spins'],
                  lattice=p['lattice'], network_distances=g['network'], physical_distances=g['physical'],
                  distance_bins=g['bins'], spin_class=g['spin_class'])
    # Reuse the existing distance analysis for actual Alpha/Window/Screening.
    # Its two metric conventions are retained verbatim.
    distance_rows, distance_summary = analyze_connections(p, dict(network_max=float(p['dref'])*(1+1e-10),
        physical_max=float(g['physical'].max())*(1+1e-10), count=settings['distance_bins']), settings['rs'])
    tables['distance'] = distance_rows
    tables['distance_summary'] = distance_summary
    for row in tables['distance_summary']:
        distances = g['network'] if row['distance_kind']=='network' else g['physical']
        values = symmetric_edges(matrices[row['source']])[:, row['layer'], row['head']]
        select = np.ones_like(g['same']) if row['spin']=='all' else g['same'] if row['spin']=='same' else ~g['same']
        stats = distance_statistics(distances[select], values[select], np.linspace(0, max(float(distances.max())*(1+1e-10), 1e-10), settings['distance_bins']+1))
        row.update(r50=stats['r50'], r95=stats['r95'])
        arrays[f"cdf_{row['distance_kind']}_{row['source']}_l{row['layer']}_h{row['head']}_{row['spin']}"] = stats['cdf']
    # QK has signed scores, so distance probability uses the explicitly named
    # affine display control; no hidden softmax or absolute-value convention.
    for layer in range(nl):
        for head in range(nh):
            for kind, distances in [('network', g['network']), ('minimum_image', g['physical'])]:
                for spin, mask in [('all', np.ones_like(g['same'])), ('same', g['same']), ('opposite', ~g['same'])]:
                    stats = distance_statistics(distances[mask], symmetric_edges(matrices['qk_affine'])[:, layer, head][mask],
                        np.linspace(0, float(distances.max())*(1+1e-10), settings['distance_bins']+1))
                    tables['distance_summary'].append(dict(distance_kind=kind, layer=layer, head=head,
                        spin=spin, source='qk_affine', **{k:v for k,v in stats.items() if k not in ('probability','cdf')}))
    for sample in range(b):
        rho = np.exp(2j*np.pi*(p['positions'][sample].reshape(n, 2) @ np.linalg.inv(p['lattice']).T) @ modes.T).sum(0)
        sf = abs(rho)**2/n
        tables['configuration'].append(dict(sample=sample, structure_factor_shell=float(sf[peak].mean()) if peak.any() else np.nan))
        arrays[f'structure_b{sample}'] = sf
    for name, a in matrices.items():
        edge = symmetric_edges(a)
        for layer in range(nl):
            for head in range(nh):
                w = edge[:, layer, head]
                f = np.array([pair_fourier(w[i], cosine[i]) for i in range(b)])
                arrays[f'fourier_{name}_l{layer}_h{head}'] = f
                arrays[f'weights_{name}_l{layer}_h{head}'] = w
                for k, mode in enumerate(modes):
                    tables['fourier'].append(dict(source=name, layer=layer, head=head, m1=int(mode[0]), m2=int(mode[1]),
                        qx=float(q[k, 0]), qy=float(q[k, 1]), value=finite_mean(f[:, k])))
                same = w[g['same']]; opposite = w[~g['same']]
                den = finite_mean(same)+finite_mean(opposite)
                selectivity = (finite_mean(same)-finite_mean(opposite))/den if den>0 else np.nan
                entropy = row_entropy(a[:, layer, head])
                residual = crossfit_residual(w, g['labels'], settings['min_bin_count'])
                valid = np.isfinite(residual['z'])
                zspec = np.array([np.nanmean(residual['z'][i, :, None]*cosine[i], axis=0)
                                 if valid[i].any() else np.full(len(modes), np.nan) for i in range(b)])
                arrays[f'residual_spectrum_{name}_l{layer}_h{head}'] = zspec
                arrays[f'residual_z_{name}_l{layer}_h{head}'] = residual['z']
                arrays[f'residual_prediction_{name}_l{layer}_h{head}'] = residual['prediction']
                # Compare on EXACTLY the same fitted support. Unfitted bins
                # must never become zero-weight evidence for a learned excess.
                fitted = np.isfinite(residual['prediction'])
                baseline = np.array([pair_fourier(np.where(fitted[i], residual['prediction'][i], 0.), cosine[i]) for i in range(b)])
                matched = np.array([pair_fourier(np.where(fitted[i], w[i], 0.), cosine[i]) for i in range(b)])
                arrays[f'conditional_fourier_{name}_l{layer}_h{head}'] = baseline
                arrays[f'matched_fourier_{name}_l{layer}_h{head}'] = matched
                for db in range(settings['distance_bins']):
                    mask = g['bins']==db
                    tables['residual'].append(dict(source=name, layer=layer, head=head, distance_bin=db,
                        distance=float((g['edges'][db]+g['edges'][db+1])/2),
                        original_mean=finite_mean(w[mask]), residual_mean=finite_mean(residual['z'][mask]),
                        valid_edges=int(valid[mask].sum()), total_edges=int(mask.sum())))
                stats = distance_statistics(g['physical'], w, np.linspace(0, max(g['physical'].max(), 1e-10)*(1+1e-10), settings['distance_bins']+1))
                network_stats = distance_statistics(g['network'], w, g['edges'])
                peak_value = finite_mean(f[:, peak])
                bg = finite_mean(abs(f[:, background]))
                tables['head'].append(dict(source=name, layer=layer, head=head,
                    width=float(p['w'][layer, head]), r50=stats['r50'], r90=stats['r90'], r95=stats['r95'], r_eff=stats['r_eff'],
                    r90_network=network_stats['r90'],
                    spin_selectivity=selectivity, entropy=finite_mean(entropy['normalized']),
                    crystal_peak=peak_value, background_abs=bg, peak_background=peak_value/bg if bg>1e-12 else np.nan,
                    residual_variance=float(np.var(residual['z'][valid], ddof=1)) if valid.sum()>1 else np.nan,
                    residual_coverage=float(valid.mean()), residual_peak=finite_mean(zspec[:, peak]),
                    crystal_increment=finite_mean(matched[:, peak]-baseline[:, peak])))
                if gradients is not None:
                    pi, gi, derivative = gradients
                    x = w[pi]; labels = g['labels'][pi]
                    for channel, ci in [('amplitude', 0), ('phase', 1)]:
                        response = abs(derivative[gi, ci, layer, head])
                        overlaps = [top_overlap(xx, yy) for xx, yy in zip(x, response)]
                        tables['phase'].append(dict(source=name, layer=layer, head=head, channel=channel,
                            samples=len(pi), pearson=corr(x, response), spearman=corr(x, response, True),
                            top10_overlap=finite_mean(overlaps), mean_abs_sensitivity=finite_mean(response)))
                        for control, lab in [('distance', g['bins'][pi]), ('spin', g['spin_class'][pi]), ('distance_spin', labels)]:
                            for row in conditional_correlations(x, response, lab):
                                tables['conditional'].append(dict(source=name, layer=layer, head=head, channel=channel, control=control, **row))
                        arrays[f'sensitivity_{name}_{channel}_l{layer}_h{head}'] = np.stack((x, response))
    equal = np.ones((b, n*(n-1)//2))
    for name, w in [('equal', equal), ('distance', np.exp(-g['network']/float(p['dref']))),
                    ('spin', 1.+g['same']), ('qk_signed', symmetric_edges(p['S'])[:, 0].mean(axis=1))]:
        arrays[f'control_{name}'] = np.array([pair_fourier(w[i], cosine[i], signed=name=='qk_signed') for i in range(b)])
    for layer in range(nl):
        vectors = p.get('head_vectors')
        pairs, summary = head_specialization(p['W'][:, layer], vectors[:, layer] if vectors is not None else None)
        tables['head_pairs'].extend(dict(layer=layer, **r) for r in pairs)
        arrays[f'head_summary_l{layer}'] = np.array(list(summary.values()))
        for sample in range(b):
            triangles = oriented_triangles(p['positions'][sample], p['lattice'])
            t = triangles['triangles']; arrays[f'triangles_b{sample}'] = t
            for name, a in [('qk_signed', p['S']), ('qk_affine', matrices['qk_affine']), ('screening', p['W'])]:
                for head in range(nh):
                    matrix = a[sample, layer, head]
                    circulation = product_circulation(matrix, t)
                    arrays[f'circulation_{name}_b{sample}_l{layer}_h{head}'] = circulation
                    for spin, mask in [('all', np.ones(len(t), bool)),
                                       ('same', np.all(p['spins'][sample][t]==p['spins'][sample][t[:, :1]], axis=1) if len(t) else np.zeros(0, bool)),
                                       ('mixed', np.any(p['spins'][sample][t]!=p['spins'][sample][t[:, :1]], axis=1) if len(t) else np.zeros(0, bool))]:
                        c = circulation[mask]
                        tables['circulation'].append(dict(sample=sample, layer=layer, head=head, source=name, spin=spin,
                            triangles=int(mask.sum()), ccw_mean=finite_mean(c), rms=np.sqrt(finite_mean(c*c)),
                            asymmetry_rms=float(np.sqrt(np.mean(symmetric_edges((matrix-matrix.T)**2)))),
                            excluded=sum(triangles['excluded'].values())))
            for head in range(nh):
                rows, mode_data = mode_projections(p['W'][sample, layer, head], p['spins'][sample], p['positions'][sample], p['lattice'], shell)
                tables['eigen'].extend(dict(sample=sample, layer=layer, head=head, **r) for r in rows)
                for key, value in mode_data.items():
                    arrays[f'eigen_{key}_b{sample}_l{layer}_h{head}'] = value
    return tables, arrays
