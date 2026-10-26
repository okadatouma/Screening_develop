"""Small weighted flag-complex PH over F2. No model/JAX imports or metric assumption.

Weights must be in [0,1]. Vertices exist at filtration 0, positive edges at
u=1-B, triangles at max(edge u). Zero edges are absent, including at u=1.
Only the 2-skeleton is needed for H0/H1. Essential intervals retain +inf.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


def symmetrize(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not len(w) or not np.isfinite(w).all():
        raise ValueError('finite nonempty square weight matrix required')
    b = (w + w.T) / 2
    if np.any(b < 0) or np.any(b > 1):
        raise ValueError('weights must be in [0,1]; map QK to (s+1)/2 explicitly')
    np.fill_diagonal(b, 0.)
    return b


def rank_weights(weights):
    """Positive-edge CDF (midrank/M), exact ties retained; zeros stay zero.

    Removes positive weight scale/distribution, NOT support-density differences.
    At a fixed positive-edge quantile ties can give different realized densities.
    """
    b = symmetrize(weights); ij = np.triu_indices(len(b), 1); x = b[ij]
    positive = x > 0
    x[positive] = rankdata(x[positive], method='average') / positive.sum()
    b[ij] = x; b[(ij[1], ij[0])] = x
    return b


def shuffled_weights(weights, rng):
    """Exact edge-weight multiset and support count; random edge placement."""
    b = symmetrize(weights); ij = np.triu_indices(len(b), 1)
    values = rng.permutation(b[ij]); b[ij] = values; b[(ij[1], ij[0])] = values
    return b


def persistence(weights, max_vertices=64):
    b = symmetrize(weights); n = len(b)
    if n > max_vertices:
        raise ValueError('reference F2 reduction limited to 64 vertices; use a standard PH backend for larger N')
    simplices = [(0., (i,)) for i in range(n)]
    simplices += [(1-b[i,j], (i,j)) for i,j in combinations(range(n),2) if b[i,j] > 0]
    simplices += [(1-min(b[i,j],b[i,k],b[j,k]), (i,j,k))
                  for i,j,k in combinations(range(n),3) if min(b[i,j],b[i,k],b[j,k]) > 0]
    simplices.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    indices = {s:i for i,(_,s) in enumerate(simplices)}
    reduced, pivots, births, paired = {}, {}, set(), set()
    diagrams = [[], []]
    for j,(time,simplex) in enumerate(simplices):
        column = ({indices[face] for face in combinations(simplex,len(simplex)-1)}
                  if len(simplex)>1 else set())
        while column and max(column) in pivots:
            column ^= reduced[pivots[max(column)]]
        reduced[j] = column
        if not column:
            births.add(j)
        else:
            i = max(column); pivots[i] = j; paired.add(i)
            dim = len(simplices[i][1])-1; birth = simplices[i][0]
            if dim < 2 and time > birth:
                diagrams[dim].append((birth,time))
    for i in sorted(births-paired):
        dim = len(simplices[i][1])-1
        if dim < 2:
            diagrams[dim].append((simplices[i][0],np.inf))
    return tuple(np.asarray(sorted(d),dtype=float).reshape(-1,2) for d in diagrams)


def betti(diagram, thresholds):
    """Strict B>t: u_birth < 1-t <= u_death. Vertices always present, even t=1."""
    d = np.asarray(diagram).reshape(-1,2)
    u = 1-np.asarray(thresholds)
    return ((d[:,0,None] < u) & (u <= d[:,1,None])).sum(axis=0)


def landscape(diagram, grid, levels=3):
    d = np.asarray(diagram).reshape(-1,2); d = d[np.isfinite(d[:,1])]
    tents = np.maximum(0., np.minimum(np.asarray(grid)[None,:]-d[:,0,None],
                                     d[:,1,None]-np.asarray(grid)[None,:]))
    return np.sort(np.vstack((tents, np.zeros((levels,len(grid))))),axis=0)[-levels:][::-1]


def summary(diagram):
    d = np.asarray(diagram).reshape(-1,2); finite = np.isfinite(d[:,1])
    life = d[finite,1]-d[finite,0]
    return dict(total_persistence=float(life.sum()), max_persistence=float(life.max(initial=0.)),
                finite_intervals=int(finite.sum()), essential_intervals=int((~finite).sum()))


def diagram_distance(left, right, order=1., bottleneck=False):
    """Exact assignment with diagonal, L_inf ground cost; essential births matched.

    Unequal essential counts -> inf. Wasserstein order >=1. No finite capping.
    """
    if not np.isfinite(order) or order < 1:
        raise ValueError('finite Wasserstein order >=1 required')
    a,b = (np.asarray(d,dtype=float).reshape(-1,2) for d in (left,right))
    for d in (a,b):
        if not np.isfinite(d[:,0]).all() or np.isnan(d).any() or np.any(d[:,1]<d[:,0]):
            raise ValueError('invalid persistence diagram')
    ea,eb = (np.sort(d[np.isinf(d[:,1]),0]) for d in (a,b))
    if len(ea)!=len(eb): return float('inf')
    essential = np.abs(ea-eb)
    a,b = (d[np.isfinite(d[:,1])] for d in (a,b)); n,m = len(a),len(b)
    cost = np.full((n+m,n+m),np.inf)
    if n and m: cost[:n,:m] = np.max(np.abs(a[:,None,:]-b[None,:,:]),axis=-1)
    if n: cost[np.arange(n),m+np.arange(n)] = (a[:,1]-a[:,0])/2
    if m: cost[n+np.arange(m),np.arange(m)] = (b[:,1]-b[:,0])/2
    cost[n:,m:] = 0.
    if bottleneck:
        candidates = np.unique(cost[np.isfinite(cost)])
        lo,hi = 0,len(candidates)-1
        while lo<hi:
            mid=(lo+hi)//2
            rows,cols=linear_sum_assignment((cost>candidates[mid]).astype(int))
            if np.all(cost[rows,cols]<=candidates[mid]): hi=mid
            else: lo=mid+1
        return float(max(candidates[lo] if len(candidates) else 0.,essential.max(initial=0.)))
    rows,cols=linear_sum_assignment(cost**order)
    return float((np.sum(cost[rows,cols]**order)+np.sum(essential**order))**(1/order))


def analyze_graph(weights, thresholds=None):
    b = symmetrize(weights)
    thresholds = np.linspace(0,1,51) if thresholds is None else np.asarray(thresholds)
    d0,d1 = persistence(b)
    edges = b[np.triu_indices(len(b),1)]
    # Select the same graph using an interior plateau value. Computing 1-t at
    # an edge/nextafter threshold can erase the strict inequality by rounding.
    curves=[]
    for t in thresholds:
        active=edges[(edges>t)&(edges>0)]
        if not len(active): curves.append((len(b),0)); continue
        excluded=edges[(edges<=t)|(edges==0)]
        representative=(active.min()+excluded.max(initial=0.))/2
        curves.append(tuple(int(betti(d,[representative])[0]) for d in (d0,d1)))
    return dict(diagrams=(d0,d1), betti=np.asarray(curves).T,
                thresholds=thresholds, density=(edges[:,None]>thresholds).mean(axis=0) if len(edges) else np.zeros(len(thresholds)),
                summaries=(summary(d0),summary(d1)),
                landscape=np.stack([landscape(d,np.linspace(0,1,51)) for d in (d0,d1)]))
