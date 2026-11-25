"""Weighted double-edge swaps preserving per-node spin-block degrees and distance bins.

Constraints hold for the supplied (e.g. top-k) graph, not at every filtration
threshold. Finite swap chains are conditional controls, not certified uniform
samples. Frozen/poorly moved graphs must not be used as evidence of significance.
"""
import numpy as np


def distance_bins(distances,bins=8):
    d=np.asarray(distances,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError('finite nonnegative square distances')
    if not np.allclose(d,d.T,atol=1e-12,rtol=0):raise ValueError('symmetric distances required')
    values=d[np.triu_indices(len(d),1)]
    cuts=np.unique(np.quantile(values,np.linspace(0,1,bins+1)))
    return np.searchsorted(cuts[1:-1],d,side='right'),cuts


def spin_degrees(graph,spins):
    a=np.asarray(graph)>0;s=np.asarray(spins)
    return np.stack([(a&(s[:,None]==s[None,:])).sum(axis=1),
                     (a&(s[:,None]!=s[None,:])).sum(axis=1)],axis=1)


def available_switches(weights,spins,admissible=None):
    """Exhaustive distinct degree/spin-preserving two-switches, ignoring distance."""
    w=np.asarray(weights);s=np.asarray(spins);edges=[tuple(e) for e in np.argwhere(np.triu(w>0,1))]
    allowed=np.ones_like(w,dtype=bool) if admissible is None else admissible
    moves=set()
    for i,(a,b) in enumerate(edges):
        for c,d in edges[i+1:]:
            if len({a,b,c,d})!=4:continue
            for u,v in ((c,d),(d,c)):
                if s[a]!=s[u] or s[b]!=s[v]:continue
                x,y=tuple(sorted((a,v))),tuple(sorted((u,b)))
                if w[x]>0 or w[y]>0 or not allowed[x] or not allowed[y]:continue
                moves.add(tuple(sorted((x,y))))
    return len(moves)


def spin_distance_shuffle(weights,spins,distances,rng,bins=8,admissible=None):
    """Relaxed control: exact spin-block/distance-bin weight multisets, NOT degrees."""
    w=np.asarray(weights);s=np.asarray(spins);n=len(w);ij=np.triu_indices(n,1)
    labels,cuts=distance_bins(distances,bins)
    allowed=np.ones_like(w,dtype=bool) if admissible is None else np.asarray(admissible,bool)
    if np.any((w>0)&~allowed):raise ValueError('original graph outside admissible support')
    classes=(s[ij[0]]>0).astype(int)+(s[ij[1]]>0).astype(int)
    values=w[ij].copy();distance_labels=labels[ij]
    for block in range(3):
        for band in np.unique(distance_labels):
            indices=np.flatnonzero((classes==block)&(distance_labels==band)&allowed[ij])
            values[indices]=rng.permutation(values[indices])
    out=np.zeros_like(w);out[ij]=values;out+=out.T
    degrees=spin_degrees(out,s)-spin_degrees(w,s)
    return out,dict(degree_rmse=float(np.sqrt(np.mean(degrees**2))),
        max_node_spin_degree_change=int(abs(degrees).max(initial=0)),distance_bin_edges=cuts.tolist(),
        exact_degrees_preserved=bool(not degrees.any()))
def rewire(weights,spins,distances,rng,*,bins=8,switches=None,max_attempts=None,admissible=None):
    w=np.asarray(weights,dtype=float).copy();s=np.asarray(spins)
    if w.ndim!=2 or w.shape[0]!=w.shape[1] or not np.isfinite(w).all() or np.any(w<0) or not np.array_equal(w,w.T) or np.any(np.diag(w)!=0):
        raise ValueError('finite nonnegative symmetric matrix with zero diagonal required')
    n=len(w)
    if s.shape!=(n,) or not np.isin(s,[-1,1]).all():raise ValueError('one +/-1 pseudospin per node required')
    labels,cuts=distance_bins(distances,bins)
    allowed=np.ones((n,n),bool) if admissible is None else np.asarray(admissible,bool)
    if allowed.shape!=w.shape or not np.array_equal(allowed,allowed.T) or np.any((w>0)&~allowed):
        raise ValueError('symmetric admissible support containing the original graph required')
    edges=[tuple(e) for e in np.argwhere(np.triu(w>0,1))]
    initial=w.copy();E=len(edges)
    target=5*E if switches is None else int(switches)
    budget=max(1000,50*target) if max_attempts is None else int(max_attempts)
    accepted=0;attempts=0
    while E>=2 and accepted<target and attempts<budget:
        attempts+=1;i,j=rng.choice(E,2,replace=False)
        a,b=edges[i];c,d=edges[j]
        if rng.integers(2):a,b=b,a
        if rng.integers(2):c,d=d,c
        if len({a,b,c,d})!=4 or s[a]!=s[c] or s[b]!=s[d]:continue
        e0,e1=(a,d),(c,b)
        if w[e0]>0 or w[e1]>0 or not allowed[e0] or not allowed[e1]:continue
        old=[int(labels[a,b]),int(labels[c,d])];new=[int(labels[e0]),int(labels[e1])]
        if sorted(old)!=sorted(new):continue
        v0,v1=w[a,b],w[c,d]
        if old!=new:v0,v1=v1,v0
        w[a,b]=w[b,a]=0;w[c,d]=w[d,c]=0
        w[e0]=w[e0[::-1]]=v0;w[e1]=w[e1[::-1]]=v1
        edges[i]=tuple(sorted(e0));edges[j]=tuple(sorted(e1));accepted+=1
    original=initial>0;final=w>0;changed=int(np.count_nonzero(np.triu(original&~final,1)))
    return w,dict(edges=E,attempts=attempts,accepted=accepted,target=target,
        target_reached=accepted>=target,changed_edge_fraction=changed/E if E else 0.,
        distance_bin_edges=cuts.tolist(),
        preserved='per-node degree separately for same/opposite spin; exact per-distance-bin edge-weight multiset; edge count; admissible support',
        sampling_status='finite constrained switch chain; uniformity/independence not certified',
        movable=bool(accepted>0 and changed>0))
