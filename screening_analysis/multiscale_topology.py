"""Two-parameter graph Betti/winding surfaces. These are NOT 1D PH diagrams."""
from itertools import combinations
import numpy as np
from screening_analysis.graph_wind import UF,minimum_image_edges


def binary_betti(adjacency):
    """Independent F2 triangle-boundary rank, retaining every isolated vertex."""
    a=np.asarray(adjacency,dtype=bool).copy()
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.array_equal(a,a.T):raise ValueError('symmetric square adjacency required')
    np.fill_diagonal(a,False);n=len(a);edges=[e for e in combinations(range(n),2) if a[e]]
    uf=UF(n)
    for i,j in edges:uf.union(i,j,np.zeros(2))
    components=len({uf.find(i)[0] for i in range(n)})
    index={e:k for k,e in enumerate(edges)};pivots={}
    for i,j,k in combinations(range(n),3):
        if not (a[i,j] and a[j,k] and a[i,k]):continue
        column=(1<<index[(i,j)])^(1<<index[(j,k)])^(1<<index[(i,k)])
        while column:
            pivot=column.bit_length()-1
            if pivot not in pivots:pivots[pivot]=column;break
            column^=pivots[pivot]
    return np.array((components,len(edges)-n+components-len(pivots)))


def distance_matrix(positions,lattice):
    delta,d,ties=minimum_image_edges(np.asarray(positions).reshape(-1,2),lattice)
    n=len(np.asarray(positions).reshape(-1,2));ij=np.triu_indices(n,1)
    matrix=np.zeros((n,n));matrix[ij]=d;matrix[(ij[1],ij[0])]=d
    return matrix,delta,ties


def surface(score, distances, delta, ties, a_grid, b_grid, support=None):
    """Exact strict score>a, d<b, with shortest-image embedded edges."""
    s=np.asarray(score,dtype=float);d=np.asarray(distances,dtype=float);n=len(s)
    if s.shape!=(n,n) or d.shape!=s.shape or not np.isfinite(s).all():raise ValueError('finite square score and distance required')
    support=np.ones_like(s,dtype=bool) if support is None else np.asarray(support,dtype=bool)
    sym=(s+s.T)/2;ij=np.triu_indices(n,1);betti=np.zeros((len(a_grid),len(b_grid),2),int)
    rank=np.zeros((len(a_grid),len(b_grid)),float);density=np.zeros_like(rank)
    for ia,a in enumerate(a_grid):
        for ib,b in enumerate(b_grid):
            adjacency=(sym>a)&(d<b)&support;np.fill_diagonal(adjacency,False)
            betti[ia,ib]=binary_betti(adjacency);active=adjacency[ij]
            density[ia,ib]=active.mean() if len(active) else 0.
            if np.any(active&ties):rank[ia,ib]=np.nan;continue
            uf=UF(n)
            for k in np.flatnonzero(active):uf.union(int(ij[0][k]),int(ij[1][k]),delta[k])
            rank[ia,ib]=uf.rank()
    return dict(betti=betti,rank=rank,density=density)


def analyze_multiscreen(similarity, window, positions, lattice, a_grid, b_grid,
                       periodic_distance=None, distance_bands=None, head_vectors=None):
    """One configuration [L,H,N,N], both per-head and head-permutation invariants.

    Default physical cutoffs use Euclidean minimum-image distance. An explicit
    periodic_distance switches the cutoff metric, NEVER the winding embedding.
    Window-weighted integrated score is sum_h(m_h*s_sym_h)/sum_h(m_h).
    Window bands below are physical distance bands, not thresholded Gaussian tails.
    """
    s=np.asarray(similarity);m=np.asarray(window)
    if s.ndim!=4 or s.shape!=m.shape or not np.isfinite(m).all() or np.any(m<0):raise ValueError('matching S/window [L,H,N,N] required')
    L,H,N,_=s.shape;d,delta,ties=distance_matrix(positions,lattice)
    cutoff=d if periodic_distance is None else np.asarray(periodic_distance)
    bands=np.asarray(distance_bands if distance_bands is not None else b_grid)
    if bands.ndim!=1 or len(bands)<2 or not np.all(np.diff(bands)>0):raise ValueError('ascending distance band edges required')
    sym=(s+s.swapaxes(-1,-2))/2;m=(m+m.swapaxes(-1,-2))/2
    per_head=[];window_heads=[];integrated=[];unions=[];consensus=[];redundancy=[];band_mass=[];band_betti=[]
    off=~np.eye(N,dtype=bool)
    for layer in range(L):
        per_head.append([surface(sym[layer,h],cutoff,delta,ties,a_grid,b_grid) for h in range(H)])
        window_heads.append([surface(sym[layer,h],cutoff,delta,ties,a_grid,b_grid,m[layer,h]>0) for h in range(H)])
        mass=m[layer].sum(axis=0)
        combined=np.divide((m[layer]*sym[layer]).sum(axis=0),mass,out=np.zeros((N,N)),where=mass>0)
        integrated.append(surface(combined,cutoff,delta,ties,a_grid,b_grid,mass>0))
        union=[];majority=[];overlaps=[]
        for a in a_grid:
            uu=[];cc=[];oo=[]
            for b in b_grid:
                adjacency=(sym[layer]>a)&(cutoff[None,:,:]<b)&(m[layer]>0)&off
                uu.append(binary_betti(adjacency.any(axis=0)));cc.append(binary_betti(adjacency.mean(axis=0)>=.5))
                values=[]
                for h,k in combinations(range(H),2):
                    union_count=np.count_nonzero(adjacency[h]|adjacency[k])
                    # Empty-vs-empty is undefined, not perfect redundancy.
                    if union_count:values.append(np.count_nonzero(adjacency[h]&adjacency[k])/union_count)
                oo.append(np.mean(values) if values else np.nan)
            union.append(uu);majority.append(cc);overlaps.append(oo)
        unions.append(union);consensus.append(majority);redundancy.append(overlaps)
        band_masks=[(cutoff>=lo)&(cutoff<hi)&off for lo,hi in zip(bands[:-1],bands[1:])]
        band_mass.append(np.stack([(m[layer]*mask).sum(axis=(-1,-2)) for mask in band_masks],axis=-1))
        band_betti.append([[surface(sym[layer,h],cutoff,delta,ties,a_grid,[np.inf],mask)['betti'][:,0]
                            for mask in band_masks] for h in range(H)])
    result=dict(a_grid=np.asarray(a_grid),b_grid=np.asarray(b_grid),distance_bands=bands,
        distance=cutoff,head_betti=np.asarray([[x['betti'] for x in layer] for layer in per_head]),
        head_rank=np.asarray([[x['rank'] for x in layer] for layer in per_head]),
        window_head_betti=np.asarray([[x['betti'] for x in layer] for layer in window_heads]),
        integrated_betti=np.asarray([x['betti'] for x in integrated]),
        integrated_rank=np.asarray([x['rank'] for x in integrated]),
        union_betti=np.asarray(unions),consensus_betti=np.asarray(consensus),head_jaccard=np.asarray(redundancy),
        window_band_mass=np.asarray(band_mass),band_betti=np.asarray(band_betti))
    baseline=surface(np.ones((N,N)),cutoff,delta,ties,[0.],b_grid)
    result['distance_only_betti']=baseline['betti'][0]
    result['distance_only_rank']=baseline['rank'][0]
    result['head_mean_betti']=result['head_betti'].mean(axis=1)
    result['head_std_betti']=result['head_betti'].std(axis=1)
    result['head_max_betti']=result['head_betti'].max(axis=1)
    totals=result['window_band_mass'].sum(axis=-1,keepdims=True)
    result['window_band_fraction']=np.divide(result['window_band_mass'],totals,
                                            out=np.full_like(result['window_band_mass'],np.nan),where=totals>0)
    if head_vectors is not None:
        c=np.asarray(head_vectors)  # [L,N,H,E]
        gram=np.einsum('lnhe,lnke->lhk',c,c)/N;den=np.trace(gram,axis1=-2,axis2=-1)
        result['head_gram']=gram
        result['cancellation_ratio']=np.divide(gram.sum(axis=(-1,-2)),den,out=np.full_like(den,np.nan),where=den>0)
        energy=np.einsum('lnhe,lnhe->lh',c,c)/N
        result['band_energy_proxy']=(energy[:,:,None]*np.nan_to_num(result['window_band_fraction'])).sum(axis=1)
        # Nonlinear TanhNorm/Gate forbid treating this proxy as an additive output decomposition.
    return result
