"""Permutation-invariant topology and conventional physical-structure baselines."""
from itertools import product
import numpy as np
from screening_analysis.graph_wind import minimum_image_edges
from screening_analysis import topology


def features(probe,count=None):
    available=len(probe['positions']);count=min(count or available,available)
    indices=np.linspace(0,available-1,count).astype(int)
    N=int(probe['NE']);L=np.asarray(probe['lattice']);area=abs(np.linalg.det(L))
    positions=np.asarray(probe['positions'])[indices].reshape(count,N,2)
    frac=np.mod(positions@np.linalg.inv(L).T,1.)
    density=np.zeros((4,4))
    for r in frac:density+=np.histogram2d(r[:,0],r[:,1],bins=4,range=((0,1),(0,1)))[0]
    density=density.ravel()/(count*N)
    # Complete Euclidean annuli fit inside half the shortest cell vector.
    translations=np.asarray([p for p in product(range(-3,4),repeat=2) if p!=(0,0)])@L.T
    rmax=.45*np.linalg.norm(translations,axis=-1).min();edges=np.linspace(0,rmax,9)
    distances=np.stack([minimum_image_edges(r,L)[1] for r in positions]);hist=np.histogram(distances,bins=edges)[0]/count
    normalizer=N*(N-1)/(2*area)*np.pi*(edges[1:]**2-edges[:-1]**2)
    gr=np.divide(hist,normalizer,out=np.zeros_like(hist,dtype=float),where=normalizer>0)
    q=np.array([(1,0),(0,1),(1,1),(1,-1),(2,0),(0,2),(2,1),(1,2)])
    rho=np.exp(2j*np.pi*np.einsum('bnd,kd->bnk',frac,q)).sum(axis=1)
    sf=np.mean(abs(rho)**2,axis=0)/N
    d=distances.ravel();distance_stats=np.r_[d.mean(),d.std(),np.quantile(d,[.1,.25,.5,.75,.9])]
    widths=[]
    for key in ('w_sigma','w_centre','w_outer','r'):
        a=np.asarray(probe[key]).ravel();a=a[np.isfinite(a)]
        widths.extend(np.quantile(a,[0,.25,.5,.75,1]) if len(a) else np.zeros(5))
    ws=np.asarray(probe['W_window'])[indices].mean(axis=(1,2)) # mean layers/heads, no head identity
    off=np.triu_indices(N,1);weight_stats=[];ph_raw=[];ph_complete=[];diagrams={}
    for i,w in enumerate(ws):
        b=topology.symmetrize(w);values=b[off]
        weight_stats.append(np.r_[np.mean(values>0),values.mean(),values.std(),np.quantile(values,[.1,.5,.9])])
        for mode,graph,collect in (('raw',b,ph_raw),('completed',(1-1e-4)*b+1e-4*(1-np.eye(N)),ph_complete)):
            ds=topology.persistence(graph);vector=[]
            for dim,diagram in enumerate(ds):
                summary=topology.summary(diagram)
                vector.extend(summary[k] for k in ('total_persistence','max_persistence','finite_intervals','essential_intervals'))
                vector.extend(topology.landscape(diagram,np.linspace(0,1,21),3).ravel())
                diagrams[f'{mode}_b{i}_H{dim}']=diagram
            collect.append(vector)
    def moments(values):
        a=np.asarray(values);return np.r_[a.mean(axis=0),a.std(axis=0)].tolist()
    result=dict(density=density.tolist(),gr=gr.tolist(),structure_factor=sf.tolist(),distances=distance_stats.tolist(),
        widths=np.asarray(widths).tolist(),weight_distribution=moments(weight_stats),topology_raw=moments(ph_raw),topology_completed=moments(ph_complete))
    return dict(features=result,diagrams=diagrams,selected_indices=indices,
        gr_definition='canonical finite-N g(r), uniform=1; full Euclidean annuli below .45 shortest lattice vector',
        gr_edges=edges,structure_factor_integer_modes=q,head_pool='mean W_window over layers and heads',
        PH_completion_floor=1e-4,physical_phase_status='not inferred')
