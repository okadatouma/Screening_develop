"""Geometrically oriented, contractible triangles of periodic embedded graphs.

Real QK/Screening circulations are descriptive, nonquantized network quantities.
They neither certify broken time reversal nor determine the sign of a Chern number.
"""
from itertools import combinations
import numpy as np
from screening_analysis.graph_wind import minimum_image_edges


def oriented_triangles(positions, lattice, area_tolerance=1e-12):
    """CCW shortest-image triangles, independent of particle/site enumeration.

    No spanning surface is chosen for noncontractible 3-edge loops. Such loops,
    degenerate shortest images and collinear triples are explicitly excluded.
    """
    x=np.asarray(positions,dtype=float).reshape(-1,2);L=np.asarray(lattice,dtype=float)
    delta,dist,ties=minimum_image_edges(x,L);ij=np.triu_indices(len(x),1)
    edge={tuple(pair):k for k,pair in enumerate(zip(*ij))}
    triangles=[];perimeters=[];longest=[];areas=[]
    excluded=dict(shortest_image_tie=0,noncontractible=0,collinear=0)
    for i,j,k in combinations(range(len(x)),3):
        ab,bc,ac=(edge[e] for e in ((i,j),(j,k),(i,k)))
        if ties[[ab,bc,ac]].any():excluded['shortest_image_tie']+=1;continue
        winding=delta[ab]+delta[bc]-delta[ac]
        if np.max(np.abs(winding))>1e-9:excluded['noncontractible']+=1;continue
        u,v=L@delta[ab],L@delta[ac]
        cross=u[0]*v[1]-u[1]*v[0]
        if abs(cross)<=area_tolerance*abs(np.linalg.det(L)):
            excluded['collinear']+=1;continue
        triangles.append((i,j,k) if cross>0 else (i,k,j))
        perimeters.append(dist[[ab,bc,ac]].sum());longest.append(dist[[ab,bc,ac]].max());areas.append(abs(cross)/2)
    return dict(triangles=np.asarray(triangles,dtype=int).reshape(-1,3),
                perimeter=np.asarray(perimeters),longest_edge=np.asarray(longest),area=np.asarray(areas),excluded=excluded)


def bounded_asymmetry(weights, epsilon=1e-8):
    w=np.asarray(weights,dtype=float)
    if not np.isfinite(epsilon) or epsilon<=0:raise ValueError('finite epsilon>0 required')
    if w.shape[-1]!=w.shape[-2] or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('finite nonnegative square weights required')
    return (w-w.swapaxes(-1,-2))/(w+w.swapaxes(-1,-2)+epsilon)


def circulation(antisymmetric, triangles):
    a=np.asarray(antisymmetric);t=np.asarray(triangles,dtype=int).reshape(-1,3)
    return a[...,t[:,0],t[:,1]]+a[...,t[:,1],t[:,2]]+a[...,t[:,2],t[:,0]]


def analyze_loops(similarity, weights, positions, lattice, epsilon=1e-8,
                  epsilon_sweep=(1e-4,1e-8,1e-12), perimeter_edges=None):
    """Single configuration [L,H,N,N]; all retained loops are oriented CCW.

    raw_forward/reverse are the signed CCW and CW circulations, NOT electron-id
    order. Reflection reverses these signs; periodic translations do not.
    Log ratios are optional diagnostics regularized before taking log.
    """
    s=np.asarray(similarity,dtype=float);w=np.asarray(weights,dtype=float)
    if s.shape!=w.shape or s.ndim!=4 or not np.isfinite(s).all():raise ValueError('matching [L,H,N,N] arrays required')
    geometry=oriented_triangles(positions,lattice);t=geometry['triangles']
    qk=circulation(s-s.swapaxes(-1,-2),t)
    screening=circulation(bounded_asymmetry(w,epsilon),t)
    values=np.asarray([circulation(bounded_asymmetry(w,e),t) for e in epsilon_sweep])
    # Each zero weight becomes epsilon BEFORE log; no log(0) intermediate.
    log_ratio=np.log(w+epsilon)-np.log(w.swapaxes(-1,-2)+epsilon)
    logs=circulation(log_ratio,t)
    perimeter=geometry['perimeter']
    edges=(np.linspace(0,3*max(np.linalg.norm(np.asarray(lattice),axis=0)),9)
           if perimeter_edges is None else np.asarray(perimeter_edges))
    statistics={}
    for name,a in (('qk',qk),('screening',screening)):
        def stat(v):
            if v.shape[-1]==0:return np.full(v.shape[:-1]+(4,),np.nan)
            return np.stack((v.mean(-1),np.mean(np.abs(v),axis=-1),np.sqrt(np.mean(v*v,axis=-1)),v.std(-1)),axis=-1)
        statistics[name]=stat(a)
        statistics[name+'_by_perimeter']=np.stack([stat(a[...,(perimeter>=lo)&(perimeter<hi)])
                    for lo,hi in zip(edges[:-1],edges[1:])],axis=-2)
    return dict(**geometry,qk=qk,screening=screening,reverse_qk=-qk,reverse_screening=-screening,
        regularized_log_circulation=logs,epsilon_values=np.asarray(epsilon_sweep),epsilon_circulations=values,
        perimeter_edges=edges,statistics=statistics,
        statistic_columns=('ccw_mean','mean_abs','rms','std'),
        interpretation='nonquantized network circulation; no automatic TR or Chern interpretation')
