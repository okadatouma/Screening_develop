"""Reuse existing graph reducer and phase-lift Ripser, with explicit scales."""
import numpy as np
from ..experiments.store import json_save
from ..paths import wannier_source


def graph_ph(folder,label,arrays):
    from screening_analysis.topology.persistence import persistence
    from ripser import ripser
    output={};rows=[]
    for name,w in arrays.items():
        w=np.asarray(w,float)
        if w.ndim==3:w=w.mean(0)
        if w.shape[0]!=w.shape[1]:raise ValueError('square graph required')
        if name=='QK':w=(np.clip(w,-1,1)+1)/2
        # Alpha/Window/Screening are already [0,1]. Other Gram graphs are explicitly supplied as bounded values.
        w=np.clip((w+w.T)/2,0,1);np.fill_diagonal(w,0)
        if len(w)<=64:
            diagrams=persistence(w)
        else:
            # Preserve exactly-zero absent edges. Full threshold1 only includes positive weights.
            d=1-w;d[w==0]=np.inf;np.fill_diagonal(d,0)
            diagrams=ripser(d,distance_matrix=True,maxdim=1,thresh=1.)['dgms']
        for dim,diag in enumerate(diagrams):
            output[f'{name}_H{dim}']=diag
            life=diag[np.isfinite(diag[:,1]),1]-diag[np.isfinite(diag[:,1]),0]
            rows.append(dict(graph=name,dim=dim,intervals=len(diag),essential=int(np.isinf(diag[:,1]).sum()),
                             longest=float(life.max(initial=0)),total_persistence=float(life.sum())))
    with (folder/f'{label}_ph.npz').open('xb') as f:np.savez_compressed(f,**output)
    json_save(folder/f'{label}_ph.json',dict(filtration='weighted flag u=1-(W+W.T)/2; zero weights absent',coeff=2,rows=rows))
    return rows


def phase_lift_ph(folder,label,k,u,phases=8):
    import subprocess
    import sys
    wannier_source()
    from wannier_ph.geometry import phase_lift,euclidean
    from wannier_ph.persistence import plateau
    points=phase_lift(k,u,phases)
    if len(points)>1600:raise ValueError('PH guard: select <=1600 phase-lift points')
    source=folder/f'{label}_lift_points.npz';target=folder/f'{label}_lift_ph.npz'
    with source.open('xb') as f:np.savez_compressed(f,points=points)
    try:
        result=subprocess.run([sys.executable,'-m','research_framework.diagnostics.ph_worker',str(source),str(target)],capture_output=True,text=True,timeout=30)
    except subprocess.TimeoutExpired:return dict(status='timeout',vertices=len(points),limit_seconds=30)
    if result.returncode==0:
        stored=np.load(target);diagrams=[stored[f'H{i}'] for i in [0,1]]
        diameter=float(stored['diameter'])
        return dict(status='ok',vertices=len(points),coeff=2,filtration='Euclidean VR, diameter epsilon',
                    width_b1_2=plateau(diagrams,2,diameter)['plateau_width'],
                    width_b1_3=plateau(diagrams,3,diameter)['plateau_width'])
    return dict(status='error',error=result.stderr[-2000:])
