"""Distance/selection/head/wavefunction audit of final checkpoints, without PH.

python -m screening_analysis.connection_audit ROOT --output NEW_DIRECTORY
All saved final walkers from each run are pooled and used by both models.
These are fixed-input diagnostics, not independent equilibrium estimates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr
from periodicwave import attention
from screening_analysis.attn_probe import probe, save_probe
from screening_analysis.checkpoint_model import load_model
from screening_analysis.graph_wind import minimum_image_edges


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def correlation(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if len(x)<3 or x.std()<1e-14 or y.std()<1e-14:
        return None
    return float(np.corrcoef(x,y)[0,1])


def residual_by_bin(values,labels):
    out=np.asarray(values,dtype=float).copy()
    for label in np.unique(labels):
        select=labels==label;out[select]-=out[select].mean()
    return out


def edge_sets(window,screening,k):
    """Statuses: 0 neither, 1 promoted, 2 demoted, 3 retained. Positive only."""
    window=np.asarray(window);screening=np.asarray(screening)
    if min(np.count_nonzero(window>0),np.count_nonzero(screening>0))<k:
        return None
    # Explicitly reject ambiguous top-k boundaries rather than inventing edges.
    sets=[]
    for values in (window,screening):
        order=np.argsort(-values,kind='stable')
        if k<len(values) and values[order[k-1]]==values[order[k]]:
            return None
        mask=np.zeros(len(values),bool);mask[order[:k]]=True;sets.append(mask)
    return sets[0].astype(int)*2+sets[1].astype(int)


def symmetric_edges(matrix):
    matrix=np.asarray(matrix)
    n=matrix.shape[-1];ij=np.triu_indices(n,1)
    return ((matrix+matrix.swapaxes(-1,-2))/2)[...,ij[0],ij[1]]


def edge_scale(logscale,n):
    """One parameter per head and undirected pair; multiply both directions."""
    ij=np.triu_indices(n,1)
    out=jnp.zeros((logscale.shape[0],n,n),dtype=logscale.dtype)
    out=out.at[:,ij[0],ij[1]].set(logscale)
    out=out.at[:,ij[1],ij[0]].set(logscale)
    return jnp.exp(out)


def angle_difference(a,b):
    return np.angle(np.exp(1j*(np.asarray(a)-np.asarray(b))))


def complex_observables(phase_angle,logabs):
    """Repository complex Network.apply returns a REAL phase angle, not exp(i phi)."""
    return jnp.stack((logabs,phase_angle))


def selection_statistics(values,distances,angles,ambiguous,spins,origins,model,output):
    n=len(spins[0]);ij=np.triu_indices(n,1)
    same=spins[:,ij[0]]==spins[:,ij[1]]
    d=distances
    bins=np.linspace(0,float(d.max())*(1+1e-12),21)
    radial=np.clip(np.searchsorted(bins,d,side='right')-1,0,19)
    angle_bin=np.clip((angles/np.pi*12).astype(int),0,11)
    curve_rows=[];edge_rows=[];selection_rows=[];head_rows=[];summary_rows=[]
    masks={}
    for sample in range(len(d)):
        origin=origins[sample]
        for source,w in values.items():
            h0,h1=w[sample,0],w[sample,1]
            x=residual_by_bin(h0,radial[sample]);y=residual_by_bin(h1,radial[sample])
            head_rows.append(dict(model=model,input_origin=origin,sample=sample,source=source,
                                  pearson=correlation(h0,h1),spearman=float(spearmanr(h0,h1).statistic),
                                  distance_residual_pearson=correlation(x,y)))
        for head in range(2):
            for k in (30,60,90):
                status=edge_sets(values['window'][sample,head],values['screening'][sample,head],k)
                if status is None:
                    continue
                masks[sample,head,k]=status
                summary_rows.append(dict(model=model,input_origin=origin,sample=sample,head=head,k=k,
                    promoted=int((status==1).sum()),demoted=int((status==2).sum()),retained=int((status==3).sum()),
                    jaccard=float((status==3).sum()/np.count_nonzero(status))))
                for edge in np.flatnonzero(status):
                    edge_rows.append(dict(model=model,input_origin=origin,sample=sample,head=head,k=k,
                        i=int(ij[0][edge]),j=int(ij[1][edge]),status=int(status[edge]),
                        periodic_distance=float(d[sample,edge]),angle_mod_pi=float(angles[sample,edge]),
                        angle_ambiguous=bool(ambiguous[sample,edge]),spin_i=float(spins[sample,ij[0][edge]]),
                        spin_j=float(spins[sample,ij[1][edge]]),same_spin=bool(same[sample,edge]),
                        window=float(values['window'][sample,head,edge]),alpha=float(values['alpha'][sample,head,edge]),
                        screening=float(values['screening'][sample,head,edge])))
                # Denominators retained: promotions out of unselected positive-Window candidates;
                # demotions out of Window top-k. Distances AND spin are stratified jointly.
                for db in np.unique(radial[sample]):
                    for spin in (False,True):
                        for ab in range(12):
                            select=(radial[sample]==db)&(same[sample]==spin)&(angle_bin[sample]==ab)&(~ambiguous[sample])
                            potential=select&(values['window'][sample,head]>0)
                            if not potential.any():
                                continue
                            selection_rows.append(dict(model=model,input_origin=origin,sample=sample,head=head,k=k,
                                distance_bin=int(db),distance_low=float(bins[db]),distance_high=float(bins[db+1]),
                                same_spin=bool(spin),angle_bin=ab,angle_low=ab*np.pi/12,angle_high=(ab+1)*np.pi/12,
                                window_candidates=int(potential.sum()),
                                promotion_opportunities=int((potential&((status==0)|(status==1))).sum()),
                                promoted=int((potential&(status==1)).sum()),
                                demotion_opportunities=int((potential&((status==2)|(status==3))).sum()),
                                demoted=int((potential&(status==2)).sum())))
    for subset in ['shared_all',*sorted(set(origins))]:
        selected=np.ones(len(d),bool) if subset=='shared_all' else np.asarray(origins)==subset
        for source,w in values.items():
            for head in range(2):
                for db in range(20):
                    for spin in ('all','same','opposite'):
                        select=(radial==db)&selected[:,None]
                        if spin!='all':select&=same if spin=='same' else ~same
                        a=w[:,head][select]
                        curve_rows.append(dict(model=model,input_subset=subset,head=head,source=source,distance_bin=db,
                            distance_low=float(bins[db]),distance_high=float(bins[db+1]),spin=spin,count=len(a),
                            mean=float(a.mean()) if len(a) else None,
                            q10=float(np.quantile(a,.1)) if len(a) else None,q90=float(np.quantile(a,.9)) if len(a) else None))
    write_csv(output/'weight-distance.csv',curve_rows)
    write_csv(output/'edge-replacements.csv',edge_rows)
    write_csv(output/'selection-strata.csv',selection_rows)
    write_csv(output/'replacement-counts.csv',summary_rows)
    write_csv(output/'head-correlations.csv',head_rows)
    return masks


def wavefunction_sensitivity(folder,checkpoint,lattice,batch,values,masks,origins,out):
    n=batch['spins'].shape[-1];ij=np.triu_indices(n,1);nh=2
    with load_model(folder,checkpoint=checkpoint,lattice=lattice) as (net,params,_,info):
        if len(params['layers']['attention'])!=1:
            raise ValueError('this audit explicitly targets one-layer two-head checkpoints')
        if not info['config']['network'].get('complex',False):
            raise ValueError('phase-angle sensitivity requires the complex-output API')
        def observables(logscale,parameters,pos,spins,atoms,charges):
            # Pass the same parameters dynamically to baseline and intervention:
            # closing over mixed-precision parameters changes XLA constant folding.
            modified=jax.tree.map(lambda x:x,parameters)
            modified['layers']['attention'][0]['_edge_scale']=edge_scale(logscale,n)
            phase,logabs=net.apply(modified,pos,spins,atoms,charges)
            return complex_observables(phase,logabs)
        evaluate=jax.jit(observables)
        derivative=jax.jit(jax.jacrev(observables,argnums=0))
        baseline=jax.jit(net.apply)
        zeros=jnp.zeros((nh,len(ij[0])),dtype=jnp.float64)
        gradients=[];baselines=[];checks=[];response_rows=[];head_vectors=[]
        for sample in range(len(batch['positions'])):
            args=[jnp.asarray(batch[key][sample]) for key in ('positions','spins','atoms','charges')]
            phase,logabs=baseline(params,*args)
            original=np.array([float(logabs),float(phase)])
            identity=np.asarray(evaluate(zeros,params,*args))
            grad=np.asarray(derivative(zeros,params,*args))
            if not np.isfinite(original).all() or not np.isfinite(grad).all():
                raise ValueError(f'undefined wavefunction/gradient at sample {sample}')
            gradients.append(grad);baselines.append(original)
            if sample in (0,len(batch['positions'])//4,len(batch['positions'])//2,3*len(batch['positions'])//4):
                for head in range(nh):
                    edge=int(np.argmax(np.linalg.norm(grad[:,head],axis=0)))
                    direction=np.zeros_like(zeros);direction[head,edge]=1
                    for eps in (1e-3,5e-4):
                        plus=np.asarray(evaluate(jnp.asarray(eps*direction),params,*args))
                        minus=np.asarray(evaluate(jnp.asarray(-eps*direction),params,*args))
                        fd=np.array([(plus[0]-minus[0])/(2*eps),angle_difference(plus[1],minus[1])/(2*eps)])
                        checks.append(dict(sample=sample,head=head,i=int(ij[0][edge]),j=int(ij[1][edge]),epsilon=eps,
                            amplitude_analytic=float(grad[0,head,edge]),amplitude_fd=float(fd[0]),
                            phase_analytic=float(grad[1,head,edge]),phase_fd=float(fd[1]),
                            identity_amplitude_error=float(abs(identity[0]-original[0])),
                            identity_phase_error=float(abs(angle_difference(identity[1],original[1])))))
            for head in range(nh):
                groups=[('all_offdiagonal',0,np.ones(len(ij[0]),bool))]
                for k in (60,90):
                    status=masks.get((sample,head,k))
                    if status is not None:
                        groups.extend((name,k,status==code) for name,code in [('promoted',1),('demoted',2),('retained',3)])
                for name,k,select in groups:
                    if not select.any():continue
                    direction=np.zeros((nh,len(ij[0])));direction[head,select]=1
                    # 10% decrease of actual directed edge weights; same coordinates, no resampling.
                    response=np.asarray(evaluate(jnp.asarray(np.log(.9)*direction),params,*args))
                    response_rows.append(dict(sample=sample,input_origin=origins[sample],head=head,k=k,group=name,
                        edges=int(select.sum()),screening_mass=float(values['screening'][sample,head,select].sum()),
                        dlogabs=float(response[0]-original[0]),dphase=float(angle_difference(response[1],original[1])),
                        linear_dlogabs=float(np.log(.9)*grad[0,head,select].sum()),
                        linear_dphase=float(np.log(.9)*grad[1,head,select].sum()),
                        mean_abs_amplitude_derivative=float(np.mean(abs(grad[0,head,select]))),
                        mean_abs_phase_derivative=float(np.mean(abs(grad[1,head,select])))))
            if (sample+1)%32==0:print(folder.parent.name,'sensitivity',sample+1,flush=True)
        np.savez_compressed(out/'connection-sensitivity.npz',derivatives=np.asarray(gradients),
                            baseline=np.asarray(baselines),edge_i=ij[0],edge_j=ij[1],origins=np.asarray(origins))
        write_csv(out/'finite-difference-checks.csv',checks)
        write_csv(out/'wavefunction-responses.csv',response_rows)
        validation=dict(max_identity_amplitude_error=max(r['identity_amplitude_error'] for r in checks),
            max_identity_phase_error=max(r['identity_phase_error'] for r in checks),
            max_amplitude_fd_error=max(abs(r['amplitude_analytic']-r['amplitude_fd']) for r in checks),
            max_phase_fd_error=max(abs(r['phase_analytic']-r['phase_fd']) for r in checks))
        if max(validation['max_identity_amplitude_error'],validation['max_identity_phase_error'])>1e-9:
            raise AssertionError('identity intervention does not reconstruct baseline')
        hf=Path(info['config']['network']['CustomScreenformer']['lj_hf_path'])
        return dict(validation=validation,checkpoint_sha256=info['checkpoint_sha256'],
                    hf_path=str(hf),hf_sha256=hashlib.sha256(hf.read_bytes()).hexdigest(),
                    intervention='W_hij and W_hji multiplied by exp(epsilon), self edges unchanged; derivative w.r.t. epsilon at 0.',
                    finite_intervention='Selected edges reduced by 10%; fixed inputs, not reoptimized or resampled.',
                    phase='Radians; raw phase response includes possible global phase. Compare configuration-dependent centered responses before physical interpretation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    runs=sorted(p for p in args.root.glob('lam*/training') if (p/'qmcjax_ckpt_079999.npz').exists())
    if len(runs)!=2:raise ValueError('exactly two final checkpoints required')
    pools=[];origins=[];lattices=[]
    for folder in runs:
        with np.load(folder/'qmcjax_ckpt_079999.npz',allow_pickle=True) as z:
            data=z['data'].item()
            pool={k:np.asarray(data[k]).reshape((-1,)+np.asarray(data[k]).shape[2:]) for k in ('positions','spins','atoms','charges')}
        pools.append(pool);origins.extend([folder.parent.name]*len(pool['positions']))
        cfg=json.loads((folder/'runtime_config.json').read_text());cfg=cfg.get('config',cfg)
        lattices.append(np.asarray(cfg['system']['pbc_lattice']))
    if not np.array_equal(lattices[0],lattices[1]):raise ValueError('shared coordinates require equal lattices')
    lattice=lattices[0];batch={k:np.concatenate([p[k] for p in pools]) for k in pools[0]}
    np.savez_compressed(args.output/'shared-inputs.npz',**batch,origins=np.asarray(origins),lattice=lattice)
    n=batch['spins'].shape[-1];ij=np.triu_indices(n,1)
    angles=[];ambiguous=[];euclidean=[]
    for positions in batch['positions']:
        delta,d,ties=minimum_image_edges(positions.reshape(n,2),lattice)
        vector=delta@lattice.T
        angles.append(np.mod(np.arctan2(vector[:,1],vector[:,0]),np.pi))
        ambiguous.append(ties|(d<1e-12));euclidean.append(d)
    angles=np.asarray(angles);ambiguous=np.asarray(ambiguous)
    all_summaries=[];width_rows=[]
    for folder in runs:
        name=folder.parent.name;out=args.output/name;out.mkdir()
        checkpoint=folder/'qmcjax_ckpt_079999.npz'
        result=probe(folder,checkpoint=checkpoint,config_path=folder/'runtime_config.json',lattice=lattice,
                     nconf=len(origins),evaluation_data=batch,collect_internal=True)
        save_probe(result,out/'shared-probe.npz')
        values={name:symmetric_edges(result[key][:,0]) for name,key in
                [('qk','S'),('alpha','Alpha'),('window','Window'),('screening','W_window')]}
        values['qk']=(values['qk']+1)/2
        if min(np.min(w) for w in values.values()) < -1e-12:raise ValueError('negative weights beyond roundoff')
        values={s:np.clip(w,0,1) for s,w in values.items()}
        distance=result['d'][:,ij[0],ij[1]]
        np.savez_compressed(out/'edge-data.npz',**values,periodic_distance=distance,euclidean_distance=np.asarray(euclidean),
                            angle_mod_pi=angles,angle_ambiguous=ambiguous,edge_i=ij[0],edge_j=ij[1],
                            spins=batch['spins'],origins=np.asarray(origins),head_vectors=result['head_vectors'])
        masks=selection_statistics(values,distance,angles,ambiguous,batch['spins'],origins,name,out)
        # Inspect all existing saved widths, without reconstructing an absent theta_0.
        for path in sorted(folder.glob('qmcjax_ckpt_*.npz')):
            with np.load(path,allow_pickle=True) as z:params=z['params'].item();step=int(z['t'])
            for layer,ap in enumerate(params['layers']['attention']):
                widths=np.logaddexp(0,np.asarray(ap['mask_omega'])[0])*float(result['dref'])
                trim=2/(1+np.exp(-np.asarray(ap['trim_rho'])[0]))
                for head,width in enumerate(widths):
                    width_rows.append(dict(model=name,step=step,layer=layer,head=head,width=float(width),
                        width_over_dref=float(width/result['dref']),half_weight_radius=float(width/2),trim_r=float(trim[head])))
        # Exact radial Window curves use the production function, not a fitted exponential.
        with load_model(folder,checkpoint=checkpoint,lattice=lattice) as (_,params,_,info):
            radius=np.linspace(0,float(result['dref']),501)
            window=np.asarray(attention.screening_window(params['layers']['attention'][0],jnp.asarray(radius[None,:]),
                jnp.asarray(result['dref']),win_shape=info['settings']['WIN_SHAPE'],freeze_w=False,
                algebraic_power=info['settings']['WIN_POWER'],freeze_eta=info['settings']['FREEZE_ETA']))[:,0,:]
            write_csv(out/'exact-window-curves.csv',[dict(head=h,distance=float(r),weight=float(window[h,i]))
                      for h in range(2) for i,r in enumerate(radius)])
        summary=wavefunction_sensitivity(folder,checkpoint,lattice,batch,values,masks,origins,out)
        vectors=np.asarray(result['head_vectors'])[:,0] # [sample,particle,head,output]
        a,b=vectors[:,:,0],vectors[:,:,1]
        dot=np.sum(a*b,axis=(1,2));norm=np.sqrt(np.sum(a*a,axis=(1,2))*np.sum(b*b,axis=(1,2)))
        write_csv(out/'head-output-overlap.csv',[dict(sample=i,input_origin=origins[i],
            cosine=float(dot[i]/norm[i]) if norm[i]>0 else None,
            cancellation_ratio=float(np.sum((a[i]+b[i])**2)/(np.sum(a[i]**2)+np.sum(b[i]**2)))) for i in range(len(origins))])
        summary.update(model=name,samples=len(origins),native_samples=origins.count(name),
                       ambiguous_angle_edges=int(ambiguous.sum()),shared_input_sha256=hashlib.sha256((args.output/'shared-inputs.npz').read_bytes()).hexdigest())
        (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
        all_summaries.append(summary)
    write_csv(args.output/'window-width-history.csv',width_rows)
    (args.output/'manifest.json').write_text(json.dumps(dict(models=all_summaries,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (Path(__file__),Path(attention.__file__),Path(__file__).parent/'checkpoint_model.py')},
        limitations='Two differently parameterized Hamiltonians/HF states, one training seed each. Fixed pooled saved walkers, not independent equilibrium samples. No PH computed.'),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
