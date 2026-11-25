"""Six-observable checkpoint analysis for an rs sweep; native/shared kept separate."""
import csv
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import attention
from screening_analysis.attn_probe import probe,save_probe
from screening_analysis.checkpoint_model import load_model
from screening_analysis.connection_audit import edge_scale,angle_difference,symmetric_edges
from screening_analysis.density_ph import density_curve,bounded_weights
from screening_analysis.topology_run import analyze_probe,graph_sources,json_safe


def write_csv(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def row_entropy(weights,include_self=False):
    """Actual nonnegative weights normalized per row; empty rows undefined."""
    w=np.asarray(weights,dtype=float).copy();n=w.shape[-1]
    if w.shape[-2]!=n or n<2 or not np.isfinite(w).all() or np.any(w < -1e-12):
        raise ValueError('finite nonnegative square weights required')
    w=np.maximum(w,0)
    if not include_self:w[...,np.arange(n),np.arange(n)]=0
    mass=w.sum(axis=-1);active=mass>0
    p=np.divide(w,mass[...,None],out=np.zeros_like(w),where=active[...,None])
    logs=np.zeros_like(p);np.log(p,out=logs,where=p>0)
    entropy=-np.sum(p*logs,axis=-1)
    entropy=np.where(active,entropy,np.nan)
    denominator=np.log(n if include_self else n-1)
    return dict(entropy=entropy,normalized=entropy/denominator if denominator>0 else np.where(active,0.,np.nan),
                effective_neighbors=np.exp(entropy),active=active,row_mass=mass)


def attention_modes(weights,spins):
    """Symmetric off-diagonal adjacency + normalized adjacency; directed spectrum too."""
    w=np.maximum(np.asarray(weights,dtype=float),0);n=w.shape[-1]
    b=(w+w.T)/2;b=b.copy();np.fill_diagonal(b,0)
    degree=b.sum(axis=-1);inv=np.divide(1,np.sqrt(degree),out=np.zeros(n),where=degree>0)
    normalized=inv[:,None]*b*inv[None,:]
    eigenvalues,vectors=np.linalg.eigh(b);order=np.arange(n-1,-1,-1)
    eigenvalues=eigenvalues[order];vectors=vectors[:,order]
    gaps=np.abs(np.diff(eigenvalues))
    separation=np.minimum(np.r_[np.inf,gaps],np.r_[gaps,np.inf])
    normalized_values,normalized_vectors=np.linalg.eigh(normalized)
    normalized_values=normalized_values[order];normalized_vectors=normalized_vectors[:,order]
    participation=1/np.sum(vectors**4,axis=0)
    s=np.array(spins,dtype=float,copy=True);s-=s.mean();norm=np.dot(s,s)
    overlap=(s@vectors)**2/norm if norm>0 else np.full(n,np.nan)
    directed=np.linalg.eigvals(w);directed=directed[np.argsort(-abs(directed),kind='stable')]
    return dict(eigenvalues=eigenvalues,eigenvectors=vectors,normalized_eigenvalues=normalized_values,
        normalized_eigenvectors=normalized_vectors,participation=participation,spin_overlap=overlap,
        directed_eigenvalues=directed,degree=degree,eigenvalue_gap=separation,
        mode_identifiable=separation>1e-8*max(1.,float(np.max(abs(eigenvalues)))))


def select_probe(p,count):
    ids=np.linspace(0,len(p['positions'])-1,min(count,len(p['positions']))).astype(int)
    size=len(p['positions']);keys={'S','W','W_window','W_intervened','Alpha','Window','d','positions','spins','atoms','charges'}
    out={k:(v[ids] if k in keys else v) for k,v in p.items()}
    out['metadata']={**p['metadata'],'parent_evaluation_input_sha256':p['metadata']['evaluation_input_sha256'],
        'analysis_subset_indices':ids.tolist(),
        'evaluation_input_sha256':hashlib.sha256(b''.join(np.ascontiguousarray(out[k]).tobytes() for k in ('positions','spins','atoms','charges'))).hexdigest()}
    return out


def summarize_probe(p,output,rs,ph_count,null_repeats):
    B,NL,NH,N,_=p['S'].shape;ij=np.triu_indices(N,1);d=p['d'][:,ij[0],ij[1]]
    same=p['spins'][:,ij[0]]==p['spins'][:,ij[1]]
    edges=np.linspace(0,float(p['dref'])*(1+1e-12),25)
    bins=np.clip(np.searchsorted(edges,d,side='right')-1,0,len(edges)-2)
    sources={s:bounded_weights(w) for s,w in graph_sources(p).items()}
    entropy_rows=[];mode_rows=[];radial_rows=[];mode_arrays={};entropy_arrays={};density_rows=[]
    for sample in range(B):
        for layer in range(NL):
            for head in range(NH):
                w=p['W'][sample,layer,head]
                mode=attention_modes(w,p['spins'][sample])
                for key,value in mode.items():mode_arrays[f'b{sample}_l{layer}_h{head}_{key}']=value
                for rank in range(N):
                    mode_rows.append(dict(sample=sample,layer=layer,head=head,rank=rank,
                        eigenvalue=float(mode['eigenvalues'][rank]),normalized_eigenvalue=float(mode['normalized_eigenvalues'][rank]),
                        participation=float(mode['participation'][rank]),spin_overlap=float(mode['spin_overlap'][rank]),
                        eigenvalue_gap=float(mode['eigenvalue_gap'][rank]),mode_identifiable=bool(mode['mode_identifiable'][rank]),
                        directed_eigenvalue_real=float(mode['directed_eigenvalues'][rank].real),
                        directed_eigenvalue_imag=float(mode['directed_eigenvalues'][rank].imag)))
                for self_edges in (False,True):
                    ent=row_entropy(w,self_edges)
                    entropy_arrays[f'b{sample}_l{layer}_h{head}_self{int(self_edges)}']=ent['entropy']
                    valid=ent['active'];average=lambda key:float(np.mean(ent[key][valid])) if valid.any() else None
                    entropy_rows.append(dict(sample=sample,layer=layer,head=head,include_self=self_edges,
                        entropy_nats=average('entropy'),entropy_normalized=average('normalized'),
                        effective_neighbors=average('effective_neighbors'),zero_row_fraction=float(1-valid.mean()),
                        offdiag_support=float(np.count_nonzero(symmetric_edges(w)>0)/len(ij[0]))))
                matrices={s:((w[sample,layer,head]+w[sample,layer,head].T)/2) for s,w in sources.items()}
                max_k=min(np.count_nonzero(w[ij]>0) for w in matrices.values())
                for s,matrix in matrices.items():
                    betti,ties,diagrams=density_curve(matrix,int(max_k))
                    for k in range(int(max_k)+1):
                        density_rows.append(dict(sample=sample,layer=layer,head=head,source=s,k=k,p=k/len(ij[0]),
                            beta0=int(betti[0,k]),beta1=int(betti[1,k]),tie_split=bool(ties[k])))
    for source,w in sources.items():
        values=symmetric_edges(w)
        for layer in range(NL):
            for head in range(NH):
                for db in range(len(edges)-1):
                    for spin in ('all','same','opposite'):
                        mask=bins==db
                        if spin!='all':mask &= same if spin=='same' else ~same
                        x=values[:,layer,head][mask]
                        radial_rows.append(dict(source=source,layer=layer,head=head,spin=spin,distance_bin=db,
                            distance_low=edges[db],distance_high=edges[db+1],distance_low_rs_aB=rs*edges[db],
                            distance_high_rs_aB=rs*edges[db+1],count=len(x),mean=float(x.mean()) if len(x) else None,
                            q10=float(np.quantile(x,.1)) if len(x) else None,q90=float(np.quantile(x,.9)) if len(x) else None))
    write_csv(output/'entropy.csv',entropy_rows);write_csv(output/'modes.csv',mode_rows)
    write_csv(output/'weight-distance.csv',radial_rows);write_csv(output/'density-betti.csv',density_rows)
    np.savez_compressed(output/'attention-modes.npz',**mode_arrays)
    np.savez_compressed(output/'row-entropies.npz',**entropy_arrays)
    ph_input=select_probe(p,ph_count)
    save_probe(ph_input,output/'ph-probe.npz')
    analyze_probe(ph_input,output/'ph',dict(id='checkpoint',phase='unknown',phase_verified=False),null_repeats=null_repeats)
    return dict(samples=B,layers=NL,heads=NH,particles=N,ph_samples=len(ph_input['positions']),
                ph_input_sha256=ph_input['metadata']['evaluation_input_sha256'])


def phase_sensitivity(net,params,batch,output,NL,NH):
    """Generic layer/head/pair log-weight response; phase API already returns angle."""
    N=batch['spins'].shape[-1];E=N*(N-1)//2;ij=np.triu_indices(N,1)
    def observe(z,p,pos,spins,atoms,charges):
        modified=jax.tree.map(lambda a:a,p)
        for layer in range(NL):
            modified['layers']['attention'][layer]['_edge_scale']=edge_scale(z[layer],N)
        phase,logabs=net.apply(modified,pos,spins,atoms,charges)
        return jnp.stack((logabs,phase))
    evaluate=jax.jit(observe);jacobian=jax.jit(jax.jacrev(observe,0));baseline=jax.jit(net.apply)
    zero=jnp.zeros((NL,NH,E),dtype=jnp.float64)
    gradients=[];valid=[];rows=[];checks=[]
    for sample in range(len(batch['positions'])):
        args=[jnp.asarray(batch[k][sample]) for k in ('positions','spins','atoms','charges')]
        original_phase,original_logabs=baseline(params,*args)
        base=np.asarray(evaluate(zero,params,*args))
        if not np.isfinite(base).all():
            gradients.append(np.full((2,NL,NH,E),np.nan));valid.append(False);continue
        grad=np.asarray(jacobian(zero,params,*args))
        if not np.isfinite(grad).all():
            gradients.append(np.full((2,NL,NH,E),np.nan));valid.append(False);continue
        identity=max(abs(float(original_logabs)-base[0]),abs(float(angle_difference(original_phase,base[1]))))
        if identity>1e-9:raise AssertionError('identity edge intervention differs from baseline')
        gradients.append(grad);valid.append(True)
        same=np.asarray(batch['spins'][sample])[ij[0]]==np.asarray(batch['spins'][sample])[ij[1]]
        for layer in range(NL):
            for head in range(NH):
                for group,mask in [('all',np.ones(E,bool)),('same',same),('opposite',~same)]:
                    direction=np.zeros_like(zero);direction[layer,head,mask]=1
                    response=np.asarray(evaluate(jnp.asarray(np.log(.9)*direction),params,*args))
                    rows.append(dict(sample=sample,layer=layer,head=head,group=group,edges=int(mask.sum()),
                        dlogabs=float(response[0]-base[0]),dphase=float(angle_difference(response[1],base[1])),
                        amplitude_derivative_sum=float(grad[0,layer,head,mask].sum()),
                        phase_derivative_sum=float(grad[1,layer,head,mask].sum()),
                        mean_abs_phase_derivative=float(abs(grad[1,layer,head,mask]).mean()) if mask.any() else None))
                if sample<2:
                    edge=int(np.argmax(np.linalg.norm(grad[:,layer,head],axis=0)))
                    direction=np.zeros_like(zero);direction[layer,head,edge]=1
                    for epsilon in (1e-3,5e-4):
                        plus=np.asarray(evaluate(jnp.asarray(epsilon*direction),params,*args))
                        minus=np.asarray(evaluate(jnp.asarray(-epsilon*direction),params,*args))
                        fd=np.array([(plus[0]-minus[0])/(2*epsilon),angle_difference(plus[1],minus[1])/(2*epsilon)])
                        checks.append(dict(sample=sample,layer=layer,head=head,edge=edge,epsilon=epsilon,
                            identity_error=identity,amplitude_error=float(abs(fd[0]-grad[0,layer,head,edge])),
                            phase_error=float(abs(fd[1]-grad[1,layer,head,edge])),
                            phase_scaled_error=float(abs(fd[1]-grad[1,layer,head,edge])/(1+abs(grad[1,layer,head,edge])))))
    np.savez_compressed(output/'connection-gradients.npz',gradients=np.asarray(gradients),valid=np.asarray(valid),edge_i=ij[0],edge_j=ij[1])
    write_csv(output/'phase-responses.csv',rows);write_csv(output/'finite-difference.csv',checks)
    return dict(samples=len(valid),valid_samples=sum(valid),identity_max=max((c['identity_error'] for c in checks),default=None),
        finite_difference_pass=bool(checks) and all(c['phase_scaled_error']<1e-4 for c in checks if c['epsilon']==5e-4),
        phase_fd_max_error=max((c['phase_error'] for c in checks),default=None),
        amplitude_fd_max_error=max((c['amplitude_error'] for c in checks),default=None),
        phase_fd_half_step_max_error=max((c['phase_error'] for c in checks if c['epsilon']==5e-4),default=None))


def analyze_checkpoint(folder,output,common_path,settings,case,checkpoint=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with load_model(folder,checkpoint=checkpoint) as (net,params,data,info):
        checkpoint=Path(info['checkpoint']);NL=net.options.num_layers;NH=net.options.num_heads
        if not info['config']['network'].get('complex',False):raise ValueError('complex angle-output network required')
        actual=info['config']['system']['make_local_energy_kwargs']
        if not np.isclose(actual['lam'],case['lam']) or not np.isclose(actual['potential_kwargs']['interaction_energy_scale'],case['rs']):
            raise ValueError('checkpoint lambda/rs differs from declared case')
        ids=np.linspace(0,len(data['positions'])-1,min(settings['native_samples'],len(data['positions']))).astype(int)
        native={k:data[k][ids] for k in ('positions','spins','atoms','charges')}
        with np.load(common_path,allow_pickle=False) as z:
            if not np.allclose(z['lattice'],info['lattice'],rtol=0,atol=1e-12):raise ValueError('shared input lattice differs')
            common={k:z[k][:settings['common_samples']] for k in native}
        if common['positions'].shape[1:]!=native['positions'].shape[1:] or common['spins'].shape[1:]!=native['spins'].shape[1:]:
            raise ValueError('common input particle shapes differ')
        summaries={};windows=[]
        for kind,batch in [('native',native),('common',common)]:
            dest=output/kind;dest.mkdir()
            p=probe(folder,checkpoint=checkpoint,config_path=info['config_path'],lattice=info['lattice'],
                    evaluation_data=batch,nconf=len(batch['positions']),collect_internal=False)
            save_probe(p,dest/'probe.npz')
            summaries[kind]=summarize_probe(p,dest,case['rs'],settings['ph_samples'],settings['null_repeats'])
            sensitivity_batch={k:v[np.linspace(0,len(v)-1,min(settings['phase_samples'],len(v))).astype(int)] for k,v in batch.items()}
            np.savez_compressed(dest/'phase-inputs.npz',**sensitivity_batch)
            summaries[kind]['phase']=phase_sensitivity(net,params,sensitivity_batch,dest,NL,NH)
            summaries[kind]['evaluation_input_sha256']=p['metadata']['evaluation_input_sha256']
            research=settings.get('research',{})
            if research.get('enabled',False):
                from screening_analysis.ph_research import run_subset
                summaries[kind]['research']=run_subset(dest,dest/'research',{k:v for k,v in research.items() if k!='enabled'})
            print(case['id'],kind,'complete',flush=True)
        radius=np.linspace(0,float(p['dref']),401)
        for layer,ap in enumerate(params['layers']['attention']):
            if 'mask_centre' in ap or 'mask_rings' in ap:raise ValueError('sweep window-width interpretation requires monotone windows')
            shape=info['settings']['WIN_SHAPE']
            scales={k:np.asarray(v) for k,v in attention.window_scales(ap,p['dref'],shape,
                freeze_eta=info['settings']['FREEZE_ETA']).items()}
            w=scales['width']
            profile=np.asarray(attention.screening_window(ap,jnp.asarray(radius[None,:]),jnp.asarray(p['dref']),
                win_shape=shape,freeze_w=False,algebraic_power=info['settings']['WIN_POWER'],
                freeze_eta=info['settings']['FREEZE_ETA']))[:,0,:]
            for head,width in enumerate(w):
                for i,r in enumerate(radius):
                    windows.append(dict(layer=layer,head=head,distance=r,distance_rs_aB=r*case['rs'],weight=profile[head,i],
                        width=width,width_rs_aB=width*case['rs'],half_weight_radius=scales['half_weight_radius'][head],
                        half_decay_radius=scales['half_decay_radius'][head],tail_floor=scales['tail_floor'][head],
                        cutoff_radius=scales['cutoff_radius'][head],eta=scales['eta'][head],window_shape=shape,
                        algebraic_power=info['settings']['WIN_POWER'] if shape=='algebraic' else None))
        write_csv(output/'window-curves.csv',windows)
        summary=dict(case=case,checkpoint=str(checkpoint),checkpoint_sha256=info['checkpoint_sha256'],
            config_sha256=hashlib.sha256(Path(info['config_path']).read_bytes()).hexdigest(),
            common_bank_sha256=hashlib.sha256(common_path.read_bytes()).hexdigest(),subsets=summaries,
            entropy='Natural log, row-normalized actual directed aggregation weights; primary excludes self, zero rows undefined.',
            modes='Off-diagonal symmetrized actual weights, raw/degree-normalized real modes; directed full spectrum also saved. No eigenvector averaging across particle permutations.',
            phase='Pair log-weight derivatives and 10% attenuation responses at fixed inputs; angle output in radians, not a Chern number.',
            phase_status='unknown; lambda/initialization are not trained-state evidence',
            limitations='Saved walkers and common fixed inputs; no independent MCMC/ESS or physical phase certificate.',
            source_sha256={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                           (__file__,attention.__file__)})
        hf=info['config']['network']['CustomScreenformer'].get('lj_hf_path')
        if hf:summary['hf_sha256']=hashlib.sha256(Path(hf).read_bytes()).hexdigest()
        (output/'summary.json').write_text(json.dumps(json_safe(summary),indent=2,ensure_ascii=False,allow_nan=False))
    (output/'finished.json').write_text(json.dumps(dict(complete=True)))
    return summary
