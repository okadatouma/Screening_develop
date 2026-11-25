"""A: connection distances; B: wavefunction response; C: variational energy.

python -m screening_analysis.connection_study --plan PLAN --mode diagnostic
python -m screening_analysis.connection_study --folder TRAINING --mode energy --output NEW_DIR
--mode scan also samples each changed Window; --mode retrain fixes all Window
widths and optimizes the remaining parameters before independent evaluation.
No scheduler jobs are submitted. See this file's five main functions below.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def analyze_connections(p, bins, rs, gradients=None):
    """Pool pairs, not per-configuration means. Keep both distance definitions."""
    from screening_analysis.connection_audit import symmetric_edges
    from screening_analysis.graph_wind import minimum_image_edges
    n=int(p['NE']);ij=np.triu_indices(n,1);rows=[];summary=[]
    # A：接続の距離分布。ネットワークの距離と物理距離は別に集計する。
    distances={'network':p['d'][:,ij[0],ij[1]],
               'minimum_image':np.array([minimum_image_edges(x.reshape(n,2),p['lattice'])[1] for x in p['positions']])}
    # Geometric reference, not evidence that the sampled state is a WC.
    a_wc=np.sqrt(2*abs(np.linalg.det(p['lattice']))/(np.sqrt(3)*n))
    same=p['spins'][:,ij[0]]==p['spins'][:,ij[1]]
    sources={'screening':symmetric_edges(p['W']),
             'alpha':symmetric_edges(p['Alpha']),'window':symmetric_edges(p['Window'])}
    for distance_kind,d in distances.items():
        edges=np.linspace(0,bins['network_max'] if distance_kind=='network' else bins['physical_max'],bins['count']+1)
        labels=np.minimum(np.searchsorted(edges,d,side='right')-1,len(edges)-2)
        for layer in range(int(p['NL'])):
            for head in range(int(p['NH'])):
                for spin in ('all','same','opposite'):
                    spin_mask=np.ones_like(same) if spin=='all' else same if spin=='same' else ~same
                    for source,values in sources.items():
                        w=values[:,layer,head];mass=w[spin_mask].sum()
                        order=np.argsort(d[spin_mask]);ordered_d=d[spin_mask][order];ordered_w=w[spin_mask][order]
                        r90=ordered_d[np.searchsorted(np.cumsum(ordered_w),.9*mass)] if mass>0 else np.nan
                        effective=np.sum(d[spin_mask]*w[spin_mask])/mass if mass>0 else np.nan
                        summary.append(dict(distance_kind=distance_kind,layer=layer,head=head,spin=spin,source=source,
                            r_eff=effective,r90=r90,r_eff_aB=rs*effective,r90_aB=rs*r90,
                            r_eff_over_a_wc=effective/a_wc if distance_kind=='minimum_image' else np.nan))
                        for b in range(len(edges)-1):
                            mask=(labels==b)&spin_mask;x=w[mask]
                            row=dict(distance_kind=distance_kind,layer=layer,head=head,spin=spin,source=source,bin=b,
                                distance_low=edges[b],distance_high=edges[b+1],distance_mid=(edges[b]+edges[b+1])/2,
                                distance_mid_aB=rs*(edges[b]+edges[b+1])/2,
                                distance_mid_over_a_wc=(edges[b]+edges[b+1])/2/a_wc if distance_kind=='minimum_image' else np.nan,
                                count=int(mask.sum()),mean=x.mean() if len(x) else np.nan,
                                total_per_configuration=x.sum()/len(d),pair_abs_amplitude=np.nan,pair_abs_phase=np.nan)
                            if gradients is not None:
                                g=gradients[:,:,layer,head]
                                row.update(pair_abs_amplitude=abs(g[:,0][mask]).mean() if mask.any() else np.nan,
                                           pair_abs_phase=abs(g[:,1][mask]).mean() if mask.any() else np.nan)
                            rows.append(row)
    return rows,summary


def analyze_response(net,params,batch,centers,widths,output,local=None,block_count=None):
    """Distance-band derivative; metric/energy only for fresh MCMC samples.

    Samples are ordered [time block, time within block, chain]. Block estimates
    include centering within the block. Errors are provisional blocking SEMs,
    not a certificate of equilibration or independence.
    """
    import jax
    import jax.numpy as jnp
    from periodicwave import networks
    nl=len(params['layers']['attention']);nh=len(params['layers']['attention'][0]['mask_omega'])
    zero=jnp.zeros((nl,nh,len(centers)));rows=[];gradients=[];energies=[];checks=[]

    def observe(epsilon,p,x,s,a,c):
        changed=jax.tree.map(lambda v:v,p)
        for layer,ap in enumerate(changed['layers']['attention']):
            ap.update(_radial_epsilon=epsilon[layer],_radial_centers=jnp.asarray(centers),_radial_widths=jnp.asarray(widths))
        phase,logabs=net.apply(changed,x,s,a,c)
        return jnp.stack((logabs,phase))

    evaluate=jax.jit(observe);derivative=jax.jit(jax.jacrev(observe,0));baseline=jax.jit(net.apply)
    energy=jax.jit(local) if local is not None else None
    for i in range(len(batch['positions'])):
        args=[jnp.asarray(batch[k][i]) for k in ('positions','spins','atoms','charges')]
        original_phase,original_abs=baseline(params,*args);base=np.asarray(evaluate(zero,params,*args))
        error=max(abs(base[0]-float(original_abs)),abs(np.angle(np.exp(1j*(base[1]-float(original_phase))))))
        if not np.isfinite(base).all() or error>1e-9:raise ValueError('radial identity/nonfinite baseline failure')
        g=np.asarray(derivative(zero,params,*args))
        if not np.isfinite(g).all():raise ValueError('nonfinite response: no samples silently discarded')
        gradients.append(g)
        if i==0:
            direction=np.zeros(zero.shape);direction.flat[np.argmax(np.linalg.norm(g,axis=0))]=1
            for step in (1e-3,5e-4):
                plus=np.asarray(evaluate(step*direction,params,*args));minus=np.asarray(evaluate(-step*direction,params,*args))
                fd=np.array([(plus[0]-minus[0])/(2*step),np.angle(np.exp(1j*(plus[1]-minus[1])))/(2*step)])
                actual=np.sum(g*direction,axis=(1,2,3));scaled=np.max(abs(fd-actual)/(1+abs(actual)))
                checks.append(dict(step=step,scaled_error=scaled,identity_error=error))
            if checks[-1]['scaled_error']>1e-4:raise ValueError('radial derivative finite-difference check failed')
        if energy is not None:
            value=np.asarray(energy(params,jax.random.PRNGKey(i),networks.FermiNetData(**dict(zip(('positions','spins','atoms','charges'),args))))[0]).item()
            if not np.isfinite(value):raise ValueError('nonfinite local energy')
            energies.append(value)
    # B：振幅と位相をまとめた複素感度。平均を引くと共通の変化を除ける。
    g=np.asarray(gradients);O=g[:,0]+1j*g[:,1];el=np.asarray(energies)
    physical=local is not None
    if physical:
        if block_count is None or block_count<4 or len(O)%block_count:raise ValueError('at least four equal MCMC blocks required')
        ob=O.reshape((block_count,-1)+O.shape[1:]);eb=el.reshape(block_count,-1)
        centered=ob-ob.mean(axis=1,keepdims=True)
        metrics=(abs(centered)**2).mean(axis=1)
        energy_blocks=2*(centered.conj()*(eb-eb.mean(axis=1,keepdims=True))[:,:,None,None,None]).mean(axis=1).real
        # C：感度と局所エネルギーの共分散。複素数の共役と実部が必要。
        energy_gradient=2*((O-O.mean(axis=0)).conj()*(el-el.mean())[:,None,None,None]).mean(axis=0).real
    for layer in range(nl):
        for head in range(nh):
            for b,r in enumerate(centers):
                value=O[:,layer,head,b];row=dict(layer=layer,head=head,bin=b,distance_mid=r,
                    mean_abs_amplitude=abs(value.real).mean(),mean_abs_phase=abs(value.imag).mean(),
                    mean_abs_complex=abs(value).mean(),empirical_centered_variance=np.mean(abs(value-value.mean())**2),
                    state_metric=np.nan,metric_block_sem=np.nan,energy_gradient=np.nan,energy_gradient_block_sem=np.nan)
                if physical:
                    row.update(state_metric=np.mean(abs(value-value.mean())**2),
                        metric_block_sem=metrics[:,layer,head,b].std(ddof=1)/np.sqrt(block_count),
                        energy_gradient=energy_gradient[layer,head,b],
                        energy_gradient_block_sem=energy_blocks[:,layer,head,b].std(ddof=1)/np.sqrt(block_count))
                rows.append(row)
    np.savez_compressed(output,gradients=g,local_energy=el,centers=centers,widths=widths,
                        finite_difference=np.asarray([[c['step'],c['scaled_error']] for c in checks]),**batch)
    return rows,el


def sample_state(net,params,data,info,args,seed,local=None,training=False):
    """Existing position+pseudospin MCMC, optional fixed-width VMC updates."""
    import jax
    import jax.numpy as jnp
    from periodicwave import mcmc,networks
    count=min(args.chains,len(data['positions']));rng=jax.random.PRNGKey(seed)
    state=networks.FermiNetData(**{k:jnp.asarray(v[:count]) for k,v in data.items()})
    batch_logabs=jax.vmap(lambda p,x,s,a,c:net.apply(p,x,s,a,c)[1],in_axes=(None,0,0,0,0))
    flips=args.spin_flips if info['config']['system'].get('pseudospin',False) else 0
    step=jax.jit(mcmc.make_mcmc_step(batch_logabs,count,steps=args.steps,ndim=2,blocks=1,spin_flips=flips))
    optimizer_state=None;history=[]
    if training:
        import optax
        from periodicwave import loss
        def logpsi(p,x,s,a,c):
            phase,logabs=net.apply(p,x,s,a,c);return logabs+1j*phase
        objective=loss.make_loss(logpsi,local,complex_output=True,clip_local_energy=0.,local_energy_chunk=1)
        gradient=jax.jit(jax.value_and_grad(objective,has_aux=True))
        optimizer=optax.adam(args.learning_rate);optimizer_state=optimizer.init(params)
    stored={k:[] for k in data};accept=[]
    iterations=args.retrain_steps if training else args.blocks*args.block_size
    for t in range(args.burn+iterations):
        rng,move,key=jax.random.split(rng,3);state,rate=step(params,state,move,args.proposal)
        if t<args.burn:continue
        if training:
            (value,aux),grad=gradient(params,key,state)
            # All widths stay fixed, including heads/layers not being scanned.
            for ap in grad['layers']['attention']:
                ap['mask_omega']=jnp.zeros_like(ap['mask_omega'])
            updates,optimizer_state=optimizer.update(grad,optimizer_state,params)
            params=optax.apply_updates(params,updates)
            if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((params,value))):raise ValueError('nonfinite VMC update')
            history.append(dict(step=t-args.burn,energy=float(np.real(aux.energy))))
        else:
            for k in stored:stored[k].append(np.asarray(getattr(state,k)))
            accept.append(np.asarray(rate))
    if training:return params,{k:np.asarray(getattr(state,k)) for k in data},history
    return {k:np.concatenate(v) for k,v in stored.items()},dict(chains=count,blocks=args.blocks,block_size=args.block_size,
        burn=args.burn,steps_per_sample=args.steps,spin_flips=flips,seed=seed,acceptance=np.mean(accept,axis=0).tolist(),
        status='fresh MCMC; equilibration and block-length convergence require user verification')


def scan_window(net,params,data,info,args,local,output):
    """One head at a time. Fresh samples for each width, no naive walker reuse."""
    import jax
    import jax.numpy as jnp
    from periodicwave import networks
    from periodicwave.CustomScreenformer import _max_periodic_distance
    rows=[];energy=jax.jit(local);dref=float(_max_periodic_distance(jnp.asarray(info['lattice'])))
    for layer,ap in enumerate(params['layers']['attention']):
        for head in range(len(ap['mask_omega'])):
            for factor in args.factors:
                modified=jax.tree.map(lambda x:x,params);p=modified['layers']['attention'][layer]
                original_width=float(jax.nn.softplus(p['mask_omega'][head]));new=original_width*factor
                p['mask_omega']=p['mask_omega'].at[head].set(new+jnp.log(-jnp.expm1(-new)))
                frozen=jax.tree.map(lambda x:x,modified)
                for mode in (('frozen','retrained') if args.mode=='retrain' else ('frozen',)):
                    state=data
                    if mode=='retrained':
                        modified,state,history=sample_state(net,frozen,data,info,args,args.seed+20001,local,training=True)
                        for a,b in zip(frozen['layers']['attention'],modified['layers']['attention']):
                            np.testing.assert_array_equal(a['mask_omega'],b['mask_omega'])
                        np.savez_compressed(output/f'retrained-l{layer}-h{head}-f{factor:g}.npz',
                            params=jax.tree.map(lambda x:np.asarray(x)[None],modified),
                            data={k:v[None] for k,v in state.items()},t=info['step']+args.retrain_steps)
                        from screening_analysis.rs_observables import write_csv
                        write_csv(output/f'training-l{layer}-h{head}-f{factor:g}.csv',history)
                    batch,sampling=sample_state(net,modified,state,info,args,args.seed+30001,local)
                    values=[]
                    for i in range(len(batch['positions'])):
                        point=networks.FermiNetData(**{k:jnp.asarray(v[i]) for k,v in batch.items()})
                        values.append(np.asarray(energy(modified,jax.random.PRNGKey(i),point)[0]).item())
                    values=np.asarray(values)
                    if not np.isfinite(values).all():raise ValueError('nonfinite scan energy')
                    blocks=values.reshape(args.blocks,-1).real.mean(axis=1)
                    rows.append(dict(layer=layer,head=head,width_factor=factor,width=original_width*factor*dref,
                        mode=mode,energy=values.real.mean(),energy_imaginary=values.imag.mean(),
                        energy_block_sem=blocks.std(ddof=1)/np.sqrt(len(blocks)),samples=len(values),
                        retrain_steps=args.retrain_steps if mode=='retrained' else 0))
                    np.savez_compressed(output/f'energy-{mode}-l{layer}-h{head}-f{factor:g}.npz',local_energy=values,**batch)
                    print('window',layer,head,factor,mode,rows[-1]['energy'],flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan',type=Path);source.add_argument('--folder',type=Path);source.add_argument('--plot-only',type=Path)
    parser.add_argument('--index',type=int);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--common',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--mode',choices=('diagnostic','energy','scan','retrain'),default='diagnostic')
    for name,default in [('samples',8),('bins',12),('chains',4),('blocks',16),('block-size',4),('burn',64),
                         ('steps',10),('spin-flips',3),('seed',41031),('retrain-steps',1000)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--proposal',type=float,default=.2);parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--factors',type=float,nargs='+',default=[.7,1.,1.3])
    parser.add_argument('--plot-python',default='/home/tomax/venvs/pyrite-bands/bin/python')
    args=parser.parse_args()
    if args.plot_only:
        # Separate plotting interpreter; numerical GPU environment need not install Matplotlib.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        root=args.plot_only
        def read(name):
            with (root/name).open() as f:return list(csv.DictReader(f))
        A=read('A-connections.csv');B=read('B-response.csv');C=read('C-energy.csv') if (root/'C-energy.csv').exists() else []
        subsets=list(dict.fromkeys(r['subset'] for r in B));heads=sorted({int(r['head']) for r in B});layers=sorted({int(r['layer']) for r in B})
        conditions=json.loads((root/'conditions.json').read_text())
        h=conditions['hamiltonian'];rs=h['potential_kwargs']['interaction_energy_scale']
        title=f"lambda={h.get('lam','unknown')}, rs={rs:g}; physical phase unverified"
        fig,axes=plt.subplots(4,len(heads)*len(layers),figsize=(6*len(heads)*len(layers),13),squeeze=False)
        fig.suptitle(title)
        for col,(layer,head) in enumerate(( (l,h) for l in layers for h in heads)):
            for subset in subsets:
                a=[r for r in A if int(r['head'])==head and int(r['layer'])==layer and r['subset']==subset and r['source']=='screening' and r['distance_kind']=='network' and r['spin']=='all']
                b=[r for r in B if int(r['head'])==head and int(r['layer'])==layer and r['subset']==subset]
                axes[0,col].plot([float(r['distance_mid']) for r in a],[float(r['mean']) for r in a],label=subset)
                axes[1,col].plot([float(r['distance_mid']) for r in a],[float(r['total_per_configuration']) for r in a],label=subset)
                for key,style in [('mean_abs_amplitude','-'),('mean_abs_phase','--')]:
                    axes[2,col].plot([float(r['distance_mid']) for r in b],[float(r[key]) for r in b],style,label=subset+' '+key)
                c=[r for r in C if int(r['head'])==head and int(r['layer'])==layer]
                if c and subset=='equilibrium':axes[3,col].errorbar([float(r['distance_mid']) for r in c],[float(r['energy_gradient']) for r in c],yerr=[float(r['energy_gradient_block_sem']) for r in c],fmt='o-')
            for row,title in enumerate(('A: mean connection per pair','A: total connection per configuration','B: band amplitude / phase response','C: energy derivative (fresh MCMC only)')):
                axes[row,col].set(title=f'{title}; layer {layer}, head {head}',xlabel='Network periodic distance');axes[row,col].grid(alpha=.2)
                if row<3:axes[row,col].legend(fontsize=7)
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(root/'overview.png',dpi=150);plt.close(fig)
        if (root/'C-window.csv').exists():
            w=read('C-window.csv');fig,ax=plt.subplots()
            for layer in layers:
                for head in heads:
                    for mode in sorted({r['mode'] for r in w}):
                        chosen=[r for r in w if int(r['layer'])==layer and int(r['head'])==head and r['mode']==mode]
                        ax.errorbar([float(r['width_factor']) for r in chosen],[float(r['energy']) for r in chosen],yerr=[float(r['energy_block_sem']) for r in chosen],fmt='o-',label=f'L{layer} H{head} {mode}')
            ax.set(xlabel='Width / learned width',ylabel='Total variational energy (code units)',title=title);ax.legend();fig.tight_layout();fig.savefig(root/'window-energy.png',dpi=150);plt.close(fig)
        return
    if any(getattr(args,k)<1 for k in ('samples','bins','chains','blocks','block_size','steps','retrain_steps')) or args.burn<0 or args.spin_flips<0:
        parser.error('counts must be positive; burn/spin-flips nonnegative')
    if args.mode!='diagnostic' and (args.blocks<4 or args.chains<2):parser.error('energy needs >=4 blocks and >=2 chains')
    if not all(np.isfinite(x) and x>0 for x in [args.proposal,args.learning_rate,*args.factors]):parser.error('finite positive scales required')
    if args.index is not None and not args.plan:parser.error('--index requires --plan')
    if args.checkpoint and args.plan:parser.error('--checkpoint is for --folder; plan pins final completed results')
    import jax
    import jax.numpy as jnp
    from screening_analysis.checkpoint_model import load_model,_factory
    from screening_analysis.attn_probe import probe
    from screening_analysis.rs_observables import write_csv,phase_sensitivity
    from screening_analysis.topology_run import json_safe
    from screening_analysis.rs_sweep import ROOT
    plan=json.loads(args.plan.read_text()) if args.plan else None
    if plan:
        if args.index is not None and not 0<=args.index<len(plan['cases']):parser.error('array index out of range')
        cases=plan['cases'] if args.index is None else [plan['cases'][args.index]]
        common=args.common or Path(plan['common_inputs'])
        if not args.common and hashlib.sha256(common.read_bytes()).hexdigest()!=plan['common_inputs_sha256']:raise ValueError('common input bank changed')
    else:cases=[dict(id=args.folder.resolve().name,phase='unknown',phase_verified=False)];common=args.common
    for case in cases:
        folder=Path(plan['root'])/case['id']/'training' if plan else args.folder
        if plan and not (folder.parent/'analysis'/'finished.json').exists():raise RuntimeError('wait for completed training and analysis: '+case['id'])
        output=(args.output/case['id'] if plan and args.output else args.output) or ROOT/'results/manual'/(Path(plan['root']).name if plan else folder.parent.name)/case['id']/'connection-study'
        with load_model(folder,checkpoint=args.checkpoint) as (net,params,data,info):
            if not info['config']['network'].get('complex',False):raise ValueError('complex wavefunction required')
            if any('mask_omega' not in p or 'mask_centre' in p or 'mask_rings' in p for p in params['layers']['attention']):raise ValueError('requires monotone Window widths')
            no_width=(info['settings']['WIN_SHAPE']=='full' or
                (info['settings']['WIN_SHAPE']=='hybrid' and info['settings']['FREEZE_ETA'] and
                 all(np.all(np.asarray(p['mask_eta'])==1) for p in params['layers']['attention'])))
            if no_width and args.mode in ('scan','retrain'):
                raise ValueError('Full Window has no width to scan; use diagnostic/energy mode')
            cfg=info['config'];kw=dict(cfg['system']['make_local_energy_kwargs']);rs=float(kw['potential_kwargs']['interaction_energy_scale'])
            if plan and (not np.isclose(rs,case['rs']) or not np.isclose(kw.get('lam',0),case['lam'])):raise ValueError('plan and checkpoint conditions differ')
            output.mkdir(parents=True,exist_ok=False)
            conditions=dict(case=case,checkpoint=info['checkpoint'],checkpoint_sha256=info['checkpoint_sha256'],
                step=info['step'],mode=args.mode,hamiltonian=kw,arguments=vars(args),sampling_convergence='unverified',
                config_sha256=hashlib.sha256(Path(info['config_path']).read_bytes()).hexdigest(),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'periodicwave/attention.py')})
            (output/'conditions.json').write_text(json.dumps(json_safe(conditions),indent=2,default=str,ensure_ascii=False))
            # Keep retrained parameter artifacts reconstructable with load_model.
            (output/'runtime_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
            seed=case.get('seed',cfg.get('seed',cfg.get('debug',{}).get('seed','unknown')))
            (output/'CONDITIONS.md').write_text(f"# 計算条件\n\nλ={kw.get('lam','unknown')}、rs={rs:g}、N={sum(cfg['system']['electrons'])}、seed={seed}、学習step={info['step']}。\n\n初期化：{case.get('initialization','unknown')}。相は未確認。実行モード：{args.mode}。詳細は conditions.json。\n",encoding='utf-8')
            ids=np.linspace(0,len(data['positions'])-1,min(args.samples,len(data['positions']))).astype(int)
            batches={'native':{k:v[ids] for k,v in data.items()}}
            if common:
                with np.load(common,allow_pickle=False) as z:
                    if not np.allclose(z['lattice'],info['lattice'],atol=1e-12,rtol=0):raise ValueError('common lattice mismatch')
                    batches['common']={k:z[k][:args.samples] for k in data}
            local=None;sampling=None
            if args.mode!='diagnostic':
                kw['lattice']=jnp.asarray(info['lattice'])
                local=_factory(cfg['system']['make_local_energy_fn'])(net.apply,jnp.asarray(data['charges'][0]),tuple(cfg['system']['electrons']),complex_output=True,**kw)
                batches['equilibrium'],sampling=sample_state(net,params,data,info,args,args.seed,local)
            a_rows=[];b_rows=[];c_rows=[];distance_summary=[];energy_summary=None
            for subset,batch in batches.items():
                p=probe(folder,checkpoint=info['checkpoint'],config_path=info['config_path'],lattice=info['lattice'],evaluation_data=batch,nconf=len(batch['positions']),collect_internal=False)
                bins=dict(network_max=float(p['dref']),physical_max=float(np.linalg.norm(info['lattice'],axis=0).sum()/2),count=args.bins)
                edges=np.linspace(0,bins['network_max'],args.bins+1);centers=(edges[:-1]+edges[1:])/2;widths=np.full(args.bins,edges[1]-edges[0])
                pair_grad=None
                if subset!='equilibrium':
                    pair_dir=output/(subset+'-pairs');pair_dir.mkdir()
                    phase_sensitivity(net,params,batch,pair_dir,int(p['NL']),int(p['NH']))
                    with np.load(pair_dir/'connection-gradients.npz') as z:
                        if not z['valid'].all():raise ValueError('invalid individual pair sensitivity')
                        pair_grad=z['gradients']
                a,dist=analyze_connections(p,bins,rs,pair_grad)
                b,el=analyze_response(net,params,batch,centers,widths,output/(subset+'-measurements.npz'),
                    local if subset=='equilibrium' else None,args.blocks if subset=='equilibrium' else None)
                a_rows.extend(dict(subset=subset,**r) for r in a);b_rows.extend(dict(subset=subset,**r) for r in b)
                distance_summary.extend(dict(subset=subset,**r) for r in dist)
                if subset=='equilibrium':
                    c_rows=[dict(subset=subset,**r) for r in b]
                    eb=el.reshape(args.blocks,-1).real.mean(axis=1)
                    energy_summary=dict(energy=el.real.mean(),energy_imaginary=el.imag.mean(),energy_block_sem=eb.std(ddof=1)/np.sqrt(args.blocks))
                print(case['id'],subset,'A/B'+('/C' if subset=='equilibrium' else ''),'complete',flush=True)
            write_csv(output/'A-connections.csv',a_rows);write_csv(output/'A-distances.csv',distance_summary)
            write_csv(output/'B-response.csv',b_rows);write_csv(output/'C-energy.csv',c_rows)
            if args.mode in ('scan','retrain'):write_csv(output/'C-window.csv',scan_window(net,params,data,info,args,local,output))
            summary=dict(case=case,mode=args.mode,energy=energy_summary,sampling=sampling,
                distance='network periodic norm and closest-image Euclidean distance kept separate; *_aB = rs times code length',
                metric='only fresh MCMC rows have state_metric; saved/common centered variances are diagnostic',
                uncertainty='Time-block SEM; inspect burn, block length and independent runs. No convergence certificate.',
                retraining='finite-budget optimization, not a certified E_min(w)',phase='unknown')
            (output/'summary.json').write_text(json.dumps(json_safe(summary),indent=2,ensure_ascii=False))
            (output/'summary.md').write_text('# 接続・波動関数・エネルギー解析\n\n'
                'A-connections.csv：距離別の1ペア当たり重み、重みの総量、個別辺感度。\n\n'
                'A-distances.csv：実効距離と90%重み半径。networkとminimum_imageは別の距離です。\n\n'
                'B-response.csv：距離帯全体を変更した振幅・位相感度。個別辺の絶対感度平均とは異なります。\n\n'
                'C-energy.csv：追加MCMCからのエネルギー勾配。diagnosticモードでは計算しません。\n\n'
                'C-window.csv：幅変更後に追加サンプリングしたエネルギー。retrainedは有限回の再学習結果です。\n\n'
                'エネルギーは系全体のコード単位です。誤差は時間ブロックによる暫定推定で、平衡化・収束・相の確定を保証しません。\n',encoding='utf-8')
            subprocess.run([args.plot_python,'-m','screening_analysis.connection_study','--plot-only',str(output)],cwd=ROOT,check=True)
            (output/'finished.json').write_text(json.dumps(dict(complete=True)))
            print(output,flush=True)


if __name__=='__main__':main()
