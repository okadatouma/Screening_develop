"""Fresh natural sampling A vs byte-identical common-input B; grouped metrics.

Labels below are lambda CONDITIONS, never physical WC/AHC labels. Actual phase
metrics stay unavailable without independent labels and converged state families.
"""
import argparse
from itertools import combinations
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from periodicwave import networks
from screening_analysis.checkpoint_model import load_model
from screening_analysis.attn_probe import probe,save_probe
from screening_analysis.physical_loops import make_network_evaluator,sample_wavefunction
from screening_analysis.benchmark_features import features
from screening_analysis import phase_metrics,topology
from screening_analysis.topology_run import json_safe


def mean_sem_by_chain(values):
    means=values.mean(axis=0)
    return float(np.std(means,ddof=1)/np.sqrt(len(means)))


def fresh_samples(folder,path,sweeps=16,burn=32):
    with load_model(folder,path) as (net,params,data,info):
        cfg=info['config'];N=sum(cfg['system']['electrons']);L=info['lattice']
        atoms,charges=data['atoms'][0],data['charges'][0]
        evaluate=make_network_evaluator(net,params,jnp.asarray(atoms),jnp.asarray(charges),
                                        'angle' if cfg['network']['complex'] else 'sign')
        sampling_seed=5107+100003*int(cfg['seed'])
        sample=sample_wavefunction(evaluate,L,N,seed=sampling_seed,chains=4,sweeps=sweeps,burn=burn,steps_per_sample=2,
                                  width=.12,pseudospin=cfg['system'].get('pseudospin',False))
        x=sample['positions'];s=sample['spins'];T,C=x.shape[:2]
        external=dict(positions=x.reshape(T*C,-1),spins=s.reshape(T*C,N),
                      atoms=np.broadcast_to(atoms,(T*C,)+atoms.shape),charges=np.broadcast_to(charges,(T*C,)+charges.shape))
        name=cfg['system']['make_local_energy_fn'];module,attribute=name.rsplit('.',1)
        if not module.startswith('periodicwave.'):raise ValueError('unknown Hamiltonian factory')
        kw=dict(cfg['system']['make_local_energy_kwargs']);kw['lattice']=jnp.asarray(kw['lattice'])
        energy=getattr(importlib.import_module(module),attribute)(net.apply,jnp.asarray(charges),tuple(cfg['system']['electrons']),complex_output=cfg['network']['complex'],**kw)
        def local(r,spin):
            d=networks.FermiNetData(positions=r,spins=spin,atoms=jnp.asarray(atoms),charges=jnp.asarray(charges))
            return energy(params,None,d)[0]
        batch=jax.jit(jax.vmap(local));els=[]
        for start in range(0,T*C,4):els.append(np.asarray(batch(jnp.asarray(external['positions'][start:start+4]),jnp.asarray(external['spins'][start:start+4]))))
        el=np.concatenate(els).reshape(T,C)
        rs=float(cfg['system']['make_local_energy_kwargs']['potential_kwargs']['interaction_energy_scale'])
        measurement=dict(energy=float(el.real.mean()),energy_per_particle_Ha=float(el.real.mean()/N/rs**2) if rs>0 else None,
            variance_abs_complex=float(np.mean(abs(el-el.mean())**2)),variance_real=float(el.real.var()),
            imaginary_energy=float(el.imag.mean()),chain_sem=mean_sem_by_chain(el.real),imaginary_chain_sem=mean_sem_by_chain(el.imag),
            nonfinite_rate=float(np.mean(~np.isfinite(el))),acceptance=sample['acceptance'],spin_acceptance=sample['spin_acceptance'],
            chains=C,sweeps=T,burn=burn,seed=sampling_seed,rng_group='training seed; independent across held-out seed groups',
            sampling_status='fresh fixed-psi MCMC, short; equilibrium/convergence not certified')
        return external,measurement,info,el


def distribution_distance(a,b,shared=True,bottleneck=True,mode='completed',count=8):
    ia=np.linspace(0,a['samples']-1,min(count,a['samples'])).astype(int)
    ib=np.linspace(0,b['samples']-1,min(count,b['samples'])).astype(int)
    pairs=list(zip(ia,ib)) if shared else [(i,j) for i in ia for j in ib]
    values=[]
    with np.load(a['diagrams'],allow_pickle=False) as left,np.load(b['diagrams'],allow_pickle=False) as right:
        for i,j in pairs:
            values.append(sum(topology.diagram_distance(left[f'{mode}_b{i}_H{d}'],right[f'{mode}_b{j}_H{d}'],bottleneck=bottleneck) for d in (0,1)))
    v=np.asarray(values);finite=v[np.isfinite(v)]
    return dict(mean=float(v.mean()),finite_mean=float(finite.mean()) if len(finite) else None,infinite=int(np.isinf(v).sum()),
                comparisons=len(v),pairing='identical configurations' if shared else 'product of natural configuration sets')


def evaluate_records(records):
    results=[]
    for view in ('A','B'):
        for mode in ('off','persistence','laplacian','weights'):
            selected=[r for r in records if r['kind']=='core' and r['mode']==mode and r['view']==view]
            if len(selected)!=6:continue
            labels=[r['candidate_parameter_label'] for r in selected];groups=[r['seed_group'] for r in selected]
            for size in (8,16,32):
                for family in selected[0]['features_by_count'][str(size)]:
                    x=[r['features_by_count'][str(size)][family] for r in selected]
                    evaluation=phase_metrics.evaluate(x,labels,groups,bootstrap_repetitions=1000)
                    results.append(dict(view=view,regularizer=mode,samples=size,feature=family,
                        label_kind='lambda condition ONLY, not WC/AHC phase',**evaluation))
    # Holm correction across ALL exploratory parameter-classification feature tests.
    p=np.array([r['permutation'].get('pvalue',1.) for r in results]);order=np.argsort(p);adjusted=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    for idx,value in zip(order,adjusted):results[idx]['holm_pvalue']=float(min(1.,value))
    return dict(phase_classification_status='unavailable: no independently verified physical phase labels',
        candidate_condition_results=results,
        limitation='3 independent paired seed groups per condition; bootstrap is conditional on fitted OOF models; p-values cannot establish physical phase identification')


def distances(records):
    rows=[]
    for view in ('A','B'):
        for reg in ('off','persistence','laplacian','weights'):
            selected=[r for r in records if r['kind']=='core' and r['mode']==reg and r['view']==view]
            for graph in ('raw','completed'):
                for metric in ('bottleneck','wasserstein1'):
                    pairs=[]
                    for a,b in combinations(selected,2):
                        if a['seed_group']==b['seed_group']:continue
                        if view=='B' and a['input_sha256']!=b['input_sha256']:raise ValueError('shared-input comparison hash mismatch')
                        d=distribution_distance(a,b,view=='B',metric=='bottleneck',graph)
                        pairs.append(dict(left=a['id'],right=b['id'],same_parameter_label=a['candidate_parameter_label']==b['candidate_parameter_label'],**d))
                    def describe(same):
                        values=np.array([d['mean'] for d in pairs if d['same_parameter_label']==same]);finite=values[np.isfinite(values)]
                        return dict(pairs=len(values),infinite=int(np.isinf(values).sum()),finite_mean=float(finite.mean()) if len(finite) else None)
                    within,between=describe(True),describe(False);margin=None
                    if within['finite_mean'] is not None and between['finite_mean'] is not None:
                        total=within['finite_mean']+between['finite_mean'];margin=(between['finite_mean']-within['finite_mean'])/total if total>0 else 0.
                    rows.append(dict(view=view,regularizer=reg,graph=graph,metric=metric,within_parameter=within,between_parameter=between,
                        finite_normalized_separation=margin,pairs=pairs,label_kind='same/different lambda, NOT verified phases'))
    return rows


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((args.benchmark/'summary.json').read_text());models=[]
    for case in cases:
        if case['status']=='complete_short_unconverged':models.append((case,Path(case['checkpoint'])))
    base=next(c for c in cases if c['id']=='N9-lam033-off-seed11')
    for step in (0,11):models.append((dict(base,id=base['id']+f'-step{step:06d}',kind='checkpoint_drift'),Path(base['checkpoint']).parent/f'qmcjax_ckpt_{step:06d}.npz'))
    base72=next(c for c in cases if c['id']=='N9-lam072-off-seed11')
    models.append((dict(base72,id=base72['id']+'-step000011',kind='checkpoint_drift'),Path(base72['checkpoint']).parent/'qmcjax_ckpt_000011.npz'))
    common={};records=[];perturbations=[]
    for case,path in models:
        folder=path.parent;model_out=out/case['id'];model_out.mkdir()
        natural,measurement,info,el=fresh_samples(folder,path,args.sweeps,args.burn)
        cfg=info['config'];key=json.dumps([cfg['system']['electrons'],np.asarray(info['lattice']).tolist(),cfg['system'].get('pseudospin',False)])
        if key not in common:
            # No held-out network supplies this covariate distribution. The pool
            # is independent of EVERY trained model and phase/candidate label.
            rng=np.random.default_rng(19021);N=sum(cfg['system']['electrons']);L=info['lattice']
            common[key]=dict(positions=(rng.uniform(size=(32,N,2))@L.T).reshape(32,-1),
                spins=rng.choice([-1.,1.],size=(32,N)) if cfg['system'].get('pseudospin',False) else np.ones((32,N)),
                atoms=np.broadcast_to(natural['atoms'][0],(32,)+natural['atoms'].shape[1:]),
                charges=np.broadcast_to(natural['charges'][0],(32,)+natural['charges'].shape[1:]))
        np.savez_compressed(model_out/'fresh_measurement.npz',local_energy=el,**natural)
        for view,data in (('A',natural),('B',common[key])):
            result=probe(folder,checkpoint=path,config_path=info['config_path'],collect_internal=True,nconf=32,evaluation_data=data)
            result['metadata'].update(comparison_view=view,input_mode='natural_resampled_external' if view=='A' else 'shared_external_configurations',
                                      measurement=json_safe(measurement) if view=='A' else None,
                                      shared_input_origin=None if view=='A' else 'fixed uniform torus / random pseudospin pool, seed19021, independent of all models and labels',
                                      warning='B is a network response comparison, not an energy expectation under this wavefunction')
            save_probe(result,model_out/f'probe_{view}.npz')
            full=features(result,count=32)
            diagram_path=model_out/f'diagrams_{view}.npz';np.savez_compressed(diagram_path,**full['diagrams'])
            by_count={str(n):features(result,count=n)['features'] for n in (8,16,32)}
            record=dict(id=case['id'],kind=case['kind'],mode=case['mode'],gamma=case['gamma'],seed=case['seed'],
                seed_group=case['seed_group'],candidate_parameter_label=case['candidate_parameter_label'],phase='unknown',phase_verified=False,
                view=view,input_sha256=result['metadata']['evaluation_input_sha256'],samples=len(result['positions']),
                features_by_count=by_count,diagrams=str(diagram_path),measurement=measurement if view=='A' else None,
                checkpoint_sha256=info['checkpoint_sha256'],step=info['step'],completed_updates=info['step']+1)
            records.append(record)
            if view=='B' and case['kind']=='core' and case['mode']=='off':
                for sigma in (.001,.01,.05):
                    rng=np.random.default_rng(19037);changed={k:np.array(v,copy=True) for k,v in data.items()}
                    changed['positions']+=sigma*rng.normal(size=changed['positions'].shape)
                    perturbed=probe(folder,checkpoint=path,config_path=info['config_path'],nconf=32,evaluation_data=changed)
                    value=features(perturbed,count=32);p=model_out/f'perturbed_{sigma:g}.npz';np.savez_compressed(p,**value['diagrams'])
                    target=dict(diagrams=str(p),samples=32)
                    delta=distribution_distance(record,target,True,True,'completed')
                    perturbations.append(dict(id=case['id'],sigma=sigma,mean_abs_weight_change=float(np.mean(abs(perturbed['W_window']-result['W_window']))),
                        completed_PH_bottleneck=delta))
        print(case['id'],measurement['energy'],measurement['variance_abs_complex'],flush=True)
    (out/'records.json').write_text(json.dumps(json_safe(records),indent=2,allow_nan=False))
    (out/'classification.json').write_text(json.dumps(json_safe(evaluate_records(records)),indent=2,allow_nan=False))
    (out/'distances.json').write_text(json.dumps(json_safe(distances(records)),indent=2,allow_nan=False))
    (out/'coordinate_perturbations.json').write_text(json.dumps(json_safe(perturbations),indent=2,allow_nan=False))
    # Matched 12-update baseline for stress and initial-wavefunction comparisons.
    references={r['candidate_parameter_label']:r for r in records if r['view']=='B' and r['id'].endswith('off-seed11-step000011')}
    stress=[]
    for r in records:
        if r['view']!='B' or r['kind'] not in ('stress','initial_wavefunction','checkpoint_drift'):continue
        reference=references[r['candidate_parameter_label']]
        if r['input_sha256']!=reference['input_sha256']:
            stress.append(dict(id=r['id'],status='different N/cell: no identical-configuration comparison'));continue
        stress.append(dict(id=r['id'],completed_updates=r['completed_updates'],reference_updates=12,
                          distance=distribution_distance(reference,r,True,True,'completed')))
    (out/'stress_comparisons.json').write_text(json.dumps(json_safe(stress),indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('benchmark',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sweeps',type=int,default=16);p.add_argument('--burn',type=int,default=32);run(p.parse_args())
