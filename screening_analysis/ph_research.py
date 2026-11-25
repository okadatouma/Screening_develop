"""Spin/degree/distance constrained PH, sensitivity PH, and aligned physical diagnostics.

audit AUDIT_DIRECTORY --output NEW_DIR [--samples 8 --repeats 8 --k 60 90]
subset ANALYSIS_SUBSET --output NEW_DIR (rs-sweep native/common directory)
No training, scheduler submission or phase certification is performed here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from screening_analysis import topology as ph
from screening_analysis.constrained_null import rewire,available_switches,spin_distance_shuffle
from screening_analysis.density_ph import bounded_weights
from screening_analysis.topology_run import graph_sources,json_safe


def write(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_probe(path):
    with np.load(path,allow_pickle=False) as z:
        p={k:z[k] for k in z.files if k!='metadata_json'}
        p['metadata']=json.loads(str(z['metadata_json']))
    return p


def fingerprint(batch,i):
    return hashlib.sha256(b''.join(np.ascontiguousarray(batch[k][i],dtype=np.float64).tobytes()
                                  for k in ('positions','spins','atoms','charges'))).hexdigest()


def align_gradients(probe,inputs,gradients,valid):
    """Exact coordinate/spin/atom/charge alignment; no nearest-neighbor matching."""
    if len(inputs['positions'])!=len(gradients) or len(valid)!=len(gradients):raise ValueError('gradient/input lengths differ')
    lookup={fingerprint(probe,i):i for i in range(len(probe['positions']))}
    pairs=[]
    for j in range(len(gradients)):
        key=fingerprint(inputs,j)
        if key not in lookup:raise ValueError('sensitivity input is absent from probe')
        if valid[j] and np.isfinite(gradients[j]).all():pairs.append((lookup[key],j))
    return pairs


def importance_graph(derivatives,n,channel):
    """Log-weight derivative already contains W: do not multiply by W again."""
    d=np.asarray(derivatives)
    values={'amplitude':abs(d[0]),'phase':abs(d[1]),'complex':np.hypot(d[0],d[1])}[channel]
    if values.shape!=(n*(n-1)//2,) or not np.isfinite(values).all():raise ValueError('finite pair derivative vector required')
    scale=float(values.max(initial=0));w=np.zeros((n,n));ij=np.triu_indices(n,1)
    if scale>0:w[ij]=values/scale
    w+=w.T
    return w,scale


def graph_features(w):
    a=ph.analyze_graph(w)
    return dict(beta1_mean=float(a['betti'][1].mean()),
                h1_finite_persistence=a['summaries'][1]['total_persistence'],
                h1_essential=a['summaries'][1]['essential_intervals'],
                components_at_full_support=int(a['betti'][0,0]),
                beta1_at_full_support=int(a['betti'][1,0])),a


def top_graph(w,k):
    n=len(w);ij=np.triu_indices(n,1);values=w[ij];order=np.argsort(-values,kind='stable')
    if np.count_nonzero(values>0)<k:return None,'insufficient_positive_edges'
    if k<len(values) and values[order[k-1]]==values[order[k]]:return None,'boundary_tie'
    out=np.zeros_like(w);ids=order[:k];out[ij[0][ids],ij[1][ids]]=values[ids];out+=out.T
    return out,'ok'


def configuration_physics(p,ids,energy_samples):
    from screening_analysis.graph_wind import minimum_image_edges
    from screening_analysis.lj_maps import first_shell
    L=p['lattice'];N=int(p['NE']);area=abs(np.linalg.det(L));shell=np.asarray(first_shell(L,N))
    reciprocal=2*np.pi*np.linalg.inv(L)
    translations=np.array([(i,j) for i in range(-3,4) for j in range(-3,4) if (i,j)!=(0,0)])@L.T
    edges=np.linspace(0,.45*np.linalg.norm(translations,axis=1).min(),17)
    norm=N*(N-1)/(2*area)*np.pi*np.diff(edges**2)
    rows=[];g=[];sq=[]
    for sample in ids:
        x=p['positions'][sample].reshape(N,2);frac=x@np.linalg.inv(L).T
        sk=abs(np.exp(2j*np.pi*frac@shell.T).sum(axis=0))**2/N
        dist=minimum_image_edges(x,L)[1];gr=np.histogram(dist,bins=edges)[0]/norm
        row=dict(sample=int(sample),structure_factor_shell=float(sk.mean()),
                 mean_pair_distance=float(dist.mean()),local_energy_real=None,local_energy_imag=None)
        row.update({f'gr_bin{i}':float(v) for i,v in enumerate(gr)})
        rows.append(row);g.append(gr);sq.append(sk)
    if energy_samples:
        import jax
        import jax.numpy as jnp
        from periodicwave import networks
        from screening_analysis.checkpoint_model import load_model,_factory
        checkpoint=Path(p['metadata']['checkpoint'])
        with load_model(checkpoint.parent,checkpoint=checkpoint,lattice=L) as (net,params,_,info):
            cfg=info['config'];kw=dict(cfg['system']['make_local_energy_kwargs']);kw['lattice']=jnp.asarray(L)
            local=_factory(cfg['system']['make_local_energy_fn'])(net.apply,jnp.asarray(p['charges'][0]),
                tuple(cfg['system']['electrons']),complex_output=True,**kw)
            evaluate=jax.jit(local)
            for row in rows[:energy_samples]:
                i=row['sample'];data=networks.FermiNetData(**{k:jnp.asarray(p[k][i]) for k in ('positions','spins','atoms','charges')})
                value=np.asarray(evaluate(params,jax.random.PRNGKey(i),data)[0]).item()
                row['local_energy_real']=float(np.real(value)) if np.isfinite(value) else None
                row['local_energy_imag']=float(np.imag(value)) if np.isfinite(value) else None
    return rows,dict(gr=np.asarray(g),r_edges=edges,structure_factor=np.asarray(sq),integer_modes=shell,
                     k_vectors=shell@reciprocal,sample_indices=np.asarray(ids))


def correlations(features,physical):
    lookup={r['sample']:r for r in physical};result=[]
    groups=sorted({(r['source'],r['control'],r['layer'],r['head']) for r in features})
    for source,control,layer,head in groups:
        selected=[r for r in features if (r['source'],r['control'],r['layer'],r['head'])==(source,control,layer,head)]
        for metric in ('beta1_mean','h1_finite_persistence','h1_essential'):
            for observable in ('structure_factor_shell','local_energy_real','mean_pair_distance',*[f'gr_bin{i}' for i in range(16)]):
                pairs=[(r[metric],lookup[r['sample']][observable]) for r in selected if lookup[r['sample']][observable] is not None]
                a=np.asarray(pairs,dtype=float).reshape(-1,2)
                status='insufficient_samples' if len(a)<5 else 'constant' if min(a.std(axis=0))<1e-12 else 'descriptive_only'
                value=float(np.corrcoef(a.T)[0,1]) if status=='descriptive_only' else None
                result.append(dict(source=source,control=control,layer=layer,head=head,ph_metric=metric,
                    physical_observable=observable,samples=len(a),pearson=value,status=status,
                    unit='paired configuration within ONE training run; no p-value/independent-seed claim'))
    return result


def analyze(p,output,*,gradients=None,gradient_inputs=None,valid=None,samples=8,repeats=8,ks=(60,90),bins=8,energy_samples=0,seed=41031):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    N=int(p['NE']);NL=int(p['NL']);NH=int(p['NH']);rng=np.random.default_rng(seed)
    if gradients is not None:
        expected=(2,NL,NH,N*(N-1)//2)
        if gradients.shape[1:]!=expected:raise ValueError(f'gradient shape expected [B,{expected}], got {gradients.shape}')
        pairs=align_gradients(p,gradient_inputs,gradients,valid)
        if not pairs:raise ValueError('no valid aligned sensitivity inputs')
        selected=np.linspace(0,len(pairs)-1,min(samples,len(pairs))).astype(int)
        pairs=[pairs[i] for i in selected];gradient_index=dict(pairs);ids=list(gradient_index)
    else:
        ids=np.linspace(0,len(p['positions'])-1,min(samples,len(p['positions']))).astype(int).tolist();gradient_index={}
    sources={s:bounded_weights(w) for s,w in graph_sources(p).items() if s in ('qk','screening')}
    features=[];nullrows=[];relaxed_rows=[];moves=[];diagrams={};curves={};importance_scales=[]
    for sample in ids:
        for layer in range(NL):
            for head in range(NH):
                matrices={s:ph.symmetrize(w[sample,layer,head]) for s,w in sources.items()}
                if sample in gradient_index:
                    for channel in ('amplitude','phase','complex'):
                        graph,scale=importance_graph(gradients[gradient_index[sample],:,layer,head],N,channel)
                        matrices['importance_'+channel]=graph
                        importance_scales.append(dict(sample=sample,layer=layer,head=head,channel=channel,maximum_absolute_importance=scale))
                for source,w in matrices.items():
                    for control,graph in [('raw',w),('positive_rank',ph.rank_weights(w))]:
                        metrics,a=graph_features(graph)
                        features.append(dict(sample=sample,layer=layer,head=head,source=source,control=control,**metrics))
                        key=f'b{sample}_l{layer}_h{head}_{source}_{control}'
                        for dim,d in enumerate(a['diagrams']):diagrams[key+f'_H{dim}']=d
                        curves[key]=a['betti']
                    for k in ks:
                        initial,status=top_graph(w,k)
                        if initial is None:
                            moves.append(dict(sample=sample,layer=layer,head=head,source=source,k=k,repeat=-1,status=status,
                                attempts=0,accepted=0,target=0,changed_edge_fraction=0.,target_reached=False,movable=False));continue
                        observed,_=graph_features(initial)
                        count_moves=available_switches(initial,p['spins'][sample],admissible=w>0)
                        for repeat in range(repeats):
                            if count_moves==0:
                                randomized=initial.copy()
                                diagnostic=dict(attempts=0,accepted=0,target=3*k,changed_edge_fraction=0.,target_reached=False,movable=False)
                            else:
                                randomized,diagnostic=rewire(initial,p['spins'][sample],p['d'][sample],rng,bins=bins,
                                    admissible=w>0,max_attempts=max(2000,100*k),switches=3*k)
                            metrics,a=graph_features(randomized)
                            moves.append(dict(sample=sample,layer=layer,head=head,source=source,k=k,repeat=repeat,
                                status='conditional_control' if diagnostic['movable'] else 'frozen_or_returned',
                                **{key:diagnostic[key] for key in ('attempts','accepted','target','changed_edge_fraction','target_reached','movable')}))
                            nullrows.append(dict(sample=sample,layer=layer,head=head,source=source,k=k,repeat=repeat,
                                movable=diagnostic['movable'],target_reached=diagnostic['target_reached'],
                                **{'observed_'+key:value for key,value in observed.items()},**{'null_'+key:value for key,value in metrics.items()}))
                            # Retain actual null diagrams, not just scalar summaries.
                            key=f'b{sample}_l{layer}_h{head}_{source}_k{k}_null{repeat}'
                            for dim,d in enumerate(a['diagrams']):diagrams[key+f'_H{dim}']=d
                            for conditional in ('spin_degree_only','spin_distance_only'):
                                if conditional=='spin_degree_only':
                                    if count_moves==0:
                                        randomized=initial.copy();diagnostic_relaxed=dict(degree_rmse=0.,max_node_spin_degree_change=0,available_two_switches=0)
                                    else:
                                        randomized,info=rewire(initial,p['spins'][sample],p['d'][sample],rng,bins=1,
                                            switches=3*k,max_attempts=max(2000,100*k),admissible=w>0)
                                        diagnostic_relaxed=dict(degree_rmse=0.,max_node_spin_degree_change=0,available_two_switches=count_moves)
                                else:
                                    randomized,info=spin_distance_shuffle(initial,p['spins'][sample],p['d'][sample],rng,bins=bins,admissible=w>0)
                                    diagnostic_relaxed=dict(degree_rmse=info['degree_rmse'],max_node_spin_degree_change=info['max_node_spin_degree_change'],available_two_switches=count_moves)
                                metrics_relaxed,ar=graph_features(randomized)
                                relaxed_rows.append(dict(sample=sample,layer=layer,head=head,source=source,k=k,repeat=repeat,
                                    conditional=conditional,**diagnostic_relaxed,
                                    **{'observed_'+key:value for key,value in observed.items()},
                                    **{'null_'+key:value for key,value in metrics_relaxed.items()}))
    physical,arrays=configuration_physics(p,ids,min(energy_samples,len(ids)))
    write(output/'ph-features.csv',features);write(output/'constrained-null.csv',nullrows)
    write(output/'constraint-hierarchy.csv',relaxed_rows)
    write(output/'switch-diagnostics.csv',moves);write(output/'importance-scales.csv',importance_scales)
    write(output/'configuration-physics.csv',physical);write(output/'within-run-correlations.csv',correlations(features,physical))
    np.savez_compressed(output/'physical-arrays.npz',**arrays)
    np.savez_compressed(output/'diagrams.npz',**diagrams);np.savez_compressed(output/'betti-curves.npz',**curves)
    summary=dict(samples=len(ids),sample_indices=ids,layers=NL,heads=NH,seed=seed,repeats=repeats,k=list(ks),distance_bins=bins,
        checkpoint_sha256=p['metadata']['checkpoint_sha256'],input_sha256=p['metadata']['evaluation_input_sha256'],
        constraints='At each reference k: exact edge count, per-node same/opposite-spin degrees, per-distance-bin weight multiset and support admissibility. NOT per-threshold degrees.',
        null_inference='Finite constrained switch chains; no uniform sampling, mixing certificate or significance claims. Check switch diagnostics.',
        bipartite_caveat='For triangle-free graphs beta1=E-V+components; fixed E and components determine beta1. Compare births and weighted filtration, not only endpoint loop counts.',
        importance='Separate filtration: |d log|psi|/d epsilon|, |d phi/d epsilon|, hypot of both. Each divided by own maximum; original scales saved; no extra W multiplication.',
        physical='Per-configuration S(q), canonical finite-N g(r), optionally exact configured local energy. Saved/fixed inputs are not certified equilibrium samples.',
        phase_status='unknown; independent twist-boundary state family missing unless separately supplied',
        source_sha256={str(Path(f).resolve()):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (__file__,__import__('screening_analysis.constrained_null',fromlist=['']).__file__)})
    (output/'summary.json').write_text(json.dumps(json_safe(summary),indent=2,ensure_ascii=False,allow_nan=False))
    return summary


def run_subset(subset,output,settings):
    p=load_probe(subset/'probe.npz')
    with np.load(subset/'connection-gradients.npz',allow_pickle=False) as z:gradients=z['gradients'];valid=z['valid']
    with np.load(subset/'phase-inputs.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    return analyze(p,output,gradients=gradients,gradient_inputs=inputs,valid=valid,**settings)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','subset'):
        p=sub.add_parser(name);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--samples',type=int,default=8);p.add_argument('--repeats',type=int,default=8)
        p.add_argument('--k',type=int,nargs='+',default=[60,90]);p.add_argument('--bins',type=int,default=8)
        p.add_argument('--energy-samples',type=int,default=0)
    args=parser.parse_args()
    settings=dict(samples=args.samples,repeats=args.repeats,ks=args.k,bins=args.bins,energy_samples=args.energy_samples)
    if min(args.samples,args.repeats,args.bins,*args.k)<1:parser.error('positive counts required')
    if args.command=='subset':run_subset(args.input,args.output,settings)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        with np.load(args.input/'shared-inputs.npz',allow_pickle=False) as z:inputs={k:z[k] for k in ('positions','spins','atoms','charges')}
        for folder in sorted(args.input.glob('lam*')):
            if not (folder/'summary.json').exists():continue
            p=load_probe(folder/'shared-probe.npz');summary=json.loads((folder/'summary.json').read_text())
            if summary['checkpoint_sha256']!=p['metadata']['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
            if summary['shared_input_sha256']!=hashlib.sha256((args.input/'shared-inputs.npz').read_bytes()).hexdigest():raise ValueError('shared input changed')
            with np.load(folder/'connection-sensitivity.npz',allow_pickle=False) as z:gradients=z['derivatives'][:,:,None]
            result=analyze(p,args.output/folder.name,gradients=gradients,gradient_inputs=inputs,
                           valid=np.isfinite(gradients).all(axis=(1,2,3,4)),**settings)
            print(folder.name,result['samples'],'complete',flush=True)


if __name__=='__main__':main()
