"""Execute available-checkpoint audit and a synthetic geometry-confound experiment.

Synthetic configurations are NOT quantum Monte Carlo or WC/AHC labels. Random
network parameters are never described as trained. All generated inputs/results
live under a new directory, leaving old checkpoints and snapshots read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from screening_analysis.topology_run import analyze_probe,compare_diagrams,json_safe


def digest_checkpoints(root):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('qmcjax_ckpt_*.npz')}


def settings(shape='cos',x64=True):
    return dict(TRIM_MAX=2.,TRIM_POW=2,WIN_SHAPE=shape,FREEZE_W=False,
                CLIP_R=None,CLIP_BETA=.3,jax_enable_x64=x64,jax_default_matmul_precision='float32')


def checkpoint_audit(out,root):
    from screening_analysis.attn_probe import probe,save_probe
    records=[]
    for name in ('screen-complex64','screen-default64','screen-complex32','screen-kfac32'):
        source=root/'tests/g3a_validation'/name
        cfg=json.loads((source/'effective-config.json').read_text())['config']
        # Runtime values reconstructed from the saved environment and unchanged
        # constructor; they are NOT encoded in the checkpoint. Record the evidence.
        x64='32' not in name; effective=settings(x64=x64)
        config_path=out/f'{name}-config.json';config_path.write_text(json.dumps(cfg,indent=2))
        checkpoints=sorted((source/'run').glob('qmcjax_ckpt_*.npz'))
        for ck in (checkpoints[0],checkpoints[len(checkpoints)//2],checkpoints[-1]):
            identifier=f'{name}-{ck.stem[-6:]}'
            a=probe(source/'run',checkpoint=ck,config_path=config_path,settings=effective,nconf=8,collect_internal=True)
            a['metadata']['pilot_setting_evidence']={
                'environment':str(source/'environment.json'),'constructor':'tests.check_g3a_vmc.build',
                'note':'explicit defaults cross-checked with source and recorded environment, not restored checkpoint metadata'}
            run=dict(id=identifier,group=name,phase='unknown',purpose='trained one-electron free-system diagnostic; no WC/AHC inference')
            summary=analyze_probe(a,out/identifier,run,null_repeats=1)
            save_probe(a,out/identifier/'probe.npz');records.append(summary)
            print(identifier,flush=True)
    for name in ('screen-complex64','screen-default64','screen-complex32','screen-kfac32'):
        runs=[r for r in records if r['run']['group']==name]
        for r in runs[1:]:compare_diagrams(out/runs[0]['run']['id']/'diagrams.npz',out/r['run']['id']/'diagrams.npz',out/f"{r['run']['id']}-drift.json")
    return records


def synthetic(out):
    import jax
    import jax.numpy as jnp
    from periodicwave import CustomScreenformer as CS,networks,envelopes,checkpoint
    from periodicwave.pbc.feature_layer import make_pbc_feature_layer
    from screening_analysis.attn_probe import probe,save_probe
    jax.config.update('jax_enable_x64',True)
    N=9; B=12; L=np.array([[3.,-1.5],[0.,3*np.sqrt(3)/2]])
    rng=np.random.default_rng(6017)
    grid=np.array([(i/3,j/3) for i in range(3) for j in range(3)])
    ordered=np.mod(grid[None,:,:]+rng.normal(0,.025,(B,N,2))+rng.uniform(size=(B,1,2)),1)@L.T
    disordered=rng.uniform(size=(B,N,2))@L.T
    modes=('cos','wide','smooth7','band','sin','gauss','fixed')
    records=[]
    for mode in modes:
        effective=settings('smooth7' if mode=='smooth7' else 'cos')
        for key in tuple(effective)[:6]:setattr(CS,key,effective[key])
        opts=dict(num_layers=2,num_heads=3,attn_dim=4,value_dim=4,mlp_dim=16,
            num_perceptrons_per_layer=1,use_layer_norm=True,mlp_activation_fct='GELU',
            use_softmask=True,use_gate=True,use_tanh_norm=True,use_fixed_value=mode=='fixed',
            use_output_scale=True,init_output_scale=0.,init_trim_r=1.9,
            init_softmask_w_factor=1.5 if mode=='wide' else .7,init_softmask_w_spread=2.)
        if mode=='band':opts['init_softmask_centres']=([0.,.4,.8],[.4,.4,.4])
        if mode in ('sin','gauss'):opts['init_softmask_rings']=([0.,.33,.67,1.],.1,1.,mode)
        fl=make_pbc_feature_layer(natoms=1,nspins=(N,0),ndim=2,lattice=jnp.asarray(L),include_r_ae=False)
        net=CS.make_fermi_net((N,0),jnp.zeros(1),ndim=2,determinants=1,
            envelope=envelopes.make_null_envelope(),feature_layer=fl,jastrow='NONE',
            jastrow_kwargs={'ndim':2},complex_output=True,pbc_lattice=jnp.asarray(L),**opts)
        for seed in (11,23,37):
            params=net.init(jax.random.PRNGKey(seed))
            replicated=jax.tree_util.tree_map(lambda a:a[None],params)
            for label,positions in (('ordered',ordered),('uniform',disordered)):
                identifier=f'synthetic-{mode}-seed{seed}-{label}'; folder=out/'inputs'/identifier
                folder.mkdir(parents=True)
                data=networks.FermiNetData(positions=jnp.asarray(positions.reshape(1,B,-1)),
                    spins=jnp.ones((1,B,N)),atoms=jnp.zeros((1,B,1,2)),charges=jnp.zeros((1,B,1)))
                ck=checkpoint.save(str(folder),0,data,replicated,None,jnp.ones(1)*.2)
                cfg=dict(seed=seed,screening_attention_settings=effective,
                    system=dict(electrons=[N,0],ndim=2,pbc_lattice=L.tolist()),
                    network=dict(network_type='CustomScreenformer',CustomScreenformer=opts,
                        make_feature_layer_fn='periodicwave.pbc.feature_layer.make_pbc_feature_layer',
                        make_feature_layer_kwargs=dict(lattice=L.tolist(),include_r_ae=False)))
                (folder/'config.json').write_text(json.dumps(cfg,indent=2))
                a=probe(folder,collect_internal=True)
                a['metadata']['synthetic']='untrained random network, prescribed geometry; NOT |psi|^2 samples or physical phase'
                summary=analyze_probe(a,out/identifier,dict(id=identifier,group=f'seed{seed}',
                    phase='unknown',synthetic_geometry=label,window=mode,seed=seed),null_repeats=2)
                save_probe(a,out/identifier/'probe.npz');records.append(summary)
                print(identifier,flush=True)
            compare_diagrams(out/f'synthetic-{mode}-seed{seed}-ordered'/'diagrams.npz',
                             out/f'synthetic-{mode}-seed{seed}-uniform'/'diagrams.npz',
                             out/f'synthetic-{mode}-seed{seed}-geometry-distances.json')
            jax.clear_caches()
    return records


def geometry_classification(records):
    """Leave-network-seed-out nearest-centroid, pool heads/layers (permutation invariant).

    Geometry labels only. No hyperparameter selection or walker-level splitting.
    This is a confound demonstration, with three seeds, not a significance claim.
    """
    results=[]
    keys=('H0_total_persistence','H1_total_persistence','H0_essential_intervals',
          'H1_essential_intervals','beta0_auc','beta1_auc','rank1_probability','rank2_probability')
    for mode in sorted({r['run']['window'] for r in records}):
        for source in ('window','trim','screening','qk'):
            for control in ('raw','rank','shuffle'):
                subset=[r for r in records if r['run']['window']==mode]
                X=np.array([[next(a for a in r['aggregates'] if a['source']==source and a['control']==control)[k] for k in keys] for r in subset])
                y=np.array([r['run']['synthetic_geometry']=='ordered' for r in subset],int)
                groups=np.array([r['run']['seed'] for r in subset]); predictions=[]
                for seed in np.unique(groups):
                    train=groups!=seed; test=~train; mean=X[train].mean(axis=0); sd=X[train].std(axis=0)
                    sd=np.where(sd>1e-10,sd,1.); z=(X-mean)/sd
                    centroids=np.stack([z[train&(y==c)].mean(axis=0) for c in (0,1)])
                    pred=np.argmin(np.sum((z[test,None,:]-centroids[None,:,:])**2,axis=-1),axis=-1)
                    predictions.extend((pred==y[test]).tolist())
                results.append(dict(window=mode,source=source,control=control,accuracy=float(np.mean(predictions)),
                    held_out_groups=3,test_units=6,label='synthetic ordered vs uniform; not WC/AHC'))
    return results


def reuse_synthetic(out,inputs):
    """Reevaluate frozen synthetic inputs with current diagnostic/analysis source."""
    from screening_analysis.attn_probe import probe,save_probe
    import jax
    records=[]
    for folder in sorted(inputs.glob('synthetic-*')):
        cfg=json.loads((folder/'config.json').read_text());identifier=folder.name
        _,mode,seed_text,label=identifier.split('-');seed=int(seed_text[4:])
        a=probe(folder,collect_internal=True)
        a['metadata']['synthetic']='untrained random network, prescribed geometry; NOT |psi|^2 samples or physical phase'
        summary=analyze_probe(a,out/identifier,dict(id=identifier,group=f'seed{seed}',
                    phase='unknown',synthetic_geometry=label,window=mode,seed=seed),null_repeats=2)
        save_probe(a,out/identifier/'probe.npz');records.append(summary)
        print(identifier,flush=True);jax.clear_caches()
    for mode in sorted({r['run']['window'] for r in records}):
        for seed in (11,23,37):
            compare_diagrams(out/f'synthetic-{mode}-seed{seed}-ordered'/'diagrams.npz',
                             out/f'synthetic-{mode}-seed{seed}-uniform'/'diagrams.npz',
                             out/f'synthetic-{mode}-seed{seed}-geometry-distances.json')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-synthetic',action='store_true')
    p.add_argument('--reuse-inputs',type=Path,help='existing pilot inputs directory; reevaluate without regenerating')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];before=digest_checkpoints(root)
    args.output.mkdir(parents=True,exist_ok=False)
    inventory={str(p.parent):len(list(p.parent.glob('qmcjax_ckpt_*.npz'))) for p in root.rglob('qmcjax_ckpt_*.npz')}
    (args.output/'checkpoint-inventory.json').write_text(json.dumps(dict(total=len(before),folders=inventory),indent=2))
    trained=checkpoint_audit(args.output,root)
    synthetic_records=([] if args.skip_synthetic else reuse_synthetic(args.output,args.reuse_inputs)
                       if args.reuse_inputs else synthetic(args.output))
    result=dict(trained_checkpoint_runs=trained,synthetic_runs=synthetic_records,
        geometry_classification=geometry_classification(synthetic_records) if synthetic_records else [],
        original_checkpoint_hashes_unchanged=all(Path(path).exists() and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in before.items()),
        original_checkpoint_sha256=before,
        physical_phase_validation='unavailable: no labeled WC/AHC checkpoint/config found',
        chern_berry_validation='not performed: no twist-family states/overlaps or physical reference labels',
        vmc_regularization_validation='not performed: no WC/AHC training baseline or independent phase validation')
    (args.output/'pilot.json').write_text(json.dumps(json_safe(result),indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
