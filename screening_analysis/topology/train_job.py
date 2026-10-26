"""Configurable fresh runs / non-destructive checkpoint continuation in new dirs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .workflow import ROOT, manifest, provenance, sha, simple_id, write


def jobs(path,output,resume=False,dry_run=False):
    data,base=manifest(path);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=resume)
    entries=data['jobs'];ids=[simple_id(r['id']) for r in entries]
    if len(set(ids))!=len(ids):raise ValueError('duplicate ids')
    defaults=data.get('defaults',{});planned=[]
    for entry in entries:
        cfg={**defaults,**entry};job=out/entry['id'];env={k:v for k,v in os.environ.items() if not k.startswith('PW_')}
        env.update({str(k):str(v) for k,v in defaults.get('env',{}).items()});env.update({str(k):str(v) for k,v in entry.get('env',{}).items()})
        env['JAX_PLATFORMS']=cfg.get('platform','cpu');env['PW_SEED']=str(cfg['seed']);env['PW_OUTPUT_DIR']=str(job/'run')
        if 'restore' in cfg:
            source=(base/cfg['restore']).resolve();command=[sys.executable,'-m','screening_analysis.topology.train_job',str(source),str(job/'run'),'--iterations',str(cfg['iterations'])]
            signature=dict(config=cfg,source_config_sha256=sha(source/'runtime_config.json'),
                source_checkpoints={p.name:sha(p) for p in source.glob('qmcjax_ckpt_*.npz')})
            if not signature['source_checkpoints']:raise ValueError('restore source has no checkpoints')
        else:
            command=[sys.executable,'-m','periodicwave.configs.run_lambda_jellium',str(cfg['n']),str(cfg['rs']),str(cfg['lam']),str(cfg['iterations']),entry['id']]
            signature=dict(config=cfg)
        planned.append(dict(id=entry['id'],command=command,environment={k:v for k,v in env.items() if k.startswith(('PW_','JAX_'))},signature=signature))
        if dry_run:continue
        done=job/'complete.json'
        if done.exists():
            previous=json.loads(done.read_text())
            if not resume or previous['signature']!=signature:raise ValueError('resume signature mismatch')
            continue
        if job.exists():raise ValueError('partial job retained; continue into a NEW job with restore set to its run directory: '+str(job))
        job.mkdir();write(job/'provenance.json',provenance(cfg['seed']));write(job/'plan.json',planned[-1])
        with (job/'run.log').open('x') as f:result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('job failed; original outputs retained: '+entry['id'])
        write(done,dict(signature=signature,status='executed; physical phase unverified'))
    write(out/('dry-run.json' if dry_run else 'plan.json'),dict(jobs=planned,provenance=provenance(),executed=not dry_run))


def continue_run(source,output,iterations):
    """Existing trainer restores params/optimizer/walkers; PRNG is NOT bitwise resumed."""
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from periodicwave import CustomScreenformer as CS,train_screen
    source=source.resolve();output=output.resolve()
    if output.exists():raise FileExistsError('continuation output must be NEW')
    config=json.loads((source/'runtime_config.json').read_text());settings=config['screening_attention_settings']
    for key in ('jax_enable_x64','jax_default_matmul_precision'):jax.config.update(key,settings[key])
    for key in ('TRIM_MAX','TRIM_POW','WIN_SHAPE','FREEZE_W','CLIP_R','CLIP_BETA'):setattr(CS,key,settings[key])
    checkpoints=list(source.glob('qmcjax_ckpt_*.npz'))
    if not checkpoints:raise ValueError('no checkpoint to restore')
    import numpy as np
    with np.load(sorted(checkpoints)[-1],allow_pickle=True) as z:last=int(z['t'])
    if iterations<=last+1:raise ValueError('iterations must exceed restored completed update count')
    before={str(p):sha(p) for p in checkpoints};config['log'].update(save_path=str(output),restore_path=str(source));config['optim']['iterations']=iterations
    output.mkdir(parents=True);write(output/'runtime_config.json',config)
    # Runtime JSON has numeric arrays, unlike the old stringifying config writer.
    for mapping,key in [(config['system'],'pbc_lattice'),(config['system']['make_local_energy_kwargs'],'lattice'),
                        (config['network']['make_feature_layer_kwargs'],'lattice')]:
        if key in mapping:mapping[key]=jnp.asarray(mapping[key])
    write(output/'continuation.json',dict(source=str(source),source_checkpoint_sha256=before,restored_t=last,total_iterations=iterations,
        rng='trainer initializes PRNG again: statistical continuation, NOT bitwise-identical trajectory',provenance=provenance(config['seed'])))
    train_screen.train(ConfigDict(config))
    assert all(sha(Path(p))==h for p,h in before.items()),'source checkpoint changed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--iterations',type=int,required=True)
    a=p.parse_args();continue_run(a.source,a.output,a.iterations)
