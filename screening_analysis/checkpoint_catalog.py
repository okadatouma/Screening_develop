"""Read-only checkpoint discovery with explicit unknown metadata and duplicate hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def find_config(folder):
    folder=Path(folder)
    for directory in (folder,folder.parent):
        for name in ('runtime_config.json','effective-config.json','config.json','manifest.json'):
            path=directory/name
            if path.exists():
                obj=json.loads(path.read_text());cfg=obj.get('config',obj)
                if 'system' in cfg or ('n' in cfg and 'lattice' in cfg):return cfg,path
    return {},None


def training_statistics(folder):
    folder=Path(folder);paths=[folder/'train_stats.csv',folder/'stats.csv']
    path=next((p for p in paths if p.exists()),None)
    if path is None:return dict(status='unavailable')
    with path.open() as f:rows=list(csv.DictReader(f))
    energies=[]
    for row in rows:
        value=row.get('energy',row.get('energy_total_code'))
        try:energies.append(float(value))
        except (TypeError,ValueError):continue
    a=np.asarray(energies);finite=a[np.isfinite(a)];tail=finite[-min(16,len(finite)):]
    return dict(status='short_or_unvalidated',file=str(path),rows=len(rows),nonfinite_energy_rows=int((~np.isfinite(a)).sum()),
                final_energy=float(finite[-1]) if len(finite) else None,
                tail_mean=float(tail.mean()) if len(tail) else None,tail_std=float(tail.std()) if len(tail) else None,
                note='training batch sequence, not an independent convergence certificate or MC SEM')


def _tree_arrays(tree,prefix=''):
    if isinstance(tree,dict):
        for key in sorted(tree):yield from _tree_arrays(tree[key],prefix+'/'+str(key))
    elif isinstance(tree,(list,tuple)):
        for i,value in enumerate(tree):yield from _tree_arrays(value,prefix+'/'+str(i))
    else:yield prefix,np.asarray(tree)


def inspect(path):
    path=Path(path);folder=path.parent;cfg,config_path=find_config(folder)
    system=cfg.get('system',{});network=cfg.get('network',{});kw=system.get('make_local_energy_kwargs',{})
    seed=cfg.get('seed',cfg.get('debug',{}).get('seed'))
    if seed is None:
        env=folder.parent/'environment.json'
        if env.exists():
            value=json.loads(env.read_text()).get('environment',{}).get('PW_SEED')
            if value is not None:seed=int(value)
    row=dict(path=str(path.resolve()),run_id=str(folder.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config_path=str(config_path.resolve()) if config_path else None,seed=seed,n=sum(system['electrons']) if 'electrons' in system else cfg.get('n'),
        lam=kw.get('lam',cfg.get('lam')),rs=kw.get('potential_kwargs',{}).get('interaction_energy_scale',cfg.get('rs')),
        delta=kw.get('delta',cfg.get('delta')),hamiltonian=system.get('make_local_energy_fn'),
        model=network.get('network_type',cfg.get('schema')),wavefunction=network.get('CustomScreenformer',{}),
        complex=network.get('complex'),jastrow=network.get('jastrow'),runtime_settings=cfg.get('screening_attention_settings'),
        phase='unknown',phase_verified=False,convergence=training_statistics(folder))
    with np.load(path,allow_pickle=True) as z:
        row['step']=int(z['t']);data=z['data'].item();params=z['params'].item()
        row['positions_shape']=list(np.asarray(data['positions']).shape)
        digest=hashlib.sha256()
        for key,array in _tree_arrays(params):
            digest.update(key.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
        row['parameter_sha256']=digest.hexdigest()
        if 'layers' in params and 'attention' in params['layers']:
            attn=params['layers']['attention'];row['layers']=len(attn)
            if len(attn):row['heads']=int(np.asarray(attn[0]['trim_rho']).shape[-1]) if 'trim_rho' in attn[0] else None
    if row['lam'] in (.33,.72) and row['rs']==15:
        row['candidate']='WC_candidate' if row['lam']==.33 else 'AHC_candidate'
        row['candidate_basis']='Hamiltonian parameters only, NOT a verified phase'
    return row


def discover(root):
    rows=[]
    for path in sorted(Path(root).rglob('qmcjax_ckpt_*.npz')):
        try:rows.append(inspect(path))
        except Exception as exc:rows.append(dict(path=str(path.resolve()),status='unreadable_or_metadata_error',error=type(exc).__name__+': '+str(exc)))
    duplicate={}
    for row in rows:
        key=row.get('checkpoint_sha256')
        if key:duplicate.setdefault(key,[]).append(row['path'])
    return dict(checkpoints=rows,total=len(rows),duplicate_checkpoint_groups=[v for v in duplicate.values() if len(v)>1],
        independent_phase_labels='none inferred; parameters and file names never certify WC/AHC/liquid/halo',
        verified_phase_count=sum(bool(r.get('phase_verified')) for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    with args.output.open('x') as f:json.dump(discover(args.root),f,indent=2)
