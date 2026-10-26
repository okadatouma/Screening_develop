"""Thin orchestration of existing diagnostics; no additional numerical definitions."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    from screening_analysis.topology_run import json_safe
    Path(path).write_text(json.dumps(json_safe(value),indent=2,ensure_ascii=False,allow_nan=False))


def provenance(seed=None):
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    versions={}
    for name in ('numpy','scipy','jax','jaxlib','folx','optax','kfac-jax','ml-collections','PyYAML'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    files=[p for folder in ('screening_analysis','periodicwave') for p in (ROOT/folder).rglob('*.py')]
    return dict(utc=datetime.now(timezone.utc).isoformat(),git_commit=git('rev-parse','HEAD'),
        git_branch=git('branch','--show-current'),git_status=git('status','--porcelain'),
        tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        python=sys.version,platform=platform.platform(),libraries=versions,seed=seed,
        environment={k:v for k,v in os.environ.items() if k.startswith(('PW_','JAX_','CUDA_VISIBLE_DEVICES'))})


def manifest(path):
    path=Path(path).resolve();text=path.read_text()
    try:data=json.loads(text)
    except json.JSONDecodeError:
        try:import yaml
        except ImportError as exc:raise ValueError('Use JSON (valid YAML 1.2), or install PyYAML for other YAML syntax') from exc
        data=yaml.safe_load(text)
    if not isinstance(data,dict):raise ValueError('manifest must be a mapping')
    return data,path.parent


def simple_id(identifier):
    if not isinstance(identifier,str) or Path(identifier).name!=identifier or identifier in ('','.','..'):
        raise ValueError('unique simple job ids required')
    return identifier


def benchmark(path,output,resume=False):
    from screening_analysis.topology_cde_run import load_probe,analyze as cde
    from screening_analysis.topology_run import analyze_probe
    from screening_analysis import phase_metrics
    import numpy as np
    data,base=manifest(path)
    for key,default in [('nconf',8),('null_repeats',1)]:
        if type(data.get(key,default)) is not int or data.get(key,default)<1:raise ValueError(key+' must be a positive integer')
    entries=data.get('runs',[])
    for entry in entries:
        if entry.get('phase') in ('WC','AHC') and not (entry.get('phase_verified') is True and entry.get('phase_evidence')):
            raise ValueError('WC/AHC needs explicit independent evidence and phase_verified=true')
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    ids=[simple_id(r['id']) for r in entries]
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('nonempty unique runs required')
    provenance_path=out/'provenance.json'
    if not provenance_path.exists():write(provenance_path,provenance(data.get('seed',20261026)))
    summaries=[];cdes=[]
    for entry in entries:
        source=(base/entry['probe']).resolve();signature=dict(entry=entry,probe_sha256=sha(source),
            nconf=data.get('nconf',8),null_repeats=data.get('null_repeats',1),
            numerical_sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'screening_analysis/topology/persistence.py',
                ROOT/'screening_analysis/topology_run.py',ROOT/'screening_analysis/topology_cde_run.py',
                ROOT/'screening_analysis/multiscale_topology.py',ROOT/'screening_analysis/directed_loops.py',
                ROOT/'screening_analysis/hidden_topology.py',ROOT/'screening_analysis/graph_wind.py']})
        folder=out/entry['id'];done=folder/'complete.json'
        if done.exists():
            if not resume or json.loads(done.read_text())['signature']!=signature:raise ValueError('resume signature mismatch: '+entry['id'])
        elif folder.exists():
            raise ValueError('incomplete task retained: move it to a different name, then resume: '+str(folder))
        else:
            folder.mkdir();probe=load_probe(source);B=probe['S'].shape[0];indices=np.linspace(0,B-1,min(B,data.get('nconf',8))).astype(int)
            # Only arrays with known sample-leading keys are sliced; no accidental
            # slicing of the lattice or head parameter arrays when B=2.
            sample_keys=('S','Alpha','Window','W_window','W','d','positions','spins','q','k','q_normalized','k_normalized',
                'v','v_normalized','v_effective','pair_features','hidden_input','hidden_output','head_vectors','head_bias','head_gram','head_cancellation_ratio')
            for key in sample_keys:
                if key in probe:probe[key]=probe[key][indices]
            run={**entry,'source_probe':str(source),'selected_configuration_indices':indices.tolist(),
                 'phase':entry.get('phase','unknown'),'group':entry.get('group',str(entry.get('seed','unknown')))}
            a=analyze_probe(probe,folder/'AB',run,rng_seed=data.get('seed',20261026),null_repeats=data.get('null_repeats',1))
            c=cde(probe,folder/'CDE',run)
            write(done,dict(signature=signature,status='complete',physical_F='not executed here; separately validated artifacts in report',
                            checkpoint_sha256=probe['metadata'].get('checkpoint_sha256')))
            print(entry['id'],a['shape'],flush=True)
        summaries.append(json.loads((folder/'AB/summary.json').read_text()));cdes.append(json.loads((folder/'CDE/summary.json').read_text()))
    write(out/'summary.json',summaries);write(out/'cde-summary.json',cdes)
    # Pool repeated checkpoints per run/group and label before group holdout.
    labeled=[r for r in summaries if r['run'].get('phase_verified') and r['run'].get('phase') in ('WC','AHC')]
    results=[]
    for source in ('window','trim','screening','qk'):
        for control in ('raw','rank','shuffle'):
            units={}
            for r in labeled:
                a=next(v for v in r['aggregates'] if v['source']==source and v['control']==control)
                features=[a[k] for k in ('H0_total_persistence','H1_total_persistence','beta0_auc','beta1_auc','rank1_probability','rank2_probability')]
                units.setdefault((r['run']['group'],r['run']['phase']),[]).append(features)
            if (units and {k[1] for k in units}=={'WC','AHC'}
                    and all(len({k[0] for k in units if k[1]==label})>=3 for label in ('WC','AHC'))):
                keys=list(units)
                results.append(dict(source=source,control=control,**phase_metrics.evaluate(
                    [np.mean(units[k],axis=0) for k in keys],[k[1] for k in keys],[k[0] for k in keys])))
    write(out/'phase-classification.json',dict(status='exploratory_verified_labels' if results else '未検証: independently verified phase labels/groups unavailable',results=results))
    write(out/'manifest.json',data)
