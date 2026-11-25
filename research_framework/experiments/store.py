"""Crash-safe run storage. Old checkpoints and input files are never overwritten."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import time
import traceback
import shutil
import numpy as np
from ..paths import ROOT,lattice_source,wannier_source


def serial(value):
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value


def digest(data):return hashlib.sha256(data).hexdigest()


def json_save(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if not replace:
        with path.open('x') as f:f.write(text)
        return
    tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def csv_save(path,rows,replace=False):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    target=path.with_suffix(f'.tmp.{os.getpid()}') if replace else path
    with target.open('w' if replace else 'x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(serial(rows))
    if replace:target.replace(path)


def source_hashes():
    sources=[]
    for parent,dirs,files in os.walk(ROOT/'research_framework'):
        dirs[:]=[d for d in dirs if d not in ['.venv','results','__pycache__','.pytest_cache']]
        sources.extend(Path(parent)/f for f in files if f.endswith('.py'))
    sources += [ROOT/p for p in ['periodicwave/CustomScreenformer.py','periodicwave/attention.py','periodicwave/network_blocks.py','periodicwave/lj_orbitals.py','periodicwave/pbc/lambda_jellium.py']]
    for project,files in [(lattice_source(),['lattice/model.py','lattice/physics.py','lattice/window_research.py','lattice/vmc.py']),
                          (wannier_source(),['wannier_ph/geometry.py','wannier_ph/persistence.py','wannier_ph/wannier.py','wannier_ph/transformer.py','vendor/berry_validation.py'])]:
        sources += [project/p for p in files]
    return {str(p):digest(p.read_bytes()) for p in sorted(sources)}


def environment():
    versions={}
    for name in ['numpy','scipy','jax','jaxlib','optax','torch','ripser','gudhi','matplotlib','PyYAML','folx','kfac-jax']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    import jax
    return dict(python=platform.python_version(),packages=versions,timezone='Asia/Tokyo',jax_enable_x64=bool(jax.config.jax_enable_x64),
                backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                note='Execution backend and Slurm allocation are recorded; existing runs/jobs are not modified')


class RunStore:
    def __init__(self,path,config,resume=False):
        self.path=Path(path);self.config=config
        self.code=source_hashes();self.config_hash=digest(json.dumps(config,sort_keys=True).encode())
        if self.path.exists():
            if not resume:raise FileExistsError(f'choose a new run or --resume: {self.path}')
            if (self.path/'config.json').read_text()!=json.dumps(serial(config),ensure_ascii=False,indent=2)+'\n':raise ValueError('resume config differs')
            if json.loads((self.path/'source_hashes.json').read_text())!=self.code:raise ValueError('resume source changed; preserve this run and create a new one')
        else:
            self.path.mkdir(parents=True,exist_ok=False)
            json_save(self.path/'config.json',config);json_save(self.path/'source_hashes.json',self.code)
            for source in self.code:
                p=Path(source);destination=self.path/'source'/p.relative_to(ROOT.parent)
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
            json_save(self.path/'environment.json',environment())
            rev=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
            (self.path/'revision.txt').write_text(rev.stdout)
            (self.path/'dirty.patch').write_text(subprocess.run(['git','diff','--','periodicwave/CustomScreenformer.py'],cwd=ROOT,capture_output=True,text=True).stdout)
        (self.path/'checkpoints').mkdir(exist_ok=True)
        (self.path/'diagnostics').mkdir(exist_ok=True)
        self.attempt_started=time.time()

    def save_checkpoint(self,step,state):
        import jax
        path=self.path/'checkpoints'/f'step_{step:07d}.pkl'
        if path.exists():raise FileExistsError(path)
        def host(value):
            # torch.Tensor -> numpy would make optimizer/RNG restoration invalid.
            return value.detach().cpu().clone() if hasattr(value,'detach') else jax.device_get(value)
        payload=dict(format='research-framework-v1',config_hash=self.config_hash,source_hashes=self.code,
                     step=int(step),state=jax.tree_util.tree_map(host,state))
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:pickle.dump(payload,f,protocol=5);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
        json_save(self.path/'latest.json',dict(checkpoint=str(path.relative_to(self.path)),step=int(step)),replace=True)

    def restore(self,path=None):
        path=Path(path) if path else self.path/json.loads((self.path/'latest.json').read_text())['checkpoint']
        # Trusted local framework checkpoint only; never fetch/unpickle external inputs.
        with path.open('rb') as f:payload=pickle.load(f)
        if payload.get('format')!='research-framework-v1' or payload['config_hash']!=self.config_hash or payload['source_hashes']!=self.code:raise ValueError('checkpoint config/source identity mismatch')
        return payload['step'],payload['state']

    def log(self,row):
        with (self.path/'history.jsonl').open('a') as f:f.write(json.dumps(serial(row),allow_nan=False)+'\n');f.flush()

    def trim_history(self,step):
        path=self.path/'history.jsonl'
        if path.exists():
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            retained=[r for r in rows if r['step']<=step]
            tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in retained));tmp.replace(path)

    def fail(self,error):
        row=dict(time=time.time(),seconds=time.time()-self.attempt_started,error=repr(error),traceback=traceback.format_exc())
        with (self.path/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def complete(self,summary):
        json_save(self.path/'summary.json',summary)
        json_save(self.path/'COMPLETE.json',dict(seconds=time.time()-self.attempt_started,config_hash=self.config_hash))

    def input(self,**arrays):
        path=self.path/'inputs.npz'
        if path.exists():
            old=np.load(path)
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],np.asarray(v)) for k,v in arrays.items()):raise ValueError('input data changed on resume')
            return
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        json_save(self.path/'input_hash.json',dict(sha256=digest(path.read_bytes()),arrays={k:list(np.shape(v)) for k,v in arrays.items()}))
