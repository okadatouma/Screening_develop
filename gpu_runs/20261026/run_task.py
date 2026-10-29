import csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from screening_analysis.topology.workflow import provenance
base=Path(__file__).resolve().parent
idx=int(sys.argv[1]); cfg=json.loads((base/'plan.json').read_text())['jobs'][idx]
hf=Path(cfg['env']['PW_LJ_HF']);kind=cfg['initial_hf_kind'];meta=base/'hf'/(f'{kind}_lam064_metadata.json' if cfg['lam']==.64 else f'{kind}_metadata.json')
m=json.loads(meta.read_text())
assert m['converged'] and abs(m['lam']-cfg['lam'])<1e-12
assert hashlib.sha256(hf.read_bytes()).hexdigest()==m['sha256']
job=base/'production'/cfg['id'];job.mkdir(parents=True,exist_ok=False)
manifest=job/'training.json';manifest.write_text(json.dumps({'jobs':[cfg]},indent=2))
env=os.environ.copy();env.update(JAX_ENABLE_X64='1',JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
subprocess.run([sys.executable,'-u','-m','screening_analysis.topology.cli','train','--manifest',str(manifest),'--output',str(job/'training')],env=env,check=True)
run=job/'training'/cfg['id']/'run'
with (run/'train_stats.csv').open() as f:rows=list(csv.DictReader(f))
e=np.array([float(r['energy']) for r in rows]);tail=e[-20000:];blocks=tail[:len(tail)//500*500].reshape(-1,500).mean(1)
a,b=np.array_split(blocks,2);sem=lambda x:float(np.std(x,ddof=1)/np.sqrt(len(x)))
err=float(np.hypot(sem(a),sem(b)));drift=float(b.mean()-a.mean())
summary=dict(updates=len(e),finite=bool(np.isfinite(e).all()),tail_mean=float(tail.mean()),block_size=500,block_sem=sem(blocks),half_drift=drift,half_drift_sem=err,stationarity_flag=bool(abs(drift)<3*err),note='Training-batch stationarity diagnostic only; not independent ground-state or Chern certification',provenance=provenance(cfg['seed']))
(job/'convergence.json').write_text(json.dumps(summary,indent=2))
assert len(e)==80000 and np.isfinite(e).all()
analysis=dict(seed=cfg['seed'],null_repeats=1,runs=[dict(id=cfg['id'],folder=str(run),config_path=str(run/'runtime_config.json'),nconf=8,group=f'seed{cfg["seed"]}',phase='unknown',phase_verified=False,initial_hf_kind=kind)])
p=job/'analysis.json';p.write_text(json.dumps(analysis,indent=2))
subprocess.run([sys.executable,'-u','-m','screening_analysis.topology_run',str(p),'--output',str(job/'topology')],env=env,check=True)
(job/'finished.json').write_text(json.dumps(dict(training_and_topology_complete=True)))
