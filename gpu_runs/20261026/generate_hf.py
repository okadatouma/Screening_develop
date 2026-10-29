import json, hashlib, time
from pathlib import Path
import numpy as np
from periodicwave.pbc.lj_hf import LJHF
out=Path(__file__).parent/'hf'
for kind,lam,target in [('wc',.33,0),('ahc',.72,1)]:
    hf=LJHF(lam,15.,L=6,gcut=2.01)
    start=hf.seed_wc(.5) if kind=='wc' else hf.seed_random(np.random.default_rng(720))
    t=time.time();r=hf.solve(start,maxit=1000,mix=.5,tol=1e-7,verbose=True)
    C=float(hf.chern(r['c']));res=float(r['hist'][-1][1])
    p=out/f'hf_N36_lam{lam:.2f}_{kind}.txt'
    info=dict(kind=kind,n=36,L=6,rs=15.,lam=lam,gcut=2.01,C=C,target_C=target,residual=res,iterations=len(r['hist']),energy=float(r['E']),seconds=time.time()-t,converged=res<1e-6 and abs(C-target)<.1)
    if info['converged']:
        with p.open('x') as f:np.savetxt(f,hf.orbitals_text(r['c']),fmt='%.17e')
        info['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();np.save(out/f'{kind}_coeff.npy',r['c'])
    (out/f'{kind}_metadata.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
