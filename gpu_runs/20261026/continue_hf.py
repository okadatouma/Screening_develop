import json,hashlib
from pathlib import Path
import numpy as np
from periodicwave.pbc.lj_hf import LJHF
out=Path(__file__).parent/'hf'
for kind,lams,target in [('wc',[.4,.48,.56,.64],0),('ahc',[.68,.64],1)]:
 c=np.load(out/f'{kind}_coeff.npy')
 for lam in lams:
  hf=LJHF(lam,15.,L=6,gcut=2.01);D=np.einsum('ka,kb->kab',c,c.conj());r=hf.solve(D,maxit=1000,mix=.5,tol=1e-7)
  C=float(hf.chern(r['c']));res=float(r['hist'][-1][1]);ok=res<1e-6 and abs(C-target)<.1
  info=dict(kind=kind,n=36,rs=15.,lam=lam,gcut=2.01,C=C,residual=res,converged=ok,used='scf' if ok else 'not_exported')
  print(json.dumps(info),flush=True)
  if not ok:break
  c=r['c']
  if lam==.64:
   p=out/f'hf_N36_lam{lam:.2f}_{kind}.txt'
   with p.open('x') as f:np.savetxt(f,hf.orbitals_text(c),fmt='%.17e')
   info['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(out/f'{kind}_lam064_metadata.json').write_text(json.dumps(info,indent=2))
