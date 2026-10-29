import json
from pathlib import Path
base=Path(__file__).resolve().parent
jobs=[]
common=dict(PW_OPTIMIZER='adam',PW_LR='.002',PW_LR_DELAY='20000',PW_BATCH='64',PW_EL_CHUNK='8',PW_FLIP_BATCH='6',PW_HEADS='2',PW_ATTN_DIM='8',PW_VALUE_DIM='8',PW_WIDTH='32',PW_DET='1',PW_LAYERS='1',PW_MCMC_STEPS='10',PW_BURN_IN='256',PW_SAVE_MIN='15',PW_WIN_SHAPE='smooth7',PW_TRIM_POW='4',PW_CHECK_NAN='1',PW_TOPO_SAMPLES='2',PW_LAPL='folx',PW_SEED_FROM_TAG='0')
for lam,kind,modes in [(.33,'wc',['off']),(.72,'ahc',['off']),(.64,'wc',['off','persistence','laplacian','weights']),(.64,'ahc',['off','persistence','laplacian','weights'])]:
 for mode in modes:
  for seed in [101,211,307]:
   jobs.append(dict(id=f'N36-lam{lam:.2f}-{kind}-{mode}-seed{seed}',n=36,rs=15,lam=lam,seed=seed,iterations=80000,platform='cuda,cpu',initial_hf_kind=kind,phase_verified=False,env={**common,'PW_LJ_HF':str(base/'hf'/f'hf_N36_lam{lam:.2f}_{kind}.txt'),'PW_TOPO_MODE':mode,'PW_TOPO_GAMMA':'0' if mode=='off' else '.1'}))
(base/'plan.json').write_text(json.dumps(dict(description='80000 updates; 3 independent seeds; HF starting phases do not certify optimized VMC phases',jobs=jobs),indent=2))
print(len(jobs),'jobs')
