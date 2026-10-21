import numpy as np, pandas as pd, glob, os
base = 'results/lambda-jellium/CustomScreenformer'
conv = (2 / 15**2) / 36 * 1000   # code units (36 electrons) -> mRy per electron
rows = []
for lam in ['0.33', '0.4', '0.48', '0.56', '0.64', '0.72']:
    for typ in ['cw', 'ca']:
        tag = typ + {'0.33':'33','0.4':'40','0.48':'48','0.56':'56','0.64':'64','0.72':'72'}[lam]
        d = f'{base}/N36_rs15_lam{lam}_{tag}'
        s = pd.read_csv(f'{d}/train_stats.csv')
        n = len(s)
        def blk(a, bs=500):
            m = len(a) // bs
            b = a[:m * bs].reshape(m, bs).mean(1)
            return a.mean(), b.std(ddof=1) / np.sqrt(m)
        last = s[s.step >= s.step.max() - 10000]
        prev = s[(s.step >= s.step.max() - 20000) & (s.step < s.step.max() - 10000)]
        e, de = blk(last.energy.values)
        ep, dep = blk(prev.energy.values)
        rows.append(dict(lam=float(lam), type='WC' if typ == 'cw' else 'AHC', tag=tag, steps=int(s.step.max()),
                         E=e, dE=de, E_prev=ep, drift=e - ep, sigma=last.locstd.mean(),
                         nminus=last.nminus.mean() / 36, pflip=last.pflip.mean(), pmove=last.pmove.mean(),
                         eimag=last.eimag.mean(), epot=last.epot.mean(), E0=s.energy.values[:50].mean()))
r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r[['lam','type','steps','E0','E','dE','drift','sigma','nminus','pflip','eimag']].round(3).to_string(index=False))
print()
for lam in sorted(r.lam.unique()):
    w = r[(r.lam == lam) & (r.type == 'WC')].iloc[0]; a = r[(r.lam == lam) & (r.type == 'AHC')].iloc[0]
    d = a.E - w.E; dd = np.hypot(a.dE, w.dE)
    print(f"lam {lam:4.2f}: E_AHC - E_WC = {d:+7.3f} ± {dd:.3f} (code)  = {d*conv:+.3f} ± {dd*conv:.3f} mRy/e   lower: {'WC' if d > 0 else 'AHC'}")
r.to_csv('summary_energies.csv', index=False)
