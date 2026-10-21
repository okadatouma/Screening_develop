import numpy as np, json, sys
from scipy.optimize import minimize
from c3hf import load_table, c3_value, find_centers, A
def density(k, G, c, lam, r):
    q = k[:, None, :] + G[None, :, :]
    nr = 1 / np.sqrt(1 + lam**2 * np.sum(q**2, -1)); qp = q[..., 0] + 1j * q[..., 1]
    ph = np.exp(1j * np.einsum('d,kgd->kg', r, q))
    up = (ph * c * nr).sum(1); dn = (ph * c * lam * qp * nr).sum(1)
    return float((abs(up)**2).sum()), float((abs(dn)**2).sum())
res = {}
for lam in ['0.33', '0.40', '0.48', '0.56', '0.64', '0.72']:
    for kind in ['wc', 'ahc']:
        k, G, c = load_table(f'/home/claude/ljhf/hf_N36_lam{lam}_{kind}.txt')
        top = find_centers(k, G, c, n=24)[:3]
        cents = []
        for val, f, r in top:
            o = minimize(lambda x: -abs(c3_value(k, G, c, A @ x)), f, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-10))
            fr = o.x % 1.0; rr = A @ fr
            v = c3_value(k, G, c, rr)
            nu, nd = density(k, G, c, float(lam), rr)
            cents.append(dict(frac=fr.tolist(), r=rr.tolist(), absM=abs(v), phase_over_2pi3=float(np.angle(v) / (2 * np.pi / 3)), n=nu + nd, nm_frac=nd / (nu + nd)))
        cents.sort(key=lambda d: -d['n'])          # 密度の高い順（いちばん上が格子点）
        res[f'{lam}_{kind}'] = cents
        print(f"lam {lam} {kind:3s}: " + "  |  ".join(
            f"f=({d['frac'][0]:.3f},{d['frac'][1]:.3f}) |M|={d['absM']:.4f} arg/(2pi/3)={d['phase_over_2pi3']:+.3f} n={d['n']:.3f} n-/n={d['nm_frac']:.2f}" for d in cents), flush=True)
json.dump(res, open('c3_hf_centers.json', 'w'), indent=1)
