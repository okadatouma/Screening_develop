"""VMC の状態の <C3> = E_{|Psi|^2}[ prod_i D_{s_i} Psi(R', S) / Psi(R, S) ]、R' = c + R3^{-1}(r - c)。
中心 c は出発点の HF の行列式の C3 中心（格子点）。HF の行列式は |<C3>| = 1。"""
import json, sys, numpy as np, jax, jax.numpy as jnp
import lj2net as LN
from c3hf import rot
cent = json.load(open('c3_hf_centers.json'))
th = 2 * np.pi / 3
Rinv = rot(-th)
out = {}
for lam in LN.LAMS:
    for typ in ['cw', 'ca']:
        run = LN.load(lam, typ)
        net, P = run['net'], run['params']
        key = f"{float(lam):.2f}_{'wc' if typ == 'cw' else 'ahc'}"
        res = []
        for ci, cen in enumerate(cent[key]):
            c = np.array(cen['r'])
            R = run['R'].reshape(-1, 36, 2); S = run['S']
            Rp = (c + (R - c) @ Rinv.T).reshape(-1, 72)
            f = jax.jit(jax.vmap(lambda x, s: net.apply(P, x, s, jnp.zeros((1, 2)), jnp.zeros((1,)))))
            ph0, la0 = f(jnp.asarray(run['R']), jnp.asarray(S))
            ph1, la1 = f(jnp.asarray(Rp), jnp.asarray(S))
            D = np.exp(1j * th * (S < 0).sum(1))
            ratio = D * np.exp(np.asarray(la1 - la0) + 1j * np.asarray(ph1 - ph0))
            m = ratio.mean(); err = ratio.std() / np.sqrt(len(ratio))
            res.append(dict(center=ci, n=cen['n'], val=[m.real, m.imag], err=float(err), absval=abs(m),
                            phase_over_2pi3=float(np.angle(m) / th)))
            print(f"lam {lam:4s} {typ}: 中心 {ci}（HF の n = {cen['n']:.1f}）  <C3> = {m.real:+.3f}{m.imag:+.3f}i ± {err:.3f}   "
                  f"|<C3>| = {abs(m):.3f}   arg/(2pi/3) = {np.angle(m)/th:+.3f}", flush=True)
        out[key] = res
json.dump(out, open('c3_vmc.json', 'w'), indent=1)
