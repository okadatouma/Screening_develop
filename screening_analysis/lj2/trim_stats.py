"""Trim が実際にどれだけ相手を切っていたか：s_ij < 1 - r_h になった (i, j, h) の割合と、窓の重みで数えた割合。"""
import sys
sys.path.insert(0, '/home/claude/lj2res/ana')
import numpy as np, jax, jax.numpy as jnp
import lj2net as LN, readout as RO
import periodicwave.CustomScreenformer as CS
out = {}
for lam in LN.LAMS:
    for typ in ('cw', 'ca'):
        run = LN.load(lam, typ)
        internals, _ = RO.make_parts(run)
        f = jax.jit(jax.vmap(internals))
        I = f(jnp.asarray(run['R'][:64], jnp.float32), jnp.asarray(run['S'][:64], jnp.float32))
        s = np.asarray(I["s"]); m = np.asarray(I["m"]); r = np.asarray(I["r"])[0]   # s: [B, h, i, j]
        thr = (1.0 - r)[None, :, None, None]
        off = ~np.eye(36, dtype=bool)[None, None]
        cut = (s < thr) & off
        frac = cut.sum() / off.sum() / s.shape[0] / s.shape[1] * s.shape[0] * s.shape[1] / 1.0
        frac = cut.mean() / off.mean()
        wfrac = (m * cut).sum() / (m * off).sum()
        out[f'{lam}_{typ}'] = (float(frac), float(wfrac), float(s[np.broadcast_to(off, s.shape)].min()), float(r.min()), float(r.max()))
        print(f'{lam:5s} {typ}  切られた対の割合 {frac:.2e}  窓の重みで {wfrac:.2e}  s の最小 {out[f"{lam}_{typ}"][2]:+.3f}  r_h {r.min():.3f}-{r.max():.3f}')
