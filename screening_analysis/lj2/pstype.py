"""擬スピンの組ごとの内容の重み：A^h_{σσ'} = Σ α^h_ij m^h_ij [s_i=σ, s_j=σ'] / Σ m^h_ij [s_i=σ, s_j=σ']（j ≠ i）。
α^h_ij は q·k による重み（Trim の relu^4）、m^h_ij は距離の窓。窓の中で、相手の擬スピンごとにどれだけ見るか。
距離ごと（d_ij のビン）にも取る。"""
import json, numpy as np, jax, jax.numpy as jnp
import lj2net as LN, readout as RO
NW = 512
edges = np.linspace(0, 6.0, 25)
out = {}
for lam in LN.LAMS:
    for typ in ['cw', 'ca']:
        run = LN.load(lam, typ)
        internals, dref = RO.make_parts(run)
        f = jax.jit(jax.vmap(internals))
        num = np.zeros((16, 2, 2)); den = np.zeros((16, 2, 2))
        numd = np.zeros((2, 2, len(edges) - 1)); dend = np.zeros((2, 2, len(edges) - 1))
        for b in range(0, NW, 64):
            S = run['S'][b:b + 64]
            I = f(jnp.asarray(run['R'][b:b + 64], jnp.float32), jnp.asarray(S, jnp.float32))
            rr = np.asarray(I['r'])[:, :, None, None]; al = (np.maximum(np.asarray(I['s']) - (1.0 - rr), 0.0) / rr) ** 4   # α（Trim）
            m = np.asarray(I['m']); d = np.asarray(I['d'])
            off = ~np.eye(36, dtype=bool)
            for a_, sa in enumerate([1, -1]):
                for b_, sb in enumerate([1, -1]):
                    msk = ((S[:, :, None] == sa) & (S[:, None, :] == sb) & off[None])[:, None]   # [B,1,i,j]
                    num[:, a_, b_] += (al * m * msk).sum((0, 2, 3))
                    den[:, a_, b_] += (m * msk).sum((0, 2, 3))
                    # 距離ごと：ヘッドで平均した α（窓の重みつき）
                    mm = (m * msk).sum(1); aa = (al * m * msk).sum(1)            # [B,i,j]
                    idx = np.digitize(d, edges) - 1
                    for kbin in range(len(edges) - 1):
                        sel = idx == kbin
                        numd[a_, b_, kbin] += aa[sel].sum(); dend[a_, b_, kbin] += mm[sel].sum()
        A = num / den
        Ad = np.where(dend > 0, numd / np.maximum(dend, 1e-30), np.nan)
        out[f'{lam}_{typ}'] = dict(A=A.tolist(), Ad=Ad.tolist())
        Am = (num.sum(0) / den.sum(0))
        print(f"lam {lam:4s} {typ}: 窓で重みをつけた α の平均（全ヘッド）  ++ {Am[0,0]:.3f}  +- {Am[0,1]:.3f}  -+ {Am[1,0]:.3f}  -- {Am[1,1]:.3f}", flush=True)
json.dump(dict(edges=edges.tolist(), runs=out), open('pstype.json', 'w'))
