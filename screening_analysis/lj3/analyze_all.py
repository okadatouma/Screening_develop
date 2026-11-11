"""lj2（同じ step の近く）と lj3 の 8 本：相の確認（C3 固有値、擬スピンの模様）とアテンション側の量。結果は ana_all.json。"""
import json, sys, time
import numpy as np, jax, jax.numpy as jnp
import lj3net as LN, readout3 as RO
from c3hf import rot

a = np.sqrt(2 * np.pi / np.sqrt(3))
A = np.array([[a, -a / 2], [0, a * np.sqrt(3) / 2]])          # 結晶の格子（列）
B = 2 * np.pi * np.linalg.inv(A).T
G6 = np.array([B[:, 0], B[:, 1], B[:, 0] + B[:, 1]]); G6 = np.concatenate([G6, -G6])
cent = json.load(open('c3_hf_centers.json'))
th = 2 * np.pi / 3
Rinv = rot(-th)
NW = 512
out = {}
for lam, typ, t2, t3 in LN.RUNS:
    for gen, tag in (('lj2', t2), ('lj3', t3)):
        t0 = time.time()
        run = LN.load(lam, tag)
        net, P = run['net'], run['params']
        R = run['R']; S = run['S']; W = R.shape[0]
        res = dict(lam=lam, type=typ, gen=gen, tag=tag, ckpt=run['ck'].split('/')[-1])
        # --- 擬スピンの模様（order.py と同じ）
        Rr = R.reshape(W, 36, 2)
        ph = np.exp(-1j * np.einsum('gd,wnd->wgn', G6, Rr))
        res['rhoG'] = float(np.abs(ph.sum(-1).mean(0)).mean() / 36)
        f = np.einsum('dk,wnk->wnd', np.linalg.inv(A), Rr) % 1.0
        nb = 18
        H, _, _ = np.histogram2d(f[..., 0].ravel(), f[..., 1].ravel(), bins=nb, range=[[0, 1], [0, 1]])
        Hm, _, _ = np.histogram2d(f[..., 0][S < 0], f[..., 1][S < 0], bins=nb, range=[[0, 1], [0, 1]])
        i, j = np.unravel_index(np.argmax(H), H.shape)
        sl = [((i + di) % nb, (j + dj) % nb) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        res['peak_nm'] = float(sum(Hm[p] for p in sl) / sum(H[p] for p in sl))
        res['nm'] = float((S < 0).mean())
        # --- C3 固有値（HF の行列式の C3 中心で）
        fn = jax.jit(jax.vmap(lambda x, s: net.apply(P, x, s, jnp.zeros((1, 2)), jnp.zeros((1,)))))
        ph0, la0 = fn(jnp.asarray(R, jnp.float32), jnp.asarray(S, jnp.float32))
        key = f"{float(lam):.2f}_{'wc' if typ == 'WC' else 'ahc'}"
        c3 = []
        for cen in cent[key]:
            c = np.array(cen['r'])
            Rp = (c + (Rr - c) @ Rinv.T).reshape(W, 72)
            ph1, la1 = fn(jnp.asarray(Rp, jnp.float32), jnp.asarray(S, jnp.float32))
            D = np.exp(1j * th * (S < 0).sum(1))
            ratio = D * np.exp(np.asarray(la1 - la0, np.float64) + 1j * np.asarray(ph1 - ph0, np.float64))
            m = ratio.mean()
            c3.append(dict(re=float(m.real), im=float(m.imag), err=float(ratio.std() / np.sqrt(W)), absval=float(abs(m)),
                           arg_over_2pi3=float(np.angle(m) / th), hf_n=cen['n']))
        res['c3'] = c3
        # --- アテンション側の量（pstype.py と run_readouts.py と同じ定義）
        internals, dref = RO.make_parts(run)
        fi = jax.jit(jax.vmap(internals))
        NH = run['cfg']['network']['CustomScreenformer']['num_heads']
        num = np.zeros((NH, 2, 2)); den = np.zeros((NH, 2, 2))
        acc = None
        off = ~np.eye(36, dtype=bool)
        for b in range(0, NW, 64):
            Sb = S[b:b + 64]
            I = fi(jnp.asarray(R[b:b + 64], jnp.float32), jnp.asarray(Sb, jnp.float32))
            al = np.asarray(I['al'], np.float64); m = np.asarray(I['m'], np.float64); s = np.asarray(I['s'], np.float64)
            for a_, sa in enumerate([1, -1]):
                for b_, sb in enumerate([1, -1]):
                    msk = ((Sb[:, :, None] == sa) & (Sb[:, None, :] == sb) & off[None])[:, None]
                    num[:, a_, b_] += (al * m * msk).sum((0, 2, 3))
                    den[:, a_, b_] += (m * msk).sum((0, 2, 3))
            wgt = np.asarray(I['wgt'], np.float64)
            same, opp = np.asarray(I['same'])[:, None], np.asarray(I['opp'])[:, None]
            d = dict(s_same_m=(s * m * same).sum((0, 2, 3)), m_same=(m * same).sum((0, 2, 3)),
                     s_opp_m=(s * m * opp).sum((0, 2, 3)), m_opp=(m * opp).sum((0, 2, 3)),
                     w_same=(wgt * same).sum((0, 2, 3)), w_opp=(wgt * opp).sum((0, 2, 3)),
                     A_same=(np.asarray(I['a_same'], np.float64) ** 2).sum(-1).sum((0, 1)),
                     A_opp=(np.asarray(I['a_opp'], np.float64) ** 2).sum(-1).sum((0, 1)),
                     cp2=(np.asarray(I['c_p'], np.float64) ** 2).sum(-1).sum((0, 1)),
                     cpsum2=((np.asarray(I['c_p'], np.float64).sum(2)) ** 2).sum(-1).sum((0, 1)),
                     call2=(np.asarray(I['c_all'], np.float64) ** 2).sum(-1).sum((0, 1)))
            acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
        Ah = num / den                                                   # [h, σ, σ']（σ = +1, −1）
        q = acc['cp2'] / acc['cp2'].sum()
        w = np.asarray(I['w'][0]); r = np.asarray(I['r'][0])
        res.update(A=Ah.tolist(), same=float(0.5 * (Ah[:, 0, 0] + Ah[:, 1, 1]).mean()), opp=float(0.5 * (Ah[:, 0, 1] + Ah[:, 1, 0]).mean()),
                   App=float(Ah[:, 0, 0].mean()), Amm=float(Ah[:, 1, 1].mean()), Apm=float(Ah[:, 0, 1].mean()), Amp=float(Ah[:, 1, 0].mean()),
                   q=q.tolist(), w=w.tolist(), r=r.tolist(), rho_p=float(acc['cpsum2'] / acc['cp2'].sum()), wq=float((q * w).sum()),
                   f_opp=(acc['A_opp'] / (acc['A_opp'] + acc['A_same'])).tolist(),
                   chi=((acc['w_opp'] / acc['m_opp']) / (acc['w_same'] / acc['m_same'])).tolist())
        res['diff'] = res['same'] - res['opp']
        res['asym_pm'] = res['App'] - res['Amm']                         # ++ と −− の差（lj2 では対称性で 0 に近い）
        out[f'{gen}_{tag}'] = res
        c = res['c3'][0]
        print(f"{gen} {lam} {typ:3s} {tag}: C3 = {c['re']:+.3f}{c['im']:+.3f}i（arg/(2π/3) {c['arg_over_2pi3']:+.2f}, |.| {c['absval']:.2f}）  山の n-/n {res['peak_nm']:.2f}  "
              f"++ {res['App']:.3f}  -- {res['Amm']:.3f}  +- {res['Apm']:.3f}  ++−−− {res['asym_pm']:+.4f}  rho_p {res['rho_p']:.3f}  <w>_q {res['wq']:.2f}  "
              f"r {r.min():.2f}-{r.max():.2f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open('ana_all.json', 'w'), indent=1)
print('ANALYZE_DONE')
