# 既存lj2/readoutの内部量をまとめ、同符号/異符号対の類似度・重み・Head寄与を集計する旧解析入口。
# 正式Diagnosticへ移行済みの主要3入口とは別であり、ここでのDeltaは新しいcoreのΔCとは異なる。
# 結果はreadouts.jsonへ保存する。
import json, numpy as np, jax, jax.numpy as jnp, time
import lj2net as LN, readout as RO
NW = 512
out = {}
for lam in LN.LAMS:
    for typ in ['cw', 'ca']:
        t0 = time.time()
        run = LN.load(lam, typ)
        internals, dref = RO.make_parts(run)
        f = jax.jit(jax.vmap(internals))
        acc = None
        for b in range(0, NW, 64):
            I = f(jnp.asarray(run['R'][b:b + 64], jnp.float32), jnp.asarray(run['S'][b:b + 64], jnp.float32))
            s, m, wgt = I['s'], I['m'], I['wgt']
            same, opp = I['same'][:, None], I['opp'][:, None]                  # [B, 1, i, j]
            d = dict(
                s_same_m=(s * m * same).sum((0, 2, 3)), m_same=(m * same).sum((0, 2, 3)),
                s_opp_m=(s * m * opp).sum((0, 2, 3)), m_opp=(m * opp).sum((0, 2, 3)),
                s_same=(s * same).sum((0, 2, 3)), n_same=(same + 0 * s).sum((0, 2, 3)),
                s_opp=(s * opp).sum((0, 2, 3)), n_opp=(opp + 0 * s).sum((0, 2, 3)),
                w_same=(wgt * same).sum((0, 2, 3)), w_opp=(wgt * opp).sum((0, 2, 3)),
                A_same=(I['a_same'] ** 2).sum(-1).sum((0, 1)), A_opp=(I['a_opp'] ** 2).sum(-1).sum((0, 1)),
                A_self=(I['a_self'] ** 2).sum(-1).sum((0, 1)),
                cp2=(I['c_p'] ** 2).sum(-1).sum((0, 1)),                             # [h]
                cpsum2=((I['c_p'].sum(2)) ** 2).sum(-1).sum((0, 1)),
                call2=(I['c_all'] ** 2).sum(-1).sum((0, 1)),
            )
            d = {k: np.asarray(v, np.float64) for k, v in d.items()}
            acc = d if acc is None else {k: acc[k] + d[k] for k in acc}
        r = np.asarray(I['r'][0]); w = np.asarray(I['w'][0])
        res = dict(
            w=w.tolist(), r=r.tolist(),
            Delta_m=(acc['s_same_m'] / acc['m_same'] - acc['s_opp_m'] / acc['m_opp']).tolist(),
            Delta=(acc['s_same'] / acc['n_same'] - acc['s_opp'] / acc['n_opp']).tolist(),
            chi=((acc['w_opp'] / acc['m_opp']) / (acc['w_same'] / acc['m_same'])).tolist(),
            f_opp=(acc['A_opp'] / (acc['A_opp'] + acc['A_same'])).tolist(),
            q=(acc['cp2'] / acc['cp2'].sum()).tolist(),
            rho_p=float(acc['cpsum2'] / acc['cp2'].sum()),
            p_share=(acc['cp2'] / acc['call2']).tolist(),
        )
        out[f'{lam}_{typ}'] = res
        print(f"lam {lam:4s} {typ}: <Delta_m> = {np.mean(res['Delta_m']):+.4f}  <chi> = {np.mean(res['chi']):.3f}  "
              f"<f_opp> = {np.mean(res['f_opp']):.3f}  rho_p = {res['rho_p']:.3f}  r_h = {np.min(r):.3f}-{np.max(r):.3f}  ({time.time() - t0:.0f}s)", flush=True)
json.dump(out, open('readouts.json', 'w'), indent=1)
