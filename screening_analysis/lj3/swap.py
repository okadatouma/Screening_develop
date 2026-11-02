"""同じ網で、WC 型と AHC 型の配置（ウォーカー）を比べる。網を固定すれば、違いは配置の違いだけから来る。
特徴：擬スピンの組（++, +−, −+, −−）ごとの窓つきの重み Σ α m / Σ m（ヘッド平均と、ヘッドごと）。
差 Δ = 特徴(AHC 型の配置) − 特徴(WC 型の配置) を、2 つの λ × 2 つの網（WC 型の網、AHC 型の網）で比べる。"""
import json, numpy as np, jax, jax.numpy as jnp
import lj3net as LN, readout3 as RO

def feats(run_net, R, S, nw=1024):
    internals, _ = RO.make_parts(run_net)
    fi = jax.jit(jax.vmap(internals))
    off = ~np.eye(36, dtype=bool)
    num = None; den = None
    for b in range(0, nw, 64):
        Sb = S[b:b + 64]
        I = fi(jnp.asarray(R[b:b + 64], jnp.float32), jnp.asarray(Sb, jnp.float32))
        al = np.asarray(I['al'], np.float64); m = np.asarray(I['m'], np.float64)
        n_ = []; d_ = []
        for sa in (1, -1):
            for sb in (1, -1):
                msk = ((Sb[:, :, None] == sa) & (Sb[:, None, :] == sb) & off[None])[:, None]
                n_.append((al * m * msk).sum((0, 2, 3))); d_.append((m * msk).sum((0, 2, 3)))
        n_ = np.stack(n_, 1); d_ = np.stack(d_, 1)                          # [H, 4]
        num = n_ if num is None else num + n_; den = d_ if den is None else den + d_
    return num / den                                                      # [H, 4]

out = {}
for gen in ('lj2', 'lj3'):
    for lam in ('0.33', '0.72'):
        tags = {typ: (t2 if gen == 'lj2' else t3) for l, typ, t2, t3 in LN.RUNS if l == lam}
        runs = {typ: LN.load(lam, tags[typ]) for typ in ('WC', 'AHC')}
        for net_typ in ('WC', 'AHC'):
            F = {cfg_typ: feats(runs[net_typ], runs[cfg_typ]['R'], runs[cfg_typ]['S']) for cfg_typ in ('WC', 'AHC')}
            D = F['AHC'] - F['WC']                                         # [H, 4]
            key = f'{gen}_{lam}_net{net_typ}'
            out[key] = dict(D=D.tolist(), F_WC=F['WC'].tolist(), F_AHC=F['AHC'].tolist())
            dm = D.mean(0)
            print(f"{gen} λ={lam} 網={net_typ:3s}：配置 AHC − WC（ヘッド平均）  ++ {dm[0]:+.4f}  +− {dm[1]:+.4f}  −+ {dm[2]:+.4f}  −− {dm[3]:+.4f}", flush=True)
    keys = [k for k in out if k.startswith(gen)]
    V = np.array([np.array(out[k]['D']).ravel() for k in keys])
    C = V @ V.T / np.outer(np.linalg.norm(V, axis=1), np.linalg.norm(V, axis=1))
    print(f"{gen}：差 Δ（ヘッド × 組 の 64 個）の余弦  " + "  ".join(f"{keys[i][4:]}~{keys[j][4:]} {C[i, j]:+.2f}" for i in range(4) for j in range(i + 1, 4)), flush=True)
json.dump(out, open('swap.json', 'w'), indent=1)
