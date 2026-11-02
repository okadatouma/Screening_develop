"""同じ λ の WC 型と AHC 型の差の「向き」が λ = 0.33 と 0.72 で揃うか（アテンション側の量を全部まとめて）。
特徴：ウォーカーごと・ヘッドごとの、擬スピンの組（++, +−, −+, −−）で分けた窓つきの重み Σ α m / Σ m（16 ヘッド × 4 = 64 個）。
(1) 差のベクトル Δ_λ = 平均_AHC − 平均_WC の、2 つの λ の間の余弦（ウォーカーのブートストラップで 95% 区間）。
(2) λ = 0.33 で LDA（w = Σ^{-1} Δ）を作り、λ = 0.72 の WC 型・AHC 型のウォーカーを w·x で分けたときの AUC（逆も）。0.5 なら向きが揃わない。"""
import json, numpy as np, jax, jax.numpy as jnp
import lj3net as LN, readout3 as RO

def features(lam, tag, nw=1024):
    run = LN.load(lam, tag)
    internals, _ = RO.make_parts(run)
    fi = jax.jit(jax.vmap(internals))
    R, S = run['R'][:nw], run['S'][:nw]
    off = ~np.eye(36, dtype=bool)
    F = []
    for b in range(0, nw, 64):
        Sb = S[b:b + 64]
        I = fi(jnp.asarray(R[b:b + 64], jnp.float32), jnp.asarray(Sb, jnp.float32))
        al = np.asarray(I['al'], np.float64); m = np.asarray(I['m'], np.float64)
        cols = []
        for sa in (1, -1):
            for sb in (1, -1):
                msk = ((Sb[:, :, None] == sa) & (Sb[:, None, :] == sb) & off[None])[:, None]
                cols.append((al * m * msk).sum((2, 3)) / np.maximum((m * msk).sum((2, 3)), 1e-12))   # [B, H]
        F.append(np.concatenate(cols, 1))
    return np.concatenate(F, 0)                                          # [W, 4H]（並び：++, +−, −+, −− の各 16 ヘッド）

def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x)) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

rng = np.random.default_rng(0)
out = {}
for gen, ti in (('lj2', 2), ('lj3', 3)):
    X = {}
    for lam, typ, t2, t3 in LN.RUNS:
        X[(lam, typ)] = features(lam, t2 if gen == 'lj2' else t3)
    D = {lam: X[(lam, 'AHC')].mean(0) - X[(lam, 'WC')].mean(0) for lam in ('0.33', '0.72')}
    cos = float(D['0.33'] @ D['0.72'] / np.linalg.norm(D['0.33']) / np.linalg.norm(D['0.72']))
    bs = []
    for _ in range(200):
        Db = {}
        for lam in ('0.33', '0.72'):
            a = X[(lam, 'AHC')]; w = X[(lam, 'WC')]
            Db[lam] = a[rng.integers(0, len(a), len(a))].mean(0) - w[rng.integers(0, len(w), len(w))].mean(0)
        bs.append(Db['0.33'] @ Db['0.72'] / np.linalg.norm(Db['0.33']) / np.linalg.norm(Db['0.72']))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res = dict(cos=cos, cos_ci=[float(lo), float(hi)])
    for tr, te in (('0.33', '0.72'), ('0.72', '0.33')):
        a, w = X[(tr, 'AHC')], X[(tr, 'WC')]
        C = 0.5 * (np.cov(a.T) + np.cov(w.T)) + 1e-6 * np.eye(a.shape[1])
        v = np.linalg.solve(C, a.mean(0) - w.mean(0))
        res[f'auc_train{tr}_on{tr}'] = float(auc(a @ v, w @ v))
        res[f'auc_train{tr}_on{te}'] = float(auc(X[(te, 'AHC')] @ v, X[(te, 'WC')] @ v))
    # 擬スピンの組の重みだけ（ヘッドで平均した 4 個）でも
    for tr, te in (('0.33', '0.72'),):
        hm = lambda Z: Z.reshape(len(Z), 4, -1).mean(2)
        a, w = hm(X[(tr, 'AHC')]), hm(X[(tr, 'WC')])
        C = 0.5 * (np.cov(a.T) + np.cov(w.T)) + 1e-9 * np.eye(4)
        v = np.linalg.solve(C, a.mean(0) - w.mean(0))
        res['auc4_train0.33_on0.72'] = float(auc(hm(X[(te, 'AHC')]) @ v, hm(X[(te, 'WC')]) @ v))
    out[gen] = res
    print(gen, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in res.items()}, flush=True)
json.dump(out, open('transfer.json', 'w'), indent=1)
