"""q·k（コサイン類似度 s）と内容の重み α が、相手の位置 r_j の関数としてどの周期の成分をもつか。
入力は (sin 2πu, cos 2πu, s)（u はセル（6a）の分数座標）。r_j をセル全体の 96×96 の格子で動かして FFT し、
結晶の周期の成分（指数 n が 6 の倍数）と 1 次の成分（|n1|,|n2|,|n1+n2| ≤ 1）の割合を出す（一様な成分は除く）。"""
import sys, json
sys.path.insert(0, '/home/claude/lj2res/ana')
import numpy as np, jax, jax.numpy as jnp
import lj2net as LN
from periodicwave import CustomScreenformer as CS

NG = 96
def feats(u, s):
    return np.concatenate([np.sin(2*np.pi*u), np.cos(2*np.pi*u), np.full(u.shape[:-1] + (1,), s)], -1)

g = (np.arange(NG) + 0.5) / NG
U = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)          # (NG², 2)
n = np.fft.fftfreq(NG, 1.0 / NG).astype(int)
N1, N2 = np.meshgrid(n, n, indexing='ij')
cryst = (N1 % 6 == 0) & (N2 % 6 == 0) & ~((N1 == 0) & (N2 == 0))
first = (np.maximum(np.abs(N1), np.abs(N2)) <= 1) & ~((N1 == 0) & (N2 == 0))

def shares(field):                                                         # field: (NG, NG)
    F = np.fft.fft2(field - field.mean())
    p = np.abs(F) ** 2; tot = p.sum()
    if tot < 1e-20 * NG**4:                                                # 定数（Trim で全部 0 など）は除く
        return (np.nan, np.nan)
    return p[cryst].sum() / tot, p[first].sum() / tot

out = {}
for lam, typ in [(l, t) for l in LN.LAMS for t in ('cw', 'ca')]:
    run = LN.load(lam, typ)
    P = run['params']['layers']; ap = P['attention'][0]
    NH, AD = 16, 16
    r = np.asarray(CS.TRIM_MAX * jax.nn.sigmoid(ap['trim_rho']))
    def qk(u, s):
        x = feats(u, s) @ np.asarray(P['embed'])
        q = (x @ np.asarray(ap['q_w'])).reshape(-1, NH, AD); k = (x @ np.asarray(ap['k_w'])).reshape(-1, NH, AD)
        return q / np.linalg.norm(q, axis=-1, keepdims=True), k / np.linalg.norm(k, axis=-1, keepdims=True)
    rng = np.random.default_rng(0)
    ui = rng.random((8, 2))                                                # 8 個の i の位置（ランダム）
    res = {'s': [], 'alpha': [], 's_query': []}
    for sj in (1, -1):                                                     # 自分の位置 r_i を動かす側（相手は固定）
        _, kj = qk(ui, sj)
        for si in (1, -1):
            qi, _ = qk(U, si)
            s = np.einsum('ihd,jhd->jhi', qi, kj)                          # (8, NH, NG²)
            for a in range(8):
                for h in range(NH):
                    res['s_query'].append(shares(s[a, h].reshape(NG, NG)))
    for si in (1, -1):
        qi, _ = qk(ui, si)                                                 # (8, NH, AD)
        for sj in (1, -1):
            _, kj = qk(U, sj)                                              # (NG², NH, AD)
            s = np.einsum('ihd,jhd->ihj', qi, kj)                          # (8, NH, NG²)
            al = (np.maximum(s - (1 - r[None, :, None]), 0) / r[None, :, None]) ** 4
            for a in range(8):
                for h in range(NH):
                    res['s'].append(shares(s[a, h].reshape(NG, NG)))
                    res['alpha'].append(shares(al[a, h].reshape(NG, NG)))
    for key in res:
        v = np.array(res[key]); v = v[~np.isnan(v[:, 0])]
        out[f'{lam}_{typ}_{key}'] = dict(cryst_mean=float(v[:, 0].mean()), cryst_max=float(v[:, 0].max()),
                                         first_mean=float(v[:, 1].mean()), first_min=float(v[:, 1].min()), n=len(v))
        print(lam, typ, key.ljust(7), 'crystal-period share: mean %.1e max %.1e | first-harmonic share: mean %.3f min %.3f  (n=%d)'
              % (v[:, 0].mean(), v[:, 0].max(), v[:, 1].mean(), v[:, 1].min(), len(v)))
json.dump(out, open('/home/claude/lj2res/ana2/harmonics.json', 'w'), indent=1)
