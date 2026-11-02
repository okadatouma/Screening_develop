"""学習で各パラメータがどれだけ動いたか：|θ_最後 − θ_初期| / |θ_初期|。初期値は学習と同じ鍵（PRNGKey(42) を 1 回 split）で作り直す。"""
import json, numpy as np, jax, jax.numpy as jnp
import lj3net as LN
out = {}
for lam, typ, t2, t3 in LN.RUNS:
    for gen, tag in (('lj2', t2), ('lj3', t3)):
        run = LN.load(lam, tag)
        p0 = run['net'].init(jax.random.split(jax.random.PRNGKey(42))[1])
        p1 = run['params']
        L0, L1 = p0['layers'], p1['layers']
        a0, a1 = L0['attention'][0], L1['attention'][0]
        def rel(x0, x1):
            x0 = np.asarray(x0, np.float64); x1 = np.asarray(x1, np.float64)
            return float(np.linalg.norm(x1 - x0) / max(np.linalg.norm(x0), 1e-12))
        r = {k: rel(a0[k], a1[k]) for k in ('q_w', 'k_w', 'v_w', 'g_w', 'attn_output', 'out_scale', 'mask_omega')}
        E0, E1 = np.asarray(L0['embed']), np.asarray(L1['embed'])
        r['embed_sincos'] = rel(E0[:4], E1[:4])
        r['embed_spin_rows'] = rel(E0[4:], E1[4:]) if gen == 'lj2' else rel(E0[10:], E1[10:])
        if gen == 'lj3':
            r['embed_harm_norm_final'] = float(np.linalg.norm(E1[4:10]))
        mlp0 = jax.tree_util.tree_leaves(L0['mlp']); mlp1 = jax.tree_util.tree_leaves(L1['mlp'])
        r['mlp'] = float(np.sqrt(sum(np.sum((np.asarray(b) - np.asarray(a)) ** 2) for a, b in zip(mlp0, mlp1)))
                         / np.sqrt(sum(np.sum(np.asarray(a) ** 2) for a in mlp0)))
        orb0 = jax.tree_util.tree_leaves(p0['orbital']); orb1 = jax.tree_util.tree_leaves(p1['orbital'])
        r['orbital'] = float(np.sqrt(sum(np.sum((np.asarray(b) - np.asarray(a)) ** 2) for a, b in zip(orb0, orb1)))
                             / np.sqrt(sum(np.sum(np.asarray(a) ** 2) for a in orb0)))
        out[f'{gen}_{tag}'] = r
        print(f"{gen} {lam} {typ:3s}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
json.dump(out, open('moved.json', 'w'), indent=1)
