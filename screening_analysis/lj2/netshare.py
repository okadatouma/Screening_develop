"""軌道のうち網の部分 eps W_s h の大きさ：eta = <||O_net||^2> / <||O_HF||^2>（N x N 行列のフロベニウスノルム、行列式 4 枚と配置で平均）。
同じ params で lj_net_scale = 0 の網を作ると O_HF e^{J/N} が出るので、差が網の部分（Jastrow の因子は両方に同じく掛かる）。"""
import json, numpy as np, jax, jax.numpy as jnp
import lj2net as LN
from periodicwave import CustomScreenformer as CS, envelopes
from periodicwave.pbc import feature_layer as FL
out = {}
for lam in LN.LAMS:
    for typ in ['cw', 'ca']:
        run = LN.load(lam, typ)
        kw0 = dict(run['kw']); kw0['lj_net_scale'] = 0.0
        L = jnp.asarray(run['L']); cfg = run['cfg']
        fl = FL.make_pbc_feature_layer(natoms=1, nspins=(36, 0), ndim=2, lattice=L, include_r_ae=False)
        net0 = CS.make_fermi_net((36, 0), jnp.zeros((1,)), ndim=2, determinants=4, envelope=envelopes.make_null_envelope(),
                                 feature_layer=fl, jastrow="NONE", jastrow_kwargs={'ndim': 2}, bias_orbitals=False,
                                 complex_output=True, pbc_lattice=L, **kw0)
        P = run['params']
        f1 = jax.jit(jax.vmap(lambda x, s: run['net'].orbitals(P, x, s, jnp.zeros((1, 2)), jnp.zeros((1,)))[0]))
        f0 = jax.jit(jax.vmap(lambda x, s: net0.orbitals(P, x, s, jnp.zeros((1, 2)), jnp.zeros((1,)))[0]))
        num = den = 0.0
        for b in range(0, 256, 64):
            x = jnp.asarray(run['R'][b:b + 64], jnp.float32); s = jnp.asarray(run['S'][b:b + 64], jnp.float32)
            O1, O0 = np.asarray(f1(x, s)), np.asarray(f0(x, s))
            num += (np.abs(O1 - O0) ** 2).sum(); den += (np.abs(O0) ** 2).sum()
        eta = num / den
        out[f'{lam}_{typ}'] = float(eta)
        print(f"lam {lam:4s} {typ}: eta = ||網の部分||^2 / ||HF の部分||^2 = {eta:.4f}  (sqrt = {np.sqrt(eta):.3f})", flush=True)
json.dump(out, open('netshare.json', 'w'), indent=1)
