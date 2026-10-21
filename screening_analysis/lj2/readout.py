"""lj2 の網の 1 層目のアテンションの中身（CustomScreenformer の apply をそのまま写したもの）と読み出し。"""
import numpy as np, jax, jax.numpy as jnp
import lj2net as LN
from periodicwave import CustomScreenformer as CS, networks
from periodicwave.pbc import feature_layer as FL


def make_parts(run):
    cfg, L = run['cfg'], jnp.asarray(run['L'])
    o = cfg['network']['CustomScreenformer']
    NH, AD, VD = o['num_heads'], o['attn_dim'], o['value_dim']
    NE = 36
    dref = float(CS._max_periodic_distance(L))
    fl = FL.make_pbc_feature_layer(natoms=1, nspins=(NE, 0), ndim=2, lattice=L, include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(L)
    P = run['params']['layers']
    ap = P['attention'][0]

    def internals(pos, spins):
        ae, ee, r_ae, r_ee = cif(pos, jnp.zeros((1, 2)), ndim=2)
        f, _ = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = jnp.concatenate((f, spins[..., None]), axis=-1)
        x = jnp.dot(f, P['embed'])
        d = r_ee[..., 0]
        q = jnp.dot(x, ap['q_w']).reshape(NE, NH, AD)
        k = jnp.dot(x, ap['k_w']).reshape(NE, NH, AD)
        v = jnp.dot(x, ap['v_w']).reshape(NE, NH, VD)
        qb, kb, vb = CS._unit_length(q), CS._unit_length(k), CS._unit_length(v)
        s = jnp.einsum('ihd,jhd->hij', qb, kb)
        r = CS.TRIM_MAX * jax.nn.sigmoid(ap['trim_rho'])
        al = jax.nn.relu(s - (1.0 - r[:, None, None])) / r[:, None, None]
        al = jnp.square(jnp.square(al))                                  # TRIM_POW = 4
        w = dref * jax.nn.softplus(ap['mask_omega'])
        t = jnp.clip(d[None] / w[:, None, None], 0.0, 1.0)
        m = 1.0 - CS._smooth7(t)                                          # smooth7 の入れ子窓
        wgt = al * m                                                      # [h, i, j]
        eye = jnp.eye(NE)
        sc = jnp.exp(ap['out_scale'])
        g = jnp.tanh(jax.nn.silu(jnp.dot(x, ap['g_w']))).reshape(NE, NH, VD)
        WO = ap['attn_output'].reshape(NH, VD, -1)

        def head_out(wg):                                                 # wg: [h, i, j] -> c_h(i): [i, h, E]
            at = jnp.einsum('hij,jhd->ihd', wg, vb)
            n = jnp.sqrt(jnp.sum(at * at, -1, keepdims=True) + 1e-12)
            at = at * jnp.tanh(n) / n
            return sc * jnp.einsum('ihd,hde->ihe', at * g, WO)
        c_all = head_out(wgt)
        c_self = head_out(wgt * eye[None])
        same = (spins[:, None] == spins[None, :]).astype(wgt.dtype) * (1 - eye)
        opp = (spins[:, None] != spins[None, :]).astype(wgt.dtype)
        a_same = jnp.einsum('hij,jhd->ihd', wgt * same[None], vb)        # 自分と同じ擬スピンの相手から集めた分（TanhNorm の前）
        a_opp = jnp.einsum('hij,jhd->ihd', wgt * opp[None], vb)
        a_self = jnp.einsum('hij,jhd->ihd', wgt * eye[None], vb)
        return dict(s=s, m=m, wgt=wgt, d=d, x=x, c_all=c_all, c_p=c_all - c_self,
                    a_same=a_same, a_opp=a_opp, a_self=a_self, same=same, opp=opp, w=w, r=r)
    return internals, dref


def verify(run, n=3):
    """写した計算の Σ_h c_h が網の attention の出力と一致するか。"""
    internals, dref = make_parts(run)
    cfg = run['cfg']; o = cfg['network']['CustomScreenformer']
    att_init, att_apply = CS.make_screening_attention(
        o['num_heads'], o['mlp_dim'], o['attn_dim'], o['value_dim'], 36, dref, use_softmask=True, use_gate=True,
        use_tanh_norm=True, init_trim_r=o['init_trim_r'], init_softmask_w_factor=o['init_softmask_w_factor'],
        init_softmask_w_spread=o['init_softmask_w_spread'], use_output_scale=True, init_output_scale=0.0)
    ap = run['params']['layers']['attention'][0]
    errs = []
    for t in range(n):
        pos = jnp.asarray(run['R'][t], jnp.float32); sp = jnp.asarray(run['S'][t], jnp.float32)
        I = internals(pos, sp)
        ref = att_apply(ap, I['x'], I['d'])
        mine = I['c_all'].sum(1)
        errs.append(float(jnp.max(jnp.abs(ref - mine)) / jnp.max(jnp.abs(ref))))
    return errs
