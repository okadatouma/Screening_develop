"""1 層目のアテンションの中身（CustomScreenformer の apply を写したもの）。lj2（sin, cos, s）と lj3（12 成分、Trim 固定）の両方に使える。
入力は網と同じ関数 CS.one_electron_inputs で作り、Trim の幅は params に trim_rho があればそれ、なければ r = 2。"""
import types
import numpy as np, jax, jax.numpy as jnp
import lj3net as LN
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
    opts = types.SimpleNamespace(input_mode=o.get('input_mode', 'sincos_s'), input_harmonics=o.get('input_harmonics'),
                                 pbc_lattice=np.asarray(run['L']))
    P = run['params']['layers']
    ap = P['attention'][0]

    def internals(pos, spins):
        ae, ee, r_ae, r_ee = cif(pos, jnp.zeros((1, 2)), ndim=2)
        f, _ = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = CS.one_electron_inputs(opts, f, ae, spins)
        x = jnp.dot(f, P['embed'])
        d = r_ee[..., 0]
        q = jnp.dot(x, ap['q_w']).reshape(NE, NH, AD)
        k = jnp.dot(x, ap['k_w']).reshape(NE, NH, AD)
        v = jnp.dot(x, ap['v_w']).reshape(NE, NH, VD)
        qb, kb, vb = CS._unit_length(q), CS._unit_length(k), CS._unit_length(v)
        s = jnp.einsum('ihd,jhd->hij', qb, kb)
        r = CS.TRIM_MAX * jax.nn.sigmoid(ap['trim_rho']) if 'trim_rho' in ap else jnp.full((NH,), CS.TRIM_MAX)
        al = jax.nn.relu(s - (1.0 - r[:, None, None])) / r[:, None, None]
        al = jnp.square(jnp.square(al))                                  # TRIM_POW = 4
        w = dref * jax.nn.softplus(ap['mask_omega'])
        t = jnp.clip(d[None] / w[:, None, None], 0.0, 1.0)
        m = 1.0 - CS._smooth7(t)                                          # smooth7 の入れ子窓
        wgt = al * m
        eye = jnp.eye(NE)
        sc = jnp.exp(ap['out_scale'])
        g = jnp.tanh(jax.nn.silu(jnp.dot(x, ap['g_w']))).reshape(NE, NH, VD)
        WO = ap['attn_output'].reshape(NH, VD, -1)

        def head_out(wg):
            at = jnp.einsum('hij,jhd->ihd', wg, vb)
            n = jnp.sqrt(jnp.sum(at * at, -1, keepdims=True) + 1e-12)
            at = at * jnp.tanh(n) / n
            return sc * jnp.einsum('ihd,hde->ihe', at * g, WO)
        c_all = head_out(wgt)
        c_self = head_out(wgt * eye[None])
        same = (spins[:, None] == spins[None, :]).astype(wgt.dtype) * (1 - eye)
        opp = (spins[:, None] != spins[None, :]).astype(wgt.dtype)
        a_same = jnp.einsum('hij,jhd->ihd', wgt * same[None], vb)
        a_opp = jnp.einsum('hij,jhd->ihd', wgt * opp[None], vb)
        a_self = jnp.einsum('hij,jhd->ihd', wgt * eye[None], vb)
        return dict(s=s, al=al, m=m, wgt=wgt, d=d, x=x, c_all=c_all, c_p=c_all - c_self,
                    a_same=a_same, a_opp=a_opp, a_self=a_self, same=same, opp=opp, w=w, r=r)
    return internals, dref


def verify(run, n=3):
    """写した計算の Σ_h c_h が網の attention の出力と一致するか（相対誤差）。"""
    internals, dref = make_parts(run)
    o = run['cfg']['network']['CustomScreenformer']
    att_init, att_apply = CS.make_screening_attention(
        o['num_heads'], o['mlp_dim'], o['attn_dim'], o['value_dim'], 36, dref, use_softmask=True, use_gate=True,
        use_tanh_norm=True, init_trim_r=o['init_trim_r'], init_softmask_w_factor=o['init_softmask_w_factor'],
        init_softmask_w_spread=o['init_softmask_w_spread'], use_output_scale=True, init_output_scale=0.0,
        trim_fixed=bool(o.get('trim_fixed', False)))
    ap = run['params']['layers']['attention'][0]
    errs = []
    for t in range(n):
        pos = jnp.asarray(run['R'][t], jnp.float32); sp = jnp.asarray(run['S'][t], jnp.float32)
        I = internals(pos, sp)
        ref = att_apply(ap, I['x'], I['d'])
        errs.append(float(jnp.max(jnp.abs(ref - I['c_all'].sum(1))) / jnp.max(jnp.abs(ref))))
    return errs
