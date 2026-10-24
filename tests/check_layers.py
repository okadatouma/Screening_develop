"""段階D: 同じ層ループのOFF/ONを旧ループ・実forwardと比較する。"""
import inspect
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.support import require_numerics, make_case
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, attention, networks, network_blocks
from periodicwave.pbc import feature_layer as FL
from tests import layers_reference as old
from tests.check_window import assert_exact

CASES = [(n,ln,'cos') for n in (1,2,3) for ln in (False,True)] + [
    (3,True,m) for m in ('smooth7','pow4','band','sin','gauss','fixed','nomask','nogate','noscale','nonorm','freeze','hooks')]
CASES += [(2,False,'fixed'),(3,False,'gauss')]


def fixture(n,ln,mode):
    settings=dict(TRIM_MAX=2.,TRIM_POW=4 if mode=='pow4' else 2,
        WIN_SHAPE='smooth7' if mode=='smooth7' else 'cos',FREEZE_W=mode=='freeze',
        CLIP_R=0.95 if mode=='hooks' else None,CLIP_BETA=0.3)
    lattice=jnp.asarray([[3.2,-1.6],[0.,2.7712812921102037]])
    dref=CS._max_periodic_distance(lattice)
    kw=dict(num_layers=n,num_heads=2,attn_dim=3,value_dim=2,mlp_dim=4,
        num_perceptrons_per_layer=2,num_electrons=2,d_ref=dref,use_layer_norm=ln,
        mlp_activation_fct='GELU',use_fixed_value=mode=='fixed',pair_dim=5,
        use_softmask=mode!='nomask',use_gate=mode!='nogate',use_output_scale=mode!='noscale',
        use_tanh_norm=mode!='nonorm')
    if mode=='band':kw['init_softmask_centres']=([0.,0.6],[0.5,0.5])
    if mode in ('sin','gauss'):kw['init_softmask_rings']=([0.,0.5,1.],0.16,1.,mode)
    with patch.multiple(CS,**settings):
        init,apply=CS.make_self_attention_block(**kw)
        old_init,old_apply=old.make_self_attention_block(**kw)
        p=init(jax.random.PRNGKey(7),4)
        previous=old_init(jax.random.PRNGKey(7),4)
    if mode=='hooks':
        for tree in (p,previous):
            for ap in tree['attention']:
                ap.update(_v_mode=jnp.asarray([1.,0.],dtype=jnp.float32),
                          _diag_only=jnp.asarray([0.,1.],dtype=jnp.float32))
    construct=networks.make_periodic_construct_input_features(lattice)
    features=FL.make_pbc_feature_layer(natoms=1,nspins=(1,1),ndim=2,lattice=lattice,include_r_ae=False)
    def inputs(pos):
        ae,ee,ra,re=construct(pos,jnp.zeros((1,2)),ndim=2)
        x,_=features.apply(ae=ae,ee=ee,r_ae=ra,r_ee=re)
        _,pair=features.apply(ae=ae,ee=ee,r_ae=ra,r_ee=re)
        pair=jnp.concatenate((pair[...,:1]/dref,pair[...,1:]),-1)
        return x,re[...,0],pair
    pos=jnp.asarray([0.23,0.41,1.17,0.89])
    return p,previous,apply,old_apply,settings,inputs,pos


CANDIDATE_AVAILABLE = ('return_aux' in inspect.getsource(CS.make_self_attention_block)
                       and not hasattr(CS, 'make_screenformer_diagnostic'))


@unittest.skipUnless(CANDIDATE_AVAILABLE,
    '撤回済み旧D候補のON/OFF厳密一致仕様専用。正式な分離診断はcheck_formal_diagnosticで検証する。')
class MultiLayerChecks(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_forward_each_layer_and_jit(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for n,ln,mode in CASES:
                with self.subTest(x64=x64,n=n,ln=ln,mode=mode):
                    p,previous,apply,old_apply,settings,inputs,pos=fixture(n,ln,mode)
                    assert_exact(self,p,previous)
                    x,d,pair=inputs(pos)
                    with patch.multiple(CS,**settings):
                        # 計算を変更しないテスト用記録で、通常forwardが実際に使った値を得る。
                        used=[];kernel=attention.screening_attention
                        def record(*args,**kwargs):
                            out=kernel(*args,**kwargs)
                            used.append((args[1],out))
                            return out
                        with patch.object(attention,'screening_attention',record):
                            normal=apply(p,x,d,pair)
                        out,layers=apply(p,x,d,pair,return_aux=True)
                        assert_exact(self,out,normal)
                        assert_exact(self,out,old_apply(p,x,d,pair))
                        self.assertEqual(len(layers),n);self.assertEqual(len(used),n)
                        _,norm=network_blocks.make_layer_norm();_,mlp=CS.make_mlp(2,'GELU')
                        for i,r in enumerate(layers):
                            assert_exact(self,r['input'],used[i][0])
                            assert_exact(self,r['attention_output'],used[i][1])
                            assert_exact(self,r['attention_residual'],r['input']+r['attention_output'])
                            first=norm(p['ln'][i][0],r['attention_residual']) if ln else r['attention_residual']
                            assert_exact(self,r['ln1_output'],first)
                            assert_exact(self,r['mlp_output'],mlp(p['mlp'][i],first))
                            assert_exact(self,r['mlp_residual'],first+r['mlp_output'])
                            second=norm(p['ln'][i][1],r['mlp_residual']) if ln else r['mlp_residual']
                            assert_exact(self,r['ln2_output'],second);assert_exact(self,r['output'],second)
                            assert_exact(self,r['output'],layers[i+1]['input'] if i+1<n else out)
                            self.assertEqual(r['attention']['q'].shape,(2,2,3))
                            self.assertEqual(r['attention']['aggregate'].shape,(2,2,5 if mode=='fixed' else 2))
                        jit_out,jit_layers=jax.jit(lambda pp,xx,dd,pf:apply(pp,xx,dd,pf,return_aux=True))(p,x,d,pair)
                        assert_exact(self,jit_out,jax.jit(apply)(p,x,d,pair))
                        assert_exact(self,jit_out,jax.jit(old_apply)(p,x,d,pair))

    def test_coordinate_hessian_parameter_gradients_with_aux(self):
        cases=[(1,True,'cos'),(2,False,'pow4'),(3,True,'smooth7'),(3,False,'fixed'),
               (2,True,'gauss'),(3,True,'freeze'),(3,False,'hooks'),(2,True,'nomask')]
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for n,ln,mode in cases:
                with self.subTest(x64=x64,n=n,ln=ln,mode=mode):
                    p,_,apply,previous,settings,inputs,pos=fixture(n,ln,mode)
                    def scalar(fun,pp,r):
                        return jnp.sum(jnp.sin(fun(pp,*inputs(r))))
                    def capture(pp,r):
                        out,layers=apply(pp,*inputs(r),return_aux=True)
                        return jnp.sum(jnp.sin(out)),layers
                    with patch.multiple(CS,**settings):
                        (value,layers),g=jax.jit(jax.value_and_grad(capture,argnums=(0,1),has_aux=True))(p,pos)
                        v0,g0=jax.jit(jax.value_and_grad(lambda pp,r:scalar(apply,pp,r),argnums=(0,1)))(p,pos)
                        vo,go=jax.jit(jax.value_and_grad(lambda pp,r:scalar(previous,pp,r),argnums=(0,1)))(p,pos)
                        assert_exact(self,(value,g),(v0,g0));assert_exact(self,(value,g),(vo,go))
                        h,_=jax.jit(jax.hessian(capture,argnums=1,has_aux=True))(p,pos)
                        assert_exact(self,h,jax.jit(jax.hessian(lambda r:scalar(apply,p,r)))(pos))
                        if mode=='freeze':
                            for ap in g[0]['attention']:np.testing.assert_array_equal(ap['mask_omega'],0.)

    def test_actual_wavefunction_trunk_entry(self):
        # 各Ansatzの軌道・行列式への接続も、同じ層applyを使うことを検証する。
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for route,n,ln in [('screen',2,False),('lambda',3,True),('hf',3,False)]:
                with self.subTest(x64=x64,route=route),tempfile.TemporaryDirectory() as directory:
                    _,_,data,lattice,opts,_=make_case(route,Path(directory))
                    opts.update(num_layers=n,use_layer_norm=ln,attn_dim=3,value_dim=2)
                    spins=(1,1) if route=='screen' else (2,0)
                    net=CS.make_fermi_net(spins,data.charges,**opts)
                    with patch.object(CS,'make_screenformer_layers',old.make_screenformer_layers):
                        oldnet=CS.make_fermi_net(spins,data.charges,**opts)
                    p=net.init(jax.random.PRNGKey(7));assert_exact(self,p,oldnet.init(jax.random.PRNGKey(7)))
                    _,trunk=CS.make_screenformer_layers(spins,1,net.options)
                    cif=networks.make_periodic_construct_input_features(lattice)
                    def evaluate(pp,r,aux):
                        ae,ee,ra,re=cif(r,data.atoms,ndim=2)
                        return trunk(pp['layers'],ae=ae,ee=ee,r_ae=ra,r_ee=re,spins=data.spins,charges=data.charges,return_aux=aux)
                    out,layers=jax.jit(lambda pp,r:evaluate(pp,r,True))(p,data.positions)
                    assert_exact(self,out,jax.jit(lambda pp,r:evaluate(pp,r,False))(p,data.positions))
                    self.assertEqual(len(layers),n)
                    # 本体Networkは変更しない。テスト内で構築時の層applyだけONの第1要素へ差し替える。
                    builder=CS.make_screenformer_layers
                    def captured_builder(*args,**kwargs):
                        init,apply=builder(*args,**kwargs)
                        return init,lambda *a,**k:apply(*a,**k,return_aux=True)[0]
                    with patch.object(CS,'make_screenformer_layers',captured_builder):
                        captured=CS.make_fermi_net(spins,data.charges,**opts)
                    def wave(fun,pp,r):return fun(pp,r,data.spins,data.atoms,data.charges)
                    def result(fun,pp,r):
                        f=lambda pp,r:(lambda ph,la:ph+0.37*la)(*wave(fun,pp,r))
                        return wave(fun,pp,r),jax.grad(f,argnums=(0,1))(pp,r),jax.hessian(f,argnums=1)(pp,r)
                    normal=jax.jit(lambda pp,r:result(net.apply,pp,r))(p,data.positions)
                    assert_exact(self,normal,jax.jit(lambda pp,r:result(captured.apply,pp,r))(p,data.positions))
                    assert_exact(self,normal,jax.jit(lambda pp,r:result(oldnet.apply,pp,r))(p,data.positions))


if __name__=='__main__':unittest.main(verbosity=2)
