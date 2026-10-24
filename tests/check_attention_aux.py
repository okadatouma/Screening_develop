"""B2: 実forwardの内部量と、取得OFF/ONの同等性を検証する。"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.support import require_numerics, make_case, ROUTES
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import attention, CustomScreenformer as CS
from tests.check_screening import make_case as attention_case, MODES
from tests.check_window import assert_exact, boundary_points
from tests.test_attention import attention_values


def entries(mode, freeze=False):
    p,x,d,settings,_,_ = attention_case(mode)
    settings['FREEZE_W'] = freeze
    options = dict(num_electrons=3,num_heads=2,attn_dim=4,value_dim=3,d_ref=1.4,
                   use_softmask=mode!='nomask',use_fixed_value=mode=='fixed',
                   use_tanh_norm=mode!='nonorm',use_gate=mode!='nogate',use_output_scale=mode!='noscale')
    def call(pp,xx,dd,aux):
        pair = jnp.stack([dd,jnp.sin(dd),jnp.cos(dd),dd*dd,jnp.ones_like(dd)],-1)
        return attention.screening_attention(pp,xx,dd,pair,options=options,
            **{k.lower():v for k,v in settings.items()},return_aux=aux)
    return p,x,d,settings,lambda p,x,d:call(p,x,d,False),lambda p,x,d:call(p,x,d,True)


class AttentionAuxChecks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64',True)

    def test_off_on_forward_and_jit(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for mode in MODES:
                with self.subTest(x64=x64,mode=mode):
                    p,x,d,_,off,on = entries(mode)
                    out,aux = on(p,x,d)
                    assert_exact(self,out,off(p,x,d))
                    compiled,compiled_aux = jax.jit(on)(p,x,d)
                    assert_exact(self,compiled,jax.jit(off)(p,x,d))
                    self.assertEqual(aux['weights_window'].shape,(2,3,3))
                    self.assertEqual(aux['gate_output'].shape,(3,10 if mode=='fixed' else 6))
                    self.assertEqual(aux['aggregate'].shape,(3,2,5 if mode=='fixed' else 3))
                    self.assertEqual(compiled_aux.keys(),aux.keys())
                    if mode=='nomask': self.assertIsNone(aux['window'])
                    if mode=='fixed':
                        self.assertIsNone(aux['v']);self.assertIsNone(aux['v_normalized'])
                        self.assertIsNone(aux['v_effective'])
                        np.testing.assert_array_equal(aux['aggregation_weights'][:,jnp.arange(3),jnp.arange(3)],0.)
                    else: self.assertIsNone(aux['pair_features'])
                    # 本体が実際に使用した重み・Valueから集約を検算する。
                    agg = (jnp.einsum('hij,ijd->ihd',aux['aggregation_weights'],aux['pair_features'])
                           if mode=='fixed' else jnp.einsum('hij,jhd->ihd',aux['aggregation_weights'],aux['v_effective']))
                    assert_exact(self,aux['aggregate'],agg)
                    # eagerでは元の射影結果と倍率から同じ順序で検算できる。
                    expected = (aux['projected_output'] if mode=='noscale'
                                else jnp.exp(p['out_scale'])*aux['projected_output'])
                    assert_exact(self,out,expected)
                    if mode=='hooks':
                        np.testing.assert_array_equal(aux['weights'][1] * (1-jnp.eye(3)),0.)
                        self.assertFalse(np.array_equal(aux['weights_window'],aux['weights']))
                        self.assertFalse(np.array_equal(aux['v_normalized'],aux['v_effective']))

    def test_derivatives_with_aux_materialized(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for mode in MODES:
                for freeze in (False,True):
                    with self.subTest(x64=x64,mode=mode,freeze=freeze):
                        p,x,d,_,off,on = entries(mode,freeze)
                        f = lambda p,x,d:jnp.sum(jnp.sin(off(p,x,d)))
                        def f_aux(p,x,d):
                            out,aux = on(p,x,d)
                            return jnp.sum(jnp.sin(out)),aux
                        value,grad = jax.jit(jax.value_and_grad(f,argnums=(0,1,2)))(p,x,d)
                        (value_aux,aux),grad_aux = jax.jit(jax.value_and_grad(f_aux,argnums=(0,1,2),has_aux=True))(p,x,d)
                        assert_exact(self,(value,grad),(value_aux,grad_aux))
                        hess,aux_hess = jax.jit(jax.hessian(f_aux,argnums=2,has_aux=True))(p,x,d)
                        assert_exact(self,jax.jit(jax.hessian(f,argnums=2))(p,x,d),hess)
                        if freeze and 'mask_omega' in grad_aux[0]:
                            np.testing.assert_array_equal(grad_aux[0]['mask_omega'],0.)
                        # 内部量の勾配を止めない。倍率なしでは射影出力の微分も同一。
                        if mode=='noscale':
                            g = jax.grad(lambda pp:jnp.sum(on(pp,x,d)[1]['projected_output']))(p)
                            assert_exact(self,g,jax.grad(lambda pp:jnp.sum(off(pp,x,d)))(p))

    def test_window_boundaries_off_on(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for mode in ('cos','smooth7','pow4','band','sin','gauss'):
                with self.subTest(x64=x64,mode=mode):
                    p,x,d,_,off,on = entries(mode)
                    points=boundary_points(p,np.float64 if x64 else np.float32)
                    def distance(t):
                        return d.at[0,1].set(t).at[1,0].set(t)
                    f=lambda t:jnp.sum(jnp.sin(off(p,x,distance(t))))
                    def f_aux(t):
                        out,aux=on(p,x,distance(t))
                        return jnp.sum(jnp.sin(out)),aux['window']
                    def expected(t):
                        return f(t),jax.grad(f)(t),jax.hessian(f)(t)
                    def actual(t):
                        (value,window),grad=jax.value_and_grad(f_aux,has_aux=True)(t)
                        hess,_=jax.hessian(f_aux,has_aux=True)(t)
                        return (value,grad,hess),window
                    values,windows=jax.jit(jax.vmap(actual))(points)
                    assert_exact(self,values,jax.jit(jax.vmap(expected))(points))
                    self.assertTrue(np.all(np.isfinite(windows)))

    def test_internal_values_against_independent_equations(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for mode in MODES[:10]:
                with self.subTest(x64=x64,mode=mode):
                    values,p = attention_values(mode)
                    x = jnp.asarray(np.arange(24).reshape(3,8)/13.-0.7)
                    d = jnp.asarray([[0.,0.3,1.1],[0.3,0.,0.8],[1.1,0.8,0.]])
                    pair = jnp.stack([d,jnp.sin(d),jnp.cos(d),d*d,jnp.ones_like(d)],-1)
                    options=dict(num_electrons=3,num_heads=2,attn_dim=4,value_dim=4,d_ref=1.4,
                        use_softmask=mode!='nomask',use_fixed_value=mode=='fixed',use_tanh_norm=mode!='nonorm',
                        use_gate=mode!='nogate',use_output_scale=True)
                    out,a = attention.screening_attention(p,x,d,pair,options=options,trim_max=2.,
                        trim_pow=4 if mode=='pow4' else 2,win_shape='smooth7' if mode=='smooth7' else 'cos',
                        freeze_w=False,clip_r=None,clip_beta=0.3,return_aux=True)
                    # 既存の独立NumPy式の許容誤差を使用する。OFF/ON自体は上で厳密比較。
                    for actual,expected in [(a['q_normalized'],values['q']),(a['k_normalized'],values['k']),
                        (a['similarity'],values['similarity']),(a['alpha'],values['alpha']),
                        (a['aggregation_weights'],values['weights']),(a['aggregate'],values['aggregate']),
                        (a['gate_output'],values['head'].reshape(3,-1)),(out,values['reference'])]:
                        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)
                    # 未正規化Q/K/V、TanhNorm後、射影直後も独立式で照合する。
                    xx=np.asarray(x)
                    for name in ('q','k'):
                        expected=(xx @ np.asarray(p[name+'_w'])).reshape(3,2,4)
                        np.testing.assert_allclose(a[name],expected,rtol=2e-6,atol=2e-7)
                    if mode!='fixed':
                        v=(xx @ np.asarray(p['v_w'])).reshape(3,2,4)
                        vn=v/np.sqrt(np.sum(v*v,-1,keepdims=True)+1e-12)
                        np.testing.assert_allclose(a['v'],v,rtol=2e-6,atol=2e-7)
                        np.testing.assert_allclose(a['v_normalized'],vn,rtol=2e-6,atol=2e-7)
                        assert_exact(self,a['v_effective'],a['v_normalized'])
                    norm=np.sqrt(np.sum(values['aggregate']**2,-1,keepdims=True)+1e-12)
                    tanh=(values['aggregate'] if mode=='nonorm' else values['aggregate']*np.tanh(norm)/norm)
                    np.testing.assert_allclose(a['tanh_output'],tanh,rtol=2e-6,atol=2e-7)
                    projection=values['head'].reshape(3,-1) @ np.asarray(p['attn_output'])
                    np.testing.assert_allclose(a['projected_output'],projection,rtol=2e-6,atol=2e-7)
                    window_weight=values['alpha']*values['mask']
                    np.testing.assert_allclose(a['weights_window'],window_weight,rtol=2e-6,atol=2e-7)
                    assert_exact(self,a['weights'],a['weights_window'])
                    assert_exact(self,out,jnp.asarray(values['output']))
                    if a['window'] is not None:
                        np.testing.assert_allclose(a['window'],values['mask'],rtol=2e-6,atol=2e-7)

    def test_network_values_and_derivatives(self):
        original = attention.screening_attention
        def with_aux(*args,**kwargs):
            return original(*args,**kwargs,return_aux=True)[0]
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for route in ROUTES:
                with self.subTest(x64=x64,route=route),tempfile.TemporaryDirectory() as directory:
                    net,p,data,_,_,_ = make_case(route,Path(directory))
                    def evaluate(pp,x):
                        wave = lambda p_,x_:net.apply(p_,x_,data.spins,data.atoms,data.charges)
                        scalar = lambda p_,x_:(lambda ph,la:ph+0.37*la)(*wave(p_,x_))
                        return (wave(pp,x),jax.jacrev(wave,argnums=1)(pp,x),
                                jax.hessian(scalar,argnums=1)(pp,x),jax.grad(scalar)(pp,x))
                    expected = jax.jit(lambda p,x:evaluate(p,x))(p,data.positions)
                    with patch.object(attention,'screening_attention',with_aux):
                        actual = jax.jit(lambda p,x:evaluate(p,x))(p,data.positions)
                    assert_exact(self,actual,expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
