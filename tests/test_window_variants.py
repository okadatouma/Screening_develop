"""Independent equations, gradients, kinetic energy and checkpoint replay."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics,make_case
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np

from periodicwave import attention,CustomScreenformer as CS
from screening_analysis.attn_probe import GLOBAL_SETTINGS,_effective_settings,probe
from screening_analysis.checkpoint_model import load_model


class WindowVariantChecks(unittest.TestCase):
    def setUp(self):
        context=jax.experimental.enable_x64(True);context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def test_equations_and_nonzero_tails_beyond_scale(self):
        params={'mask_omega':jnp.log(jnp.expm1(jnp.array([1.,2.]))),
                'mask_eta_logit':jnp.log(jnp.array([.2,.7])/jnp.array([.8,.3]))}
        distance=jnp.array([[0.,1.,2.,4.]])
        t=np.array(distance)[None]/np.array([1.,2.])[:,None,None]
        expected={'gaussian':np.exp(-t*t),'algebraic':1/(1+t**6),
                  'hybrid':np.array([.8,.3])[:,None,None]*np.exp(-t*t)+np.array([.2,.7])[:,None,None],
                  'full':np.ones((2,1,4))}
        for shape,value in expected.items():
            actual=jax.jit(lambda d:attention.screening_window(params,d,1.,win_shape=shape,freeze_w=False,algebraic_power=6))(distance)
            np.testing.assert_allclose(actual,value,rtol=1e-12,atol=1e-12)
            self.assertTrue(np.all(np.asarray(actual)>0))
        gaussian=float(expected['gaussian'][0,0,-1]);algebraic=float(expected['algebraic'][0,0,-1])
        self.assertGreater(algebraic,gaussian)

    def test_origin_second_derivative_width_and_eta_gradients(self):
        p={'mask_omega':jnp.array([.2]),'mask_eta_logit':jnp.array([-.5])}
        xi=float(jax.nn.softplus(p['mask_omega'][0]))
        for shape,power,curvature in [('gaussian',4,-2/xi**2),('algebraic',2,-2/xi**2),('algebraic',4,0.)]:
            f=lambda d:attention.screening_window(p,jnp.reshape(d,(1,1)),1.,win_shape=shape,freeze_w=False,algebraic_power=power)[0,0,0]
            self.assertAlmostEqual(float(jax.grad(f)(0.)),0.,places=12)
            self.assertAlmostEqual(float(jax.grad(jax.grad(f))(0.)),curvature,places=12)
        for shape in ('gaussian','algebraic','hybrid','full'):
            def cost(params,freeze=False):
                return attention.screening_window(params,jnp.array([[.7]]),1.,win_shape=shape,freeze_w=freeze).sum()
            gradient=jax.grad(cost)(p);frozen=jax.grad(lambda params:cost(params,True))(p)
            self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(gradient)))
            np.testing.assert_array_equal(frozen['mask_omega'],0.)
            if shape!='full':self.assertGreater(abs(float(gradient['mask_omega'][0])),0.)
            else:np.testing.assert_array_equal(gradient['mask_omega'],0.)
            if shape=='hybrid':self.assertGreater(float(gradient['mask_eta_logit'][0]),0.)

    def test_hybrid_mixed_precision_remains_bounded_for_ph(self):
        p={'mask_omega':jnp.zeros(17,dtype=jnp.float32),
           'mask_eta_logit':jnp.linspace(-3.,3.,17,dtype=jnp.float32)}
        window=attention.screening_window(p,jnp.array([[0.,.2,1.]],dtype=jnp.float64),1.,
            win_shape='hybrid',freeze_w=False)
        np.testing.assert_array_equal(np.asarray(window)[:,0,0],1.)
        self.assertTrue(np.all((np.asarray(window)>=0)&(np.asarray(window)<=1)))

    def test_fixed_hybrid_endpoints_and_width_reporting(self):
        p={'mask_omega':jnp.array([.3]),'mask_eta':jnp.array([0.])}
        d=jnp.array([[0.,1.,2.]])
        gaussian=attention.screening_window(p,d,1.,win_shape='gaussian',freeze_w=False)
        hybrid=attention.screening_window(p,d,1.,win_shape='hybrid',freeze_w=False,freeze_eta=True)
        np.testing.assert_array_equal(gaussian,hybrid)
        p['mask_eta']=jnp.array([1.])
        full=attention.screening_window(p,d,1.,win_shape='hybrid',freeze_w=False,freeze_eta=True)
        np.testing.assert_array_equal(full,1.)
        derivative=jax.grad(lambda params:attention.screening_window(params,d,1.,win_shape='hybrid',freeze_w=False,freeze_eta=True).sum())(p)
        np.testing.assert_array_equal(derivative['mask_eta'],0.)
        for shape in ('cos','smooth7','gaussian','algebraic'):
            scales=attention.window_scales(p,1.,shape)
            radius=scales['half_weight_radius'][0]
            actual=attention.screening_window(p,radius.reshape(1,1),1.,win_shape=shape,freeze_w=False)
            np.testing.assert_allclose(actual,.5,atol=1e-12)
        p['mask_eta']=jnp.array([.2])
        radius=attention.window_scales(p,1.,'hybrid',freeze_eta=True)['half_weight_radius'][0]
        actual=attention.screening_window(p,radius.reshape(1,1),1.,win_shape='hybrid',freeze_w=False,freeze_eta=True)
        np.testing.assert_allclose(actual,.5,atol=1e-12)
        p['mask_eta']=jnp.array([1.])
        self.assertTrue(np.isnan(attention.window_scales(p,1.,'full')['width']).all())
        self.assertTrue(np.isnan(attention.window_scales(p,1.,'hybrid',freeze_eta=True)['half_weight_radius']).all())
        self.assertTrue(np.isnan(attention.window_scales(p,1.,'hybrid',freeze_eta=True)['width']).all())

    def test_new_metadata_is_required_and_old_metadata_replays(self):
        old=dict(TRIM_MAX=2.,TRIM_POW=4,WIN_SHAPE='smooth7',FREEZE_W=False,CLIP_R=None,CLIP_BETA=.3,
                 jax_enable_x64=True,jax_default_matmul_precision='float32')
        effective,provenance=_effective_settings({},old,None)
        self.assertEqual(effective['WIN_POWER'],4)
        self.assertIn('legacy_irrelevant_defaults',provenance)
        for shape in ('gaussian','algebraic','hybrid','full'):
            with self.assertRaises(ValueError):_effective_settings({},dict(old,WIN_SHAPE=shape),None)

    def test_factory_checkpoint_replay_probe_and_folx_kinetic_energy(self):
        from screening_analysis.train_topology_pilot import build,data_from_key
        for shape,fixed,eta in [('gaussian',False,.2),('algebraic',False,.2),('hybrid',False,.2),('hybrid',True,0.),('full',False,.2)]:
            with self.subTest(shape=shape,fixed=fixed),tempfile.TemporaryDirectory() as tmp,patch.multiple(
                    CS,WIN_SHAPE=shape,WIN_POWER=6,WIN_ETA=eta,FREEZE_ETA=fixed,TRIM_POW=4,FREEZE_W=False,CLIP_R=None):
                net,_,_,_,_,cfg=build(2)
                params=net.init(jax.random.PRNGKey(91));data=data_from_key(jax.random.PRNGKey(19),2)
                cfg['screening_attention_settings']={key:getattr(CS,key) for key in GLOBAL_SETTINGS}
                cfg['screening_attention_settings'].update(jax_enable_x64=True,jax_default_matmul_precision='float32')
                folder=Path(tmp);(folder/'runtime_config.json').write_text(json.dumps(cfg))
                np.savez_compressed(folder/'qmcjax_ckpt_000000.npz',t=0,
                    params=jax.tree.map(lambda x:np.asarray(x)[None],params),data={k:np.asarray(v)[None] for k,v in dict(data).items()})
                inputs=[getattr(data,k)[0] for k in ('positions','spins','atoms','charges')]
                expected=np.asarray(net.apply(params,*inputs))
                with patch.multiple(CS,WIN_SHAPE='cos',WIN_POWER=2,WIN_ETA=.7,FREEZE_ETA=not fixed):
                    with load_model(folder) as (loaded,restored,_,info):
                        np.testing.assert_allclose(np.asarray(loaded.apply(restored,*inputs)),expected,atol=1e-12)
                        self.assertEqual(info['settings']['WIN_POWER'],6)
                    self.assertEqual(CS.WIN_POWER,2)
                p=probe(folder,config_path=folder/'runtime_config.json',nconf=1,collect_internal=False)
                ap=params['layers']['attention'][0]
                expected_window=attention.screening_window(ap,jnp.asarray(p['d'][0]),p['dref'],win_shape=shape,
                    freeze_w=False,algebraic_power=6,freeze_eta=fixed)
                np.testing.assert_allclose(p['Window'][0,0],expected_window,atol=1e-12)
                # Actual lambda-jellium Hamiltonian with the production folx path.
                _,p2,point,_,_,energy=make_case('lambda',tmp)
                value=energy(p2,jax.random.PRNGKey(3),point)[0]
                self.assertTrue(np.isfinite(np.asarray(value)).all())


if __name__=='__main__':unittest.main()
