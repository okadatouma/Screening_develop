"""B1のforward移動を固定した旧演算と比較。python -m tests.check_screening"""
import unittest
from unittest.mock import patch

from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import attention, CustomScreenformer as CS
from tests import screening_reference as old
from tests.check_window import assert_exact

MODES = ('cos', 'smooth7', 'pow4', 'band', 'sin', 'gauss', 'fixed',
         'nomask', 'nogate', 'nonorm', 'noscale', 'hooks')


def make_case(mode):
    options = dict(num_electrons=3, num_heads=2, attn_dim=4, value_dim=3, d_ref=1.4,
                   use_softmask=mode != 'nomask', use_fixed_value=mode == 'fixed',
                   use_tanh_norm=mode != 'nonorm', use_gate=mode != 'nogate',
                   use_output_scale=mode != 'noscale')
    settings = dict(TRIM_MAX=2., TRIM_POW=4 if mode == 'pow4' else 2,
                    WIN_SHAPE='smooth7' if mode == 'smooth7' else 'cos',
                    FREEZE_W=False, CLIP_R=0.95 if mode == 'hooks' else None, CLIP_BETA=0.3)
    kwargs = dict(options, embed_dim=8, pair_dim=5)
    if mode == 'band':
        kwargs['init_softmask_centres'] = ([0., 0.6], [0.5, 0.5])
    if mode in ('sin', 'gauss'):
        kwargs['init_softmask_rings'] = ([0., 0.5, 1.], 0.16, 1., mode)
    init, apply = CS.make_screening_attention(**kwargs)
    p = init(jax.random.PRNGKey(29))
    if mode == 'hooks':
        p = dict(p, _v_mode=jnp.asarray([1., 0.], dtype=jnp.float32),
                 _diag_only=jnp.asarray([0., 1.], dtype=jnp.float32))
    x = jnp.arange(24).reshape(3, 8) / 13. - 0.7
    d = jnp.asarray([[0., 0.3, 1.1], [0.3, 0., 0.8], [1.1, 0.8, 0.]])
    pair = jnp.stack([d, jnp.sin(d), jnp.cos(d), d*d, jnp.ones_like(d)], -1)
    kw = dict(options=options, **{k.lower(): v for k,v in settings.items()})
    reference = lambda pp,xx,dd: old.screening_attention(pp,xx,dd,pair,**kw)
    actual = lambda pp,xx,dd: apply(pp,xx,dd,pair)
    return p,x,d,settings,actual,reference


class ScreeningChecks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64', True)

    def test_forward_jit_all_modes(self):
        self.assertIs(CS._unit_length, attention._unit_length)
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for mode in MODES:
                with self.subTest(x64=x64, mode=mode):
                    p,x,d,settings,actual,reference = make_case(mode)
                    with patch.multiple(CS, **settings):
                        assert_exact(self, actual(p,x,d), reference(p,x,d))
                        assert_exact(self, jax.jit(actual)(p,x,d), jax.jit(reference)(p,x,d))

    def test_gradients_and_hessian_all_modes(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for mode in MODES:
                with self.subTest(x64=x64, mode=mode):
                    p,x,d,settings,actual,reference = make_case(mode)
                    def derivatives(fn):
                        objective = lambda pp,xx,dd: jnp.sum(jnp.sin(fn(pp,xx,dd)))
                        return jax.jit(lambda pp,xx,dd: (
                            jax.grad(objective, argnums=(0,1,2))(pp,xx,dd),
                            jax.hessian(objective, argnums=2)(pp,xx,dd)))
                    with patch.multiple(CS, **settings):
                        assert_exact(self, derivatives(actual)(p,x,d), derivatives(reference)(p,x,d))

    def test_fixed_pair_gradient_and_hessian(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            with patch.multiple(CS, TRIM_MAX=2., TRIM_POW=2, WIN_SHAPE='cos',
                                FREEZE_W=False, CLIP_R=None, CLIP_BETA=0.3):
                init, apply = CS.make_screening_attention(2,8,4,3,2,1.4,
                                                         use_fixed_value=True,pair_dim=5)
                p = init(jax.random.PRNGKey(29))
                x = jnp.arange(16).reshape(2,8)/13.-0.7
                d = jnp.asarray([[0.,0.37],[0.37,0.]])
                pair = jnp.arange(20).reshape(2,2,5)/17.
                def evaluate(pp,xx,dd,ee):
                    objective = lambda p_,x_,d_,e_: jnp.sum(jnp.sin(apply(p_,x_,d_,e_)))
                    return (apply(pp,xx,dd,ee),
                            jax.grad(objective,argnums=(0,1,2,3))(pp,xx,dd,ee),
                            jax.hessian(objective,argnums=3)(pp,xx,dd,ee))
                with patch.object(attention,'screening_attention',old.screening_attention):
                    expected = jax.jit(lambda p,x,d,e: evaluate(p,x,d,e))(p,x,d,pair)
                actual = jax.jit(lambda p,x,d,e: evaluate(p,x,d,e))(p,x,d,pair)
                assert_exact(self,actual,expected)
                np.testing.assert_array_equal(actual[1][3][jnp.arange(2),jnp.arange(2)],0.)

    def test_global_changes_and_cached_jit(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            p,x,d,settings,actual,_ = make_case('cos')
            with patch.multiple(CS, **settings):
                compiled = jax.jit(actual)
                initial = compiled(p,x,d)
                CS.TRIM_MAX, CS.TRIM_POW = 1.7, 4
                CS.WIN_SHAPE, CS.FREEZE_W = 'smooth7', True
                CS.CLIP_R, CS.CLIP_BETA = 0.9, 0.4
                assert_exact(self, compiled(p,x,d), initial)
                with patch.object(attention, 'screening_attention', old.screening_attention):
                    expected = jax.jit(lambda pp,xx,dd: actual(pp,xx,dd))(p,x,d)
                observed = jax.jit(lambda pp,xx,dd: actual(pp,xx,dd))(p,x,d)
                assert_exact(self, observed, expected)
                self.assertFalse(np.array_equal(observed,initial))


if __name__ == '__main__':
    unittest.main(verbosity=2)
