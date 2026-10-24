"""段階Aの局所検証: .venv-pw/bin/python -m tests.check_window（固定基準は更新しない）。"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.support import make_case, require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import attention, CustomScreenformer as CS
from tests import window_reference as old

MODES = ('cos', 'smooth7', 'band', 'sin', 'gauss')


def parameters(mode):
    # 学習本体と同じ明示float32の窓パラメータ。距離の精度は別に切り替える。
    p = dict(mask_omega=jnp.asarray([-0.2, 0.3], dtype=jnp.float32))
    if mode == 'band':
        p['mask_centre'] = jnp.asarray([0., 0.6], dtype=jnp.float32)
    if mode in ('sin', 'gauss'):
        p.update(mask_rings=jnp.asarray([0., 0.5, 1.], dtype=jnp.float32),
                 mask_envp=jnp.asarray(1. if mode == 'sin' else 3., dtype=jnp.float32))
    if mode == 'gauss':
        p['mask_gauss'] = jnp.asarray(1., dtype=jnp.float32)
    return p


def assert_exact(test, left, right):
    test.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        test.assertEqual(a.shape, b.shape)
        test.assertEqual(a.dtype, b.dtype)
        test.assertTrue(np.all(np.isfinite(a)))
        test.assertTrue(np.all(np.isfinite(b)))
        np.testing.assert_array_equal(a, b)


def boundary_points(p, dtype):
    # 原点・幅・帯窓端・輪境界と遷移帯の端・d_refの両側を検査。
    w = np.asarray(1.4 * jax.nn.softplus(p['mask_omega']))
    edges = [0., 0.37, 1.4, *w]
    if 'mask_centre' in p:
        c = 1.4 * np.asarray(p['mask_centre'])
        edges.extend([*c, *(c - w), *(c + w)])
    if 'mask_rings' in p:
        b = 1.4 * np.asarray(p['mask_rings'])
        edges.extend([*b, *(b - w[0]), *(b + w[0])])
    values = []
    for edge in np.asarray(edges, dtype=dtype):
        values.extend([edge, np.nextafter(edge, dtype(-np.inf)), np.nextafter(edge, dtype(np.inf)),
                       dtype(edge - 1e-4), dtype(edge + 1e-4)])
    return jnp.asarray(np.unique(np.maximum(np.asarray(values, dtype=dtype), 0.)))


class WindowChecks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64', True)

    def test_values_coordinate_derivatives_and_jit_at_boundaries(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for mode in MODES:
                for freeze in (False, True):
                    with self.subTest(x64=x64, mode=mode, freeze=freeze):
                        p = parameters(mode)
                        kw = dict(win_shape='smooth7' if mode == 'smooth7' else 'cos', freeze_w=freeze)
                        def evaluate(fn):
                            f = lambda d: fn(p, d[None, None], 1.4, **kw)[:, 0, 0]
                            return lambda d: (f(d), jax.jacfwd(f)(d), jax.jacfwd(jax.jacfwd(f))(d))
                        points = boundary_points(p, np.float64 if x64 else np.float32)
                        assert_exact(self, evaluate(attention.screening_window)(points[1]), evaluate(old.screening_window)(points[1]))
                        actual = jax.jit(jax.vmap(evaluate(attention.screening_window)))(points)
                        expected = jax.jit(jax.vmap(evaluate(old.screening_window)))(points)
                        assert_exact(self, actual, expected)

    def test_parameter_gradient_hessian_and_freeze(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for mode in MODES:
                for freeze in (False, True):
                    with self.subTest(x64=x64, mode=mode, freeze=freeze):
                        p = parameters(mode)
                        kw = dict(win_shape='smooth7' if mode == 'smooth7' else 'cos', freeze_w=freeze)
                        d = jnp.asarray([[0., 0.37], [0.37, 0.81]])
                        def objective(fn):
                            return lambda pp: jnp.sum(jnp.sin(fn(pp, d, 1.4, **kw)) * jnp.asarray([0.7, 1.3])[:, None, None])
                        grad = jax.grad(objective(attention.screening_window))(p)
                        assert_exact(self, grad, jax.grad(objective(old.screening_window))(p))
                        assert_exact(self, jax.hessian(objective(attention.screening_window))(p), jax.hessian(objective(old.screening_window))(p))
                        for name, value in grad.items():
                            if freeze or name in ('mask_envp', 'mask_gauss'):
                                np.testing.assert_array_equal(value, jnp.zeros_like(value))
                        if not freeze:
                            self.assertGreater(float(jnp.linalg.norm(grad['mask_omega'])), 0.)

    def test_global_settings_and_jit_trace_timing(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            with patch.multiple(CS, WIN_SHAPE='cos', FREEZE_W=False, TRIM_MAX=2., TRIM_POW=2, CLIP_R=None):
                init, apply = CS.make_screening_attention(2, 8, 4, 4, 3, 1.4)
                p = init(jax.random.PRNGKey(29))
                x = jnp.arange(24).reshape(3, 8) / 13. - 0.7
                d = jnp.asarray([[0., 0.3, 1.1], [0.3, 0., 0.8], [1.1, 0.8, 0.]])
                fn = lambda pp, dd: jnp.sum(apply(pp, x, dd))
                eager = lambda: jax.value_and_grad(fn, argnums=(0, 1))(p, d)
                with patch.object(attention, 'screening_window', old.screening_window):
                    expected = eager()
                assert_exact(self, eager(), expected)
                compiled = jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))
                initial = compiled(p, d)
                CS.WIN_SHAPE, CS.FREEZE_W = 'smooth7', True
                CS.TRIM_MAX, CS.TRIM_POW, CS.CLIP_R = 1.7, 4, 0.9
                # キャッシュ済みJITは旧設定、新しいtraceは現在設定を読む。
                assert_exact(self, compiled(p, d), initial)
                fresh = lambda pp,dd: fn(pp,dd)
                with patch.object(attention, 'screening_window', old.screening_window):
                    expected = jax.jit(jax.value_and_grad(fresh, argnums=(0,1)))(p,d)
                fresh_new = lambda pp,dd: fn(pp,dd)
                assert_exact(self, jax.jit(jax.value_and_grad(fresh_new, argnums=(0,1)))(p,d), expected)
                self.assertFalse(np.array_equal(initial[0], expected[0]))
                np.testing.assert_array_equal(expected[1][0]['mask_omega'], 0.)
                with patch.object(attention, 'screening_window', old.screening_window):
                    expected_eager = eager()
                assert_exact(self, eager(), expected_eager)

    def test_network_jit_and_derivatives(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for route in ('screen', 'lambda', 'hf'):
                with self.subTest(x64=x64, route=route), tempfile.TemporaryDirectory() as directory:
                    net, p, data, _, _, _ = make_case(route, Path(directory))
                    def evaluate(pp, x):
                        wave = lambda p_,x_: net.apply(p_, x_, data.spins, data.atoms, data.charges)
                        scalar = lambda p_,x_: (lambda ph,la: ph + 0.37*la)(*wave(p_,x_))
                        return (wave(pp,x), jax.jacrev(wave, argnums=1)(pp,x),
                                jax.hessian(scalar, argnums=1)(pp,x), jax.grad(scalar)(pp,x))
                    with patch.object(attention, 'screening_window', old.screening_window):
                        expected = jax.jit(lambda pp,x: evaluate(pp,x))(p,data.positions)
                    actual = jax.jit(lambda pp,x: evaluate(pp,x))(p,data.positions)
                    assert_exact(self, actual, expected)

    def test_gaussian_folx(self):
        import folx
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            p = parameters('gauss')
            def evaluate(fn):
                f = lambda d: fn(p,d.reshape(2,2),1.4,win_shape='cos',freeze_w=False)
                return folx.forward_laplacian(f, sparsity_threshold=0)(jnp.asarray([0.,0.37,0.37,0.81]))
            actual, expected = evaluate(attention.screening_window), evaluate(old.screening_window)
            assert_exact(self, actual.x, expected.x)
            assert_exact(self, actual.laplacian, expected.laplacian)

    def test_no_window_and_legacy_helper_names(self):
        self.assertIs(CS._smooth7, attention._smooth7)
        self.assertIs(CS._ring_steps, attention._ring_steps)
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            init, apply = CS.make_screening_attention(2, 8, 4, 4, 3, 1.4, use_softmask=False)
            p = init(jax.random.PRNGKey(29))
            self.assertNotIn('mask_omega', p)
            x = jnp.arange(24).reshape(3, 8) / 13. - 0.7
            d = jnp.zeros((3, 3))
            expected = apply(p, x, d)
            with patch.object(attention, 'screening_window', side_effect=AssertionError('Windowなしで呼ばれた')):
                assert_exact(self, apply(p,x,d), expected)
                self.assertTrue(np.all(np.isfinite(jax.jit(apply)(p,x,d))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
