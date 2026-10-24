"""窓研究の独立検算と小規模配線テスト。重い大学計算の成功を意味しない。"""
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from periodicwave.window_experiments import windows
from periodicwave.window_experiments.network import build
from periodicwave.window_experiments.run import parser, configuration, optimizer, save_state, restore_state, initial_data
from periodicwave.window_experiments.analyze import core_statistics, minimum_image_distance, blocking

jax.config.update('jax_enable_x64', os.environ.get('PW_TEST_X64', '1') == '1')


def config(**updates):
    c = configuration(parser().parse_args(['--smoke', '--out', '/unused']))
    c.update(updates)
    return c


class WindowExperiments(unittest.TestCase):
    def test_features(self):
        L = jnp.array([[3., -.8], [0., 2.5]])
        ae = jnp.array([[[.2, .4]], [[1.2, .8]]]); spins = jnp.array([1., -1.])
        for mode, dim in [('E0', 5), ('E1', 11), ('E2', 12), ('E3', 18)]:
            f = windows.input_features(ae, spins, L, mode)
            self.assertEqual(f.shape, (2, dim))
            np.testing.assert_allclose(windows.input_features(ae+L[:, 0], spins, L, mode), f,
                                       atol=3e-5 if f.dtype == jnp.float32 else 1e-13)
            np.testing.assert_array_equal(windows.input_features(ae[::-1], spins[::-1], L, mode), f[::-1])
        e3 = windows.input_features(ae, spins, L, 'E3')
        np.testing.assert_array_equal(e3[0, 12:], 0); np.testing.assert_array_equal(e3[1, 6:12], 0)

    def test_order_and_pair_symmetry(self):
        for mode in ('W1', 'W2'):
            c = config(window=mode, pair_windows=True, heads=8, min_gap_fraction=.02)
            p = windows.initialize(c, 3., jnp.float32)
            p['increments'] = jnp.linspace(-4, 4, 8)
            p['pair_shift'] = jnp.linspace(-10, 10, 24).reshape(3, 8)
            s = np.asarray(windows.scales(p, c, 3.))
            self.assertTrue(np.all(np.diff(s, axis=1) > 0)); np.testing.assert_array_equal(s[:, -1], 3.)
            d = jnp.array([[0., .4], [.4, 0.]])
            m, _ = windows.evaluate(p, d, jnp.array([1., -1.]), c, 3.)
            np.testing.assert_array_equal(m[:, 0, 1], m[:, 1, 0])

    def test_window_reference_and_derivatives(self):
        dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        tol = 1e-12 if jax.config.x64_enabled else 2e-5
        for mode in ('W0', 'W1', 'W2', 'W3'):
            c = config(window=mode, basis_edges=[0., .5, 1., 3.])
            p = windows.initialize(c, 3., dtype)
            d = jnp.array([[0., .3], [.3, 0.]], dtype)
            m, _ = windows.evaluate(p, d, jnp.array([1., -1.]), c, 3.)
            self.assertTrue(np.all(np.asarray(m) >= 0))
            if mode == 'W2': np.testing.assert_allclose(m.sum(0), 1, atol=tol)
            if mode == 'W3': np.testing.assert_allclose(jax.nn.softmax(p['logits'], -1).sum(-1), 1, atol=tol)
            if mode in ('W0', 'W1'):
                scales = windows.scales(p, c, 3.)[0]
                ref = .5*(1+jnp.cos(jnp.pi*jnp.clip(d[None]/scales[:, None, None], 0, 1)))
                np.testing.assert_array_equal(m, ref)
            f = lambda x: windows.evaluate(p, jnp.array([[0., x], [x, 0.]]), jnp.array([1., -1.]), c, 3.)[0].sum()
            for x in (0., .2-1e-4, .2, .2+1e-4, .5):
                self.assertTrue(np.isfinite(jax.grad(f)(x)))
                self.assertTrue(np.isfinite(jax.grad(jax.grad(f))(x)))
                np.testing.assert_allclose(jax.jit(f)(x), f(x), atol=tol, rtol=tol)

    def test_core_cosine_definition(self):
        # i=0をcore Query、+対はj=1。i=2をcore Query、-対はj=3。
        pos = np.array([[[.05, 0], [1, 0], [2.05, 0], [3, 0]]])
        spins = np.array([[1., 1., -1., -1.]])
        C = np.zeros((1, 2, 4, 4)); C[:, :, 0, 1] = [.2, .3]; C[:, :, 2, 3] = [.8, -.2]
        # 大きい自己項・非core Queryが統計へ混入しない。
        C[:, :, 0, 0] = 999; C[:, :, 3, 2] = 888
        m = np.ones_like(C); m[0, 1, 2, 3] = 0
        result = core_statistics(C, C**2, 2*C, m, pos, spins, np.eye(2)*4, [[0, 0], [2, 0]], .1)
        np.testing.assert_allclose(result['core_C_pp'], [.2, .3])
        self.assertAlmostEqual(result['core_delta_C'][0], .6)
        self.assertTrue(np.isnan(result['core_C_mm'][1]))
        self.assertAlmostEqual(result['core_alpha_mm'][0], .64)
        self.assertAlmostEqual(result['core_W_aggregation_mm'][0], 1.6)

    def test_core_periodic_oblique(self):
        L = np.array([[2., -1.], [0., np.sqrt(3)]])
        p = np.array([[.01, .02], [1.4, 1.]])
        d = minimum_image_distance(p, [[0, 0]], L)
        np.testing.assert_allclose(minimum_image_distance(p+L[:, 1], [[0, 0]], L), d, atol=1e-14)
        brute = [min(np.linalg.norm(x-L@np.array([i, j])) for i in range(-3, 4) for j in range(-3, 4)) for x in p]
        np.testing.assert_allclose(d, brute, atol=1e-14)

    def test_core_uses_window_not_alpha_or_head_output(self):
        pos = np.array([[[0, 0], [1, 0], [2, 0], [3, 0]]], dtype=float)
        s = np.array([[1., 1., 1., -1.]])
        C = np.zeros((1, 1, 4, 4)); C[0, 0, 0, 1:3] = [.2, .8]
        m = np.ones_like(C); m[0, 0, 0, 2] = 3
        result = core_statistics(C, C**4, m*C**4, m, pos, s, np.eye(2)*5, [[0, 0]], .1)
        self.assertAlmostEqual(result['core_C_pp'][0], .65)
        self.assertTrue(np.isnan(result['core_delta_C'][0]))

    def test_all_input_window_forwards(self):
        for mode in ('E0', 'E1', 'E2', 'E3'):
            for win in ('W0', 'W1', 'W2', 'W3'):
                c = config(input=mode, window=win)
                net, diagnose, _ = build(c); params = net.init(jax.random.PRNGKey(7))
                data = initial_data(c, jax.random.PRNGKey(9)); args = [v[0] for v in (data.positions, data.spins, data.atoms, data.charges)]
                phase, value = jax.jit(net.apply)(params, *args)
                self.assertTrue(np.isfinite(value)); self.assertTrue(np.isfinite(phase))
                _, layers, f = jax.jit(diagnose)(params, *args)
                self.assertEqual(len(layers), c['layers']); self.assertEqual(f.shape[0], c['n'])
                self.assertEqual(layers[0]['attention']['window'].shape, (c['heads'], c['n'], c['n']))

    def test_fixed_value_position_gate_derivatives(self):
        c = config(input='E3', window='W2', value='pair', gate='position', pair_windows=True, layers=2)
        net, _, _ = build(c); params = net.init(jax.random.PRNGKey(7))
        d = initial_data(c, jax.random.PRNGKey(9)); x, s, a, z = d.positions[0], d.spins[0], d.atoms[0], d.charges[0]
        f = lambda y: net.apply(params, y, s, a, z)[1]
        self.assertTrue(np.isfinite(jax.jit(jax.grad(f))(x)).all())
        self.assertTrue(np.isfinite(jax.jit(jax.hessian(f))(x)).all())
        grad = jax.jit(jax.grad(lambda p: net.apply(p, x, s, a, z)[1]))(params)
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
        self.assertGreater(np.linalg.norm(grad['layers']['qk_extra']), 0)

    def test_optimizer_freeze_release_and_checkpoint(self):
        c = config(); net, _, _ = build(c); p = net.init(jax.random.PRNGKey(3))
        tx = optimizer(c, p); state = tx.init(p); grad = jax.tree.map(jnp.ones_like, p)
        updates, state = tx.update(grad, state, p); new = optax.apply_updates(p, updates)
        oldw = p['layers']['attention'][0]['window']['increments']
        np.testing.assert_array_equal(oldw, new['layers']['attention'][0]['window']['increments'])
        updates, state = tx.update(grad, state, new); new = optax.apply_updates(new, updates)
        self.assertFalse(np.array_equal(oldw, new['layers']['attention'][0]['window']['increments']))
        data = initial_data(c, jax.random.PRNGKey(5)); key = jax.random.PRNGKey(17)
        with tempfile.TemporaryDirectory() as tmp:
            save_state(Path(tmp), 1, data, new, state, .2, key)
            step, dr, pr, sr, width, kr = restore_state(Path(tmp), c['batch'])
            self.assertEqual(step, 2); np.testing.assert_array_equal(kr, key)
            for x, y in zip(jax.tree.leaves((data, new, state)), jax.tree.leaves((dr, pr, sr))):
                np.testing.assert_array_equal(x, y)
            ur, _ = tx.update(grad, sr, pr); u, _ = tx.update(grad, state, new)
            for x, y in zip(jax.tree.leaves(ur), jax.tree.leaves(u)): np.testing.assert_array_equal(x, y)

    def test_hf_metadata_rejects_wrong_lambda(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp)/'hf.txt'; np.savetxt(table, [[0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0]])
            meta = Path(tmp)/'hf.json'; meta.write_text(json.dumps({'lam': .64}))
            args = parser().parse_args(['--smoke', '--out', tmp, '--hf', str(table), '--hf-meta', str(meta), '--label', 'WC'])
            with self.assertRaisesRegex(ValueError, 'lam'): configuration(args)

    def test_blocking_does_not_call_std_sem(self):
        result = blocking(np.ones((4, 2)))
        self.assertIsNone(result['sem_blocking']); self.assertFalse(result['plateau'])

    def test_suite_preserves_paired_conditions_and_fails_on_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp)/'plan.json'
            args = [sys.executable, '-m', 'periodicwave.window_experiments.suite', '--plan', str(plan),
                '--out-root', str(Path(tmp)/'runs'), '--wc-hf', '/university/WC.txt', '--ahc-hf', '/university/AHC.txt',
                '--wc-meta', '/university/WC.json', '--ahc-meta', '/university/AHC.json', '--phase', 'inputs']
            r = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            jobs = json.loads(plan.read_text())['jobs']; self.assertEqual(len(jobs), 24)
            self.assertTrue(all('--freeze-window' in j['argv'] and '--require-gpu' in j['argv'] for j in jobs))
            self.assertEqual(len({j['argv'][j['argv'].index('--out')+1] for j in jobs}), 24)
            r = subprocess.run(args+['--out', '/forbidden'], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)

    def test_missing_hf_and_incomplete_core_stop(self):
        with self.assertRaises(ValueError):
            configuration(parser().parse_args(['--out', '/unused', '--lam', '.64']))
        with self.assertRaises(ValueError):
            minimum_image_distance([[0., 0.]], [[0., 0.]], np.zeros((2, 2)))


if __name__ == '__main__': unittest.main()
