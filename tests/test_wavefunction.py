"""[同等性/物理] 軌道、行列式、位相、反対称性、PBC、座標微分。"""
import os
import tempfile
import unittest

from tests.support import require_numerics, make_case, ROUTES, assert_tree_close
require_numerics()

import jax
import jax.numpy as jnp
import numpy as np
import folx


class WavefunctionChecks(unittest.TestCase):
    def test_values_orbitals_and_determinants(self):
        with tempfile.TemporaryDirectory() as tmp:
            for route in ROUTES:
                with self.subTest(route=route):
                    net, params, data, _, _, _ = make_case(route, tmp)
                    args = (data.positions, data.spins, data.atoms, data.charges)
                    phase, logabs = net.apply(params, *args)
                    orbitals = net.orbitals(params, *args)
                    self.assertEqual(len(orbitals), 1)
                    self.assertEqual(orbitals[0].shape, (2, 2, 2))
                    determinant_sum = np.linalg.det(np.asarray(orbitals[0])).sum()
                    np.testing.assert_allclose(jnp.exp(logabs + 1j * phase), determinant_sum,
                                               rtol=1e-9, atol=1e-10)
                    np.testing.assert_allclose(logabs, np.log(abs(determinant_sum)), atol=1e-9)
                    np.testing.assert_allclose(jnp.exp(1j * phase), determinant_sum / abs(determinant_sum), atol=1e-9)
                    assert_tree_close(self, params, net.init(jax.random.PRNGKey(7)), rtol=0, atol=0)

    def test_fermion_exchange_and_individual_electron_pbc(self):
        with tempfile.TemporaryDirectory() as tmp:
            for route in ROUTES:
                with self.subTest(route=route):
                    net, params, d, lattice, _, _ = make_case(route, tmp)
                    psi = lambda x, s: (lambda ph, la: jnp.exp(la + 1j * ph))(
                        *net.apply(params, x, s, d.atoms, d.charges))
                    original = psi(d.positions, d.spins)
                    exchange = psi(d.positions.reshape(2, 2)[::-1].reshape(-1), d.spins[::-1])
                    np.testing.assert_allclose(exchange, -original, rtol=1e-8, atol=1e-9)
                    for electron in range(2):
                        for axis in range(2):
                            shifted = d.positions.reshape(2, 2).at[electron].add(lattice[:, axis]).reshape(-1)
                            np.testing.assert_allclose(psi(shifted, d.spins), original, rtol=1e-8, atol=1e-9)

    def test_first_second_derivatives_and_forward_laplacian(self):
        with tempfile.TemporaryDirectory() as tmp:
            for route in ROUTES:
                with self.subTest(route=route):
                    net, params, d, _, _, _ = make_case(route, tmp)
                    for component in (0, 1):
                        f = lambda x: net.apply(params, x, d.spins, d.atoms, d.charges)[component]
                        grad, hess = jax.grad(f)(d.positions), jax.hessian(f)(d.positions)
                        forward = folx.forward_laplacian(f, sparsity_threshold=0)(d.positions)
                        np.testing.assert_allclose(forward.jacobian.dense_array.reshape(-1), grad,
                                                   rtol=1e-7, atol=1e-8)
                        np.testing.assert_allclose(forward.laplacian, jnp.trace(hess), rtol=1e-7, atol=1e-8)
                        h = 1e-4
                        delta = jnp.asarray([h, 0., 0., 0.])
                        first = (f(d.positions + delta) - f(d.positions - delta)) / (2 * h)
                        second = (f(d.positions + delta) - 2 * f(d.positions) + f(d.positions - delta)) / h**2
                        np.testing.assert_allclose(first, grad[0], rtol=2e-5, atol=2e-6)
                        np.testing.assert_allclose(second, hess[0, 0], rtol=2e-4, atol=2e-5)

    def test_hf_only_matches_synthetic_bloch_spinors(self):
        with tempfile.TemporaryDirectory() as tmp:
            net, p, d, _, options, _ = make_case("hf", tmp)
            from periodicwave import CustomScreenformer as CS
            reference_options = dict(options, lj_net_scale=0., lj_jastrow=False)
            hf_only = CS.make_fermi_net((2, 0), d.charges, **reference_options)
            table = np.loadtxt(options["lj_hf_path"])
            k = table[:, :2]
            r = np.asarray(d.positions).reshape(2, 2)
            u = np.stack([np.ones(2), 0.4 * (k[:, 0] + 1j * k[:, 1])])
            u /= np.sqrt(1 + 0.4**2 * np.sum(k * k, -1))[None]
            rows = np.where(np.asarray(d.spins)[:, None] > 0, u[0], u[1])
            expected = np.exp(1j * r @ k.T) * rows
            actual = hf_only.orbitals(p, d.positions, d.spins, d.atoms, d.charges)[0]
            np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-9)

    def test_frozen_numerical_baseline(self):
        directory = os.environ.get("PW_BASELINE_DIR")
        if not directory:
            self.skipTest("数値基準未指定。変更前に tests.baseline capture を実行する")
        from tests.baseline import numerical_snapshot, compare
        arrays, info = numerical_snapshot()
        # G2c changes reporting only: compare every G2b numerical value exactly.
        compare(directory, arrays, info, exact=True)
