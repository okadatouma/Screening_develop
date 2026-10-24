"""[物理/同等性] G1の規格化VMC勾配と、変更していないMCMCの検証。"""
import unittest

from tests.support import require_numerics, assert_tree_close
require_numerics()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from periodicwave import loss, mcmc, networks, train, train_screen


def two_state_data():
    return networks.FermiNetData(positions=jnp.asarray([[-1., 0.], [1., 0.]]),
                               spins=jnp.ones((2, 1)), atoms=jnp.zeros((2, 1, 2)),
                               charges=jnp.zeros((2, 1)))


def objective(complex_output=False, chunk=0, clip=0.):
    network = lambda p, x, s, a, c: p["theta"] * x[0] * (1.+0j if complex_output else 1.)
    # 2状態のHermitian演算子 H=diag(-1,+1)。θ=0では両状態が等確率。
    local = lambda p, k, d: (d.positions[0] + (0j if complex_output else 0.), None)
    return loss.make_loss(network, local, complex_output=complex_output,
                          local_energy_chunk=chunk, clip_local_energy=clip,
                          clip_from_median=False)


def gradient(complex_output=False):
    f = objective(complex_output)
    return jax.grad(lambda p: f(p, jax.random.PRNGKey(0), two_state_data())[0])({"theta": jnp.asarray(0.)})["theta"]


class VMCEstimator(unittest.TestCase):
    def test_current_real_and_complex_gradient_coefficients(self):
        np.testing.assert_allclose(gradient(False), 2., atol=1e-12)
        np.testing.assert_allclose(gradient(True), 2., atol=1e-12)

    def test_complex_gradient_against_normalized_exact_expectation(self):
        def exact(theta):
            x = jnp.asarray([-1., 1.])
            weights = jnp.exp(2 * theta * x)
            return jnp.sum(weights * x) / weights.sum()
        np.testing.assert_allclose(gradient(True), jax.grad(exact)(0.), atol=1e-12)

    def test_K04_real_gradient_should_equal_normalized_exact_derivative(self):
        # E(θ)=tanh(2θ)、E'(0)=2。固定標本平均の差分ではない。
        np.testing.assert_allclose(gradient(False), 2., atol=1e-12)

    def test_imaginary_energy_contributes_to_phase_gradient(self):
        # H=[[0,i],[-i,0]]、ψ=(exp(-iθ),exp(iθ))。E=-sin(2θ)。
        network = lambda p, x, s, a, c: 1j * p["theta"] * x[0]
        local = lambda p, k, d: (-1j * d.positions[0], None)
        f = loss.make_loss(network, local, complex_output=True)
        got = jax.grad(lambda p: f(p, jax.random.PRNGKey(0), two_state_data())[0])({"theta": jnp.asarray(0.)})
        np.testing.assert_allclose(got["theta"], -2., atol=1e-12)

    def test_chunked_and_unbatched_loss_gradient_match(self):
        p, key, d = {"theta": jnp.asarray(0.)}, jax.random.PRNGKey(0), two_state_data()
        for complex_output in (False, True):
            f, g = objective(complex_output, 0), objective(complex_output, 1)
            assert_tree_close(self, f(p, key, d), g(p, key, d))
            assert_tree_close(self, jax.value_and_grad(f, has_aux=True)(p, key, d),
                              jax.value_and_grad(g, has_aux=True)(p, key, d))

    def test_clipping_changes_gradient_not_reported_mean(self):
        d, p, key = two_state_data(), {"theta": jnp.asarray(0.)}, jax.random.PRNGKey(0)
        d.positions = jnp.asarray([[-2., 0.], [5., 0.]])
        a, b = objective(False, clip=0.), objective(False, clip=0.2)
        np.testing.assert_allclose(a(p, key, d)[0], b(p, key, d)[0])
        self.assertGreater(abs(float(jax.grad(lambda pp: a(pp, key, d)[0])(p)["theta"]
                                     - jax.grad(lambda pp: b(pp, key, d)[0])(p)["theta"])), 0.1)

    def test_one_adam_update_both_trainers(self):
        f, d = objective(False), two_state_data()
        p, key = {"theta": jnp.asarray(0.)}, jax.random.PRNGKey(2)
        optimizer = optax.adam(0.01)
        state = optimizer.init(p)
        # 独立な厳密式 E(theta)=tanh(2 theta), E'(0)=2 を使う。
        updates, expected_state = optimizer.update({"theta": jnp.asarray(2.)}, state, p)
        expected = optax.apply_updates(p, updates)
        for module in (train, train_screen):
            with self.subTest(trainer=module.__name__):
                actual, actual_state, value, _ = module.make_opt_update_step(f, optimizer)(p, d, state, key)
                assert_tree_close(self, actual, expected)
                assert_tree_close(self, actual_state, expected_state)
                np.testing.assert_allclose(value, 0., atol=1e-12)


class MCMCChecks(unittest.TestCase):
    @staticmethod
    def logabs(p, x, s, a, c):
        return -0.5 * jnp.sum(x*x, axis=-1) + 0.2 * jnp.sum(s, axis=-1)

    def test_all_electron_move_against_explicit_proposal_and_acceptance(self):
        d = two_state_data()
        key = jax.random.PRNGKey(31)
        lp = 2 * self.logabs(None, d.positions, d.spins, d.atoms, d.charges)
        next_key, proposal_key = jax.random.split(key)
        proposed = d.positions + 0.2 * jax.random.normal(proposal_key, d.positions.shape)
        lp2 = 2 * self.logabs(None, proposed, d.spins, d.atoms, d.charges)
        final_key, accept_key = jax.random.split(next_key)
        accept = jnp.log(jax.random.uniform(accept_key, lp.shape)) < lp2 - lp
        actual, returned_key, actual_lp, count = mcmc.mh_update(
            None, self.logabs, d, key, lp, 0., stddev=0.2, ndim=2)
        np.testing.assert_array_equal(actual.positions, jnp.where(accept[:, None], proposed, d.positions))
        np.testing.assert_array_equal(actual.spins, d.spins)
        np.testing.assert_array_equal(actual_lp, jnp.where(accept, lp2, lp))
        np.testing.assert_array_equal(returned_key, final_key)
        self.assertEqual(float(count), float(accept.sum()))

    def test_pseudospin_flip_acceptance(self):
        d = two_state_data()
        key = jax.random.PRNGKey(37)
        lp = 2 * self.logabs(None, d.positions, d.spins, d.atoms, d.charges)
        key1, proposal_key = jax.random.split(key)
        idx = jax.random.randint(proposal_key, (2,), 0, 1)
        proposal = d.spins * (1 - 2 * jax.nn.one_hot(idx, 1, dtype=d.spins.dtype))
        lp2 = 2 * self.logabs(None, d.positions, proposal, d.atoms, d.charges)
        final_key, accept_key = jax.random.split(key1)
        accept = jnp.log(jax.random.uniform(accept_key, lp.shape)) < lp2 - lp
        actual, returned_key, actual_lp, count = mcmc.mh_spin_flip(None, self.logabs, d, key, lp, 0.)
        np.testing.assert_array_equal(actual.spins, jnp.where(accept[:, None], proposal, d.spins))
        np.testing.assert_array_equal(actual.positions, d.positions)
        np.testing.assert_array_equal(actual_lp, jnp.where(accept, lp2, lp))
        np.testing.assert_array_equal(returned_key, final_key)
        self.assertEqual(float(count), float(accept.sum()))

    def test_K02_block_move_currently_raises_argument_error(self):
        d = two_state_data()
        with self.assertRaisesRegex(TypeError, "mh_accept"):
            mcmc.make_mcmc_step(self.logabs, 2, steps=1, ndim=1, blocks=2)(
                None, d, jax.random.PRNGKey(0), 0.2)
