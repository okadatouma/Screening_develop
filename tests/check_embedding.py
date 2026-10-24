"""F1: compare learned embedding against archived pre-F1 source, exactly.

Current behavior tests; physical PBC/equivariance checks remain in check_interfaces.
No reference snapshots are regenerated and no module import/folx registration is replayed.
"""
import ast
from pathlib import Path
import tempfile
import unittest

from tests.support import require_numerics, make_case
require_numerics()
import jax
import jax.numpy as jnp
from periodicwave import network_blocks as NB, networks
from periodicwave import CustomPsiformer as CP, CustomScreenformer as CS, SlaterNet as SN
from tests.check_window import assert_exact


ARCHIVE = Path(__file__).parent / 'f1_validation'
ANSATZ = (('psiformer', CP, 'make_psiformer_layers'),
          ('screen', CS, 'make_screenformer_layers'),
          ('slater', SN, 'make_SlaterNet_layers'))


def previous_layers(module, name):
    """Load only the old factory; unchanged blocks/features use their real code."""
    source = (ARCHIVE / (module.__name__.split('.')[-1] + '-before.py.txt')).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.get_source_segment(source, node), str(ARCHIVE / name), 'exec'), namespace)
    return namespace[name]


def old_embedding(w, f, s):
    features = jnp.concatenate((f, s[..., None]), axis=-1)
    return jnp.dot(features, w)


class EmbeddingChecks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64', True)
        jax.clear_caches()

    def test_initializer_bare_weight_and_subkeys(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for features, width in ((4, 8), (5, 3), (12, 7)):
                with self.subTest(x64=x64, features=features, width=width):
                    key = jax.random.PRNGKey(17)
                    old = NB.init_linear_layer(key, features + 1, width, include_bias=False)['w']
                    new = NB.init_input_embedding(key, features, width)
                    assert_exact(self, old, new)
                    self.assertEqual(new.shape, (features + 1, width))

    def test_embedding_values_jit_derivatives_and_compilation(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for n, features, width in ((2, 4, 8), (3, 5, 3), (4, 12, 7)):
                with self.subTest(x64=x64, shape=(n, features, width)):
                    w = NB.init_input_embedding(jax.random.PRNGKey(17), features, width)
                    f = jnp.sin(jnp.arange(n * features).reshape(n, features) * .17)
                    s = jnp.where(jnp.arange(n) % 2, -1., 1.)
                    args = (w, f, s)
                    old, new = old_embedding, NB.apply_input_embedding
                    assert_exact(self, old(*args), new(*args))
                    assert_exact(self, jax.jit(old)(*args), jax.jit(new)(*args))
                    self.assertEqual(str(jax.make_jaxpr(old)(*args)), str(jax.make_jaxpr(new)(*args)))
                    def loss(fn):
                        return lambda ww, ff: jnp.sum(jnp.sin(fn(ww, ff, s)))
                    for arg in (0, 1):
                        assert_exact(self, jax.jit(jax.grad(loss(old), argnums=arg))(w, f),
                                     jax.jit(jax.grad(loss(new), argnums=arg))(w, f))
                    assert_exact(self, jax.jit(jax.hessian(loss(old), argnums=1))(w, f),
                                 jax.jit(jax.hessian(loss(new), argnums=1))(w, f))
                    # Same entry name avoids a textual module-name-only difference.
                    def entry(fn):
                        def evaluate(ww, ff, ss): return fn(ww, ff, ss)
                        return jax.jit(evaluate).lower(*args).compiler_ir(dialect='hlo').as_hlo_text()
                    self.assertEqual(entry(old), entry(new))

    def test_three_trunks_full_rng_tree_jit_coordinate_and_parameter_derivatives(self):
        for x64 in (False, True):
            jax.config.update('jax_enable_x64', x64)
            for route, module, maker in ANSATZ:
                with self.subTest(x64=x64, route=route), tempfile.TemporaryDirectory() as tmp:
                    net, full, data, lattice, _, _ = make_case(route, tmp)
                    old_init, old_apply = previous_layers(module, maker)((1, 1), 1, net.options)
                    new_init, new_apply = getattr(module, maker)((1, 1), 1, net.options)
                    key = jax.random.PRNGKey(23)
                    old_dim, old_params = old_init(key)
                    new_dim, p = new_init(key)
                    self.assertEqual(old_dim, new_dim)
                    assert_exact(self, old_params, p)  # all downstream draws, not just embed
                    self.assertEqual(p['input'], {})
                    self.assertEqual(p['embed'].ndim, 2)
                    geometry = networks.make_periodic_construct_input_features(lattice)
                    def evaluate(fn):
                        def apply(pp, r):
                            ae, ee, ra, re = geometry(r, data.atoms, ndim=2)
                            return fn(pp, ae=ae, ee=ee, r_ae=ra, r_ee=re,
                                      spins=data.spins, charges=data.charges)
                        return apply
                    old, new = evaluate(old_apply), evaluate(new_apply)
                    r = data.positions
                    assert_exact(self, old(p, r), new(p, r))
                    assert_exact(self, jax.jit(old)(p, r), jax.jit(new)(p, r))
                    def loss(fn): return lambda pp, rr: jnp.sum(jnp.sin(fn(pp, rr)))
                    for arg in (0, 1):
                        assert_exact(self, jax.jit(jax.grad(loss(old), argnums=arg))(p, r),
                                     jax.jit(jax.grad(loss(new), argnums=arg))(p, r))
                    assert_exact(self, jax.jit(jax.hessian(loss(old), argnums=1))(p, r),
                                 jax.jit(jax.hessian(loss(new), argnums=1))(p, r))
