"""LayerNorm移動の局所検証。実行: .venv-pw/bin/python -m tests.check_layernorm"""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.support import ROOT, ROUTES, make_case, require_numerics

require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import checkpoint, CustomPsiformer, CustomScreenformer, SlaterNet
from periodicwave import network_blocks


def reference_layer_norm():
    """移動前の演算・初期化を固定した参照（本体とは独立）。"""
    def init(param_shape: int):
        params = {}
        params['scale'] = jnp.ones(param_shape)
        params['offset'] = jnp.zeros(param_shape)
        return params

    def apply(params, inputs: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=axis, keepdims=True)
        variance = jnp.var(inputs, axis=axis, keepdims=True)
        eps = 1e-5
        inv = params['scale'] * jax.lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + params['offset']

    return init, apply


def assert_exact_tree(test, left, right):
    test.assertEqual(jax.tree_util.tree_structure(left), jax.tree_util.tree_structure(right))
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        test.assertEqual(a.dtype, b.dtype)
        np.testing.assert_array_equal(a, b)


class LayerNormChecks(unittest.TestCase):
    def test_public_names(self):
        for module in (CustomPsiformer, CustomScreenformer, SlaterNet):
            self.assertIs(module.make_layer_norm, network_blocks.make_layer_norm)
        init, apply = network_blocks.make_layer_norm()
        self.assertEqual(list(inspect.signature(init).parameters), ['param_shape'])
        self.assertEqual(list(inspect.signature(apply).parameters), ['params', 'inputs', 'axis'])
        self.assertEqual(inspect.signature(apply).parameters['axis'].default, -1)

    def test_both_precisions_and_derivatives(self):
        try:
            for x64 in (False, True):
                jax.config.update('jax_enable_x64', x64)
                old_init, old_apply = reference_layer_norm()
                init, apply = network_blocks.make_layer_norm()
                for axis, shape in ((-1, 8), (0, (2, 1))):
                    with self.subTest(x64=x64, axis=axis):
                        p, old_p = init(shape), old_init(shape)
                        assert_exact_tree(self, p, old_p)
                        x = jnp.arange(16).reshape(2, 8) / 7
                        # 非自明なscale/offsetでも比較する。
                        p = dict(scale=p['scale'] * 1.2, offset=p['offset'] + 0.3)
                        assert_exact_tree(self, apply(p, x, axis), old_apply(p, x, axis))
                        compiled = jax.jit(lambda p,x: apply(p,x,axis))
                        old_compiled = jax.jit(lambda p,x: old_apply(p,x,axis))
                        assert_exact_tree(self, compiled(p,x), old_compiled(p,x))
                        objective = lambda f: lambda p,x: jnp.sum(jnp.sin(f(p,x,axis)))
                        for transform in (jax.grad, jax.hessian):
                            assert_exact_tree(self, transform(objective(apply), argnums=1)(p,x), transform(objective(old_apply), argnums=1)(p,x))
                        assert_exact_tree(self, jax.grad(objective(apply))(p,x), jax.grad(objective(old_apply))(p,x))
                        self.assertEqual(str(jax.make_jaxpr(lambda p,x: apply(p,x,axis))(p,x)), str(jax.make_jaxpr(lambda p,x: old_apply(p,x,axis))(p,x)))
        finally:
            jax.config.update('jax_enable_x64', True)

    def test_network_jit_both_precisions(self):
        # 同じ入力で移動前の式を差し込んだネットワークとJIT経路を比較する。
        try:
            for x64 in (False, True):
                jax.config.update('jax_enable_x64', x64)
                for route in ROUTES:
                    with self.subTest(x64=x64, route=route), tempfile.TemporaryDirectory() as directory:
                        net, params, data, _, _, _ = make_case(route, Path(directory))
                        with patch.object(network_blocks, 'make_layer_norm', reference_layer_norm):
                            old_net, old_params, _, _, _, _ = make_case(route, Path(directory))
                        assert_exact_tree(self, params, old_params)
                        args = (params, data.positions, data.spins, data.atoms, data.charges)
                        assert_exact_tree(self, jax.jit(net.apply)(*args), jax.jit(old_net.apply)(*args))
        finally:
            jax.config.update('jax_enable_x64', True)

    def test_pre_move_checkpoints(self):
        root = ROOT / 'tests/layernorm_validation/before-checkpoints'
        for record in json.loads((root / 'evaluations.json').read_text()):
            with self.subTest(route=record['route'], x64=record['x64']), tempfile.TemporaryDirectory() as directory:
                jax.config.update('jax_enable_x64', record['x64'])
                path = ROOT / record['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])
                net, initial, _, _, _, _ = make_case(record['route'], Path(directory))
                t, data, params, state, width = checkpoint.restore(str(path), batch_size=2)
                params = jax.tree_util.tree_map(lambda a: jnp.asarray(a[0]), params)
                assert_exact_tree(self, params, initial)
                data = jax.tree_util.tree_map(lambda a: jnp.asarray(a[0, 0]), data)
                phase, logabs = net.apply(params, data.positions, data.spins, data.atoms, data.charges)
                self.assertEqual(float(phase), record['phase'])
                self.assertEqual(float(logabs), record['logabs'])
                self.assertEqual(t, 1)
                self.assertIsNone(state)
        jax.config.update('jax_enable_x64', True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
