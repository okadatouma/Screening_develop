"""[同等性] Attentionを小さい配列と独立したNumPy式で比較する。"""
import math
import unittest
from unittest.mock import patch

from tests.support import require_numerics
require_numerics()

import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomPsiformer as CP, CustomScreenformer as CS


def attention_values(mode):
    """本体と参照式の内部量。グローバル変更はこの呼出中だけに限定。"""
    x = jnp.asarray(np.arange(24).reshape(3, 8) / 13. - 0.7)
    d = jnp.asarray([[0., 0.3, 1.1], [0.3, 0., 0.8], [1.1, 0.8, 0.]])
    pair = jnp.stack([d, jnp.sin(d), jnp.cos(d), d * d, jnp.ones_like(d)], -1)
    power = 4 if mode == "pow4" else 2
    shape = "smooth7" if mode == "smooth7" else "cos"
    with patch.multiple(CS, TRIM_MAX=2., TRIM_POW=power, WIN_SHAPE=shape,
                        FREEZE_W=False, CLIP_R=None):
        if mode == "softmax":
            init, apply = CP.make_multi_head_attention(2, 8, 4, 4, 3)
        else:
            rings = ([0., 0.5, 1.], 0.16, 3., mode) if mode in ("sin", "gauss") else None
            centres = ([0., 0.6], [0.5, 0.5]) if mode == "band" else None
            init, apply = CS.make_screening_attention(
                2, 8, 4, 4, 3, 1.4, init_softmask_rings=rings, init_softmask_centres=centres,
                use_softmask=mode != "nomask", use_gate=mode != "nogate", use_tanh_norm=mode != "nonorm",
                use_fixed_value=mode == "fixed", pair_dim=5)
        params = init(jax.random.PRNGKey(29))
        actual = apply(params, x) if mode == "softmax" else apply(params, x, d, pair)
        p = {k: np.asarray(v) for k, v in params.items()}
        xx, dd = np.asarray(x), np.asarray(d)
        q = (xx @ p["q_w"]).reshape(3, 2, 4)
        k = (xx @ p["k_w"]).reshape(3, 2, 4)
        unit = lambda a: a / np.sqrt(np.sum(a * a, -1, keepdims=True) + 1e-12)
        v = None if mode == "fixed" else (xx @ p["v_w"]).reshape(3, 2, 4)
        if mode == "softmax":
            similarity = np.einsum("ihd,jhd->hij", q, k) / 2.
            exp_s = np.exp(similarity - similarity.max(-1, keepdims=True))
            alpha = exp_s / exp_s.sum(-1, keepdims=True)
            mask = np.ones_like(alpha)
        else:
            q, k = unit(q), unit(k)
            if v is not None:
                v = unit(v)
            similarity = np.einsum("ihd,jhd->hij", q, k)
            r = (2. / (1. + np.exp(-p["trim_rho"])))[:, None, None]
            alpha = (np.maximum(similarity - (1. - r), 0.) / r) ** power
            w = (None if mode == "nomask" else 1.4 * np.logaddexp(0., p["mask_omega"]))
            if mode == "nomask":
                mask = np.ones_like(alpha)
            elif mode in ("sin", "gauss"):
                b = 1.4 * p["mask_rings"][:, None, None]
                if mode == "gauss":
                    erf = np.vectorize(math.erf)
                    t = 1. + 0.5 * (erf((dd[None] - b) / (w[0] * np.sqrt(2)))
                                     + erf((-dd[None] - b) / (w[0] * np.sqrt(2))))
                else:
                    t = 0.5 * (1. + np.sin(np.pi / 2 * np.clip((dd[None] - b) / w[0], -1, 1)))
                t[0], t[-1] = 1., 0.
                g = 0.5 * (1. + np.cos(np.pi * np.clip(dd / 1.4, 0, 1) ** 3))
                mask = (t[:-1] - t[1:]) * g[None]
            else:
                t = (np.clip((dd[None] - 1.4 * p["mask_centre"][:, None, None]) / w[:, None, None], -1, 1)
                     if mode == "band" else np.clip(dd[None] / w[:, None, None], 0, 1))
                mask = (1. - (35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7)
                        if shape == "smooth7" else 0.5 * (1. + np.cos(np.pi * t)))
        weights = alpha * mask
        if mode == "fixed":
            weights = weights * (1. - np.eye(3))[None]
            aggregate = np.einsum("hij,ijd->ihd", weights, np.asarray(pair))
        else:
            aggregate = np.einsum("hij,jhd->ihd", weights, v)
        head = aggregate
        if mode not in ("softmax", "nonorm"):
            norm = np.sqrt(np.sum(head * head, -1, keepdims=True) + 1e-12)
            head = head * np.tanh(norm) / norm
        if mode not in ("softmax", "nogate"):
            gate = xx @ p["g_w"]
            silu = gate / (1. + np.exp(-gate))
            head = head * np.tanh(silu).reshape(head.shape)
        expected = head.reshape(3, -1) @ p["attn_output"]
        if mode != "softmax":
            expected = expected * np.exp(p["out_scale"])
        return dict(q=q, k=k, similarity=similarity, alpha=alpha, mask=mask,
                    weights=weights, aggregate=aggregate, head=head,
                    output=np.asarray(actual), reference=expected), params


class AttentionEquivalence(unittest.TestCase):
    def test_attention_equations_and_internal_values(self):
        for mode in ("softmax", "cos", "smooth7", "pow4", "band", "sin", "gauss", "fixed",
                     "nomask", "nogate", "nonorm"):
            with self.subTest(mode=mode):
                values, _ = attention_values(mode)
                np.testing.assert_allclose(values["output"], values["reference"], rtol=2e-6, atol=2e-7)
                for value in values.values():
                    self.assertTrue(np.all(np.isfinite(value)))

    def test_freeze_window_stops_parameter_gradient_only(self):
        with patch.multiple(CS, TRIM_MAX=2., TRIM_POW=2, WIN_SHAPE="cos", CLIP_R=None):
            init, apply = CS.make_screening_attention(2, 8, 4, 4, 3, 1.4)
            p = init(jax.random.PRNGKey(29))
            x = jnp.asarray(np.arange(24).reshape(3, 8) / 13. - 0.7)
            d = jnp.asarray([[0., 0.3, 1.1], [0.3, 0., 0.8], [1.1, 0.8, 0.]])
            with patch.object(CS, "FREEZE_W", False):
                before = apply(p, x, d)
                learned = jax.grad(lambda pp: jnp.sum(apply(pp, x, d)))(p)
            with patch.object(CS, "FREEZE_W", True):
                after = apply(p, x, d)
                frozen = jax.grad(lambda pp: jnp.sum(apply(pp, x, d)))(p)
            np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(frozen["mask_omega"], 0.)
            self.assertGreater(float(jnp.linalg.norm(learned["mask_omega"])), 1e-12)
            np.testing.assert_allclose(learned["q_w"], frozen["q_w"], atol=1e-12)

    def test_gaussian_ring_partition_and_zero_origin_slope(self):
        b = jnp.asarray([0., 0.5, 1.])
        def windows(x):
            steps = CS._ring_steps(x[None, None], b, 0.12, gauss=True)
            return (steps[:-1] - steps[1:])[:, 0, 0]
        for d in (0., 0.17, 0.5, 0.91, 1.4):
            result = windows(jnp.asarray(d))
            self.assertAlmostEqual(float(result.sum()), 1., places=12)
            self.assertGreaterEqual(float(result.min()), -1e-12)
        np.testing.assert_allclose(jax.jacfwd(windows)(jnp.asarray(0.)), 0., atol=1e-12)
