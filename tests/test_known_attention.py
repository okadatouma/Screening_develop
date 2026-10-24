"""[解析/API] E1移行後のprobeを独立式と比較する。旧式との差はcheck_attn_probeで保存。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics, make_case, batch_data
require_numerics()

import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, networks
from screening_analysis import attn_probe


def probe_weights(power=2, shape="cos"):
    with tempfile.TemporaryDirectory(prefix="pw-probe-") as tmp:
        with patch.multiple(CS, TRIM_MAX=2., TRIM_POW=power, WIN_SHAPE=shape,
                            FREEZE_W=False, CLIP_R=None):
            net, params, d, lattice, _, _ = make_case("screen", tmp)
            options = dict(num_layers=1, num_heads=2, attn_dim=4, value_dim=4, mlp_dim=8,
                           num_perceptrons_per_layer=1, use_layer_norm=True,
                           mlp_activation_fct="GELU", use_softmask=True, use_gate=True,
                           use_tanh_norm=True, init_trim_r=1.9, init_softmask_w_factor=0.8,
                           init_softmask_w_spread=2.5, use_output_scale=True, init_output_scale=0.)
            cfg = {"system": {"ndim": 2, "electrons": [1, 1], "pbc_lattice":
                               "unserializable object: " + np.array2string(np.asarray(lattice), precision=17)},
                   "network": {"network_type": "CustomScreenformer", "CustomScreenformer": options,
                        "make_feature_layer_fn": "periodicwave.pbc.feature_layer.make_pbc_feature_layer",
                        "make_feature_layer_kwargs": {"lattice": np.asarray(lattice).tolist(), "include_r_ae": False}}}
            Path(tmp, "config.json").write_text(json.dumps(cfg))
            batch = batch_data(d)
            np.savez(Path(tmp, "qmcjax_ckpt_000000.npz"),
                     t=0,
                     params=jax.tree_util.tree_map(lambda a: np.asarray(a)[None], params),
                     data={name: np.asarray(value)[None] for name, value in dict(batch).items()})
            settings=dict(TRIM_MAX=2.,TRIM_POW=power,WIN_SHAPE=shape,FREEZE_W=False,
                          CLIP_R=None,CLIP_BETA=CS.CLIP_BETA,jax_enable_x64=bool(jax.config.x64_enabled),
                          jax_default_matmul_precision=jax.config.jax_default_matmul_precision)
            observed = attn_probe.probe(tmp, nconf=1,settings=settings,lattice=np.asarray(lattice))["W"][0, 0]
            construct = networks.make_periodic_construct_input_features(lattice)
            ae, ee, ra, re = construct(d.positions, d.atoms, ndim=2)
            one, _ = net.options.feature_layer.apply(ae=ae, ee=ee, r_ae=ra, r_ee=re)
            x = jnp.concatenate([one, d.spins[:, None]], -1) @ params["layers"]["embed"]
            p = params["layers"]["attention"][0]
            q = CS._unit_length((x @ p["q_w"]).reshape(2, 2, 4))
            k = CS._unit_length((x @ p["k_w"]).reshape(2, 2, 4))
            s = jnp.einsum("ihd,jhd->hij", q, k)
            r = (2 * jax.nn.sigmoid(p["trim_rho"]))[:, None, None]
            alpha = (jax.nn.relu(s - (1-r)) / r) ** power
            w = CS._max_periodic_distance(lattice) * jax.nn.softplus(p["mask_omega"])
            t = jnp.clip(re[..., 0][None] / w[:, None, None], 0., 1.)
            mask = (1 - (35*t**4 - 84*t**5 + 70*t**6 - 20*t**7) if shape == "smooth7"
                    else 0.5 * (1 + jnp.cos(jnp.pi*t)))
            return observed, np.asarray(alpha * mask)


class AnalysisAttention(unittest.TestCase):
    def test_default_analysis_weights_match(self):
        actual, expected = probe_weights()
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_K01_trim_power_four_analysis_should_match(self):
        actual, expected = probe_weights(power=4)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_K01_smooth7_analysis_should_match(self):
        actual, expected = probe_weights(shape="smooth7")
        np.testing.assert_allclose(actual, expected, atol=1e-9)
