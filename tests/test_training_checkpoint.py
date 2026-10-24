"""[統合/同等性] 実際のtrainerで1 step。一時フォルダ以外には保存しない。"""
import json
import os
from pathlib import Path
import tempfile
import unittest

from tests.support import require_numerics, make_case, ROUTES, batch_data, assert_tree_close
require_numerics()

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from periodicwave import base_config, checkpoint, train, train_screen
from periodicwave.utils import custom_logging


def training_config(route, directory, optimizer="adam"):
    net, params, data, lattice, options, _ = make_case(route, directory)
    cfg = base_config.default()
    cfg.batch_size = 2
    cfg.debug.deterministic = True
    cfg.optim.optimizer = optimizer
    cfg.optim.iterations = 1
    cfg.optim.clip_local_energy = 0.
    cfg.mcmc.burn_in = 0
    cfg.mcmc.steps = 1
    cfg.mcmc.move_width_updater = "const"
    cfg.system.electrons = (2, 0) if route in ("lambda", "hf") else (1, 1)
    cfg.system.ndim = 2
    cfg.system.pbc_lattice = np.asarray(lattice)
    cfg.system.make_local_energy_fn = ("periodicwave.pbc.lambda_jellium.local_energy"
                                       if route in ("lambda", "hf")
                                       else "periodicwave.pbc.hamiltonian.local_energy")
    cfg.system.make_local_energy_kwargs = {
        "lattice": np.asarray(lattice), "convergence_radius": 2, "potential_type": "Coulomb",
        "potential_kwargs": {"interaction_energy_scale": 2.},
        "kinetic_kwargs": {"laplacian_method": "folx"}}
    if route in ("lambda", "hf"):
        cfg.system.make_local_energy_kwargs.update(lam=0.4, delta=2., flip_batch=1)
        cfg.system.pseudospin = True
        cfg.system.init_pseudospin = "random"
        cfg.mcmc.spin_flips = 1
        cfg.optim.local_energy_chunk = 1
    cfg.network.complex = True
    cfg.network.determinants = 2
    cfg.network.jastrow = "NONE"
    cfg.network.jastrow_kwargs = {"ndim": 2}
    cfg.network.make_feature_layer_fn = "periodicwave.pbc.feature_layer.make_pbc_feature_layer"
    cfg.network.make_feature_layer_kwargs = {"lattice": np.asarray(lattice), "include_r_ae": False}
    cfg.network.make_envelope_fn = "periodicwave.envelopes.make_null_envelope"
    name = {"psiformer": "CustomPsiformer", "slater": "SlaterNet"}.get(route, "CustomScreenformer")
    cfg.network.network_type = name
    excluded = {"ndim", "determinants", "envelope", "feature_layer", "jastrow", "jastrow_kwargs",
                "complex_output", "bias_orbitals", "pbc_lattice"}
    cfg.network[name] = ml_collections.ConfigDict({k: v for k, v in options.items() if k not in excluded})
    cfg.log.save_path = str(Path(directory) / (route + "-" + optimizer))
    return cfg


class CheckpointContract(unittest.TestCase):
    def test_roundtrip_fields_and_completed_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, params, d, _, _, _ = make_case("hf", tmp)
            batch = batch_data(d)
            devices = jax.device_count()
            data = jax.tree_util.tree_map(lambda x: jnp.stack([x]*devices), batch)
            params = jax.tree_util.tree_map(lambda x: jnp.stack([x]*devices), params)
            width = jnp.full((devices,), 0.2)
            path = checkpoint.save(tmp, 3, data, params, None, width)
            with np.load(path, allow_pickle=True) as z:
                self.assertEqual(set(z.files), {"t", "data", "params", "opt_state", "mcmc_width"})
            t, restored_data, restored_params, state, restored_width = checkpoint.restore(path, devices*2)
            self.assertEqual(t, 4)
            self.assertIsNone(state)
            assert_tree_close(self, data, restored_data, rtol=0, atol=0)
            assert_tree_close(self, params, restored_params, rtol=0, atol=0)
            np.testing.assert_array_equal(width, restored_width)

    def test_current_config_array_serialization_is_lossy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ml_collections.ConfigDict({"lattice": np.eye(2)})
            path = Path(tmp) / "config.json"
            custom_logging.save_config_dict_as_json(cfg, str(path))
            saved = json.loads(path.read_text())
            self.assertIsInstance(saved["lattice"], str)
            self.assertIn("unserializable object", saved["lattice"])


class TrainingIntegration(unittest.TestCase):
    def run_routes(self, optimizer):
        if jax.device_count() != 1:
            self.skipTest("この小系の統合試験は単一device用。多GPUは別途実機検証")
        with tempfile.TemporaryDirectory(prefix="pw-training-") as tmp:
            for route in ROUTES:
                with self.subTest(route=route, optimizer=optimizer):
                    cfg = training_config(route, tmp, optimizer)
                    module = train if route in ("psiformer", "slater") else train_screen
                    module.train(cfg)
                    path = Path(cfg.log.save_path) / "qmcjax_ckpt_000000.npz"
                    self.assertTrue(path.is_file())
                    t, data, p, state, width = checkpoint.restore(str(path), 2)
                    self.assertEqual(t, 1)
                    self.assertIsNotNone(state)
                    for leaf in jax.tree_util.tree_leaves((data, p, state, width)):
                        self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
                    stats = Path(cfg.log.save_path, "train_stats.csv").read_text().splitlines()
                    self.assertEqual(len(stats), 2)
                    self.assertIn("energy", stats[0])
                    if route in ("lambda", "hf"):
                        self.assertIn("pflip", stats[0])

    def test_adam_training_all_routes(self):
        self.run_routes("adam")

    @unittest.skipUnless(os.environ.get("PW_RUN_KFAC") == "1", "PW_RUN_KFAC=1 でCPU KFAC統合試験を実行")
    def test_kfac_training_all_routes(self):
        self.run_routes("kfac")
