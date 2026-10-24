"""[静的契約] 依存なしで実行。数値計算が成功したことを意味しない。"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.baseline import metadata, source_inventory
from tests.support import ROOT, PACKAGES


def source(path):
    return ast.parse(ROOT.joinpath(path).read_text())


def function(path, name):
    return next(n for n in source(path).body if isinstance(n, ast.FunctionDef) and n.name == name)


def block_signature():
    tree = source("periodicwave/mcmc.py")
    accept = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mh_accept")
    block = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mh_block_update")
    call = next(n for n in ast.walk(block) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "mh_accept")
    signature = inspect.Signature([inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                   for a in accept.args.args])
    return signature, len(call.args)


def deterministic_seed_expression(path):
    tree = source(path)
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "cfg.debug.deterministic")
    assignment = next(n for n in branch.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "seed" for t in n.targets))
    return ast.Expression(assignment.value)


class ReproducibilityContracts(unittest.TestCase):
    def test_recorded_source_manifest_has_baseline_identity(self):
        recorded = json.loads(ROOT.joinpath("tests/source_baseline.json").read_text())
        self.assertEqual(recorded["commit"], "038730ef3bd06e738356e2a6f8d8b0adbb3e77cc")
        self.assertEqual(len(recorded["sha256"]), 170)
        for name, value in recorded["sha256"].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(len(value), 64)
        # ソースの同一hashを日常の数値同等性テストの条件にはしない。

    def test_all_current_runtime_sources_parse(self):
        inventory = source_inventory()
        self.assertTrue(inventory["sha256"])
        self.assertIn("periodicwave/pbc/lambda_jellium.py", inventory["sha256"])

    def test_representative_entry_routes(self):
        routes = {"periodicwave/configs/run_2deg.py": "train",
                  "periodicwave/configs/run_2deg_screen.py": "train_screen",
                  "periodicwave/configs/run_lambda_jellium.py": "train_screen"}
        for path, trainer in routes.items():
            imports = [alias.name for node in source(path).body if isinstance(node, ast.ImportFrom)
                       and node.module == "periodicwave" for alias in node.names]
            self.assertIn(trainer, imports, path)
        text = ROOT.joinpath("periodicwave/configs/run_lambda_jellium.py").read_text()
        self.assertIn("periodicwave.pbc.lambda_jellium.local_energy", text)
        self.assertIn("PW_LJ_HF", text)

    def test_environment_inventory_covers_hidden_physics_switches(self):
        names = set(source_inventory()["pw_environment_names"])
        self.assertTrue({"PW_WIN_SHAPE", "PW_TRIM_POW", "PW_FREEZE_W", "PW_SEED",
                         "PW_LJ_HF", "PW_FLIPS", "PW_EL_CHUNK", "PW_DELTA_MULT"} <= names)

    def test_metadata_reports_missing_packages_without_importing_them(self):
        info = metadata()
        self.assertEqual(set(info["versions"]), set(PACKAGES))
        self.assertIn("PW_WIN_SHAPE", info["environment"])
        self.assertIn("python", info)

    def test_existing_baseline_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp, "research-data.txt")
            sentinel.write_text("keep")
            before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
            result = subprocess.run([sys.executable, "-m", "tests.baseline", "capture", tmp],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("上書きしません", result.stderr)
            self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(), before)

    def test_checkpoint_schema_excludes_rng_and_resumes_at_next_step(self):
        save = function("periodicwave/checkpoint.py", "save")
        call = next(n for n in ast.walk(save) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "savez")
        self.assertEqual({k.arg for k in call.keywords}, {"t", "data", "params", "opt_state", "mcmc_width"})
        restore = function("periodicwave/checkpoint.py", "restore")
        expression = next(n.value for n in ast.walk(restore) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "t" for t in n.targets))
        self.assertIsInstance(expression, ast.BinOp)
        self.assertIsInstance(expression.op, ast.Add)
        self.assertEqual(ast.literal_eval(expression.right), 1)

    def test_seed_override_differs_between_current_trainers(self):
        with patch.dict(os.environ, {"PW_SEED": "91"}):
            ordinary = eval(compile(deterministic_seed_expression("periodicwave/train.py"), "seed", "eval"), {"os": os})
            screen = eval(compile(deterministic_seed_expression("periodicwave/train_screen.py"), "seed", "eval"), {"os": os})
        self.assertEqual(ordinary, 42)
        self.assertEqual(screen, 91)

    def test_K02_actual_block_call_cannot_bind_to_accept_signature(self):
        signature, arguments = block_signature()
        with self.assertRaises(TypeError):
            signature.bind(*([None] * arguments))

    @unittest.expectedFailure
    def test_K02_block_call_should_satisfy_accept_interface(self):
        signature, arguments = block_signature()
        signature.bind(*([None] * arguments))

    def test_config_is_currently_saved_before_deterministic_override(self):
        tree = source("periodicwave/configs/run_lambda_jellium.py")
        save_line = next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "save_config_dict_as_json")
        override_line = next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign)
                             and any(ast.unparse(t) == "cfg.debug.deterministic" for t in n.targets))
        self.assertGreater(override_line, save_line)
