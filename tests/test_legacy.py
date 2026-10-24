"""既存検算を独立プロセスで実行。図用の出力先も一時領域に隔離する。"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.support import ROOT, require_numerics


@unittest.skipUnless(os.environ.get("PW_RUN_LEGACY") == "1", "PW_RUN_LEGACY=1 で既存の長いCPU検算も実行")
class LegacyPhysics(unittest.TestCase):
    def run_script(self, name):
        require_numerics()
        with tempfile.TemporaryDirectory(prefix="pw-legacy-") as tmp:
            path = Path(tmp, name)
            shutil.copyfile(ROOT / name, path)
            Path(tmp, "screening_analysis").mkdir()
            env = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
            run = subprocess.run([sys.executable, str(path)], cwd=tmp, env=env,
                                 capture_output=True, text=True, timeout=900)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            # Gaussian旧版比較は参照ファイル未指定なので未実行（READMEに明記）。

    def test_existing_screenformer(self):
        self.run_script("test_screenformer.py")

    def test_existing_lambda_jellium(self):
        self.run_script("test_lambda_jellium.py")

    def test_existing_gaussian_rings(self):
        self.run_script("test_gauss_rings.py")
