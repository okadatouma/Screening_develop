"""Restart checks preserve old data and require a fresh challenge measurement."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from screening_analysis.converged_window_study import run_case, sampling_passes, validate_resume_output
from screening_analysis.restart_converged_window import prepare_restart, save


class RestartTests(unittest.TestCase):
    def fixture(self, root, drifting=False):
        (root/"source-production/screening_analysis").mkdir(parents=True)
        case=dict(id="test",n=9,rs=4.,lam=.72,seed=101,index=0,mode="learned",width_factor=1.,initialization="plane")
        plan=dict(cases=[case],energy_tolerance_Ha_per_electron=5e-5,
            minimum_stage_count=4,max_updates=400000,initial_stages=[dict(updates=30000,lr=.001)]*4,
            challenge_updates=10000,challenge_lr=.0001,extension_updates=40000,extension_lr=.00003,hf={},
            measurement=dict(max_rhat=1.05,min_ess=400,max_structure_rhat=1.1))
        save(root/"plan.json",plan)
        directory=root/"runs/test";directory.mkdir(parents=True)
        (directory/"reference").mkdir();(directory/"reference/orbitals.txt").write_text("reference")
        result=None
        for i in range(4):
            target=(i+1)*30000;stage=directory/f"stage-{i:02d}-{target:06d}";stage.mkdir()
            np.savez(stage/f"qmcjax_ckpt_{target-1:06d}.npz",t=target-1)
            save(stage/"stage_complete.json",dict(target=target))
            measurement=directory/(stage.name+"-measurement");measurement.mkdir()
            result=dict(step=target,checkpoint=str(stage/f"qmcjax_ckpt_{target-1:06d}.npz"),sampling_pass=False,
                energy_Ha=dict(mean=-1.+(.01*i if drifting else 0),sem=5e-5,rhat=1.,ess=1000),
                imaginary_Ha=dict(mean=0.,sem=1e-5),structure_factor=dict(rhat=1.))
            save(measurement/"summary.json",result)
            save(directory/f"stage-{i:02d}.json",dict(output=str(stage),measurement_output=str(measurement),target=target))
        return case,plan,result

    def test_relaxation_does_not_relax_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan,result=self.fixture(Path(tmp))
            self.assertFalse(sampling_passes(result,plan))
            plan["energy_tolerance_Ha_per_electron"]=5e-4
            self.assertTrue(sampling_passes(result,plan))
            result["energy_Ha"]["rhat"]=1.2
            self.assertFalse(sampling_passes(result,plan))

    def test_reused_candidate_runs_challenge_without_retraining_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/"old";original.mkdir();case,_,result=self.fixture(original)
            before=(original/"runs/test/stage-03-120000-measurement/summary.json").read_bytes()
            destination=Path(tmp)/"new";audit=prepare_restart(original,destination,5e-4)
            self.assertTrue(audit[0]["challenge_next"])
            self.assertEqual(audit[0]["previous_step"],120000)
            def execute(command,**kwargs):
                spec=json.loads(Path(command[-1]).read_text())
                self.assertEqual(spec["target"],130000)
                self.assertEqual(spec["lr"],.0001)
                if command[-2]=="train-stage":
                    stage=Path(spec["output"]);stage.mkdir()
                    np.savez(stage/"qmcjax_ckpt_129999.npz",t=129999)
                else:
                    output=Path(spec["measurement_output"]);output.mkdir()
                    save(output/"summary.json",dict(result,step=130000,sampling_pass=True))
            with patch("screening_analysis.converged_window_study.subprocess.run",side_effect=execute) as calls:
                run_case(destination,0)
            self.assertEqual(calls.call_count,2)
            finished=json.loads((destination/"runs/test/finished.json").read_text())
            self.assertEqual(finished["status"],"converged_to_protocol")
            self.assertEqual(before,(original/"runs/test/stage-03-120000-measurement/summary.json").read_bytes())

    def test_partial_checkpoint_is_copied_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/"old";original.mkdir();self.fixture(original,drifting=True)
            stage=original/"runs/test/stage-04-160000";stage.mkdir()
            np.savez(stage/"qmcjax_ckpt_130000.npz",t=130000)
            save(original/"runs/test/stage-04.json",dict(output=str(stage),measurement_output=str(stage)+"-measurement",target=160000))
            destination=Path(tmp)/"new";audit=prepare_restart(original,destination,5e-4)
            self.assertEqual(audit[0]["partial"]["step"],130001)
            copied=destination/"runs/test/stage-04-160000"
            validate_resume_output(copied,160000)
            with self.assertRaises(ValueError):validate_resume_output(copied,170000)
            (copied/"qmcjax_ckpt_130000.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):validate_resume_output(copied,160000)


if __name__=="__main__":unittest.main()
