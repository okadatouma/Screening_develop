"""CLI/package contracts; synthetic fixtures never carry physical phase labels."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from screening_analysis import topology
from screening_analysis.topology import persistence as public_persistence
from screening_analysis.topology.cli import main
from screening_analysis.topology.workflow import manifest,simple_id


class CliChecks(unittest.TestCase):
    def test_legacy_public_api(self):
        from screening_analysis.topology.persistence import persistence
        self.assertIs(public_persistence,persistence)
        self.assertTrue(callable(topology.summary))

    def test_commands_and_default_manifest(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:main(['--help'])
        self.assertEqual(exc.exception.code,0)
        path=Path(__file__).resolve().parents[1]/'screening_analysis/topology/experiments.yaml'
        data,base=manifest(path)
        self.assertEqual(len(data['runs']),12)
        self.assertTrue(all((base/r['probe']).is_file() for r in data['runs']))
        self.assertTrue(all(r['phase']=='unknown' and not r['phase_verified'] for r in data['runs']))

    def test_training_dry_run_and_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'job.yaml';config.write_text(json.dumps(dict(defaults=dict(n=4,rs=15,lam=.33,iterations=2,platform='cpu'),jobs=[dict(id='one',seed=71)])))
            main(['train','--manifest',str(config),'--output',str(root/'plan'),'--dry-run'])
            data=json.loads((root/'plan/dry-run.json').read_text());self.assertFalse(data['executed'])
            self.assertEqual(data['jobs'][0]['environment']['PW_SEED'],'71')
            self.assertFalse((root/'plan/one/run').exists())
            with self.assertRaises(FileExistsError):main(['train','--manifest',str(config),'--output',str(root/'plan'),'--dry-run'])

    def test_unverified_phase_rejected_before_output(self):
        from screening_analysis.topology.workflow import benchmark
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'manifest.json'
            p.write_text(json.dumps(dict(runs=[dict(id='wc',phase='WC',phase_verified=False)])))
            with self.assertRaises(ValueError):benchmark(p,root/'output')
            self.assertFalse((root/'output').exists())

    def test_invalid_sample_count_rejected(self):
        from screening_analysis.topology.workflow import benchmark
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'manifest.json';p.write_text(json.dumps(dict(nconf=0,runs=[])))
            with self.assertRaises(ValueError):benchmark(p,root/'output')
            self.assertFalse((root/'output').exists())

    def test_invalid_ids(self):
        for value in ['../bad','.','..','']:
            with self.assertRaises(ValueError):simple_id(value)


if __name__=='__main__':unittest.main()
