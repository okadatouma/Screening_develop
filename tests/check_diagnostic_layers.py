"""D'試作と保存した実測記録の検証。Productionの比較許容誤差は変更しない。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tests.support import require_numerics, ROOT, make_case
require_numerics()
import numpy as np
import jax
import jax.numpy as jnp
from periodicwave import CustomScreenformer as CS, checkpoint
from tests.diagnostic_layers import diagnostic_namespace, make_diagnostic


class DiagnosticChecks(unittest.TestCase):
    def test_source_and_configuration_are_isolated(self):
        source=Path(CS.__file__).read_bytes()
        factory=CS.make_self_attention_block
        original=CS.WIN_SHAPE
        ns=diagnostic_namespace()
        archived=(Path(__file__).parent/'layers_validation/before-source.txt').read_bytes()
        self.assertEqual(ns['source_sha256'],hashlib.sha256(archived).hexdigest())
        recorded=(Path(__file__).parent/'layers_validation/candidate.sha256').read_text().strip()
        self.assertEqual(ns['candidate_sha256'],recorded)
        ns['WIN_SHAPE']='diagnostic-space-only'
        self.assertEqual(CS.WIN_SHAPE,original)
        self.assertIs(CS.make_self_attention_block,factory)
        self.assertEqual(Path(CS.__file__).read_bytes(),source)

    def test_saved_experiment_has_complete_coverage(self):
        for name,count in (('matrix-v1',30),('matrix-wide-v1',6)):
            path=Path(__file__).parent/f'diagnostic_validation/{name}/summary.json'
            if not path.exists():self.skipTest('D-prime計測が未完了: '+name)
            result=json.loads(path.read_text())
            self.assertTrue(result['production_source_unchanged'])
            self.assertEqual(len(result['cases']),count)
            for case in result['cases']:
                self.assertTrue(case['production_unchanged'])
                self.assertTrue(case['discarded_aux_exact'])
                self.assertTrue(case['eager_relations_exact'])
                for metric in case['metrics'].values():
                    self.assertTrue(np.isfinite([metric['max_abs'],metric['rms'],metric['reference_rms']]).all())

    def test_checkpoint_and_discarded_aux(self):
        root=ROOT/'tests/layernorm_validation/before-checkpoints'
        try:
            for record in json.loads((root/'evaluations.json').read_text()):
                if record['route']!='screen':continue
                jax.config.update('jax_enable_x64',record['x64'])
                path=ROOT/record['path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])
                with tempfile.TemporaryDirectory() as directory:
                    net,_,_,_,_,_=make_case('screen',directory)
                _,data,params,_,_=checkpoint.restore(str(path),batch_size=2)
                params=jax.tree_util.tree_map(lambda x:jnp.asarray(x[0]),params)
                data=jax.tree_util.tree_map(lambda x:jnp.asarray(x[0,0]),data)
                diagnostic=make_diagnostic(net,(1,1),data,diagnostic_namespace())
                production=jax.jit(lambda p,r:jnp.stack(net.apply(p,r,data.spins,data.atoms,data.charges)))
                args=(params,data.positions)
                before=production(*args)
                out,_=jax.jit(diagnostic)(*args)
                print('checkpoint diagnostic max_abs',record['x64'],float(jnp.max(jnp.abs(out-before))))
                np.testing.assert_array_equal(before,production(*args))
                np.testing.assert_array_equal(before,jax.jit(lambda p,r:diagnostic(p,r)[0])(*args))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])
        finally:jax.config.update('jax_enable_x64',True)

    def test_saved_values_and_layer_connections(self):
        path=Path(__file__).parent/'diagnostic_validation/error-summary.json'
        if not path.exists():self.skipTest('D-prime計測が未完了')
        for case in json.loads(path.read_text())['cases']:
            directory=path.parent/case['directory']
            archive=directory/(case['case']+'.npz')
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),case['archive_sha256'])
            with np.load(archive) as arrays:
                for i in range(case['layers']-1):
                    np.testing.assert_array_equal(arrays[f"aux_layer{i}_['output']"],
                                                  arrays[f"aux_layer{i+1}_['input']"])
                for name,measurement in case['metrics'].items():
                    ref=arrays[name+'_production_reference'].astype(np.float64)
                    diag=arrays[name+'_diagnostic'].astype(np.float64)
                    self.assertEqual(float(np.max(np.abs(diag-ref))),measurement['max_abs'])


if __name__=='__main__':unittest.main()
