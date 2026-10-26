import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from screening_analysis.topology_run import analyze_probe,phase_classification


class RunnerChecks(unittest.TestCase):
    def test_sources_outputs_and_protection(self):
        # Directed weights, exact zeros and ties, and a winding triangle.
        W=np.array([[1.,.8,.4],[.6,1.,.8],[.4,.8,1.]])[None,None,None]
        probe=dict(S=2*W-1,Alpha=W,W_window=W*.5,Window=np.ones_like(W)*.5,
            positions=np.array([[[.05,.1],[.38,.1],[.71,.1]]]),lattice=np.eye(2),
            d=np.array([[[0,.33,.34],[.33,0,.33],[.34,.33,0]]]),dref=1.,metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'new'
            summary=analyze_probe(probe,target,{'id':'test'},null_repeats=1)
            self.assertEqual(len(summary['aggregates']),12)
            self.assertEqual(summary['shape'],[1,1,1,3])
            with np.load(target/'curves.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['screening_raw_density'],z['screening_shuffle_density'])
            with np.load(target/'diagrams.npz',allow_pickle=False) as z:
                self.assertTrue(any(k.endswith('H0') for k in z.files))
            self.assertTrue((target/'curves.svg').read_text().startswith('<svg'))
            self.assertIn('critical_cutoff1',json.loads((target/'geometry.json').read_text())[0])
            with self.assertRaises(FileExistsError):analyze_probe(probe,target,{'id':'test'})
            self.assertEqual(phase_classification([summary])['status'],'unavailable')

    def test_grouped_phase_evaluation(self):
        records=[]
        keys=('H0_total_persistence','H1_total_persistence','H0_essential_intervals',
          'H1_essential_intervals','beta0_auc','beta1_auc','rank1_probability','rank2_probability')
        for seed in range(3):
            for phase in ('WC','AHC'):
                for checkpoint in range(2):
                    aggregates=[dict(source=source,control=control,**{k:float(phase=='AHC') for k in keys})
                        for source in ('window','trim','screening','qk') for control in ('raw','rank','shuffle')]
                    records.append(dict(run=dict(group=f'seed{seed}',phase=phase,phase_evidence='test fixture only'),aggregates=aggregates))
        result=phase_classification(records)
        self.assertEqual(result['status'],'exploratory')
        for row in result['results']:
            self.assertEqual(row['groups'],3)
            self.assertEqual(row['confusion_wc_ahc'],[[3,0],[0,3]])


if __name__=='__main__':unittest.main()
