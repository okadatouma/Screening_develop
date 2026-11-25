"""Plot-environment tests: no JAX/model/checkpoint needed."""
import unittest
import numpy as np
from screening_analysis.physical_phase_report import eigenspace_groups,scale_adjusted_correlations

class ReportStatistics(unittest.TestCase):
    def test_degenerate_projector_independent_of_eigenbasis(self):
        rng=np.random.default_rng(3)
        v=rng.normal(size=(3,2));s=rng.normal(size=3);s/=np.linalg.norm(s)
        U=np.eye(3);theta=.7;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        rotated=U.copy();rotated[:,1:]=U[:,1:]@R
        rows=[]
        for u in (U,rotated):
            overlap=abs(u.T@v)**2/np.sum(v*v,axis=0)
            rows.append(eigenspace_groups(np.array([1.,-.5,-.5]),overlap,abs(s@u)**2))
        self.assertEqual(rows[0][1]['dimension'],2)
        self.assertAlmostEqual(rows[0][1]['spin_projector_overlap'],rows[1][1]['spin_projector_overlap'])
        self.assertAlmostEqual(rows[0][1]['crystal_projector_overlap'],rows[1][1]['crystal_projector_overlap'])

    def test_log_weight_response_factor_is_removed(self):
        w=np.linspace(.01,1,50);response=2*w
        r=scale_adjusted_correlations(w,response,w)
        self.assertAlmostEqual(r['pearson_positive_support'],1)
        self.assertTrue(np.isnan(r['pearson_scale_adjusted']))
        self.assertEqual(r['active_fraction'],1)
        zero=scale_adjusted_correlations(w,response,np.zeros_like(w))
        self.assertTrue(np.isnan(zero['pearson_scale_adjusted']));self.assertEqual(zero['active_edges'],0)

if __name__=='__main__':unittest.main()
