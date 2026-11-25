"""Numerical convergence criteria must reject noisy and trapped plateaus."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from screening_analysis.converged_window_study import time_statistics,equivalent,check_convergence,dump
from screening_analysis.converged_window_report import minimum_lengths,pair_curves,mass_radius


class ConvergenceTests(unittest.TestCase):
    def test_equivalence_is_not_failure_to_detect_a_difference(self):
        a=dict(mean=-1.,sem=.1);b=dict(mean=-1.,sem=.1)
        self.assertFalse(equivalent(a,b,.001)[0])
        a["sem"]=b["sem"]=.00001
        self.assertTrue(equivalent(a,b,.001)[0])
        b["mean"]=-1.01
        self.assertFalse(equivalent(a,b,.001)[0])

    def test_sampling_and_challenge_are_required(self):
        p=dict(energy_tolerance_Ha_per_electron=5.e-5)
        history=[dict(sampling_pass=True,energy_Ha=dict(mean=-1.,sem=1.e-6)) for _ in range(3)]
        self.assertFalse(check_convergence(history,p)["converged"])
        self.assertTrue(check_convergence(history,p,challenge=True)["converged"])
        history[-1]["sampling_pass"]=False
        self.assertFalse(check_convergence(history,p,challenge=True)["converged"])

    def test_chain_offsets_are_not_hidden_in_small_time_error(self):
        rng=np.random.default_rng(12)
        good=rng.normal(size=(2048,8))
        bad=good+np.arange(8)[None,:]
        a,b=time_statistics(good),time_statistics(bad)
        self.assertLess(a["rhat"],1.02)
        self.assertGreater(b["rhat"],1.5)
        self.assertGreater(b["sem"],a["sem"]*10)
        self.assertAlmostEqual(b["mean"],bad.mean())

    def test_serialization_handles_concrete_device_arrays(self):
        import jax.numpy as jnp
        import json
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"a.json"
            dump(path,dict(matrix=jnp.eye(2),bad=float("inf")))
            result=json.loads(path.read_text())
            self.assertEqual(result["matrix"],[[1.,0.],[0.,1.]])
            self.assertIsNone(result["bad"])


class PhysicalLengthTests(unittest.TestCase):
    def test_minimum_image_invariance_and_brute_force(self):
        L=np.array([[3.,-1.5],[0.,2.598076211353316]])
        rng=np.random.default_rng(3);delta=rng.uniform(-3,3,(100,2))
        observed=minimum_lengths(delta,L)
        shifts=np.array([(i,j) for i in range(-5,6) for j in range(-5,6)])@L.T
        expected=np.linalg.norm(delta[:,None]-shifts,axis=-1).min(-1)
        np.testing.assert_allclose(observed,expected,atol=1.e-12)
        np.testing.assert_allclose(observed,minimum_lengths(delta+L[:,0]-2*L[:,1],L),atol=1.e-12)

    def test_uniform_pair_normalization_and_density_background(self):
        L=np.array([[6.,-3.],[0.,3*np.sqrt(3)]])
        rng=np.random.default_rng(14);pos=rng.uniform(size=(4096,9,2))@L.T
        gr,bg,_=pair_curves(pos,L,np.linspace(0,2.,17))
        self.assertLess(abs(np.mean(gr)-1),.03)
        self.assertLess(np.max(abs(bg-1)),.04)
        self.assertIsNone(mass_radius(np.arange(3),np.zeros(3)))
        self.assertEqual(mass_radius(np.arange(3),np.array([0,1,0])),1.)


if __name__=="__main__":unittest.main()
