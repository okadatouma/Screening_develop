import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from screening_analysis.phase_evidence import evaluate_grid,run
from screening_analysis.berry_validation import qwz_manybody_grid


class PhaseEvidenceChecks(unittest.TestCase):
    def test_known_manybody_chern_refinement_and_no_false_phase_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[]
            for grid in (6,10):
                states,gap=qwz_manybody_grid(grid,-1.)
                rng=np.random.default_rng(grid)
                states*=np.exp(1j*rng.uniform(-np.pi,np.pi,(grid,grid)))[...,None]
                name=f'grid{grid}.npz'
                np.savez(root/name,states=states,theta_x=2*np.pi*np.arange(grid)/grid,
                    theta_y=2*np.pi*np.arange(grid)/grid,gap_lower_bounds=np.full((grid,grid),gap))
                entries.append(dict(path=name))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(physical_system='QWZ benchmark, NOT lambda-jellium',
                basis_id='fixed-site Fock basis',grids=entries,state_origin='exact diagonalization')))
            result=run(manifest,root/'out')
            self.assertTrue(result['chern_grid_stable']);self.assertTrue(result['positive_supplied_gap_bounds'])
            self.assertFalse(result['phase_verified']);self.assertEqual(result['physical_phase'],'unknown')
            for grid in result['grids']:self.assertAlmostEqual(grid['chern'],-1.,places=10)

    def test_explicit_twist_grid_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wrong.npz';np.savez(path,states=np.ones((3,3,2)),theta_x=[0,1,2],theta_y=[0,1,2])
            with self.assertRaisesRegex(ValueError,'uniform twist grid'):evaluate_grid(path)

    def test_overlap_interface_matches_explicit_state_vectors(self):
        from screening_analysis.berry_validation import berry_links
        states,gap=qwz_manybody_grid(6,-1.)
        x=np.sum(states.conj()*np.roll(states,-1,axis=0),axis=-1)
        y=np.sum(states.conj()*np.roll(states,-1,axis=1),axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'links.npz'
            np.savez(path,overlap_x=x,overlap_y=y,overlap_includes_boundary_sewing=np.array(True),
                overlap_error_x=np.full(x.shape,.01),theta_x=2*np.pi*np.arange(6)/6,theta_y=2*np.pi*np.arange(6)/6)
            result=evaluate_grid(path)
            np.testing.assert_allclose(result['curvature'],berry_links(states)['curvature'],atol=1e-14)
            self.assertGreater(result['minimum_supplied_link_snr_x'],1.)


if __name__=='__main__':unittest.main()
