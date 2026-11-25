import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from screening_analysis.rs_sweep import ROOT,cases_from_config,prepare,read_plan,shared_bank,prepare_reference
from screening_analysis.rs_observables import row_entropy,attention_modes,phase_sensitivity


class RSSweepChecks(unittest.TestCase):
    def config(self):return json.loads((ROOT/'rs_sweep.example.json').read_text())

    def test_default_plan_has_distinct_conditions_and_no_phase_certificates(self):
        cases=cases_from_config(self.config())
        self.assertEqual(len(cases),54)
        self.assertEqual(len({c['id'] for c in cases}),54)
        self.assertTrue(all(c['phase']=='unknown' and not c['phase_verified'] for c in cases))

    def test_invalid_scan_and_owned_environment(self):
        for key,value in [('rs',[0,15]),('rs',[15,15]),('seeds',[101,101]),('iterations',0),('n',5)]:
            c=self.config();c[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):cases_from_config(c)
        c=self.config();c['training_environment']['PW_LJ_HF']='wrong.txt'
        with self.assertRaises(ValueError):cases_from_config(c)

    def test_common_bank_and_hash_protection(self):
        bank=shared_bank(4,5)
        np.testing.assert_array_equal(bank['positions'],shared_bank(4,5)['positions'])
        self.assertEqual(bank['positions'].shape,(5,8))
        np.testing.assert_array_equal(bank['spins'][::2],1.)
        with tempfile.TemporaryDirectory() as tmp:
            c=self.config();c.update(n=4,rs=[2],seeds=[1],branches=[dict(label='plane',lam=0.,initialization='plane')])
            config=Path(tmp)/'config.json';config.write_text(json.dumps(c))
            output=Path(tmp)/'prepared';plan=prepare(config,output)
            self.assertFalse((output/plan['cases'][0]['id']).exists())
            read_plan(output/'plan.json',0)
            with (output/'common-inputs.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):read_plan(output/'plan.json',0)

    def test_plane_reference_matches_periodic_supercell(self):
        case=dict(n=4,lam=0.,rs=2.,initialization='plane')
        with tempfile.TemporaryDirectory() as tmp:
            path=prepare_reference(case,dict(gcut=1.01),Path(tmp)/'ref')
            rows=np.loadtxt(path);L=shared_bank(4,1)['lattice']
            np.testing.assert_allclose(rows[:,:2]@L/(2*np.pi),np.rint(rows[:,:2]@L/(2*np.pi)),atol=1e-12)
            self.assertEqual(len(np.unique(rows[:,:2],axis=0)),4)
            self.assertFalse(json.loads((path.parent/'metadata.json').read_text())['converged'])

    def test_incompatible_hf_seed_fails_before_preparing_jobs(self):
        c=self.config();c.update(n=4,rs=[15],seeds=[101],branches=[c['branches'][2]])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'HF seed table does not match'):
                prepare(p,Path(tmp)/'out')
            self.assertFalse((Path(tmp)/'out').exists())

    def test_entropy_uniform_zero_and_scale_invariance(self):
        weights=np.ones((4,4));result=row_entropy(weights)
        np.testing.assert_allclose(result['entropy'],np.log(3))
        np.testing.assert_allclose(result['normalized'],1.)
        np.testing.assert_allclose(result['effective_neighbors'],3.)
        np.testing.assert_allclose(row_entropy(123*weights)['entropy'],result['entropy'])
        empty=row_entropy(np.zeros((4,4)))
        self.assertTrue(np.isnan(empty['entropy']).all());self.assertFalse(empty['active'].any())

    def test_modes_independent_of_particle_labels_and_no_spin_mutation(self):
        rng=np.random.default_rng(41);w=rng.uniform(size=(5,5));spins=np.array([1.,1.,1.,-1.,-1.]);original=spins.copy()
        result=attention_modes(w,spins)
        np.testing.assert_array_equal(spins,original)
        p=rng.permutation(5);other=attention_modes(w[p][:,p],spins[p])
        np.testing.assert_allclose(result['eigenvalues'],other['eigenvalues'],atol=1e-12)
        np.testing.assert_allclose(result['normalized_eigenvalues'],other['normalized_eigenvalues'],atol=1e-12)
        np.testing.assert_allclose(result['participation'],other['participation'],atol=1e-12)
        np.testing.assert_allclose(result['spin_overlap'],other['spin_overlap'],atol=1e-12)
        np.testing.assert_allclose(result['normalized_eigenvalues'][0],1.,atol=1e-12)

    def test_multi_layer_phase_sensitivity_known_answer(self):
        import jax
        import jax.numpy as jnp
        class KnownNetwork:
            def apply(self,p,pos,spins,atoms,charges):
                phase=jnp.asarray(0.);amplitude=jnp.asarray(0.)
                for layer,ap in enumerate(p['layers']['attention']):
                    scale=ap.get('_edge_scale',jnp.ones((2,4,4)))
                    amplitude=amplitude+scale[0,0,1]
                    phase=phase+(layer+1)*(scale[1,0,1]+scale[1,1,0])
                return jnp.angle(jnp.exp(1j*phase)),amplitude
        with jax.experimental.enable_x64(True),tempfile.TemporaryDirectory() as tmp:
            batch={k:v for k,v in shared_bank(4,1).items() if k!='lattice'}
            result=phase_sensitivity(KnownNetwork(),{'layers':{'attention':[{},{}]}},batch,Path(tmp),2,2)
            self.assertEqual(result['valid_samples'],1);self.assertTrue(result['finite_difference_pass'])
            with np.load(Path(tmp)/'connection-gradients.npz') as z:
                self.assertEqual(z['gradients'].shape,(1,2,2,2,6))
                np.testing.assert_allclose(z['gradients'][0,1,:,1,0],[2,4],atol=1e-12)


if __name__=='__main__':unittest.main()
