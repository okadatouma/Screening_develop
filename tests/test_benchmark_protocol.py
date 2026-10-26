"""Read-only reconstruction and natural/shared-input separation contracts."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from tests.check_attn_probe import fixture
from screening_analysis.attn_probe import probe
from screening_analysis.checkpoint_catalog import inspect,discover
from screening_analysis.checkpoint_model import load_model
from screening_analysis.benchmark_features import features


class ProtocolChecks(unittest.TestCase):
    def test_shared_probe_identical_inputs_and_readonly_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            net,p,data,settings,L,ck=fixture(tmp)
            digest=hashlib.sha256(ck.read_bytes()).hexdigest()
            external={k:np.asarray(getattr(data,k)).copy() for k in ('positions','spins','atoms','charges')}
            external['positions']+=np.array([.07,.02,-.01,.03])
            a=probe(tmp,settings=settings,lattice=L,evaluation_data=external)
            b=probe(tmp,settings=settings,lattice=L,evaluation_data=external)
            self.assertEqual(a['metadata']['input_mode'],'shared_external_configurations')
            self.assertEqual(a['metadata']['evaluation_input_sha256'],b['metadata']['evaluation_input_sha256'])
            np.testing.assert_array_equal(a['W_window'],b['W_window'])
            self.assertEqual(hashlib.sha256(ck.read_bytes()).hexdigest(),digest)
            bad=dict(external,positions=np.zeros((2,8)))
            with self.assertRaisesRegex(ValueError,'dimensions'):probe(tmp,settings=settings,lattice=L,evaluation_data=bad)

    def test_physical_model_reconstruction_uses_same_wavefunction(self):
        with tempfile.TemporaryDirectory() as tmp:
            net,p,data,settings,L,ck=fixture(tmp)
            file=Path(tmp)/'config.json';cfg=json.loads(file.read_text())
            cfg['system']['pbc_lattice']=L.tolist();cfg['network']['make_feature_layer_kwargs']['lattice']=L.tolist()
            cfg['network'].update(determinants=2,complex=True,bias_orbitals=False,jastrow='NONE',jastrow_kwargs={'ndim':2},
                                 make_envelope_fn='periodicwave.envelopes.make_null_envelope',make_envelope_kwargs={})
            cfg['screening_attention_settings']=settings;file.write_text(json.dumps(cfg))
            with load_model(tmp) as (restored,params,samples,info):
                args=[jnp.asarray(samples[k][0]) for k in ('positions','spins','atoms','charges')]
                expected=net.apply(p,*args);actual=restored.apply(params,*args)
                np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_catalog_never_converts_candidate_parameters_to_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,_,_,ck=fixture(tmp)
            file=Path(tmp)/'config.json';cfg=json.loads(file.read_text())
            cfg['system']['make_local_energy_kwargs']=dict(lam=.33,potential_kwargs={'interaction_energy_scale':15})
            file.write_text(json.dumps(cfg));r=inspect(ck)
            self.assertEqual(r['candidate'],'WC_candidate');self.assertEqual(r['phase'],'unknown');self.assertFalse(r['phase_verified'])
            copy=Path(tmp)/'copy';copy.mkdir();shutil.copyfile(ck,copy/ck.name)
            inventory=discover(tmp);self.assertEqual(inventory['total'],2)
            self.assertEqual(len(inventory['duplicate_checkpoint_groups']),1)

    def test_geometric_baselines_identical_under_shared_inputs(self):
        rng=np.random.default_rng(50);B,L,H,N=8,1,2,4
        w=rng.uniform(.1,.8,(B,L,H,N,N));p=dict(positions=rng.uniform(0,4,(B,2*N)),NE=N,
            lattice=np.eye(2)*4,W_window=w,w_sigma=np.ones((L,H)),w_centre=np.zeros((L,H)),w_outer=np.ones((L,H)),r=np.ones((L,H)))
        q=dict(p,W_window=w*.7,w_sigma=p['w_sigma']*1.2)
        a=features(p)['features'];b=features(q)['features']
        for family in ('density','gr','structure_factor','distances'):np.testing.assert_array_equal(a[family],b[family])
        self.assertGreater(np.linalg.norm(np.array(a['weight_distribution'])-np.array(b['weight_distribution'])),0.)


if __name__=='__main__':unittest.main()
