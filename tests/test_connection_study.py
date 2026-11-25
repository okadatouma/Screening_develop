"""Check intervention physics, pooled statistics and the A/B/C integration."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from periodicwave import attention,CustomScreenformer as CS
from screening_analysis.connection_study import analyze_connections,analyze_response,sample_state,scan_window


class ConnectionStudyChecks(unittest.TestCase):
    def setUp(self):
        context=jax.experimental.enable_x64(True);context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def test_radial_intervention_identity_permutation_and_coordinate_derivative(self):
        from tests.test_connection_audit import ConnectionAuditChecks
        params,_=ConnectionAuditChecks().fixture()
        x=jnp.arange(12,dtype=float).reshape(4,3)/7
        options=dict(num_electrons=4,num_heads=2,attn_dim=2,value_dim=2,d_ref=2.,use_softmask=False,
                     use_fixed_value=False,use_tanh_norm=False,use_gate=False,use_output_scale=False)
        def forward(p,x,d):
            return attention.screening_attention(p,x,d,options=options,trim_max=2.,trim_pow=4,
                win_shape='smooth7',freeze_w=False,clip_r=None,clip_beta=.3,return_aux=True)
        positions=jnp.array([0.,.3,.8,1.4]);d=(positions[:,None]-positions[None,:])**2
        changed={**params,'_radial_centers':jnp.array([.5]),'_radial_widths':jnp.array([.7]),'_radial_epsilon':jnp.zeros((2,1))}
        baseline,aux=forward(params,x,d);identity,_=forward(changed,x,d)
        np.testing.assert_array_equal(baseline,identity)
        changed['_radial_epsilon']=jnp.array([[.2],[-.3]])
        value,record=forward(changed,x,d)
        np.testing.assert_array_equal(np.diagonal(record['weights'],axis1=-2,axis2=-1),np.diagonal(aux['weights'],axis1=-2,axis2=-1))
        order=np.array([2,0,3,1]);permuted,_=forward(changed,x[order],d[order][:,order])
        np.testing.assert_allclose(permuted,value[order],rtol=1e-12,atol=1e-12)
        def scalar(t):
            distances=d.at[0,1].set(t).at[1,0].set(t)
            return jnp.sum(forward(changed,x,distances)[0]**2)
        point=.2;eps=1e-5
        derivative=jax.grad(scalar)(point)
        np.testing.assert_allclose(derivative,(scalar(point+eps)-scalar(point-eps))/(2*eps),rtol=1e-6,atol=1e-9)
        self.assertGreater(abs(float(derivative)),1e-10)
        self.assertTrue(np.isfinite(jax.grad(jax.grad(scalar))(.5+.7)))

    def test_connection_mass_effective_distance_and_empty_spin_group(self):
        w=np.zeros((2,1,1,2,2));w[0,0,0,0,1]=w[0,0,0,1,0]=1;w[1,0,0,0,1]=w[1,0,0,1,0]=3
        d=np.array([[[0,1],[1,0]],[[0,2],[2,0]]])
        p=dict(NE=2,NL=1,NH=1,W=w,Alpha=w,Window=w,d=d,lattice=np.eye(2)*10,
               positions=np.array([[0,0,1,0],[0,0,2,0]]),spins=np.ones((2,2)))
        rows,summary=analyze_connections(p,dict(network_max=4,physical_max=5,count=2),rs=12)
        result=next(r for r in summary if r['source']=='screening' and r['distance_kind']=='minimum_image' and r['spin']=='all')
        self.assertAlmostEqual(result['r_eff'],1.75);self.assertEqual(result['r90'],2)
        self.assertAlmostEqual(result['r_eff_aB'],21.)
        empty=next(r for r in rows if r['spin']=='opposite')
        self.assertEqual(empty['count'],0);self.assertTrue(np.isnan(empty['mean']))

    def test_energy_covariance_matches_exact_normalized_state_and_gauge_invariance(self):
        class ToyNet:
            def __init__(self,gauge):self.gauge=gauge
            def apply(self,p,x,s,a,c):
                e=p['layers']['attention'][0].get('_radial_epsilon',jnp.zeros((1,1))).sum()
                return e*(3*x[0]+self.gauge),e*(x[0]+2*self.gauge)
        params={'layers':{'attention':[{'mask_omega':jnp.array([0.])}]}}
        x=np.tile([-1.,1.],4)
        batch=dict(positions=np.column_stack([x,np.zeros(8)]),spins=np.ones((8,1)),atoms=np.zeros((8,1,2)),charges=np.zeros((8,1)))
        def local(p,key,data):return data.positions[0]+0j,None
        with tempfile.TemporaryDirectory() as tmp:
            results=[]
            for gauge in (0.,10.):
                rows,_=analyze_response(ToyNet(gauge),params,batch,np.array([1.]),np.array([1.]),Path(tmp)/f'{gauge}.npz',local,4)
                results.append(rows[0])
            # E(epsilon)=tanh(2 epsilon), exact diagonal two-state Hamiltonian.
            eps=1e-5;exact=(np.tanh(2*eps)-np.tanh(-2*eps))/(2*eps)
            for row in results:
                self.assertAlmostEqual(row['energy_gradient'],exact,places=8)
                self.assertAlmostEqual(row['state_metric'],10.,places=10)
            rows,_=analyze_response(ToyNet(0),params,batch,np.array([1.]),np.array([1.]),Path(tmp)/'diagnostic.npz')
            self.assertTrue(np.isnan(rows[0]['state_metric']))

    def test_production_sampling_and_fixed_width_retraining(self):
        from screening_analysis.train_topology_pilot import build,data_from_key
        with patch.multiple(CS,WIN_SHAPE='smooth7',TRIM_POW=4,FREEZE_W=False,CLIP_R=None):
            net,_,_,_,_,cfg=build(2);params=net.init(jax.random.PRNGKey(4))
            data={k:np.asarray(v) for k,v in dict(data_from_key(jax.random.PRNGKey(8),2)).items()}
            from periodicwave.pbc import hamiltonian
            local=hamiltonian.local_energy(net.apply,jnp.zeros(1),(4,0),complex_output=True,lattice=jnp.eye(2)*4,
                convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':'default'})
            args=argparse.Namespace(chains=2,steps=1,spin_flips=1,proposal=.2,burn=1,blocks=4,block_size=1,
                retrain_steps=1,learning_rate=1e-4,factors=[1.],mode='retrain',seed=4)
            info=dict(config=cfg,lattice=np.eye(2)*4,step=0)
            fresh,metadata=sample_state(net,params,data,info,args,4,local)
            self.assertEqual(fresh['positions'].shape,(8,8));self.assertEqual(metadata['chains'],2)
            with tempfile.TemporaryDirectory() as tmp:
                rows=scan_window(net,params,data,info,args,local,Path(tmp))
                self.assertEqual(len(rows),4)
                self.assertTrue(all(np.isfinite(r['energy']) for r in rows))
                self.assertTrue((Path(tmp)/'retrained-l0-h0-f1.npz').exists())


if __name__=='__main__':unittest.main()
