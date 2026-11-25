import tempfile
from pathlib import Path
import unittest
import numpy as np
from screening_analysis.constrained_null import distance_bins,spin_degrees,rewire,available_switches,spin_distance_shuffle
from screening_analysis.ph_research import importance_graph,align_gradients,graph_features,top_graph,analyze


class ResearchPHChecks(unittest.TestCase):
    def test_switch_invariants_and_nontrivial_randomization(self):
        rng=np.random.default_rng(17);n=12;s=np.r_[np.ones(6),-np.ones(6)]
        w=np.zeros((n,n))
        for i in range(6):
            for j in (i,(i+1)%6,(i+2)%6):w[i,j+6]=w[j+6,i]=rng.uniform(.1,1)
        d=np.ones((n,n));np.fill_diagonal(d,0)
        randomized,record=rewire(w,s,d,rng,bins=1,switches=80)
        self.assertTrue(record['target_reached']);self.assertTrue(record['movable'])
        np.testing.assert_array_equal(spin_degrees(w,s),spin_degrees(randomized,s))
        np.testing.assert_array_equal(np.sort(w[np.triu_indices(n,1)]),np.sort(randomized[np.triu_indices(n,1)]))
        self.assertFalse(np.any((randomized>0)&(s[:,None]==s[None,:])))
        # No triangles: H1 endpoint fixed by E-V+C, so loop-count differences
        # cannot establish exceptional structure at fixed connectivity.
        for graph in (w,randomized):
            features,_=graph_features(graph)
            self.assertEqual(features['beta1_at_full_support'],18-12+features['components_at_full_support'])

    def test_distance_bin_weight_multisets_and_support(self):
        rng=np.random.default_rng(12);n=10;s=rng.choice([-1,1],n)
        d=rng.uniform(size=(n,n));d=(d+d.T)/2;np.fill_diagonal(d,0)
        w=rng.uniform(.1,1,(n,n));w=(w+w.T)/2;w[d>.55]=0;np.fill_diagonal(w,0)
        allowed=d<.6
        random,record=rewire(w,s,d,rng,bins=3,switches=20,max_attempts=2000,admissible=allowed)
        labels,_=distance_bins(d,3);ij=np.triu_indices(n,1)
        for label in np.unique(labels):
            select=labels[ij]==label
            np.testing.assert_array_equal(np.sort(w[ij][select]),np.sort(random[ij][select]))
        np.testing.assert_array_equal(spin_degrees(w,s),spin_degrees(random,s))
        self.assertFalse(np.any((random>0)&~allowed))

    def test_frozen_complete_bipartite_is_reported(self):
        s=np.array([1,1,-1,-1]);w=(s[:,None]!=s[None,:]).astype(float)
        result,info=rewire(w,s,np.ones((4,4)),np.random.default_rng(2),switches=5,max_attempts=100)
        self.assertFalse(info['movable']);self.assertEqual(info['accepted'],0)
        np.testing.assert_array_equal(result,w)
        self.assertEqual(available_switches(w,s),0)

    def test_relaxed_control_preserves_spin_and_distance_weight_multisets(self):
        rng=np.random.default_rng(7);s=np.r_[np.ones(5),-np.ones(5)];n=len(s)
        d=rng.uniform(size=(n,n));d=(d+d.T)/2
        w=np.zeros((n,n))
        for i in range(5):w[i,5+i]=w[5+i,i]=.1+i/10
        random,info=spin_distance_shuffle(w,s,d,rng,bins=2)
        labels,_=distance_bins(d,2);ij=np.triu_indices(n,1)
        for band in np.unique(labels):
            mask=(labels[ij]==band)
            np.testing.assert_array_equal(np.sort(w[ij][mask]),np.sort(random[ij][mask]))
        self.assertFalse(np.any((random>0)&(s[:,None]==s[None,:])))

    def test_importance_complex_derivative_and_zero(self):
        graph,scale=importance_graph([[3,0,0],[4,0,0]],3,'complex')
        self.assertEqual(scale,5);self.assertEqual(graph[0,1],1)
        zero,scale=importance_graph(np.zeros((2,3)),3,'phase')
        self.assertEqual(scale,0);self.assertFalse(zero.any())

    def test_exact_input_alignment_and_mismatch(self):
        p=dict(positions=np.arange(12).reshape(3,4),spins=np.ones((3,2)),atoms=np.zeros((3,1,2)),charges=np.zeros((3,1)))
        inputs={k:v[[2,0]] for k,v in p.items()}
        pairs=align_gradients(p,inputs,np.ones((2,2,1,1,1)),[True,True])
        self.assertEqual(pairs,[(2,0),(0,1)])
        inputs['spins'][0,0]=-1
        with self.assertRaises(ValueError):align_gradients(p,inputs,np.ones((2,2,1,1,1)),[True,True])

    def test_small_pipeline(self):
        n=4;rng=np.random.default_rng(6);w=rng.uniform(.1,1,(1,1,1,n,n));w=(w+w.swapaxes(-1,-2))/2
        p=dict(S=2*w-1,Alpha=w,W_window=w,Window=w,d=np.ones((1,n,n)),dref=np.array(2.),
               positions=rng.uniform(size=(1,2*n)),spins=np.array([[1.,1.,-1.,-1.]]),atoms=np.zeros((1,1,2)),charges=np.zeros((1,1)),
               lattice=np.eye(2)*3,NE=n,NL=1,NH=1,metadata=dict(checkpoint_sha256='test',evaluation_input_sha256='test'))
        with tempfile.TemporaryDirectory() as temp:
            result=analyze(p,Path(temp)/'out',samples=1,repeats=1,ks=(3,),energy_samples=0)
            self.assertEqual(result['samples'],1)
            for name in ['diagrams.npz','constrained-null.csv','configuration-physics.csv','within-run-correlations.csv']:
                self.assertTrue((Path(temp)/'out'/name).exists())


if __name__=='__main__':unittest.main()
