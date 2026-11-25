import unittest
import tempfile
from pathlib import Path
import numpy as np
from screening_analysis.density_ph import analyze, bounded_weights, density_curve, distance_shuffle


class DensityPHChecks(unittest.TestCase):
    def test_saved_probe_pipeline(self):
        rng=np.random.default_rng(3)
        b=rng.uniform(.1,.9,(4,4)); b=(b+b.T)/2; np.fill_diagonal(b,0)
        window=b.copy();window[0,1]=window[1,0]=0
        dist=rng.uniform(.1,2,(4,4));dist=(dist+dist.T)/2;np.fill_diagonal(dist,0)
        expand=lambda a:a[None,None,None,:,:]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'probe.npz'
            np.savez(path,S=expand(2*b-1),Alpha=expand(b),Window=expand(window),
                     W_window=expand(b*window),d=dist[None],dref=np.array(1.),metadata_json=np.array('{}'))
            summary=analyze(path,Path(temp)/'out',repeats=2)
            self.assertEqual(summary['common_max_k'],5)
            self.assertEqual(summary['possible_edges'],6)
            self.assertTrue((Path(temp)/'out/curves.csv').exists())
            self.assertTrue((Path(temp)/'out/diagrams-edge-count.npz').exists())

    def test_roundoff_clipping_is_explicit_and_bounded(self):
        np.testing.assert_array_equal(bounded_weights([[-1e-15,1+1e-15]]),[[0,1]])
        with self.assertRaises(ValueError):
            bounded_weights([[-.001,1.]])

    def test_square_then_diagonal(self):
        b = np.zeros((4,4))
        for weight,(i,j) in zip([.9,.8,.7,.6,.2],[(0,1),(1,2),(2,3),(0,3),(0,2)]):
            b[i,j]=b[j,i]=weight
        curve, ties, diagrams = density_curve(b)
        np.testing.assert_array_equal(curve[1], [0,0,0,0,1,0])
        np.testing.assert_allclose(diagrams[1], [[4,5]])
        self.assertFalse(ties.any())

    def test_zero_edges_never_added_and_ties_flagged(self):
        b = np.array([[0,.8,.8],[.8,0,0],[.8,0,0]])
        curve, split, _ = density_curve(b)
        self.assertEqual(curve.shape, (2,3))
        np.testing.assert_array_equal(split,[False,True,False])
        with self.assertRaises(ValueError):
            density_curve(b,3)

    def test_monotone_weights_and_vertex_permutation(self):
        rng = np.random.default_rng(7)
        b = rng.uniform(.01,1,(7,7)); b=(b+b.T)/2; np.fill_diagonal(b,0)
        original=density_curve(b,12)[0]
        np.testing.assert_array_equal(original,density_curve(b**3,12)[0])
        perm=rng.permutation(7)
        np.testing.assert_array_equal(original,density_curve(b[perm][:,perm],12)[0])

    def test_every_k_against_boundary_matrix_oracle(self):
        from tests.test_topology import oracle_betti
        rng=np.random.default_rng(21)
        for _ in range(8):
            b=rng.choice([0.,.2,.7],(5,5)); b=(b+b.T)/2; np.fill_diagonal(b,0)
            ij=np.triu_indices(5,1); priority=rng.permutation(len(ij[0]))
            values=b[ij]; positive=np.flatnonzero(values>0)
            order=positive[np.lexsort((priority[positive],-values[positive]))]
            curve,_,_=density_curve(b,priorities=priority)
            for k in range(len(order)+1):
                graph=np.zeros_like(b); edge=order[:k]
                graph[ij[0][edge],ij[1][edge]]=1; graph+=graph.T
                np.testing.assert_array_equal(curve[:,k],oracle_betti(graph,.5))

    def test_distance_null_preserves_bin_multisets(self):
        rng=np.random.default_rng(12)
        b=rng.choice([0.,.2,.6],(8,8)); b=(b+b.T)/2; np.fill_diagonal(b,0)
        d=rng.uniform(0,3,(8,8)); d=(d+d.T)/2
        shuffled,cuts=distance_shuffle(b,d,rng,bins=4)
        ij=np.triu_indices(8,1); labels=np.searchsorted(cuts[1:-1],d[ij],side='right')
        for label in np.unique(labels):
            np.testing.assert_array_equal(np.sort(b[ij][labels==label]),np.sort(shuffled[ij][labels==label]))


if __name__=='__main__':
    unittest.main()
