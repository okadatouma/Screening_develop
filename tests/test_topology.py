"""Independent small-complex, assignment and torus-embedding checks."""
import itertools
import unittest
import numpy as np
from screening_analysis import topology as ph
from screening_analysis.graph_wind import minimum_image_edges, weighted_winding


def gf2rank(a):
    a=np.asarray(a,dtype=int).copy(); rank=0
    for col in range(a.shape[1]):
        rows=np.flatnonzero(a[rank:,col])
        if not len(rows): continue
        pivot=rank+rows[0]; a[[rank,pivot]]=a[[pivot,rank]]
        for r in range(rank+1,len(a)):
            if a[r,col]: a[r]^=a[rank]
        rank+=1
        if rank==len(a): break
    return rank


def oracle_betti(b,t):
    n=len(b); edges=[e for e in itertools.combinations(range(n),2) if b[e]>t]
    triangles=[s for s in itertools.combinations(range(n),3) if all(e in edges for e in itertools.combinations(s,2))]
    d1=np.zeros((n,len(edges)),int); d2=np.zeros((len(edges),len(triangles)),int)
    for j,e in enumerate(edges): d1[list(e),j]=1
    for j,s in enumerate(triangles):
        for e in itertools.combinations(s,2):d2[edges.index(e),j]=1
    r1,r2=gf2rank(d1),gf2rank(d2)
    return n-r1,len(edges)-r1-r2


class TopologyChecks(unittest.TestCase):
    def test_all_four_vertex_flag_complexes(self):
        ij=np.triu_indices(4,1)
        for code in range(64):
            b=np.zeros((4,4)); b[ij]=[(code>>k)&1 for k in range(6)]; b+=b.T
            result=ph.analyze_graph(b,[0,.5,1])
            for k,t in enumerate((0,.5,1)):
                np.testing.assert_array_equal(result['betti'][:,k],oracle_betti(b,t))

    def test_weighted_ties_and_zero_edges(self):
        rng=np.random.default_rng(32)
        for _ in range(12):
            b=rng.choice([0.,.2,.7,1.],(6,6)); b=ph.symmetrize(b)
            ts=np.unique(np.r_[b.ravel(),np.nextafter(b.ravel(),-np.inf),1.])
            result=ph.analyze_graph(b,ts)
            for k,t in enumerate(ts[ts>=0]):
                idx=np.flatnonzero(ts==t)[0]
                np.testing.assert_array_equal(result['betti'][:,idx],oracle_betti(b,t))
            perm=rng.permutation(6)
            other=ph.persistence(b[perm][:,perm])
            for a,c in zip(result['diagrams'],other):
                self.assertEqual(ph.diagram_distance(a,c,bottleneck=True),0.)

    def test_square_diagonal_kills_cycle(self):
        b=np.zeros((4,4))
        for i,j in ((0,1),(1,2),(2,3),(0,3)):b[i,j]=b[j,i]=.8
        b[0,2]=b[2,0]=.3
        d0,d1=ph.persistence(b)
        np.testing.assert_allclose(d1,[[.2,.7]])
        self.assertAlmostEqual(ph.summary(d1)['total_persistence'],.5)
        np.testing.assert_allclose(ph.landscape(d1,[.2,.45,.7],1),[[0,.25,0]],atol=1e-15)
        self.assertEqual(np.isinf(d0[:,1]).sum(),1)
        b[0,2]=b[2,0]=0
        self.assertTrue(np.isinf(ph.persistence(b)[1][0,1]))

    def test_distances_and_essential_parts(self):
        a=np.array([[.1,.7],[.2,.4]]); empty=np.empty((0,2))
        self.assertAlmostEqual(ph.diagram_distance(a,empty),.4)
        self.assertAlmostEqual(ph.diagram_distance(a,empty,bottleneck=True),.3)
        self.assertAlmostEqual(ph.diagram_distance(a,empty,order=2),np.sqrt(.1))
        self.assertEqual(ph.diagram_distance(empty,empty,bottleneck=True),0.)
        self.assertEqual(ph.diagram_distance([[0,np.inf]],empty),np.inf)
        self.assertAlmostEqual(ph.diagram_distance([[.2,np.inf]],[[.3,np.inf]]),.1)
        self.assertEqual(ph.diagram_distance(a,a),0.)

    def test_controls(self):
        b=np.array([[0,.1,.8],[.1,0,0],[.8,0,0]])
        np.testing.assert_array_equal(ph.rank_weights(b),ph.rank_weights(b**2))
        other=ph.shuffled_weights(b,np.random.default_rng(3))
        np.testing.assert_array_equal(np.sort(b[np.triu_indices(3,1)]),np.sort(other[np.triu_indices(3,1)]))
        np.testing.assert_array_equal(ph.analyze_graph(b)['density'],ph.analyze_graph(other)['density'])

    def test_known_winding_and_strict_threshold(self):
        x=np.array([[.05,.1],[.38,.1],[.71,.1]])
        b=np.full((3,3),.7); np.fill_diagonal(b,0)
        r=weighted_winding(x,np.eye(2),b,[.7,np.nextafter(.7,0),0])
        np.testing.assert_array_equal(r['rank'],[0,1,1])
        self.assertEqual(r['critical_rank1'],.7); self.assertTrue(np.isnan(r['critical_rank2']))
        # Two independent 3-cycles with no shared edges.
        x=np.vstack((x,[[.17,.05],[.17,.38],[.17,.71]])); b=np.zeros((6,6))
        for block in (slice(0,3),slice(3,6)):b[block,block]=.7
        np.fill_diagonal(b,0)
        self.assertEqual(weighted_winding(x,np.eye(2),b,[0])['rank'][0],2)

    def test_skew_minimum_image_and_invariances(self):
        L=np.array([[1.,.92],[0.,.3]]); rng=np.random.default_rng(123)
        f=rng.uniform(size=(9,2)); x=f@L.T
        dd,dist,ties=minimum_image_edges(x,L)
        shifts=np.array(list(itertools.product(range(-7,8),repeat=2)))
        ij=np.triu_indices(9,1); delta=f[ij[1]]-f[ij[0]]
        expected=np.linalg.norm((delta[:,None,:]-shifts)@L.T,axis=-1).min(axis=1)
        np.testing.assert_allclose(dist,expected,atol=1e-14)
        self.assertGreater(np.max(np.linalg.norm((delta-np.rint(delta))@L.T,axis=-1)-dist),.05)
        b=ph.symmetrize(rng.uniform(size=(9,9))); ref=weighted_winding(x,L,b)
        p=rng.permutation(9)
        variants=((x+np.array([.34,-.23]),b),(x+rng.integers(-5,6,(9,2))@L.T,b),(x[p],b[p][:,p]))
        for xx,bb in variants:
            out=weighted_winding(xx,L,bb)
            np.testing.assert_array_equal(ref['rank'],out['rank'])
            np.testing.assert_allclose([ref['critical_rank1'],ref['critical_rank2']],
                                       [out['critical_rank1'],out['critical_rank2']],equal_nan=True)

    def test_ambiguous_embedding_rejected(self):
        with self.assertRaisesRegex(ValueError,'degenerate'):
            weighted_winding([[0,0],[.5,0]],np.eye(2),[[0,1],[1,0]])
        r=weighted_winding([[0,0],[.5,0]],np.eye(2),np.zeros((2,2)))
        self.assertTrue(np.all(r['rank']==0))


if __name__=='__main__':unittest.main()
