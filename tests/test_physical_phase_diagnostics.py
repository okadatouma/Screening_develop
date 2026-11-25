"""Independent small-graph definitions and a real diagnostic-to-report smoke test."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from screening_analysis import physical_phase_metrics as m
from screening_analysis.physical_phase_diagnostics import (read_config, validate_probe,
    wavefunction_diagnostics, write_csv, write_json, compare, finite_difference_checks)
from screening_analysis.directed_loops import oriented_triangles


def small_probe():
    rng = np.random.default_rng(811)
    b, n, h = 6, 4, 2
    s = rng.uniform(-.8, .95, (b, 1, h, n, n))
    r = np.full((1, h), 1.8)
    alpha = (np.maximum(s-(1-r[None, :, :, None, None]), 0)/r[None, :, :, None, None])**2
    positions = rng.uniform(0., 1., (b, n*2))
    x = positions.reshape(b, n, 2)
    d = np.linalg.norm(x[:, :, None]-x[:, None, :], axis=-1)
    window = np.broadcast_to(np.exp(-d[:, None, None]**2), s.shape).copy()
    return dict(S=s, Alpha=alpha, Window=window, W=alpha*window, W_window=alpha*window,
        r=r, w=np.ones((1,h))*2, d=d, dref=3., positions=positions,
        spins=np.tile([1,-1,1,-1], (b,1)), NE=n, NL=1, NH=h, lattice=np.eye(2)*4,
        head_vectors=rng.normal(size=(b,1,n,h,3)),
        metadata=dict(settings=dict(TRIM_POW=2), fixed_value=False))


class PhysicalPhaseMetrics(unittest.TestCase):
    def test_fourier_definition_signed_and_structure_identity(self):
        x = np.array([[.1,.2], [.4,.6], [.8,.2]])
        lattice = np.array([[2., .7], [0., 1.6]])
        modes, q = m.reciprocal_grid(lattice, 2)
        c = m.pair_cosines(x, lattice, modes)
        np.testing.assert_allclose(q @ lattice, 2*np.pi*modes, atol=1e-14)
        f = m.pair_fourier(np.ones(3), c)
        rho = np.exp(1j*x@q.T).sum(0)
        np.testing.assert_allclose(1+2/3*(f*3), abs(rho)**2/3, atol=1e-14)
        self.assertAlmostEqual(f[(modes==0).all(1)][0], 1)
        self.assertTrue(np.isnan(m.pair_fourier(np.zeros(3), c)).all())
        np.testing.assert_allclose(m.pair_fourier([1,-1,0], c, True), (c[0]-c[1])/2)
        with self.assertRaises(ValueError): m.pair_fourier([1,-1,0], c)

    def test_fourier_periodicity_and_particle_permutation(self):
        rng=np.random.default_rng(5);x=rng.uniform(0,1,(5,2));L=np.array([[2.,.4],[0.,2.]])
        w=rng.uniform(0,1,(5,5));w=(w+w.T)/2;ij=np.triu_indices(5,1)
        modes,_=m.reciprocal_grid(L,3)
        actual=m.pair_fourier(w[ij],m.pair_cosines(x,L,modes))
        order=np.array([2,4,0,1,3]);shift=x.copy();shift[1]+=L[:,0]-2*L[:,1]
        np.testing.assert_allclose(actual,m.pair_fourier(w[ij],m.pair_cosines(shift,L,modes)),atol=1e-14)
        np.testing.assert_allclose(actual,m.pair_fourier(w[order][:,order][ij],m.pair_cosines(x[order],L,modes)),atol=1e-14)

    def test_distance_mass_quantiles_and_invalid_bins(self):
        r=m.distance_statistics([1,2,3],[1,2,7],[0,1.5,2.5,3.5])
        np.testing.assert_allclose(r['probability'],[.1,.2,.7]);self.assertEqual(r['r90'],3)
        self.assertEqual(r['r50'],3);self.assertAlmostEqual(r['r_eff'],2.6);self.assertEqual(r['cdf'][-1],1)
        self.assertTrue(np.isnan(m.distance_statistics([1],[0],[0,2])['r90']))
        with self.assertRaises(ValueError):m.positive_weights([-1e-3])
        np.testing.assert_array_equal(m.positive_weights([-1e-14,1]),[0,1])

    def test_crossfit_has_no_evaluation_leak_and_handles_empty_constant(self):
        y=np.array([[1.,2.,3.],[10.,11.,12.],[2.,3.,4.],[11.,12.,13.]])
        labels=np.zeros_like(y,dtype=int)
        r=m.crossfit_residual(y,labels,3)
        self.assertAlmostEqual(r['prediction'][0,0],11.5)
        changed=y.copy();changed[0]=100
        r2=m.crossfit_residual(changed,labels,3)
        np.testing.assert_array_equal(r2['prediction'][::2],r['prediction'][::2])
        self.assertTrue(np.isnan(m.crossfit_residual(y,labels,100)['z']).all())
        self.assertTrue(np.isnan(m.crossfit_residual(np.ones_like(y),labels,3)['z']).all())

    def test_js_effective_heads_and_signed_output_cancellation(self):
        w=np.ones((2,2,3,3));vectors=np.ones((2,3,2,4));vectors[:,:,1]*=-1
        pairs,s=m.head_specialization(w,vectors)
        self.assertAlmostEqual(s['edge_effective_heads'],1.)
        self.assertAlmostEqual(s['cancellation_ratio'],0.)
        self.assertAlmostEqual(pairs[1]['js_nats'],0.)
        self.assertTrue(np.isnan(m.head_specialization(np.zeros_like(w))[1]['edge_effective_heads']))
        order=[1,0];_,t=m.head_specialization(w[:,order],vectors[:,:,order])
        self.assertEqual(s,t)

    def test_product_circulation_and_passive_transformations(self):
        x=np.array([[.1,.1],[.8,.2],[.3,.9]])
        L=np.eye(2)*4;a=np.array([[0,2,7],[3,0,4],[5,6,0.]])
        def vals(x,L,a):return m.product_circulation(a,oriented_triangles(x,L)['triangles'])
        expected=2*4*5-7*6*3
        self.assertAlmostEqual(vals(x,L,a)[0],expected)
        np.testing.assert_allclose(vals(x,L,(a+a.T)/2),0.)
        R=np.diag([-1.,1.]);np.testing.assert_allclose(vals(x@R.T,R@L,a),-vals(x,L,a))
        R=np.array([[0.,-1.],[1.,0.]]);np.testing.assert_allclose(vals(x@R.T,R@L,a),vals(x,L,a))
        perm=[2,0,1];np.testing.assert_allclose(vals(x[perm],L,a[perm][:,perm]),vals(x,L,a))
        shifted=x.copy();shifted[0]+=L[:,0];np.testing.assert_allclose(vals(shifted,L,a),vals(x,L,a))
        self.assertEqual(m.product_circulation(a,np.empty((0,3),int)).size,0)

    def test_rayleigh_ritz_raw_normalized_zero_degree_and_scale(self):
        w=np.array([[1,2,0],[3,1,4],[0,1,1.]])
        x=np.array([[0,0],[.2,.1],[.3,.4]]);L=np.eye(2)*3;s=[1,-1,1];shell=np.array([[1,0],[0,1]])
        rows,arrays=m.mode_projections(w,s,x,L,shell)
        self.assertTrue(all(r['rayleigh_ritz_pass'] for r in rows))
        rows2,arrays2=m.mode_projections(7*w,s,x,L,shell)
        np.testing.assert_allclose(arrays['matrix'],arrays2['matrix'])
        self.assertAlmostEqual(rows[0]['spin_rayleigh']*7,rows2[0]['spin_rayleigh'])
        self.assertAlmostEqual(rows[1]['spin_rayleigh'],rows2[1]['spin_rayleigh'])
        zero,_=m.mode_projections(np.zeros((3,3)),[1,1,1],x,L,shell)
        self.assertTrue(np.isnan(zero[0]['spin_rayleigh']));self.assertEqual(zero[0]['lambda_min'],0)
        # Complete graph K3 has eigenvalues 1,-1/2,-1/2 when normalized.
        _,arr=m.mode_projections(np.ones((3,3)),s,x,L,shell)
        np.testing.assert_allclose(arr['eigenvalues'],[1,-.5,-.5],atol=1e-14)

    def test_statistics_no_pseudoreplication_and_ties(self):
        self.assertAlmostEqual(m.corr([1,2,3],[3,5,8],True),1.)
        self.assertTrue(np.isnan(m.corr([1,1],[2,3])))
        self.assertTrue(np.isnan(m.top_overlap(np.ones(10),np.arange(10))))
        s=m.block_summary(np.repeat([1.,2.,3.,4.],2),4)
        self.assertAlmostEqual(s['block_sem'],np.std([1,2,3,4],ddof=1)/2)
        self.assertTrue(np.isnan(m.block_summary(np.arange(8))['block_sem']))

    def test_complete_small_probe_reproducibility_and_ritz(self):
        p=small_probe();self.assertTrue(validate_probe(p)['pass_definitions'])
        settings=dict(distance_bins=3,qmax=3,min_bin_count=3,rs=4)
        tables,a=m.analyze_probe(p,settings)
        tables2,b=m.analyze_probe(p,settings)
        self.assertTrue(all(r['rayleigh_ritz_pass'] for r in tables['eigen']))
        self.assertEqual(a.keys(),b.keys())
        for key in a:np.testing.assert_allclose(a[key],b[key],equal_nan=True)
        self.assertTrue(all(r['r90_network']<=3 for r in tables['head']))
        self.assertEqual(len(tables['head']),8)

    def test_production_probe_independent_definition(self):
        from tests.check_attn_probe import fixture
        from screening_analysis.attn_probe import probe
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,_=fixture(Path(tmp))
            p=probe(tmp,settings=settings,lattice=L,collect_internal=True)
            self.assertLess(validate_probe(p)['alpha_formula_max_error'],1e-12)
            tables,arrays=m.analyze_probe(p,dict(distance_bins=3,qmax=3,min_bin_count=2,rs=4))
            self.assertTrue(all(row['rayleigh_ritz_pass'] for row in tables['eigen']))
            self.assertIn('fourier_screening_l0_h0',arrays)
            with self.assertRaises(AssertionError):
                validate_probe(dict(p,Alpha=p['Alpha']*.9))

    def test_coordinate_instability_and_head_ablation_with_toy_wavefunction(self):
        import jax.numpy as jnp
        class ToyNet:
            class options:value_dim=1
            def apply(self,p,x,s,a,c):
                scale=p['layers']['attention'][0]['attn_output'].sum()
                return scale*jnp.sum(x*x),scale*jnp.sum(x)
        params=dict(layers=dict(attention=[dict(attn_output=jnp.ones((2,1)))]))
        p=small_probe();p.update(atoms=np.zeros((6,1,2)),charges=np.zeros((6,1)))
        rows,g=wavefunction_diagnostics(ToyNet(),params,p,dict(coordinate_epsilon=1e-4,coordinate_samples=1,ablation_samples=1))
        self.assertEqual(rows[0]['status'],'ok');self.assertEqual(len(g),1)
        self.assertEqual(len([r for r in rows if r['kind']=='head_ablation']),2)
        class Singular(ToyNet):
            def apply(self,p,x,s,a,c):return jnp.nan,jnp.nan
        rows,g=wavefunction_diagnostics(Singular(),params,p,dict(coordinate_epsilon=1e-4,coordinate_samples=1,ablation_samples=1))
        self.assertEqual(rows[0]['status'],'nonfinite wavefunction/derivative');self.assertFalse(g)

    def test_config_and_empty_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.json';path.write_text(json.dumps(dict(output='unused',samples=2,phase_samples=3)))
            with self.assertRaises(ValueError):read_config(path)
            path.write_text(json.dumps(dict(output='unused',unknown=True)))
            with self.assertRaises(ValueError):read_config(path)
        self.assertEqual(compare([]),[])

    def test_saved_finite_difference_requires_both_channels_and_checks(self):
        g=np.ones((1,2,1,1,1))
        r=dict(sample='0',layer='0',head='0',edge='0',epsilon='0.0005',
            amplitude_error='1e-9',phase_scaled_error='1e-9',identity_error='0')
        self.assertTrue(finite_difference_checks([r],g)['passed'])
        self.assertFalse(finite_difference_checks([],g)['passed'])
        self.assertFalse(finite_difference_checks([dict(r,amplitude_error='nan')],g)['passed'])
        self.assertFalse(finite_difference_checks([dict(r,amplitude_error='0.01')],g)['passed'])


if __name__=='__main__':unittest.main()
