"""Independent C/D/E/F checks; no production forward/AD changes or snapshot capture."""
import itertools
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from screening_analysis import topology as ph
from screening_analysis.multiscale_topology import binary_betti,analyze_multiscreen
from screening_analysis.directed_loops import oriented_triangles,analyze_loops,bounded_asymmetry
from screening_analysis.hidden_topology import pool_hidden,point_cloud_ph,grouped_centroid
from screening_analysis.physical_loops import (dvr_basis,uniform_quadrature,one_body_samples,
    density_summary,physical_loop_statistics,independent_loop_products,TorusBasis,sample_wavefunction)
from screening_analysis.berry_validation import (berry_links,qwz_manybody_grid,qwz_kubo_integral,qwz_twist_hamiltonian,
                                                slater_berry_links,qwz_occupied_grid)


def slater_evaluator(basis,coefficients,conjugate=False):
    def evaluate(x,s):
        values=np.linalg.det(basis(x,s)@coefficients)
        if conjugate:values=values.conj()
        logabs=np.full(len(values),-np.inf)
        np.log(np.abs(values),out=logabs,where=np.abs(values)>0)
        return np.angle(values),logabs
    return evaluate


class MultiscaleChecks(unittest.TestCase):
    def test_binary_boundary_rank_all_four_vertex_graphs(self):
        ij=np.triu_indices(4,1)
        for code in range(64):
            a=np.zeros((4,4));a[ij]=[(code>>k)&1 for k in range(6)];a+=a.T
            np.testing.assert_array_equal(binary_betti(a),ph.analyze_graph(a,[0])['betti'][:,0])

    def fixture(self):
        rng=np.random.default_rng(13);x=rng.uniform(size=(5,2));s=rng.uniform(-1,1,(2,3,5,5));m=rng.uniform(size=s.shape)
        c=rng.normal(size=(2,5,3,4));return x,s,m,c

    def test_head_particle_and_periodic_invariance(self):
        x,s,m,c=self.fixture();args=dict(lattice=np.eye(2),a_grid=[-.4,0,.4],b_grid=[.1,.3,.7],distance_bands=[0,.3,.7])
        ref=analyze_multiscreen(s,m,x,head_vectors=c,**args)
        hp=np.array([2,0,1]);p=np.array([3,1,4,0,2])
        transformed=analyze_multiscreen(s[:,hp][:,:,p][:,:,:,p],m[:,hp][:,:,p][:,:,:,p],
            x[p]+np.array([.43,-.25])+np.array([[1,2],[0,-1],[2,0],[-2,4],[0,0]]),
            head_vectors=c[:,p][:,:,hp],**args)
        for key in ('head_mean_betti','head_std_betti','head_max_betti','integrated_betti','integrated_rank',
                    'union_betti','consensus_betti','head_jaccard','cancellation_ratio','band_energy_proxy'):
            np.testing.assert_allclose(ref[key],transformed[key],atol=1e-12,equal_nan=True)

    def test_two_parameter_strict_thresholds(self):
        x=np.array([[.05,.1],[.38,.1],[.71,.1]]);s=np.ones((1,1,3,3))*.5
        result=analyze_multiscreen(s,np.ones_like(s),x,np.eye(2),[.5,.49],[.1,.35,.7],distance_bands=[0,.35,.7])
        np.testing.assert_array_equal(result['head_betti'][0,0,0,:,0],[3,3,3])
        np.testing.assert_array_equal(result['head_rank'][0,0,1],[0,1,1])
        # A filled flag triangle can have zero H1 while its embedded edges wind.
        self.assertEqual(result['head_betti'][0,0,1,1,1],0)

    def test_full_CDE_pipeline_head_invariance_and_phase_gate(self):
        from screening_analysis.topology_cde_run import analyze,evaluate_phases
        import xml.etree.ElementTree as ET
        rng=np.random.default_rng(619);B,L,H,N,E=4,2,3,5,6
        s=rng.uniform(-1,1,(B,L,H,N,N));m=rng.uniform(size=s.shape)
        probe=dict(S=s,Window=m,W_window=m*(s+1)/2,positions=rng.uniform(size=(B,N,2)),lattice=np.eye(2),dref=1.,
            head_vectors=rng.normal(size=(B,L,N,H,E)),hidden_output=rng.normal(size=(B,L,N,E)),metadata={})
        p=np.array([2,0,1]);permuted=dict(probe)
        for key in ('S','Window','W_window'):permuted[key]=probe[key][:,:,p]
        permuted['head_vectors']=probe['head_vectors'][:,:,:,p]
        with tempfile.TemporaryDirectory() as tmp:
            a=analyze(probe,Path(tmp)/'a',a_grid=[-1,0,1],b_grid=[0,.3,.8])
            b=analyze(permuted,Path(tmp)/'b',a_grid=[-1,0,1],b_grid=[0,.3,.8])
            for key in a['features']:np.testing.assert_allclose(a['features'][key],b['features'][key],atol=2e-13)
            self.assertEqual(evaluate_phases([a,b])['status'],'unavailable')
            for path in Path(tmp).glob('*/*.svg'):ET.parse(path)


class DirectedChecks(unittest.TestCase):
    def fixture(self):
        x=np.array([[.1,.1],[.35,.1],[.2,.32]])
        s=np.array([[0,.6,-.2],[-.2,0,.6],[.6,-.2,0.]])[None,None]
        w=np.maximum(s,0);return x,s,w

    def test_chirality_permutation_reflection_reverse(self):
        x,s,w=self.fixture();ref=analyze_loops(s,w,x,np.eye(2))
        np.testing.assert_allclose(ref['qk'],2.4)
        for p in itertools.permutations(range(3)):
            p=np.asarray(p);r=analyze_loops(s[:,:,p][:,:,:,p],w[:,:,p][:,:,:,p],x[p],np.eye(2))
            np.testing.assert_allclose(r['qk'],ref['qk'])
        reflected=x.copy();reflected[:,0]*=-1
        r=analyze_loops(s,w,reflected,np.eye(2));np.testing.assert_allclose(r['qk'],-ref['qk'])
        np.testing.assert_array_equal(ref['reverse_screening'],-ref['screening'])

    def test_periodic_ambiguity_and_zero_regularization(self):
        x,s,w=self.fixture()
        ref=analyze_loops(s,w,x,np.eye(2))
        other=analyze_loops(s,w,x+np.array([[1,2],[-2,0],[0,-3]])+.17,np.eye(2))
        np.testing.assert_allclose(ref['qk'],other['qk'])
        wind=oriented_triangles([[.05,.1],[.38,.1],[.71,.1]],np.eye(2))
        self.assertEqual(wind['excluded']['noncontractible'],1)
        tie=oriented_triangles([[0,0],[.5,0],[.2,.2]],np.eye(2))
        self.assertEqual(tie['excluded']['shortest_image_tie'],1)
        for epsilon in (1e-2,1e-6,1e-12):
            a=bounded_asymmetry(np.array([[0,0,1e-10],[0,0,0],[0,0,0]]),epsilon)
            self.assertTrue(np.isfinite(a).all());self.assertLessEqual(np.max(np.abs(a)),1)
        self.assertTrue(np.isfinite(ref['regularized_log_circulation']).all())
        zero=analyze_loops(s,w*0,x,np.eye(2));np.testing.assert_array_equal(zero['screening'],0)
        self.assertGreater(np.max(np.abs(bounded_asymmetry(w*1e-9,1e-2)-bounded_asymmetry(w*1e-9,1e-12))),.9)

    def test_orthogonal_qk_invariance_but_not_general_GL(self):
        rng=np.random.default_rng(81);q=rng.normal(size=(5,4));k=rng.normal(size=(5,4));o=np.linalg.qr(rng.normal(size=(4,4)))[0]
        def score(q,k):
            q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+1e-12)
            k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-12)
            return q@k.T
        np.testing.assert_allclose(score(q,k),score(q@o,k@o),atol=1e-14)
        A=np.diag([3.,.4,1.,2.]);self.assertGreater(np.max(np.abs(score(q,k)-score(q@A,k@np.linalg.inv(A).T))),.1)


class HiddenChecks(unittest.TestCase):
    def test_pooling_and_rotation_invariants(self):
        rng=np.random.default_rng(7);h=rng.normal(size=(12,2,5,8));p=rng.permutation(5)
        for method in ('mean','mean_std','moments','gram_spectrum'):
            np.testing.assert_allclose(pool_hidden(h,method),pool_hidden(h[:,:,p],method),atol=1e-13)
        o=np.linalg.qr(rng.normal(size=(8,8)))[0]
        np.testing.assert_allclose(pool_hidden(h,'gram_spectrum'),pool_hidden(h@o,'gram_spectrum'),atol=1e-13)

    def test_mean_collapse_and_euclidean_PH(self):
        h=np.array([[[1.,0.],[-1,0]],[[2.,0],[-2,0]],[[3.,0],[-3,0]]])
        self.assertEqual(np.max(np.abs(pool_hidden(h,'mean'))),0)
        self.assertGreater(np.std(pool_hidden(h,'mean_std')[:,2]),0)
        theta=np.arange(16)*2*np.pi/16;points=np.stack((np.cos(theta),np.sin(theta)),axis=-1)
        r=point_cloud_ph(points);self.assertGreater(r['summaries'][1]['max_persistence'],1.)
        r2=point_cloud_ph(points*3)
        self.assertAlmostEqual(r2['summaries'][1]['max_persistence'],3*r['summaries'][1]['max_persistence'])
        self.assertEqual(point_cloud_ph(np.zeros((4,3)))['summaries'][0]['essential_intervals'],1)

    def test_group_holdout(self):
        x=np.array([[0],[10],[.1],[10.1],[-.1],[9.9]])
        r=grouped_centroid(x,[0,1]*3,[1,1,2,2,3,3])
        self.assertEqual(r['accuracy'],1);self.assertEqual(r['independent_groups'],3)


class PhysicalChecks(unittest.TestCase):
    def test_exact_two_particle_1RDM_conjugation_and_trace(self):
        basis=dvr_basis([[1.,.2],[0.,1.]])
        q,qs,qw=uniform_quadrature(basis.lattice,(3,3),shift=(0,0))
        np.testing.assert_allclose(np.einsum('qa,q,qb->ab',basis(q).conj(),qw,basis(q)),np.eye(9),atol=1e-14)
        rng=np.random.default_rng(19);u=np.linalg.qr(rng.normal(size=(9,2))+1j*rng.normal(size=(9,2)))[0]
        pairs=np.array(list(itertools.product(range(9),repeat=2)));x=q[pairs];s=np.ones((len(x),2))
        evaluate=slater_evaluator(basis,u);ph0,la=evaluate(x,s);weights=np.exp(2*la);keep=weights>1e-20
        local=one_body_samples(evaluate,x[keep],s[keep],basis,q,qs,qw,chunk=10)
        result=density_summary(local,sample_weights=weights[keep]);expected=u.conj()@u.T
        np.testing.assert_allclose(result['G'],expected,atol=2e-14)
        self.assertAlmostEqual(result['trace'].real,2,places=13);self.assertLess(result['hermiticity_residual'],1e-13)
        chi=rng.normal(size=9);rephased=TorusBasis(basis.lattice,basis.modes,basis.unitary@np.diag(np.exp(1j*chi)),basis.centres)
        local_phase=one_body_samples(evaluate,x[keep],s[keep],rephased,q,qs,qw)
        expected_phase=np.exp(1j*(chi[:,None]-chi[None,:]))*expected
        np.testing.assert_allclose(density_summary(local_phase,sample_weights=weights[keep])['G'],expected_phase,atol=2e-14)
        def global_phase(xx,ss):
            phase,logabs=evaluate(xx,ss);return phase+2.83,logabs
        np.testing.assert_allclose(one_body_samples(global_phase,x[keep],s[keep],basis,q,qs,qw),local,atol=2e-13)
        np.testing.assert_allclose(one_body_samples(evaluate,x[keep,::-1],s[keep,::-1],basis,q,qs,qw),local,atol=2e-13)
        local_conj=one_body_samples(slater_evaluator(basis,u,True),x[keep],s[keep],basis,q,qs,qw)
        np.testing.assert_allclose(density_summary(local_conj,sample_weights=weights[keep])['G'],expected.conj(),atol=2e-14)

    def test_pseudospin_offdiagonal_coherence(self):
        basis=dvr_basis(np.eye(2),(1,1),components=(1,-1));q,qs,qw=uniform_quadrature(np.eye(2),(1,1),components=(1,-1))
        c=np.array([[1.],[2j]])/np.sqrt(5);evaluate=slater_evaluator(basis,c)
        x=q[:,None,:];s=qs[:,None];_,la=evaluate(x,s)
        local=one_body_samples(evaluate,x,s,basis,q,qs,qw)
        g=density_summary(local,sample_weights=np.exp(2*la))['G']
        np.testing.assert_allclose(g,c.conj()@c.T,atol=1e-14);self.assertAlmostEqual(np.trace(g).real,1)
        self.assertGreater(abs(g[0,1].imag),.3)

    def test_projected_particle_trace_is_not_total_particle_number(self):
        full=TorusBasis(np.eye(2),[[0,0],[1,0]]);truncated=TorusBasis(np.eye(2),[[0,0]])
        c=np.array([[.5],[np.sqrt(3)/2]])
        q,qs,qw=uniform_quadrature(np.eye(2),(8,8));x=q[:,None,:];s=np.ones((len(q),1))
        evaluate=slater_evaluator(full,c);_,la=evaluate(x,s)
        local=one_body_samples(evaluate,x,s,truncated,q,qs,qw)
        g=density_summary(local,np.exp(2*la))['G']
        np.testing.assert_allclose(g,[[.25]],atol=1e-14)

    def test_fixed_basis_loops_gauge_TR_and_basis_permutation(self):
        basis=dvr_basis(np.eye(2));rng=np.random.default_rng(23)
        u=np.linalg.qr(rng.normal(size=(9,2))+1j*rng.normal(size=(9,2)))[0];g=u.conj()@u.T
        r=physical_loop_statistics(g,basis);self.assertGreater(np.max(np.abs(r['imaginary_loop'])),1e-4)
        chi=rng.normal(size=9);gg=np.exp(1j*(chi[:,None]-chi[None,:]))*g
        np.testing.assert_allclose(physical_loop_statistics(gg,basis)['imaginary_loop'],r['imaginary_loop'],atol=1e-15)
        np.testing.assert_allclose(physical_loop_statistics(g.conj(),basis)['imaginary_loop'],-r['imaginary_loop'],atol=1e-15)
        p=rng.permutation(9);b=SimpleNamespace(size=9,centres=basis.centres[p],lattice=basis.lattice)
        np.testing.assert_allclose(np.sort(physical_loop_statistics(g[p][:,p],b)['imaginary_loop']),np.sort(r['imaginary_loop']),atol=1e-15)

    def test_independent_chain_product_and_no_trace_forcing(self):
        rng=np.random.default_rng(5);g=rng.normal(size=(4,3,3))+1j*rng.normal(size=(4,3,3));t=np.array([[0,1,2]])
        r=independent_loop_products(g,t)
        expected=np.mean([g[a,0,1]*g[b,1,2]*g[c,2,0] for a,b,c in itertools.permutations(range(4),3)])
        np.testing.assert_allclose(r['product'],[expected]);self.assertIn('jackknife_sem_imag',r)
        sample=np.tile(np.eye(2)[None]*.2,(4,1,1));self.assertAlmostEqual(density_summary(sample)['trace'].real,.4)
        with self.assertRaisesRegex(ValueError,'nodes'):
            one_body_samples(lambda x,s:(np.zeros(len(x)),np.full(len(x),-np.inf)),np.zeros((1,1,2)),np.ones((1,1)),
                dvr_basis(np.eye(2),(1,1)),np.zeros((1,2)),np.ones(1),np.ones(1))

    def test_fresh_periodic_sampler_independent_streams(self):
        evaluate=lambda x,s:(np.zeros(len(x)),np.zeros(len(x)))
        r=sample_wavefunction(evaluate,np.eye(2),1,seed=3,chains=4,sweeps=32,burn=4,steps_per_sample=1,pseudospin=True)
        self.assertEqual(r['positions'].shape,(32,4,1,2));np.testing.assert_array_equal(r['acceptance'],1.)
        np.testing.assert_array_equal(r['spin_acceptance'],1.)
        self.assertTrue(np.all((r['positions']>=0)&(r['positions']<1)))
        self.assertGreater(np.max(np.abs(r['positions'][:,0]-r['positions'][:,1])),.1)


class BerryChecks(unittest.TestCase):
    def test_lambda_jellium_C3_spinor_representation_and_eigenstates(self):
        from screening_analysis.rotation_probe import c3_local_ratios,c3_expectation
        L=np.array([[3.,-1.5],[0.,3*np.sqrt(3)/2]]);theta=2*np.pi/3
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        q0=2*np.pi*np.linalg.inv(L).T[:,0];momenta=np.stack([np.linalg.matrix_power(rotation,j)@q0 for j in range(3)])
        lam=.7;U=np.diag([1,np.exp(1j*theta)])
        def h(q):
            plus=q[0]+1j*q[1]
            return np.array([[lam**2*np.dot(q,q),-lam*plus.conjugate()],[-lam*plus,1]])
        np.testing.assert_allclose(h(rotation@q0),U@h(q0)@U.conj().T,atol=1e-14)
        rng=np.random.default_rng(51);x=(rng.uniform(size=(40,1,2))@L.T);s=rng.choice([-1.,1.],(40,1))
        for angular in (0,1,2):
            def evaluate(x,s):
                coefficients=np.exp(-1j*angular*theta*np.arange(3))
                orbital=np.where(s[:,0,None]==1,1.,lam*(momenta[:,0]+1j*momenta[:,1]))
                z=np.sum(coefficients*orbital*np.exp(1j*x[:,0]@momenta.T),axis=-1)
                return np.angle(z),np.log(abs(z))
            ratio=c3_local_ratios(evaluate,x,s,L);expected=np.exp(1j*angular*theta)
            np.testing.assert_allclose(ratio,expected,atol=1e-12)
            self.assertLess(c3_expectation(ratio)['candidate_eigenstate_residual'],1e-23)
            with self.assertRaisesRegex(ValueError,'no C3'):
                c3_local_ratios(evaluate,x,s,np.eye(2))

    def test_actual_twist_H_derivatives(self):
        theta=np.array([.43,.17]);h,dx,dy=qwz_twist_hamiltonian(theta,derivatives=True);step=1e-6
        np.testing.assert_allclose(h,h.conj().T,atol=1e-15)
        for axis,derivative in enumerate((dx,dy)):
            d=np.eye(2)[axis]*step
            fd=(qwz_twist_hamiltonian(theta+d)-qwz_twist_hamiltonian(theta-d))/(2*step)
            np.testing.assert_allclose(derivative,fd,atol=1e-10)
        kx=(2*np.pi*np.arange(2)+theta[0])/2;ky=(2*np.pi*np.arange(2)+theta[1])/2
        bands=np.array([np.sqrt(np.sin(x)**2+np.sin(y)**2+(-1+np.cos(x)+np.cos(y))**2) for x in kx for y in ky])
        np.testing.assert_allclose(np.linalg.eigvalsh(h),np.sort(np.r_[bands,-bands]),atol=2e-15)

    def test_manybody_chern_independent_kubo_and_refinement(self):
        for mass,expected in ((-1.,-1),(1.,1),(3.,0)):
            values=[]
            for grid in (6,10):
                states,gap=qwz_manybody_grid(grid,mass);r=berry_links(states)
                self.assertGreater(gap,1.9);self.assertAlmostEqual(r['chern'],expected,places=12)
                values.append(r['chern'])
            self.assertAlmostEqual(qwz_kubo_integral(24,mass),expected,places=8)

    def test_U1_and_complex_conjugation(self):
        z,gap=qwz_manybody_grid(6);r=berry_links(z);rng=np.random.default_rng(8)
        phase=np.exp(1j*rng.uniform(-np.pi,np.pi,z.shape[:2]))
        np.testing.assert_allclose(berry_links(z*phase[...,None])['curvature'],r['curvature'],atol=1e-14)
        self.assertAlmostEqual(berry_links(z.conj())['chern'],-r['chern'])
        with self.assertRaisesRegex(ValueError,'overlap'):
            berry_links(np.tile(np.eye(3)[:,None,:],(1,3,1)))

    def test_slater_determinant_links_and_occupied_gauge(self):
        v,gap=qwz_occupied_grid(6,size=(2,2));z,_=qwz_manybody_grid(6)
        direct=berry_links(z);slater=slater_berry_links(v)
        np.testing.assert_allclose(direct['curvature'],slater['curvature'],atol=1e-14)
        rng=np.random.default_rng(37)
        rotations=np.linalg.qr(rng.normal(size=(6,6,4,4))+1j*rng.normal(size=(6,6,4,4)))[0]
        np.testing.assert_allclose(slater_berry_links(v@rotations)['curvature'],slater['curvature'],atol=1e-14)


if __name__=='__main__':unittest.main()
