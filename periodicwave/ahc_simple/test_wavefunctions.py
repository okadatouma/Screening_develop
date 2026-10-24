"""実行：python -m unittest periodicwave.ahc_simple.test_wavefunctions -v

小規模の数学検算。既存のAHC/WC研究データやVMC最適化は検証対象にしない。
float64/complex128の正則な小行列で絶対/相対2e-13を基本基準とする。
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .spinor import lower_spinor, pseudospin, select_component, one_particle_hamiltonian
from .pfaffian import canonical_pairs, slater_geminal, pfaffian4, pfaffian
from .wavefunctions import (
    load_hf_table, make_bloch_orbitals, bloch_spinor, bloch_orbital, unprojected_bloch_spinor,
    orbital_matrix, slater_wavefunction, periodic_distance, relative_displacement,
    jastrow, backflow, geminal_matrix, wavefunction,
)
from .example import demo_data


class WavefunctionTests(unittest.TestCase):
    def assertComplexClose(self, actual, expected, atol=2e-13):
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=atol)

    def test_spinor_eigenvectors_normalization_and_analytic_pseudospin(self):
        for lam in (0.,.3,.72):
            for k in ([0.,0.],[.7,-.4],[-1.2,.9]):
                k=np.asarray(k);u=lower_spinor(k,lam);q2=k@k
                self.assertComplexClose(np.vdot(u,u),1.)
                self.assertComplexClose(one_particle_hamiltonian(k,lam,2.)@u,[0.,0.])
                expected=np.array([2*lam*k[0],2*lam*k[1],1-lam**2*q2])/(1+lam**2*q2)
                self.assertComplexClose(pseudospin(u),expected)
                upper=np.array([-np.conj(u[1]),np.conj(u[0])])
                self.assertComplexClose(one_particle_hamiltonian(k,lam,2.)@upper,2*(1+lam**2*q2)*upper)
        # 任意の二成分出力は、下側バンドだけという制約を持たない。
        self.assertGreater(np.linalg.norm(one_particle_hamiltonian([.7,.2],.72)@np.array([0.,1.])),0.)

    def test_bloch_component_and_two_spinor_representations(self):
        k=np.array([.2,-.3]);G=np.array([[0.,0.],[1.,.5],[-.4,.9]])
        coeff=np.array([1.+.2j,.3-.1j,-.2+.4j]);r=np.array([.4,.7])
        value=bloch_spinor(r,k,G,coeff,.72)
        direct=sum(c*lower_spinor(k+g,.72)*np.exp(1j*(k+g)@r) for g,c in zip(G,coeff))
        self.assertComplexClose(value,direct)
        for s in (-1,1):self.assertComplexClose(bloch_orbital(r,s,k,G,coeff,.72),select_component(value,s))
        R,S,discrete,_,momenta=demo_data()
        spinors=make_bloch_orbitals(momenta,np.zeros((1,2)),np.ones((4,1)),.7,spinor_output=True)
        self.assertComplexClose(orbital_matrix(R,S,discrete),orbital_matrix(R,S,spinors,spinor_output=True))
        self.assertComplexClose(slater_wavefunction(R,S,discrete),slater_wavefunction(R,S,spinors,spinor_output=True))

    def test_hf_six_column_reader_and_single_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp,'hf.txt')
            table=np.array([[0,0,0,0,1,.2],[0,0,1,0,.3,-.1],
                            [.5,0,0,0,.4,.3],[.5,0,1,0,.7,-.2]])
            np.savetxt(path,table,fmt='%.17g');original=path.read_bytes()
            k,G,c=load_hf_table(path)
            self.assertEqual(k.shape,(2,2));self.assertEqual(G.shape,(2,2))
            self.assertComplexClose(c,[[1+.2j,.3-.1j],[.4+.3j,.7-.2j]])
            self.assertEqual(path.read_bytes(),original)
            np.savetxt(path,table[:1]);self.assertEqual(load_hf_table(path)[2].shape,(1,1))

    def test_unprojected_complex_spinor_slater(self):
        from functools import partial
        R,S,_,_,k=demo_data();G=np.zeros((1,2))
        spinors=[partial(unprojected_bloch_spinor,k=ka,G=G,coeff=[[1.+.1j*a,.3+.2j*a]]) for a,ka in enumerate(k)]
        discrete=[lambda r,s,orb=orb:select_component(orb(r),s) for orb in spinors]
        self.assertComplexClose(slater_wavefunction(R,S,spinors,spinor_output=True),slater_wavefunction(R,S,discrete))

    def test_pfaffian4_sign_and_complex_phase(self):
        F=np.array([[0,1+2j,2-3j,.7+.9j],[-1-2j,0,1.1-.4j,-.5+1j],
                    [-2+3j,-1.1+.4j,0,3-1j],[-.7-.9j,.5-1j,-3+1j,0]])
        expected=(1+2j)*(3-1j)-(2-3j)*(-.5+1j)+(.7+.9j)*(1.1-.4j)
        self.assertComplexClose(pfaffian4(F),expected)
        swap=[1,0,2,3];self.assertComplexClose(pfaffian(F[np.ix_(swap,swap)]),-expected)

    def test_general_pfaffian_matches_independent_recursive_expansion(self):
        def expansion(F):
            if len(F)==0:return 1.+0j
            answer=0j
            for j in range(1,len(F)):
                indices=[i for i in range(len(F)) if i not in (0,j)]
                answer+=(-1)**(j+1)*F[0,j]*expansion(F[np.ix_(indices,indices)])
            return answer
        rng=np.random.default_rng(4)
        for n in (0,2,4,6,8):
            X=(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))/3
            F=X-X.T
            self.assertComplexClose(pfaffian(F),expansion(F))
            self.assertComplexClose(pfaffian(F)**2,np.linalg.det(F))

    def test_slater_pfaffian_identity(self):
        rng=np.random.default_rng(15)
        for n in (2,4,6,8):
            # well-conditionedな複素Φ。実数だけの検算では位相規約を検査できない。
            Phi=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
            F=slater_geminal(Phi)
            self.assertComplexClose(F,-F.T)
            self.assertComplexClose(pfaffian(canonical_pairs(n)),1.)
            self.assertComplexClose(pfaffian(F),np.linalg.det(Phi))

    def test_invalid_pfaffian_and_exact_singular_state(self):
        with self.assertRaises(ValueError):pfaffian(np.zeros((3,3)))
        with self.assertRaises(ValueError):pfaffian(np.eye(4))
        self.assertEqual(pfaffian(np.zeros((6,6))),0j)

    def test_jastrow_zero_complex_and_pair_channels(self):
        R=np.array([[0.,0.],[.3,.4]]);S=np.array([1,-1])
        p=dict(a=[.2,.3,.4+.1j],b=[.6,.7,.8])
        self.assertComplexClose(jastrow(R,S,p),-(.4+.1j)*np.exp(-.5/.8))
        self.assertEqual(jastrow(R,S,dict(a=0.,b=.8)),0j)
        with self.assertRaises(ValueError):jastrow(R,S,dict(a=1.,b=0.))

    def test_backflow_equivariance_spin_modes_and_zero_limit(self):
        R,S,_,_,_=demo_data();L=np.eye(2)*4;p=[2,0,3,1]
        for parameters,spins in ((dict(eta=.08,b=1.),None),(dict(eta=[.08,-.03,.05],b=1.),S),
                                 (dict(eta=.08,b=1.,mode='minimum_image'),None)):
            moved=backflow(R,parameters,L,spins=spins)
            permuted=backflow(R[p],parameters,L,spins=None if spins is None else S[p])
            self.assertComplexClose(permuted,moved[p])
        np.testing.assert_array_equal(backflow(R,dict(eta=0.,b=1.),L),R)
        with self.assertRaises(ValueError):backflow(R,dict(eta=[1.,2.,3.],b=1.),L)

    def test_pbc_minimum_image_and_smooth_boundary_derivative(self):
        L=np.array([[3.,-1.5],[0.,2.6]]);dr=L@np.array([.49,-.49])
        mic=relative_displacement(dr,L,'minimum_image')
        self.assertLess(np.linalg.norm(mic),np.linalg.norm(dr))
        self.assertComplexClose(periodic_distance(dr+L[:,0],L),periodic_distance(dr,L))
        def shift(x):return relative_displacement(np.array([x,.7]),np.eye(2)*4)
        h=1e-4
        for x in (0.,4.):
            derivative=(shift(x+h)-shift(x-h))/(2*h)
            self.assertComplexClose(derivative,[1.,0.],atol=5e-9)

    def test_all_models_exchange_periodicity_and_finite_values(self):
        L=np.array([[4.,-1.],[0.,3.5]])
        for n in (4,6):
            R,S,orbitals,extra,_=demo_data(n,lattice=L)
            J=dict(a=[.08+.01j,.1-.02j,.12+.03j],b=.8)
            BF=dict(eta=[.05,-.02,.04],b=1.2)
            models=[{},dict(jastrow_parameters=J),dict(backflow_parameters=BF),
                    dict(jastrow_parameters=J,backflow_parameters=BF),
                    dict(use_pfaffian=True,eta=.2,geminal_orbitals=extra),
                    dict(use_pfaffian=True,eta=.2,geminal_orbitals=extra,jastrow_parameters=J,backflow_parameters=BF)]
            swap=np.arange(n);swap[:2]=[1,0]
            for options in models:
                psi=wavefunction(R,S,orbitals,lattice=L,**options)
                self.assertTrue(np.isfinite(psi));self.assertGreater(abs(psi),1e-10)
                self.assertComplexClose(wavefunction(R[swap],S[swap],orbitals,lattice=L,**options),-psi)
                for axis in range(2):
                    moved=R.copy();moved[0]+=L[:,axis]
                    self.assertComplexClose(wavefunction(moved,S,orbitals,lattice=L,**options),psi)

    def test_common_bloch_twist_is_preserved(self):
        L=np.eye(2)*4;R,S,_,_,k=demo_data();twist=np.array([.17,-.11])
        orbitals=make_bloch_orbitals(k+twist,np.zeros((1,2)),np.ones((4,1)),.7)
        for pf in (False,True):
            options=dict(use_pfaffian=pf,backflow_parameters=dict(eta=.05,b=1.))
            psi=wavefunction(R,S,orbitals,lattice=L,**options)
            moved=R.copy();moved[0]+=L[:,0]
            self.assertComplexClose(wavefunction(moved,S,orbitals,lattice=L,**options),np.exp(1j*(twist@L[:,0]))*psi)

    def test_zero_corrections_geminal_and_nontrivial_pair_extension(self):
        R,S,orbitals,extra,_=demo_data();psi=slater_wavefunction(R,S,orbitals)
        options=dict(jastrow_parameters=dict(a=0.,b=1.),backflow_parameters=dict(eta=0.,b=1.))
        self.assertComplexClose(wavefunction(R,S,orbitals,**options),psi)
        self.assertComplexClose(wavefunction(R,S,orbitals,use_pfaffian=True,eta=0.,**options),psi)
        F=geminal_matrix(R,S,*extra);self.assertComplexClose(F,-F.T)
        # 一組の外積だけはrank<=2なので、N=4単独のPfはゼロ。HF項が重要。
        self.assertComplexClose(pfaffian(F),0.)
        ratios=[]
        for dx in (0.,.31):
            r=R.copy();r[0,0]+=dx
            ratios.append(wavefunction(r,S,orbitals,use_pfaffian=True,eta=.2,geminal_orbitals=extra)/slater_wavefunction(r,S,orbitals))
        self.assertGreater(abs(ratios[0]-ratios[1]),1e-6)

    def test_pfaffian_directional_derivative_without_conjugation(self):
        rng=np.random.default_rng(12);a=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
        F=a-a.T;a=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));dF=a-a.T
        exact=.5*pfaffian(F)*np.trace(np.linalg.solve(F,dF))
        h=1e-5;finite=(pfaffian(F+h*dF)-pfaffian(F-h*dF))/(2*h)
        np.testing.assert_allclose(finite,exact,atol=2e-8,rtol=2e-8)

    @unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('folx'),
                         '既存JAX/folx環境がないためProduction HF照合は未実行')
    def test_existing_production_hf_orbitals(self):
        import jax
        import jax.numpy as jnp
        from periodicwave import CustomScreenformer as CS, envelopes, lj_orbitals
        from periodicwave.pbc import feature_layer
        with tempfile.TemporaryDirectory() as tmp,jax.experimental.enable_x64():
            R,S,_,_,k=demo_data();L=np.eye(2)*4
            G=np.array([[0.,0.],[np.pi/2,0.],[0.,np.pi/2]])
            c=np.tile([1.,.1+.2j,.05-.1j],(4,1))
            orbitals=make_bloch_orbitals(k,G,c,.7)
            table=np.array([[*ka,*g,z.real,z.imag] for ka,row in zip(k,c) for g,z in zip(G,row)])
            path=Path(tmp,'hf.txt');np.savetxt(path,table,fmt='%.17g');original=path.read_bytes()
            for a,b in zip(load_hf_table(path),lj_orbitals.load_hf_table(path)):self.assertComplexClose(a,b)
            net=CS.make_fermi_net((4,0),jnp.zeros(1),ndim=2,determinants=1,complex_output=True,
                envelope=envelopes.make_null_envelope(),jastrow='NONE',jastrow_kwargs={'ndim':2},
                feature_layer=feature_layer.make_pbc_feature_layer(natoms=1,nspins=(4,0),lattice=jnp.asarray(L),include_r_ae=False),
                pbc_lattice=jnp.asarray(L),num_layers=0,mlp_dim=4,num_heads=1,attn_dim=2,value_dim=2,
                num_perceptrons_per_layer=1,use_layer_norm=False,mlp_activation_fct='GELU',
                lj_hf_path=str(path),lj_lam=.7,lj_net_scale=0.,lj_jastrow=False)
            p=net.init(jax.random.PRNGKey(7));args=(jnp.asarray(R.ravel()),jnp.asarray(S),jnp.zeros((1,2)),jnp.zeros(1))
            actual=net.orbitals(p,*args)[0][0]
            self.assertComplexClose(actual,orbital_matrix(R,S,orbitals))
            ph,la=net.apply(p,*args);self.assertComplexClose(np.exp(complex(la)+1j*complex(ph)),slater_wavefunction(R,S,orbitals))
            self.assertEqual(path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
