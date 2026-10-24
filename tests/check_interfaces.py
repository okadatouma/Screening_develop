"""F0。CurrentContracts=現行仕様保持、IndependentPhysics=独立した数式検算。

本体や既存テストを変更しない。既存33件と区別して明示実行する。
"""
import itertools
import math
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics, make_case
require_numerics()
import attr
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import networks, network_blocks as NB, envelopes, jastrows
from periodicwave import CustomScreenformer as CS, CustomPsiformer as CP, SlaterNet as SN, hamiltonian
from periodicwave.pbc import feature_layer as FL, hamiltonian as PH, lambda_jellium as LJ


def periodic_distance(lattice, displacement):
    """NumPy参照。実空間変位 -> 周期距離（最小像距離ではない）。"""
    lattice=np.asarray(lattice,dtype=np.float64)
    u=np.asarray(displacement,dtype=np.float64)@np.linalg.inv(lattice).T
    a=1-np.cos(2*np.pi*u);b=np.sin(2*np.pi*u);g=lattice.T@lattice
    return np.sqrt(np.einsum('...i,ij,...j->...',a,g,a)+np.einsum('...i,ij,...j->...',b,g,b))/(2*np.pi)


def features(layer,geometry):
    # geometry tuple順(ae,ee,r_ae,r_ee)とFeatureApplyの引数順は異なる。
    ae,ee,ra,re=geometry
    return layer.apply(ae=ae,ee=ee,r_ae=ra,r_ee=re)


def controlled_orbitals(complex_output=True,jastrow='NONE',lattice=None,envelope=None):
    """Attentionと切り離した共変trunkで、軌道構成だけを検算する。"""
    def init(key):return 3, {}
    def apply(params,*,ae,spins,**kwargs):
        return jnp.stack((jnp.sin(ae[:,0,0]),jnp.cos(ae[:,0,1]),spins),axis=-1)
    opts=networks.BaseNetworkOptions(ndim=2,determinants=2,complex_output=complex_output,
        pbc_lattice=lattice,envelope=envelope or envelopes.make_null_envelope(),
        jastrow=jastrows.JastrowType[jastrow],jastrow_kwargs=dict(ndim=2,interaction_strength=2.))
    init_o,apply_o=networks.make_orbitals((2,1),jnp.zeros(1),opts,(init,apply))
    return init_o,apply_o,apply,opts


class CurrentContracts(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_geometry_shapes_orientation_and_masked_derivatives(self):
        L=jnp.asarray([[3.2,-1.6],[0.,2.7712812921102037]])
        x=jnp.asarray([.23,.41,1.17,.89]);atoms=jnp.zeros((1,2))
        periodic=networks.make_periodic_construct_input_features(L)
        ae,ee,ra,re=periodic(x,atoms,ndim=2)
        np.testing.assert_array_equal(ae[:,0],x.reshape(2,2))
        np.testing.assert_array_equal(ee[0,1],x[2:]-x[:2])
        self.assertEqual(ra.shape,(2,1))  # 現行PBC実装。docstringの末尾1軸とは異なる。
        self.assertEqual(networks.construct_input_features(x,atoms,2)[2].shape,(2,1,1))
        self.assertEqual(re.shape,(2,2,1))
        for fn in (lambda r:periodic(r,atoms,2)[3],
                   lambda r:features(FL.make_pbc_feature_layer(1,(1,1),2,lattice=L),periodic(r,atoms,2))[1][...,:1]):
            for value in (fn(x),jax.jacfwd(fn)(x),jax.jacfwd(jax.jacrev(fn))(x)):
                np.testing.assert_array_equal(value[np.arange(2),np.arange(2)],0.)

    def test_embedding_rng_tree_and_evaluation_all_three_ansatz(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for route,module,maker,block in (('psiformer',CP,'make_psiformer_layers','make_self_attention_block'),
                    ('screen',CS,'make_screenformer_layers','make_self_attention_block'),
                    ('slater',SN,'make_SlaterNet_layers','make_SlaterNet_block')):
                with self.subTest(x64=x64,route=route),tempfile.TemporaryDirectory() as tmp:
                    net,p,d,L,_,_=make_case(route,tmp)
                    _,trunk_key=jax.random.split(jax.random.PRNGKey(7))
                    key,_=jax.random.split(trunk_key)  # feature用split。現在の未使用subkeyも保持。
                    _,embed_key=jax.random.split(key)
                    weight_key,_=jax.random.split(embed_key)
                    shape=p['layers']['embed'].shape
                    expected=jax.random.normal(weight_key,shape)/jnp.sqrt(float(shape[0]))
                    np.testing.assert_array_equal(p['layers']['embed'],expected)
                    self.assertEqual(p['layers']['input'],{})
                    self.assertEqual(str(expected.dtype),'float64' if x64 else 'float32')
                    geom=networks.make_periodic_construct_input_features(L)(d.positions,d.atoms,2)
                    one,_=features(net.options.feature_layer,geom)
                    reference=jnp.concatenate((one,d.spins[:,None]),axis=-1)@p['layers']['embed']
                    # テスト内でtrunk後段だけ恒等にし、実際の埋込み入口を観察する。
                    with patch.object(module,block,return_value=(None,lambda pp,x,*a,**k:x)):
                        _,apply=getattr(module,maker)((1,1),1,net.options)
                    result=apply(p['layers'],ae=geom[0],ee=geom[1],r_ae=geom[2],r_ee=geom[3],spins=d.spins,charges=d.charges)
                    np.testing.assert_array_equal(result,reference)
                    rev=networks.make_periodic_construct_input_features(L)(d.positions.reshape(2,2)[::-1].reshape(-1),d.atoms,2)
                    permuted=apply(p['layers'],ae=rev[0],ee=rev[1],r_ae=rev[2],r_ee=rev[3],spins=d.spins[::-1],charges=d.charges)
                    np.testing.assert_array_equal(permuted,result[::-1])
        jax.clear_caches()

    def test_orbital_packing_merged_jastrow_and_envelope_position(self):
        x=jnp.asarray([.23,.41,1.17,.89,.65,1.4]);s=jnp.asarray([1.,1.,-1.]);a=jnp.zeros((1,2));c=jnp.zeros(1)
        L=jnp.eye(2)*3
        init,apply,trunk,_=controlled_orbitals(jastrow='SIMPLE_EE',lattice=L)
        p=init(jax.random.PRNGKey(11));geom=networks.make_periodic_construct_input_features(L)(x,a,2)
        h=trunk({},ae=geom[0],spins=s)
        raw=np.asarray(h)@np.asarray(p['orbital'][0]['w'])
        packed=raw[:,::2]+1j*raw[:,1::2]
        base=packed.reshape(3,2,3).transpose(1,0,2)
        pair=np.asarray(x).reshape(3,2)
        dist=periodic_distance(L,pair[None,:,:]-pair[:,None,:])[np.triu_indices(3,1)]
        J=np.sum(-(2./3)/(1+dist)) # nspins_merged=(3,0): 全てparallel。alpha=1。
        actual=apply(p,x,s,a,c)[0]
        np.testing.assert_allclose(actual,base*np.exp(J/3),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(actual),np.linalg.det(base)*np.exp(J),rtol=1e-11,atol=1e-12)
        altered={**p,'jastrow':{**p['jastrow'],'ee_anti':jnp.array([9.])}}
        np.testing.assert_array_equal(apply(altered,x,s,a,c)[0],actual)
        # 非周期幾何では非null EnvelopeのPRE_DETERMINANTを独立検算。
        init_e,apply_e,trunk_e,_=controlled_orbitals(lattice=None,envelope=envelopes.make_isotropic_envelope())
        pe=init_e(jax.random.PRNGKey(11));ge=networks.construct_input_features(x,a,2)
        raw=np.asarray(trunk_e({},ae=ge[0],spins=s))@np.asarray(pe['orbital'][0]['w'])
        z=raw[:,::2]+1j*raw[:,1::2]
        expected=z*np.exp(-np.linalg.norm(np.asarray(x).reshape(3,2),axis=1))[:,None]
        np.testing.assert_allclose(apply_e(pe,x,s,a,c)[0],expected.reshape(3,2,3).transpose(1,0,2),atol=1e-12)


    def test_node_and_near_node_current_behavior(self):
        for complex_output in (False,True):
            dtype=jnp.complex128 if complex_output else jnp.float64
            f=lambda t:NB.logdet_matmul([jnp.diag(jnp.array([1.,t],dtype=dtype))[None]])
            for t in (-2.**-8,2.**-8,-2.**-20,2.**-20):
                ph,la=f(t)
                np.testing.assert_allclose(la,np.log(abs(t)),atol=1e-12)
                sign=jnp.exp(1j*ph) if complex_output else ph
                np.testing.assert_allclose(sign,np.sign(t),atol=1e-12)
                np.testing.assert_allclose(jax.grad(lambda v:f(v)[1])(t),1/t,rtol=1e-12)
                np.testing.assert_allclose(jax.hessian(lambda v:f(v)[1])(t),-1/t**2,rtol=1e-12)
            self.assertFalse(np.isfinite(np.asarray(f(0.)[1])))
        # 各行列は非特異でも、和が0の節は別に存在する。
        matrices=jnp.stack((jnp.eye(2,dtype=jnp.complex128),jnp.diag(jnp.array([-1.,1.],dtype=jnp.complex128))))
        _,la=NB.logdet_matmul([matrices]);self.assertTrue(np.isneginf(la))


class IndependentPhysics(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_periodic_norm_analytic_first_second_derivatives(self):
        lengths=np.array([3.2,2.7]);r=np.array([.43,.67])
        D=np.sqrt(np.sum((lengths/np.pi*np.sin(np.pi*r/lengths))**2))
        G=lengths/(2*np.pi)*np.sin(2*np.pi*r/lengths)
        grad=G/D;hess=np.diag(np.cos(2*np.pi*r/lengths))/D-np.outer(G,G)/D**3
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            L=jnp.diag(jnp.asarray(lengths));x=jnp.asarray(r)
            f=lambda y:FL.periodic_norm(L.T@L,y/jnp.asarray(lengths))
            for observed,expected in ((f(x),D),(jax.grad(f)(x),grad),(jax.hessian(f)(x),hess),
                                      (jax.jit(jax.hessian(f))(x),hess)):
                eps=np.finfo(np.asarray(x).dtype).eps
                np.testing.assert_allclose(observed,expected,rtol=32*eps,atol=32*eps)
            geom=networks.make_periodic_construct_input_features(L)
            g=lambda y:geom(jnp.concatenate((jnp.zeros(2),y)),jnp.zeros((1,2)),2)[3][0,1,0]
            np.testing.assert_allclose(jax.grad(g)(x),grad,rtol=32*eps,atol=32*eps)
            np.testing.assert_allclose(jax.hessian(g)(x),hess,rtol=32*eps,atol=32*eps)

    def test_features_values_covariance_and_pbc(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            L=jnp.asarray([[3.2,-1.6],[0.,2.7712812921102037]])
            r=jnp.asarray([[.23,.41],[1.17,.89],[.65,1.4]]);a=jnp.zeros((1,2));perm=np.array([2,0,1])
            geom=networks.make_periodic_construct_input_features(L)
            for include in (False,True):
                for rescale in (False,True):
                    fl=FL.make_pbc_feature_layer(1,(2,1),2,rescale,lattice=L,include_r_ae=include)
                    one,pair=features(fl,geom(r.reshape(-1),a,2))
                    u=np.asarray(r,dtype=np.float64)@np.linalg.inv(np.asarray(L,dtype=np.float64)).T
                    d=periodic_distance(L,np.asarray(r));d=np.log1p(d) if rescale else d
                    ref=np.concatenate(([d[:,None]] if include else [])+[np.sin(2*np.pi*u),np.cos(2*np.pi*u)],axis=1)
                    dr=np.asarray(r)[None,:,:]-np.asarray(r)[:,None,:]
                    frac=dr@np.linalg.inv(np.asarray(L,dtype=np.float64)).T
                    distance=periodic_distance(L,dr);np.fill_diagonal(distance,0.)
                    refpair=np.concatenate((distance[...,None],np.sin(2*np.pi*frac),np.cos(2*np.pi*frac)),axis=-1)
                    eps=np.finfo(np.asarray(r).dtype).eps
                    np.testing.assert_allclose(one,ref,rtol=32*eps,atol=32*eps)
                    np.testing.assert_allclose(pair,refpair,rtol=32*eps,atol=32*eps)
                    dims,pp=fl.init();self.assertEqual(dims,(5 if include else 4,5));self.assertEqual(pp,{})
                    op,ep=features(fl,geom(r[perm].reshape(-1),a,2))
                    np.testing.assert_array_equal(op,one[perm]);np.testing.assert_array_equal(ep,pair[perm][:,perm])
                    shifted=r.at[1].add(L[:,0]);os,es=features(fl,geom(shifted.reshape(-1),a,2))
                    np.testing.assert_allclose(os,one,atol=32*eps,rtol=32*eps)
                    np.testing.assert_allclose(es,pair,atol=32*eps,rtol=32*eps)

    def test_determinant_sum_real_sign_and_complex_phase(self):
        for complex_output in (False,True):
            a=np.array([[[1.,2.],[.3,1.1]],[[-.2,.4],[.7,.8]]])
            if complex_output:a=a+1j*np.array([[[.2,0.],[.1,-.3]],[[.4,.2],[0.,.3]]])
            value=np.linalg.det(a).sum();ph,la=NB.logdet_matmul([jnp.asarray(a)])
            expected=value/abs(value)
            np.testing.assert_allclose(jnp.exp(1j*ph) if complex_output else ph,expected,atol=1e-12)
            np.testing.assert_allclose(la,np.log(abs(value)),atol=1e-12)

    def test_hf_nn_correction_dynamic_spins_and_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            net,p,d,L,kw,_=make_case('hf',tmp)
            noJ=CS.make_fermi_net((2,0),d.charges,**dict(kw,lj_jastrow=False))
            _,trunk=CS.make_screenformer_layers((2,0),1,net.options)
            table=np.loadtxt(kw['lj_hf_path']);k=table[:,:2];lam=kw['lj_lam'];r=np.asarray(d.positions).reshape(2,2)
            for spins in (jnp.array([1.,-1.]),jnp.array([-1.,1.]),jnp.array([1.,1.]),jnp.array([-1.,-1.])):
                geom=networks.make_periodic_construct_input_features(L)(d.positions,d.atoms,2)
                h=trunk(p['layers'],ae=geom[0],ee=geom[1],r_ae=geom[2],r_ee=geom[3],spins=spins,charges=d.charges)
                raw=np.asarray(h)@np.asarray(p['orbital'][0]['w'])
                out=(raw[:,::2]+1j*raw[:,1::2]).reshape(2,2,2,2)
                nn=np.where(np.asarray(spins)[:,None,None]>0,out[:,0],out[:,1])
                u=np.where(np.asarray(spins)[:,None]>0,1.+0j,lam*(k[:,0]+1j*k[:,1])[None,:])
                hf=np.exp(1j*r@k.T)*u/np.sqrt(1+lam**2*np.sum(k*k,axis=1))[None,:]
                expected=(hf[:,None,:]+kw['lj_net_scale']*nn).transpose(1,0,2)
                actual=noJ.orbitals(p,d.positions,spins,d.atoms,d.charges)[0]
                np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)
                self.assertGreater(np.max(np.abs(actual-hf[None])),1e-6)
                distance=float(periodic_distance(L,r[1]-r[0]))
                key='alpha_pp' if np.all(np.asarray(spins)>0) else 'alpha_mm' if np.all(np.asarray(spins)<0) else 'alpha_pm'
                gamma=kw['lj_cusps'][('alpha_pp','alpha_mm','alpha_pm').index(key)]
                alpha=np.logaddexp(0.,np.asarray(p['jastrow_ss'][key])[0])
                J=-gamma*alpha*np.exp(-distance/alpha)
                withJ=net.orbitals(p,d.positions,spins,d.atoms,d.charges)[0]
                np.testing.assert_allclose(withJ,expected*np.exp(J/2),rtol=1e-11,atol=1e-12)
                ph,la=net.apply(p,d.positions,spins,d.atoms,d.charges)
                phe,lae=net.apply(p,d.positions.reshape(2,2)[::-1].reshape(-1),spins[::-1],d.atoms,d.charges)
                np.testing.assert_allclose(jnp.exp(lae+1j*phe),-jnp.exp(la+1j*ph),rtol=1e-9,atol=1e-12)
        jax.clear_caches()

    def test_lambda_spin_flip_and_weighted_complex_kinetic(self):
        x=jnp.asarray([.23,.41,1.17,.89]);s=jnp.asarray([1.,-1.]);lam=.4;delta=2.
        b=np.array([[.2,-.1],[-.3,.15]]);q=np.array([[.7,-.2],[-.4,.5]]);offset=np.array([.1,-.2])
        def f(p,r,spins,atoms,charges):
            rr=r.reshape(2,2);idx=(spins<0).astype(int)
            return jnp.sum(jnp.asarray(q)[idx]*rr)+.1*jnp.sum(rr*rr),jnp.sum(jnp.asarray(offset)[idx])+jnp.sum(jnp.asarray(b)[idx]*rr)-.2*jnp.sum(rr*rr)
        for spins in (s,-s):
            rr=np.asarray(x).reshape(2,2);idx=(np.asarray(spins)<0).astype(int)
            z=b[idx]-.4*rr+1j*(q[idx]+.2*rr)
            coeff=.5+delta*lam**2*(np.asarray(spins)>0)
            expected=-np.sum(coeff*(2*(-.4+.2j)+np.sum(z*z,axis=1)))+delta*np.sum(np.asarray(spins)<0)
            for i in range(2):
                other=1-idx[i];zz=b[other]-.4*rr[i]+1j*(q[other]+.2*rr[i])
                ratio=np.exp(offset[other]-offset[idx[i]]+(b[other]-b[idx[i]])@rr[i]+1j*(q[other]-q[idx[i]])@rr[i])
                expected+=1j*delta*lam*ratio*(zz[0]-1j*float(spins[i])*zz[1])
            d=networks.FermiNetData(positions=x,spins=spins,atoms=jnp.zeros((1,2)),charges=jnp.zeros(1))
            for method in ('default','folx'):
                for batch in (0,1):
                    local=LJ.local_energy(f,d.charges,(2,0),lattice=jnp.eye(2)*4,lam=lam,delta=delta,
                        potential_type='none',flip_batch=batch,kinetic_kwargs={'laplacian_method':method})
                    energy,aux=local(None,None,d)
                    np.testing.assert_allclose(energy,expected,atol=1e-10,rtol=1e-10)
                    np.testing.assert_array_equal(aux,[0.,1.])

    def test_ewald_independent_background_self_and_length_scaling(self):
        L=np.array([[3.2,-1.6],[0.,2.7712812921102037]]);rs=2.;cut=5
        A=abs(np.linalg.det(L));eta=2.8/np.sqrt(A)
        indices=list(itertools.product(range(-cut,cut+1),repeat=2));nz=[v for v in indices if v!=(0,0)]
        real=[L@v for v in indices];rnz=[L@v for v in nz];rec=[2*np.pi*np.linalg.inv(L).T@v for v in nz]
        reciprocal=lambda r:sum(2*np.pi/A*math.erfc(np.linalg.norm(k)/(2*eta))/np.linalg.norm(k)*np.cos(k@r) for k in rec)
        M=sum(math.erfc(eta*np.linalg.norm(v))/np.linalg.norm(v) for v in rnz)+reciprocal(np.zeros(2))-2*eta/np.sqrt(np.pi)
        B=2*np.pi/A/eta/np.sqrt(np.pi)
        for n in (1,2,3):
            r=np.array([[.23,.41],[1.17,.89],[.65,1.4]])[:n];ee=r[None,:,:]-r[:,None,:]
            ref=n*M/2-n*n*B/2
            for i in range(n):
                for j in range(i+1,n):
                    u=np.linalg.solve(L,ee[i,j]);d=L@((u+.5)%1-.5)
                    ref+=sum(math.erfc(eta*np.linalg.norm(d-v))/np.linalg.norm(d-v) for v in real)+reciprocal(d)
            p=PH.make_2DCoulomb_potential(jnp.asarray(L),jnp.zeros(1),jnp.zeros(1),cut,rs)
            np.testing.assert_allclose(p(None,jnp.asarray(ee)),rs*ref,atol=1e-11,rtol=1e-11)
            scale=1.7;scaled=PH.make_2DCoulomb_potential(jnp.asarray(L*scale),jnp.zeros(1),jnp.zeros(1),cut,rs)
            np.testing.assert_allclose(scaled(None,jnp.asarray(ee*scale)),rs*ref/scale,atol=1e-11,rtol=1e-11)

    def test_moire_value_derivatives_and_local_energy(self):
        a=3.;U=.7;phi=.2;L=jnp.asarray([[a,a/2],[0.,np.sqrt(3)*a/2]])
        G=np.array([[0.,4*np.pi/(np.sqrt(3)*a)],[-2*np.pi/a,-2*np.pi/(np.sqrt(3)*a)],[2*np.pi/a,-2*np.pi/(np.sqrt(3)*a)]])
        x=jnp.asarray([.23,.41]);fn=PH.make_triangular_potential(L,jnp.zeros((1,2)),jnp.zeros(1),U,phi)
        f=lambda r:fn(r.reshape(1,1,2))
        phase=G@np.asarray(x)+phi
        np.testing.assert_allclose(f(x),-2*U*np.cos(phase).sum(),atol=1e-12)
        np.testing.assert_allclose(jax.grad(f)(x),2*U*(np.sin(phase)[:,None]*G).sum(0),atol=1e-12)
        np.testing.assert_allclose(jax.hessian(f)(x),2*U*np.einsum('i,ij,ik->jk',np.cos(phase),G,G),atol=1e-12)
        np.testing.assert_allclose(f(jnp.zeros(2)),-6*U*np.cos(phi),atol=1e-12)
        for i in range(2):np.testing.assert_allclose(f(x+L[:,i]),f(x),atol=1e-12)
        q=jnp.array([.7,-.2]);wave=lambda p,r,s,a,c:(q@r,jnp.asarray(0.))
        local=PH.local_energy(wave,jnp.zeros(1),(1,0),complex_output=True,lattice=L,potential_type='CoulombMoire',
            potential_kwargs=dict(interaction_energy_scale=0.,moire_lattice_vectors=L,moire_potential_strength=U,moire_potential_phi=phi),
            kinetic_kwargs={'laplacian_method':'default'},convergence_radius=2)
        d=networks.FermiNetData(positions=x,spins=jnp.ones(1),atoms=jnp.zeros((1,2)),charges=jnp.zeros(1))
        np.testing.assert_allclose(local(None,None,d)[0],float(q@q)/2+f(x),atol=1e-12)


class KnownInconsistency(unittest.TestCase):
    """既存の不整合の再現。PASSは当該経路の物理的正しさを意味しない。"""
    def test_current_periodic_isotropic_envelope_shape_incompatibility(self):
        init,apply,_,_=controlled_orbitals(lattice=jnp.eye(2)*3,envelope=envelopes.make_isotropic_envelope())
        p=init(jax.random.PRNGKey(11))
        with self.assertRaisesRegex((TypeError,ValueError),'broadcast|incompatible'):
            apply(p,jnp.asarray([.23,.41,1.17,.89,.65,1.4]),jnp.asarray([1.,1.,-1.]),jnp.zeros((1,2)),jnp.zeros(1))


if __name__=='__main__':unittest.main()
