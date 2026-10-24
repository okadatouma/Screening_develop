"""G2b: analytic periodic waves and actual corrected Hamiltonian paths.

Constant-output ERROR characterizations are not physical correctness.
The pre-G2b source and all investigation results are preserved separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import unittest
from tests.support import require_numerics
require_numerics()
import folx
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import hamiltonian as H, networks, loss, constants
from periodicwave.pbc import hamiltonian as PH, lambda_jellium as LJ
from tests.check_vmc_gradients import serial

RECORDS=[]
PARAMETERS={'amplitude':[.2,-.1,0.,0.,0.,0.],
            'phase':[0.,0.,.4,-.17,0.,0.],
            'mixed':[.2,-.1,.4,-.17,.08,.12]}


def wave(p,x,s,a,c):
    """a=sum(alpha cos x+beta cos y+u s), phi=sum(theta sin x+eta cos y+v s)."""
    r=x.reshape(-1,2)
    return (jnp.sum(p[2]*jnp.sin(r[:,0])+p[3]*jnp.cos(r[:,1]))+p[5]*s.sum(),
            jnp.sum(p[0]*jnp.cos(r[:,0])+p[1]*jnp.cos(r[:,1]))+p[4]*s.sum())


def data(n=1,spin=1.):
    r=np.array([[.31,-.27],[1.17,.89]])[:n]
    return networks.FermiNetData(positions=jnp.asarray(r.ravel()),
        spins=jnp.asarray([spin] if n==1 else [1.,-1.]),atoms=jnp.zeros((1,2)),charges=jnp.zeros(1))


def analytic(p,d):
    """Independent NumPy first derivatives and Hessian diagonal, [N,2]."""
    p=np.asarray(p);r=np.asarray(d.positions).reshape(-1,2)
    ga=np.stack((-p[0]*np.sin(r[:,0]),-p[1]*np.sin(r[:,1])),axis=-1)
    gp=np.stack((p[2]*np.cos(r[:,0]),-p[3]*np.sin(r[:,1])),axis=-1)
    ha=np.stack((-p[0]*np.cos(r[:,0]),-p[1]*np.cos(r[:,1])),axis=-1)
    hp=np.stack((-p[2]*np.sin(r[:,0]),-p[3]*np.cos(r[:,1])),axis=-1)
    lap=np.sum(ha+1j*hp+(ga+1j*gp)**2,axis=1)
    return ga,gp,ha,hp,lap


def close_parts(actual,expected,x64):
    # Analytic elementary operations: fixed 32 eps, real/imag checked separately.
    eps=np.finfo(np.float64 if x64 else np.float32).eps
    for a,b in ((np.real(actual),np.real(expected)),(np.imag(actual),np.imag(expected))):
        np.testing.assert_allclose(a,b,rtol=32*eps,atol=32*eps)


def ordinary(method,potential='Coulomb'):
    kw={'interaction_energy_scale':0.}
    if potential=='CoulombMoire':
        kw.update(moire_lattice_vectors=jnp.eye(2)*4,moire_potential_strength=.3,moire_potential_phi=.2)
    return PH.local_energy(wave,jnp.zeros(1),(1,0),complex_output=True,lattice=jnp.eye(2)*(2*np.pi),
        potential_type=potential,convergence_radius=2,potential_kwargs=kw,
        kinetic_kwargs={'laplacian_method':method})


class G2Checks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64',True)
        jax.clear_caches()

    def test_folx_and_ad_intermediates_against_analytic_derivatives(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for name,values in PARAMETERS.items():
                for n in (1,2):
                    p=jnp.asarray(values);d=data(n);ga,gp,ha,hp,lap=analytic(p,d)
                    fn=lambda x:wave(p,x,d.spins,d.atoms,d.charges)
                    for jit in (False,True):
                        wrapped=lambda x:folx.forward_laplacian(fn,sparsity_threshold=0)(x)
                        ph,la=(jax.jit(wrapped) if jit else wrapped)(d.positions)
                        close_parts(la.jacobian.dense_array,ga.ravel(),x64)
                        close_parts(ph.jacobian.dense_array,gp.ravel(),x64)
                        close_parts(la.laplacian,ha.sum(),x64);close_parts(ph.laplacian,hp.sum(),x64)
                        for k,g,h in ((0,gp,hp),(1,ga,ha)):
                            close_parts(jax.grad(lambda x:fn(x)[k])(d.positions),g.ravel(),x64)
                            close_parts(jnp.trace(jax.hessian(lambda x:fn(x)[k])(d.positions)),h.sum(),x64)
                        raw=-.5*(la.laplacian+1j*ph.laplacian+jnp.sum((la.jacobian.dense_array+1j*ph.jacobian.dense_array)**2))
                        close_parts(raw,-.5*lap.sum(),x64)
                        wrong=-.5*(ha.sum()+1j*hp.sum()+np.sum(np.abs(ga+1j*gp)**2))
                        RECORDS.append(serial(dict(test='analytic_intermediates',x64=x64,case=name,n=n,jit=jit,parameters=p,positions=d.positions,spins=d.spins,grad_a=ga,grad_phi=gp,lap_a=la.laplacian,lap_phi=ph.laplacian,raw_folx_kinetic=raw,expected=-.5*lap.sum(),incorrect_conjugated_square=wrong)))

    def test_actual_ordinary_routes_preserve_complex_energy(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for name,values in PARAMETERS.items():
                for n in (1,2):
                    p=jnp.asarray(values);d=data(n);expected=-.5*analytic(p,d)[4].sum()
                    for method in ('default','folx'):
                        f=ordinary(method)
                        root=H.local_kinetic_energy(wave,complex_output=True,laplacian_method=method)
                        for jit in (False,True):
                            call=lambda pp,dd:f(pp,None,dd)[0]
                            got=(jax.jit(call) if jit else call)(p,d)
                            kg=(jax.jit(root) if jit else root)(p,d)
                            close_parts(kg,expected,x64)
                            close_parts(got,expected,x64)
                            if name in ('phase','mixed'):
                                # Two-electron imaginary terms can partially cancel.
                                # Require a signal above roundoff, not an arbitrary .01.
                                self.assertGreater(abs(expected.imag),128*np.finfo(np.float64 if x64 else np.float32).eps)
                                self.assertGreater(abs(complex(got).imag),128*np.finfo(np.float64 if x64 else np.float32).eps)
                            RECORDS.append(serial(dict(test='PBC_complex_energy',x64=x64,case=name,n=n,method=method,jit=jit,actual=got,root_kinetic=kg,analytic=expected,error_to_physics=got-expected)))

    def test_moire_real_potential_preserves_complex_energy(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            p=jnp.asarray(PARAMETERS['mixed']);d=data();r=np.asarray(d.positions)
            b=4*np.pi/(np.sqrt(3)*4)
            G=np.array([[0,b],[b*np.sin(2*np.pi/3),b*np.cos(2*np.pi/3)]])
            G=np.vstack((G,-G.sum(axis=0)))
            v=-.6*np.cos(G@r+.2).sum();expected=-.5*analytic(p,d)[4].sum()+v
            for method in ('default','folx'):
                got=jax.jit(lambda pp,dd:ordinary(method,'CoulombMoire')(pp,None,dd)[0])(p,d)
                close_parts(got,expected,x64)
                RECORDS.append(serial(dict(test='moire',x64=x64,method=method,potential=v,actual=got,analytic=expected)))

    def test_lambda_weighted_laplacian_and_spin_flip_full_complex_energy(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for n,spin in ((1,1.),(1,-1.),(2,1.)):
                p=jnp.asarray(PARAMETERS['mixed']);d=data(n,spin);ga,gp,ha,hp,lap=analytic(p,d);s=np.asarray(d.spins)
                for lam in (0.,.4):
                    delta=2.;c=.5+delta*lam**2*(s>0)
                    ratio=np.exp(-2*s*(float(p[4])+1j*float(p[5])))
                    flip=ratio*((ga+1j*gp)[:,0]-1j*s*(ga+1j*gp)[:,1])
                    expected=-np.sum(c*lap)+delta*np.sum(s<0)+1j*delta*lam*flip.sum()
                    for method in ('default','folx'):
                        f=LJ.local_energy(wave,d.charges,(n,0),lattice=jnp.eye(2)*(2*np.pi),lam=lam,delta=delta,potential_type='none',kinetic_kwargs={'laplacian_method':method},flip_batch=1)
                        w=LJ.make_weighted_laplacian(wave,method)
                        for jit in (False,True):
                            call=lambda pp,dd:f(pp,None,dd)[0]
                            got=(jax.jit(call) if jit else call)(p,d)
                            weighted=(jax.jit(w) if jit else w)(p,d,jnp.asarray(c,dtype=p.dtype))
                            close_parts(weighted,np.sum(c*lap),x64);close_parts(got,expected,x64)
                            close_parts(LJ.make_flip_terms(wave,flip_batch=1)(p,d,*wave(p,d.positions,d.spins,d.atoms,d.charges)),flip,x64)
                            RECORDS.append(serial(dict(test='lambda_full_complex',x64=x64,n=n,spin=spin,lam=lam,delta=delta,method=method,jit=jit,c=c,weighted=weighted,expected_weighted=np.sum(c*lap),spin_flip=flip,actual=got,analytic=expected)))

    def test_literal_constant_outputs_are_a_separate_known_contract_error(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64);d=data();p=jnp.asarray(PARAMETERS['phase'])
            for which in ('logabs','phase'):
                def f(pp,x,s,a,c):
                    ph,la=wave(pp,x,s,a,c)
                    return (ph,jnp.zeros((),x.dtype)) if which=='logabs' else (jnp.zeros((),x.dtype),la)
                # Each constant has a well-defined zero gradient/laplacian.
                pp=p if which=='logabs' else jnp.asarray(PARAMETERS['amplitude'])
                expected=-.5*analytic(pp,d)[4].sum()
                out=folx.forward_laplacian(lambda x:f(pp,x,d.spins,d.atoms,d.charges),sparsity_threshold=0)(d.positions)
                idx=1 if which=='logabs' else 0
                self.assertFalse(hasattr(out[idx],'laplacian'))
                for route in ('root','pbc','lambda'):
                    for method in ('default','folx'):
                        if route=='root':fn=H.local_kinetic_energy(f,complex_output=True,laplacian_method=method)
                        elif route=='pbc':
                            el=PH.local_energy(f,d.charges,(1,0),complex_output=True,lattice=jnp.eye(2)*4,convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':method})
                            fn=lambda p,d:el(p,None,d)[0]
                        else:
                            el=LJ.local_energy(f,d.charges,(1,0),lattice=jnp.eye(2)*4,lam=0.,delta=2.,potential_type='none',kinetic_kwargs={'laplacian_method':method})
                            fn=lambda p,d:el(p,None,d)[0]
                        if method=='folx':
                            with self.assertRaises(AttributeError) as error:jax.jit(fn)(pp,d)
                            RECORDS.append(dict(test='known_constant_output_ERROR_not_physics_PASS',x64=x64,constant=which,route=route,method=method,error=str(error.exception),folx_output_type=type(out[idx]).__name__))
                        else:close_parts(jax.jit(fn)(pp,d),expected,x64)

    def test_G1_vmc_phase_gradient_with_actual_local_energy(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for route,method,lam in (('pbc','default',0.),('pbc','folx',0.),('lambda','default',0.),('lambda','folx',0.),('lambda','default',.4),('lambda','folx',.4)):
                p=jnp.asarray([0.,.4]);d=torus_batch();local=torus_local(route,method,lam)
                theta=float(p[1])  # FD and the closed form use the same represented input.
                expected=np.array([-2*lam*theta,(1+2*lam**2)*theta/2]) if route=='lambda' else np.array([0.,theta/2])
                physical=expected.copy()
                for repeat,chunk in ((1,0),(1,3),(2,0)):
                    batch=jax.tree_util.tree_map(lambda a:jnp.concatenate([a]*repeat,axis=0),d)
                    f=loss.make_loss(torus_logpsi,local,complex_output=True,local_energy_chunk=chunk,clip_local_energy=0.)
                    (value,aux),g=jax.jit(jax.value_and_grad(f,has_aux=True))(p,jax.random.PRNGKey(7),batch)
                    close_parts(g,expected,x64)
                    # Independent periodic Dirichlet energy, not mean(EL) FD.
                    ad=jax.grad(torus_energy)(p,lam if route=='lambda' else 0.,2. if route=='lambda' else 0.)
                    close_parts(ad,physical,x64)
                    h=1e-5;fd=np.array([(torus_energy_numpy(np.asarray(p,dtype=float)+h*e,lam if route=='lambda' else 0.,2. if route=='lambda' else 0.)-torus_energy_numpy(np.asarray(p,dtype=float)-h*e,lam if route=='lambda' else 0.,2. if route=='lambda' else 0.))/(2*h) for e in np.eye(2)])
                    np.testing.assert_allclose(fd,physical,atol=2e-9,rtol=2e-9)
                    RECORDS.append(serial(dict(test='G1_with_actual_Hamiltonian',x64=x64,route=route,method=method,lam=lam,repeat=repeat,chunk=chunk,actual_gradient=g,expected_implementation=expected,physical_gradient=physical,independent_AD=ad,finite_difference=fd,energy=value,local_energy=aux.local_energy)))

    def test_clipped_complex_scores_match_independent_estimator(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            p=jnp.asarray([0.,.4]);d=torus_batch();x=np.asarray(d.positions)[:,0].astype(float)
            theta=float(p[1]);physical_el=theta**2*np.cos(x)**2/2+1j*theta*np.sin(x)/2
            for method in ('default','folx'):
                el=physical_el
                for clip in (5.,.25):
                    # Independent existing mean-window, center=True clip policy.
                    clipped=[]
                    for component in (el.real,el.imag):
                        center=component.mean();width=clip*np.abs(component-center).mean()
                        clipped.append(np.clip(component,center-width,center+width))
                    residual=(clipped[0]+1j*clipped[1]);residual-=residual.mean()
                    expected=2*np.real(np.mean(residual[:,None]*np.stack((np.cos(x),-1j*np.sin(x)),axis=-1),axis=0))
                    f=loss.make_loss(torus_logpsi,torus_local('pbc',method,0.),complex_output=True,clip_local_energy=clip,clip_from_median=False,center_at_clipped_energy=True)
                    (_,aux),g=jax.jit(jax.value_and_grad(f,has_aux=True))(p,jax.random.PRNGKey(7),d)
                    close_parts(g,expected,x64)
                    RECORDS.append(serial(dict(test='G1_clipped_score',x64=x64,method=method,clip=clip,actually_clipped=not np.allclose(clipped[0]+1j*clipped[1],el,atol=1e-14,rtol=0),gradient=g,expected_score=expected,unclipped_physical_gradient=[0.,theta/2])))


def torus_wave(p,x,s,a,c):
    # Zero coefficients retain coordinate-dependent output representations;
    # literal constants are tested separately, never silently replaced in Production.
    return p[1]*jnp.sin(x[0]),p[0]*jnp.cos(x[0])


def torus_logpsi(p,x,s,a,c):
    ph,la=torus_wave(p,x,s,a,c);return la+1j*ph


def torus_batch():
    x=np.tile(np.arange(16)*2*np.pi/16+.13,2);n=len(x)
    return networks.FermiNetData(positions=jnp.asarray(np.stack((x,np.zeros(n)),axis=-1)),spins=jnp.asarray(np.repeat([1.,-1.],16)[:,None]),atoms=jnp.zeros((n,1,2)),charges=jnp.zeros((n,1)))


def torus_local(route,method,lam):
    opts=dict(complex_output=True,lattice=jnp.eye(2)*(2*np.pi),kinetic_kwargs={'laplacian_method':method})
    if route=='pbc':return PH.local_energy(torus_wave,jnp.zeros(1),(1,0),convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},**opts)
    return LJ.local_energy(torus_wave,jnp.zeros(1),(1,0),lam=lam,delta=2.,potential_type='none',**opts)


def torus_energy(p,lam,delta):
    # Integration by parts on the torus: <|grad psi|^2>/2, plus lambda terms.
    x=jnp.arange(128,dtype=p.dtype)*2*jnp.pi/128
    w=jnp.exp(2*p[0]*jnp.cos(x));w=w/w.sum()
    return (1+delta*lam**2)/2*jnp.sum(w*(p[0]**2*jnp.sin(x)**2+p[1]**2*jnp.cos(x)**2))+delta/2-delta*lam*p[1]*jnp.sum(w*jnp.cos(x))


def torus_energy_numpy(p,lam,delta):
    x=np.arange(256)*2*np.pi/256;w=np.exp(2*p[0]*np.cos(x));w/=w.sum()
    return (1+delta*lam**2)/2*np.sum(w*(p[0]**2*np.sin(x)**2+p[1]**2*np.cos(x)**2))+delta/2-delta*lam*p[1]*np.sum(w*np.cos(x))


class DeviceChecks(unittest.TestCase):
    @unittest.skipUnless(jax.local_device_count()==2,'separate two-CPU-device process required')
    def test_batch_device_local_energy_and_G1_gradient(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for route,method,lam in (('pbc','default',0.),('pbc','folx',0.),('lambda','folx',.4)):
                p=jnp.asarray([0.,.4]);d=torus_batch();d=jax.tree_util.tree_map(lambda a:a.reshape((2,16)+a.shape[1:]),d)
                f=loss.make_loss(torus_logpsi,torus_local(route,method,lam),complex_output=True,local_energy_chunk=3)
                def step(q,k,dd):
                    (v,a),g=jax.value_and_grad(f,has_aux=True)(q,k,dd)
                    return v,a.local_energy,constants.pmean(g)
                value,el,g=constants.pmap(step)(jnp.stack([p,p]),jax.random.split(jax.random.PRNGKey(7),2),d)
                theta=float(p[1])
                expected=np.array([-2*lam*theta,(1+2*lam**2)*theta/2]) if route=='lambda' else np.array([0.,theta/2])
                close_parts(g[0],expected,x64);close_parts(g[1],expected,x64)
                RECORDS.append(serial(dict(test='isolated_two_devices',x64=x64,route=route,method=method,lam=lam,local_energy=el,gradient=g,expected=expected,energy=value)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);parser.add_argument('--device-probe',action='store_true');args=parser.parse_args()
    if args.report.exists():parser.error('existing report is protected')
    cls=DeviceChecks if args.device_probe else G2Checks
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    from tests.baseline import metadata
    import inspect
    folx_source=Path(inspect.getfile(folx.forward_laplacian))
    report=dict(version='g2b-v1',environment=metadata(),devices=[str(d) for d in jax.devices()],xla_flags=os.environ.get('XLA_FLAGS'),folx_source=str(folx_source),folx_source_sha256=hashlib.sha256(folx_source.read_bytes()).hexdigest(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),records=RECORDS,note='Complex PBC energies are checked against analytic physics; constant-output errors remain characterized separately.')
    with args.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
