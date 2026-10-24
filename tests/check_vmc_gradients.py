"""Independent two-state VMC investigation; Production is never patched.

p=(alpha, theta, gamma), psi_x=exp(alpha*x+i*(theta*x+gamma)), x=(-1,+1).
Exact physics (normalized full sum) and current-rule characterizations are
separate classes. CurrentBehavior explicitly uses the archived pre-G1 loss.
A characterization PASS does not mean the VMC rule is correct.
Run --device-probe in a separate two-CPU-device process (see tests/README.md).
"""
import argparse
import json
import os
from pathlib import Path
import unittest

from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import loss, networks, constants

X=np.array([-1.,1.])
H_DIAG=np.diag([-1.,1.]).astype(complex)
H_PHASE=np.array([[0,1j],[-1j,0]])
H_MIXED=np.array([[.4,.7+.6j],[.7-.6j,-.2]])
RECORDS=[]
CASES=(('amplitude',H_DIAG,[0.,0.,0.],[1,1]),
       ('phase_zero',H_PHASE,[0.,0.,0.],[1,1]),
       ('phase_nonzero',H_PHASE,[0.,.31,.2],[1,1]),
       ('mixed_p20_80',H_MIXED,[np.log(2)/2,.31,.2],[1,4]),
       ('mixed_p80_20',H_MIXED,[-np.log(2)/2,-.27,-.1],[4,1]))


def full_sum(p,H):
    psi=np.exp(p[0]*X+1j*(p[1]*X+p[2]))
    prob=np.abs(psi)**2;prob/=prob.sum()
    el=(H@psi)/psi
    O=np.stack((X,1j*X,1j*np.ones(2)),axis=1)
    energy=np.vdot(psi,H@psi)/np.vdot(psi,psi)
    gradient=2*np.real(np.sum(prob[:,None]*(el-energy)[:,None]*O.conj(),axis=0))
    return energy,prob,el,O,gradient


def exact_jax(p,H):
    x=jnp.asarray(X,dtype=p.dtype)
    psi=jnp.exp(p[0]*x+1j*(p[1]*x+p[2]))
    return (jnp.vdot(psi,H@psi)/jnp.vdot(psi,psi)).real


def data_for(states):
    x=jnp.asarray(X)[np.asarray(states)]
    n=len(states)
    return networks.FermiNetData(positions=jnp.stack((x,jnp.zeros_like(x)),axis=1),
        spins=jnp.ones((n,1)),atoms=jnp.zeros((n,1,2)),charges=jnp.zeros((n,1)))


def make_current(H,complex_output=True,clip=0.,median=False,center=True,chunk=0,real_el=False):
    H=jnp.asarray(H,dtype=jnp.complex128 if jax.config.x64_enabled else jnp.complex64)
    def network(p,x,s,a,c):
        value=p[0]*x[0]
        return value+1j*(p[1]*x[0]+p[2]) if complex_output else value
    def local(p,key,d):
        x=jnp.asarray(X,dtype=p.dtype)
        z=jnp.exp(p[0]*x+1j*(p[1]*x+p[2]))
        el=(H@z)/z
        value=el[(d.positions[0]>0).astype(jnp.int32)]
        return (value.real if real_el or not complex_output else value),None
    return loss.make_loss(network,local,clip_local_energy=clip,clip_from_median=median,
        center_at_clipped_energy=center,complex_output=complex_output,local_energy_chunk=chunk)


def evaluate(f,p,d,jit=False):
    def objective(pp):
        value,aux=f(pp,jax.random.PRNGKey(0),d)
        return value.real,aux
    return (jax.jit if jit else lambda g:g)(jax.value_and_grad(objective,has_aux=True))(p)


def current_algebra(el,O,complex_output=True,clip=0.,median=False,center=True):
    """Pre-G1 NumPy rule and the independent centered-score reference."""
    mean=el.mean();clipped=el.copy()
    if clip>0:
        c=np.median(el.real) if median else mean
        def component(v,c):
            width=clip*np.mean(np.abs(v-c))
            return np.clip(v,c-width,c+width)
        clipped=component(el.real,np.real(c))
        if complex_output:clipped=clipped+1j*component(el.imag,np.imag(c))
        field=clipped.mean() if center else mean  # scalar returned by clip_local_values
        diff=clipped-field
    else:
        field=el                         # vector kept in AuxiliaryLossData
        diff=el-mean
    centered=2*np.real(np.mean(diff[:,None]*O.conj(),axis=0))
    if complex_output:
        effective=diff+field
        term1=2*np.real(np.sum(effective[:,None]*O.conj(),axis=0))
        term2=2*np.real(np.sum(np.asarray(field)[...,None]*O.real,axis=0))
        gradient=(term1-term2)/len(el)
    else:
        term1=np.sum(O.real*diff.real[:,None],axis=0);term2=np.zeros(3)
        gradient=term1/len(el)
    return gradient,centered,dict(mean=mean,field=field,diff=diff,clipped=clipped,
        term1=term1,twice_term2=term2,field_shape=list(np.shape(field)))


def serial(value):
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    a=np.asarray(value)
    return dict(real=a.real.tolist(),imag=a.imag.tolist()) if np.iscomplexobj(a) else a.tolist()


def close(a,b,x64):
    # Two-state arithmetic, 32 machine eps; original tests/tolerances untouched.
    eps=np.finfo(np.float64 if x64 else np.float32).eps
    np.testing.assert_allclose(a,b,rtol=32*eps,atol=32*eps)


def sample(case):
    name,H,p,counts=case
    return name,H,jnp.asarray(p),np.repeat(np.arange(2),counts)


class Base(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)


class ExactPhysics(Base):
    def test_normalized_sum_ad_finite_difference_covariance_and_conjugate(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for case in CASES:
                name,H,p,indices=sample(case);energy,prob,el,O,g=full_sum(np.asarray(p),H)
                ad=jax.grad(exact_jax)(p,jnp.asarray(H,dtype=jnp.complex128 if x64 else jnp.complex64))
                eps=1e-5;fd=[]
                for k in range(3):
                    shift=np.eye(3)[k]*eps
                    fd.append((full_sum(np.asarray(p,dtype=float)+shift,H)[0].real-full_sum(np.asarray(p,dtype=float)-shift,H)[0].real)/(2*eps))
                np.testing.assert_allclose(fd,g,atol=2e-9,rtol=2e-9)
                close(ad,g,x64)
                # Rational base probabilities are represented by exact state counts.
                close(np.bincount(indices,minlength=2)/len(indices),prob,x64)
                self.assertLess(abs(energy.imag),1e-14)
                wrong_conjugate=2*np.real(np.sum(prob[:,None]*(el-energy)[:,None]*O,axis=0))
                RECORDS.append(serial(dict(category='independent_physics',x64=x64,case=name,p=p,H=H,
                    energy=energy,probabilities=prob,local_energy=el,O=O,exact_gradient=g,ad=ad,
                    finite_difference=fd,wrong_conjugate=wrong_conjugate)))
        np.testing.assert_array_equal(full_sum(np.zeros(3),H_PHASE)[4],[0.,-2.,0.])


    def test_test_only_centered_surrogate_derivative(self):
        # Proposed score contraction only, in tests. No Production loss is patched.
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for case in CASES:
                name,H,p,indices=sample(case)
                E,prob,el,O,correct=full_sum(np.asarray(p),H)
                weights=jnp.asarray(prob);residual=jnp.asarray(el-E)
                def frozen_score(q):
                    x=jnp.asarray(X,dtype=q.dtype)
                    logpsi=q[0]*x+1j*(q[1]*x+q[2])
                    return 2*jnp.real(jnp.sum(weights*jax.lax.stop_gradient(residual)*jnp.conj(logpsi)))
                got=jax.jit(jax.grad(frozen_score))(p)
                close(got,correct,x64)
                RECORDS.append(serial(dict(category='test_only_proposed_centered_contraction',x64=x64,
                    case=name,gradient=got,exact=correct)))


class CurrentBehavior(Base):
    def setUp(self):
        # Pin historical assertions to the preserved implementation, not G1.
        # Only this test module's reference changes; periodicwave.loss is untouched.
        import sys
        import types
        global loss
        self.production_loss = loss
        name = 'tests._pre_g1_loss'
        if name not in sys.modules:
            module = types.ModuleType(name)
            sys.modules[name] = module
            source = Path(__file__).parent / 'g1_validation/loss.py-before.txt'
            exec(compile(source.read_text(), str(source), 'exec'), module.__dict__)
        loss = sys.modules[name]

    def tearDown(self):
        global loss
        loss = self.production_loss
        super().tearDown()

    def test_complex_jvp_first_mismatch(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for case in CASES:
                name,H,p,indices=sample(case);energy,prob,el,O,correct=full_sum(np.asarray(p),H)
                f=make_current(H);d=data_for(indices);(_,aux),actual=evaluate(f,p,d)
                predicted,centered,trace=current_algebra(el[indices],O[indices])
                close(actual,predicted,x64);close(aux.local_energy,el[indices],x64)
                # At the exact distribution, amplitude is right, phase gets an extra term.
                close(np.asarray(actual)[0],correct[0],x64)
                close(np.asarray(actual)-correct,2*np.real(np.mean((1j*el[indices].imag)[:,None]*O[indices].conj(),axis=0)),x64)
                if name=='phase_zero':np.testing.assert_array_equal(actual,[0.,-4.,0.])
                if name=='amplitude':np.testing.assert_array_equal(actual,[2.,0.,0.])
                RECORDS.append(serial(dict(category='current_bug_reproduction',x64=x64,case=name,
                    gradient=actual,correct=correct,centered_reference=centered,trace=trace,
                    physics_max_abs_error=np.max(np.abs(np.asarray(actual)-correct)))))

    def test_real_amplitude_missing_two(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for alpha,counts in ((0.,[1,1]),(np.log(2)/2,[1,4]),(-np.log(2)/2,[4,1])):
                p=jnp.asarray([alpha,0.,0.]);indices=np.repeat(np.arange(2),counts)
                _,_,el,O,g=full_sum(np.asarray(p),H_DIAG);O=O.real
                f=make_current(H_DIAG,complex_output=False);(_,aux),actual=evaluate(f,p,data_for(indices))
                predicted,centered,trace=current_algebra(el[indices].real,O[indices],False)
                close(actual,predicted,x64);close(2*np.asarray(actual),g,x64)
                RECORDS.append(serial(dict(category='current_real_bug_reproduction',x64=x64,p=p,
                    gradient=actual,correct=g,trace=trace)))

    def test_batch_replication_chunk_jit_and_mixed_direction(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            name,H,p,indices=sample(CASES[3]);_,_,el,O,_=full_sum(np.asarray(p),H)
            predicted,_,_=current_algebra(el[indices],O[indices]);direction=jnp.asarray([.3,-.7,.2])
            for repeat in (1,2,4):
                for chunk in (0,1,2):
                    f=make_current(H,chunk=chunk);d=data_for(np.tile(indices,repeat))
                    (_,aux),actual=evaluate(f,p,d,jit=True);close(actual,predicted,x64)
                    _,tangent=jax.jvp(lambda pp:f(pp,jax.random.PRNGKey(0),d)[0].real,(p,),(direction,))
                    close(tangent,np.dot(predicted,np.asarray(direction)),x64)
                    RECORDS.append(serial(dict(category='batch_chunk_jit',x64=x64,repeat=repeat,chunk=chunk,
                        gradient=actual,directional_jvp=tangent,direction=direction)))

    def test_clipping_field_switch_and_real_energy_ablation(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            _,H,p,indices=sample(CASES[3]);_,_,el,O,exact=full_sum(np.asarray(p),H);d=data_for(indices)
            for clip in (0.,.25,1.,100.):
                for median,center in ((False,True),(False,False),(True,True),(True,False)):
                    f=make_current(H,clip=clip,median=median,center=center)
                    (value,aux),actual=evaluate(f,p,d)
                    predicted,centered,trace=current_algebra(el[indices],O[indices],True,clip,median,center)
                    close(actual,predicted,x64);close(value,el[indices].mean().real,x64)
                    self.assertEqual(np.shape(aux.clipped_energy),tuple(trace['field_shape']))
                    RECORDS.append(serial(dict(category='clipping_characterization_not_exact_energy_gradient',
                        x64=x64,clip=clip,median=median,center=center,gradient=actual,
                        exact_unclipped_gradient=exact,centered_clipped_reference=centered,trace=trace)))
            # Same raw energies, no actual outlier clipped, different code branch.
            p=jnp.zeros(3);d=data_for([0,1])
            off=evaluate(make_current(H_PHASE),p,d)[1]
            on=evaluate(make_current(H_PHASE,clip=100.),p,d)[1]
            real_only=evaluate(make_current(H_PHASE,real_el=True),p,d)[1]
            np.testing.assert_array_equal(off,[0.,-4.,0.]);np.testing.assert_array_equal(on,[0.,-2.,0.])
            np.testing.assert_array_equal(real_only,[0.,0.,0.])
            RECORDS.append(serial(dict(category='clipping_branch_and_imaginary_energy_ablation',x64=x64,
                off=off,on_no_values_clipped=on,imaginary_energy_discarded=real_only)))

    def test_finite_batch_global_phase_and_centering(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            p=jnp.zeros(3);idx=np.array([0,0,0,1]);d=data_for(idx)
            _,_,el,O,exact=full_sum(np.zeros(3),H_PHASE)
            for clip in (0.,100.):
                (_,aux),actual=evaluate(make_current(H_PHASE,clip=clip),p,d)
                predicted,centered,trace=current_algebra(el[idx],O[idx],clip=clip)
                close(actual,predicted,x64);close(centered[2],0.,x64)
                RECORDS.append(serial(dict(category='finite_batch_not_exact_distribution',x64=x64,
                    clip=clip,gradient=actual,centered_empirical_reference=centered,
                    full_state_exact=exact,trace=trace)))

    def test_auxiliary_jvp_zero_and_linearity_contract(self):
        p=jnp.zeros(3);d=data_for([0,1]);f=make_current(H_PHASE)
        def all_outputs(pp):return f(pp,jax.random.PRNGKey(0),d)
        primal0,t0=jax.jvp(all_outputs,(p,),(jnp.zeros_like(p),))
        primal1,t1=jax.jvp(all_outputs,(p,),(jnp.asarray([0.,1.,0.]),))
        primal2,t2=jax.jvp(all_outputs,(p,),(jnp.asarray([0.,2.,0.]),))
        np.testing.assert_array_equal(t0[0],0.)
        close(t2[0],2*t1[0],True)
        self.assertNotEqual(float(t0[1].variance),0.)
        np.testing.assert_array_equal(t1[1].variance,t2[1].variance)
        raw=all_outputs(p)[0]
        RECORDS.append(serial(dict(category='auxiliary_jvp_contract_separate_from_main_gradient',
            ordinary_primal_dtype=str(raw.dtype),jvp_primal_dtype=str(primal0[0].dtype),
            zero_direction_main=t0[0],zero_direction_aux_variance=t0[1].variance,
            one_direction_aux_variance=t1[1].variance,two_direction_aux_variance=t2[1].variance)))


class DeviceAggregation(Base):
    @unittest.skipUnless(jax.local_device_count()>=2,'separate two-CPU-device process required')
    def test_pmap_local_scores_then_trainer_pmean(self):
        self.assertEqual(jax.local_device_count(),2)
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for complex_output,H in ((True,H_PHASE),(False,H_DIAG)):
                for indices in ([0,1,0,1],[0,0,0,1]):
                    p=jnp.zeros(3);batch=data_for(indices)
                    sharded=jax.tree_util.tree_map(lambda a:a.reshape((2,2)+a.shape[1:]),batch)
                    for clip in (0.,.25,100.):
                        f=make_current(H,complex_output=complex_output,clip=clip)
                        def step(pp,kk,dd):
                            def objective(q):
                                v,a=f(q,kk,dd);return v.real,a
                            (v,a),g=jax.value_and_grad(objective,has_aux=True)(pp)
                            return v,g,constants.pmean(g)
                        result=constants.pmap(step)(jnp.stack([p,p]),jax.random.split(jax.random.PRNGKey(0),2),sharded)
                        standalone=evaluate(f,p,batch)[1]
                        close(result[2][0],standalone,x64);close(result[2][1],standalone,x64)
                        RECORDS.append(serial(dict(category='isolated_two_cpu_devices',x64=x64,
                            complex_output=complex_output,indices=indices,clip=clip,
                            local_gradient=result[1],pmean_gradient=result[2],standalone=standalone,
                            global_reported_energy=result[0])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device-probe',action='store_true')
    args=parser.parse_args()
    classes=(DeviceAggregation,) if args.device_probe else (ExactPhysics,CurrentBehavior)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    from tests.baseline import metadata
    report=dict(environment=metadata(),runtime=dict(devices=[str(d) for d in jax.devices()],
        backend=jax.default_backend(),xla_flags=os.environ.get('XLA_FLAGS'),x64=bool(jax.config.x64_enabled)),
        methods_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),records=RECORDS,
        note='Characterization PASS reproduces current bugs, not physical correctness. No Production patch.')
    with args.report.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
