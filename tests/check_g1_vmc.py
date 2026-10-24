"""G1 physical oracle: normalized full sums, AD/FD and centered score covariance.

No old snapshot or research input is rewritten. Clipped scores are checked against
an independent NumPy estimator, not declared exact energy derivatives.
"""
import argparse
import json
import os
from pathlib import Path
import unittest
from tests import check_vmc_gradients as ref
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
import optax
from periodicwave import constants, train, train_screen

RECORDS=[]
PHYSICAL_CASES=ref.CASES+(
    ('amplitude_p20_80',ref.H_DIAG,[np.log(2)/2,0.,0.],[1,4]),
    ('amplitude_p80_20',ref.H_DIAG,[-np.log(2)/2,0.,0.],[4,1]))

class CorrectedGradient(ref.Base):
    def test_full_sum_covariance_ad_fd_and_production_jit(self):
        # Rational exact probabilities are represented by counts, so the VMC
        # batch estimator here equals the complete normalized-state derivative.
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for case in PHYSICAL_CASES:
                name,H,p,idx=ref.sample(case)
                energy,prob,el,O,expected=ref.full_sum(np.asarray(p),H)
                ad=jax.grad(ref.exact_jax)(p,jnp.asarray(H))
                step=1e-5
                fd=np.array([(ref.full_sum(np.asarray(p,dtype=float)+e*step,H)[0].real-ref.full_sum(np.asarray(p,dtype=float)-e*step,H)[0].real)/(2*step) for e in np.eye(3)])
                np.testing.assert_allclose(fd,expected,atol=2e-9,rtol=2e-9)
                for complex_output in ((False,True) if name.startswith('amplitude') else (True,)):
                    for jit in (False,True):
                        (value,aux),g=ref.evaluate(ref.make_current(H,complex_output=complex_output),p,ref.data_for(idx),jit)
                        ref.close(g,expected,x64);ref.close(g,ad,x64)
                        ref.close(value,energy.real,x64);ref.close(aux.local_energy,el[idx],x64)
                        RECORDS.append(ref.serial(dict(test='exact_energy_gradient',x64=x64,case=name,complex_output=complex_output,jit=jit,parameters=p,H=H,indices=idx,probabilities=prob,energy=energy,local_energy=el,expected=expected,ad=ad,finite_difference=fd,actual=g)))
        # E=tanh(2 alpha) and E=-sin(2 theta), both at zero.
        np.testing.assert_array_equal(ref.full_sum(np.zeros(3),ref.H_DIAG)[4],[2,0,0])
        np.testing.assert_array_equal(ref.full_sum(np.zeros(3),ref.H_PHASE)[4],[0,-2,0])

    def test_clipping_off_inactive_active_and_centering(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for case in (ref.CASES[0],ref.CASES[1],ref.CASES[3]):
                name,H,p,idx=ref.sample(case)
                for complex_output in ((False,True) if name=='amplitude' else (True,)):
                    _,_,el,O,exact=ref.full_sum(np.asarray(p),H)
                    if not complex_output:el,O=el.real,O.real
                    for clip in (0.,.25,1.,100.):
                        for median,center in ((False,True),(False,False),(True,True),(True,False)):
                            old,expected,trace=ref.current_algebra(el[idx],O[idx],complex_output,clip,median,center)
                            f=ref.make_current(H,complex_output,clip,median,center)
                            data=ref.data_for(idx)
                            (value,aux),g=ref.evaluate(f,p,data,jit=True)
                            ref.close(g,expected,x64);ref.close(value,el[idx].mean().real,x64)
                            direction=jnp.asarray([.3,-.7,.2])
                            tangent=lambda v:jax.jvp(lambda q:f(q,jax.random.PRNGKey(0),data)[0].real,(p,),(v,))[1]
                            ref.close(tangent(direction),np.dot(expected,np.asarray(direction)),x64)
                            ref.close(tangent(2*direction),2*tangent(direction),x64)
                            np.testing.assert_array_equal(tangent(jnp.zeros_like(p)),0.)
                            self.assertEqual(np.shape(aux.clipped_energy),tuple(trace['field_shape']))
                            active=not np.allclose(trace['clipped'],el[idx],rtol=0,atol=1e-14)
                            if clip==0 or not active:ref.close(g,exact,x64)
                            RECORDS.append(ref.serial(dict(test='clipped_score_not_exact_gradient_when_active',x64=x64,case=name,complex_output=complex_output,clip=clip,median=median,center=center,active=active,expected=expected,exact_unclipped=exact,actual=g,old_rule=old,trace=trace)))
            # An unbalanced empirical batch is not the exact distribution.
            p=jnp.zeros(3);idx=np.array([0,0,0,1]);_,_,el,O,_=ref.full_sum(np.zeros(3),ref.H_PHASE)
            for clip in (0.,100.):
                g=ref.evaluate(ref.make_current(ref.H_PHASE,clip=clip),p,ref.data_for(idx))[1]
                np.testing.assert_array_equal(g,[0.,-1.5,0.])
                RECORDS.append(ref.serial(dict(test='finite_batch_centered_not_full_sum',x64=x64,clip=clip,actual=g,expected=[0.,-1.5,0.])))

    def test_batch_chunk_and_main_jvp_linear_directional_derivative(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            _,H,p,idx=ref.sample(ref.CASES[3]);expected=ref.full_sum(np.asarray(p),H)[4]
            v=jnp.asarray([.3,-.7,.2]);w=jnp.asarray([-.4,.1,.8])
            for repeat in (1,2,4):
                for chunk in (0,1,2):
                    f=ref.make_current(H,chunk=chunk);d=ref.data_for(np.tile(idx,repeat));key=jax.random.PRNGKey(0)
                    value=lambda q:f(q,key,d)[0].real
                    g=ref.evaluate(f,p,d,jit=True)[1];ref.close(g,expected,x64)
                    tangent=lambda direction:jax.jvp(value,(p,),(direction,))[1]
                    a,b,ab,twice,zero=[tangent(t) for t in (v,w,v+w,2*v,jnp.zeros_like(p))]
                    ref.close(ab,a+b,x64);ref.close(twice,2*a,x64);np.testing.assert_array_equal(zero,0.)
                    ref.close(a,np.dot(expected,np.asarray(v)),x64)
                    h=1e-5;fd=(ref.full_sum(np.asarray(p,dtype=float)+h*np.asarray(v),H)[0].real-ref.full_sum(np.asarray(p,dtype=float)-h*np.asarray(v),H)[0].real)/(2*h)
                    # FD has O(h^2) truncation/cancellation, unlike the AD oracle.
                    # Use the same FD bound as ExactPhysics, not machine epsilon.
                    np.testing.assert_allclose(a,fd,rtol=2e-9 if x64 else 32*np.finfo(np.float32).eps,
                                               atol=2e-9 if x64 else 32*np.finfo(np.float32).eps)
                    RECORDS.append(ref.serial(dict(test='replication_chunk_jit_main_jvp',x64=x64,repeat=repeat,chunk=chunk,actual=g,expected=expected,jvp=a,finite_difference=fd,zero=zero,additivity_error=ab-a-b)))

    def test_auxiliary_contract_is_separate_and_still_known_invalid(self):
        # Preserve K10. This is not a PASS for the complete auxiliary JVP contract.
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            p=jnp.zeros(3);d=ref.data_for([0,1]);f=ref.make_current(ref.H_PHASE);key=jax.random.PRNGKey(0)
            fun=lambda q:f(q,key,d)
            primal,t0=jax.jvp(fun,(p,),(jnp.zeros_like(p),))
            _,t1=jax.jvp(fun,(p,),(jnp.asarray([0.,1.,0.]),))
            _,t2=jax.jvp(fun,(p,),(jnp.asarray([0.,2.,0.]),))
            np.testing.assert_array_equal([t0[0],t1[0],t2[0]],[0.,-2.,-4.])
            np.testing.assert_array_equal(t0[1].variance,1.)
            np.testing.assert_array_equal(t1[1].variance,t2[1].variance)
            g_aux=ref.evaluate(f,p,d)[1];g_only=jax.grad(lambda q:fun(q)[0].real)(p)
            np.testing.assert_array_equal(g_aux,g_only)
            RECORDS.append(ref.serial(dict(test='K10_unchanged_auxiliary_JVP_violation',x64=x64,zero_aux_variance=t0[1].variance,one_aux_variance=t1[1].variance,two_aux_variance=t2[1].variance,main_gradient=g_only,raw_dtype=str(fun(p)[0].dtype),jvp_primal_dtype=str(primal[0].dtype))))

    def test_adam_trainers_against_independent_physical_gradient(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for case in (ref.CASES[0],ref.CASES[1],ref.CASES[3]):
                name,H,p,idx=ref.sample(case);d=ref.data_for(idx)
                exact=jnp.asarray(ref.full_sum(np.asarray(p),H)[4],dtype=p.dtype)
                for complex_output in ((False,True) if name=='amplitude' else (True,)):
                    f=ref.make_current(H,complex_output);opt=optax.adam(.01);state=opt.init(p)
                    measured=ref.evaluate(f,p,d)[1]
                    ref.close(measured,exact,x64)
                    # Adam can amplify roundoff in a mathematically zero score
                    # (the global phase). Test its wiring with the measured score,
                    # recording the independently known exact score separately.
                    updates,expected_state=opt.update(measured,state,p);expected=optax.apply_updates(p,updates)
                    ideal_updates,_=opt.update(exact,state,p)
                    for trainer in (train,train_screen):
                        actual,s,value,aux=trainer.make_opt_update_step(f,opt)(p,d,state,jax.random.PRNGKey(2))
                        ref.close(actual,expected,x64)
                        for a,b in zip(jax.tree_util.tree_leaves(s),jax.tree_util.tree_leaves(expected_state)):ref.close(a,b,x64)
                        RECORDS.append(ref.serial(dict(test='adam_physical_gradient',x64=x64,case=name,complex_output=complex_output,trainer=trainer.__name__,exact_gradient=exact,measured_gradient=measured,ideal_parameters=optax.apply_updates(p,ideal_updates),expected=expected,actual=actual)))

class DeviceGradient(ref.Base):
    @unittest.skipUnless(jax.local_device_count()==2,'isolated two-CPU-device process required')
    def test_pmap_global_center_and_trainer_average_match_covariance(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for complex_output,H in ((False,ref.H_DIAG),(True,ref.H_PHASE)):
                p=jnp.zeros(3)
                for idx in ([0,1,0,1],[0,0,0,1]):
                    batch=ref.data_for(idx);shards=jax.tree_util.tree_map(lambda a:a.reshape((2,2)+a.shape[1:]),batch)
                    _,_,el,O,_=ref.full_sum(np.zeros(3),H)
                    if not complex_output:el,O=el.real,O.real
                    for clip in (0.,.25,100.):
                        _,expected,trace=ref.current_algebra(el[idx],O[idx],complex_output,clip)
                        f=ref.make_current(H,complex_output,clip=clip)
                        def step(q,k,d):
                            g=jax.grad(lambda pp:f(pp,k,d)[0].real)(q)
                            return g,constants.pmean(g)
                        local,g=constants.pmap(step)(jnp.stack([p,p]),jax.random.split(jax.random.PRNGKey(0),2),shards)
                        ref.close(g[0],expected,x64);ref.close(g[1],expected,x64)
                        ref.close(g[0],ref.evaluate(f,p,batch)[1],x64)
                        RECORDS.append(ref.serial(dict(test='two_device_global_mean',x64=x64,complex_output=complex_output,indices=idx,clip=clip,local=local,actual=g,expected=expected)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device-probe',action='store_true')
    args=parser.parse_args()
    if args.report.exists():parser.error('existing report is protected')
    classes=(DeviceGradient,) if args.device_probe else (CorrectedGradient,ref.ExactPhysics)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes))
    from tests.baseline import metadata
    report=dict(version='g1-v1',environment=metadata(),devices=[str(d) for d in jax.devices()],xla_flags=os.environ.get('XLA_FLAGS'),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),records=RECORDS+ref.RECORDS)
    with args.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
