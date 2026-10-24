"""G3a residual studies only. Production and old validation assets are read-only.

sample: fixed two-electron determinants, independent seeds and nested run lengths.
probes: localize precision/graph differences, without replacing Production formulas.
reference: independent square-cell Ewald calibration for the proposed G3b model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import folx
from jax.flatten_util import ravel_pytree
from periodicwave import checkpoint, loss, mcmc, networks
from periodicwave.pbc import hamiltonian as PH
from tests.check_g3a_vmc import (L, Q, build, blocking, exact_slater_energy,
                                orbital_coefficients, serial)


def restored(directory, batch):
    paths=sorted((directory/'run').glob('qmcjax_ckpt_*.npz'))
    _,_,p,_,_=checkpoint.restore(str(paths[-1]),batch)
    return jax.tree_util.tree_map(lambda a:jnp.asarray(a[0]),p)


def sample(args):
    meta=json.loads((args.case/'summary.json').read_text())
    assert meta['route']=='slater' and meta['particles']==2 and meta['x64']
    complex_output=meta['complex_output'];net,_=build('slater',2,complex_output)
    p=restored(args.case,meta['batch']);c=orbital_coefficients(p,complex_output)
    reference=float(exact_slater_energy(p,complex_output))
    logabs=lambda pp,x,s,a,c:net.apply(pp,x,s,a,c)[1]
    batch=32
    step=mcmc.make_mcmc_step(jax.vmap(logabs,in_axes=(None,0,0,0,0)),batch,
                             steps=1,ndim=2,blocks=1)
    def chain(d,key):
        def one(carry,_):
            d,k=carry;k,sub=jax.random.split(k);d,ac=step(p,d,sub,1.)
            return (d,k),(d.positions,ac)
        return jax.lax.scan(one,(d,key),None,length=args.sweeps+128)
    chain=jax.jit(chain)
    rows=[]
    for seed in (2303,2304,2305,2306):
        key,k=jax.random.split(jax.random.PRNGKey(seed))
        d=networks.FermiNetData(positions=jax.random.uniform(k,(batch,4),minval=0,maxval=L),
             spins=jnp.ones((batch,2)),atoms=jnp.zeros((batch,1,2)),charges=jnp.zeros((batch,1)))
        (final,key),(pos,accept)=chain(d,key)
        pos=np.asarray(pos[128:]);accept=np.asarray(accept[128:]);r=pos.reshape(-1,2,2)
        f=np.stack((np.sin(Q*r[...,0]),np.sin(Q*r[...,1]),
                    np.cos(Q*r[...,0]),np.cos(Q*r[...,1]),np.ones(r.shape[:-1])),axis=-1)
        mat=f@c;lap=-Q*Q*(mat-c[4])
        det=mat[:,0,0]*mat[:,1,1]-mat[:,0,1]*mat[:,1,0]
        el=-.5*(lap[:,0,0]*mat[:,1,1]+mat[:,0,0]*lap[:,1,1]
                 -lap[:,0,1]*mat[:,1,0]-mat[:,0,1]*lap[:,1,0])/det
        el=el.reshape(args.sweeps,batch)
        if not np.isfinite(el).all():raise AssertionError('nonfinite local energy')
        # Independently evaluated Fourier determinant vs actual Production EL.
        local=PH.local_energy(net.apply,jnp.zeros(1),(2,0),complex_output=complex_output,
            lattice=jnp.eye(2)*L,convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},
            kinetic_kwargs={'laplacian_method':'folx'})
        actual=np.asarray(jax.jit(jax.vmap(lambda x:local(p,None,
            networks.FermiNetData(positions=x,spins=jnp.ones(2),atoms=jnp.zeros((1,2)),charges=jnp.zeros(1)))[0]))(jnp.asarray(pos[:8].reshape(-1,4))))
        delta=float(np.max(np.abs(actual-el[:8].ravel())))
        np.testing.assert_allclose(actual,el[:8].ravel(),rtol=1e-11,atol=1e-11)
        lengths=[t for t in (1024,4096,16384) if t<=args.sweeps]
        for t in lengths:
            st=blocking(el[:t].real);im=blocking(el[:t].imag)
            sem=st['conservative_block_sem'];var=st['local_std']**2
            rows.append(dict(seed=seed,sweeps=t,samples=t*batch,energy=st,imaginary=im,
                difference=st['mean']-reference,z=(st['mean']-reference)/sem,
                acceptance=float(accept[:t].mean()),
                tau_int_half_convention=.5*(sem/st['iid_sem'])**2,
                effective_samples=var/sem**2,production_local_energy_max_abs=delta))
        np.savez(args.output/f'seed-{seed}.npz',local_energy=el,acceptance=accept,
                 final_positions=np.asarray(final.positions),final_rng=np.asarray(key))
    pools=[]
    for t in sorted({r['sweeps'] for r in rows}):
        selected=[r for r in rows if r['sweeps']==t]
        mean=float(np.mean([r['energy']['mean'] for r in selected]))
        sem=float(np.sqrt(sum(r['energy']['conservative_block_sem']**2 for r in selected))/len(selected))
        spread=float(np.std([r['energy']['mean'] for r in selected],ddof=1)/np.sqrt(len(selected)))
        pools.append(dict(sweeps=t,seeds=len(selected),samples=t*batch*len(selected),mean=mean,
            independent_chain_block_sem=sem,between_seed_sem=spread,difference=mean-reference,z=(mean-reference)/sem))
    return dict(case=str(args.case),reference=reference,ground=Q*Q/2,
        variational_gap=reference-Q*Q/2,initialization_seed=42,
        note='Training init remains seed42. Independent seeds below change frozen measurement chains, not training initialization.',
        burn=128,walkers=batch,rows=rows,pooled=pools,
        parameter_sha256=hashlib.sha256(np.asarray(ravel_pytree(p)[0]).tobytes()).hexdigest())


def metric(a,b):
    a=np.asarray(a);b=np.asarray(b);d=np.abs(a-b)
    return dict(max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),
                reference_rms=float(np.sqrt(np.mean(np.abs(b)**2))))


def stability(args):
    directory=Path('tests/g3a_followup/frozen-complex24')
    target=1.2347060624819985
    rows=[]
    for label,start,stop in (('burn128',0,None),('burn512',384,None),
                              ('burn1024',896,None),('first_half',0,8192),('second_half',8192,None)):
        stats=[]
        for seed in (2303,2304,2305,2306):
            with np.load(directory/f'seed-{seed}.npz') as z:el=z['local_energy'][start:stop].real
            stats.append(blocking(el))
        mean=float(np.mean([s['mean'] for s in stats]))
        sem=float(np.sqrt(sum(s['conservative_block_sem']**2 for s in stats))/4)
        rows.append(dict(window=label,mean=mean,block_sem=sem,z=(mean-target)/sem,
                         all_plateaus=all(s['plateau_last3'] for s in stats)))
    return dict(fixed_reference=target,rows=rows,
                warning='Windows share samples. Burn comparisons are sensitivity diagnostics, not independent replications.')


def probes(args):
    results={}
    # 1-token dot_product_attention must be the identity on Value, mathematically.
    q=jnp.array([[[.3,-.7]]]);k=jnp.array([[[.2,.5]]]);v=jnp.array([[[.9,-.1]]])
    results['one_token_attention_identity']={}
    for method in ('jacfwd','jacrev'):
        diff=getattr(jax,method)(lambda w:jax.nn.dot_product_attention(q,k,w))(v)
        results['one_token_attention_identity'][method]=metric(diff.reshape(2,2),np.eye(2))
    # Compare raw orbital and log/phase Hessian identities at the saved NN state.
    for route,name in (('psiformer','psiformer-complex64'),('screen','screen-complex64')):
        directory=Path('tests/g3a_validation')/name;meta=json.loads((directory/'summary.json').read_text())
        p=restored(directory,meta['batch']);net,_=build(route,1,True)
        x=jnp.array([.31,.73]);s=jnp.ones(1);a=jnp.zeros((1,2));c=jnp.zeros(1)
        def orb(x):return net.orbitals(p,x,s,a,c)[0][0,0,0]
        z=orb(x)
        direct=-.5*(jnp.trace(jax.hessian(lambda x:orb(x).real)(x))+
                      1j*jnp.trace(jax.hessian(lambda x:orb(x).imag)(x)))/z
        vals={}
        for method in ('default','folx'):
            local=PH.local_energy(net.apply,c,(1,0),complex_output=True,lattice=jnp.eye(2)*L,
                convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},
                kinetic_kwargs={'laplacian_method':method})
            d=networks.FermiNetData(positions=x,spins=s,atoms=a,charges=c)
            vals[method]=serial(local(p,None,d)[0])
        # Fourth-order finite differences of the raw orbital, not of log(psi).
        fd=[]
        evaluate=jax.jit(orb)
        for h in (.016,.008,.004,.002):
            lap=0j
            for axis in range(2):
                e=np.eye(2)[axis]*h
                lap+=complex((-evaluate(x+2*e)+16*evaluate(x+e)-30*evaluate(x)
                             +16*evaluate(x-e)-evaluate(x-2*e))/(12*h*h))
            fd.append(dict(h=h,energy=serial(-.5*lap/complex(z)),difference_from_orbital_ad=float(abs(-.5*lap/complex(z)-complex(direct)))))
        results[route]=dict(direct_orbital_ad=serial(direct),production=vals,orbital_finite_difference=fd)
        if route=='psiformer':
            # Counterfactual test ONLY: for a single token softmax is exactly 1.
            # This is not installed as an alternative Production implementation.
            with patch.object(jax.nn,'dot_product_attention',side_effect=lambda q,k,v,*a,**kw:v):
                alternate,_=build('psiformer',1,True)
                def altorb(x):return alternate.orbitals(p,x,s,a,c)[0][0,0,0]
                altz=altorb(x)
                altdirect=-.5*(jnp.trace(jax.hessian(lambda x:altorb(x).real)(x))+
                              1j*jnp.trace(jax.hessian(lambda x:altorb(x).imag)(x)))/altz
                loc=PH.local_energy(alternate.apply,c,(1,0),complex_output=True,lattice=jnp.eye(2)*L,
                    convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},
                    kinetic_kwargs={'laplacian_method':'default'})
                alte=loc(p,None,networks.FermiNetData(positions=x,spins=s,atoms=a,charges=c))[0]
            results[route]['one_token_identity_counterfactual']=dict(
                forward_orbital_difference=float(abs(altz-z)),direct_ad=serial(altdirect),
                log_ad=serial(alte),difference=float(abs(alte-altdirect)))
        # Identify graph-sensitive EL and score separately on the identical stored batch.
        if route=='screen':
            with np.load(directory/'frozen_samples.npz') as f:
                positions=jnp.asarray(f['positions'].reshape(-1,2));old_el=f['local_energy'].ravel()
                old_score=f['score_direction'].ravel();direction=ravel_pytree(p)[1](jnp.asarray(f['direction']))
            def lp(pp,x,s,a,c):
                ph,la=net.apply(pp,x,s,a,c);return la+1j*ph
            bd=networks.FermiNetData(positions=positions,spins=jnp.ones((len(positions),1)),
                atoms=jnp.zeros((len(positions),1,2)),charges=jnp.zeros((len(positions),1)))
            # local is folx here; retain the existing G3a setting.
            scores=jax.jit(lambda pp,v:jax.jvp(lambda pp:jax.vmap(lp,in_axes=(None,0,0,0,0))(
                pp,bd.positions,bd.spins,bd.atoms,bd.charges),(pp,),(v,))[1])(p,direction)
            contrasts=[]
            for chunk in (0,32,128):
                fn=loss.make_loss(lp,local,complex_output=True,local_energy_chunk=chunk)
                def along(t):return fn(jax.tree_util.tree_map(lambda a,b:a+t*b,p,direction),jax.random.PRNGKey(0),bd)[0].real
                tangent=jax.jit(lambda t:jax.jvp(along,(t,),(jnp.ones(()),))[1])(jnp.zeros(()))
                native=jax.jit(lambda pp,dp:jax.jvp(lambda pp:fn(pp,jax.random.PRNGKey(0),bd)[0].real,
                                                  (pp,),(dp,))[1])(p,direction)
                perturbed=jax.tree_util.tree_map(lambda a,b:a+jnp.zeros(())*b,p,direction)
                promoted=[jax.tree_util.keystr(k) for (k,a),b in zip(
                    jax.tree_util.tree_flatten_with_path(p)[0],jax.tree_util.tree_leaves(perturbed)) if a.dtype!=b.dtype]
                aux=jax.jit(fn)(p,jax.random.PRNGKey(0),bd)[1];el=np.asarray(aux.local_energy)
                score=np.asarray(scores);ref=2*np.real(np.mean((el-el.mean())*score.conj()))
                contrasts.append(dict(chunk=chunk,production_tangent=float(tangent),separate_score_contraction=float(ref),
                    contraction_delta=float(tangent)-ref,native_pytree_jvp=float(native),native_delta=float(native)-ref,
                    scalar_perturbation_promoted_leaves=promoted,
                    local_energy_vs_saved=metric(el,old_el),score_vs_saved=metric(score,old_score)))
            results['screen_graphs']=contrasts
        jax.clear_caches()
    # x64 network parameters contain explicitly f32 Screening scalar leaves.
    net,_=build('screen',1,True);p=net.init(jax.random.PRNGKey(42))
    leaves=jax.tree_util.tree_flatten_with_path(p)[0]
    results['screen_parameter_dtypes']={jax.tree_util.keystr(k):str(v.dtype) for k,v in leaves}
    results['jax_attention_implementation']='jax/_src/nn/functions.py:966 casts padded logits to float32 before softmax (jax 0.7.2); this is independent of the Screening mixed parameter dtypes.'
    return results


def attention_precision(args):
    """Test-only float64 softmax counterfactual; never used for training."""
    from tests.support import make_case
    def double_attention(q,k,v):
        logits=jnp.einsum('ihd,jhd->hij',q,k)/jnp.sqrt(q.shape[-1])
        w=jax.nn.softmax(logits,axis=-1)
        return jnp.einsum('hij,jhd->ihd',w,v)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for alternate in (False,True):
            context=patch.object(jax.nn,'dot_product_attention',side_effect=double_attention) if alternate else patch.object(jax.nn,'dot_product_attention',wraps=jax.nn.dot_product_attention)
            with context:
                net,p,d,_,_,_=make_case('psiformer',tmp)
                values=[]
                for component in (0,1):
                    f=lambda x:net.apply(p,x,d.spins,d.atoms,d.charges)[component]
                    h=jnp.trace(jax.hessian(f)(d.positions))
                    fl=folx.forward_laplacian(f,sparsity_threshold=0)(d.positions).laplacian
                    values.append(dict(component=component,value=float(f(d.positions)),
                        hessian_trace=float(h),folx_laplacian=float(fl),difference=float(fl-h)))
            rows.append(dict(test_only_float64_attention=alternate,components=values))
            jax.clear_caches()
    return dict(conditions='Existing 2-electron/two-head/GELU/LayerNorm test case, seed7, float64',
                rows=rows,production_modified=False,
                caution='Counterfactual changes legacy forward arithmetic; a potential fix requires separate authorization and versioned numerical baselines.')


def kfac_structure(args):
    """Observe the original failing weighted sum; no replacement update."""
    import kfac_jax
    from tests.check_g3a_vmc import config
    from periodicwave import train_screen
    observations=[]
    original=kfac_jax.utils.weighted_sum_of_objects
    def observe(objects,coefficients):
        try:return original(objects,coefficients)
        except ValueError:
            accumulator=kfac_jax.utils.scalar_mul(objects[0],coefficients[0])
            left=jax.tree_util.tree_flatten_with_path(accumulator)[0]
            right=jax.tree_util.tree_leaves(objects[1])
            differences=[]
            for (key,a),b in zip(left,right):
                if a.dtype!=b.dtype or a.shape!=b.shape:
                    differences.append(dict(path=jax.tree_util.keystr(key),
                        scaled_first_dtype=str(a.dtype),second_dtype=str(b.dtype),
                        first_shape=list(a.shape),second_shape=list(b.shape)))
            observations.append(dict(coefficients=[str(getattr(c,'dtype',type(c))) for c in coefficients],
                                     mismatched_leaves=differences))
            raise
    cfg=config('screen',1,True,'kfac',2,64,args.output/'run')
    try:
        with patch.object(kfac_jax.utils,'weighted_sum_of_objects',side_effect=observe):train_screen.train(cfg)
    except ValueError as e:failure=dict(type=type(e).__name__,message=str(e))
    else:raise AssertionError('known KFAC structure error did not recur')
    return dict(status='ERROR (known K08 reproduced, not corrected)',error=failure,
                observations=observations,production_modified=False)


def ewald_reference(args):
    # Independent scalar Ewald sum; vary alpha and cutoff, no Production routines.
    def independent(r,alpha,cut):
        m=np.array([(i,j) for i in range(-cut,cut+1) for j in range(-cut,cut+1)])
        nz=np.any(m!=0,axis=1);R=L*m;G=Q*m[nz];g=np.linalg.norm(G,axis=1)
        rr=np.linalg.norm(np.asarray(r)-R,axis=1)
        real=sum(math.erfc(alpha*x)/x for x in rr)
        recip=(2*np.pi/L**2)*sum(math.erfc(x/(2*alpha))/x*y for x,y in zip(g,np.cos(G@r)))
        rn=np.linalg.norm(R[nz],axis=1)
        mad=sum(math.erfc(alpha*x)/x for x in rn)+(2*np.pi/L**2)*sum(math.erfc(x/(2*alpha))/x for x in g)-2*alpha/np.sqrt(np.pi)
        bg=2*np.sqrt(np.pi)/(L**2*alpha)
        xi=mad-bg
        return real+recip-bg+xi,xi  # N=2: pair-periodic potential plus N*xi/2.
    displacements=[np.array([.3,.7]),np.array([1.1,-.9]),np.array([1.9,1.7])]
    oracle=[independent(r,2.8/L,12)[0] for r in displacements]
    alpha_checks=[]
    for alpha in (1.5/L,2.8/L,4./L):
        alpha_checks.append(dict(alpha=alpha,values=[independent(r,alpha,16)[0] for r in displacements]))
    production=[]
    for cut in (2,3,5,7):
        f=PH.make_2DCoulomb_potential(jnp.eye(2)*L,jnp.zeros((1,2)),jnp.zeros(1),truncation_limit=cut,interaction_energy_scale=1.)
        values=[]
        for r in displacements:
            ee=jnp.asarray([[np.zeros(2),r],[-r,np.zeros(2)]])
            values.append(float(f(None,ee)))
        production.append(dict(cutoff=cut,values=values,vs_independent=metric(values,oracle)))
    xi=independent(displacements[0],2.8/L,12)[1]
    def finite_basis_energy(c):
        # Non-orthogonal Slater density matrix, independent of Production EL.
        freq=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
        inv=np.linalg.inv(c.conj().T@c)
        kinetic=float(np.trace(inv@(c.conj().T@(np.sum(freq*freq,axis=1)[:,None]*Q*Q/2*c))).real)
        pair=0j
        for q in {(int(x),int(y)) for x,y in (freq[:,None]-freq[None,:]).reshape(-1,2)}-{(0,0)}:
            shift=np.all(freq[:,None]-freq[None,:]==q,axis=-1)
            rq=c.conj().T@shift@c
            pair+=(2*np.pi/(L**2*Q*np.linalg.norm(q)))/2*(
                np.trace(inv@rq)*np.trace(inv@rq.conj().T)-np.trace(inv@rq@inv@rq.conj().T))
        return dict(kinetic=kinetic,pair=float(pair.real),background_self=xi,total=kinetic+float(pair.real)+xi)
    plane=np.zeros((5,2),dtype=complex);plane[0,0]=1;plane[1,1]=1
    standing=plane.copy();standing[1,1]=.5;standing[2,1]=.5
    finite_refs={'plane':finite_basis_energy(plane),'standing':finite_basis_energy(standing)}
    current=Path('tests/g3a_followup/slater-real-two64-96')
    if current.exists():
        from tests.check_g3a_vmc import fourier_coefficients
        finite_refs['prepared_real_slater96']=finite_basis_energy(fourier_coefficients(restored(current,128),False))
    # Exact expectation for determinant (1, exp(iQx)); real (1,cos Qx)
    # is a superposition of orthogonal +/- total-momentum sectors, same energy.
    e=Q*Q/2-2*np.pi/(L**2*Q)+xi
    tol=32*np.finfo(np.float64).eps
    for row in alpha_checks:
        np.testing.assert_allclose(row['values'],oracle,atol=tol,rtol=tol)
    for row in production:
        if row['cutoff']>=5:np.testing.assert_allclose(row['values'],oracle,atol=tol,rtol=tol)
    for name in ('plane','standing'):
        np.testing.assert_allclose(finite_refs[name]['total'],e,atol=tol,rtol=tol)
    return dict(model='N=2 fully polarized square 2DEG; 3D 1/r interaction confined to 2D, uniform neutralizing background, periodic Ewald',
        L=L,area=L*L,density=2/L**2,rs=L/np.sqrt(2*np.pi),kinetic_prefactor=-.5,
        interaction_energy_scale=1.,twist=[0.,0.],units='Hartree atomic units; TOTAL energy, divide by 2 for per electron',
        independent_displacements=[x.tolist() for x in displacements],independent_values=oracle,
        alpha_invariance=alpha_checks,production_cutoff=production,xi_with_background=xi,
        analytic_plane_determinant_expectation=e,per_electron=e/2,
        finite_fourier_slater_references=finite_refs,
        independent_calibration_checks='PASS at 32eps on the specified points/states; not a whole-cell error bound',
        reference_definition='q!=0 Fourier pair coefficient 2*pi/(A*|q|), q=0 omitted, constant N*xi/2; E=Q^2/2-2*pi/(A*Q)+xi. A calibration state, NOT interacting ground energy.',
        production_interacting_training_executed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('sample','probes','reference','attention','kfac','stability'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',type=Path)
    parser.add_argument('--sweeps',type=int,default=16384)
    args=parser.parse_args()
    if args.mode=='sample' and args.case is None:parser.error('sample requires --case')
    if args.output.exists():parser.error('existing results are read-only')
    args.output.mkdir(parents=True)
    jax.config.update('jax_enable_x64',True)
    result={'sample':sample,'probes':probes,'reference':ewald_reference,
            'attention':attention_precision,'kfac':kfac_structure,'stability':stability}[args.mode](args)
    (args.output/'summary.json').write_text(json.dumps(result,default=serial,indent=2,allow_nan=False))
    print('Recorded',args.mode,'in',args.output)
