"""G3a: existing networks/free PBC Hamiltonian, exact/spectral references.

Production is unchanged. Files are created only in a new requested directory.
N=1 ground energy=0; N=2 polarized ground energy=(2*pi/L)^2/2, L=4.
SlaterNet with num_layers=0 is an existing finite-Fourier special case, not a
new Ansatz. Nonlinear one-layer networks use converged Fourier quadrature.
"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import traceback
from unittest.mock import patch
import numpy as np
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from periodicwave import base_config,checkpoint,envelopes,loss,mcmc,networks,train,train_screen
from periodicwave import SlaterNet,CustomPsiformer as CP,CustomScreenformer as CS
from periodicwave.pbc import feature_layer,hamiltonian as PH

L=4.
Q=2*np.pi/L
MAKERS={'slater':SlaterNet,'psiformer':CP,'screen':CS}
NAMES={'slater':'SlaterNet','psiformer':'CustomPsiformer','screen':'CustomScreenformer'}


def serial(x):
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    a=np.asarray(x)
    if np.iscomplexobj(a):return dict(real=a.real.tolist(),imag=a.imag.tolist())
    return a.tolist()


def build(route,n,complex_output,layers=None):
    if layers is None:layers=0 if route=='slater' else 1
    opts=dict(ndim=2,determinants=1,envelope=envelopes.make_null_envelope(),feature_layer=feature_layer.make_pbc_feature_layer(natoms=1,nspins=(n,0),ndim=2,lattice=jnp.eye(2)*L,include_r_ae=False),jastrow='NONE',jastrow_kwargs={'ndim':2},complex_output=complex_output,bias_orbitals=True,pbc_lattice=jnp.eye(2)*L,num_layers=layers,mlp_dim=8,num_perceptrons_per_layer=1,use_layer_norm=False,mlp_activation_fct='GELU')
    if route!='slater':opts.update(num_heads=1,attn_dim=4,value_dim=4)
    net=MAKERS[route].make_fermi_net((n,0),jnp.zeros(1),**opts)
    return net,opts


def config(route,n,complex_output,opt,iterations,batch,out):
    cfg=base_config.default();cfg.batch_size=batch;cfg.debug.deterministic=True
    cfg.optim.optimizer=opt;cfg.optim.iterations=iterations;cfg.optim.clip_local_energy=0.
    cfg.optim.lr.rate=.01 if opt=='adam' else .02;cfg.optim.lr.decay=0.
    cfg.mcmc.steps=8;cfg.mcmc.burn_in=16;cfg.mcmc.move_width=1.;cfg.mcmc.move_width_updater='const';cfg.mcmc.blocks=1
    cfg.system.electrons=(n,0);cfg.system.ndim=2;cfg.system.pbc_lattice=np.eye(2)*L
    cfg.system.make_local_energy_fn='periodicwave.pbc.hamiltonian.local_energy'
    cfg.system.make_local_energy_kwargs={'lattice':np.eye(2)*L,'potential_type':'Coulomb','convergence_radius':2,'potential_kwargs':{'interaction_energy_scale':0.},'kinetic_kwargs':{'laplacian_method':'folx'}}
    cfg.network.network_type=NAMES[route];cfg.network.complex=complex_output;cfg.network.determinants=1;cfg.network.bias_orbitals=True;cfg.network.jastrow='NONE';cfg.network.jastrow_kwargs={'ndim':2}
    cfg.network.make_envelope_fn='periodicwave.envelopes.make_null_envelope'
    cfg.network.make_feature_layer_fn='periodicwave.pbc.feature_layer.make_pbc_feature_layer'
    cfg.network.make_feature_layer_kwargs={'lattice':np.eye(2)*L,'include_r_ae':False}
    _,opts=build(route,n,complex_output)
    exclude={'ndim','determinants','envelope','feature_layer','jastrow','jastrow_kwargs','complex_output','bias_orbitals','pbc_lattice'}
    import ml_collections
    cfg.network[NAMES[route]]=ml_collections.ConfigDict({k:v for k,v in opts.items() if k not in exclude})
    cfg.log.save_path=str(out);cfg.log.save_frequency=0.;return cfg


def orbital_coefficients(p,complex_output,xp=np):
    # Independent coefficient contraction; basis is sin x,sin y,cos x,cos y,1.
    a=xp.asarray(p['layers']['embed'])@xp.asarray(p['orbital'][0]['w'])
    b=xp.asarray(p['orbital'][0]['b'])
    if complex_output:a=a[:,::2]+1j*a[:,1::2];b=b[::2]+1j*b[1::2]
    c=xp.concatenate((a[:4],(a[4]+b)[None,:]),axis=0)
    return c


def fourier_coefficients(p,complex_output,xp=np):
    c=orbital_coefficients(p,complex_output,xp)
    return xp.stack((c[4],(c[2]-1j*c[0])/2,(c[2]+1j*c[0])/2,(c[3]-1j*c[1])/2,(c[3]+1j*c[1])/2))


def exact_slater_energy(p,complex_output,xp=np):
    c=fourier_coefficients(p,complex_output,xp);s=c.conj().T@c
    t=c.conj().T@(xp.asarray([0.,Q*Q/2,Q*Q/2,Q*Q/2,Q*Q/2])[:,None]*c)
    return xp.trace(xp.linalg.solve(s,t)).real


def analytic_orbitals(p,r,complex_output):
    r=np.asarray(r).reshape(-1,2);c=orbital_coefficients(p,complex_output)
    f=np.stack((np.sin(Q*r[:,0]),np.sin(Q*r[:,1]),np.cos(Q*r[:,0]),np.cos(Q*r[:,1]),np.ones(len(r))),axis=1)
    mat=f@c;lap=-(Q*Q)*(mat-c[4]);psi=np.linalg.det(mat)
    kinetic=-.5*np.einsum('ji,ij',np.linalg.inv(mat),lap)
    return mat,psi,kinetic


def spectral_energy_fn(net,grid):
    # Dirichlet/Parseval form: no log derivatives and no local-energy routine.
    r=np.stack(np.meshgrid((np.arange(grid)+.137)*L/grid,(np.arange(grid)+.137)*L/grid,indexing='ij'),axis=-1).reshape(-1,2)
    kk=np.fft.fftfreq(grid,d=1/grid);k2=(kk[:,None]**2+kk[None,:]**2)*(Q*Q)
    def energy(p):
        pos=jnp.asarray(r,dtype=jax.tree_util.tree_leaves(p)[0].dtype)
        z=jax.vmap(lambda x:net.orbitals(p,x,jnp.ones(1),jnp.zeros((1,2)),jnp.zeros(1))[0][0,0,0])(pos).reshape(grid,grid)
        a=jnp.fft.fft2(z)/(grid*grid);w=jnp.abs(a)**2
        return jnp.sum(jnp.asarray(k2)*w)/jnp.sum(w)/2
    return jax.jit(energy)


def blocking(series):
    """Time blocks within independent walkers; never treat locstd/EW as SEM."""
    a=np.asarray(series,dtype=float)
    if a.ndim==1:a=a[:,None]
    T,B=a.shape;rows=[]
    for block in (1,4,16,32,64,128):
        nb=T//block
        if nb<4:continue
        z=a[:nb*block].reshape(nb,block,B).mean(axis=1).ravel()
        rows.append(dict(block=block,blocks_per_walker=nb,independent_block_count=len(z),sem=float(z.std(ddof=1)/np.sqrt(len(z)))))
    chosen=max((r['sem'] for r in rows if r['block']>=16),default=rows[-1]['sem'])
    tail=[r['sem'] for r in rows[-3:]]
    plateau=(max(tail)/min(tail)<1.5) if min(tail)>0 else max(tail)==0
    return dict(mean=float(a.mean()),local_std=float(a.std()),iid_sem=float(a.std(ddof=1)/np.sqrt(a.size)),conservative_block_sem=chosen,blocking=rows,plateau_last3=bool(plateau))


def sampled_check(net,p,energy,n,complex_output,reference,refgrad,out,sweeps=1024,reference_fn=None,reference_cos=None,reference_budget=0.):
    B=32;key=jax.random.PRNGKey(2301+n)
    key,k=jax.random.split(key);pos=jax.random.uniform(k,(B,2*n),minval=0.,maxval=L)
    d=networks.FermiNetData(positions=pos,spins=jnp.ones((B,n)),atoms=jnp.zeros((B,1,2)),charges=jnp.zeros((B,1)))
    logabs=lambda pp,x,s,a,c:net.apply(pp,x,s,a,c)[1]
    bn=jax.vmap(logabs,in_axes=(None,0,0,0,0));step=mcmc.make_mcmc_step(bn,B,steps=1,ndim=2,blocks=1)
    def sample(dd,kk):
        def one(carry,_):
            data,k=carry;k,sub=jax.random.split(k);data,accept=step(p,data,sub,1.)
            return (data,k),(data.positions,accept)
        return jax.lax.scan(one,(dd,kk),None,length=sweeps+128)
    (_,lastkey),(positions,accept)=jax.jit(sample)(d,key);positions=positions[128:];accept=accept[128:]
    flat=positions.reshape(-1,2*n)
    eval_el=jax.jit(jax.vmap(lambda x:energy(p,None,networks.FermiNetData(positions=x,spins=jnp.ones(n),atoms=jnp.zeros((1,2)),charges=jnp.zeros(1)))[0]))
    el=np.asarray(eval_el(flat)).reshape(sweeps,B)
    es=blocking(el.real);ims=blocking(el.imag)
    flatp,unravel=ravel_pytree(p);v=jax.random.normal(jax.random.PRNGKey(71),flatp.shape);v=v/jnp.linalg.norm(v);direction=unravel(v)
    def score(x):
        def lp(q):
            ph,la=net.apply(q,x,jnp.ones(n),jnp.zeros((1,2)),jnp.zeros(1));return la+1j*ph if complex_output else la
        return jax.jvp(lp,(p,),(direction,))[1]
    O=np.asarray(jax.jit(jax.vmap(score))(flat)).reshape(sweeps,B)
    gs=blocking(2*np.real((el-reference)*O.conj()))
    expected_direction=float(np.dot(np.asarray(ravel_pytree(refgrad)[0]),np.asarray(v)))
    production_data=networks.FermiNetData(positions=flat,spins=jnp.ones((len(flat),n)),atoms=jnp.zeros((len(flat),1,2)),charges=jnp.zeros((len(flat),1)))
    def production_log(q,x,s,a,c):
        ph,la=net.apply(q,x,s,a,c);return la+1j*ph if complex_output else la
    production_loss=loss.make_loss(production_log,energy,complex_output=complex_output,local_energy_chunk=128)
    def along(t):return production_loss(jax.tree_util.tree_map(lambda a,b:a+t*b,p,direction),jax.random.PRNGKey(0),production_data)[0].real
    _,production_direction=jax.jit(lambda t,v:jax.jvp(along,(t,),(v,)))(jnp.zeros(()),jnp.ones(()))
    centered_score=2*np.real(np.mean((el-el.mean())*O.conj()))
    h=1e-5
    with jax.experimental.enable_x64():
        p64=jax.tree_util.tree_map(lambda a:jnp.asarray(a,dtype=jnp.float64),p)
        v64=jax.tree_util.tree_map(lambda a:jnp.asarray(a,dtype=jnp.float64),direction)
        pp=jax.tree_util.tree_map(lambda a,b:a+h*b,p64,v64);pm=jax.tree_util.tree_map(lambda a,b:a-h*b,p64,v64)
        fd=float((reference_fn(pp)-reference_fn(pm))/(2*h)) if reference_fn else None
    # Torus observables, not the unwrapped position density.
    x=np.asarray(positions)[...,0];cx=np.cos(Q*x)
    obs=blocking(cx)
    np.savez(out/'frozen_samples.npz',positions=np.asarray(positions),local_energy=el,score_direction=O,acceptance=np.asarray(accept),direction=np.asarray(v),final_rng=np.asarray(lastkey))
    result=dict(energy=es,imaginary_energy=ims,gradient_direction=gs,exact_energy=reference,exact_direction=expected_direction,energy_z=(es['mean']-reference)/es['conservative_block_sem'] if es['conservative_block_sem'] else None,gradient_z=(gs['mean']-expected_direction)/gs['conservative_block_sem'] if gs['conservative_block_sem'] else None,cos_x=obs,acceptance=float(np.asarray(accept).mean()),chains=B,sweeps=sweeps,burn_in=128)
    eps=np.finfo(np.asarray(flatp).dtype).eps
    budget=float(32*eps*max(1.,abs(reference))+reference_budget)
    result.update(production_direction=float(production_direction),independent_centered_score=float(centered_score),independent_energy_finite_difference=fd,reference_cos_x=reference_cos,numerical_budget=budget,reference_truncation_budget=reference_budget,
        energy_compatible=bool(abs(es['mean']-reference)<=4.5*es['conservative_block_sem']+budget),
        gradient_compatible=bool(abs(gs['mean']-expected_direction)<=4.5*gs['conservative_block_sem']+32*eps*max(1.,abs(expected_direction))),
        imaginary_mean_compatible=bool(abs(ims['mean'])<=4.5*ims['conservative_block_sem']+budget),
        production_contraction_compatible=bool(abs(float(production_direction)-centered_score)<=32*eps*max(1.,abs(centered_score))),
        cos_x_compatible=bool(abs(obs['mean']-reference_cos)<=4.5*obs['conservative_block_sem']+budget) if reference_cos is not None else None)
    return result


def exact_cos_x(p,complex_output):
    c=fourier_coefficients(p,complex_output);freq=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    delta=freq[:,None,:]-freq[None,:,:]
    op=.5*((np.abs(delta[...,0])==1)&(delta[...,1]==0))
    S=c.conj().T@c;B=c.conj().T@op@c
    return float(np.trace(np.linalg.solve(S,B)).real/c.shape[1])


def spectral_cos_x(net,p,grid=256):
    x=(np.arange(grid)+.137)*L/grid;r=np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2)
    z=jax.jit(jax.vmap(lambda x:net.orbitals(p,x,jnp.ones(1),jnp.zeros((1,2)),jnp.zeros(1))[0][0,0,0]))(jnp.asarray(r))
    w=np.abs(np.asarray(z))**2
    return float(np.sum(w*np.cos(Q*r[:,0]))/np.sum(w))


def run_case(args):
    out=args.output
    if out.exists():raise FileExistsError('existing output is protected: '+str(out))
    out.mkdir(parents=True)
    jax.config.update('jax_enable_x64',not args.float32)
    net,opts=build(args.route,args.particles,not args.real)
    cfg=config(args.route,args.particles,not args.real,args.optimizer,args.iterations,args.batch,out/'run')
    cfg.mcmc.burn_in=16
    cfg.system.make_local_energy_kwargs['kinetic_kwargs']['laplacian_method']=args.laplacian
    (out/'effective-config.json').write_text(json.dumps(cfg.to_dict(),default=serial,indent=2,allow_nan=False))
    from tests.baseline import metadata
    (out/'environment.json').write_text(json.dumps(metadata(),indent=2))
    initial=net.init(jax.random.split(jax.random.PRNGKey(42))[1])
    np.savez(out/'initial_params.npz',params=initial)
    trainer=train_screen if args.route=='screen' else train
    log_errors=[]
    def log(fmt,*a,**kw):
        try:logging.LogRecord('g3a',logging.INFO,'',0,fmt,a,None).getMessage()
        except (TypeError,ValueError) as e:log_errors.append(dict(format=fmt,error=str(e)))
    try:
        with patch.object(trainer.logging,'info',side_effect=log):trainer.train(cfg)
    except Exception as e:
        (out/'training-error.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),known_candidate='K08 if float64 Screening KFAC'),indent=2))
        raise
    with (out/'run/train_stats.csv').open() as f:rows=list(csv.DictReader(f))
    energy=PH.local_energy(net.apply,jnp.zeros(1),(args.particles,0),complex_output=not args.real,lattice=jnp.eye(2)*L,convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':args.laplacian})
    def lp(p,x,s,a,c):
        ph,la=net.apply(p,x,s,a,c);return la if args.real else la+1j*ph
    objective=loss.make_loss(lp,energy,complex_output=not args.real,clip_local_energy=0.)
    gradfn=jax.jit(jax.value_and_grad(objective,has_aux=True))
    opt=optax.chain(optax.scale_by_adam(**cfg.optim.adam),optax.scale_by_schedule(lambda t:cfg.optim.lr.rate*(1+t/cfg.optim.lr.delay)**(-cfg.optim.lr.decay)),optax.scale(-1.))
    prev=initial;state=opt.init(prev);trace=[];paths=sorted((out/'run').glob('qmcjax_ckpt_*.npz'))
    if len(paths)!=args.iterations:raise AssertionError('every iteration checkpoint was required')
    if args.route=='slater':
        reference=lambda p:exact_slater_energy(p,not args.real,jnp)
    else:
        reference=spectral_energy_fn(net,64)
    reference=jax.jit(reference);refgradient=jax.jit(jax.grad(reference))
    for t,path in enumerate(paths):
        next_t,data,p,restored_state,width=checkpoint.restore(str(path),args.batch)
        p=jax.tree_util.tree_map(lambda a:jnp.asarray(a[0]),p);d=jax.tree_util.tree_map(lambda a:jnp.asarray(a[0]),data)
        (_,aux),g=gradfn(prev,jax.random.PRNGKey(0),d)
        eref=float(reference(prev));postref=float(reference(p));ground=0. if args.particles==1 else Q*Q/2
        if eref<ground-32*np.finfo(np.asarray(ravel_pytree(prev)[0]).dtype).eps or postref<ground-32*np.finfo(np.asarray(ravel_pytree(p)[0]).dtype).eps:raise AssertionError('variational bound violated')
        update_error=None
        if args.optimizer=='adam':
            upd,state=opt.update(g,state,prev);expected=optax.apply_updates(prev,upd)
            update_error=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(p)))
        trace.append(dict(step=t,reported_energy=float(rows[t]['energy']),local_complex_variance=float(aux.variance),gradient_norm=float(jnp.linalg.norm(ravel_pytree(g)[0])),reference_gradient_norm=float(jnp.linalg.norm(ravel_pytree(refgradient(prev))[0])),reference_before=eref,reference_after=postref,acceptance=float(rows[t]['pmove']),adam_independent_update_max_abs=update_error,restored_next_step=next_t))
        prev=p
    p=prev
    # Pointwise amplitude/phase, PBC and exchange; independent local energy for finite Fourier case.
    r=jnp.asarray([.31,.73,1.27,2.11][:2*args.particles]);s=jnp.ones(args.particles);a=jnp.zeros((1,2));c=jnp.zeros(1)
    ph,la=net.apply(p,r,s,a,c);psi=np.exp(complex(la)+1j*complex(ph)) if not args.real else float(ph)*np.exp(float(la))
    checks={}
    for i in range(args.particles):
        shifted=r.at[2*i].add(L);ph2,la2=net.apply(p,shifted,s,a,c);psi2=np.exp(complex(la2)+1j*complex(ph2)) if not args.real else float(ph2)*np.exp(float(la2));checks['pbc_'+str(i)]=float(abs(psi-psi2))
    if args.particles==2:
        rr=r.reshape(2,2)[::-1].ravel();ph2,la2=net.apply(p,rr,s[::-1],a,c);ps2=np.exp(complex(la2)+1j*complex(ph2)) if not args.real else float(ph2)*np.exp(float(la2));checks['exchange_error']=float(abs(psi+ps2))
    with jax.experimental.enable_x64():
        p64=jax.tree_util.tree_map(lambda a:jnp.asarray(a,dtype=jnp.float64),p)
        if args.route=='slater':
            matrix,exactpsi,exactel=analytic_orbitals(p64,r,not args.real)
            actualel=energy(p,None,networks.FermiNetData(positions=r,spins=s,atoms=a,charges=c))[0]
            checks.update(psi_error=float(abs(psi-exactpsi)),energy_error=float(abs(complex(actualel)-exactel)),analytic_psi=serial(exactpsi),analytic_energy=serial(exactel))
            reference_function=lambda pp:exact_slater_energy(pp,not args.real,jnp)
            reference_final=float(reference_function(p64));refgrad=jax.grad(reference_function)(p64);ref_convergence=None;ref_budget=0.
            ref_cos=exact_cos_x(jax.tree_util.tree_map(lambda a:np.asarray(a),p64),not args.real)
        else:
            values={str(k):float(spectral_energy_fn(net,k)(p64)) for k in (32,64,128,256)}
            reference_function=spectral_energy_fn(net,256)
            reference_final=values['256'];ref_convergence=values
            ref_budget=2*abs(values['256']-values['128'])
            refgrad=jax.grad(reference_function)(p64);ref_cos=spectral_cos_x(net,p64)
            z=np.asarray(net.orbitals(p,r,s,a,c)[0][0,0,0]);checks['psi_error']=float(abs(psi-z))
            def orb(x):return net.orbitals(p,x,s,a,c)[0][0,0,0]
            zlap=jnp.trace(jax.hessian(lambda x:orb(x).real)(r))+1j*jnp.trace(jax.hessian(lambda x:orb(x).imag)(r))
            direct=-.5*zlap/orb(r)
            actualel=energy(p,None,networks.FermiNetData(positions=r,spins=s,atoms=a,charges=c))[0]
            checks.update(energy_error=float(abs(complex(actualel)-complex(direct))),analytic_psi=serial(z),analytic_energy=serial(direct))
    sampling=sampled_check(net,p,energy,args.particles,not args.real,reference_final,refgrad,out,sweeps=args.sweeps,reference_fn=reference_function,reference_cos=ref_cos,reference_budget=ref_budget)
    # Round-trip an actual complete saved state without changing its format.
    saved=checkpoint.restore(str(paths[-1]),args.batch)
    roundtrip=out/'roundtrip';roundtrip.mkdir()
    copied=checkpoint.save(str(roundtrip),saved[0]-1,saved[1],saved[2],saved[3],saved[4]);again=checkpoint.restore(copied,args.batch)
    if jax.tree_util.tree_structure(saved)!=jax.tree_util.tree_structure(again):raise AssertionError('checkpoint tree changed')
    for x,y in zip(jax.tree_util.tree_leaves(saved),jax.tree_util.tree_leaves(again)):np.testing.assert_array_equal(x,y)
    epsilon=np.finfo(np.asarray(ravel_pytree(p)[0]).dtype).eps
    point_ok=all(checks[k]<=32*epsilon*(max(1.,abs(complex(actualel))) if k=='energy_error' else max(1.,abs(psi))) for k in checks if k.endswith('error') or k.startswith('pbc_'))
    physics_pass=point_ok and all(sampling[k] for k in ('energy_compatible','gradient_compatible','imaginary_mean_compatible','production_contraction_compatible','cos_x_compatible'))
    plateau_ok=all(sampling[k]['plateau_last3'] or sampling[k]['local_std']<=sampling['numerical_budget'] for k in ('energy','gradient_direction','cos_x','imaginary_energy'))
    physics_reference_pass=point_ok and all(sampling[k] for k in ('energy_compatible','gradient_compatible','imaginary_mean_compatible','cos_x_compatible'))
    summary=dict(version='g3a-v1',laplacian_method=args.laplacian,blocking_plateau_pass=plateau_ok,physics_reference_pass=bool(physics_reference_pass),route=args.route,particles=args.particles,complex_output=not args.real,x64=not args.float32,optimizer=args.optimizer,iterations=args.iterations,batch=args.batch,box_length=L,hamiltonian='PBC Coulomb with interaction_energy_scale=0 (free)',ground_energy=0. if args.particles==1 else Q*Q/2,log_errors=log_errors,trace=trace,point_checks=checks,reference_final=reference_final,reference_convergence=ref_convergence,sampling=sampling,checkpoint_format_unchanged=True,checkpoint_roundtrip_exact=True,PRNG_not_in_production_checkpoint=True,point_checks_pass=point_ok,core_physics_pass=bool(physics_pass),production_logging_healthy=not bool(log_errors),statistical_gate='4.5 conservative-block SEM + predeclared 32eps numerical budget + measured reference truncation',reference_grid_converged=(ref_budget<1e-7 if ref_convergence else True))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(final_reference=reference_final,initial_reference=trace[0]['reference_before'],sampling=sampling,log_errors=log_errors),indent=2))
    if not physics_pass or not plateau_ok or (ref_convergence is not None and ref_budget>=1e-7) or log_errors:raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--route',choices=MAKERS,default='slater');p.add_argument('--real',action='store_true');p.add_argument('--float32',action='store_true');p.add_argument('--particles',type=int,choices=(1,2),default=1);p.add_argument('--optimizer',choices=('adam','kfac'),default='adam');p.add_argument('--laplacian',choices=('default','folx'),default='folx');p.add_argument('--iterations',type=int,default=24);p.add_argument('--batch',type=int,default=128);p.add_argument('--sweeps',type=int,default=1024);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.route!='slater' and args.particles!=1:p.error('nonlinear spectral reference is N=1 only')
    run_case(args)
