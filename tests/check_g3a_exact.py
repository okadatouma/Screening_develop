"""Existing HF orbitals, lambda=0, exact eigenstates and torus MCMC laws."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
from periodicwave import CustomScreenformer as CS,checkpoint,loss,mcmc,networks,hamiltonian as H
from periodicwave.pbc import lambda_jellium as LJ
from tests.check_g3a_vmc import L,Q,build,blocking,serial


def run(args):
    out=args.output
    if out.exists():raise FileExistsError('existing results protected')
    out.mkdir(parents=True);jax.config.update('jax_enable_x64',not args.float32)
    records=[]
    for name,n,table in (
        ('plane',1,[[Q,0,0,0,1,0],[Q,0,0,Q,0,0]]),
        ('standing',1,[[0,0,Q,0,.5,0],[0,0,-Q,0,.5,0]]),
        ('two_fermions',2,[[0,0,0,0,1,0],[Q,0,0,0,1,0]])):
        case=out/name;case.mkdir();file=case/'hf.txt';np.savetxt(file,table,fmt='%.17g')
        _,opts=build('screen',n,True,layers=0)
        opts.update(lj_hf_path=str(file),lj_lam=0.,lj_net_scale=0.,lj_jastrow=False,lj_cusps=(0.,0.,0.))
        net=CS.make_fermi_net((n,0),jnp.zeros(1),**opts);p=net.init(jax.random.PRNGKey(7))
        r=jnp.asarray([.31,.73,1.27,2.11][:2*n]);s=jnp.ones(n);a=jnp.zeros((1,2));c=jnp.zeros(1)
        if name=='plane':expected_psi=np.exp(1j*Q*float(r[0]))
        elif name=='standing':expected_psi=np.cos(Q*float(r[0]))
        else:expected_psi=np.exp(1j*Q*float(r[2]))-np.exp(1j*Q*float(r[0]))
        ph,la=net.apply(p,r,s,a,c);z=np.exp(complex(la)+1j*complex(ph))
        eps=np.finfo(np.float32 if args.float32 else np.float64).eps;tol=32*eps
        np.testing.assert_allclose(z,expected_psi,rtol=tol,atol=tol)
        for i in range(n):
            ph1,la1=net.apply(p,r.at[2*i].add(L),s,a,c)
            np.testing.assert_allclose(np.exp(complex(la1)+1j*complex(ph1)),z,rtol=tol,atol=tol)
        if n==2:
            ph1,la1=net.apply(p,r.reshape(n,2)[::-1].ravel(),s[::-1],a,c)
            np.testing.assert_allclose(np.exp(complex(la1)+1j*complex(ph1)),-z,rtol=tol,atol=tol)
        energies={}
        for method in ('default','folx'):
            local=LJ.local_energy(net.apply,c,(n,0),lattice=jnp.eye(2)*L,lam=0.,delta=2.,potential_type='none',kinetic_kwargs={'laplacian_method':method})
            d=networks.FermiNetData(positions=r,spins=s,atoms=a,charges=c)
            e=jax.jit(lambda pp,dd:local(pp,None,dd)[0])(p,d)
            np.testing.assert_allclose(e.real,Q*Q/2,rtol=tol,atol=tol);np.testing.assert_allclose(e.imag,0.,atol=tol)
            energies[method]=serial(e)
        B=64;steps=2048;key=jax.random.PRNGKey(79+n)
        key,k=jax.random.split(key);bd=networks.FermiNetData(positions=jax.random.uniform(k,(B,2*n),minval=0,maxval=L),spins=jnp.ones((B,n)),atoms=jnp.zeros((B,1,2)),charges=jnp.zeros((B,1)))
        logabs=lambda pp,x,s,a,c:net.apply(pp,x,s,a,c)[1]
        step=mcmc.make_mcmc_step(jax.vmap(logabs,in_axes=(None,0,0,0,0)),B,steps=1,ndim=2,blocks=1)
        def chain(dd,key):
            def one(carry,_):
                d,k=carry;k,sub=jax.random.split(k);d,accept=step(p,d,sub,1.)
                return (d,k),(d.positions,accept)
            return jax.lax.scan(one,(dd,key),None,length=steps+128)
        (final_data,final_key),(pos,accept)=jax.jit(chain)(bd,key);pos=np.asarray(pos[128:]);accept=np.asarray(accept[128:])
        x=pos[...,0]
        features={'cos_x':(np.cos(Q*x),0.),'sin_x':(np.sin(Q*x),0.),'cos_y':(np.cos(Q*pos[...,1]),0.)}
        if name=='standing':features['cos_2x']=(np.cos(2*Q*x),.5)
        if n==2:features['cos_relative_x']=(np.cos(Q*(pos[...,0]-pos[...,2])),-.5)
        stats={}
        for label,(values,target) in features.items():
            st=blocking(values);st['expected']=target
            st['compatible']=bool(abs(st['mean']-target)<=4.5*st['conservative_block_sem']+tol)
            if not st['compatible'] or not st['plateau_last3']:raise AssertionError('MCMC law/blocking failed: '+label)
            stats[label]=st
        # Use the same existing Production loss; all NN parameters are inactive at net_scale=0.
        def logpsi(pp,x,s,a,c):
            ph,la=net.apply(pp,x,s,a,c);return la+1j*ph
        objective=loss.make_loss(logpsi,local,complex_output=True)
        (_,aux),g=jax.jit(jax.value_and_grad(objective,has_aux=True))(p,jax.random.PRNGKey(0),final_data)
        norm=float(jnp.sqrt(sum(jnp.sum(a*a) for a in jax.tree_util.tree_leaves(g))))
        np.testing.assert_allclose(norm,0.,atol=tol)
        ck=checkpoint.save(str(case),0,jax.tree_util.tree_map(lambda x:jnp.stack([x]),final_data),jax.tree_util.tree_map(lambda x:jnp.stack([x]),p),None,jnp.asarray([1.]))
        restored=checkpoint.restore(ck,B);assert restored[0]==1
        np.savez(case/'samples.npz',positions=pos,acceptance=accept,final_validation_rng=np.asarray(final_key),local_energy=np.asarray(aux.local_energy))
        records.append(dict(name=name,n=n,lambda_value=0.,delta=2.,HF_NN_scale=0.,Jastrow=False,expected_energy=Q*Q/2,expected_wavefunction=serial(expected_psi),actual_wavefunction=serial(z),energies=energies,gradient_norm=norm,acceptance=float(accept.mean()),statistics=stats,hf_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        jax.clear_caches()
    # Known literal-constant limitation is not silently classified as a healthy path.
    f=lambda p,x,s,a,c:(Q*x[0],jnp.asarray(0.,x.dtype))
    d=networks.FermiNetData(positions=jnp.asarray([.31,.73]),spins=jnp.ones(1),atoms=jnp.zeros((1,2)),charges=jnp.zeros(1))
    try:H.local_kinetic_energy(f,complex_output=True,laplacian_method='folx')(None,d)
    except AttributeError as e:constant_issue=dict(status='ERROR (known)',error=str(e))
    else:raise AssertionError('known literal constant error unexpectedly disappeared')
    from tests.baseline import metadata
    (out/'summary.json').write_text(json.dumps(dict(version='g3a-v1',x64=not args.float32,environment=metadata(),eigenstates=records,literal_constant_folx=constant_issue,production_modified=False),indent=2,allow_nan=False))
    print('Exact HF eigenstates, antisymmetry, PBC, zero gradient and MCMC laws checked:',len(records))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--float32',action='store_true');p.add_argument('--output',type=Path,required=True);run(p.parse_args())
