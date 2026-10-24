"""G3a supplementary checks on saved samples/states; never overwrites a run."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import optax
from periodicwave import checkpoint,loss,networks,lj_orbitals
from periodicwave.pbc import hamiltonian as PH
from tests.check_g3a_vmc import L,Q,build,blocking,exact_slater_energy,serial


def histograms(root):
    results=[]
    bins=12;edges=np.arange(bins+1)*L/bins
    for dirname in ('exact64-v2','exact32-v2'):
        for name in ('plane','standing','two_fermions'):
            with np.load(root/dirname/name/'samples.npz') as z:r=z['positions']
            x=r[...,0] if name!='two_fermions' else r[...,0]-r[...,2]
            x=np.mod(x,L)
            target=np.ones(bins)/bins
            if name=='standing':target+=(np.sin(2*Q*edges[1:])-np.sin(2*Q*edges[:-1]))/(4*np.pi)
            if name=='two_fermions':target-=(np.sin(Q*edges[1:])-np.sin(Q*edges[:-1]))/(2*np.pi)
            rows=[]
            for i in range(bins):
                st=blocking(((x>=edges[i])&(x<edges[i+1])).astype(float));delta=st['mean']-target[i]
                rows.append(dict(bin=i,expected=float(target[i]),actual=st['mean'],block_sem=st['conservative_block_sem'],z=float(delta/st['conservative_block_sem']),compatible=bool(abs(delta)<=4.5*st['conservative_block_sem']),plateau=st['plateau_last3']))
            results.append(dict(case=dirname+'/'+name,law='(1-cos(Q*relative_x))/L' if name=='two_fermions' else ('(1+cos(2Q*x))/L' if name=='standing' else '1/L'),bins=rows))
    return results


def precision_pairs():
    rows=[]
    for complex_output in (False,True):
        jax.config.update('jax_enable_x64',True)
        net64,_=build('slater',1,complex_output);p64=net64.init(jax.random.split(jax.random.PRNGKey(42))[1])
        r=np.array([.31,.73]);values=[]
        for x64 in (True,False):
            jax.config.update('jax_enable_x64',x64);net,_=build('slater',1,complex_output)
            p=jax.tree_util.tree_map(lambda a:jnp.asarray(np.asarray(a),dtype=jnp.float64 if x64 else jnp.float32),p64)
            d=networks.FermiNetData(positions=jnp.asarray(r),spins=jnp.ones(1),atoms=jnp.zeros((1,2)),charges=jnp.zeros(1))
            ph,la=net.apply(p,d.positions,d.spins,d.atoms,d.charges)
            es={}
            for method in ('default','folx'):
                local=PH.local_energy(net.apply,d.charges,(1,0),complex_output=complex_output,lattice=jnp.eye(2)*L,convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':method})
                es[method]=complex(local(p,None,d)[0])
            values.append(dict(x64=x64,phase=float(ph),logabs=float(la),local_energy=es,exact_energy=float(exact_slater_energy(jax.tree_util.tree_map(lambda a:np.asarray(a,dtype=float),p),complex_output))))
        delta=max(abs(values[0]['local_energy'][m]-values[1]['local_energy'][m]) for m in ('default','folx'))
        rows.append(dict(complex_output=complex_output,same_parameters_cast=True,values=serial(values),max_local_energy_difference=float(delta),compatible=bool(delta<=32*np.finfo(np.float32).eps)))
    jax.config.update('jax_enable_x64',True)
    return rows


def adam_states(root):
    names=('slater-complex64-v2','slater-real64','slater-complex32-v2','slater-real32-v2','slater-two64','screen-complex64','screen-default64','screen-complex32','psiformer-complex64','psiformer-default64')
    records=[]
    for name in names:
        directory=root/name;r=json.loads((directory/'summary.json').read_text());jax.config.update('jax_enable_x64',r['x64'])
        net,_=build(r['route'],r['particles'],r['complex_output']);n=r['particles']
        local=PH.local_energy(net.apply,jnp.zeros(1),(n,0),complex_output=r['complex_output'],lattice=jnp.eye(2)*L,convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':r.get('laplacian_method','folx')})
        def logpsi(p,x,s,a,c):
            ph,la=net.apply(p,x,s,a,c);return la+1j*ph if r['complex_output'] else la
        f=loss.make_loss(logpsi,local,complex_output=r['complex_output'])
        grad=jax.jit(jax.grad(lambda p,d:f(p,jax.random.PRNGKey(0),d)[0].real))
        with np.load(directory/'initial_params.npz',allow_pickle=True) as a:p=a['params'].item()
        opt=optax.chain(optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8,eps_root=0.),optax.scale_by_schedule(lambda t:.01*jnp.power(1+t/10000.,0.)),optax.scale(-1.))
        state=opt.init(p);errors=[]
        for path in sorted((directory/'run').glob('qmcjax_ckpt_*.npz')):
            _,d,newp,s,width=checkpoint.restore(str(path),r['batch'])
            d=jax.tree_util.tree_map(lambda a:jnp.asarray(a[0]),d)
            g=grad(p,d);_,state=opt.update(g,state,p)
            stored=jax.tree_util.tree_map(lambda a:jnp.asarray(a[0]),s)
            a=jax.tree_util.tree_leaves(state);b=jax.tree_util.tree_leaves(stored)
            if len(a)!=len(b):raise AssertionError('optimizer-state leaf count changed')
            error=max(float(jnp.max(jnp.abs(x-y))) for x,y in zip(a,b));scale=max(1.,max(float(jnp.max(jnp.abs(x))) for x in a))
            errors.append(dict(max_abs=error,scale=scale,compatible=bool(error<=32*np.finfo(np.float64 if r['x64'] else np.float32).eps*scale)))
            p=jax.tree_util.tree_map(lambda a:jnp.asarray(a[0]),newp)
        records.append(dict(case=name,steps=len(errors),state_checks=errors,all_compatible=all(e['compatible'] for e in errors)))
        jax.clear_caches()
    return records


def run(args):
    if args.output.exists():raise FileExistsError('existing report protected')
    h=histograms(args.root);pp=precision_pairs();states=adam_states(args.root)
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp,'single_hf.txt');np.savetxt(file,[[Q,0,0,0,1,0]])
        try:lj_orbitals.load_hf_table(file)
        except IndexError as e:hf_issue=dict(status='ERROR(input limitation)',error=str(e),note='valid single row is loaded as 1D; no Production fix. Exact plane used two G rows with a zero coefficient.')
        else:hf_issue=dict(status='unexpected success')
    summary=dict(version='g3a-v1',histograms=h,paired_precision=pp,adam_state_verification=states,hf_single_row=hf_issue,histogram_all_compatible=all(b['compatible'] for a in h for b in a['bins']),histogram_all_plateaus=all(b['plateau'] for a in h for b in a['bins']),paired_precision_compatible=all(r['compatible'] for r in pp),adam_states_compatible=all(r['all_compatible'] for r in states))
    with args.output.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in summary.items() if isinstance(v,bool)},indent=2))
    if not all(summary[k] for k in ('histogram_all_compatible','histogram_all_plateaus','paired_precision_compatible','adam_states_compatible')):raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('tests/g3a_validation'));p.add_argument('--output',type=Path,required=True);run(p.parse_args())
