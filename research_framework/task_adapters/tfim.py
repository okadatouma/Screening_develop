"""Reuse original LatticeWavefunction, TFIM, exact observables and CMI."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ..paths import lattice_source
from ..experiments.store import json_save,csv_save
from ..diagnostics.information import born_mi,quantum_mi
from ..diagnostics.topology import graph_ph

lattice_source()
from lattice.model import LatticeWavefunction,ModelConfig
from lattice.physics import TFIM,configurations,normalized_state,state_observables,enumerate_model
from lattice.window_research import physical_metrics,cmi_matrix,AttentionAverage,width_rows,attention_cmi_rows
from lattice.vmc import score_loss,metropolis


def build_model(c):
    p,m,w=c['physics'],c['model'],c['attention']['window']
    mode='trainable' if w['mode'].startswith('trainable') else ('smooth_fixed' if w['shape']=='sigmoid' else 'hard')
    radius='full' if w['mode']=='full' or mode!='hard' else w['width']
    return LatticeWavefunction(ModelConfig(n=p['n'],width=m['width'],heads=m['heads'],head_dim=m['head_dim'],depth=m['depth'],
        radius=radius,window_mode=mode,window_init=2. if w['width']=='full' else w['width'],window_temperature=w['temperature']))


def intervene(model,params,spec):
    target=copy.copy(model);changed=copy.deepcopy(params);kind=spec['type'];li=spec.get('layer',0);head=spec.get('head',0)
    if kind=='head_ablation':
        p=changed['layers'][li]['attention'];d=model.config.head_dim
        p['attn_output']=p['attn_output'].at[head*d:(head+1)*d].set(0)
    elif kind=='window_width':
        changed['layers'][li]['attention']['_research_override_width']=jnp.asarray(spec['width'])
        original=target.attention
        def attention(p,x,distance,**kwargs):
            if '_research_override_width' in p:
                kwargs['window']=kwargs['window'].at[head].set(jax.nn.sigmoid((p['_research_override_width']-distance)/target.config.window_temperature))
            return original(p,x,distance,**kwargs)
        target.attention=attention
    elif kind in ['remove_connections','distance_band','patch_connections']:
        n=model.config.n;scale=np.ones((model.config.heads,n,n))
        if kind=='remove_connections':
            for i,j in spec.get('edges',[(0,1),(1,0)]):scale[:,i,j]=0
        elif kind=='patch_connections':
            cut=n//2;scale[:,:cut,cut:]=0;scale[:,cut:,:cut]=0
        else:scale[:,np.asarray(model.distance)>spec.get('radius',1.)]=0
        changed['layers'][li]['attention']['_edge_scale']=jnp.asarray(scale)
    else:raise ValueError('incompatible TFIM intervention')
    return target,changed


class TFIMTask:
    def __init__(self,config):
        self.c=config;self.model=build_model(config)
        p=config['physics'];self.h=TFIM(p['n'],p['J'],p['J']*p['g'])

    def run(self,store,resume=False,until=None):
        c=self.c;t=c['training'];model=self.model;n=c['physics']['n']
        spins=jnp.asarray(configurations(n));energy0,ed,residual=self.h.ground_state();exact=state_observables(ed,self.h)
        probe=np.asarray(spins)[np.random.default_rng(917).choice(len(spins),min(c['evaluation']['probe_samples'],len(spins)),replace=False)]
        store.input(spins=spins,ed=ed,probe=probe)
        optimizer=optax.chain(optax.clip_by_global_norm(t['gradient_clip']),optax.adam(t['learning_rate']))
        if resume and (store.path/'latest.json').exists():
            start,state=store.restore();params,opt,key,walkers=(state[k] for k in ['params','optimizer','key','walkers']);store.trim_history(start)
        else:
            key,initkey,walkkey=jax.random.split(jax.random.PRNGKey(c['experiment']['seed']),3)
            params=model.init(initkey);opt=optimizer.init(params);start=0
            walkers=2*jax.random.bernoulli(walkkey,shape=(t['walkers'],n)).astype(jnp.int8)-1
            if t['method']=='vmc':
                # Honor the configured pre-training equilibration on a fresh run.
                walkers,key,_=jax.jit(lambda p,z,k:metropolis(model,p,z,k,t['burn_in']*n))(params,walkers,key)
            store.save_checkpoint(0,dict(params=params,optimizer=opt,key=key,walkers=walkers))
        initial=model.init(jax.random.split(jax.random.PRNGKey(c['experiment']['seed']),3)[1])
        def normalized(p):
            logs=model.batch(p,spins);weights=jax.nn.softmax(2*logs)
            return logs,weights
        @jax.jit
        def update(params,opt,key,walkers):
            if t['method']=='exact':
                z=spins;_,prob=normalized(params)
            else:
                walkers,key,_=metropolis(model,params,walkers,key,t['sweeps']*n)
                z=walkers;prob=None
            energies=self.h.local_energy(model.batch,params,z)
            def loss(p):
                return score_loss(model,p,z,energies,prob)+c['attention']['window']['regularization']*model.window_cost(p)
            grad=jax.grad(loss)(params);updates,opt=optimizer.update(grad,opt,params);params=optax.apply_updates(params,updates)
            return params,opt,key,walkers,jnp.mean(energies) if prob is None else jnp.dot(prob,energies)
        def evaluate(p):
            psi=enumerate_model(model,p,batch_size=min(128,len(spins)))
            result,obs=physical_metrics(psi,ed,self.h,exact)
            return result,psi,obs
        stop=t['steps'] if until is None else until
        if not start<=stop<=t['steps']:raise ValueError('until outside checkpoint/configured interval')
        if start==0:
            before,_,_=evaluate(params);store.log(dict(step=0,**before,widths=np.stack([np.asarray(model.widths(l)) for l in params['layers']]).tolist()))
        for step in range(start+1,stop+1):
            params,opt,key,walkers,mc=update(params,opt,key,walkers)
            if not np.isfinite(float(mc)):raise FloatingPointError('nonfinite TFIM energy')
            if step%max(1,t['steps']//10)==0 or step==stop:
                value,_,_=evaluate(params);store.log(dict(step=step,**value,widths=np.stack([np.asarray(model.widths(l)) for l in params['layers']]).tolist(),parity_in_loss=False))
            if step%t['checkpoint_every']==0 or step==stop:store.save_checkpoint(step,dict(params=params,optimizer=opt,key=key,walkers=walkers))
        if stop<t['steps']:return dict(paused_at=stop)
        final,psi,obs=evaluate(params)
        np.savez_compressed(store.path/'states.npz',variational=psi,ed=ed,spins=np.asarray(spins))
        for label,state in [('ED',ed),('NN',psi)]:
            np.savez_compressed(store.path/'diagnostics'/f'{label}_information.npz',Born_MI=born_mi(state,n),
                               Born_CMI=cmi_matrix(state,n),quantum_MI=quantum_mi(state,n))
        # All models/stages see the same exact uniform configurations for internal probes.
        for label,p in [('before',initial),('after',params)]:
            _,aux=jax.jit(jax.vmap(lambda z:model.apply(p,z,diagnostic=True)))(jnp.asarray(probe))
            arrays={f'layer{i}_{key}':np.asarray(v) for i,a in enumerate(aux) for key,v in a.items() if v is not None}
            np.savez_compressed(store.path/'diagnostics'/f'{label}_attention.npz',**arrays)
            if c['evaluation']['ph']:
                a=aux[-1]
                hc=np.asarray(a['head_contribution']).mean((0,2));hc=hc/np.maximum(np.linalg.norm(hc,axis=-1,keepdims=True),1e-12)
                graph_ph(store.path/'diagnostics',label,dict(QK=np.asarray(a['similarity']).mean((0,1)),Alpha=np.asarray(a['alpha']).mean((0,1)),
                     Window=np.asarray(a['window']).mean((0,1)),Screening=np.asarray(a['aggregation_weights']).mean((0,1)),head_output=(hc@hc.T+1)/2))
        attention=AttentionAverage(model)(params,np.asarray(spins),psi**2,batch_size=min(64,len(spins)))
        widths=width_rows(model,params,attention,dict(**final,measure='exact Born'))
        csv_save(store.path/'diagnostics/head_widths.csv',widths)
        csv_save(store.path/'diagnostics/attention_cmi.csv',attention_cmi_rows(attention,cmi_matrix(ed,n),cmi_matrix(psi,n),np.asarray(model.distance),{}))
        changes=[]
        if c['evaluation']['interventions']:
            specs=[dict(type='head_ablation',head=h) for h in range(model.config.heads)]+[dict(type='remove_connections'),dict(type='distance_band',radius=1.),dict(type='window_width',width=.75),dict(type='patch_connections')]
            for spec in specs:
                from ..interventions import apply_intervention
                changed,p=apply_intervention(c,model,params,spec)
                state=enumerate_model(changed,p,batch_size=min(128,len(spins)))
                value,_=physical_metrics(state,ed,self.h,exact)
                changes.append(dict(intervention=spec,**value,delta_energy=value['energy']-final['energy'],
                                    state_expectation='exact enumeration of INTERVENED state, renormalized'))
            json_save(store.path/'interventions.json',changes)
        summary=dict(task='lattice_tfim',**final,ed_energy=energy0,ed_residual=residual,
                     steps=stop,training_method=t['method'],parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(params)),
                     widths=np.stack([np.asarray(model.widths(l)) for l in params['layers']]),
                     parity_in_loss=False,observables=obs,exact=exact,n=n)
        store.complete(summary);return summary
