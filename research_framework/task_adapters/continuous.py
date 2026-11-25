"""Original fermion factories/Hamiltonian/Metropolis; fresh Born sampling after interventions."""
import copy
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from periodicwave import networks,mcmc
from periodicwave.pbc import lambda_jellium
from ..factory import make_wavefunction
from ..experiments.store import json_save,csv_save
from ..diagnostics.topology import graph_ph


def synthetic_hf(path,n,lattice):
    """Plane-wave reference, NOT a self-consistent HF solution or old checkpoint."""
    modes=sorted([(i,j) for i in range(-n,n+1) for j in range(-n,n+1)],key=lambda z:(z[0]**2+z[1]**2,z))[:n]
    k=2*np.pi*np.asarray(modes)@np.linalg.inv(lattice)
    table=np.c_[k,np.zeros((n,2)),np.ones(n),np.zeros(n)]
    with Path(path).open('x') as f:np.savetxt(f,table,fmt='%.17g')
    return str(path)


class ContinuousTask:
    def __init__(self,c):self.c=c;self.wave=None

    def build(self,folder):
        p=self.c['physics'];reference=p['hf_path']
        if reference is None:
            path=folder/'reference_plane_waves.txt'
            reference=str(path) if path.exists() else synthetic_hf(path,p['n'],p['lattice'])
        self.wave=make_wavefunction(self.c['wavefunction'],'continuous_electrons',full_config=self.c,reference_path=reference)
        self.local=lambda_jellium.local_energy(self.wave.production_apply,self.wave.charges,(p['n'],0),
                    complex_output=True,lattice=self.wave.lattice,lam=p['lam'],delta=p['delta'],potential_type=p['potential'],
                    potential_kwargs=dict(interaction_energy_scale=p['rs']),kinetic_kwargs=dict(laplacian_method='default'),return_components=False)
        self.energy_batch=jax.jit(jax.vmap(lambda pp,x,s:self.local(pp,jax.random.PRNGKey(0),networks.FermiNetData(
                    positions=x,spins=s,atoms=self.wave.atoms,charges=self.wave.charges))[0],in_axes=(None,0,0)))
        self.log_batch=jax.vmap(lambda pp,x,s,a,q:self.wave.production_apply(pp,x,s,a,q)[1],in_axes=(None,0,0,0,0))
        self.sample=jax.jit(self.sampler,static_argnums=3)

    def initial_walkers(self,key,count):
        n=self.c['physics']['n'];fraction=jax.random.uniform(key,(count,n,2),minval=.001,maxval=.999)
        positions=(fraction@self.wave.lattice.T).reshape(count,-1)
        # All-plus starts on the nonzero plane-wave reference support. Spin moves remain enabled.
        return networks.FermiNetData(positions=positions,spins=jnp.ones((count,n)),
            atoms=jnp.broadcast_to(self.wave.atoms,(count,1,2)),charges=jnp.zeros((count,1)))

    def sampler(self,params,data,key,steps):
        lp=2*self.log_batch(params,data.positions,data.spins,data.atoms,data.charges)
        def move(_,carry):
            data,key,lp,count=carry
            data,key,lp,count=mcmc.mh_update(params,self.log_batch,data,key,lp,count,stddev=.35,ndim=2)
            data,key,lp,count=mcmc.mh_spin_flip(params,self.log_batch,data,key,lp,count)
            fraction=data.positions.reshape(-1,2)@jnp.linalg.inv(self.wave.lattice).T
            data=networks.FermiNetData(positions=((fraction%1)@self.wave.lattice.T).reshape(data.positions.shape),
                                      spins=data.spins,atoms=data.atoms,charges=data.charges)
            return data,key,lp,count
        data,key,_,accept=jax.lax.fori_loop(0,steps,move,(data,key,lp,jnp.asarray(0)))
        return data,key,accept/jnp.maximum(2*steps*len(data.spins),1)

    def measured_arrays(self,params,seed,samples=None):
        """Equivalent measurement with arrays retained for physical diagnostics."""
        e=self.c['evaluation'];count=e['chains'];steps=e['samples'] if samples is None else samples
        key,initkey=jax.random.split(jax.random.PRNGKey(seed));data=self.initial_walkers(initkey,count)
        move=self.sample;data,key,_=move(params,data,key,e['burn_in'])
        xx=[];ss=[];ee=[]
        for _ in range(steps):
            data,key,_=move(params,data,key,e['thinning'])
            ee.append(np.asarray(self.energy_batch(params,data.positions,data.spins)));xx.append(np.asarray(data.positions));ss.append(np.asarray(data.spins))
        energies=np.asarray(ee)
        if not np.isfinite(energies).all():raise FloatingPointError('nonfinite local-energy measurement')
        blocks=np.asarray([v.mean() for v in np.array_split(energies.real,min(8,len(energies)))])
        result=dict(energy=float(energies.real.mean()),energy_imag=float(energies.imag.mean()),local_variance=float(energies.real.var()),
                    block_sem=float(blocks.std(ddof=1)/np.sqrt(len(blocks))) if len(blocks)>1 else None,
                    samples=int(energies.size),chains=count,sampling='fresh Born chains; explicit burn-in; pilot MC not certified converged')
        return result,np.asarray(xx).reshape(-1,self.c['physics']['n']*2),np.asarray(ss).reshape(-1,self.c['physics']['n']),energies

    def diagnostics(self,store,label,params,x,s):
        wave=self.wave;count=min(len(x),self.c['evaluation']['probe_samples']);x=x[:count];s=s[:count]
        logs=np.asarray(jax.vmap(lambda xx,ss:wave.apply(params,xx,ss)[1])(jnp.asarray(x),jnp.asarray(s)))
        valid=np.isfinite(logs)
        if not valid.all():
            json_save(store.path/'diagnostics'/f'{label}_support_mask.json',dict(valid=valid,input_count=len(valid),
                note='zero-amplitude nodes excluded from derivative diagnostics; this subset is not a matched full-input comparison'))
            x=x[valid];s=s[valid];count=len(x)
        if count==0:return
        evaluate=jax.jit(jax.vmap(lambda xx,ss:wave.internal(params,xx,ss)))
        features,trace=evaluate(jnp.asarray(x),jnp.asarray(s))
        arrays=dict(features=np.asarray(features),positions=x,spins=s);last=trace[-1]['attention']
        from screening_analysis.head_contrib import projected_heads
        for li,layer in enumerate(trace):
            for key,value in layer['attention'].items():
                if value is not None:arrays[f'layer{li}_{key}']=np.asarray(value)
            ap=params['layers']['attention'][li]
            contribution,_=jax.vmap(lambda gate:projected_heads(gate,ap['attn_output'],self.c['model']['heads'],jnp.exp(ap['out_scale'])))(layer['attention']['gate_output'])
            arrays[f'layer{li}_head_contribution']=np.asarray(contribution)
            norm=np.sum(np.asarray(contribution)**2,axis=(-1,-2))
            arrays[f'layer{li}_head_reinforcement']=np.sum(np.asarray(contribution).sum(-2)**2,-1)/np.maximum(norm,1e-20)
        if self.c['analysis']['phase_sensitivity']:
            h=self.c['model']['heads'];n=self.c['physics']['n']
            def sensitivity(xx,ss):
                def phase(log_scale):
                    changed=dict(params);layers=dict(params['layers']);att=list(layers['attention']);att[-1]=dict(att[-1],_edge_scale=jnp.exp(log_scale));layers['attention']=att;changed['layers']=layers
                    return wave.apply(changed,xx,ss)[0]
                return jax.grad(phase)(jnp.zeros((h,n,n)))
            sens=np.asarray(jax.jit(jax.vmap(sensitivity))(jnp.asarray(x),jnp.asarray(s)))
            arrays['phase_log_edge_sensitivity']=sens
        np.savez_compressed(store.path/'diagnostics'/f'{label}_internal.npz',**arrays)
        if self.c['evaluation']['ph']:
            graphs=dict(QK=np.asarray(last['similarity']).mean((0,1)),Alpha=np.asarray(last['alpha']).mean((0,1)),
                        Window=np.asarray(last['window']).mean((0,1)),Screening=np.asarray(last['aggregation_weights']).mean((0,1)))
            if 'phase_log_edge_sensitivity' in arrays:
                sense=abs(arrays['phase_log_edge_sensitivity']).mean((0,1));graphs['phase_sensitivity']=sense/max(float(sense.max()),1e-20)
            hc=arrays[f'layer{len(trace)-1}_head_contribution'].mean((0,2))
            hc=hc/np.maximum(np.linalg.norm(hc,axis=-1,keepdims=True),1e-12);graphs['head_output']=(hc@hc.T+1)/2
            graph_ph(store.path/'diagnostics',label,graphs)
        # Reciprocal modes of both learned candidate set and physical crystal shell.
        modes=np.unique(np.r_[wave.embedding.modes,wave.embedding.crystal_modes],axis=0)
        frac=x.reshape(count,-1,2)@np.linalg.inv(self.c['physics']['lattice']).T
        phase=2*np.pi*np.einsum('bnd,qd->bnq',frac,modes)
        rho=np.exp(1j*phase).sum(1);structure=(abs(rho)**2/self.c['physics']['n']).mean(0)
        similarity=np.asarray(last['similarity']).mean(1);weights=np.asarray(last['aggregation_weights']).mean(1)
        delta=phase[:,:,None,:]-phase[:,None,:,:];cos=np.cos(delta)
        mask=1-np.eye(self.c['physics']['n']);same=s[:,:,None]==s[:,None,:]
        spectrum=[]
        for qi,q in enumerate(modes):
            denominator=np.sum(weights*mask)
            spectrum.append(dict(qx=int(q[0]),qy=int(q[1]),structure_factor_probe=float(structure[qi]),
                QK_Fourier=float(np.sum(similarity*cos[...,qi]*mask)/max(np.sum(abs(similarity)*mask),1e-20)),
                Screening_Fourier=float(np.sum(weights*cos[...,qi]*mask)/max(denominator,1e-20)),
                Spin_Fourier=float(np.sum(weights*cos[...,qi]*s[:,:,None]*s[:,None,:]*mask)/max(denominator,1e-20))))
        csv_save(store.path/'diagnostics'/f'{label}_fourier.csv',spectrum)
        selection={}
        for name,sel in [('same',same),('opposite',~same)]:
            sel=sel*mask;selection[name]=float(np.sum(similarity*sel)/sel.sum()) if sel.sum() else None
        json_save(store.path/'diagnostics'/f'{label}_spin_selection.json',selection)

    def run(self,store,resume=False,until=None):
        self.build(store.path);c=self.c;t=c['training'];wave=self.wave
        rng=np.random.default_rng(917);probe_x=(rng.uniform(.01,.99,(c['evaluation']['probe_samples'],c['physics']['n'],2))@np.asarray(wave.lattice).T).reshape(c['evaluation']['probe_samples'],-1)
        probe_s=rng.choice([-1.,1.],(len(probe_x),c['physics']['n']))
        store.input(probe_positions=probe_x,probe_spins=probe_s)
        optimizer=optax.chain(optax.clip_by_global_norm(t['gradient_clip']),optax.adam(t['learning_rate']))
        key,initkey,walkkey=jax.random.split(jax.random.PRNGKey(c['experiment']['seed']),3)
        initial=wave.init(initkey)
        if resume and (store.path/'latest.json').exists():
            start,state=store.restore();params,opt,key,data=(state[k] for k in ['params','optimizer','key','walkers']);store.trim_history(start)
        else:
            params=initial;opt=optimizer.init(params);start=0;data=self.initial_walkers(walkkey,t['walkers'])
            data,key,_=jax.jit(self.sampler,static_argnums=3)(params,data,key,t['burn_in'])
            store.save_checkpoint(0,dict(params=params,optimizer=opt,key=key,walkers=data))
        @jax.jit
        def update(params,opt,data,key):
            data,key,accept=self.sampler(params,data,key,t['sweeps'])
            energies=self.energy_batch(params,data.positions,data.spins);center=energies.mean()
            coefficients=jax.lax.stop_gradient(energies-center)
            def loss(p):
                phase,logabs=jax.vmap(wave.production_apply,in_axes=(None,0,0,0,0))(p,data.positions,data.spins,data.atoms,data.charges)
                return 2*jnp.mean(coefficients.real*logabs+coefficients.imag*phase)+wave.penalty(p)
            grad=jax.grad(loss)(params);change,opt=optimizer.update(grad,opt,params);params=optax.apply_updates(params,change)
            return params,opt,data,key,center,accept,optax.global_norm(grad)
        stop=t['steps'] if until is None else until
        if not start<=stop<=t['steps']:raise ValueError('until out of range')
        for step in range(start+1,stop+1):
            update_started=time.monotonic()
            params,opt,data,key,energy,accept,grad=update(params,opt,data,key)
            if not np.isfinite(np.asarray([energy.real,energy.imag,accept,grad])).all():raise FloatingPointError('nonfinite VMC update')
            row=dict(step=step,energy=float(energy.real),energy_imag=float(energy.imag),acceptance=float(accept),gradient_norm=float(grad),
                     update_seconds=time.monotonic()-update_started)
            if wave.embedding.learned:row['fourier_amplitudes']=np.asarray(params['layers']['embedding_extra']['fourier_amplitude']).tolist()
            row['widths']=[np.asarray(wave.attention.widths(ap,c['model']['heads'],float(cs_distance(wave.lattice)))).tolist() for ap in params['layers']['attention']]
            store.log(row)
            if step%t['checkpoint_every']==0 or step==stop:store.save_checkpoint(step,dict(params=params,optimizer=opt,key=key,walkers=data))
        if stop<t['steps']:return dict(paused_at=stop)
        self.diagnostics(store,'before',initial,probe_x,probe_s);self.diagnostics(store,'after',params,probe_x,probe_s)
        before,bx,bs,be=self.measured_arrays(initial,c['experiment']['seed']+9000)
        final,x,s,energies=self.measured_arrays(params,c['experiment']['seed']+10000)
        np.savez_compressed(store.path/'evaluation.npz',positions=x,spins=s,local_energy=energies,initial_energy=be)
        self.diagnostics(store,'born_after',params,x,s)
        if wave.twisted:
            from ..wavefunctions.twist import twist_overlaps
            twists=[[0.,0.],[.2,0.],[.2,.2],[0.,.2]];twist_rows=[]
            if c['physics']['n']<=2:
                for mesh in [3,4]:
                    value,states=twist_overlaps(wave,params,twists,mesh)
                    np.savez_compressed(store.path/'diagnostics'/f'twist_overlap_mesh{mesh}.npz',overlap=value.pop('overlap'),states=states,twists=twists)
                    twist_rows.append(dict(mesh=mesh,**value,chern_interpretable=False))
                json_save(store.path/'diagnostics/twist_overlap_summary.json',dict(rows=twist_rows,
                    gauge='periodic phi_theta; physical quasi-periodic Psi includes exp(i kappa.sum r)',
                    limitation='small rectangle, finite quadrature; full twist-torus sewing and gap tracking not performed'))
        if c['wavefunction']['type'] in ['hf_nn_jastrow','hf','hf_jastrow','twist_dependent']:
            no_nn=copy.deepcopy(params);no_nn['orbital'][0]['w']=jnp.zeros_like(no_nn['orbital'][0]['w'])
            orbital=wave.network.orbitals(params,jnp.asarray(x[0]),jnp.asarray(s[0]),wave.atoms,wave.charges)[0]
            hf_j=wave.network.orbitals(no_nn,jnp.asarray(x[0]),jnp.asarray(s[0]),wave.atoms,wave.charges)[0]
            json_save(store.path/'diagnostics/hf_correction.json',dict(relative_orbital_correction=float(jnp.linalg.norm(orbital-hf_j)/jnp.maximum(jnp.linalg.norm(hf_j),1e-12)),
                    jastrow_retained=c['wavefunction']['jastrow'],pure_hf_reference=not c['wavefunction']['jastrow'],
                    note='NN orbital correction zeroed at same input. Jastrow, if enabled, remains; not pure HF in that case.'))
        changes=[]
        if c['evaluation']['interventions']:
            specs=[dict(type='head_ablation',head=h) for h in range(c['model']['heads'])]+[dict(type='distance_band',radius=1.),dict(type='window_width',width=1.),dict(type='remove_spin')]
            if c['wavefunction']['complex_phase']:specs.append(dict(type='zero_phase_channel'))
            if wave.embedding.onehot:specs.append(dict(type='replace_spin'))
            if wave.embedding.high:specs.append(dict(type='remove_six_theta'))
            if wave.embedding.learned:specs.append(dict(type='remove_fourier'))
            for index,spec in enumerate(specs):
                from ..interventions import apply_intervention
                _,p=apply_intervention(c,wave,params,spec)
                value,xx,ss,ee=self.measured_arrays(p,c['experiment']['seed']+11000+index,samples=max(8,c['evaluation']['samples']//2))
                changes.append(dict(intervention=spec,**value,delta_energy=value['energy']-final['energy'],
                                    baseline_sem=final['block_sem'],resampled=True))
                np.savez_compressed(store.path/'diagnostics'/f'intervention_{index}.npz',positions=xx,spins=ss,local_energy=ee)
            json_save(store.path/'interventions.json',changes)
        # Born structure factor uses ALL fresh evaluation inputs, not the uniform probe.
        modes=np.unique(np.r_[wave.embedding.modes,wave.embedding.crystal_modes],axis=0)
        phase=2*np.pi*np.einsum('bnd,qd->bnq',x.reshape(-1,c['physics']['n'],2)@np.linalg.inv(c['physics']['lattice']).T,modes)
        sq=np.mean(abs(np.exp(1j*phase).sum(1))**2/c['physics']['n'],0)
        summary=dict(task='continuous_electrons',**final,before=before,steps=stop,n=c['physics']['n'],
                     parameter_count=sum(v.size for v in jax.tree_util.tree_leaves(params)),
                     fourier_modes=modes,structure_factor=sq,
                     hf_reference='supplied table' if c['physics']['hf_path'] else 'synthetic plane-wave table; not self-consistent HF',
                     jastrow_active=c['wavefunction']['jastrow'] and c['wavefunction']['type']!='hf',
                     pure_hf=c['wavefunction']['type']=='hf',
                     physical_phase_certified=False,warning='small VMC pilot; energies/interventions need longer chains and independent seeds')
        store.complete(summary);return summary


def cs_distance(lattice):
    from periodicwave.CustomScreenformer import _max_periodic_distance
    return _max_periodic_distance(lattice)
