"""Matched-seed VMC ablation of three DIFFERENT regularizers on free N4 fermions.

Uses production Network/MCMC/Hamiltonian/VMC gradients/Optax update functions.
This controlled small-system experiment has no WC/AHC labels. All outputs new.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from periodicwave import CustomScreenformer as CS,networks,envelopes,loss,mcmc,train_screen
from periodicwave.pbc import feature_layer,hamiltonian
from periodicwave import topology_regularization as topo
from periodicwave.window_experiments.run import save_state


SETTINGS=dict(TRIM_MAX=2.,TRIM_POW=2,WIN_SHAPE='cos',FREEZE_W=False,CLIP_R=None,CLIP_BETA=.3,
              jax_enable_x64=True,jax_default_matmul_precision='float32')


def build(batch=16):
    N=4;L=jnp.eye(2)*4
    options=dict(num_layers=1,num_heads=2,attn_dim=4,value_dim=4,mlp_dim=16,
        num_perceptrons_per_layer=1,use_layer_norm=False,mlp_activation_fct='GELU',
        use_softmask=True,use_gate=True,use_tanh_norm=True,use_fixed_value=False,
        use_output_scale=True,init_output_scale=0.,init_trim_r=1.9,init_softmask_w_factor=.8,init_softmask_w_spread=2.)
    net=CS.make_fermi_net((N,0),jnp.zeros(1),ndim=2,determinants=1,
        feature_layer=feature_layer.make_pbc_feature_layer(natoms=1,nspins=(N,0),ndim=2,lattice=L,include_r_ae=False),
        envelope=envelopes.make_null_envelope(),jastrow='NONE',jastrow_kwargs={'ndim':2},
        bias_orbitals=True,complex_output=True,pbc_lattice=L,**options)
    diagnostic=CS.make_screenformer_diagnostic((N,0),net.options)
    batch_logabs=jax.vmap(lambda p,x,s,a,c:net.apply(p,x,s,a,c)[1],in_axes=(None,0,0,0,0))
    sample=jax.jit(mcmc.make_mcmc_step(batch_logabs,batch,steps=2,ndim=2,blocks=1))
    energy=hamiltonian.local_energy(net.apply,jnp.zeros(1),(N,0),complex_output=True,lattice=L,
        convergence_radius=2,potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':'default'})
    def logpsi(p,x,s,a,c):
        phase,logabs=net.apply(p,x,s,a,c);return logabs+1j*phase
    base=loss.make_loss(logpsi,energy,complex_output=True,clip_local_energy=0.,local_energy_chunk=4)
    evaluate=jax.jit(base)
    cfg=dict(system=dict(electrons=[N,0],ndim=2,pbc_lattice=np.asarray(L).tolist(),pseudospin=False,
            make_local_energy_fn='periodicwave.pbc.hamiltonian.local_energy',make_local_energy_kwargs=dict(lattice=np.asarray(L).tolist(),
            convergence_radius=2,potential_type='Coulomb',potential_kwargs={'interaction_energy_scale':0.},kinetic_kwargs={'laplacian_method':'default'})),
        network=dict(network_type='CustomScreenformer',complex=True,determinants=1,bias_orbitals=True,jastrow='NONE',jastrow_kwargs={'ndim':2},
            make_feature_layer_fn='periodicwave.pbc.feature_layer.make_pbc_feature_layer',make_feature_layer_kwargs=dict(lattice=np.asarray(L).tolist(),include_r_ae=False),
            make_envelope_fn='periodicwave.envelopes.make_null_envelope',make_envelope_kwargs={},CustomScreenformer=options),
        screening_attention_settings=SETTINGS,phase_status='unknown; free N4 numerical validation, not WC/AHC')
    return net,diagnostic,sample,base,evaluate,cfg


def data_from_key(key,batch=16):
    return networks.FermiNetData(positions=jax.random.uniform(key,(batch,8),minval=0.,maxval=4.),
        spins=jnp.ones((batch,4)),atoms=jnp.zeros((batch,1,2)),charges=jnp.zeros((batch,1)))


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','float32')
    for key in ('TRIM_MAX','TRIM_POW','WIN_SHAPE','FREEZE_W','CLIP_R','CLIP_BETA'):setattr(CS,key,SETTINGS[key])
    net,diagnostic,sample,base,evaluate,cfg=build(args.batch)
    common=data_from_key(jax.random.PRNGKey(9001),args.batch)
    stability={mode:jax.jit(topo.make_regularized_loss(lambda p,k,d:(jnp.asarray(0.),None),net,diagnostic,
                        dict(mode=mode,gamma=1.,sampling_score=False,max_samples=4))) for mode in ('persistence','laplacian','weights')}
    variants=[('off',0.)]+[(mode,gamma) for mode in ('persistence','laplacian','weights') for gamma in (.1,1.)]
    rows=[]
    for mode,gamma in variants:
        config=topo.configuration(dict(mode=mode,gamma=gamma,max_samples=4))
        wrapped=topo.make_regularized_loss(base,net,diagnostic,config)
        optimizer=optax.adam(.003)
        update=jax.jit(train_screen.make_opt_update_step(wrapped,optimizer))
        for seed in (11,23,37):
            identifier=f'{mode}-g{gamma:g}-seed{seed}';folder=out/identifier;folder.mkdir()
            key,kp,kd=jax.random.split(jax.random.PRNGKey(seed),3);params=net.init(kp);data=data_from_key(kd,args.batch);state=optimizer.init(params)
            run_cfg={**cfg,'seed':seed,'optim':dict(optimizer='adam',iterations=args.iterations,topology_regularization=config),
                      'log':dict(save_path=str(folder))}
            (folder/'runtime_config.json').write_text(json.dumps(run_cfg,indent=2))
            manifest=dict(id=identifier,seed=seed,seed_group=f'seed{seed}',label='unknown',label_verified=False,
                model='free N4 polarized fermions, L=4 square',regularizer=config,iterations=args.iterations,
                exact_ground_energy=3*(2*np.pi/4)**2/2,energy_units='total code units',init='random Screenformer; same seed/initial samples across variants')
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
            for _ in range(args.burn):key,k=jax.random.split(key);data,_=sample(params,data,k,.6)
            history=[];status='complete';nan_steps=0
            fields=['step','energy','objective','penalty','variance_abs_complex','variance_real','position_acceptance','seconds']
            with (folder/'stats.csv').open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for t in range(args.iterations):
                    start=time.perf_counter();key,ks,ke=jax.random.split(key,3);data,accept=sample(params,data,ks,.6)
                    proposed,new_state,objective,aux=update(params,data,state,ke)
                    jax.block_until_ready(objective)
                    valid=all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves((proposed,new_state,objective,aux.local_energy)))
                    if not valid:nan_steps+=1;status='failed_nonfinite';break
                    params,state=proposed,new_state;energy=float(aux.energy.real);el=np.asarray(aux.local_energy)
                    row=dict(step=t,energy=energy,objective=float(objective),penalty=(float(objective)-energy)/gamma if gamma else 0.,
                        variance_abs_complex=float(np.var(el)),variance_real=float(el.real.var()),position_acceptance=float(accept),seconds=time.perf_counter()-start)
                    writer.writerow(row);f.flush();history.append(row)
                    if t in (0,args.iterations//2-1,args.iterations-1):save_state(folder,t,data,params,state,.6,key)
            measured=None
            if status=='complete':
                key,kd=jax.random.split(key);fresh=data_from_key(kd,args.batch)
                for _ in range(args.eval_burn):key,k=jax.random.split(key);fresh,_=sample(params,fresh,k,.6)
                samples=[]
                for _ in range(args.eval_sweeps):
                    key,ks,ke=jax.random.split(key,3);fresh,_=sample(params,fresh,ks,.6)
                    _,aux=evaluate(params,ke,fresh);samples.append(np.asarray(aux.local_energy))
                el=np.asarray(samples);chain_means=el.real.mean(axis=0)
                measurement=dict(energy=float(el.real.mean()),imaginary_energy=float(el.imag.mean()),
                    variance_abs_complex=float(np.mean(abs(el-el.mean())**2)),variance_real=float(el.real.var()),
                    independent_chain_sem=float(chain_means.std(ddof=1)/np.sqrt(len(chain_means))),
                    sweeps=args.eval_sweeps,chains=args.batch,burn=args.eval_burn)
                measured=measurement
                np.savez_compressed(folder/'measurement.npz',local_energy=el)
                invariant={name:float(fn(params,jax.random.PRNGKey(9002),common)[0]) for name,fn in stability.items()}
            else:invariant={}
            summary=dict(**manifest,status=status,nan_steps=nan_steps,nan_rate_observed=nan_steps/max(1,len(history)+nan_steps),
                measurement=measured,common_configuration_stability=invariant,
                median_update_seconds_excluding_first3=float(np.median([r['seconds'] for r in history[3:]])) if len(history)>3 else None,
                first_step_seconds=history[0]['seconds'] if history else None,training_energy_final=history[-1]['energy'] if history else None,
                convergence_status='short finite-budget trajectory, not independently converged')
            (folder/'summary.json').write_text(json.dumps(summary,indent=2));rows.append(summary)
            print(identifier,status,None if measured is None else measured['energy'],flush=True)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    (out/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path(topo.__file__),Path(train_screen.__file__))},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=64);parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--burn',type=int,default=32);parser.add_argument('--eval-burn',type=int,default=32);parser.add_argument('--eval-sweeps',type=int,default=32)
    run(parser.parse_args())
