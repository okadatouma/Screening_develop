"""Small continuous-system study with independent energy-equivalence checks.

Reuses the original config, Screenformer, Hamiltonian, VMC gradient and MCMC.
All changes to the legacy trainer are process-local adapters for this study.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    from screening_analysis.physical_phase_diagnostics import write_json
    def materialize(v):
        if isinstance(v,dict):return {k:materialize(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [materialize(x) for x in v]
        if hasattr(v,"shape") and hasattr(v,"dtype"):return materialize(np.asarray(v).tolist())
        if isinstance(v,Path):return str(v)
        return v
    write_json(path, materialize(value))


def prepare(destination):
    destination=Path(destination).resolve(); destination.mkdir(parents=True,exist_ok=False)
    cases=[]
    for n in (9,16):
        for rs, candidate, initial in ((4.,"liquid_candidate","plane"),(25.,"solid_candidate","wc")):
            for seed in (101,202,303):
                variants=[("learned",1.),("full",1.)]
                if n==9:variants += [("learned",.5),("learned",2.),("fixed",1.)]
                for mode, factor in variants:
                    cases.append(dict(n=n,rs=rs,lam=.72,candidate=candidate,initialization=initial,
                        seed=seed,mode=mode,width_factor=factor))
                if n==9:
                    for mode in ("learned","full"):
                        cases.append(dict(n=n,rs=rs,lam=.72,candidate=candidate,
                            initialization="wc" if initial=="plane" else "plane",
                            seed=seed,mode=mode,width_factor=1.))
    for i,c in enumerate(cases):
        c["id"]=f'N{c["n"]}_rs{c["rs"]:g}_{c["initialization"]}_{c["mode"]}_w{c["width_factor"]:g}_s{c["seed"]}'
        c["index"]=i
    protocol=dict(schema="continuous-converged-window-v1",cases=cases,
        energy_tolerance_Ha_per_electron=5.e-5,minimum_stage_count=4,max_updates=400000,
        initial_stages=[dict(updates=30000,lr=.001),dict(updates=30000,lr=.0003),
                        dict(updates=30000,lr=.0001),dict(updates=30000,lr=.00003)],
        extension_updates=40000,extension_lr=.00003,challenge_updates=10000,challenge_lr=.0001,
        training=dict(batch=128,layers=1,heads=2,width=32,attn_dim=8,value_dim=8,determinants=1,
                      mcmc_steps=20,burn=512,spin_flips=3,global_hops=1,dtype="float64"),
        measurement=dict(chains=16,times=2048,max_times=16384,burn=1024,steps=20,
                         min_ess=400,max_rhat=1.05,max_structure_rhat=1.1),
        hf=dict(gcut=2.01,max_iterations=2000,mix=.3,tolerance=1.e-7),
        policy="Phase candidate labels only; width/physical length equality is a hypothesis. No width or topology penalty.")
    dump(destination/"plan.json",protocol)
    source=destination/"source";source.mkdir()
    for name in ("periodicwave","screening_analysis"):
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns("__pycache__","results*","topology_*20*","*.npz","*.png","*.svg","*.pdf","*.csv","*.log"))
    shutil.copy2(ROOT/"continuous_window.job",source/"continuous_window.job")
    shutil.copy2(ROOT/"continuous_window_report.job",source/"continuous_window_report.job")
    (destination/"logs").mkdir()
    dump(destination/"source_sha256.json",{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in source.rglob("*") if p.is_file()})
    print(json.dumps(dict(root=str(destination),cases=len(cases))))


def install_training_adapters(stage, previous, case, settings):
    """Process-local adapters: no edits to existing trainer or checkpoints.

    Retain Adam NamedTuples in a scalar object, and add symmetric one-electron
    global hops to the existing local/pseudospin Metropolis kernel.
    """
    import jax
    import jax.numpy as jnp
    from periodicwave import checkpoint, mcmc, networks, train_screen
    from screening_analysis.sampler import make_hop
    real_save=checkpoint.save
    def save(path,t,data,params,opt_state,width):
        holder=np.empty((),dtype=object);holder[()]=opt_state
        return real_save(path,t,data,params,holder,width)
    checkpoint.save=save
    real_mcmc=mcmc.make_mcmc_step
    from periodicwave.pbc.lattices import _triangular_lattice_vecs_periodic_potential
    lattice=_triangular_lattice_vecs_periodic_potential(np.sqrt(2*np.pi/np.sqrt(3)),case["n"])[0]
    def make_step(batch_network,*args,**kwargs):
        local=real_mcmc(batch_network,*args,**kwargs)
        hop=make_hop(batch_network,case["n"],lattice)
        def step(params,data,key,width):
            k1,k2=jax.random.split(key)
            data,rate=local(params,data,k1,width)
            lp=2*batch_network(params,data.positions,data.spins,data.atoms,data.charges)
            pos,_,_,_=hop(params,data.positions,data.spins,data.atoms,data.charges,k2,lp,1)
            return networks.FermiNetData(positions=pos,spins=data.spins,atoms=data.atoms,charges=data.charges),rate
        return step
    mcmc.make_mcmc_step=make_step
    real_train=train_screen.train
    def train(cfg,*args,**kwargs):
        cfg.log.restore_path=str(previous) if previous else ""
        cfg.optim.lr.decay=0. # explicit constant LR within each stage
        cfg.optim.clip_local_energy=0.
        cfg.optim.reset_if_nan=False
        dump(Path(stage)/"runtime_config.json",cfg.to_dict())
        return real_train(cfg,*args,**kwargs)
    train_screen.train=train


def train_stage(spec):
    import jax
    jax.config.update("jax_enable_x64",True)
    case=spec["case"];p=spec["protocol"];t=p["training"]
    stage=Path(spec["output"])
    # Explicit seeds for each stage: legacy checkpoints contain no RNG state.
    # Optimizer/walkers continue, but no bitwise replay across stage boundaries
    # is claimed. Streams are never silently reused between stages.
    env=dict(PW_OUTPUT_DIR=str(stage),PW_LJ_HF=spec["reference"],PW_OPTIMIZER="adam",
        PW_LR=str(spec["lr"]),PW_LR_DELAY="1000000000",PW_BATCH=str(t["batch"]),
        PW_EL_CHUNK="8",PW_FLIP_BATCH="3",PW_LAYERS=str(t["layers"]),PW_HEADS=str(t["heads"]),
        PW_ATTN_DIM=str(t["attn_dim"]),PW_VALUE_DIM=str(t["value_dim"]),PW_WIDTH=str(t["width"]),
        PW_DET=str(t["determinants"]),PW_MCMC_STEPS=str(t["mcmc_steps"]),PW_BURN_IN=str(t["burn"]),
        PW_FLIPS=str(t["spin_flips"]),PW_WIN_SHAPE="full" if case["mode"]=="full" else "gaussian",
        PW_FREEZE_W="1" if case["mode"]=="fixed" else "0",PW_TRIM_POW="4",
        PW_W_LO=str(.5*np.sqrt(2*np.pi/np.sqrt(3))*case["width_factor"]/np.sqrt(np.log(2))),
        PW_W_HI=str(.8*np.sqrt(2*np.pi/np.sqrt(3))*case["width_factor"]/np.sqrt(np.log(2))),
        PW_INIT_LOCALIZE="0",PW_INIT_SPIN="random",PW_SEED=str(spec["stage_seed"]),
        PW_SEED_FROM_TAG="0",PW_TOPO_MODE="off",PW_TOPO_GAMMA="0",PW_DELTA_MULT="1",
        PW_COULOMB="1",PW_CHECK_NAN="1",PW_SAVE_MIN="10",PW_LAPL="folx")
    for key in list(os.environ):
        if key.startswith("PW_"): del os.environ[key]
    os.environ.update(env)
    install_training_adapters(stage,spec.get("previous"),case,p)
    sys.argv=["run_lambda_jellium",str(case["n"]),str(case["rs"]),str(case["lam"]),str(spec["target"]),case["id"]]
    runpy.run_module("periodicwave.configs.run_lambda_jellium",run_name="__main__")


def time_statistics(values):
    """Conservative blocking, split-chain Rhat, initial-positive-pair ESS.

    Mean is never trimmed or clipped. Both chain/time axes are retained.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or min(x.shape)<4 or not np.isfinite(x).all():raise ValueError("finite [time,chain] with >=4 entries required")
    times,chains=x.shape
    chain_mean=x.mean(0);se_chain=chain_mean.std(ddof=1)/np.sqrt(chains)
    block_se=[]
    for size in (max(1,times//64),max(1,times//32),max(1,times//16),max(1,times//8)):
        count=times//size
        b=x[:count*size].reshape(count,size,chains).mean(1).mean(1)
        block_se.append(dict(size=size,blocks=count,sem=float(b.std(ddof=1)/np.sqrt(count))))
    centered=x-x.mean(0,keepdims=True);var=float(np.mean(centered**2));tau=1.
    if var>1.e-28:
        ac=[float(np.mean(centered[:-lag]*centered[lag:])/var) for lag in range(1,min(times//2,1024))]
        for k in range(0,len(ac)-1,2):
            pair=ac[k]+ac[k+1]
            if pair<=0:break
            tau+=2*pair
    half=times//2;split=np.concatenate([x[:half].T,x[-half:].T],axis=0)
    within=float(np.mean(split.var(1,ddof=1)));between=float(half*split.mean(1).var(ddof=1))
    rhat=float(np.sqrt(((half-1)/half*within+between/half)/within)) if within>1.e-28 else (1. if between<1.e-28 else float("inf"))
    return dict(mean=float(x.mean()),sem=float(max(se_chain,*[b["sem"] for b in block_se])),
        chain_sem=float(se_chain),blocking=block_se,rhat=rhat,ess=float(x.size/max(tau,1)),iat=float(tau),
        time_samples=times,chains=chains)


def equivalent(previous,current,tolerance):
    """Require the whole approximate 95% interval inside +/- tolerance."""
    bound=abs(current["mean"]-previous["mean"])+1.96*np.hypot(current["sem"],previous["sem"])
    return bool(np.isfinite(bound) and bound<=tolerance),float(bound)


def check_convergence(history,protocol,challenge=False):
    if len(history)<3:return dict(converged=False,reason="need_three_fixed_checkpoint_measurements")
    recent=history[-3:];tol=protocol["energy_tolerance_Ha_per_electron"]
    sampling=all(r["sampling_pass"] for r in recent)
    comparisons=[equivalent(a["energy_Ha"],b["energy_Ha"],tol) for a,b in zip(recent[:-1],recent[1:])]
    return dict(converged=bool(sampling and all(x[0] for x in comparisons) and challenge),
        stationary_candidate=bool(sampling and all(x[0] for x in comparisons)),
        sampling_pass=sampling,equivalence_bounds_Ha=[x[1] for x in comparisons],
        tolerance_Ha=tol,challenge_completed=challenge,
        reason="statistical_energy_equivalence_with_mixing_and_LR_challenge; not_ground_state_certificate")


def sampling_passes(result, protocol):
    """Apply the declared precision and mixing thresholds to fixed measurements."""
    energy=result["energy_Ha"];imag=result["imaginary_Ha"];m=protocol["measurement"]
    tol=protocol["energy_tolerance_Ha_per_electron"]
    return bool(energy["sem"] <= tol/3 and energy["rhat"] <= m["max_rhat"] and
        energy["ess"] >= m["min_ess"] and
        result["structure_factor"]["rhat"] <= m["max_structure_rhat"] and
        abs(imag["mean"]) <= 3*imag["sem"]+tol)


def validate_resume_output(output, target):
    """Only explicitly snapshotted, verified partial stages may be resumed."""
    manifest=json.loads((output/"resume_checkpoint.json").read_text())
    checkpoint=output/manifest["checkpoint"]
    if manifest["target"] != target or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError(f"Resume checkpoint mismatch: {output}")
    with np.load(checkpoint,allow_pickle=True) as z:
        if not 0 <= int(z["t"])+1 <= target:
            raise ValueError(f"Resume step exceeds target: {checkpoint}")


def measure(spec):
    import jax
    jax.config.update("jax_enable_x64",True)
    import jax.numpy as jnp
    from periodicwave import mcmc,networks,CustomScreenformer as CS,attention
    from screening_analysis.checkpoint_model import load_model,_factory
    from screening_analysis.sampler import make_hop
    from screening_analysis.lj_maps import first_shell
    case=spec["case"];p=spec["protocol"];m=p["measurement"]
    output=Path(spec["measurement_output"]);output.mkdir(parents=True,exist_ok=False)
    with load_model(Path(spec["output"])) as (net,params,data,info):
        lattice=np.asarray(info["lattice"]);n=case["n"];chains=m["chains"]
        seed=spec["stage_seed"]+1000000
        rng=jax.random.PRNGKey(seed);rng,kx,ks=jax.random.split(rng,3)
        position=jax.random.uniform(kx,(chains,n,2),dtype=jnp.float64)@jnp.asarray(lattice).T
        spins=jnp.where(jax.random.bernoulli(ks,shape=(chains,n)),1.,-1.)
        state=networks.FermiNetData(positions=position.reshape(chains,-1),spins=spins,
            atoms=jnp.zeros((chains,1,2)),charges=jnp.zeros((chains,1)))
        batch=jax.vmap(lambda pp,x,s,a,c:net.apply(pp,x,s,a,c)[1],in_axes=(None,0,0,0,0))
        hop=make_hop(batch,n,lattice)
        localstep=mcmc.make_mcmc_step(batch,chains,steps=m["steps"],ndim=2,blocks=1,spin_flips=p["training"]["spin_flips"])
        kw=dict(info["config"]["system"]["make_local_energy_kwargs"]);kw["lattice"]=jnp.asarray(lattice)
        local=_factory(info["config"]["system"]["make_local_energy_fn"])(net.apply,jnp.zeros(1),(n,0),complex_output=True,**kw)
        energies=jax.jit(jax.vmap(lambda pp,k,d:local(pp,k,d)[0],in_axes=(None,0,0)))
        with np.load(info["checkpoint"],allow_pickle=True) as z:width=float(z["mcmc_width"].ravel()[0])
        @jax.jit
        def move(state,rng):
            rng,k1,k2=jax.random.split(rng,3)
            new,rate=localstep(params,state,k1,width)
            lp=2*batch(params,new.positions,new.spins,new.atoms,new.charges)
            pos,_,_,ha=hop(params,new.positions,new.spins,new.atoms,new.charges,k2,lp,2)
            return networks.FermiNetData(positions=pos,spins=new.spins,atoms=new.atoms,charges=new.charges),rng,rate,ha
        for _ in range(m["burn"]):state,rng,_,_=move(state,rng)
        values=[];positions=[];spin_values=[];accept=[];hops=[];total=0;target=m["times"]
        shell=first_shell(lattice,n)
        while True:
            for t in range(total,target):
                state,rng,rate,ha=move(state,rng)
                rng,ke=jax.random.split(rng)
                e=np.asarray(energies(params,jax.random.split(ke,chains),state))
                if not np.isfinite(e).all():raise FloatingPointError("nonfinite independent local energy; no samples discarded")
                values.append(e);positions.append(np.asarray(state.positions).reshape(chains,n,2));spin_values.append(np.asarray(state.spins))
                accept.append(np.asarray(rate));hops.append(float(ha))
            array=np.asarray(values);scale=n*case["rs"]**2
            energy=time_statistics(array.real/scale);imag=time_statistics(array.imag/scale)
            rho=np.exp(2j*np.pi*np.einsum("tcnd,qd->tcnq",np.asarray(positions)@np.linalg.inv(lattice).T,shell)).sum(2)
            sf=np.mean(abs(rho)**2/n,axis=-1);structure=time_statistics(sf)
            sampling_pass=sampling_passes(dict(energy_Ha=energy,imaginary_Ha=imag,
                structure_factor=structure),p)
            if sampling_pass or target>=m["max_times"]:break
            total=target;target=min(2*target,m["max_times"])
        np.savez_compressed(output/"samples.npz",positions=np.asarray(positions),spins=np.asarray(spin_values),
            local_energy=array,lattice=lattice,acceptance=np.asarray(accept),hop_acceptance=np.asarray(hops))
        summary=dict(case=case,step=spec["target"],stage_lr=spec["lr"],checkpoint=info["checkpoint"],
            checkpoint_sha256=info["checkpoint_sha256"],energy_Ha=energy,imaginary_Ha=imag,
            structure_factor=structure,sampling_pass=sampling_pass,measurement_seed=seed,
            acceptance=np.mean(accept,axis=0),global_hop_acceptance=float(np.mean(hops)),
            proposal_width=width,independent_uniform_starts=True,parameters_fixed=True)
        dref=float(CS._max_periodic_distance(jnp.asarray(lattice)))
        curves=[]
        from screening_analysis.attn_probe import probe,save_probe
        flat_positions=np.asarray(positions).reshape(-1,n*2);flat_spins=np.asarray(spin_values).reshape(-1,n)
        ids=np.linspace(0,len(flat_positions)-1,min(256,len(flat_positions))).astype(int)
        eval_data=dict(positions=flat_positions[ids],spins=flat_spins[ids],atoms=np.zeros((len(ids),1,2)),charges=np.zeros((len(ids),1)))
        pr=probe(Path(spec["output"]),checkpoint=Path(info["checkpoint"]),config_path=info["config_path"],
            lattice=lattice,evaluation_data=eval_data,nconf=len(ids),collect_internal=False)
        save_probe(pr,output/"attention_probe.npz")
        for li,ap in enumerate(params["layers"]["attention"]):
            scales={k:np.asarray(v) for k,v in attention.window_scales(ap,dref,info["settings"]["WIN_SHAPE"]).items()}
            for head in range(p["training"]["heads"]):
                curves.append(dict(layer=li,head=head,**{k:float(v[head]) for k,v in scales.items()}))
        summary["window_scales"]=curves
        dump(output/"summary.json",summary)
        print(json.dumps(dict(step=spec["target"],energy_Ha=energy,sampling_pass=sampling_pass)),flush=True)


def run_case(root,index,pilot=False):
    root=Path(root).resolve();p=json.loads((root/"plan.json").read_text());case=dict(p["cases"][index])
    if pilot:
        case["id"]="pilot-"+str(index);p=dict(p)
        p["training"]=dict(p["training"],batch=8,width=8,attn_dim=2,value_dim=2,burn=4,mcmc_steps=2)
        p["measurement"]=dict(p["measurement"],chains=4,times=16,max_times=16,burn=8,steps=2)
        stages=[dict(updates=3,lr=.001),dict(updates=3,lr=.0003)]
    else:stages=list(p["initial_stages"])
    directory=root/("pilot" if pilot else "runs")/case["id"]
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"finished.json").exists():return
    dump(directory/"case.json",case)
    from screening_analysis.rs_sweep import prepare_reference
    reference=directory/"reference/orbitals.txt"
    if not reference.exists():prepare_reference(case,p["hf"],reference.parent)
    history=[];previous=None;target=0;challenge=False;challenge_ever=False;stage_index=0
    resume=directory/"resume_state.json"
    if resume.exists():
        state=json.loads(resume.read_text())
        if pilot or state["case"] != case or state["protocol_sha256"] != hashlib.sha256((root/"plan.json").read_bytes()).hexdigest():
            raise ValueError("Resume case/protocol mismatch")
        history=[json.loads((directory/x).read_text()) for x in state["measurements"]]
        if history:
            previous=directory/state["previous"];target=history[-1]["step"];stage_index=len(history)
            challenge=bool(stage_index>=p["minimum_stage_count"] and check_convergence(history,p).get("stationary_candidate",False))
        dump(directory/"convergence.json",dict(check_convergence(history,p),history=history))
    try:
        while True:
            if stage_index<len(stages):stage=stages[stage_index]
            else:stage=dict(updates=p["challenge_updates"] if challenge else p["extension_updates"],
                            lr=p["challenge_lr"] if challenge else p["extension_lr"])
            target+=stage["updates"]
            if not pilot:target=min(target,p["max_updates"])
            output=directory/f"stage-{stage_index:02d}-{target:06d}"
            measurement_output=output.parent/(output.name+"-measurement")
            spec=dict(case=case,protocol=p,reference=str(reference),output=str(output),
                previous=str(previous) if previous else None,measurement_output=str(measurement_output),
                lr=stage["lr"],target=target,stage_seed=case["seed"]+10007*(stage_index+1))
            specpath=directory/f"stage-{stage_index:02d}.json";dump(specpath,spec)
            if not (output/"stage_complete.json").exists():
                if output.exists():
                    if not (output/"resume_checkpoint.json").exists():
                        raise RuntimeError(f"Partial stage preserved; explicit recovery required: {output}")
                    validate_resume_output(output,target)
                with (directory/f"train-{stage_index:02d}.log").open("x") as log:
                    subprocess.run([sys.executable,"-u","-m",__package__+".converged_window_study","train-stage",str(specpath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                with np.load(sorted(output.glob("qmcjax_ckpt_*.npz"))[-1],allow_pickle=True) as z:
                    if int(z["t"])+1!=target:raise RuntimeError("Training did not reach target")
                dump(output/"stage_complete.json",dict(target=target))
            if not (measurement_output/"summary.json").exists():
                with (directory/f"measure-{stage_index:02d}.log").open("x") as log:
                    subprocess.run([sys.executable,"-u","-m",__package__+".converged_window_study","measure",str(specpath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((measurement_output/"summary.json").read_text());history.append(result)
            previous=output;stage_index+=1
            if pilot:
                if stage_index==len(stages):
                    dump(directory/"finished.json",dict(status="pilot_pass",steps=target,energy_finite=True,optimizer_continuation=True));break
                continue
            diagnosis=check_convergence(history,p,challenge=challenge)
            challenge_ever=challenge_ever or challenge
            dump(directory/"convergence.json",dict(diagnosis,history=history))
            if stage_index>=p["minimum_stage_count"] and diagnosis["converged"]:
                dump(directory/"finished.json",dict(status="converged_to_protocol",steps=target,measurement=str(measurement_output),
                    training=str(output),ground_state_certified=False,phase_verified=False));break
            if target>=p["max_updates"]:
                dump(directory/"finished.json",dict(status="unconverged_budget_limit",steps=target,measurement=str(measurement_output),
                    training=str(output),ground_state_certified=False,phase_verified=False));break
            challenge=bool(stage_index>=p["minimum_stage_count"] and diagnosis.get("stationary_candidate",False) and not challenge)
        print(json.dumps(dict(case=case["id"],steps=target,status="finished")),flush=True)
    except Exception:
        (directory/"failure.txt").write_text(traceback.format_exc());raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare");q.add_argument("root",type=Path)
    for name in ("case","pilot"):
        q=sub.add_parser(name);q.add_argument("root",type=Path);q.add_argument("index",type=int)
    for name in ("train-stage","measure"):
        q=sub.add_parser(name);q.add_argument("spec",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":prepare(args.root)
    elif args.command in ("case","pilot"):run_case(args.root,args.index,pilot=args.command=="pilot")
    else:{"train-stage":train_stage,"measure":measure}[args.command](json.loads(args.spec.read_text()))


if __name__=="__main__":main()
