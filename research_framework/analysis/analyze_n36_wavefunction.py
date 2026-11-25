"""Wavefunction changes and head ablations on the same eight saved probes."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from research_framework.wavefunctions.continuous import ContinuousWavefunction


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    jax.config.update("jax_enable_x64",True)
    hashes={}
    def read(p):
        b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return b
    def load(p):return json.loads(read(p))
    def save(name,rows):
        with (out/name).open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def difference(p0,l0,p1,l1):
        delta=np.angle(np.exp(1j*(p1-p0)))
        center=np.angle(np.mean(np.exp(1j*delta)))
        relative=np.angle(np.exp(1j*(delta-center)))
        dl=l1-l0
        return dict(logabs_centered_rms=float(np.std(dl)),logabs_delta_mean=float(dl.mean()),
                    phase_centered_rms=float(np.sqrt(np.mean(relative**2))),
                    phase_coherence=float(abs(np.mean(np.exp(1j*delta)))))
    rows=[];actions=[];representations=[];stored={}
    def cka(x,y):
        x=x-x.mean(0);y=y-y.mean(0)
        return float(np.linalg.norm(x.T@y)**2/max(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y),1e-30))
    plan=load(root/"plan.json")
    for case in plan:
        d=root/case["name"];c=case["config"];kind=c["embedding"]["type"]+("_L1" if c["embedding"]["sparse_penalty"] else "")
        seed=c["experiment"]["seed"]
        with np.load(d/"inputs.npz") as z:x=jnp.asarray(z["probe_positions"]);s=jnp.asarray(z["probe_spins"])
        read(d/"inputs.npz");read(d/"config.json")
        before=pickle.loads(read(d/"checkpoints/step_0000000.pkl"))["state"]["params"]
        after=pickle.loads(read(d/"checkpoints/step_0010000.pkl"))["state"]["params"]
        before=jax.tree_util.tree_map(jnp.asarray,before);after=jax.tree_util.tree_map(jnp.asarray,after)
        wave=ContinuousWavefunction(c["wavefunction"],c)
        evaluate=jax.jit(jax.vmap(wave.apply,in_axes=(None,0,0)))
        p0,l0=map(np.asarray,evaluate(before,x,s));p1,l1=map(np.asarray,evaluate(after,x,s))
        assert np.isfinite(np.r_[p0,l0,p1,l1]).all()
        rows.append(dict(name=case["name"],kind=kind,seed=seed,**difference(p0,l0,p1,l1)))
        stored[case["name"]+"_before"]=np.stack([p0,l0]);stored[case["name"]+"_after"]=np.stack([p1,l1])
        for head in [0,1,2,3,"all"]:
            pp=dict(after);layers=dict(pp["layers"]);att=list(layers["attention"]);ap=dict(att[0]);weight=ap["attn_output"]
            ap["attn_output"]=jnp.zeros_like(weight) if head=="all" else weight.at[int(head)*8:(int(head)+1)*8].set(0)
            att[0]=ap;layers["attention"]=att;pp["layers"]=layers
            pi,li=map(np.asarray,evaluate(pp,x,s))
            actions.append(dict(name=case["name"],kind=kind,seed=seed,intervention="remove_head_"+str(head),**difference(p1,l1,pi,li)))
        if kind.startswith("learned_fourier"):
            for qi in range(6):
                pp=dict(after);layers=dict(pp["layers"]);extra=dict(layers["embedding_extra"])
                mask=np.ones(17);mask[4+2*qi:6+2*qi]=0;extra["_feature_mask"]=jnp.asarray(mask)
                layers["embedding_extra"]=extra;pp["layers"]=layers
                pi,li=map(np.asarray,evaluate(pp,x,s))
                actions.append(dict(name=case["name"],kind=kind,seed=seed,intervention="remove_fourier_"+str(qi),**difference(p1,l1,pi,li)))
        theta=2*np.pi*(np.asarray(x).reshape(-1,2)@np.linalg.inv(c["physics"]["lattice"]).T)
        target={"position_low":np.c_[np.sin(theta),np.cos(theta)],
                "position_crystal":np.c_[np.sin(theta@np.array([[6,0],[0,6],[6,-6]]).T),np.cos(theta@np.array([[6,0],[0,6],[6,-6]]).T)],
                "spin":np.asarray(s).reshape(-1,1)}
        for label in ["before","after"]:
            path=d/"diagnostics"/(label+"_internal.npz")
            with np.load(path) as a:f=a["features"].reshape(-1,32)
            read(path)
            for name,y in target.items():representations.append(dict(name=case["name"],kind=kind,seed=seed,stage=label,target=name,cka=cka(f,y)))
        print("wavefunction",case["name"],rows[-1],flush=True)
    save("wavefunction_changes.csv",rows);save("wavefunction_ablations.csv",actions);save("feature_cka.csv",representations)
    np.savez_compressed(out/"matched_probe_wavefunctions.npz",**stored)
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    (out/"input_hashes.json").write_text(json.dumps(hashes,indent=2))
    (out/"METHOD.txt").write_text("Eight identical uniform probes, not Born samples. Log-amplitude differences centered to remove global normalization; phase differences circularly centered to remove global phase. These are function responses on probes, not physical fidelity or expected energy. Head ablation zeroes the actual projection rows. CKA uses centered final feature matrices across all 8x36 matched particles, not independent physical samples. No training/sampling performed.\n")
    print("DONE",out,flush=True)


if __name__=="__main__":main()
