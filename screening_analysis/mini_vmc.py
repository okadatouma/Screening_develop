# -*- coding: utf-8 -*-
"""CPU でも回る最小の VMC。構成の比較だけが目的で、収束は狙わない。

    python screening_analysis/mini_vmc.py --mode bin8 --steps 200
"""
import argparse, os, sys, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); sys.path.insert(0,HERE)
import jax, jax.numpy as jnp, optax
from periodicwave import CustomScreenformer as CS, networks, envelopes
from periodicwave.pbc import feature_layer as FL, hamiltonian as HAM

AEE=np.sqrt(2*np.pi/np.sqrt(3))

def build(mode, n=5, nup=13, ndn=12, rs=30.0, NH=16, U=0.0):
    a1=AEE*np.array([1.,0.]); a2=AEE*np.array([-0.5,np.sqrt(3)/2])
    Lnp=np.column_stack([n*a1,n*a2]); L=jnp.asarray(Lnp)
    dref=float(CS._max_periodic_distance(Lnp)); nspins=(nup,ndn); charges=jnp.zeros(1)
    fl=FL.make_pbc_feature_layer(natoms=1,nspins=nspins,ndim=2,lattice=L,include_r_ae=False)
    kw={}
    if mode=="base": NB=NH
    elif mode.startswith("band"):
        NB=int(mode[4:]); D=dref/max(NB-1,1); c=D*np.arange(NB)
        kw=dict(init_softmask_centres=(c/dref,np.full(NB,D)/dref), use_fixed_value=True)
    elif mode.startswith("env"):
        # 全体窓 g を輪切り： env8 / env16、既定 p = 3、tau = 0.5（遷移帯 = 輪幅の半分）
        NB=int(mode[3:]); P_EXP=float(os.environ.get("ENV_P",3.0)); TAU=float(os.environ.get("ENV_TAU",0.5))
        e=np.linspace(0.0,dref,NB+1); dl=TAU*(e[1]-e[0])/2
        kw=dict(init_softmask_rings=(e/dref,dl/dref,P_EXP), use_fixed_value=True)
    elif mode=="fixv": NB=NH; kw=dict(use_fixed_value=True)
    else: raise ValueError(mode)
    net=CS.make_fermi_net(nspins,charges,ndim=2,determinants=4,
        envelope=envelopes.make_null_envelope(),feature_layer=fl,jastrow="NONE",jastrow_kwargs={},
        bias_orbitals=False,complex_output=False,pbc_lattice=L,num_layers=1,num_heads=NB,
        attn_dim=16,value_dim=16,mlp_dim=64,num_perceptrons_per_layer=2,use_layer_norm=True,
        mlp_activation_fct="GELU",init_trim_r=1.9,**kw)
    le=HAM.local_energy(f=net.apply,charges=charges,nspins=nspins,complex_output=False,use_scan=False,
        lattice=L,potential_type="Coulomb",
        potential_kwargs=dict(interaction_energy_scale=rs,moire_lattice_vectors=L,
                              moire_potential_strength=U,moire_potential_phi=np.pi/4),
        kinetic_kwargs=dict(laplacian_method="folx"))
    return net,le,Lnp,dref,nup+ndn,nspins


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",default="base"); ap.add_argument("--steps",type=int,default=200)
    ap.add_argument("--batch",type=int,default=128); ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--rs",type=float,default=30.0); ap.add_argument("--U",type=float,default=0.0)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    net,le,Lnp,dref,NE,nspins=build(a.mode,rs=a.rs,U=a.U)
    L=jnp.asarray(Lnp)
    key=jax.random.PRNGKey(a.seed)
    key,sk=jax.random.split(key); params=net.init(sk)
    logabs=lambda p,x,s,at,ch: net.apply(p,x,s,at,ch)[1]
    blog=jax.vmap(logabs,in_axes=(None,0,0,0,0))
    B=a.batch
    sp=jnp.asarray(np.tile(np.array([1.]*nspins[0]+[-1.]*nspins[1]),(B,1)))
    at=jnp.zeros((B,1,2)); ch=jnp.zeros((B,1))
    # 三角格子の上に置いて少しずらす（学習の出発点）
    ij=np.stack(np.meshgrid(np.arange(5),np.arange(5),indexing="ij"),-1).reshape(-1,2)
    R0=ij@np.stack([AEE*np.array([1.,0.]),AEE*np.array([-0.5,np.sqrt(3)/2])])
    rng=np.random.default_rng(a.seed)
    x=jnp.asarray((R0[None]+0.25*rng.normal(size=(B,NE,2))).reshape(B,-1))

    @jax.jit
    def mcmc(p,x,key,w):
        def body(c,_):
            x,lp,key,acc=c
            key,k1,k2=jax.random.split(key,3)
            x2=x+w*jax.random.normal(k1,x.shape)
            lp2=2.*blog(p,x2,sp,at,ch)
            ok=jnp.log(jax.random.uniform(k2,(B,)))<(lp2-lp)
            return (jnp.where(ok[:,None],x2,x),jnp.where(ok,lp2,lp),key,acc+ok.mean()),None
        lp=2.*blog(p,x,sp,at,ch)
        (x,lp,key,acc),_=jax.lax.scan(body,(x,lp,key,0.),None,length=8)
        return x,key,acc/8

    @jax.jit
    def step(p,opt_state,x,key):
        dat=networks.FermiNetData(positions=x,spins=sp,atoms=at,charges=ch)
        ks=jax.random.split(key,B)
        E=jnp.real(jax.vmap(le,in_axes=(None,0,0))(p,ks,dat)[0])
        Ec=jnp.clip(E,jnp.percentile(E,2),jnp.percentile(E,98))
        dE=Ec-jnp.mean(Ec)
        g=jax.grad(lambda q: 2.*jnp.mean(jax.lax.stop_gradient(dE)*blog(q,x,sp,at,ch)))(p)
        upd,opt_state=opt.update(g,opt_state,p)
        return optax.apply_updates(p,upd),opt_state,jnp.mean(Ec),jnp.std(Ec)

    opt=optax.chain(optax.clip_by_global_norm(10.0),optax.adam(a.lr))
    opt_state=opt.init(params)
    w=jnp.asarray(0.18); t0=time.time(); hist=[]
    for i in range(a.steps):
        key,k1,k2=jax.random.split(key,3)
        x,_,acc=mcmc(params,x,k1,w)
        w=w*jnp.where(acc>0.55,1.1,jnp.where(acc<0.45,1/1.1,1.0))
        params,opt_state,E,S=step(params,opt_state,x,k2)
        hist.append((float(E),float(S)))
        if (i+1)%20==0 or i==0:
            print("  step %4d  E = %10.2f   局所エネルギーの標準偏差 = %9.2f   pmove %.2f  (%.0f s)"
                  % (i+1,E,S,acc,time.time()-t0),flush=True)
    H=np.array(hist)
    print("\n  [%s] 最初の 20 step 平均 E = %.1f  →  最後の 20 step 平均 E = %.1f   （差 %+.1f）"
          % (a.mode,H[:20,0].mean(),H[-20:,0].mean(),H[-20:,0].mean()-H[:20,0].mean()))
    print("  [%s] 局所エネルギーの標準偏差 : 最初 %.1f  →  最後 %.1f" % (a.mode,H[:20,1].mean(),H[-20:,1].mean()))
    np.save(os.path.join(HERE,"mini_%s.npy"%a.mode),H)

if __name__=="__main__":
    main()
