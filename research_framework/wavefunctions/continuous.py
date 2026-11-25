"""Original orbital/Jastrow factories, with opt-in embedding and quasi-periodic twist.

Gauge: Psi_theta(R)=exp(i sum_i kappa.r_i) phi_theta(R), kappa=L^-T theta.
phi_theta is periodic in particle coordinates. Differentiating the FULL Psi in
the original Hamiltonian includes the twist kinetic/cross terms. The raw theta
conditioning does not impose an incorrect periodic Bloch-frame condition.
"""
from dataclasses import replace
import copy
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as cs, envelopes, networks
from periodicwave.pbc import feature_layer
from ..factory import make_embedding, make_attention


class ContinuousWavefunction:
    def __init__(self,config,full_config,reference_path=None):
        self.config=config;self.full=full_config
        p=full_config['physics'];m=full_config['model'];b=full_config['embedding'];a=full_config['attention']
        n=p['n'];self.lattice=jnp.asarray(p['lattice']);self.atoms=jnp.zeros((1,2));self.charges=jnp.zeros(1)
        self.twisted=config['type']=='twist_dependent'
        self.embedding=make_embedding(b,'continuous_electrons',p,m,self.twisted)
        self.attention=make_attention(a,'continuous_electrons')
        original_embedding=(b['type']=='baseline' and b['spin_encoding']=='scalar' and not self.twisted and m['capacity_mode']=='natural')
        use_hf=config['type'] in ['hf_nn_jastrow','hf','hf_jastrow','twist_dependent']
        hf=p['hf_path'] or reference_path
        if use_hf and not hf:raise ValueError('HF ansatz requires a supplied/generated HF table')
        options=dict(ndim=2,determinants=1,envelope=envelopes.make_null_envelope(),
                     feature_layer=feature_layer.make_pbc_feature_layer(1,(n,0),ndim=2,lattice=self.lattice,include_r_ae=False),
                     jastrow='NONE',complex_output=config['complex_phase'],pbc_lattice=self.lattice,num_layers=m['depth'],
                     num_heads=m['heads'],attn_dim=m['head_dim'],value_dim=m['head_dim'],mlp_dim=m['width'],
                     num_perceptrons_per_layer=m['perceptrons'],use_layer_norm=m['layer_norm'],mlp_activation_fct='TANH',
                     input_embedding=None if original_embedding else self.embedding,
                     attention_factory=self.attention)
        if use_hf:
            from periodicwave.lj_orbitals import cusp_values
            options.update(lj_hf_path=str(hf),lj_lam=p['lam'],lj_net_scale=0. if config['type'] in ['hf','hf_jastrow'] else config['net_scale'],
                           lj_jastrow=config['jastrow'] and config['type']!='hf',lj_cusps=cusp_values(p['rs'],p['lam'],p['delta']) if p['potential']=='Coulomb' else (0.,0.,0.))
        self.network=cs.make_fermi_net((n,0),self.charges,**options)
        self.diagnostic=cs.make_screenformer_diagnostic((n,0),self.network.options)
        self.options=options

    def init(self,key):return self.network.init(key)

    def conditioned(self,params,twist):
        if not self.twisted:return params
        layers=dict(params['layers']);extra=dict(layers.get('embedding_extra',{}));extra['_twist']=jnp.asarray(twist)
        layers['embedding_extra']=extra
        return dict(params,layers=layers)

    def periodic_apply(self,params,pos,spins,twist=None):
        twist=jnp.asarray(self.full['physics']['twist'] if twist is None else twist)
        phase,logabs=self.network.apply(self.conditioned(params,twist),pos,spins,self.atoms,self.charges)
        if not self.config['complex_phase']:phase=jnp.where(phase<0,jnp.pi,0.)
        return phase,logabs

    def apply(self,params,pos,spins,twist=None):
        twist=jnp.asarray(self.full['physics']['twist'] if twist is None else twist)
        phase,logabs=self.periodic_apply(params,pos,spins,twist)
        if self.twisted:
            kappa=jnp.linalg.solve(self.lattice.T,twist)
            phase=phase+jnp.sum(pos.reshape(-1,2)@kappa)
        return phase,logabs

    def production_apply(self,params,pos,spins,atoms,charges):
        del atoms,charges
        return self.apply(params,pos,spins)

    def internal(self,params,pos,spins):
        return self.diagnostic(self.conditioned(params,self.full['physics']['twist']),pos,spins,self.atoms,self.charges)

    def penalty(self,params):
        result=jnp.asarray(0.)
        extra=params['layers'].get('embedding_extra',{})
        if self.embedding.learned:result=result+self.embedding.penalty(extra)
        win=self.full['attention']['window']
        if win['regularization']:
            widths=jnp.stack([self.attention.widths(p,self.full['model']['heads'],1.) for p in params['layers']['attention']])
            result=result+win['regularization']*jnp.mean(widths/win['upper'])
        return result

    def intervene(self,params,spec):
        changed=copy.deepcopy(params);kind=spec['type'];head=spec.get('head',0);layer=spec.get('layer',0)
        if kind in ['remove_six_theta','remove_fourier','remove_spin','replace_spin']:
            if self.network.options.input_embedding is None:
                if kind not in ['remove_spin','replace_spin']:raise ValueError('feature group absent')
                # Scalar-spin feature contributes through the last row of original W.
                changed['layers']['embed']=changed['layers']['embed'].at[-1].set(0)
            else:
                extra=changed['layers']['embedding_extra']
                if kind=='replace_spin':extra['_spin_replacement']=jnp.asarray([.5,.5] if self.embedding.onehot else [0.])
                else:extra['_feature_mask']=self.embedding.feature_mask(kind)
            return self,changed
        ap=changed['layers']['attention'][layer]
        if kind=='head_ablation':
            projection=ap['attn_output'];d=self.full['model']['head_dim']
            ap['attn_output']=projection.at[head*d:(head+1)*d].set(0)
        elif kind=='window_width':ap['_window_override']=jnp.full(self.full['model']['heads'],spec['width'])
        elif kind in ['distance_band','remove_connections']:
            if kind=='remove_connections' and 'edges' in spec:raise ValueError('electron-index edge removal breaks permutation symmetry; use a physical distance kernel')
            ap['_smooth_edge_radius']=jnp.asarray(spec.get('radius',1.))
        elif kind=='zero_phase_channel':
            # Actual imaginary NN ORBITAL coefficients, not discarding determinant phase.
            w=changed['orbital'][0]['w'];changed['orbital'][0]['w']=w.at[:,1::2].set(0)
        else:raise ValueError('unsupported continuous intervention: '+kind)
        return self,changed
