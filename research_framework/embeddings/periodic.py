"""PBC-compatible Fourier features; crystal reciprocal indices derive from S.

L = A S (columns): q_crystal = 2pi L^-T S^T m. Thus the Fourier index
relative to the simulation cell is S^T m, never an unconditional integer 6.
"""
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import network_blocks


class PeriodicEmbedding:
    def __init__(self, config, physics, model, twist=False):
        self.config,self.physics,self.model=config,physics,model
        self.twist=twist
        s=np.asarray(physics['crystal_supercell'],int)
        self.crystal_modes=np.array([[1,0],[0,1],[1,-1]])@s
        self.modes=np.asarray(config['modes'],int)
        self.reciprocal=jnp.linalg.inv(jnp.asarray(physics['lattice']))
        self.onehot=config['spin_encoding']=='onehot' or config['type'] in ['spin_onehot','six_theta_onehot']
        self.high=config['type'] in ['six_theta','six_theta_onehot']
        self.learned=config['type']=='learned_fourier'

    def dimensions(self, base=4):
        return base+(6 if self.high else 0)+(2*len(self.modes) if self.learned else 0)+(2 if self.onehot else 1)+(2 if self.twist else 0)

    def init(self,key,base_dim,width):
        if base_dim!=4:raise ValueError('research Fourier/spin embedding requires 2D and exactly one ghost atom without radial features')
        features=self.dimensions(base_dim)
        rows=self.model['embedding_budget'] if self.model['capacity_mode']=='padded' else features
        if rows<features:raise ValueError('embedding_budget smaller than active input feature dimension')
        weight=network_blocks.init_linear_layer(key,rows,width,include_bias=False)['w']
        extra={}
        if self.learned:extra['fourier_amplitude']=jnp.full(len(self.modes),self.config['amplitude_initial'])
        elif self.model['capacity_mode']=='padded':extra['feature_gain']=jnp.ones(len(self.modes))
        return weight,extra

    def features(self,base,spins,ae,extra=None):
        extra=extra or {}
        parts=[base]
        theta=2*jnp.pi*(ae[:,0,:]@self.reciprocal.T)
        if self.high:
            angles=theta@jnp.asarray(self.crystal_modes).T
            parts.append(jnp.stack((jnp.cos(angles),jnp.sin(angles)),-1).reshape(len(spins),-1))
        if self.learned:
            angles=theta@jnp.asarray(self.modes).T
            amp=extra['fourier_amplitude']
            parts.append((jnp.stack((jnp.cos(angles),jnp.sin(angles)),-1)*amp[None,:,None]).reshape(len(spins),-1))
        spin=jnp.stack(((1+spins)/2,(1-spins)/2),-1) if self.onehot else spins[:,None]
        if '_spin_replacement' in extra:spin=jnp.broadcast_to(extra['_spin_replacement'],spin.shape)
        parts.append(spin)
        if self.twist:
            parts.append(jnp.broadcast_to(extra.get('_twist',jnp.zeros(2))/(2*jnp.pi),(len(spins),2)))
        features=jnp.concatenate(parts,-1)
        if 'feature_gain' in extra:
            # Active gain controls match the learned-amplitude parameter budget;
            # nominal parameter count matching does not imply identical function classes.
            features=features*extra['feature_gain'][jnp.arange(features.shape[-1])%len(self.modes)]
        if '_feature_mask' in extra:features=features*extra['_feature_mask']
        return features

    def apply(self,weight,extra,base,spins,ae):
        features=self.features(base,spins,ae,extra)
        if self.learned and self.config['normalize_projection']:
            # Remove the immediate a_q * W_q rescaling ambiguity at the input.
            block=weight[4:4+2*len(self.modes)].reshape(len(self.modes),2,-1)
            block=block/jnp.sqrt(jnp.sum(block**2,axis=(1,2),keepdims=True)+1e-12)
            weight=weight.at[4:4+2*len(self.modes)].set(block.reshape(-1,weight.shape[-1]))
        if features.shape[-1]<weight.shape[0]:features=jnp.pad(features,((0,0),(0,weight.shape[0]-features.shape[-1])))
        return features@weight

    def penalty(self,extra):
        return self.config['sparse_penalty']*jnp.mean(jnp.abs(extra['fourier_amplitude'])) if self.learned else jnp.asarray(0.)

    def feature_mask(self,kind):
        mask=np.ones(self.dimensions())
        if kind=='remove_six_theta':
            if not self.high:raise ValueError('six_theta features absent')
            mask[4:10]=0
        elif kind=='remove_fourier':
            if not self.learned:raise ValueError('learned Fourier features absent')
            mask[4:4+2*len(self.modes)]=0
        elif kind=='remove_spin':
            end=self.dimensions()-(2 if self.twist else 0)
            mask[end-(2 if self.onehot else 1):end]=0
        else:raise ValueError(kind)
        return jnp.asarray(mask)
