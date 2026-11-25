"""Opt-in Window factory; original QK/Trim/Value/TanhNorm/gate/projection reused."""
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import attention as shared


class ScreeningFactory:
    def __init__(self,config):self.config=config

    def widths(self,params,heads,d_ref):
        c=self.config['window'];mode=c['mode']
        if mode=='baseline':return d_ref*jax.nn.softplus(params['mask_omega'])
        if mode.startswith('trainable'):
            values=c['lower']+(c['upper']-c['lower'])*jax.nn.sigmoid(params['window_logit'])
            return jnp.broadcast_to(values,(heads,))
        return jnp.full(heads,c['upper'] if c['width']=='full' else c['width'])

    def window(self,params,d,heads,d_ref):
        c=self.config['window']
        if c['mode']=='baseline':
            from periodicwave import CustomScreenformer as cs
            window=shared.screening_window(params,d,d_ref,win_shape=cs.WIN_SHAPE,freeze_w=cs.FREEZE_W,
                                            algebraic_power=cs.WIN_POWER,freeze_eta=cs.FREEZE_ETA)
        elif c['mode']=='full' or c['width']=='full':window=jnp.ones((heads,)+d.shape)
        else:
            widths=self.widths(params,heads,d_ref)
            if c['shape']=='sigmoid':window=jax.nn.sigmoid((widths[:,None,None]-d)/c['temperature'])
            elif c['shape']=='hard':window=(d[None]<=widths[:,None,None]).astype(d.dtype)
            else:
                virtual=dict(mask_omega=jnp.log(jnp.expm1(widths/d_ref)))
                window=shared.screening_window(virtual,d,d_ref,win_shape=c['shape'],freeze_w=False)
        if '_window_override' in params:
            override=params['_window_override']
            if self.config['window']['shape']=='sigmoid':window=jax.nn.sigmoid((override[:,None,None]-d)/c['temperature'])
            else:
                virtual=dict(mask_omega=jnp.log(jnp.expm1(override/d_ref)))
                window=shared.screening_window(virtual,d,d_ref,win_shape='cos',freeze_w=False)
        return window

    def __call__(self,num_heads,embed_dim,attn_dim,value_dim,num_electrons,d_ref,**kwargs):
        from periodicwave import CustomScreenformer as cs
        base_init,base_apply=cs.make_screening_attention(num_heads,embed_dim,attn_dim,value_dim,num_electrons,d_ref,**kwargs)
        c=self.config['window']
        options=dict(num_electrons=num_electrons,num_heads=num_heads,attn_dim=attn_dim,value_dim=value_dim,d_ref=d_ref,
                     **{k:kwargs[k] for k in ['use_softmask','use_fixed_value','use_tanh_norm','use_gate','use_output_scale']})
        def init(key):
            params=base_init(key)
            if c['mode'].startswith('trainable'):
                fraction=(c['width']-c['lower'])/(c['upper']-c['lower'])
                count=num_heads if c['mode']=='trainable_per_head' else 1
                params['window_logit']=jnp.full(count,np.log(fraction/(1-fraction)))
            return params
        def apply(params,x,d,pair_feat=None,*,return_aux=False):
            if c['mode']=='baseline' and '_window_override' not in params and '_smooth_edge_radius' not in params:
                return base_apply(params,x,d,pair_feat,return_aux=return_aux)
            window=self.window(params,d,num_heads,d_ref)
            if '_smooth_edge_radius' in params:
                radius=params['_smooth_edge_radius'];z=jnp.clip(d/radius,0,1)
                window=window*(1-shared._smooth7(z))[None]
            return shared.screening_attention(params,x,d,pair_feat,options=options,
                trim_max=cs.TRIM_MAX,trim_pow=cs.TRIM_POW,win_shape=cs.WIN_SHAPE,freeze_w=cs.FREEZE_W,
                clip_r=cs.CLIP_R,clip_beta=cs.CLIP_BETA,return_aux=return_aux,window=window,
                algebraic_power=cs.WIN_POWER,freeze_eta=cs.FREEZE_ETA)
        return init,apply
