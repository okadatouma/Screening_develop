# Copyright 2023 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND. See the License for permissions and limitations.
# Derived from CustomScreenformer.py; experimental QK/window inputs added.
"""実験用Screenformer。通常のNetwork/trainer/Diagnostic入口を置き換えない。

Q/Kへだけ高調波を追加。Value・残差・MLP・HF構成は既存の式を使う。
単一の層ループをforwardと独立JIT診断で共有する。診断値はProduction内部のbit列ではない。
"""
# 新しい窓研究だけで使うNetwork。既存Productionのモデル構築・診断APIとは別の入口。
# forwardと診断は同じapply_layersを共有し、内部量が必要なときだけreturn_aux=Trueにする。
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, attention, envelopes, networks, network_blocks, lj_orbitals
from periodicwave.pbc import feature_layer
from . import windows


def build(config):
    """Network, 独立JIT可能な診断関数, d_refを返す。

    従来のembed/attention/mlp/orbitalとHF構成を再利用。追加パラメータは
    layers.qk_extra、任意のgate_position、各attention.windowだけ。
    """
    N=config['n'];L=jnp.asarray(config['lattice']);dref=float(CS._max_periodic_distance(np.asarray(L)))
    hi=config['w_max'] or dref
    base=CS.make_fermi_net((N,0),jnp.zeros(1),ndim=2,complex_output=True,
        determinants=config['determinants'],envelope=envelopes.make_null_envelope(),
        feature_layer=feature_layer.make_pbc_feature_layer(natoms=1,nspins=(N,0),ndim=2,lattice=L,include_r_ae=False),
        jastrow='NONE',jastrow_kwargs={'ndim':2},bias_orbitals=True,pbc_lattice=L,
        num_layers=config['layers'],num_heads=config['heads'],attn_dim=config['attn_dim'],value_dim=config['value_dim'],
        mlp_dim=config['width'],num_perceptrons_per_layer=2,use_layer_norm=config['layer_norm'],mlp_activation_fct='GELU',
        use_fixed_value=config['value']=='pair',use_gate=config['gate']!='off',use_tanh_norm=True,
        init_softmask_w_factor=np.sqrt(config['w_first']*hi)/dref,
        init_softmask_w_spread=np.sqrt(hi/config['w_first']),use_output_scale=True,init_output_scale=config['output_scale'])
    options=base.options
    legacy_init,_=CS.make_screenformer_layers((N,0),1,options)
    _,mlp_apply=CS.make_mlp(2,'GELU');_,ln_apply=network_blocks.make_layer_norm()
    geometry=networks.make_periodic_construct_input_features(L)
    H=config['heads'];mode=config['input'];extra_dims={'E0':0,'E1':6,'E2':7,'E3':13}[mode]

    def init_layers(key):
        dims,p=legacy_init(key)
        # 既存の初期化キーは消費し直さない。追加射影はゼロ、元の初期関数を保つ。
        if extra_dims:p['qk_extra']=jnp.zeros((extra_dims,dims),p['embed'].dtype)
        if config['gate']=='position':
            p['gate_position']=network_blocks.init_linear_layer(jax.random.fold_in(key,771),10,dims,include_bias=False)['w']
        for ap in p['attention']:
            ap.pop('mask_omega',None)
            ap['window']=windows.initialize(config,dref,p['embed'].dtype)
        return dims,p

    def apply_layers(p,*,ae,r_ae,ee,r_ee,spins,charges,return_aux=False):
        raw,_=options.feature_layer.apply(ae=ae,r_ae=r_ae,ee=ee,r_ee=r_ee,**p['input'])
        x=network_blocks.apply_input_embedding(p['embed'],raw,spins)
        f=windows.input_features(ae,spins,L,mode,config['harmonic'])
        if mode=='E1':extra=f[:,5:]
        elif mode=='E2':extra=jnp.concatenate((f[:,4:10],f[:,10:12].sum(-1,keepdims=True)),axis=-1)
        elif mode=='E3':extra=jnp.concatenate((f[:,4:6].sum(-1,keepdims=True),f[:,6:]),axis=-1)
        # E2/E3のone-hotは(s,1)の線形基底へ書き直す。元のs射影はxに含まれる。
        gate_stream=None
        if config['gate']=='position':
            fpos=windows.input_features(ae,jnp.ones_like(spins),L,'E2',config['harmonic'])[:,:10]
            gate_stream=fpos@p['gate_position']  # Rだけの関数。後続層でもSを入れない。
        d=r_ee[...,0];pair=None
        if config['value']=='pair':
            _,pair=options.feature_layer.apply(ae=ae,r_ae=r_ae,ee=ee,r_ee=r_ee,**p['input'])
            pair=jnp.concatenate((pair[...,:1]/dref,pair[...,1:]),axis=-1)
        core_options=dict(num_electrons=N,num_heads=H,attn_dim=config['attn_dim'],value_dim=config['value_dim'],
            d_ref=dref,use_softmask=True,use_fixed_value=config['value']=='pair',use_tanh_norm=True,
            use_gate=config['gate']!='off',use_output_scale=True)
        records=[] if return_aux else None
        for l in range(config['layers']):
            ap=p['attention'][l]
            wc = config if config.get('cut_layer') in (None, l) else dict(config, cut_radius=None)
            m,detail=windows.evaluate(ap['window'],d,spins,wc,dref)
            qk=None if mode=='E0' else x+extra@p['qk_extra']
            result=attention.screening_attention(ap,x,d,pair,options=core_options,
                trim_max=config['trim_max'],trim_pow=config['trim_pow'],win_shape='cos',freeze_w=False,
                clip_r=None,clip_beta=.3,return_aux=return_aux,window=m,qk_stream=qk,gate_stream=gate_stream)
            if return_aux:
                out,aux=result;record=dict(input=x,attention=aux,window_details=detail,attention_output=out)
            else:out=result
            x=x+out
            if return_aux:record['attention_residual']=x
            if config['layer_norm']:x=ln_apply(p['ln'][l][0],x)
            if return_aux:record['ln1_output']=x
            mlp=mlp_apply(p['mlp'][l],x);x=x+mlp
            if return_aux:record.update(mlp_output=mlp,mlp_residual=x)
            if config['layer_norm']:x=ln_apply(p['ln'][l][1],x)
            if return_aux:record['output']=x;records.append(record)
        return (x,tuple(records),f) if return_aux else x

    if config.get('hf'):
        k,G,c=lj_orbitals.load_hf_table(config['hf'])
        init,orbital=lj_orbitals.make_lj_orbitals((N,0),jnp.zeros(1),options,(init_layers,apply_layers),
            k,G,c,config['lam'],lj_orbitals.cusp_values(config['rs'],config['lam'],config['delta']),
            config['net_scale'],config['jastrow'])
    else:
        init,orbital=networks.make_orbitals((N,0),jnp.zeros(1),options,(init_layers,apply_layers))

    def apply(p,pos,spins,atoms,charges):return network_blocks.logdet_matmul(orbital(p,pos,spins,atoms,charges))
    def diagnose(p,pos,spins,atoms,charges):
        ae,ee,ra,re=geometry(pos,atoms,ndim=2)
        return apply_layers(p['layers'],ae=ae,r_ae=ra,ee=ee,r_ee=re,spins=spins,charges=charges,return_aux=True)
    return networks.Network(options=options,init=init,apply=apply,orbitals=orbital),diagnose,dref
