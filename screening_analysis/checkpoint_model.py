"""Read-only known-factory reconstruction for fresh sampling, not a new forward."""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from periodicwave import CustomScreenformer as CS
from screening_analysis.attn_probe import _effective_settings,lattice_from_config,SETTINGS
from screening_analysis.checkpoint_catalog import find_config


def _factory(name):
    if not name.startswith('periodicwave.'):raise ValueError('only repository factories supported')
    module,attribute=name.rsplit('.',1);return getattr(importlib.import_module(module),attribute)


@contextmanager
def load_model(folder,checkpoint=None,settings=None,lattice=None):
    folder=Path(folder);cfg,config_path=find_config(folder)
    if not cfg or 'system' not in cfg:raise ValueError('exact structured network/system config required')
    if cfg['network']['network_type']!='CustomScreenformer':raise ValueError('this loader targets standard CustomScreenformer checkpoints')
    effective,provenance=_effective_settings(cfg,settings,None)
    if isinstance(cfg['system']['pbc_lattice'],str) and lattice is None:raise ValueError('original precision lattice required; no rounded reconstruction')
    L=lattice_from_config(cfg) if lattice is None else np.asarray(lattice)
    path=Path(checkpoint) if checkpoint else sorted(folder.glob('qmcjax_ckpt_*.npz'))[-1]
    globals_before={key:getattr(CS,key) for key in SETTINGS[:6]}
    try:
        for key in SETTINGS[:6]:setattr(CS,key,effective[key])
        with jax.experimental.enable_x64(effective['jax_enable_x64']),jax.default_matmul_precision(effective['jax_default_matmul_precision']):
            with np.load(path,allow_pickle=True) as z:raw=z['params'].item();data=z['data'].item();step=int(z['t'])
            devices=np.asarray(data['positions']).shape[0]
            for leaf in jax.tree.leaves(raw):
                a=np.asarray(leaf)
                if a.ndim<1 or a.shape[0]!=devices or any(not np.array_equal(a[0],v) for v in a[1:]):
                    raise ValueError('parameter replicas disagree')
            params=jax.tree.map(lambda a:jnp.asarray(np.asarray(a)[0]),raw)
            data={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in data.items()}
            nspins=tuple(cfg['system']['electrons']);network=cfg['network'];ndim=cfg['system']['ndim']
            feature_kwargs=dict(network.get('make_feature_layer_kwargs',{}))
            if 'lattice' in feature_kwargs:
                if isinstance(feature_kwargs['lattice'],str):raise ValueError('exact feature lattice required')
                feature_kwargs['lattice']=jnp.asarray(feature_kwargs['lattice'])
            fl=_factory(network['make_feature_layer_fn'])(natoms=data['charges'].shape[-1],nspins=nspins,ndim=ndim,**feature_kwargs)
            envelope=_factory(network['make_envelope_fn'])(**network.get('make_envelope_kwargs',{}))
            net=CS.make_fermi_net(nspins,jnp.asarray(data['charges'][0]),ndim=ndim,
                determinants=network['determinants'],feature_layer=fl,envelope=envelope,
                jastrow=network.get('jastrow','NONE'),jastrow_kwargs=network.get('jastrow_kwargs',{}),
                bias_orbitals=network.get('bias_orbitals',False),complex_output=network.get('complex',False),
                pbc_lattice=jnp.asarray(L),**dict(network['CustomScreenformer']))
            info=dict(config=cfg,config_path=str(config_path),checkpoint=str(path),step=step,lattice=L,
                settings=effective,settings_provenance=provenance,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            yield net,params,data,info
    finally:
        for key,value in globals_before.items():setattr(CS,key,value)
