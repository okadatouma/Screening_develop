"""Strict YAML/JSON schema and task compatibility checks before model creation."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import yaml

DEFAULTS = dict(
    experiment=dict(name='pilot', task='lattice_tfim', seed=101, stage='pilot'),
    physics=dict(n=6,J=1.,g=1.,lattice=[[4.,0.],[0.,4.]],crystal_supercell=[[2,0],[0,2]],
                 lam=0.,delta=1.,rs=1.,potential='none',hf_path=None,twist=[0.,0.],nk=8,
                 train_masses=[-3.,-1.4,-1.,-.6,.6,1.,1.4,3.],test_masses=[2.8,.8,-1.2]),
    model=dict(width=12,heads=2,head_dim=4,depth=1,perceptrons=1,layer_norm=False,
               capacity_mode='natural',embedding_budget=20),
    embedding=dict(type='baseline',spin_encoding='scalar',modes=[[1,0],[0,1],[1,-1]],
                   amplitude_initial=1.,sparse_penalty=0.,normalize_projection=True),
    attention=dict(type='screening',phase_transport=False,window=dict(mode='fixed',shape='hard',
                   width='full',lower=.05,upper=4.,temperature=.25,regularization=0.)),
    wavefunction=dict(type='baseline',complex_phase=True,jastrow=False,net_scale=.05),
    training=dict(method='exact',steps=150,learning_rate=.01,checkpoint_every=50,
                  walkers=16,burn_in=8,sweeps=2,gradient_clip=5.),
    evaluation=dict(samples=32,chains=4,burn_in=16,thinning=2,probe_samples=8,ph=True,
                    interventions=True,phase_samples=8),
    analysis=dict(energy=True,attention=True,persistent_homology=True,structure_factor=False,
                  phase_sensitivity=True,entanglement=True,berry=False))


def _merge(default, supplied, prefix=''):
    if not isinstance(supplied,dict): raise ValueError(f'{prefix or "config"}: mapping required')
    unknown=set(supplied)-set(default)
    if unknown: raise ValueError(f'unknown fields {prefix}: {sorted(unknown)}')
    out=deepcopy(default)
    for key,value in supplied.items():
        out[key]=_merge(default[key],value,prefix+key+'.') if isinstance(default[key],dict) else deepcopy(value)
    return out


def _positive(value,name,integer=False,zero=False):
    if isinstance(value,bool) or (integer and type(value) is not int) or not isinstance(value,(float,int)) or not np.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name}: finite {"integer " if integer else ""}{"nonnegative" if zero else "positive"} required')


def validate_config(supplied):
    c=_merge(DEFAULTS,supplied)
    e,p,m,b,a,w,t,v,x=(c[k] for k in ['experiment','physics','model','embedding','attention','wavefunction','training','evaluation','analysis'])
    task=e['task'];win=a['window']
    if task not in ['continuous_electrons','lattice_tfim','qwz_topology']: raise ValueError('unknown task')
    if not isinstance(e['name'],str) or not e['name'] or '/' in e['name'] or '..' in e['name']:raise ValueError('safe nonempty experiment name required')
    _positive(e['seed'],'seed',True,True)
    if e['seed']>=2**31:raise ValueError('seed must be <2**31')
    for key in ['n','nk']: _positive(p[key],key,True)
    for key in ['width','heads','head_dim','depth','perceptrons','embedding_budget']:_positive(m[key],key,True)
    if m['capacity_mode'] not in ['natural','padded']:raise ValueError('capacity_mode must be natural or padded')
    for group in [x]:
        if any(type(value) is not bool for value in group.values()):raise ValueError('analysis flags must be boolean')
    for key in ['ph','interventions']:
        if type(v[key]) is not bool:raise ValueError(f'evaluation.{key} must be boolean')
    v['ph']=v['ph'] and x['persistent_homology']
    for key in ['complex_phase','jastrow']:
        if type(w[key]) is not bool:raise ValueError(f'{key} must be boolean')
    if b['spin_encoding'] not in ['scalar','onehot']:raise ValueError('invalid spin encoding')
    if b['type'] not in ['baseline','six_theta','spin_onehot','six_theta_onehot','learned_fourier','complex','gauge_covariant','projector']:raise ValueError('unknown embedding')
    modes=np.asarray(b['modes'])
    if modes.ndim != 2 or modes.shape[1]!=2 or not len(modes) or not np.isfinite(modes).all() or not np.array_equal(modes,np.rint(modes)) or np.any(np.all(modes==0,axis=1)):
        raise ValueError('Fourier modes must be nonzero integer reciprocal indices [Q,2]')
    if len({tuple(q) for q in modes}) != len(modes):raise ValueError('duplicate Fourier modes')
    if a['type'] not in ['screening','real','complex','gauge_covariant']:raise ValueError('unknown attention type')
    if win['mode'] not in ['baseline','fixed','full','trainable','trainable_per_head']:raise ValueError('unknown Window mode')
    if win['shape'] not in ['hard','cos','smooth7','sigmoid','gaussian']:raise ValueError('unknown Window shape')
    for key in ['temperature','upper']:_positive(win[key],f'window.{key}')
    _positive(win['lower'],'window.lower',zero=True)
    if win['upper']<=win['lower']:raise ValueError('window upper must exceed lower')
    if win['width']!='full':_positive(win['width'],'window.width')
    trainable=win['mode'].startswith('trainable')
    if trainable and (win['width']=='full' or not win['lower']<win['width']<win['upper'] or win['shape']=='hard'):raise ValueError('trainable width needs interior initialization and differentiable shape')
    _positive(win['regularization'],'window.regularization',zero=True)
    if win['regularization'] and not trainable:raise ValueError('Window regularization requires trainable mode')
    _positive(b['sparse_penalty'],'sparse_penalty',zero=True)
    if b['sparse_penalty'] and b['type']!='learned_fourier':raise ValueError('Fourier penalty requires learned Fourier')
    for key in ['steps','checkpoint_every','walkers','burn_in','sweeps']:_positive(t[key],f'training.{key}',True)
    for key in ['learning_rate','gradient_clip']:_positive(t[key],key)
    for key in ['samples','chains','burn_in','thinning','probe_samples','phase_samples']:_positive(v[key],f'evaluation.{key}',True)
    if task=='continuous_electrons':
        if a['type']!='screening':raise ValueError('continuous adapter preserves original Screening; complex/covariant attention belongs to QWZ adapter')
        if b['type'] in ['gauge_covariant','projector']:raise ValueError('Bloch projector/covariance is not an electron-position embedding')
        if win['shape'] in ['hard','sigmoid'] and win['mode'] not in ['baseline','full']:raise ValueError('continuous kinetic derivatives require origin-smooth cos/smooth7/gaussian Window')
        if w['type'] not in ['baseline','hf_nn_jastrow','hf','hf_jastrow','complex','twist_dependent']:raise ValueError('incompatible continuous Wavefunction')
        if t['method']!='vmc':raise ValueError('continuous task requires VMC training, never uniform-input energy averages')
        if x['entanglement'] or x['berry']:raise ValueError('continuous entanglement/Berry need separate state-family estimators; unavailable in this adapter')
        lat=np.asarray(p['lattice'],float);crystal=np.asarray(p['crystal_supercell'])
        if lat.shape!=(2,2) or not np.isfinite(lat).all() or abs(np.linalg.det(lat))<1e-10:raise ValueError('invertible column-vector lattice required')
        if crystal.shape!=(2,2) or not np.array_equal(crystal,np.rint(crystal)) or abs(np.linalg.det(crystal))<1:raise ValueError('integer invertible crystal supercell required')
        if w['type'].startswith('hf') and p['hf_path'] is not None and not Path(p['hf_path']).is_file():raise ValueError('HF table missing')
        if p['potential'] not in ['none','Coulomb']:raise ValueError('unsupported potential')
        if not w['complex_phase'] and (p['lam']!=0 or w['type'].startswith('hf') or w['type']=='twist_dependent'):raise ValueError('spin-orbit/HF/twist family requires complex orbital determinants')
        if np.any(np.asarray(p['twist'])!=0) and w['type']!='twist_dependent':raise ValueError('nonzero boundary twist requires twist_dependent wavefunction')
    elif task=='lattice_tfim':
        if not 2<=p['n']<=16:raise ValueError('TFIM exact diagnostics require 2<=N<=16; N>16 estimators are not implemented in this adapter')
        if b['type']!='baseline' or a['type']!='screening' or w['type']!='baseline':raise ValueError('TFIM adapter supports original site embedding and positive real lattice wavefunction only')
        if b['spin_encoding']!='scalar' or a['phase_transport']:raise ValueError('unsupported TFIM spin/phase representation')
        if x['berry'] or x['structure_factor']:raise ValueError('select TFIM correlation/entanglement diagnostics instead')
        if win['shape'] not in ['hard','sigmoid']:raise ValueError('TFIM uses existing hard or sigmoid Window')
        if win['shape']=='hard' and win['width']!='full' and type(win['width']) is not int:raise ValueError('TFIM hard radius must be an integer in lattice sites')
        if trainable and win['mode']!='trainable_per_head':raise ValueError('existing TFIM trainable Window is per-head')
        if trainable and (win['lower']!=0 or win['upper']!=p['n']//2):raise ValueError('TFIM existing bound is [0,floor(N/2)]')
        if t['method'] not in ['exact','vmc'] or (t['method']=='exact' and p['n']>10):raise ValueError('TFIM exact-training guard N<=10; use existing VMC for larger N')
    else:
        if m['width']%m['heads']:raise ValueError('QWZ width must be divisible by heads')
        if p['nk']<4 or p['nk']>32:raise ValueError('QWZ mesh guard: 4<=nk<=32')
        if v['ph'] and p['nk']**2*v['phase_samples']>1600:raise ValueError('phase-lift PH guard is 1600 points; reduce nk/phase_samples or disable PH')
        if b['type'] not in ['baseline','projector','complex','gauge_covariant']:raise ValueError('electron crystal/spin embedding does not apply to QWZ')
        if a['type']=='screening':raise ValueError('QWZ selects real/complex/gauge_covariant attention explicitly')
        if w['type'] not in ['projector','local_atlas']:raise ValueError('QWZ outputs projector or local_atlas, not many-electron determinants')
        if t['method']!='denoise' or x['energy'] or x['entanglement'] or x['structure_factor']:raise ValueError('QWZ denoises states; no many-body energy/entanglement diagnostics')
        if a['type']=='gauge_covariant' and b['type']!='gauge_covariant':raise ValueError('charged covariant attention requires charged covariant embedding')
        if b['type']=='gauge_covariant' and a['type']!='gauge_covariant':raise ValueError('covariant embedding requires covariant transport throughout output projection')
        if a['type']=='complex' and b['type']!='complex':raise ValueError('native complex attention requires complex embedding')
        if a['phase_transport'] != (a['type']=='gauge_covariant'):raise ValueError('phase_transport must match gauge_covariant attention')
        if w['type']=='local_atlas' and b['type']=='projector':raise ValueError('atlas regression requires explicit chart inputs; choose baseline/complex/covariant')
        if win['mode']=='baseline':raise ValueError('QWZ has no inherited Screening Window')
        if any(abs(float(mass))<1e-8 or abs(abs(float(mass))-2)<1e-8 for mass in p['train_masses']+p['test_masses']):raise ValueError('gapless QWZ masses not allowed')
    return c


def load_config(path):
    path=Path(path)
    return validate_config(json.loads(path.read_text()) if path.suffix=='.json' else yaml.safe_load(path.read_text()))
