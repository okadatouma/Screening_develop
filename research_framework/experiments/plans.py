"""One-change-at-a-time pilots and unsubmitted production plans."""
from copy import deepcopy
import json
from pathlib import Path
from ..config import validate_config
from .store import json_save,digest,source_hashes


def continuous_base():
    return validate_config(dict(experiment=dict(name='continuous_baseline',task='continuous_electrons'),
        physics=dict(n=2,lam=0.,delta=1.,lattice=[[4.,0.],[0.,4.]],crystal_supercell=[[2,0],[0,2]]),
        model=dict(width=8,heads=2,head_dim=2),
        attention=dict(type='screening',window=dict(mode='fixed',shape='cos',width=1.5,upper=3.)),
        wavefunction=dict(type='hf_nn_jastrow',complex_phase=True,jastrow=False),
        training=dict(method='vmc',steps=12,learning_rate=.002,checkpoint_every=6,walkers=12,burn_in=12,sweeps=2),
        evaluation=dict(samples=16,chains=4,burn_in=24,thinning=2,probe_samples=8),
        analysis=dict(entanglement=False,structure_factor=True)))


def qwz_base():
    return validate_config(dict(experiment=dict(name='qwz_baseline',task='qwz_topology'),
        physics=dict(nk=8),model=dict(width=12,heads=2),
        attention=dict(type='real',window=dict(mode='full',shape='gaussian',width='full',upper=5.)),
        wavefunction=dict(type='projector'),training=dict(method='denoise',steps=150,learning_rate=.003),
        analysis=dict(energy=False,entanglement=False,berry=True)))


def pilot_cases():
    cases=[]
    def add(base,name,group,changes=None,seed=101):
        c=deepcopy(base);c['experiment'].update(name=name,seed=seed,stage=group)
        for section,values in (changes or {}).items():
            for key,value in values.items():
                if isinstance(value,dict):c[section][key].update(value)
                else:c[section][key]=value
        c=validate_config(c);identifier=digest(json.dumps(c,sort_keys=True).encode())[:10]
        cases.append(dict(name=f'{name}_s{seed}_{identifier}',group=group,config=c))
    base=continuous_base()
    for seed in [101,202]:
        for kind in ['baseline','six_theta','spin_onehot','six_theta_onehot','learned_fourier']:
            add(base,'electron_'+kind,'B_embedding',{'embedding':{'type':kind}},seed)
    for kind in ['baseline','six_theta_onehot','learned_fourier']:
        add(base,'electron_padded_'+kind,'B_capacity',{'embedding':{'type':kind},'model':{'capacity_mode':'padded','embedding_budget':12}},101)
    for name,changes in [
        ('trainable',{'attention':{'window':{'mode':'trainable_per_head','width':1.5,'lower':.1,'upper':3.,'shape':'cos'}}}),
        ('locality',{'attention':{'window':{'mode':'trainable_per_head','width':1.5,'lower':.1,'upper':3.,'shape':'cos','regularization':.01}}}),
        ('hf',{'wavefunction':{'type':'hf','jastrow':False}}),
        ('complex_slater',{'wavefunction':{'type':'complex'}}),
        ('real_slater',{'wavefunction':{'type':'baseline','complex_phase':False}}),
        ('twist',{'wavefunction':{'type':'twist_dependent'},'physics':{'twist':[.25,-.15]}})]:
        add(base,'electron_'+name,'B_wave_attention',changes)
    for name,kind,jastrow in [('HF','hf',False),('HF_J','hf_jastrow',True),('HF_NN','hf_nn_jastrow',False),('HF_NN_J','hf_nn_jastrow',True)]:
        add(base,'electron_coulomb_'+name,'B_coulomb_ansatz',{'wavefunction':{'type':kind,'jastrow':jastrow},'physics':{'potential':'Coulomb','rs':1.}})
    tfim=validate_config(dict(physics=dict(n=8),training=dict(steps=180,learning_rate=.01,checkpoint_every=90)))
    for seed in [101,202]:
        for g in [1.,1.5]:
            for radius in [1,2,4,'full']:
                add(tfim,f'tfim_g{g}_R{radius}','C_window',{'physics':{'g':g},'attention':{'window':{'width':radius}}},seed)
            for label,mode,penalty in [('smooth_fixed','fixed',0.),('trainable','trainable_per_head',0.),('regularized','trainable_per_head',.01)]:
                add(tfim,f'tfim_g{g}_{label}','C_window',{'physics':{'g':g},'attention':{'window':{'mode':mode,'shape':'sigmoid','width':2.,'lower':0.,'upper':4.,'regularization':penalty}}},seed)
    add(tfim,'tfim_depth2','C_depth',{'model':{'depth':2}})
    add(tfim,'tfim_heads4','C_heads',{'model':{'heads':4,'head_dim':2}})
    base=qwz_base()
    variants=[('real',{}),('projector',{'embedding':{'type':'projector'}}),
        ('complex_twochannel',{'embedding':{'type':'complex'}}),
        ('native_complex',{'embedding':{'type':'complex'},'attention':{'type':'complex'}}),
        ('gauge_covariant',{'embedding':{'type':'gauge_covariant'},'attention':{'type':'gauge_covariant','phase_transport':True}}),
        ('atlas_real',{'wavefunction':{'type':'local_atlas'}}),
        ('atlas_complex',{'wavefunction':{'type':'local_atlas'},'embedding':{'type':'complex'},'attention':{'type':'complex'}}),
        ('atlas_covariant',{'wavefunction':{'type':'local_atlas'},'embedding':{'type':'gauge_covariant'},'attention':{'type':'gauge_covariant','phase_transport':True}})]
    for seed in [101,202]:
        for name,changes in variants:add(base,'qwz_'+name,'D_geometry',changes,seed)
    for label,mode,penalty in [('fixed','fixed',0.),('shared','trainable',0.),('per_head','trainable_per_head',0.),('locality','trainable_per_head',.01)]:
        add(base,'qwz_covariant_'+label,'D_window',{'embedding':{'type':'gauge_covariant'},
            'attention':{'type':'gauge_covariant','phase_transport':True,'window':{'mode':mode,'shape':'gaussian','width':2.,'lower':.1,'upper':5.,'regularization':penalty}}})
    return cases


def production_cases():
    """Explicit plans only; never submit to Slurm automatically."""
    cases=[]
    for n in [12,16]:
        for seed in [101,202,303]:
            c=validate_config(dict(experiment=dict(name=f'tfim_N{n}_long',seed=seed,stage='production'),physics=dict(n=n),
                training=dict(method='vmc',steps=10000,walkers=64,burn_in=128,sweeps=4,checkpoint_every=500),
                attention=dict(window=dict(mode='trainable_per_head',shape='sigmoid',width=2.,lower=0.,upper=n//2))))
            cases.append(dict(name=f'tfim_N{n}_s{seed}',group='unrun_production',config=c))
    root=Path(__file__).resolve().parents[2]
    reference=root/'results/head-count-sweep-20261031/heads-4/wc-h4-lam0p72-rs25-seed101/reference/orbitals.txt'
    if reference.exists():
        c=continuous_base();c['physics'].update(n=36,lam=.72,rs=25.,delta=25.,potential='Coulomb',hf_path=str(reference),crystal_supercell=[[6,0],[0,6]])
        # Real lattice and Hamiltonian MUST be taken from the archived candidate settings before execution.
        runtime=json.loads((reference.parents[1]/'training/runtime_config.json').read_text())
        c['physics']['lattice']=runtime['system']['pbc_lattice']
        c['experiment'].update(name='electron_N36_archived_geometry',stage='production')
        c['model'].update(width=32,heads=4,head_dim=8)
        c['training'].update(steps=10000,walkers=64,burn_in=256,checkpoint_every=500)
        cases.append(dict(name='electron_N36_archived_geometry',group='unrun_production',config=validate_config(c),
                          executable=False,reason='large continuum run: default-Hessian local energy is verified only at small N; schedule after memory/timing pilot and estimator benchmarking'))
    return cases


def make_plan(output,tier='pilot'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cases=pilot_cases() if tier=='pilot' else production_cases()
    json_save(root/'plan.json',cases);json_save(root/'source_hashes.json',source_hashes())
    json_save(root/'RESOURCE_PLAN.json',dict(tier=tier,cases=len(cases),backend='CPU pilot; production not submitted',
        maximum_concurrent_runs=3,threads_per_run=2,existing_jobs='untouched',per_case_timeout_seconds=600,
        H2='not requested; reference H1 only',note='Single-change and limited matched-dimension comparisons; not a full Cartesian product.'))
    return cases
