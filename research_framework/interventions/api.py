"""One API for physical state interventions; evaluation remains physics-specific."""
def validate_intervention(config,spec):
    import math
    task=config['experiment']['task']
    allowed={
        'continuous_electrons':{'head_ablation','remove_connections','distance_band','window_width','remove_fourier','remove_six_theta','remove_spin','replace_spin','zero_phase_channel'},
        'lattice_tfim':{'head_ablation','remove_connections','distance_band','window_width','patch_connections'},
        'qwz_topology':{'head_ablation','remove_connections','distance_band','window_width','zero_phase_channel','disable_gauge_transport','patch_connections'}}[task]
    if not isinstance(spec,dict) or spec.get('type') not in allowed:raise ValueError('intervention incompatible with task')
    if type(spec.get('head',0)) is not int or type(spec.get('layer',0)) is not int:raise ValueError('head/layer must be integer indices')
    if not 0<=spec.get('head',0)<config['model']['heads'] or not 0<=spec.get('layer',0)<config['model']['depth']:raise ValueError('intervention head/layer out of range')
    if task=='continuous_electrons' and spec['type']=='remove_connections' and 'edges' in spec:raise ValueError('particle-index connection removal breaks fermionic equivariance; use a distance kernel')
    if spec['type']=='disable_gauge_transport' and config['attention']['type']!='gauge_covariant':raise ValueError('no gauge transport to disable')
    if spec['type']=='patch_connections' and task=='qwz_topology' and config['wavefunction']['type']!='local_atlas':raise ValueError('patches absent')
    if spec['type']=='zero_phase_channel' and task=='continuous_electrons' and not config['wavefunction']['complex_phase']:raise ValueError('real orbital ansatz has no imaginary orbital channel')
    for key in ['width','radius']:
        if key in spec and (isinstance(spec[key],bool) or not isinstance(spec[key],(int,float)) or not math.isfinite(spec[key]) or spec[key]<=0):raise ValueError('intervention width/radius must be finite and positive')
    return dict(spec)


def apply_intervention(config,model,params,spec):
    spec=validate_intervention(config,spec);task=config['experiment']['task']
    if task=='continuous_electrons':return model.intervene(params,spec)
    if task=='lattice_tfim':
        from ..task_adapters.tfim import intervene
        return intervene(model,params,spec)
    import torch
    class Intervened(torch.nn.Module):
        def __init__(self,base,spec):super().__init__();self.base=base;self.spec=spec;self.atlas=base.atlas
        def forward(self,batch,intervention=None,mask=None):
            if intervention:raise ValueError('compose interventions explicitly, not by silently overriding one')
            return self.base(batch,self.spec,mask)
    return Intervened(model,spec),params
