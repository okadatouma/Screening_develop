"""Factories are task-aware; incompatible Hilbert spaces are never interchanged."""
def make_embedding(config, task='continuous_electrons', physics=None, model=None, twist=False):
    if task=='continuous_electrons':
        from .embeddings.periodic import PeriodicEmbedding
        return PeriodicEmbedding(config,physics,model,twist)
    if task=='qwz_topology':
        from .embeddings.phase import PhaseEmbedding
        return PhaseEmbedding(config,model)
    if task=='lattice_tfim':
        if config['type']!='baseline':raise ValueError('only original TFIM embedding supported')
        return None
    raise ValueError(task)


def make_attention(config, task='continuous_electrons'):
    if task=='continuous_electrons':
        from .attentions.screening import ScreeningFactory
        return ScreeningFactory(config)
    if task=='qwz_topology':
        from .attentions.phase import PhaseAttention
        return lambda width,heads:PhaseAttention(config,width,heads)
    if task=='lattice_tfim':return config
    raise ValueError(task)


def make_wavefunction(config, task='continuous_electrons', **context):
    if task=='continuous_electrons':
        from .wavefunctions.continuous import ContinuousWavefunction
        return ContinuousWavefunction(config,**context)
    if task=='lattice_tfim':
        from .task_adapters.tfim import build_model
        return build_model(context['full_config'])
    if task=='qwz_topology':
        from .wavefunctions.qwz import QWZNetwork
        return QWZNetwork(context['full_config'])
    raise ValueError(task)


def make_task(config):
    from .config import validate_config
    c=validate_config(config);task=c['experiment']['task']
    if task=='continuous_electrons':
        from .task_adapters.continuous import ContinuousTask
        return ContinuousTask(c)
    if task=='lattice_tfim':
        from .task_adapters.tfim import TFIMTask
        return TFIMTask(c)
    from .task_adapters.qwz import QWZTask
    return QWZTask(c)
