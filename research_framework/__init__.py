"""Composable research adapters; existing production defaults remain unchanged."""
from .config import load_config, validate_config
from .factory import make_embedding, make_attention, make_wavefunction, make_task

__all__ = ['load_config','validate_config','make_embedding','make_attention','make_wavefunction','make_task']
