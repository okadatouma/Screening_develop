"""Resolve actual neighboring projects, verify imports, and record source origins."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECTS=ROOT.parent


def add_source(name, environment):
    path=Path(os.environ.get(environment,PROJECTS/name)).resolve()
    if not path.is_dir():raise FileNotFoundError(f'{name}: set {environment} to source checkout')
    if str(path) not in sys.path:sys.path.append(str(path))
    return path


def lattice_source():return add_source('Lattice_screening','RF_LATTICE_SOURCE')
def wannier_source():return add_source('wannier_ph','RF_WANNIER_SOURCE')
