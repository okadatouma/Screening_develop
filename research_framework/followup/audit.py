"""Read-only inventory taken before follow-up experiments."""
import json
import subprocess
from pathlib import Path
from ..experiments.store import digest, json_save
from ..paths import ROOT


def audit(destination):
    rf = ROOT/'research_framework'
    roots = [rf/'results/2026-11-06_pilot', rf/'results/2026-11-06_fourier_full_candidates']
    records = []
    for root in roots:
        for p in sorted(root.glob('*/summary.json')):
            value = json.loads(p.read_text())
            records.append(dict(path=str(p), sha256=digest(p.read_bytes()),
                                task=value.get('task'), n=value.get('n'), steps=value.get('steps')))
    # Only stable artifacts are protected: live cluster output necessarily changes.
    stable = [p for root in roots for p in root.rglob('*')
              if p.is_file() and 'source' not in p.parts and p.suffix in ['.pkl', '.csv', '.json', '.md']]
    stable += [p for p in rf.rglob('*.py')
               if not any(x in p.parts for x in ['results', '.venv', 'followup', '__pycache__'])]
    stable += list(rf.glob('*REPORT_JA.md'))
    inventory = dict(
        existing_summaries=records,
        protected_sha256={str(p): digest(p.read_bytes()) for p in sorted(set(stable))},
        cluster_status=subprocess.run(['squeue','-u','tomax','-o','%.18i %.30j %.9T %.10M'], capture_output=True,text=True).stdout,
        classifications=[
            dict(item='Embedding/Attention/Wavefunction switches', implementation='existing', verification='legacy pilots and saved tests; reuse'),
            dict(item='N=36 five-way matched-initial-state embedding comparison', implementation='components existing', verification='unverified; N=2 pilots excluded'),
            dict(item='N=36 baseline Slurm 2881', implementation='existing running experiment', verification='not a matched-init comparison'),
            dict(item='Fourier amplitude and sparse penalty', implementation='existing', verification='candidate-specific physical response unverified'),
            dict(item='QWZ projector/atlas/gauge covariance', implementation='existing', verification='local reconstruction verified; global processing unverified'),
            dict(item='PH/CMI/correlation length/S2/parity', implementation='existing', verification='archived evidence only; no recomputation in follow-up')],
        archive_references=[str(rf/'COMPARISON_REPORT_JA.md'),str(rf/'IMPLEMENTATION_REPORT_JA.md'),
                            str(rf/'results/2026-11-06_pilot/reports_corrected'),
                            str(ROOT/'results/analysis/physical-phase-diagnostics-20261102/summary.md'),
                            str(ROOT.parent/'wannier_ph/results/2026-11-05_initial')])
    json_save(destination, inventory)
    return inventory


def verify_protected(path):
    values=json.loads(Path(path).read_text())['protected_sha256']
    changed=[p for p,h in values.items() if not Path(p).is_file() or digest(Path(p).read_bytes())!=h]
    if changed: raise RuntimeError(f'Existing artifacts changed: {changed}')
    return len(values)
