from pathlib import Path
import json
from types import SimpleNamespace
import pytest
from research_framework import cluster
from research_framework.paths import ROOT


def frozen(tmp_path):
    plan=ROOT/'research_framework/results/2026-11-06_production_plan/plan.json'
    return cluster.prepare(plan,tmp_path/'cluster')


def test_freeze_keeps_original_plan_and_immutable_hf(tmp_path):
    root=frozen(tmp_path)
    old=json.loads((root/'original_plan.json').read_text());new=json.loads((root/'plan.json').read_text())
    assert len(new)==7 and not old[6]['executable'] and not new[6]['executable']
    assert old[6]['config']['physics']['hf_path']!=new[6]['config']['physics']['hf_path']
    assert Path(old[6]['config']['physics']['hf_path']).read_bytes()==Path(new[6]['config']['physics']['hf_path']).read_bytes()
    cluster.verify(root)
    target=root/'source/Screening_topology_gpu/research_framework/config.py'
    with target.open('a') as f:f.write('\n# tampered snapshot\n')
    with pytest.raises(ValueError,match='frozen source/input changed'):cluster.verify(root)


def test_submission_dependencies_and_duplicate_guard(tmp_path,monkeypatch):
    root=frozen(tmp_path);calls=[]
    def fake(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout=str(9000+len(calls))+'\n',returncode=0)
    monkeypatch.setattr(cluster.subprocess,'run',fake)
    jobs=cluster.submit(root)
    assert jobs==dict(tfim_array='9001',n36_pilot='9002',n36_production='9003',collect='9004')
    assert '--dependency=afterok:9002' in calls[2]
    assert '--dependency=afterany:9001:9003' in calls[3]
    assert '--kill-on-invalid-dep=yes' in calls[2]
    with pytest.raises(FileExistsError):cluster.submit(root)
    assert len(calls)==4


def test_worker_cannot_bypass_missing_n36_gate(tmp_path,monkeypatch):
    root=frozen(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','1234')
    with pytest.raises(FileNotFoundError):cluster.worker(root,'n36')
    assert not (root/'electron_N36_archived_geometry').exists()
