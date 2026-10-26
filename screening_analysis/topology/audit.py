"""Read-only source/checkpoint audit and final evidence index."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from .workflow import ROOT,provenance,sha,write


def main():
    sa=ROOT/'screening_analysis';out=ROOT/'results/screening_topology';load=lambda p:json.loads(p.read_text())
    evidence={}
    original=load(sa/'topology_results_2026-10-26_final/pilot.json')
    hashes=original['original_checkpoint_sha256']
    assert len(hashes)==563 and all(sha(p)==h for p,h in hashes.items())
    evidence['original_563_checkpoint_hashes']='all unchanged'
    previous=load(sa/'topology_benchmark_2026-10-26/checkpoint_inventory.json')
    assert previous['total']==1334 and all(sha(r['path'])==r['checkpoint_sha256'] for r in previous['checkpoints'])
    evidence['previous_1334_checkpoint_hashes']='all unchanged'
    oldph=next(v for p,v in original['trained_checkpoint_runs'][0]['source_sha256'].items() if p.endswith('/topology.py'))
    assert oldph==sha(sa/'topology/persistence.py');evidence['PH_module_move']='byte-identical implementation'
    for folder in ('topology_g_results_2026-10-26','topology_benchmark_2026-10-26'):
        hashes=load(sa/folder/'source.json');assert all(sha(ROOT/p)==h for p,h in hashes.items())
    evidence['production_training_source_hashes']='unchanged since G/lambda experiments'
    catalog=load(out/'scan/inventory.json');assert catalog['total']==1336
    assert not [r for r in catalog['checkpoints'] if 'error' in r]
    assert all(sha(r['path'])==r['checkpoint_sha256'] for r in catalog['checkpoints'])
    evidence['checkpoint_catalog']=dict(total=1336,read_errors=0,verified_phase_count=catalog['verified_phase_count'])
    status=load(out/'smoke-release/status.json');log=(out/'smoke-release/tests.txt').read_text()
    assert status['returncode']==0 and 'Ran 51 tests' in log and '\nOK\n' in log;evidence['selected_unit_tests']=51
    assert len(list((out/'benchmark').glob('*/complete.json')))==12
    assert len(list((out/'layer-stress').glob('*/complete.json')))==1
    assert len(load(out/'benchmark/summary.json'))==12
    evidence['new_AB_CDE']=dict(core_views=12,configurations_per_view=8,layer_stress_views=1)
    comparison=load(out/'internal-comparison-final/metrics.json');assert len(comparison['results'])==72
    for r in comparison['results']:
        assert r['independent_groups']==3 and r['test_units']==6 and r['holm_pvalue']==1
        assert r['permutation']['pvalue']>=.25
    evidence['internal_condition_comparisons']=72
    physical=load(out/'physical-candidates/summary.json');assert len(physical['results'])==4
    assert all(r['samples']==16 and r['chains']==4 and r['triangles']==144 and not r['phase_verified'] for r in physical['results'])
    evidence['actual_N9_1RDM']=dict(models=2,independent_training_seeds=1,quadrature_sides=[4,6],samples_per_model=16,physical_phase='unverified')
    continuation=[]
    for directory in [out/'continuation-smoke-final',out/'job-smoke/continue-one-update/run']:
        with np.load(directory/'qmcjax_ckpt_000024.npz',allow_pickle=True) as z:
            assert int(z['t'])==24
        continuation.append(dict(path=str(directory),checkpoint_sha256=sha(directory/'qmcjax_ckpt_000024.npz')))
    evidence['continuation_updates']=continuation
    evidence['main_training_updates']=2052;evidence['continuation_smoke_updates']=2
    evidence['scaleup']=dict(jobs=len(load(out/'scaleup-plan/dry-run.json')['jobs']),executed=False)
    svg=list(out.rglob('*.svg'))
    for p in svg:ET.parse(p)
    evidence['svg_xml_valid_count']=len(svg);evidence['raster_visual_QA']='not performed'
    docs=[sa/'topology/RESEARCH_REPORT.md',out/'README.md']
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in target and (p.parent/target).resolve()!=out/'validation.json':assert (p.parent/target).exists(),(p,target)
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True);assert check.returncode==0,check.stdout
    evidence['git_diff_check']='passed'
    logs=out/'validation-logs';logs.mkdir(exist_ok=True)
    for p in ('topology-g-existing-probe.txt','topology-g-snapshot64.txt','topology-g-snapshot32.txt'):
        source=sa/'topology_g_benchmark_report_2026-10-26_final/validation'/p
        shutil.copyfile(source,logs/p)
    for src,dest in [('/tmp/topology-continuation.txt','continuation-initial-failure.txt'),('/tmp/topology-continuation-final.txt','continuation-success.txt'),('/tmp/topology-job-smoke.txt','cli-job-success.txt'),('/tmp/topology-physical-candidates.txt','physical-candidates.txt')]:
        shutil.copyfile(src,logs/dest)
    evidence['provenance']=provenance(20261026)
    evidence['status']='passed within stated scope; physical WC/AHC and actual Screenformer Chern unverified'
    write(out/'validation.json',evidence);assert (out/'validation.json').exists();print(json.dumps({k:v for k,v in evidence.items() if k!='provenance'},indent=2))


if __name__=='__main__':main()
