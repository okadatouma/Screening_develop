"""Read-only artifact audit; run from repository root with .venv-pw/bin/python."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np

root=Path.cwd();sa=root/'screening_analysis';out=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
old=load(sa/'topology_results_2026-10-26_final/pilot.json')['original_checkpoint_sha256']
checks['preexisting_checkpoint_count']=len(old)
checks['preexisting_checkpoint_mismatches']=[p for p,h in old.items() if hashfile(Path(p))!=h]
assert len(old)==563 and not checks['preexisting_checkpoint_mismatches']
inv=load(sa/'topology_benchmark_2026-10-26/checkpoint_inventory.json')
checks['catalog_count']=inv['total'];checks['verified_phase_count']=inv['verified_phase_count']
checks['unreadable_checkpoints']=[r['path'] for r in inv['checkpoints'] if 'error' in r]
checks['catalog_checkpoint_mismatches']=[r['path'] for r in inv['checkpoints'] if hashfile(Path(r['path']))!=r['checkpoint_sha256']]
assert inv['total']==1334 and inv['verified_phase_count']==0 and not checks['unreadable_checkpoints'] and not checks['catalog_checkpoint_mismatches']
g=load(sa/'topology_g_results_2026-10-26/summary.json');b=load(sa/'topology_benchmark_2026-10-26/summary.json')
assert len(g)==21 and all(r['status']=='complete' and r['nan_steps']==0 for r in g)
assert len(b)==35 and all(r['status']=='complete_short_unconverged' and r['returncode']==0 for r in b)
checks.update(free_system_runs=len(g),lambda_runs=len(b),updates=sum(r['iterations'] for r in g+b))
assert checks['updates']==2052
training_rows=0
for rows,parent,file in [(g,sa/'topology_g_results_2026-10-26','stats.csv'),(b,sa/'topology_benchmark_2026-10-26','train_stats.csv')]:
 for row in rows:
  records=list(csv.DictReader((parent/row['id']/file).open()))
  assert len(records)==row['iterations']
  keys=['energy','objective','variance_abs_complex'] if file=='stats.csv' else ['energy','locstd']+(['objective','topology_penalty'] if row['mode']!='off' else [])
  assert all(np.isfinite(float(r[k])) for r in records for k in keys)
  training_rows+=len(records)
checks['finite_training_rows']=training_rows
source={}
for folder in ['topology_g_results_2026-10-26','topology_benchmark_2026-10-26']:
 values=load(sa/folder/'source.json');mismatch=[p for p,h in values.items() if hashfile(root/p)!=h];source[folder]=dict(files=len(values),mismatches=mismatch)
 assert not mismatch
checks['training_source_hashes']=source
assert load(out.parent/'source.json')['sha256']==hashfile(sa/'report_topology_benchmark.py')
records=load(sa/'topology_benchmark_analysis_2026-10-26_final/records.json');assert len(records)==76
core=[r for r in records if r['kind']=='core'];assert len(core)==48
shared=[r for r in core if r['view']=='B'];assert len({r['input_sha256'] for r in shared})==1
for feature in ('density','gr','structure_factor','distances'):
 values=[r['features_by_count']['32'][feature] for r in shared]
 assert all(np.array_equal(values[0],v) for v in values[1:])
assert all(r['phase']=='unknown' and not r['phase_verified'] for r in records)
checks['evaluated_models']=len(records)//2;checks['primary_models']=len(core)//2
checks['shared_core_input_sha256']=shared[0]['input_sha256']
classify=load(sa/'topology_benchmark_analysis_2026-10-26_final/classification.json')['candidate_condition_results']
assert len(classify)==192
for r in classify:
 assert r['test_units']==6 and r['independent_groups']==3 and r['permutation']['exact'] and r['permutation']['permutations']==8
 assert r['permutation']['pvalue']>=.25 and r['holm_pvalue']==1
 subset=[v for v in core if v['view']==r['view'] and v['mode']==r['regularizer']]
 for fold in r['folds']:
  train={subset[i]['seed_group'] for i in fold['train_indices']};test={subset[i]['seed_group'] for i in fold['test_indices']}
  assert not train&test
checks['seed_disjoint_classification_tests']=len(classify)
stress=load(sa/'topology_benchmark_analysis_2026-10-26_final/stress_comparisons.json')
assert all(r['reference_updates']==12 for r in stress if 'reference_updates' in r)
assert any(r['id']=='stress-N4' and 'different N' in r['status'] for r in stress)
sym=load(sa/'topology_benchmark_symmetry_2026-10-26_final/summary.json')
assert len(sym['results'])==38
assert sym['source_sha256']==hashfile(sa/'benchmark_symmetry.py')
checks['ewald_c3_max_difference']=max(r['ewald_C3_max_difference'] for r in sym['results'])
checks['c3_residual_range']=[min(r['eigenstate_candidate_residual'] for r in sym['results']),max(r['eigenstate_candidate_residual'] for r in sym['results'])]
svgs=list(out.parent.glob('*.svg'))
for path in svgs:ET.parse(path)
checks['svg_xml_valid']=[p.name for p in svgs];checks['raster_visual_QA']='not performed'
logs=['topology-g-all45.txt','topology-g-existing-probe.txt','topology-g-snapshot64.txt','topology-g-snapshot32.txt','topology-g-protocol-tests.txt']
for name in logs:shutil.copyfile(Path('/tmp')/name,out/name)
assert 'Ran 45 tests' in (out/logs[0]).read_text() and '\nOK\n' in (out/logs[0]).read_text()
assert 'Ran 6 tests' in (out/logs[1]).read_text() and '\nOK\n' in (out/logs[1]).read_text()
assert '数値基準との比較に成功' in (out/logs[2]).read_text() and '数値基準との比較に成功' in (out/logs[3]).read_text()
checks['test_logs']={name:hashfile(out/name) for name in logs}
checks['status']='passed; physical phases and actual Screenformer Chern remain UNVERIFIED'
(out/'verification.json').write_text(json.dumps(checks,indent=2,allow_nan=False))
print(json.dumps(checks,indent=2))
