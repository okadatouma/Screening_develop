"""Re-evaluate a corrected intervention at immutable saved checkpoints, no retraining."""
import json
import pickle
import numpy as np
import torch
from ..experiments.retries import resolve
from ..experiments.store import json_save,source_hashes
from ..task_adapters.qwz import dataset,QWZTask
from ..factory import make_wavefunction
from ..interventions import apply_intervention


def corrected_width_interventions(root):
    torch.set_num_threads(2)
    records=[]
    for case in json.loads((root/'plan.json').read_text()):
        c=case['config']
        if c['experiment']['task']!='qwz_topology':continue
        path=resolve(root,case['name'])
        if not (path/'COMPLETE.json').exists():continue
        destination=path/'interventions_width_corrected.json'
        if destination.exists():continue
        checkpoint=path/json.loads((path/'latest.json').read_text())['checkpoint']
        with checkpoint.open('rb') as f:saved=pickle.load(f)
        model=make_wavefunction(c['wavefunction'],'qwz_topology',full_config=c);model.load_state_dict(saved['state']['model'])
        task=QWZTask(c);test=dataset(c,c['physics']['test_masses']);baseline,arrays=task.evaluate(model,test)
        old=np.load(path/'diagnostics/after_internal.npz')['projector']
        error=float(abs(arrays['projector']-old).max())
        if error>3e-6:raise ArithmeticError('default checkpoint predictions changed during width-only correction')
        spec=dict(type='window_width',width=1.5);target,_=apply_intervention(c,model,None,spec)
        current,changed=task.evaluate(target,test)
        fine=dataset(c,c['physics']['test_masses'],n=min(2*c['physics']['nk'],32));fine_rows,_=task.evaluate(target,fine)
        result=[]
        for row,ref,frow in zip(current,baseline,fine_rows):
            row['chern_mesh_stable']=bool(abs(row['chern']-frow['chern'])<1e-6)
            row['conditional_interpretation_screen']=bool(row['admissible_screen'] and frow['admissible_screen'] and row['chern_mesh_stable'])
            result.append(dict(intervention=spec,**row,delta_projector_mse=row['projector_mse']-ref['projector_mse'],
                               correction='full-window width override previously retained full window; original diagnostics preserved'))
        json_save(destination,result)
        json_save(path/'width_reanalysis_provenance.json',dict(source_hashes=source_hashes(),checkpoint=str(checkpoint),
                    baseline_max_difference=error,training_repeated=False,original_interventions_retained=True))
        np.savez_compressed(path/'diagnostics/width_corrected.npz',projector=changed['projector'],states=changed['states'])
        records.append(dict(run=case['name'],baseline_max_difference=error,rows=len(result)))
    json_save(root/'width_reanalysis_summary.json',records)
