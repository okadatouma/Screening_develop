"""Post-training causal communication control for the new holonomy benchmark."""
import copy
from pathlib import Path
import json
import numpy as np
import torch
from .holonomy import HolonomyModel,inputs
from .qwz_run import score
from ..experiments.store import csv_save,json_save


def analyze(root):
    root=Path(root);torch.set_num_threads(2)
    a=np.load(root/'test.npz')
    b={k:torch.tensor(a[k]) for k in ['z','carrier','coords','valid','distance','chart','target']}
    rows=[]
    for p in sorted(root.glob('*/final.pt')):
        folder=p.parent;summary=json.loads((folder/'summary.json').read_text())
        kind=summary['kind'];m=HolonomyModel(kind)
        m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'])
        if not len(m.blocks):continue
        local=copy.deepcopy(m)
        for block in local.blocks:
            block.config=copy.deepcopy(block.config)
            block.config['window'].update(mode='fixed',shape='hard',width=1.)
        restricted=score(local,b)
        rows.append(dict(kind=kind,seed=summary['seed'],baseline_mse=summary['after']['mse'],
                         all_layers_local_mse=restricted['mse'],delta_mse=restricted['mse']-summary['after']['mse']))
    csv_save(root/'all_layers_local_control.csv',rows)
    return rows
