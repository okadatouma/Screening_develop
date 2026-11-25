"""Capacity controls use the EXACT saved main-benchmark data and fixed budget."""
import json
from pathlib import Path
import numpy as np
import torch
from .holonomy import inputs,subset,HolonomyModel,gauge
from .controls import GlobalMLP
from .qwz_run import score,snapshot
from ..experiments.store import json_save,csv_save,source_hashes,digest


def load(path):
    a=np.load(path)
    return {k:torch.tensor(a[k]) if k in ['z','carrier','coords','valid','distance','chart','target'] else a[k] for k in a.files}


def run(main,output):
    main=Path(main);root=Path(output);root.mkdir(exist_ok=False);torch.set_num_threads(2)
    protocol=json.loads((main/'protocol.json').read_text());steps=protocol['steps']
    train=load(main/'train.npz');val=load(main/'validation.npz');test=load(main/'test.npz');long=load(main/'longer_loop.npz')
    json_save(root/'protocol.json',dict(main=str(main.resolve()),steps=steps,learning_rate=.003,batch_size=64,seeds=protocol['seeds'],
        source_hashes=source_hashes(),data_sha256={p.name:digest(p.read_bytes()) for p in main.glob('*.npz')},
        capacity='Full Attention 5690 real DOF vs flat MLP 5474 and pooled MLP width42 5924; same optimizer, seeds, batches, updates'))
    rows=[]
    for seed in protocol['seeds']:
        for kind in ['global_flat_mlp','pooled_mlp_matched']:
            torch.manual_seed(seed)
            model=GlobalMLP() if kind=='global_flat_mlp' else HolonomyModel('pooled_mlp',width=42)
            folder=root/f'{kind}_s{seed}';folder.mkdir()
            before=score(model,test);snapshot(model,subset(test,slice(0,16)),folder/'before_internal.npz')
            torch.save(model.state_dict(),folder/'initial.pt')
            optimizer=torch.optim.Adam(model.parameters(),lr=.003);generator=torch.Generator().manual_seed(seed+5000);history=[]
            for step in range(1,steps+1):
                ix=torch.randint(len(train['z']),(64,),generator=generator);b=subset(train,ix)
                loss=((model(inputs(b))-b['target'])**2).mean();optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                if step%50==0:history.append(dict(step=step,train_mse=float(loss.detach()),validation_mse=score(model,val)['mse']))
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),batch_rng=generator.get_state()),folder/'final.pt')
            snapshot(model,subset(test,slice(0,16)),folder/'after_internal.npz');csv_save(folder/'history.csv',history)
            # Flattened MLP has a fixed input size: length extrapolation is undefined.
            result=dict(kind=kind,seed=seed,before=before,after=score(model,test),real_parameter_dof=sum(p.numel() for p in model.parameters()),
                        longer_loop=score(model,long) if kind=='pooled_mlp_matched' else None)
            json_save(folder/'summary.json',result);rows.append(result);print(result,flush=True)
    json_save(root/'summary.json',rows)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.main,a.output)
