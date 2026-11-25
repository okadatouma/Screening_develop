"""Reproducible held-out benchmark and interventions for the NEW task only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .holonomy import dataset,contract,HolonomyModel,subset,gauge,inputs
from ..experiments.store import json_save,csv_save,digest

KINDS=['local_mlp','pooled_mlp','local_attention','full_attention','gauge_covariant','trainable_window']


def score(model,batch,spec=None):
    with torch.no_grad():
        y=model(inputs(batch),spec)
    target=batch['target']
    angle=torch.atan2(y[:,1],y[:,0])-torch.atan2(target[:,1],target[:,0])
    return dict(mse=float(((y-target)**2).mean()),
                angular_mae=float(abs(torch.atan2(torch.sin(angle),torch.cos(angle))).mean()),
                mean_output_norm=float(torch.linalg.vector_norm(y,dim=-1).mean()))


def snapshot(model,batch,path):
    with torch.no_grad():prediction,record=model(inputs(batch),record=True)
    values=dict(prediction=prediction.numpy(),representation=record['representation'].numpy(),target=batch['target'].numpy())
    for i,t in enumerate(record['trace']):
        for k in ['Q','K','QK','weights','head_contribution','widths']:
            values[f'layer{i}_{k}']=t[k].numpy()
    with Path(path).open('xb') as f:np.savez_compressed(f,**values)


def run(root,steps=800,seeds=(101,202,303),groups=256,kinds=KINDS):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    train=dataset(groups,701);val=dataset(64,702);test=dataset(96,703);long=dataset(48,704,length=16)
    checks={k:contract(b) for k,b in [('train',train),('validation',val),('test',test),('longer_loop',long)]}
    # This immutable contract and the exact data are written BEFORE model training.
    json_save(root/'TASK_CONTRACT.json',checks)
    for name,b in [('train',train),('validation',val),('test',test),('longer_loop',long)]:
        with (root/f'{name}.npz').open('xb') as f:
            np.savez_compressed(f,**{k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in b.items()})
    json_save(root/'protocol.json',dict(steps=steps,seeds=seeds,kinds=kinds,width=24,heads=3,depth=2,
        batch_size=64,optimizer='Adam',learning_rate=.003,checkpoint_selection='final fixed step; no test-set selection',
        data_sha256={p.name:digest(p.read_bytes()) for p in root.glob('*.npz')},
        interpretation='real controls get exact pointwise gauge removal; covariant blocks get flat input carrier transport; no hidden physical frames',
        capacity='same channel width/depth; complex parameters count twice in real DOF. Parameter counts are reported, not asserted equal.'))
    rows=[];interventions=[]
    for seed in seeds:
        for kind in kinds:
            torch.manual_seed(seed);model=HolonomyModel(kind)
            folder=root/f'{kind}_s{seed}';folder.mkdir()
            before=score(model,test);snapshot(model,subset(test,slice(0,16)),folder/'before_internal.npz')
            torch.save(dict(model=model.state_dict(),seed=seed,step=0),folder/'initial.pt')
            optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            generator=torch.Generator().manual_seed(seed+5000)
            history=[];started=time.monotonic()
            for step in range(1,steps+1):
                ix=torch.randint(len(train['z']),(64,),generator=generator)
                b=subset(train,ix);prediction=model(inputs(b));loss=((prediction-b['target'])**2).mean()
                if not torch.isfinite(loss):raise FloatingPointError(f'{kind} loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                if step%50==0 or step==steps:
                    history.append(dict(step=step,train_mse=float(loss.detach()),validation_mse=score(model,val)['mse']))
            snapshot(model,subset(test,slice(0,16)),folder/'after_internal.npz')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=steps,
                            rng=torch.get_rng_state(),batch_rng=generator.get_state()),folder/'final.pt')
            csv_save(folder/'history.csv',history)
            result=dict(kind=kind,seed=seed,steps=steps,before=before,after=score(model,test),
                        longer_loop=score(model,long),validation=score(model,val),
                        real_parameter_dof=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
                        seconds=time.monotonic()-started)
            torch.manual_seed(9000+seed)
            changed=gauge(test,torch.randn(test['z'].shape[:2])*3)
            with torch.no_grad():result['gauge_prediction_max_delta']=float(abs(model(inputs(changed))-model(inputs(test))).max())
            # A circular-mean predictor is fit on TRAIN targets only.
            mean=train['target'].mean(0)
            result['constant_test_mse']=float(((test['target']-mean)**2).mean())
            for layer in range(len(model.blocks)):
                for head in range(3):
                    spec=dict(type='head_ablation',layer=layer,head=head)
                    interventions.append(dict(kind=kind,seed=seed,**spec,**score(model,test,spec),baseline_mse=result['after']['mse']))
                for typ in ['patch_connections','disable_gauge_transport']:
                    spec=dict(type=typ,layer=layer)
                    interventions.append(dict(kind=kind,seed=seed,**spec,**score(model,test,spec),baseline_mse=result['after']['mse']))
            # Remote-edge counterfactual: target stays the physical twin's target,
            # while a distant input patch is exchanged. This is an intervention,
            # not a new valid labeled training example.
            remote=dict(test);remote['z']=test['z'].clone()
            mask=test['distance'][0]>2
            remote['z'][:,mask]=test['z'][torch.arange(len(test['z']))^1][:,mask]
            with torch.no_grad():
                pred=model(inputs(test));counter=model(inputs(remote))
                result['remote_exchange_prediction_rms']=float(torch.sqrt(((pred-counter)**2).mean()))
                result['remote_exchange_twin_target_mse']=float(((counter-test['target'][torch.arange(len(pred))^1])**2).mean())
            json_save(folder/'summary.json',result);rows.append(result)
            print(json.dumps(dict(kind=kind,seed=seed,test=result['after']['mse'],constant=result['constant_test_mse'])),flush=True)
    json_save(root/'summary.json',rows);csv_save(root/'interventions.csv',interventions)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=800)
    p.add_argument('--groups',type=int,default=256);p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    a=p.parse_args();run(a.output,a.steps,a.seeds,a.groups)
