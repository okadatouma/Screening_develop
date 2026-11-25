"""QWZ denoising/atlas studies reuse the physical reference implementation."""
import copy
import math
import numpy as np
import torch
from ..paths import wannier_source
from ..factory import make_wavefunction
from ..experiments.store import json_save,csv_save
from ..diagnostics.topology import graph_ph,phase_lift_ph

wannier_source()
from wannier_ph.geometry import qwz,projector,berry_data,gauge_phase
from wannier_ph.wannier import canonical_atlas,cylinder_frame


def dataset(c,masses,n=None,gauge='none',seed=0):
    n=n or c['physics']['nk'];atlas=c['wavefunction']['type']=='local_atlas'
    zs=[];ps=[];valids=[];covers=[]
    for mass in masses:
        k,u,_=qwz(n,mass);p=projector(u);ps.append(p.reshape(-1,2,2))
        chi=gauge_phase(k,gauge,seed)
        if atlas:
            frame,valid,_=canonical_atlas(u,tolerance=.1)
            frame=frame*np.exp(1j*np.stack([chi,-.7*chi+.4],-1))[...,None,:]
            zs.append(frame.transpose(3,0,1,2).reshape(2*n*n,2));valids.append(valid.transpose(2,0,1).reshape(-1))
        else:zs.append((u*np.exp(1j*chi)[...,None]).reshape(-1,2));valids.append(np.ones(n*n,bool))
        w=np.clip((p[...,0,0].real-.15)/.7,0,1);covers.append(np.stack([w,1-w],-1).reshape(n*n,2))
    kk=k.reshape(-1,2);nodes=np.tile(kk,(2 if atlas else 1,1));delta=(nodes[:,None,:]-nodes[None,:,:]+np.pi)%(2*np.pi)-np.pi
    coords=np.stack([np.cos(nodes[:,0]),np.sin(nodes[:,0]),np.cos(nodes[:,1]),np.sin(nodes[:,1])],-1)
    b=len(masses)
    return dict(z=torch.tensor(np.asarray(zs),dtype=torch.complex64),valid=torch.tensor(np.asarray(valids),dtype=torch.bool),
                target_projector=torch.tensor(np.asarray(ps),dtype=torch.complex64),mass=torch.tensor(masses,dtype=torch.float32),
                cover=torch.tensor(np.asarray(covers),dtype=torch.float32),coords=torch.tensor(np.broadcast_to(coords,(b,*coords.shape)).copy(),dtype=torch.float32),
                distance=torch.tensor(np.linalg.norm(delta,axis=-1),dtype=torch.float32),
                chart=torch.tensor(np.repeat(np.arange(2 if atlas else 1),n*n)),k=k)


def transform_gauge(batch,chi):
    out=dict(batch);out['z']=batch['z']*torch.exp(1j*chi)[...,None]
    return out


class QWZTask:
    def __init__(self,c):self.c=c

    @torch.no_grad()
    def evaluate(self,model,batch,intervention=None):
        model.eval();result=model(batch,intervention);p=result['projector'].numpy();n=int(np.sqrt(p.shape[1]));rows=[];states=[]
        for i,mass in enumerate(batch['mass'].numpy()):
            eigen,vec=np.linalg.eigh(p[i]);u=vec[...,-1].reshape(n,n,2);states.append(u)
            berry=berry_data(u);target=berry_data(qwz(n,float(mass))[1]);clutch=cylinder_frame(u)[2]
            row=dict(mass=float(mass),projector_mse=float(np.mean(abs(p[i]-batch['target_projector'][i].numpy())**2)),
                     chern=berry['chern'],target_chern=target['chern'],curvature_rmse=float(np.sqrt(np.mean((berry['curvature']-target['curvature'])**2))),
                     min_link=berry['minimum_link_overlap'],max_flux=berry['max_plaquette_phase'],
                     minimum_decoding_norm=float(result['raw_norm'][i].min()),transition_winding_chern=-clutch,
                     hermitian_error=float(abs(p[i]-p[i].conj().swapaxes(-1,-2)).max()),
                     idempotence_error=float(abs(p[i]@p[i]-p[i]).max()),rank_error=float(abs(np.trace(p[i],axis1=-2,axis2=-1)-1).max()))
            row['admissible_screen']=bool(row['minimum_decoding_norm']>.03 and row['min_link']>.2 and row['max_flux']<math.pi-.2)
            if model.atlas:
                frames=result['frame'][i].reshape(2,n*n,2);truth=batch['z'][i].reshape(2,n*n,2)
                mask=batch['valid'][i].reshape(2,n*n).all(0)
                t=(frames[0].conj()*frames[1]).sum(-1);tt=(truth[0].conj()*truth[1]).sum(-1)
                row['transition_mse']=float(abs(t[mask]-tt[mask]).square().mean()) if mask.any() else None
                row['transition_overlap_count']=int(mask.sum())
            rows.append(row)
        arrays=dict(projector=p,states=np.asarray(states),raw_output=result['raw_output'].numpy(),features=result['features'].numpy())
        if 'frame' in result:arrays['local_frames']=result['frame'].numpy()
        for li,record in enumerate(result['records']):
            arrays.update({f'layer{li}_{key}':value.detach().numpy() for key,value in record.items()})
        return rows,arrays

    def diagnostics(self,store,tag,rows,arrays,batch,lift=True):
        with (store.path/'diagnostics'/f'{tag}_internal.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        if not self.c['evaluation']['ph']:return
        li=self.c['model']['depth']-1
        for i,row in enumerate(rows):
            graphs=dict(QK=np.tanh(arrays[f'layer{li}_QK'][i]).mean(0),Alpha=arrays[f'layer{li}_Alpha'][i].mean(0),
                        Window=arrays[f'layer{li}_Window'].mean(0),Screening=arrays[f'layer{li}_weights'][i].mean(0))
            # QWZ uses softmax; the common label Screening denotes actual aggregation graph only.
            graph_ph(store.path/'diagnostics',f'{tag}_mass{i}',graphs)
            from ripser import ripser
            features=arrays['features'][i]
            if np.iscomplexobj(features):features=np.c_[features.real,features.imag]
            dgm=ripser(features,maxdim=1)['dgms']
            np.savez_compressed(store.path/'diagnostics'/f'{tag}_mass{i}_feature_ph.npz',**{f'H{j}':d for j,d in enumerate(dgm)})
            if lift:
                data=phase_lift_ph(store.path/'diagnostics',f'{tag}_mass{i}',batch['k'],arrays['states'][i],self.c['evaluation']['phase_samples'])
                json_save(store.path/'diagnostics'/f'{tag}_mass{i}_lift_summary.json',data)

    def run(self,store,resume=False,until=None):
        c=self.c;t=c['training'];torch.set_num_threads(2);torch.manual_seed(c['experiment']['seed']);torch.use_deterministic_algorithms(True)
        model=make_wavefunction(c['wavefunction'],'qwz_topology',full_config=c)
        optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'])
        train=dataset(c,c['physics']['train_masses']);test=dataset(c,c['physics']['test_masses'])
        store.input(train_z=train['z'].numpy(),test_z=test['z'].numpy(),train_projector=train['target_projector'].numpy(),
                    test_projector=test['target_projector'].numpy(),k=test['k'])
        if resume and (store.path/'latest.json').exists():
            start,state=store.restore();model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);store.trim_history(start)
        else:
            start=0;store.save_checkpoint(0,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
        # Recreate the exact untrained model under a local RNG fork, preserving resumed RNG.
        with torch.random.fork_rng():
            torch.manual_seed(c['experiment']['seed']);initial=make_wavefunction(c['wavefunction'],'qwz_topology',full_config=c)
        stop=t['steps'] if until is None else until
        if not start<=stop<=t['steps']:raise ValueError('until out of range')
        for step in range(start+1,stop+1):
            model.train();mask=(torch.rand(train['valid'].shape)>.2).float()
            result=model(train,mask=mask);loss=model.loss(result,train)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite denoising loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip']);optimizer.step()
            if step%max(1,t['steps']//10)==0 or step==stop:store.log(dict(step=step,loss=float(loss.detach()),widths=[a.widths().detach().tolist() for a in model.attentions],chern_in_loss=False))
            if step%t['checkpoint_every']==0 or step==stop:store.save_checkpoint(step,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
        if stop<t['steps']:return dict(paused_at=stop)
        all_rows=[]
        for tag,current in [('before',initial),('after',model)]:
            rows,arrays=self.evaluate(current,test)
            all_rows += [dict(stage=tag,**row) for row in rows]
            self.diagnostics(store,tag,rows,arrays,test)
        trained_rows=rows;trained_arrays=arrays
        # Direct numerical covariance test at identical checkpoint, not a Chern-label task.
        generator=torch.Generator().manual_seed(312)
        chi=(torch.rand(test['valid'].shape,generator=generator)*2-1)*torch.pi
        gauged=transform_gauge(test,chi);grows,garrays=self.evaluate(model,gauged)
        gauge=dict(projector_max_difference=float(abs(garrays['projector']-trained_arrays['projector']).max()),
                   chern_changes=[b['chern']-a['chern'] for a,b in zip(trained_rows,grows)],
                   supplied_links=c['attention']['type']=='gauge_covariant',
                   interpretation='Berry links and atlas cover were SUPPLIED, not discovered')
        if 'local_frames' in trained_arrays:
            gauge['frame_covariance_error']=float(abs(garrays['local_frames']-trained_arrays['local_frames']*np.exp(1j*chi.numpy())[...,None]).max())
        json_save(store.path/'gauge_robustness.json',gauge)
        all_rows += [dict(stage='random_gauge',**row) for row in grows]
        fine=dataset(c,c['physics']['test_masses'],n=min(2*c['physics']['nk'],32));fine_rows,_=self.evaluate(model,fine)
        all_rows += [dict(stage='fine_mesh',**row) for row in fine_rows]
        changes=[]
        if c['evaluation']['interventions']:
            specs=[dict(type='head_ablation',head=h) for h in range(c['model']['heads'])]+[dict(type='remove_connections'),dict(type='distance_band',radius=1.5),dict(type='window_width',width=1.5)]
            if c['embedding']['type']!='projector':specs.append(dict(type='zero_phase_channel'))
            if c['attention']['type']=='gauge_covariant':specs.append(dict(type='disable_gauge_transport'))
            if model.atlas:specs.append(dict(type='patch_connections'))
            for ii,spec in enumerate(specs):
                from ..interventions import apply_intervention
                target,_=apply_intervention(c,model,None,spec)
                current,arr=self.evaluate(target,test);frows,_=self.evaluate(target,fine)
                for row,baseline,frow in zip(current,trained_rows,frows):
                    row['chern_mesh_stable']=bool(abs(row['chern']-frow['chern'])<1e-6)
                    row['conditional_interpretation_screen']=bool(row['admissible_screen'] and frow['admissible_screen'] and row['chern_mesh_stable'])
                    changes.append(dict(intervention=spec,**row,delta_projector_mse=row['projector_mse']-baseline['projector_mse']))
                np.savez_compressed(store.path/'diagnostics'/f'intervention_{ii}.npz',projector=arr['projector'],states=arr['states'])
            json_save(store.path/'interventions.json',changes)
        csv_save(store.path/'qwz_metrics.csv',all_rows)
        summary=dict(task='qwz_topology',steps=stop,parameter_count=sum(p.numel() for p in model.parameters()),
                     projector_mse=float(np.mean([r['projector_mse'] for r in trained_rows])),metrics=trained_rows,gauge=gauge,
                     physical_gap_certified=False,output_type=c['wavefunction']['type'],
                     information_supplied='local charged input sections, Berry links and chart cover for covariant design; cannot count preserved input topology as discovery')
        store.complete(summary);return summary
