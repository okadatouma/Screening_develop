"""Saved-diagram and local-frame plots; no new training or physical labels."""
import json
import numpy as np
from ..experiments.retries import resolve


def detailed_plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=json.loads((root/'plan.json').read_text());out=root/'figures';out.mkdir(exist_ok=True)
    selected=[c for c in cases if c['config']['experiment']['name'] in ['qwz_real','qwz_projector','qwz_gauge_covariant'] and c['config']['experiment']['seed']==101]
    fig,axes=plt.subplots(len(selected),4,figsize=(14,3.3*len(selected)))
    for ri,case in enumerate(selected):
        folder=resolve(root,case['name'])/'diagnostics'
        for ci,name in enumerate(['QK flag','aggregation flag','feature VR','physical phase-lift VR']):
            ax=axes[ri,ci];limit=1.
            for stage in ['before','after']:
                if ci<2:
                    archive=np.load(folder/f'{stage}_mass1_ph.npz');d=archive['QK_H1' if ci==0 else 'Screening_H1']
                else:d=np.load(folder/f'{stage}_mass1_{"feature_ph" if ci==2 else "lift_ph"}.npz')['H1']
                finite=d[np.isfinite(d[:,1])]
                if len(finite):limit=max(limit,float(finite.max()))
                # Essential graph bars are shown above finite ones, not silently dropped.
                ax.scatter(d[:,0],np.minimum(d[:,1],limit*1.03),s=6,label=stage,alpha=.7)
            ax.plot([0,limit],[0,limit],'k:',lw=.6);ax.set(title=case['config']['experiment']['name']+'\n'+name,
                    xlabel='birth (own filtration units)',ylabel='death; infinity clipped for display')
            ax.legend(fontsize=7)
    fig.suptitle('H1 at heldout m=0.8: distinct spaces and filtrations, not equal invariants')
    fig.tight_layout();fig.savefig(out/'internal_ph_vs_phase_lift.png',dpi=150);plt.close(fig)
    atlas=[c for c in cases if c['config']['wavefunction']['type']=='local_atlas']
    if atlas:
        from ..task_adapters.qwz import dataset
        fig,axes=plt.subplots(1,2,figsize=(12,4));labels=[]
        for index,case in enumerate(atlas):
            folder=resolve(root,case['name'])/'diagnostics';batch=dataset(case['config'],case['config']['physics']['test_masses'])
            target=batch['z'].numpy();valid=batch['valid'].numpy();labels.append(case['config']['experiment']['name']+f" s{case['config']['experiment']['seed']}")
            for stage,offset in [('before',-.12),('after',.12)]:
                frame=np.load(folder/f'{stage}_internal.npz')['local_frames'];overlap=np.sum(target.conj()*frame,-1)
                phase=np.angle(overlap[valid]);amplitude=abs(frame[valid])-abs(target[valid])
                axes[0].scatter(index+offset,np.sqrt(np.mean(phase**2)),label=stage if index==0 else None)
                axes[1].scatter(index+offset,np.sqrt(np.mean(amplitude**2)),label=stage if index==0 else None)
        for ax,name in zip(axes,['local frame phase RMS (target chart gauge)','local component magnitude RMS']):
            ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel=name);ax.tick_params(axis='x',labelrotation=70,labelsize=7);ax.legend()
        fig.tight_layout();fig.savefig(out/'atlas_amplitude_phase.png',dpi=150);plt.close(fig)
    tfim=[c for c in cases if c['config']['experiment']['name'] in ['tfim_g1.0_R1','tfim_g1.0_R2','tfim_g1.0_Rfull','tfim_g1.0_trainable'] and c['config']['experiment']['seed']==101]
    if tfim:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for i,case in enumerate(tfim):
            folder=resolve(root,case['name']);s=json.loads((folder/'summary.json').read_text());label=case['config']['experiment']['name']
            if i==0:
                axes[0].plot(s['exact']['connected_correlation'],'k.-',label='ED');axes[1].plot(s['exact']['S2'],'k.-',label='ED')
            axes[0].plot(s['observables']['connected_correlation'],'o--',label=label)
            axes[1].plot(s['observables']['S2'],'o--',label=label)
            info=np.load(folder/'diagnostics/NN_information.npz')
            axes[2].scatter(np.mean(s['widths']),info['Born_CMI'][np.triu_indices(s['n'],1)].mean(),label=label)
        for ax,xy in zip(axes,[('ring distance','connected Czz'),('subsystem length','Renyi-2 S2 (nats)'),('mean head width','mean Born CMI (nats)')]):ax.set(xlabel=xy[0],ylabel=xy[1]);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'tfim_ed_correlation_entanglement.png',dpi=150);plt.close(fig)
