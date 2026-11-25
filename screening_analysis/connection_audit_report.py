"""Summarize completed connection_audit output and draw scientific figures.

Run in an environment with numpy + matplotlib; does not import JAX/models.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def finite_mean(rows,key):
    values=[float(r[key]) for r in rows if r[key] not in ('','None')]
    return float(np.mean(values)) if values else None


def standardized_selection(rows,model,head,k,kind):
    origins=sorted({r['input_origin'] for r in rows});index={s:i for i,s in enumerate(origins)}
    exposure=np.zeros((len(origins),20,2,12));events=np.zeros_like(exposure)
    for r in rows:
        if int(r['head'])!=head or int(r['k'])!=k:continue
        ix=(index[r['input_origin']],int(r['distance_bin']),int(r['same_spin']=='True'),int(r['angle_bin']))
        exposure[ix]+=int(r[kind+'_opportunities'])
        events[ix]+=int(r['promoted' if kind=='promotion' else 'demoted'])
    # Spin: same common origin/distance/angle distribution for both spin classes.
    common=(exposure[:,:,0,:]>=5)&(exposure[:,:,1,:]>=5)
    weights=exposure.sum(axis=2)*common
    spin=[]
    for s in range(2):
        rates=np.divide(events[:,:,s,:],exposure[:,:,s,:],out=np.zeros_like(weights),where=exposure[:,:,s,:]>0)
        spin.append(float(np.sum(weights*rates)/weights.sum()) if weights.sum()>0 else None)
    # Angle: same common origin/distance/spin distribution for all 12 angles.
    valid=(exposure>=5).all(axis=3)
    weight_angle=exposure.sum(axis=3)*valid
    angular=[]
    for a in range(12):
        rates=np.divide(events[:,:,:,a],exposure[:,:,:,a],out=np.zeros_like(weight_angle),where=exposure[:,:,:,a]>0)
        angular.append(float(np.sum(weight_angle*rates)/weight_angle.sum()) if weight_angle.sum()>0 else None)
    return dict(model=model,head=head,k=k,kind=kind,
        raw_opposite_rate=float(events[:,:,0,:].sum()/exposure[:,:,0,:].sum()) if exposure[:,:,0,:].sum()>0 else None,
        raw_same_rate=float(events[:,:,1,:].sum()/exposure[:,:,1,:].sum()) if exposure[:,:,1,:].sum()>0 else None,
        standardized_opposite_rate=spin[0],standardized_same_rate=spin[1],
        spin_standardization_exposure_fraction=float(weights.sum()/exposure.sum()) if exposure.sum()>0 else 0,
        standardized_angular_rates=angular,
        angle_standardization_exposure_fraction=float(weight_angle.sum()/exposure.sum()) if exposure.sum()>0 else 0,
        minimum_count_per_cell=5)


def sensitivity_summary(rows,model,subset='shared_all'):
    if subset!='shared_all':rows=[r for r in rows if r['input_origin']==subset]
    results=[]
    for head in (0,1):
        for k in (0,60,90):
            for group in ('all_offdiagonal','promoted','demoted','retained'):
                selected=[r for r in rows if int(r['head'])==head and int(r['k'])==k and r['group']==group]
                if not selected:continue
                da=np.array([float(r['dlogabs']) for r in selected]);dp=np.array([float(r['dphase']) for r in selected])
                mean_phase=np.angle(np.exp(1j*dp).mean())
                centered_phase=np.angle(np.exp(1j*(dp-mean_phase)))
                linear_a=np.array([float(r['linear_dlogabs']) for r in selected])
                linear_p=np.array([float(r['linear_dphase']) for r in selected])
                results.append(dict(model=model,input_subset=subset,head=head,k=k,group=group,samples=len(selected),
                    mean_edges=finite_mean(selected,'edges'),mean_screening_mass=finite_mean(selected,'screening_mass'),
                    mean_abs_amplitude_derivative=finite_mean(selected,'mean_abs_amplitude_derivative'),
                    mean_abs_phase_derivative=finite_mean(selected,'mean_abs_phase_derivative'),
                    raw_amplitude_response_rms=float(np.sqrt(np.mean(da**2))),
                    centered_amplitude_response_rms=float(np.std(da)),
                    raw_phase_response_rms=float(np.sqrt(np.mean(dp**2))),
                    centered_phase_response_rms=float(np.sqrt(np.mean(centered_phase**2))),
                    amplitude_linearization_rmse=float(np.sqrt(np.mean((da-linear_a)**2))),
                    phase_linearization_rmse=float(np.sqrt(np.mean(np.angle(np.exp(1j*(dp-linear_p)))**2)))))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output
    models=sorted(p for p in root.glob('lam*') if (p/'summary.json').exists())
    if len(models)!=2:raise ValueError('two completed model audits required')
    plt.rcParams.update({'font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    colors=['#2675b6','#d47c11']
    width_rows=read(root/'window-width-history.csv')
    widths=[];replacement=[];head_stats=[];selection=[];sens=[];alpha_spin=[]
    paired_data=[];composition=[]
    for modelindex,path in enumerate(models):
        model=path.name
        rows=read(path/'weight-distance.csv')
        exact=read(path/'exact-window-curves.csv')
        for col,source in enumerate(('window','alpha')):
            ax=axs[modelindex,col]
            for head in (0,1):
                shared=[r for r in rows if r['source']==source and int(r['head'])==head and r['input_subset']=='shared_all' and r['spin']=='all' and r['count']!='0']
                native=[r for r in rows if r['source']==source and int(r['head'])==head and r['input_subset']==model and r['spin']=='all' and r['count']!='0']
                center=lambda r:(float(r['distance_low'])+float(r['distance_high']))/2
                ax.plot([center(r) for r in shared],[float(r['mean']) for r in shared],color=colors[head],label=f'H{head} shared')
                ax.fill_between([center(r) for r in shared],[float(r['q10']) for r in shared],[float(r['q90']) for r in shared],color=colors[head],alpha=.12)
                ax.plot([center(r) for r in native],[float(r['mean']) for r in native],':',color=colors[head],label=f'H{head} native')
                if source=='window':
                    e=[r for r in exact if int(r['head'])==head]
                    ax.plot([float(r['distance']) for r in e],[float(r['weight']) for r in e],'--',color=colors[head],alpha=.5)
            ax.set(title=f'{model}: {source}',xlabel='Periodic-norm distance',ylabel='Weight');ax.grid(alpha=.2);ax.legend(fontsize=8)
        with np.load(path/'edge-data.npz') as z:
            d=z['periodic_distance'];alpha=z['alpha'];spins=z['spins'];origins=z['origins'];ii=z['edge_i'];jj=z['edge_j']
            same=spins[:,ii]==spins[:,jj]
            # Ordinary fixed-k graph descriptors, not a new PH computation.
            for source in ('qk','alpha','window','screening'):
                for sample in range(len(d)):
                    for head in (0,1):
                        weights=z[source][sample,head]
                        for k in (60,90):
                            order=np.argsort(-weights,kind='stable')
                            if np.count_nonzero(weights>0)<k or weights[order[k-1]]==weights[order[k]]:continue
                            chosen=order[:k];graph=np.zeros((len(spins[0]),len(spins[0])),dtype=np.int64)
                            graph[ii[chosen],jj[chosen]]=1;graph+=graph.T
                            triangles=int(np.trace(graph@graph@graph)//6)
                            unseen=set(range(len(graph)));components=0
                            while unseen:
                                components+=1;stack=[unseen.pop()]
                                while stack:
                                    adjacent=set(np.flatnonzero(graph[stack.pop()]))&unseen
                                    unseen-=adjacent;stack.extend(adjacent)
                            composition.append(dict(model=model,sample=sample,input_origin=str(origins[sample]),head=head,
                                source=source,k=k,same_spin_fraction=float(np.mean(same[sample,chosen])),triangles=triangles,
                                components=components,ordinary_graph_cycle_rank=k-len(graph)+components))
            bins=np.linspace(0,d.max()*(1+1e-12),21);labels=np.searchsorted(bins,d,side='right')-1
            for head in (0,1):
                means=[[],[]];weights=[]
                for origin in np.unique(origins):
                    for db in range(20):
                        base=(origins[:,None]==origin)&(labels==db)
                        selected=[alpha[:,head][base&(same==s)] for s in (False,True)]
                        if min(map(len,selected))<20:continue
                        weights.append(sum(map(len,selected)))
                        for s in (0,1):means[s].append(selected[s].mean())
                adjusted=[float(np.average(v,weights=weights)) for v in means]
                alpha_spin.append(dict(model=model,head=head,adjusted_opposite_alpha=adjusted[0],adjusted_same_alpha=adjusted[1],
                    definition='Standardized across common input-origin and distance bins, at least 20 pairs per spin class per stratum.'))
            paired_data.append({k:z[k] for k in ('alpha','window','screening','origins')})
        for head in (0,1):
            history=sorted([r for r in width_rows if r['model']==model and int(r['head'])==head],key=lambda r:int(r['step']))
            first,last=history[0],history[-1]
            widths.append(dict(model=model,head=head,first_step=int(first['step']),first_width=float(first['width']),
                               final_width=float(last['width']),half_weight_radius=float(last['half_weight_radius']),
                               final_trim_r=float(last['trim_r'])))
        counts=read(path/'replacement-counts.csv')
        for subset in ('shared_all',model):
            for head in (0,1):
                for k in (30,60,90):
                    selected=[r for r in counts if int(r['head'])==head and int(r['k'])==k and (subset=='shared_all' or r['input_origin']==subset)]
                    replacement.append(dict(model=model,input_subset=subset,head=head,k=k,graphs=len(selected),
                        mean_promoted=finite_mean(selected,'promoted'),mean_jaccard=finite_mean(selected,'jaccard')))
        correlations=read(path/'head-correlations.csv');output=read(path/'head-output-overlap.csv')
        for subset in ('shared_all',model):
            for source in ('qk','alpha','window','screening'):
                selected=[r for r in correlations if r['source']==source and (subset=='shared_all' or r['input_origin']==subset)]
                head_stats.append(dict(model=model,input_subset=subset,source=source,
                    pearson=finite_mean(selected,'pearson'),spearman=finite_mean(selected,'spearman'),
                    distance_residual_pearson=finite_mean(selected,'distance_residual_pearson'),
                    projected_output_cosine=finite_mean([r for r in output if subset=='shared_all' or r['input_origin']==subset],'cosine'),
                    cancellation_ratio=finite_mean([r for r in output if subset=='shared_all' or r['input_origin']==subset],'cancellation_ratio')))
        strata=read(path/'selection-strata.csv')
        for head in (0,1):
            for k in (60,90):
                for kind in ('promotion','demotion'):
                    selection.append(standardized_selection(strata,model,head,k,kind))
        responses=read(path/'wavefunction-responses.csv')
        sens.extend(sensitivity_summary(responses,model))
        sens.extend(sensitivity_summary(responses,model,model))
    fig.suptitle('Shared 128 inputs vs native 64 walkers; bands = descriptive 10th-90th percentiles')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/f'weight-distance.{ext}',dpi=170)
    plt.close(fig)
    # Spin-resolved Alpha curves make the within-distance selection visible.
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,path in zip(axs,models):
        rows=read(path/'weight-distance.csv')
        for head in (0,1):
            for spin,style in [('same','-'),('opposite','--')]:
                chosen=[r for r in rows if r['source']=='alpha' and int(r['head'])==head and r['input_subset']=='shared_all' and r['spin']==spin and r['count']!='0']
                ax.plot([(float(r['distance_low'])+float(r['distance_high']))/2 for r in chosen],
                        [float(r['mean']) for r in chosen],style,color=colors[head],label=f'H{head}, {spin} spin')
        ax.set(title=path.name,xlabel='Periodic-norm distance',ylabel='Mean Alpha weight');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Both models evaluated on exactly the same 128 positions AND pseudospins')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/f'alpha-spin-distance.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,model in zip(axs,[p.name for p in models]):
        for head in (0,1):
            h=sorted([r for r in width_rows if r['model']==model and int(r['head'])==head],key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in h],[float(r['width']) for r in h],marker='.',label=f'H{head}',color=colors[head])
        ax.set(title=model,xlabel='Saved checkpoint step',ylabel='Window cutoff width (periodic-norm units)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/f'window-width-history.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,model in zip(axs,[p.name for p in models]):
        for head in (0,1):
            r=next(r for r in selection if r['model']==model and r['head']==head and r['k']==90 and r['kind']=='promotion')
            angles=(np.arange(12)+.5)*15
            rates=[np.nan if x is None else x for x in r['standardized_angular_rates']]
            ax.plot(angles,rates,'o-',label=f"H{head}; coverage {r['angle_standardization_exposure_fraction']:.0%}",color=colors[head])
        ax.set(title=model,xlabel='Pair angle modulo 180 degrees',ylabel='Adjusted promotion probability, k=90');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Common origin/distance/spin strata; at least 5 opportunities per angle cell')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/f'angle-selection.{ext}',dpi=170)
    plt.close(fig)
    paired=[]
    for source in ('alpha','window','screening'):
        for head in (0,1):
            a,b=paired_data[0][source][:,head],paired_data[1][source][:,head]
            paired.append(dict(source=source,head=head,mean_model72_minus33=float(np.mean(b-a)),
                mean_per_configuration_rms_difference=float(np.mean(np.sqrt(np.mean((b-a)**2,axis=1)))),
                note='Head indices local; no claim of optimal cross-model head alignment.'))
    summary=dict(widths=widths,replacements=replacement,head_correlations=head_stats,
        alpha_spin=alpha_spin,selection=selection,sensitivity=sens,paired_shared_inputs=paired,
        validation=[json.loads((p/'summary.json').read_text())['validation'] for p in models],
        caveats='All statistics descriptive over fixed saved walkers. Different lambda/HF models and one seed each; no phase labels or significance claims. Angular dependence is geometric anisotropy, not evidence of topology.')
    (root/'aggregate-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    write(root/'sensitivity-summary.csv',sens);write(root/'replacement-summary.csv',replacement)
    write(root/'top-edge-composition.csv',composition)
    print(json.dumps({k:summary[k] for k in ('widths','alpha_spin','validation')},indent=2))


if __name__=='__main__':main()
