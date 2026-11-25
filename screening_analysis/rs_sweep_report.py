"""Collect completed sweep analyses and draw PH/rs trends (requires Matplotlib).

python -m screening_analysis.rs_sweep_report PLAN --output NEW_DIRECTORY
Missing cases are listed; completed cases can be replotted without retraining.
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


def average(rows,key):
    values=[float(r[key]) for r in rows if r[key] not in ('','None','nan')]
    return float(np.mean(values)) if values else None


def ph_figure(path,out,title):
    colors=dict(window='#2675b6',trim='#d47c11',screening='#18845a',qk='#8e44ad')
    fig,axs=plt.subplots(2,4,figsize=(13,6))
    with np.load(path,allow_pickle=False) as z:
        for col,(source,color) in enumerate(colors.items()):
            for dim in (0,1):
                ax=axs[dim,col];essential=0
                for key in z.files:
                    if not key.startswith(source+'_raw_') or not key.endswith(f'_H{dim}'):continue
                    d=z[key];finite=np.isfinite(d[:,1]);essential+=int((~finite).sum())
                    ax.scatter(d[finite,0],d[finite,1],s=5,alpha=.15,color=color)
                    ax.scatter(d[~finite,0],np.full((~finite).sum(),1.04),s=12,marker='^',alpha=.3,color=color)
                ax.plot([0,1],[0,1],color='#999',linewidth=.6)
                ax.set(xlim=(-.02,1.06),ylim=(-.02,1.08),title=f'{source}, H{dim}; essential={essential}',xlabel='Birth',ylabel='Death')
    fig.suptitle(title+'; triangles at y=1.04 denote infinite deaths, not finite lifetimes')
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    parser.add_argument('--analysis-dir',default='analysis',help='Per-case analysis directory, e.g. analysis-recomputed')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());args.output.mkdir(parents=True,exist_ok=False)
    from screening_analysis.rs_sweep import write_conditions
    write_conditions(plan,args.output)
    rows=[];missing=[]
    for case in plan['cases']:
        folder=Path(plan['root'])/case['id']/args.analysis_dir
        if not (folder/'finished.json').exists():missing.append(case['id']);continue
        summary=json.loads((folder/'summary.json').read_text())
        if summary['case']!=case:raise ValueError('case metadata mismatch')
        windows=read(folder/'window-curves.csv')
        unique={(r['layer'],r['head']):float(r['width']) for r in windows}
        figure_dir=args.output/case['id'];figure_dir.mkdir()
        for subset in ('native','common'):
            data=folder/subset;ent=read(data/'entropy.csv');modes=read(data/'modes.csv')
            radial=read(data/'weight-distance.csv');phase=read(data/'phase-responses.csv') if (data/'phase-responses.csv').exists() else []
            density=read(data/'density-betti.csv');M=case['n']*(case['n']-1)//2;k=max(1,round(.1*M))
            ph_selected=[r for r in density if r['source']=='screening' and int(r['k'])==k and r['tie_split']=='False']
            expected_graphs=summary['subsets'][subset]['samples']*summary['subsets'][subset]['layers']*summary['subsets'][subset]['heads']
            phase_values=[]
            for layer,head in unique:
                values=np.array([float(r['dphase']) for r in phase if r['group']=='all' and r['layer']==layer and r['head']==head])
                if len(values):
                    center=np.angle(np.exp(1j*values).mean());residual=np.angle(np.exp(1j*(values-center)))
                    phase_values.append(float(np.sqrt(np.mean(residual**2))))
            phase_info=summary['subsets'][subset]['phase']
            # Bin index 6 has centre d/dref = 6.5/24, fixed across rs.
            alpha=lambda spin:average([r for r in radial if r['source']=='trim' and r['spin']==spin and int(r['distance_bin'])==6],'mean')
            rows.append(dict(case=case['id'],branch=case['branch'],lam=case['lam'],rs=case['rs'],seed=case['seed'],subset=subset,
                window_min=min(unique.values()),window_max=max(unique.values()),
                window_min_rs_aB=min(unique.values())*case['rs'],window_max_rs_aB=max(unique.values())*case['rs'],
                alpha_same_at_dref_0p271=alpha('same'),alpha_opposite_at_dref_0p271=alpha('opposite'),
                entropy_normalized=average([r for r in ent if r['include_self']=='False'],'entropy_normalized'),
                entropy_zero_row_fraction=average([r for r in ent if r['include_self']=='False'],'zero_row_fraction'),
                second_normalized_eigenvalue=average([r for r in modes if r['rank']=='1'],'normalized_eigenvalue'),
                minimum_normalized_eigenvalue=average([r for r in modes if int(r['rank'])==case['n']-1],'normalized_eigenvalue'),
                leading_participation=average([r for r in modes if r['rank']=='0' and r.get('mode_identifiable')=='True'],'participation'),
                phase_centered_rms_headmean=float(np.mean(phase_values)) if phase_values and phase_info['finite_difference_pass'] else None,
                phase_valid_samples=phase_info['valid_samples'],phase_fd_pass=phase_info['finite_difference_pass'],
                screening_beta1_density_0p1=average(ph_selected,'beta1') if len(ph_selected)==expected_graphs else None,
                density_beta1_valid_graphs=len(ph_selected),density_beta1_expected_graphs=expected_graphs,phase='unknown'))
            ph_figure(data/'ph/diagrams.npz',figure_dir/f'{subset}-ph-diagrams.png',case['id']+' '+subset)
            fig,axs=plt.subplots(1,2,figsize=(10,4))
            for layer,head in unique:
                for spin,style in [('same','-'),('opposite','--')]:
                    selected=[r for r in radial if r['source']=='trim' and r['layer']==layer and r['head']==head and r['spin']==spin and r['mean']!='']
                    axs[0].plot([(float(r['distance_low'])+float(r['distance_high']))/2 for r in selected],
                        [float(r['mean']) for r in selected],style,label=f'L{layer} H{head} {spin}')
                selected=[r for r in windows if r['layer']==layer and r['head']==head]
                axs[1].plot([float(r['distance']) for r in selected],[float(r['weight']) for r in selected],label=f'L{layer} H{head}')
            axs[0].set(title='Alpha vs periodic distance',xlabel='Distance',ylabel='Alpha')
            axs[1].set(title='Exact Window',xlabel='Distance',ylabel='Window')
            for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
            fig.suptitle(case['id']+' '+subset);fig.tight_layout();fig.savefig(figure_dir/f'{subset}-radial.png',dpi=150);plt.close(fig)
            with np.load(data/'probe.npz',allow_pickle=False) as probe_data, np.load(data/'attention-modes.npz',allow_pickle=False) as mode_data:
                n=int(probe_data['NE']);heads=min(4,int(probe_data['NH']));lattice=probe_data['lattice']
                positions=probe_data['positions'][0].reshape(n,2)
                positions=np.mod(positions@np.linalg.inv(lattice).T,1)@lattice.T
                fig,axes=plt.subplots(min(3,n),heads,figsize=(3.4*heads,8),squeeze=False)
                for head in range(heads):
                    vectors=mode_data[f'b0_l0_h{head}_eigenvectors'];values=mode_data[f'b0_l0_h{head}_eigenvalues']
                    identifiable=mode_data[f'b0_l0_h{head}_mode_identifiable']
                    ranks=[0,1,n-1] if n>=3 else list(range(n))
                    for plotrow,rank in enumerate(ranks):
                        ax=axes[plotrow,head];limit=max(float(abs(vectors[:,rank]).max()),1e-12)
                        dots=ax.scatter(positions[:,0],positions[:,1],c=vectors[:,rank],cmap='coolwarm',vmin=-limit,vmax=limit)
                        ax.set_title(f'Head {head}, mode {rank}: {values[rank]:.3g}'+(' [degenerate]' if not identifiable[rank] else ''))
                        ax.set_aspect('equal');fig.colorbar(dots,ax=ax,shrink=.6)
                fig.suptitle(case['id']+'\n'+subset+'; first configuration, layer 0; eigenvector sign arbitrary',fontsize=9)
                fig.tight_layout();fig.savefig(figure_dir/f'{subset}-mode-snapshot.png',dpi=150);plt.close(fig)
    if rows:
        with (args.output/'observables.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        fields=[('screening_beta1_density_0p1','Screening beta1 at p~0.1; positive edges only'),
                ('window_max','Maximum Window width, code units'),('alpha_same_at_dref_0p271','Alpha same spin at d/dref~0.271'),
                ('minimum_normalized_eigenvalue','Minimum normalized eigenvalue (bipartite: -1)'),
                ('entropy_normalized','Off-diagonal normalized connection entropy'),
                ('phase_centered_rms_headmean','Centered phase response RMS, 10% attenuation')]
        branches=sorted({r['branch'] for r in rows})
        for subset in ('native','common'):
            fig,axs=plt.subplots(3,2,figsize=(12,11))
            for ax,(field,title) in zip(axs.flat,fields):
                for branch in branches:
                    for seed in sorted({r['seed'] for r in rows if r['branch']==branch}):
                        chosen=sorted([r for r in rows if r['branch']==branch and r['seed']==seed and r['subset']==subset and r[field] is not None],key=lambda r:r['rs'])
                        ax.plot([r['rs'] for r in chosen],[r[field] for r in chosen],'o-',label=f'{branch} seed{seed}',linewidth=1)
                ax.set(title=title,xlabel='rs');ax.grid(alpha=.2)
            axs.flat[0].legend(fontsize=7)
            fig.suptitle(f'{subset}: independent seeds shown separately; branches are candidate labels')
            fig.tight_layout();fig.savefig(args.output/f'{subset}-rs-trends.png',dpi=170);fig.savefig(args.output/f'{subset}-rs-trends.svg');plt.close(fig)
    (args.output/'status.json').write_text(json.dumps(dict(completed_cases=len(rows)//2,missing=missing,
        note='No trained physical phase labels assigned. Trends describe saved native walkers and fixed common inputs, not certified equilibrium measurements.'),indent=2))
    print(f'{len(rows)//2} completed cases; {len(missing)} missing')


if __name__=='__main__':main()
