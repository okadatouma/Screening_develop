"""Render matched-density and fixed-input history results with Matplotlib.

python -m screening_analysis.density_ph_plot ROOT
Requires matplotlib; no JAX/model imports or checkpoint writes.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--density-dir',default='density-ph-final')
    args=parser.parse_args()
    output=args.root/args.density_dir
    files=sorted(output.glob('*/curves.csv'))
    colors=dict(qk='#8e44ad',trim='#d47c11',window='#2675b6',screening='#18845a',distance='#333333')
    fig,axs=plt.subplots(1,len(files),figsize=(13,4.8),squeeze=False)
    for ax,path in zip(axs[0],files):
        summary=json.loads((path.parent/'summary.json').read_text())
        possible_edges=summary['possible_edges']
        with path.open() as f: rows=list(csv.DictReader(f))
        groups={(r['sample'],r['layer'],r['head']) for r in rows}
        max_k=min(max(int(r['k']) for r in rows if (r['sample'],r['layer'],r['head'])==g) for g in groups)
        for source,color in colors.items():
            for control in ('raw','distance_shuffle'):
                by_k={}
                for r in rows:
                    if r['source']==source and r['control']==control and r['common_untied']=='True' and int(r['k'])<=max_k:
                        by_k.setdefault(int(r['k']),[]).append(int(r['beta1']))
                ks=sorted(by_k)
                label=source if control=='raw' else f'{source}: distance null'
                style=':' if control!='raw' else '--' if source in ('trim','distance') else '-'
                ax.plot(np.array(ks)/possible_edges,[np.mean(by_k[k]) for k in ks],style,color=color,
                        linewidth=1.6,alpha=.65 if control!='raw' else 1,label=label)
        ax.set(title=path.parent.name,xlabel=f'Edge density p = k/{possible_edges}',ylabel='Mean Betti-1')
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Equal edge counts; all 16 configuration/head graphs supported; distance null = 8 shuffles')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/f'matched-density-comparison.{ext}',dpi=170)
    plt.close(fig)
    history=args.root/'density-ph-history'
    with (history/'history.csv').open() as f: rows=list(csv.DictReader(f))
    runs=sorted({r['run'] for r in rows})
    fig,axs=plt.subplots(1,len(runs),figsize=(12,4.5),squeeze=False)
    for ax,run in zip(axs[0],runs):
        selected=sorted([r for r in rows if r['run']==run],key=lambda r:int(r['step']))
        for source,color in colors.items():
            ax.plot([int(r['step']) for r in selected],[float(r['fixed_k_'+source]) for r in selected],
                    '--' if source in ('trim','distance') else '-',marker='o',color=color,label=source)
        ax.set(title=f"{run}: fixed k = 0..{selected[0]['history_fixed_max_k']}",
               xlabel='Checkpoint step',ylabel='Mean Betti-1 over fixed k and inputs')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Fixed final configurations: parameter-change diagnostic, not historical equilibrium sampling')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(history/f'fixed-input-history.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
