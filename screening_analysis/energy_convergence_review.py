"""Plot training energies and report block-based stationarity diagnostics.

No JAX initialization or training; source CSVs are read-only.

このコードで計算結果を見て、step数でどのくらいエネルギーが下がっているのか、収束しているのかを確認する
出力は convergence-review/ 内のPNG・PDF・SVG
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def block_stats(values, size):
    n = len(values) // size
    means = values[-n*size:].reshape(n, size).mean(1)
    return dict(size=size, count=n, mean=float(means.mean()),
                sem=float(means.std(ddof=1)/np.sqrt(n)))


def review(folder):
    path = folder / 'training' / 'train_stats.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r['step']) for r in rows])
    energy = np.array([float(r['energy']) for r in rows])
    if len(energy) < 20000 or not np.isfinite(energy).all():
        raise ValueError('Requires at least 20000 finite energy records')
    if not np.all(np.diff(steps) == 1):
        raise ValueError('Expected contiguous steps')
    tail = energy[-20000:]
    blocks = tail.reshape(-1, 500).mean(1)
    x = steps[-20000:].reshape(-1, 500).mean(1)
    a, b = blocks[:20], blocks[20:]
    delta = float(b.mean()-a.mean())
    delta_sem = float(np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)))
    xc = x-x.mean()
    slope = float(xc @ (blocks-blocks.mean()) / (xc @ xc))
    residual = blocks-(blocks.mean()+slope*xc)
    slope_sem = float(np.sqrt((residual@residual)/(len(blocks)-2)/(xc@xc)))
    corr = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    result = dict(source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  records=len(energy), first_step=int(steps[0]), last_step=int(steps[-1]),
                  energy_units='total code units, not per-electron Hartree',
                  last20k_mean=float(tail.mean()),
                  tail_blocks=[block_stats(tail, s) for s in (250,500,1000,2000,4000)],
                  previous10k_mean=float(a.mean()), final10k_mean=float(b.mean()),
                  final_minus_previous10k=delta, delta_sem_500=delta_sem,
                  delta_z_500=delta/delta_sem if delta_sem else None,
                  tail_slope_per10k=slope*10000, tail_slope_sem_per10k=slope_sem*10000,
                  tail_slope_z=slope/slope_sem if slope_sem else None,
                  detrended_block_lag1_correlation=corr,
                  windows10k=[dict(first_step=int(steps[i]),last_step=int(steps[i+9999]),
                                   mean=float(energy[i:i+10000].mean()))
                               for i in range(0,len(energy)-9999,10000)],
                  limitations='Training batch means under changing parameters; block SEM/OLS assume approximately independent blocks. No proof of optimizer convergence or ground state; independent fixed-checkpoint energy and multiple seeds still needed.')
    return steps, energy, result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    results={}
    for row,(name,title,color) in enumerate([
        ('lam0.33-wc',r'$\lambda=0.33$ (WC HF initialization)','#2166ac'),
        ('lam0.72-ahc',r'$\lambda=0.72$ (AHC HF initialization)','#b35806')]):
        steps,energy,result=review(a.root/name)
        results[name]=result
        for col,tail in enumerate((False,True)):
            ax=axes[row,col]
            s=steps[-20000:] if tail else steps
            e=energy[-20000:] if tail else energy
            ax.plot(s,e,color='0.65',alpha=.22,lw=.35,rasterized=True,label='Batch energy')
            bs=s.reshape(-1,500).mean(1);be=e.reshape(-1,500).mean(1)
            ax.plot(bs,be,color=color,lw=1.5,label='500-update block mean')
            # Population spread is not an error bar; show an explicit block SEM instead.
            if tail:
                sem=np.array([block_stats(e[i:i+10000],500)['sem'] for i in (0,10000)])
                halfmeans=np.array([e[:10000].mean(),e[10000:].mean()])
                halfsteps=np.array([s[:10000].mean(),s[10000:].mean()])
                ax.errorbar(halfsteps,halfmeans,yerr=sem,fmt='o',color='black',capsize=5,
                            label='10k mean +/- block SEM (500)')
                ax.set_ylim(be.min()-.5,be.max()+.5)
                ax.text(.03,.04,f"Last - previous 10k: {result['final_minus_previous10k']:+.4f}\n"
                        f"Tail slope / 10k: {result['tail_slope_per10k']:+.4f}",
                        transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            else:
                ax.axvspan(60000,79999,color=color,alpha=.08,label='Last 20k updates')
            ax.set_title(title+ (' — last 20k (zoomed)' if tail else ' — full history'))
            ax.set_xlabel('Training step (zero-based)')
            ax.set_ylabel('Total energy (code units)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8,loc='upper right')
            ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
    fig.suptitle('VMC training energy: full histories and late-stage drift\n'
                 'Training diagnostics only; zoom panels clip individual batch fluctuations',fontsize=13)
    for suffix in ('png','pdf','svg'):
        fig.savefig(a.output/f'energy-convergence.{suffix}',dpi=180)
    plt.close(fig)
    (a.output/'diagnostics.json').write_text(json.dumps(results,indent=2))
    with (a.output/'window-means.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['condition','first_step','last_step','mean_total_code_energy'])
        for name,r in results.items():
            for w in r['windows10k']:writer.writerow([name,w['first_step'],w['last_step'],w['mean']])
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
