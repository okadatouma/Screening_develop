"""Finite-cell physical correlations, connection ranges and convergence reports.

Pure NumPy/SciPy/Matplotlib entry; all original results remain read-only.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ["run","status"]
    with Path(path).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def minimum_lengths(delta,lattice):
    """Exact nearest image for the well-conditioned triangular cells used here."""
    delta=np.asarray(delta);frac=delta@np.linalg.inv(lattice).T;frac-=np.rint(frac)
    shifts=np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1)])
    return np.linalg.norm((frac[...,None,:]-shifts)@lattice.T,axis=-1).min(-1)


def pair_curves(positions,lattice,edges,grid=64):
    n=positions.shape[1];ii,jj=np.triu_indices(n,1);area=abs(np.linalg.det(lattice))
    lengths=minimum_lengths(positions[:,ii]-positions[:,jj],lattice)
    count=np.histogram(lengths,bins=edges)[0]
    annulus=np.pi*np.diff(edges**2)
    gr=count/(len(positions)*n*(n-1)/2*annulus/area)
    fractional=np.mod(positions@np.linalg.inv(lattice).T,1.)
    def histogram(values):
        h=np.histogram2d(*values.reshape(-1,2).T,bins=grid,range=((0,1),(0,1)))[0]
        return h/h.sum()
    # Cross-fit different chain groups, avoiding the finite-sample self spike
    # in the autocorrelation of a single empirical one-body density.
    density_a=histogram(fractional[::2]);density_b=histogram(fractional[1::2])
    density=(density_a+density_b)/2
    fa,fb=np.fft.fft2(density_a),np.fft.fft2(density_b)
    autocorrelation=np.fft.ifft2((fa*fb.conj()+fb*fa.conj())/2).real
    axis=np.arange(grid)/grid
    mesh=np.stack(np.meshgrid(axis,axis,indexing="ij"),axis=-1)
    distance=minimum_lengths(mesh@lattice.T,lattice)
    probability=np.histogram(distance,bins=edges,weights=autocorrelation)[0]
    uniform=np.histogram(distance,bins=edges,weights=np.full((grid,grid),1/grid**2))[0]
    background=np.divide(probability,uniform,out=np.full_like(probability,np.nan),where=uniform>0)
    return gr,background,density


def mass_radius(radius,mass,fraction=.9):
    good=np.isfinite(mass)&(mass>0)
    if not good.any():return None
    order=np.argsort(radius[good]);r=np.asarray(radius)[good][order];w=np.asarray(mass)[good][order]
    return float(r[min(np.searchsorted(np.cumsum(w),fraction*w.sum()),len(r)-1)])


def physical_analysis(measurement,case,output):
    from screening_analysis.lj_maps import first_shell
    from scipy.signal import find_peaks
    with np.load(measurement/"samples.npz") as z:
        original=z["positions"];lattice=z["lattice"]
    times,chains,n,_=original.shape
    # Preserve time blocks and take the same bounded number from every block.
    blocks=min(16,times//2);parts=np.array_split(np.arange(times),blocks)
    curves=[];backgrounds=[];positions=[]
    shortest=min(np.linalg.norm(lattice@np.array(v)) for v in ((1,0),(0,1),(1,1)))
    edges=np.linspace(0,.45*shortest,33);radius=(edges[:-1]+edges[1:])/2
    for part in parts:
        ids=part[np.linspace(0,len(part)-1,min(len(part),max(1,4096//blocks//chains))).astype(int)]
        pos=original[ids].reshape(-1,n,2)
        gr,bg,_=pair_curves(pos,lattice,edges)
        curves.append(gr);backgrounds.append(bg);positions.append(pos)
    positions=np.concatenate(positions)
    curves=np.asarray(curves);backgrounds=np.asarray(backgrounds)
    gr=curves.mean(0);bg=backgrounds.mean(0);connected=gr-bg
    sem=(curves-backgrounds).std(0,ddof=1)/np.sqrt(blocks)
    annulus=np.pi*np.diff(edges**2)
    # Explicit noise threshold prevents estimating a range from a pure noise tail.
    mass=np.maximum(abs(connected)-2*sem,0)*annulus
    r90=mass_radius(radius,mass)
    bootstrap=[];rng=np.random.default_rng(20261105)
    for _ in range(128):
        ids=rng.integers(blocks,size=blocks)
        value=(curves[ids]-backgrounds[ids]).mean(0)
        r=mass_radius(radius,np.maximum(abs(value)-2*sem,0)*annulus)
        if r is not None:bootstrap.append(r)
    interval=np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap)>=100 else None
    # Exponential envelope fits are only attempted when several resolved peaks
    # exist; a small crystal cell often cannot resolve a decay length at all.
    peaks,_=find_peaks(abs(connected),prominence=max(float(np.nanmedian(sem))*2,1.e-8))
    a=np.sqrt(2*np.pi/np.sqrt(3))
    peaks=peaks[(radius[peaks]>.5*a)&(abs(connected[peaks])>3*sem[peaks])]
    xi=None;fit_reason="fewer_than_three_resolved_envelope_peaks"
    if len(peaks)>=3:
        coefficient=np.polyfit(radius[peaks],np.log(abs(connected[peaks])),1)
        if coefficient[0]<0:
            xi=float(-1/coefficient[0]);fit_reason="finite_cell_exponential_envelope_fit"
        else:fit_reason="no_decaying_envelope"
    qmax=int(np.sqrt(n))+2
    modes=np.array([(i,j) for i in range(-qmax,qmax+1) for j in range(-qmax,qmax+1) if i or j])
    fractional=positions@np.linalg.inv(lattice).T
    rho=np.exp(2j*np.pi*np.einsum("bnd,qd->bnq",fractional,modes)).sum(1)
    total=(abs(rho)**2).mean(0)/n;disconnected=abs(rho.mean(0))**2/n
    sf_connected=np.maximum(total-disconnected,0)
    shell=np.asarray(first_shell(lattice,n))
    q=modes@(2*np.pi*np.linalg.inv(lattice))
    shell_mask=np.any(np.all(modes[:,None,:]==shell[None,:,:],axis=-1),axis=1)
    small=modes[np.argsort(np.linalg.norm(q,axis=1))[:6]]
    curvature_lengths=[]
    for Q in shell:
        center=np.flatnonzero(np.all(modes==Q,axis=1))
        neighbors=np.any(np.all(modes[:,None,:]==(Q+small)[None,:,:],axis=-1),axis=1)
        if len(center) and neighbors.any():
            peak=sf_connected[center[0]];adjacent=sf_connected[neighbors].mean()
            dq=np.mean(np.linalg.norm(small@(2*np.pi*np.linalg.inv(lattice)),axis=1))
            if adjacent>0 and peak>adjacent:curvature_lengths.append(np.sqrt(peak/adjacent-1)/dq)
    xi_peak=float(np.mean(curvature_lengths)) if curvature_lengths else None
    rows=[dict(r=float(r),r_aB=float(r*case["rs"]),gr=float(g),one_body_product=float(b),
               connected_distinct_canonical=float(c),block_sem=float(s)) for r,g,b,c,s in zip(radius,gr,bg,connected,sem)]
    table(output/"correlations.csv",rows)
    table(output/"structure_factor.csv",[dict(m1=int(m[0]),m2=int(m[1]),qx=float(k[0]),qy=float(k[1]),
        S_total=float(v),S_disconnected=float(d),S_connected=float(c)) for m,k,v,d,c in zip(modes,q,total,disconnected,sf_connected)])
    result=dict(correlation_R90=r90,correlation_R90_CI95=interval,
        xi_envelope=xi,xi_envelope_status=fit_reason,xi_connected_peak_curvature=xi_peak,
        radial_cutoff=float(edges[-1]),cell_limited_R90=bool(r90 is not None and r90>.85*edges[-1]),
        cell_limited_peak_xi=bool(xi_peak is not None and xi_peak>.5*shortest),
        S_Bragg_total=float(total[shell_mask].mean()),S_Bragg_connected=float(sf_connected[shell_mask].mean()),
        S_Bragg_disconnected=float(disconnected[shell_mask].mean()),correlation_configurations=len(positions),
        pair_definition="canonical distinct-pair excess over measured one-body density product, grid64; not full number-fluctuation connected correlator",
        R90_definition="annulus-integrated max(abs(pair_excess)-2*block_SEM,0), within finite radial cutoff",
        peak_xi_definition="finite-grid Lorentzian-curvature proxy around geometric Bragg shell, only when S_connected(Q)>neighbors; not assumed exponential correlation length")
    (output/"physical_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def connection_analysis(measurement,case):
    with np.load(measurement/"attention_probe.npz") as z:
        p={k:z[k] for k in z.files if k!="metadata_json"}
    n=case["n"];ii,jj=np.triu_indices(n,1)
    positions=p["positions"].reshape(-1,n,2)
    physical=minimum_lengths(positions[:,ii]-positions[:,jj],p["lattice"])
    network=p["d"][:,ii,jj]
    rows=[]
    for li in range(int(p["NL"])):
        for h in range(int(p["NH"])):
            w=(p["W"][:,li,h]+p["W"][:,li,h].transpose(0,2,1))/2
            mass=w[:,ii,jj];sumw=mass.sum()
            rows.append(dict(layer=li,head=h,network_R90=mass_radius(network.ravel(),mass.ravel()),
                physical_R90=mass_radius(physical.ravel(),mass.ravel()),
                physical_R_eff=float((physical*mass).sum()/sumw) if sumw>0 else None,
                network_R_eff=float((network*mass).sum()/sumw) if sumw>0 else None,
                mean_row_mass=float(p["W"][:,li,h].sum(-1).mean()),
                diagnostic_configurations=len(physical)))
    return rows


def report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan=json.loads((root/"plan.json").read_text());output=root/"report";output.mkdir(exist_ok=True)
    figures=output/"figures";figures.mkdir(exist_ok=True)
    runs=[];heads=[];trajectories=[];failures=[]
    for case in plan["cases"]:
        folder=root/"runs"/case["id"]
        if (folder/"failure.txt").exists():failures.append(case["id"])
        if not (folder/"finished.json").exists():continue
        done=json.loads((folder/"finished.json").read_text());measurement=Path(done["measurement"])
        summary=json.loads((measurement/"summary.json").read_text())
        cached=folder/"physical_summary.json"
        physical=json.loads(cached.read_text()) if cached.exists() else physical_analysis(measurement,case,folder)
        energy=summary["energy_Ha"]
        row=dict(case,status=done["status"],steps=done["steps"],energy_Ha=energy["mean"],
            energy_sem_Ha=energy["sem"],energy_ESS=energy["ess"],energy_Rhat=energy["rhat"],
            sampling_pass=summary["sampling_pass"],**{k:v for k,v in physical.items() if not isinstance(v,(dict,list))})
        runs.append(row)
        for a,scales in zip(connection_analysis(measurement,case),summary["window_scales"]):
            heads.append({**row,**a,**scales})
        history=json.loads((folder/"convergence.json").read_text())["history"]
        for entry in history:
            stage_folder=Path(entry["checkpoint"]).parent
            stage_measurement=stage_folder.parent/(stage_folder.name+"-measurement")
            if entry["step"]==done["steps"]:stage_physical=physical
            elif (stage_folder/"physical_summary.json").exists():
                stage_physical=json.loads((stage_folder/"physical_summary.json").read_text())
            else:stage_physical=physical_analysis(stage_measurement,case,stage_folder)
            for scale in entry["window_scales"]:
                trajectories.append(dict(case,step=entry["step"],stage_lr=entry["stage_lr"],
                    energy_Ha=entry["energy_Ha"]["mean"],energy_sem_Ha=entry["energy_Ha"]["sem"],
                    sampling_pass=entry["sampling_pass"],correlation_R90=stage_physical["correlation_R90"],
                    xi_connected_peak_curvature=stage_physical["xi_connected_peak_curvature"],**scale))
    table(output/"runs.csv",runs);table(output/"window_correlation_comparison.csv",heads);table(output/"trajectories.csv",trajectories)
    comparisons=[]
    for a in runs:
        if a["mode"]!="learned":continue
        for control_mode in ("full","fixed"):
            b=next((b for b in runs if (a["n"],a["rs"],a["seed"],a["initialization"])
                ==(b["n"],b["rs"],b["seed"],b["initialization"]) and b["mode"]==control_mode),None)
            if b:comparisons.append(dict(run=a["id"],control=b["id"],N=a["n"],rs=a["rs"],
                seed=a["seed"],delta_energy_Ha=a["energy_Ha"]-b["energy_Ha"],
                uncertainty_Ha=float(np.hypot(a["energy_sem_Ha"],b["energy_sem_Ha"])),
                both_converged=bool(a["status"]==b["status"]=="converged_to_protocol")))
    table(output/"paired_controls.csv",comparisons)
    grouped=[]
    for key in {(r["n"],r["rs"],r["initialization"],r["mode"],r["width_factor"]) for r in runs}:
        data=[r for r in runs if (r["n"],r["rs"],r["initialization"],r["mode"],r["width_factor"])==key and r["status"]=="converged_to_protocol"]
        if not data:continue
        energy=np.array([r["energy_Ha"] for r in data]);sem=float(energy.std(ddof=1)/np.sqrt(len(data))) if len(data)>1 else None
        grouped.append(dict(n=key[0],rs=key[1],initialization=key[2],mode=key[3],width_factor=key[4],
            converged_seeds=len(data),energy_Ha_mean=float(energy.mean()),energy_seed_sem_Ha=sem))
    table(output/"grouped_converged.csv",grouped)
    (output/"status.json").write_text(json.dumps(dict(expected=len(plan["cases"]),finished=len(runs),
        converged=sum(r["status"]=="converged_to_protocol" for r in runs),failures=failures),indent=2)+"\n")
    if runs:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(n,rs) in zip(axes.flat,((9,4),(9,25),(16,4),(16,25))):
            for name in {v["id"] for v in trajectories if v["n"]==n and v["rs"]==rs}:
                data=[v for v in trajectories if v["id"]==name and v["head"]==0]
                ax.errorbar([v["step"] for v in data],[v["energy_Ha"] for v in data],yerr=[v["energy_sem_Ha"] for v in data],alpha=.6,marker='.',label=name if len(runs)<8 else None)
            ax.set(title=f'N={n}, rs={rs}; independent checkpoints',xlabel='updates',ylabel='Energy / electron (Ha)');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(figures/"energy_convergence.png",dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,rs in zip(axes,(4.,25.)):
            for name in {v["id"] for v in trajectories if v["n"]==9 and v["rs"]==rs and v["mode"]=="learned" and v["seed"]==101}:
                for h in (0,1):
                    data=[v for v in trajectories if v["id"]==name and v["head"]==h]
                    ax.plot([v["step"] for v in data],[v["half_weight_radius"] for v in data],'.-',label=name+f' h{h}')
            ax.set(title=f'N9 rs={rs}; seed101',xlabel='updates',ylabel='Window half-height (network metric)');ax.grid(alpha=.2);ax.legend(fontsize=5)
        fig.tight_layout();fig.savefig(figures/"width_trajectories.png",dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,y,label in zip(axes,("correlation_R90","xi_connected_peak_curvature","physical_R90"),
                              ("Connected pair R90 (finite cell)","Connected SF curvature proxy","Actual attention physical R90")):
            for rs,color in ((4.,'#2876ac'),(25.,'#d37426')):
                data=[r for r in heads if r["rs"]==rs and r["mode"]=="learned" and r["status"]=="converged_to_protocol" and r.get(y) is not None]
                ax.scatter([r["half_weight_radius"] for r in data],[r[y] for r in data],c=color,s=25,label=f'rs={rs}')
            ax.set(xlabel='Window half-height (network metric)',ylabel=label);ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(figures/"window_correlation.png",dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,rs in zip(axes,(4.,25.)):
            for r in runs:
                if r["rs"]!=rs or r["n"]!=9 or r["seed"]!=101:continue
                with (root/"runs"/r["id"]/"correlations.csv").open() as f:data=list(csv.DictReader(f))
                ax.plot([float(v["r"]) for v in data],[float(v["connected_distinct_canonical"]) for v in data],label=r["mode"]+f' w{r["width_factor"]} '+r["initialization"])
            ax.set(title=f'N9 rs={rs}; seed101',xlabel='physical minimum-image r (rs aB)',ylabel='canonical distinct pair excess');ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(figures/"physical_correlations.png",dpi=170);plt.close(fig)
    lines=["# 連続系：エネルギー収束後のWindowと物理的長さ", "",
        f'終了 {len(runs)}/{len(plan["cases"])} run、収束判定通過 {sum(r["status"]=="converged_to_protocol" for r in runs)}、失敗 {len(failures)}。',"",
        "仮説：過去の幅と相関の不一致が最適化不足によるなら、収束後に対応が明確になるか。対応するという結論は前提にしない。", "",
        "N=9,16、同じλ=0.72、rs=4/25、3seed。N9では3初期幅、固定Gaussian、Full、逆のHF/平面波初期化も比較。N16では標準幅とFullを確認。相の名前は候補である。",
        "Gaussian窓、既存の非正規化Value集約とTFIMとは別の連続Hamiltonianを使用。幅・PHの正則化なし。batch128、20 MCMC stepsと対称な単一電子global hopを追加。",
        "30000更新ずつLR=0.001,0.0003,0.0001,0.00003。その後必要に応じて40000ずつ延長。独立固定checkpoint測定3回の隣接差で、95%誤差範囲が±5e-5 Ha/電子内、SEM<=その1/3、ESS>=400、split Rhat<=1.05、構造因子Rhat<=1.1を要求。",
        "候補通過後もLR=1e-4で10000更新する再加速検証を行う。最大400000更新で未達ならunconverged_budget_limitと記録し、収束済み比較に混ぜない。",
        "各段階は16本の独立一様初期配置、擬スピン反転とglobal hop、burn1024。2048→最大16384 time samples/chainへ延長。無外れ値除去。誤差は時間blockingとchain間SEMの大きい方。",
        "既存checkpointはRNGを持たないため、各段階の乱数を別seedで明示的に再初期化し、Adam状態とwalkerを継続。段階を跨ぐ乱数軌道の完全一致は主張しない。",
        "", "## 実測結果", "", "|N|rs|初期化|窓|初期倍率|seed|更新数|判定|E/電子 [Ha]|SEM|pair R90|",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:lines.append(f'|{r["n"]}|{r["rs"]}|{r["initialization"]}|{r["mode"]}|{r["width_factor"]}|{r["seed"]}|{r["steps"]}|{r["status"]}|{r["energy_Ha"]:.8g}|{r["energy_sem_Ha"]:.3g}|{r["correlation_R90"]}|')
    selected=[r for r in runs if r["status"]=="converged_to_protocol" and r["mode"]=="learned"]
    if selected:
        lines += ["", "## 収束通過runの対応関係（探索的）", ""]
        for n in (9,16):
            for rs in (4.,25.):
                data=[r for r in selected if r["n"]==n and r["rs"]==rs]
                if not data:continue
                x=[];y=[];actual=[]
                for r in data:
                    h=[v for v in heads if v["id"]==r["id"]]
                    if r["correlation_R90"] is not None:
                        x.append(np.mean([v["half_weight_radius"] for v in h]));y.append(r["correlation_R90"])
                        actual.append(np.mean([v["physical_R90"] for v in h if v["physical_R90"] is not None]))
                correlation=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and min(np.std(x),np.std(y))>1.e-12 else None
                ratios=np.asarray(actual)/np.asarray(y) if y else np.array([])
                lines.append(f'N={n}, rs={rs}: 収束run {len(data)}、run平均半値幅とpair R90のPearson={correlation}、実Attention physical R90 / pair R90の中央値={float(np.median(ratios)) if len(ratios) else None}。複数初期化・幅・seedを含む記述統計で、対応の因果や一致を確定しない。')
        for mode in ("full","fixed"):
            data=[v for v in comparisons if v["both_converged"] and ("_"+mode+"_") in v["control"]]
            if data:lines.append(f'{mode}に対する収束済み対応比較 {len(data)} 組：学習幅側−対照のエネルギー差中央値 {np.median([v["delta_energy_Ha"] for v in data]):.6g} Ha/電子。')
    lines += ["", "## 解釈と限界", "",
        "エネルギーの統計的安定は大域的最小値・厳密基底状態の保証ではない。異なる初期化、幅、Fullのseed別結果をpaired_controls.csvで確認する。",
        "固体では長距離秩序や一体密度の周期性を単一の有限相関長に押し込めない。S_total、S_disconnected=|<rho_q>|²/N、S_connectedを分ける。",
        "実空間ではcanonical distinct-pair excess=g(r)−測定一体密度積のradial averageを用いる。grid64による離散化、有限Nの規約、時間ブロックの不確実性を明記する。",
        "物理的距離とネットワークの滑らかな周期距離は異なる。名目幅の一致だけでは判定せず、実Attentionのphysical_R90/R_effも比較する。",
        "pair R90は指定したノイズ閾値と有限r範囲での接続質量距離であり、指数相関長とは呼ばない。3個以上の有意な減衰包絡ピークがない場合xi_envelopeは未解像とする。",
        "構造因子の曲率長も有限格子Lorentzian仮定のproxyであり、セル長への張り付きを明示する。N9/16で熱力学極限の相関長や相を確定しない。",
        "結果が揃った後、収束通過run内の幅と相関proxy、初期幅・seed依存、Fullとのエネルギー差を評価する。収束しなかったrunでは当初の仮説を判定しない。"]
    (output/"REPORT_JA.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);args=p.parse_args();report(args.root.resolve())
