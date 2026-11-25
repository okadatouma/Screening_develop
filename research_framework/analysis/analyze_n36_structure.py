"""Matched-probe, all-seed energy and persistent-homology reanalysis.

Only reads saved outputs. Nulls preserve the edge-weight multiset and support
count, not degrees, geometry, or wavefunction. PH is not a physical invariant.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from scipy.stats import rankdata

from screening_analysis.topology.persistence import persistence, symmetrize, landscape, diagram_distance

KINDS = ["baseline", "six_theta", "spin_onehot", "six_theta_onehot",
         "learned_fourier", "learned_fourier_L1"]
SEEDS = [101, 202, 303]
GRAPHS = ["QK", "Screening", "Window", "PhaseResponse", "FeatureVR"]


def serial(x):
    if isinstance(x, dict): return {str(k): serial(v) for k,v in x.items()}
    if isinstance(x, (tuple,list,np.ndarray)): return [serial(v) for v in x]
    if isinstance(x, np.generic): return serial(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--nulls",type=int,default=199)
    args=p.parse_args();root=args.root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest={};cache={};diagrams={};rows=[];runrows=[];nullrows=[];changes=[];amp_rows=[]
    def read(path):
        b=path.read_bytes();manifest[str(path)]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));return b
    def js(path):return json.loads(read(path))
    def arr(path):
        with np.load(io.BytesIO(read(path)),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    def savejson(name,x): (out/name).write_text(json.dumps(serial(x),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    def savecsv(name,rs):
        with (out/name).open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(serial(rs))
    def figsave(fig,name):
        fig.savefig(out/(name+".png"),dpi=155,bbox_inches="tight")
        fig.savefig(out/(name+".pdf"),bbox_inches="tight");plt.close(fig)
    def rank(w):
        w=w.copy();ix=np.triu_indices(len(w),1);v=w[ix].copy();ok=v>0
        if ok.any():v[ok]=rankdata(v[ok],method="average")/ok.sum()
        w[ix]=v;w[(ix[1],ix[0])]=v;return w
    def ph(w):
        w=symmetrize(w);key=w.tobytes()
        if key in cache:return cache[key]
        distance=1-w;distance[w==0]=np.inf;np.fill_diagonal(distance,0)
        dg=ripser(distance,distance_matrix=True,maxdim=1,thresh=1.,coeff=2)["dgms"]
        values={}
        for dim,d in enumerate(dg):
            finite=d[np.isfinite(d[:,1])];essential=d[~np.isfinite(d[:,1])]
            life=finite[:,1]-finite[:,0]
            # Integral of beta over u in [0,1]. Infinite bars remain infinite in
            # saved diagrams and contribute only their observed interval here.
            values.update({f"H{dim}_area_0_1":float(life.sum()+np.sum(1-essential[:,0])),
                           f"H{dim}_finite_total":float(life.sum()),
                           f"H{dim}_longest_finite":float(life.max(initial=0)),
                           f"H{dim}_essential":len(essential)})
        values["edge_density"]=float(np.mean(w[np.triu_indices(len(w),1)]>0))
        values["landscape"]=landscape(dg[1],np.linspace(0,1,101),levels=3)
        cache[key]=(dg,values);return dg,values
    def null_area(ws,rng):
        # A full null draw shuffles each of the eight graph edge multisets.
        vals=[]
        for w in ws:
            ix=np.triu_indices(len(w),1);z=w.copy();v=rng.permutation(w[ix]);z[ix]=v;z[(ix[1],ix[0])]=v
            vals.append(ph(z)[1]["H1_area_0_1"])
        return float(np.mean(vals))

    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    (out/"analyze_n36_structure.py").write_bytes(read(Path(__file__).resolve()))
    plan=js(root/"plan.json");rng=np.random.default_rng(20261109)
    curves={};rawgraphs={};baseline_probe=None;window_max_change=0.;reference_checked=False
    for case in plan:
        d=root/case["name"];c=case["config"];seed=c["experiment"]["seed"]
        assert (d/"FOLLOWUP_COMPLETE.json").exists()
        kind=c["embedding"]["type"]+("_L1" if c["embedding"]["sparse_penalty"] else "")
        hist=[json.loads(x) for x in read(d/"history.jsonl").splitlines() if x]
        e=np.array([v["energy"] for v in hist]);assert len(e)==10000 and np.isfinite(e).all()
        curves[(kind,seed)]=e
        a={label:arr(d/"diagnostics"/(label+"_internal.npz")) for label in ["before","after"]}
        assert np.array_equal(a["before"]["positions"],a["after"]["positions"])
        assert np.array_equal(a["before"]["spins"],a["after"]["spins"])
        if baseline_probe is None:baseline_probe=(a["before"]["positions"],a["before"]["spins"])
        assert all(np.array_equal(x,y) for x,y in zip(baseline_probe,(a["before"]["positions"],a["before"]["spins"])))
        delta_window=float(np.max(abs(a["before"]["layer0_window"]-a["after"]["layer0_window"])))
        window_max_change=max(window_max_change,delta_window)
        metrics=dict(name=case["name"],kind=kind,seed=seed,
                     first1000=float(e[:1000].mean()),last1000=float(e[-1000:].mean()),
                     training_drop=float(e[-1000:].mean()-e[:1000].mean()),
                     mean_all10000=float(e.mean()),tail_change=float(e[-1000:].mean()-e[-2000:-1000].mean()),
                     last2000_slope_per1000=float(np.polyfit(np.arange(2000),e[-2000:],1)[0]*1000),
                     hf_relative_orbital_correction=js(d/"diagnostics/hf_correction.json")["relative_orbital_correction"])
        stage_graphs={}
        for label,x in a.items():
            phase=x["phase_log_edge_sensitivity"]
            metrics[label+"_phase_rms"]=float(np.sqrt(np.mean(phase**2)))
            metrics[label+"_phase_max"]=float(np.max(abs(phase)))
            fourier=list(csv.DictReader(io.StringIO(read(d/"diagnostics"/(label+"_fourier.csv")).decode())))
            metrics[label+"_crystal_qk"]=float(np.mean([abs(float(v["QK_Fourier"])) for v in fourier if max(abs(int(v["qx"])),abs(int(v["qy"])))==6]))
            spin=js(d/"diagnostics"/(label+"_spin_selection.json"))
            metrics[label+"_spin_selectivity"]=spin["same"]-spin["opposite"]
            for graph in GRAPHS:
                if graph=="FeatureVR":
                    feat=x["features"];norm=np.linalg.norm(feat,axis=-1,keepdims=True)
                    assert np.all(norm>1e-12)
                    unit=feat/norm;cos=np.einsum("bid,bjd->bij",unit,unit)
                    ws=(1-np.sqrt(np.maximum(0.,2-2*np.clip(cos,-1,1)))/2)[:,None]
                elif graph=="PhaseResponse":
                    ws=abs(phase);mx=ws.max(axis=(-1,-2),keepdims=True)
                    ws=np.divide(ws,mx,out=np.zeros_like(ws),where=mx>0)
                else:
                    key={"QK":"similarity","Screening":"aggregation_weights","Window":"window"}[graph]
                    ws=x["layer0_"+key]
                    if graph=="QK":ws=(np.clip(ws,-1,1)+1)/2
                ws=np.array([[symmetrize(np.clip(w,0,1)) for w in ww] for ww in ws])
                stage_graphs[(label,graph)]=ws
                rawgraphs[(case["name"],label,graph)]=ws
                for ci in range(ws.shape[0]):
                    for hi in range(ws.shape[1]):
                        for variant in ["raw","rank"]:
                            w=ws[ci,hi] if variant=="raw" else rank(ws[ci,hi])
                            dg,st=ph(w)
                            if not reference_checked and graph=="QK" and label=="after":
                                old=persistence(w)
                                for di in [0,1]:
                                    assert len(old[di])==len(dg[di])
                                    assert diagram_distance(old[di],dg[di],bottleneck=True)<1e-6
                                reference_checked=True
                            row=dict(name=case["name"],kind=kind,seed=seed,stage=label,graph=graph,
                                     variant=variant,configuration=ci,head=hi,
                                     **{k:v for k,v in st.items() if k!="landscape"})
                            rows.append(row)
                            tag=f'{case["name"]}__{label}__{graph}__{variant}__c{ci}h{hi}'
                            for di in [0,1]:diagrams[tag+f"__H{di}"]=dg[di]
            print("PH measured",case["name"],label,flush=True)
        for graph in GRAPHS:
            for variant in ["raw","rank"]:
                prev=stage_graphs[("before",graph)];post=stage_graphs[("after",graph)]
                distances=[];deltas=[];pre=[];last=[]
                for ci,hi in itertools.product(range(prev.shape[0]),range(prev.shape[1])):
                    st=[]
                    for ww in [prev[ci,hi],post[ci,hi]]:st.append(ph(ww if variant=="raw" else rank(ww))[1])
                    distances.append(float(np.sqrt(np.mean((st[1]["landscape"]-st[0]["landscape"])**2))))
                    pre.append(st[0]["H1_area_0_1"]);last.append(st[1]["H1_area_0_1"])
                    deltas.append(last[-1]-pre[-1])
                changes.append(dict(name=case["name"],kind=kind,seed=seed,graph=graph,variant=variant,
                                    before_H1_area=float(np.mean(pre)),after_H1_area=float(np.mean(last)),
                                    delta_H1_area=float(np.mean(deltas)),landscape_L2=float(np.mean(distances))))
        if "fourier_amplitudes" in hist[-1]:
            for i,amp in enumerate(hist[-1]["fourier_amplitudes"]):
                amp_rows.append(dict(name=case["name"],kind=kind,seed=seed,candidate=i,
                                     qx=c["embedding"]["modes"][i][0],qy=c["embedding"]["modes"][i][1],amplitude=amp))
        runrows.append(metrics)

    # Conditional null on per-configuration HEAD-MEAN graphs. Never average
    # adjacency across unrelated configurations / electron-index assignments.
    for ci,case in enumerate(plan):
        rr=runrows[ci]
        for graph in ["QK","Screening"]:
            ws=rawgraphs[(case["name"],"after",graph)].mean(axis=1)
            ws=np.array([symmetrize(w) for w in ws])
            observed=float(np.mean([ph(w)[1]["H1_area_0_1"] for w in ws]))
            draws=np.array([null_area(ws,rng) for _ in range(args.nulls)])
            upper=(1+np.sum(draws>=observed))/(args.nulls+1)
            lower=(1+np.sum(draws<=observed))/(args.nulls+1)
            nullrows.append(dict(name=case["name"],kind=rr["kind"],seed=rr["seed"],graph=graph,
                                 observed=observed,null_mean=float(draws.mean()),null_sd=float(draws.std(ddof=1)),
                                 difference=float(observed-draws.mean()),p_two_sided=float(min(1,2*min(upper,lower))),
                                 permutations=args.nulls))
            print("null",case["name"],graph,"observed",round(observed,4),"null",round(draws.mean(),4),flush=True)
            # Avoid retaining every shuffled diagram in memory.
            cache.clear()
    ps=np.array([r["p_two_sided"] for r in nullrows]);order=np.argsort(ps)
    adjusted=np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
    for i,q in zip(order,adjusted):nullrows[i]["BH_q"]=float(min(q,1))
    grouped=[]
    for kind in KINDS:
        rs=[r for r in runrows if r["kind"]==kind]
        g=dict(kind=kind)
        for k in ["first1000","last1000","training_drop","mean_all10000","before_crystal_qk","after_crystal_qk","after_phase_rms","hf_relative_orbital_correction"]:
            g[k+"_mean"]=float(np.mean([r[k] for r in rs]));g[k+"_seed_sd"]=float(np.std([r[k] for r in rs],ddof=1))
        for graph in GRAPHS:
            for variant in ["raw","rank"]:
                selected=[r for r in changes if r["kind"]==kind and r["graph"]==graph and r["variant"]==variant]
                g[graph+"_"+variant+"_area_delta"]=float(np.mean([r["delta_H1_area"] for r in selected]))
                g[graph+"_"+variant+"_landscape_change"]=float(np.mean([r["landscape_L2"] for r in selected]))
        grouped.append(g)
    contrasts=[]
    for kind in KINDS[1:]:
        vals=[]
        for seed in SEEDS:
            base=next(r for r in runrows if r["kind"]=="baseline" and r["seed"]==seed)
            v=next(r for r in runrows if r["kind"]==kind and r["seed"]==seed)
            delta=v["last1000"]-base["last1000"];vals.append(delta)
            contrasts.append(dict(kind=kind,seed=seed,last1000_minus_baseline=delta,
                                  all10000_mean_minus_baseline=v["mean_all10000"]-base["mean_all10000"]))
        # Exact two-sided sign-flip test across the THREE independent seeds.
        observed=abs(np.mean(vals));null=[abs(np.mean(np.array(vals)*s)) for s in itertools.product([-1,1],repeat=3)]
        print("contrast",kind,"delta",vals,"exact_seed_p",np.mean(np.array(null)>=observed-1e-12),flush=True)
    savecsv("run_structure.csv",runrows);savecsv("group_structure.csv",grouped)
    savecsv("ph_graphs.csv",rows);savecsv("ph_changes.csv",changes)
    savecsv("conditional_edge_nulls.csv",nullrows);savecsv("energy_paired_contrasts.csv",contrasts)
    savecsv("fourier_final_amplitudes.csv",amp_rows)
    np.savez_compressed(out/"persistence_diagrams.npz",**diagrams)
    savejson("summary.json",dict(timestamp=now.isoformat(),runs=18,groups=grouped,
              conditional_nulls=nullrows,window_max_absolute_change=window_max_change,
              matched_uniform_probes_all_runs=True,reference_reducer_checked=reference_checked,
              null_definition="8 per-configuration, head-mean graphs; independently shuffled edge multiset including zeros; two-sided MC p, BH across 36 tests; conditional descriptive test only",
              inference_unit="3 independent seeds. Configurations/heads are not independent training replications."))
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.2})
    colors={s:f"C{i}" for i,s in enumerate(SEEDS)}
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,kind in zip(axes.flat,KINDS):
        for seed in SEEDS:
            e=curves[(kind,seed)];ax.plot(np.arange(100,10001,100),e.reshape(100,100).mean(1),color=colors[seed],label=str(seed))
        ax.set(title=kind,xlabel="Training update",ylabel="Training energy (stored units)")
    axes[0,0].legend(title="seed");fig.tight_layout();figsave(fig,"training_energy_all18")
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for j,kind in enumerate(KINDS):
        rs=[r for r in runrows if r["kind"]==kind]
        for r in rs:
            axes[0].scatter(j,r["last1000"],color=colors[r["seed"]])
            axes[1].plot([j-.15,j+.15],[r["before_crystal_qk"],r["after_crystal_qk"]],"o-",color=colors[r["seed"]],alpha=.7)
            axes[2].scatter(j,r["after_phase_rms"],color=colors[r["seed"]])
    for ax in axes:ax.set_xticks(range(6),KINDS,rotation=65,ha="right")
    axes[0].set_ylabel("Last 1000 training energy")
    axes[1].set_ylabel("Crystal-shell |QK Fourier|: before -> after")
    axes[2].set_ylabel("Wavefunction phase / log-edge sensitivity RMS")
    axes[2].set_yscale("log");fig.tight_layout();figsave(fig,"training_and_wavefunction_response")
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ai,(graph,variant) in enumerate(itertools.product(["QK","Screening","FeatureVR"],["raw","rank"])):
        ax=axes.flat[ai]
        for j,kind in enumerate(KINDS):
            for seed in SEEDS:
                v=next(r for r in changes if r["kind"]==kind and r["seed"]==seed and r["graph"]==graph and r["variant"]==variant)
                ax.plot([j-.15,j+.15],[v["before_H1_area"],v["after_H1_area"]],"o-",color=colors[seed],alpha=.7)
        ax.set_title(graph+" / "+variant+" / before -> after")
        ax.set_ylabel("Mean integral of beta1 over u in [0,1]")
        ax.set_xticks(range(6),KINDS,rotation=65,ha="right")
    fig.tight_layout();figsave(fig,"persistent_homology_before_after")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,graph in zip(axes,["QK","Screening"]):
        vs=[r for r in nullrows if r["graph"]==graph]
        ax.errorbar(range(18),[r["null_mean"] for r in vs],yerr=[2*r["null_sd"] for r in vs],fmt=".",color="gray",label="Edge-shuffle null mean +/- 2 SD")
        ax.scatter(range(18),[r["observed"] for r in vs],c=[KINDS.index(r["kind"]) for r in vs],cmap="tab10",label="Observed")
        ax.set_xticks(range(18),[r["kind"]+"/"+str(r["seed"]) for r in vs],rotation=90,fontsize=6)
        ax.set(title=graph+" / head-mean graphs",ylabel="Mean integrated beta1");ax.legend(fontsize=7)
    fig.tight_layout();figsave(fig,"persistent_homology_edge_nulls")

    lines=["# N36・全18条件：エネルギー低下と内部構造の評価", "",
           f"作成開始 {now:%Y-%m-%d %H:%M} JST。全18条件は学習・個別除去解析まで完了。", "",
           "## 評価できる範囲", "",
           "固定した同一の8配置・擬スピンを、全方式・全seedの学習前後で比較した。独立評価鎖の未収束があっても、この固定入力上でのネットワーク構造の変化、位相の依存性、学習ログの変化は測定できる。物理的な基底状態・相・変分エネルギーの優劣とは区別する。", "",
           "## 学習エネルギーと機能的応答", "",
           "エネルギーはコード内部単位。平均±は3 seedの標準偏差。学習中のwalkerでの値であり、独立評価された期待値ではない。", "",
           "|方式|末尾1000更新 E|最初1000→末尾1000の低下|結晶QK 前→後|学習後の位相応答RMS|",
           "|---|---:|---:|---:|---:|"]
    for g in grouped:
        lines.append(f'|{g["kind"]}|{g["last1000_mean"]:.2f} ± {g["last1000_seed_sd"]:.2f}|{g["training_drop_mean"]:.2f}|{g["before_crystal_qk_mean"]:.4f} → {g["after_crystal_qk_mean"]:.4f}|{g["after_phase_rms_mean"]:.4g}|')
    lines += ["", "![学習エネルギー](training_energy_all18.png)", "",
              "baselineとsix_thetaの同一seedの比較はenergy_paired_contrasts.csv。3 seedは独立な繰り返しで、各配置・各headを独立seedとして数えない。3 seedの両側符号反転検定は最小p=0.25なので、同方向の差が揃っても通常の5%有意差とは呼ばない。", "",
              "学習前の位相応答はNN軌道readoutをゼロ初期化したため全条件ゼロ。学習後の非ゼロ応答は、内部edgeが波動関数の位相に実際に寄与することを示す。ただし選んだ規約と8入力での局所微分であり、相の秩序変数ではない。HF相対軌道補正は保存された1配置での値で、全分布の距離ではない。", "",
              "![エネルギー・波数・波動関数応答](training_and_wavefunction_response.png)", "",
              "## パーシステントホモロジー", "",
              "保存時にはPHが無効だったため、保存された内部配列から今回新規に計算した。粒子36点のQK、実集約重みScreening、固定距離Window、位相応答の絶対値、最終粒子特徴のFeatureVRを解析。QK等は対称化した重みw∈[0,1]のflag filtration u=1-w、FeatureVRは単位化特徴間Euclidean距離/2。H0/H1・F2係数。ゼロ重みの辺はu=1でも不在。", "",
              "各配置・各headで計算してから指標を平均し、異なる配置の隣接行列を平均していない。FeatureVRはhead軸なし。rawと、正の辺の順位に変換したrankの両方を比較。rankは重み分布を除くが、ゼロ辺の支持・密度差は残る。", "",
              "H1面積はu∈[0,1]のBetti数積分。essential barは保存図では∞を保持し、積分では観測区間だけを寄与させた。面積が大きいほど良い波動関数という基準ではない。", "",
              f"固定Windowの前後最大差は{window_max_change:.3g}。そのPHは学習前から存在し、学習による獲得と解釈しない。学習前も内部QKにPHがあり得るが、波動関数の位相応答はゼロだった。PHの存在と波動関数への機能的寄与は別々に調べる必要がある。", "",
              "![PH学習前後](persistent_homology_before_after.png)", "",
              "## 辺シャッフル対照", "",
              f"学習後のQK/Screeningについて、各配置でhead平均した8グラフを用いた。{args.nulls}回の対照それぞれで、各グラフの辺重み（ゼロを含む）の多重集合を保持して配置だけを無作為化。重み分布と支持辺数を揃えた対照で、次数・距離・擬スピン関係は保存しない。H1面積の両側Monte Carlo pを算出し36検定でBH補正した。これは保存グラフに条件付けた事後探索で、3 seed間の方式差や物理相に関する有意差ではない。", "",
              f"BH q<0.05：{sum(r['BH_q']<.05 for r in nullrows)}/{len(nullrows)}。全結果はconditional_edge_nulls.csv。差がある場合も幾何・入力特徴・学習のどれが寄与したかは前後比較と併せて判断する。", "",
              "![辺シャッフル対照](persistent_homology_edge_nulls.png)", "",
              "## 保存物と検証", "",
              "run_structure.csv：18条件の学習・QK・位相応答。ph_graphs.csv：各配置/headのPH。ph_changes.csv：同じ配置/headでの前後変化。persistence_diagrams.npz：H0/H1図（∞を保持）。fourier_final_amplitudes.csv：全seedの振幅。summary.json：集計・手法。", "",
              "Ripserの結果は既存reference reducerと照合した。完了runの全入力SHA256を解析終了時にも確認する。元の学習・checkpoint・診断ファイルは変更しない。", ""]
    (out/"REPORT_JA.md").write_text("\n".join(lines))
    for path,v in manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==v["sha256"]
    savejson("input_manifest.json",manifest)
    savejson("validation.json",dict(input_files_unchanged=len(manifest),all18_complete=True,
               matched_probes=True,reference_reducer_checked=reference_checked,
               window_max_absolute_change=window_max_change,ph_rows=len(rows),null_tests=len(nullrows)))
    print("DONE",out,flush=True)
    print(json.dumps(serial(grouped),indent=2),flush=True)


if __name__=="__main__":main()
