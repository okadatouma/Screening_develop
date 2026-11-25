"""Offline audit of saved N36 runs; never writes to a training run.

Run from repository root with the research_framework CPU virtual environment.
An existing output directory is rejected. Energies retain their stored units.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import pickle
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from screening_analysis.converged_window_study import time_statistics

LABELS = ["baseline", "six_theta", "spin_onehot", "six_theta_onehot",
          "learned_fourier", "learned_fourier_L1"]
SEEDS = [101, 202, 303]
MODES = np.array([[1, 0], [0, 1], [1, -1], [6, 0], [0, 6], [6, -6]])


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}

    def read(path):
        data = path.read_bytes()
        manifest[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data

    def load(path):
        return json.loads(read(path))

    def arrays(path):
        with np.load(io.BytesIO(read(path)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def write_json(name, value):
        (out / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n")

    def write_csv(name, rows):
        if not rows:
            return
        with (out / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(clean(rows))

    def figure(fig, name):
        fig.savefig(out / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    def stats(x):
        result = time_statistics(x)
        result["chain_means"] = x.mean(axis=0).tolist()
        result["second_minus_first_half"] = float(x[len(x)//2:].mean() - x[:len(x)//2].mean())
        result["min"] = float(x.min())
        result["max"] = float(x.max())
        # ESS here is a within-chain autocorrelation estimate. It is not a
        # multi-chain effective sample size or a pass certificate when Rhat fails.
        return result

    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    (out / "analyze_n36_snapshot.py").write_bytes(read(Path(__file__).resolve()))
    frozen_source = root / "source/Screening_topology_gpu"
    for relative in ["research_framework/task_adapters/continuous.py",
                     "research_framework/followup/continuous.py"]:
        read(frozen_source / relative)
    read(Path(__file__).resolve().parents[2] / "screening_analysis/converged_window_study.py")
    plan = load(root / "plan.json")
    protocol = load(root / "protocol.json")
    status = subprocess.run(["squeue", "-u", "tomax", "-o",
                             "%.22i %.30j %.10T %.12M %R"],
                            capture_output=True, text=True)
    (out / "squeue.txt").write_text(status.stdout + status.stderr)
    details, runs, ablations, structure, probes, initial = [], [], [], [], [], {}
    curves, amplitudes, eval_series = {}, {}, {}

    for case in plan:
        d = root / case["name"]
        c = case["config"]
        seed = c["experiment"]["seed"]
        kind = c["embedding"]["type"]
        if c["embedding"]["sparse_penalty"]:
            kind += "_L1"
        history_bytes = read(d / "history.jsonl")
        lines = history_bytes.splitlines()
        # A live last line may be partial: omit it only if it has no newline.
        if not history_bytes.endswith(b"\n"):
            lines = lines[:-1]
        history = [json.loads(line) for line in lines if line.strip()]
        steps = np.array([row["step"] for row in history])
        energy = np.array([row["energy"] for row in history])
        assert np.isfinite(energy).all() and np.all(np.diff(steps) == 1)
        curves[(kind, seed)] = (steps, energy)
        if "fourier_amplitudes" in history[-1]:
            amplitudes[(kind, seed)] = (steps, np.array([row["fourier_amplitudes"] for row in history]))
        complete = (d / "FOLLOWUP_COMPLETE.json").exists()
        row = dict(name=case["name"], kind=kind, seed=seed, complete=complete,
                   last_step=int(steps[-1]), train_last1000=float(energy[-1000:].mean()),
                   train_previous1000=float(energy[-2000:-1000].mean()),
                   train_acceptance_last1000=float(np.mean([v["acceptance"] for v in history[-1000:]])),
                   energy=None, reported_sem=None, conservative_sem=None, rhat=None,
                   within_chain_ess=None, initial_energy=None, initial_rhat=None,
                   eval_minus_train=None, delta_from_initial=None, half_drift=None,
                   train_spin_mean=None, eval_spin_mean=None, imag_mean=None,
                   imag_sem=None, extreme_sample_variance_fraction=None)
        detail = dict(name=case["name"], complete=complete, last_step=int(steps[-1]))
        if not complete:
            runs.append(row)
            details.append(detail)
            continue
        summary = load(d / "summary.json")
        a = arrays(d / "evaluation.npz")
        e, b = a["local_energy"], a["initial_energy"]
        assert e.shape == (64, 8) and b.shape == e.shape
        assert np.isfinite(e).all() and np.isfinite(b).all()
        assert np.isclose(e.real.mean(), summary["energy"], atol=1e-10, rtol=0)
        es, bs, ims = stats(e.real), stats(b.real), stats(e.imag)
        eval_series[(kind, seed)] = e.real
        if seed in initial:
            assert np.array_equal(b, initial[seed]["array"]), "HF starts duplicated across kinds must match"
        else:
            initial[seed] = dict(array=b, stats=bs)
        latest = load(d / "latest.json")
        checkpoint = pickle.loads(read(d / latest["checkpoint"]))  # trusted local output
        assert checkpoint["step"] == 10000 == latest["step"] == steps[-1]
        walkers = checkpoint["state"]["walkers"]
        centered = (e.real - e.real.mean())**2
        row.update(energy=summary["energy"], reported_sem=summary["block_sem"],
                   conservative_sem=es["sem"], rhat=es["rhat"], within_chain_ess=es["ess"],
                   initial_energy=bs["mean"], initial_rhat=bs["rhat"],
                   eval_minus_train=es["mean"] - row["train_last1000"],
                   delta_from_initial=es["mean"] - bs["mean"],
                   half_drift=es["second_minus_first_half"],
                   train_spin_mean=float(np.asarray(walkers.spins).mean()),
                   eval_spin_mean=float(a["spins"].mean()), imag_mean=ims["mean"], imag_sem=ims["sem"],
                   extreme_sample_variance_fraction=float(centered.max()/centered.sum()))
        detail.update(energy=es, initial_energy=bs, imaginary_energy=ims,
                      source_summary=summary, latest_checkpoint=latest,
                      hf_correction=load(d / "diagnostics/hf_correction.json"))
        n = c["physics"]["n"]
        shape = (*e.shape, n, 2)
        frac = a["positions"].reshape(shape) @ np.linalg.inv(c["physics"]["lattice"]).T
        rho = np.exp(2j*np.pi*np.einsum("tcnd,qd->tcnq", frac, MODES)).sum(2)
        sq = abs(rho)**2/n
        for qi, q in enumerate(MODES):
            ss = stats(sq[..., qi])
            structure.append(dict(name=case["name"], kind=kind, seed=seed, qx=int(q[0]),
                                  qy=int(q[1]), sq=ss["mean"], conservative_sem=ss["sem"], rhat=ss["rhat"]))
            idx = list(map(list, summary["fourier_modes"])).index(q.tolist())
            assert np.isclose(ss["mean"], summary["structure_factor"][idx], atol=1e-10)
        for label in ["before", "after", "born_after"]:
            text = read(d / "diagnostics" / (label + "_fourier.csv")).decode()
            for v in csv.DictReader(io.StringIO(text)):
                probes.append(dict(name=case["name"], kind=kind, seed=seed, label=label,
                                   **{k: float(value) for k, value in v.items()}))
        if (d / "candidate_response.json").exists():
            candidate = load(d / "candidate_response.json")
            detail["candidate_response"] = candidate
            for v in candidate["rows"]:
                ca = arrays(d / "diagnostics" / f'candidate_{v["candidate"]}_born.npz')
                cs = stats(ca["local_energy"].real)
                assert np.isclose(cs["mean"], v["energy"], atol=1e-10)
                ablations.append(dict(name=case["name"], kind=kind, seed=seed,
                                      candidate=v["candidate"], qx=v["qx"], qy=v["qy"],
                                      amplitude=v["amplitude"], delta_energy=v["delta_energy"],
                                      rhat=cs["rhat"], conservative_sem=cs["sem"],
                                      reference_reported_sem=v["baseline_sem"],
                                      nominal_combined_sem=float(np.hypot(cs["sem"], v["baseline_sem"])),
                                      phase_circular_rms=v["phase_circular_rms"]))
        runs.append(row)
        details.append(detail)

    completed = [r for r in runs if r["complete"]]
    groups = []
    for kind in LABELS:
        rs = [r for r in completed if r["kind"] == kind]
        energies = np.array([r["energy"] for r in rs])
        groups.append(dict(kind=kind, n=len(rs), seeds=[r["seed"] for r in rs],
                           energy_mean=float(energies.mean()),
                           energy_seed_sd=float(energies.std(ddof=1)) if len(rs)>1 else None,
                           train_last1000_mean=float(np.mean([r["train_last1000"] for r in rs])),
                           rhat_min=min(r["rhat"] for r in rs), rhat_max=max(r["rhat"] for r in rs)))
    write_csv("run_summary.csv", runs)
    write_csv("group_summary.csv", groups)
    write_csv("structure_factor.csv", structure)
    write_csv("fourier_ablation.csv", ablations)
    write_csv("attention_fourier_probes.csv", probes)
    write_json("diagnostics.json", dict(timestamp=now.isoformat(), protocol=protocol,
               completed=len(completed), planned=len(plan), groups=groups, runs=details,
               unique_initial_hf={s: v["stats"] for s, v in initial.items()},
               interpretation="All energy Rhat values fail 1.05; no ranking or causal mode-importance claim.",
               ess_limitation="Within-chain ESS is only descriptive when chains disagree; 64 time points are short."))

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, LABELS):
        for si, seed in enumerate(SEEDS):
            steps, en = curves[(kind, seed)]
            count = len(en)//100
            ax.plot(steps[:count*100].reshape(-1,100).mean(1),
                    en[:count*100].reshape(-1,100).mean(1), label=f"seed {seed}", color=f"C{si}")
            rr = next(r for r in runs if r["kind"] == kind and r["seed"] == seed)
            if rr["complete"]:
                ax.errorbar(10200+si*130, rr["energy"], yerr=rr["conservative_sem"],
                            fmt="o", color=f"C{si}", capsize=3)
        ax.set_title(kind)
        ax.set_xlabel("Training update; dots at right = fresh fixed-state measurement")
        ax.set_ylabel("Stored energy units")
    axes[0,0].legend()
    fig.suptitle("N36: 100-update averages vs fresh-chain evaluation (dots: descriptive SEM)")
    fig.tight_layout()
    figure(fig, "training_vs_evaluation")

    fig, axes = plt.subplots(1,2,figsize=(14,7))
    y = np.arange(len(completed))
    ax=axes[0]
    ax.errorbar([r["energy"] for r in completed], y,
                xerr=[r["conservative_sem"] for r in completed], fmt="o", capsize=3,
                label="Fresh evaluation, descriptive SEM")
    ax.scatter([r["train_last1000"] for r in completed], y, marker="x", label="Training last 1000")
    ax.scatter([r["initial_energy"] for r in completed], y, marker="|", label="Fresh initial HF")
    ax.set_yticks(y, [f'{r["kind"]} / {r["seed"]}' for r in completed])
    ax.set_xlabel("Stored energy units (not Ha/e)")
    ax.legend(fontsize=8)
    ax=axes[1]
    ax.barh(y,[r["rhat"] for r in completed],color="C3")
    ax.axvline(1.05,color="black",linestyle="--",label="Diagnostic threshold 1.05")
    ax.set_yticks(y,[])
    ax.set_xlabel("Energy split-chain Rhat (64 times x 8 chains)")
    ax.legend()
    fig.suptitle("All completed runs fail the energy mixing diagnostic; no physical ranking")
    fig.tight_layout()
    figure(fig,"energy_and_mixing")

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axes.flat,[("baseline",101),("six_theta",303),
                                ("learned_fourier",202),("six_theta_onehot",303)]):
        e=eval_series[key]
        ax.plot(np.arange(1,len(e)+1),e,alpha=.65,linewidth=.8)
        ax.set_title(f"{key[0]} / seed {key[1]}")
        ax.set_xlabel("Retained time after 256 burn-in sweeps (thinning=4)")
        ax.set_ylabel("Local energy, stored units")
    fig.suptitle("Fresh evaluation: 8 chains show persistent offsets and drift")
    fig.tight_layout()
    figure(fig,"evaluation_chains")

    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ki,kind in enumerate(LABELS[-2:]):
        for si,seed in enumerate(SEEDS):
            ax=axes[ki,si]
            steps,amp=amplitudes[(kind,seed)]
            for qi,q in enumerate(MODES):
                ax.plot(steps[::50],amp[::50,qi],label=str(tuple(q)))
            rr=next(r for r in runs if r["kind"]==kind and r["seed"]==seed)
            ax.set_title(f'{kind} / {seed}' + (" (training)" if not rr["complete"] else ""))
            ax.set_xlabel("Training update")
            ax.set_ylabel("Fourier amplitude (coordinate, not importance)")
    axes[0,0].legend(ncol=2,fontsize=7)
    fig.suptitle("L1 suppresses several coordinates; mode importance needs converged interventions")
    fig.tight_layout()
    figure(fig,"fourier_amplitudes")

    all_done = len(completed) == len(plan)
    text=["# N36 CPUジョブの解析" + ("（全条件完了）" if all_done else "（途中スナップショット）"), "",
          f"確認日時：{now:%Y年%m月%d日 %H:%M} JST。対象：Slurm 2910、N=36、rs=25、λ=0.72。", "",
          f"解析完了は{len(completed)}/18条件。" + ("全方式・全seedを集計した。" if all_done else "残りは末尾の実行状態表に記載。これは全条件完了後の最終報告ではない。"), "",
          "## 結論", "",
          "学習は進んでいるが、完了条件すべてで独立評価のエネルギー鎖間診断が不良。現状のエネルギー表からEmbeddingの優劣、結晶化、AHC、Fourier波数の重要性は確定できない。", "",
          f"完了{len(completed)}条件のエネルギーsplit-Rhatは{min(r['rhat'] for r in completed):.3f}〜{max(r['rhat'] for r in completed):.3f}で、診断目安1.05以下を通る条件はない。保存標本は64時点×8鎖=512個。学習前HFの独立評価もRhat={min(v['stats']['rhat'] for v in initial.values()):.3f}〜{max(v['stats']['rhat'] for v in initial.values()):.3f}。", "",
          "短い独立鎖では、学習中の低いエネルギーが再現されない。これはサンプリングの問題を示す強い証拠だが、現在の標本だけでは最適化不良・評価推定の偏り・実装上の原因を分離できない。低い学習エネルギーも真の期待値とは保証できない。", "",
          "## エネルギーの比較（記述統計）", "",
          "以下は保存されたコード内部のエネルギー単位。Ha/電子と混同しない。±はseed間標準偏差であり、平均の誤差や物理的に信頼できる信頼区間ではない。評価鎖が未収束なので、方式間の平均を物理的な優劣として順位付けしない。", "",
          "|方式|完了seed数|独立評価の平均 ± seed標準偏差|学習末尾1000更新の平均|評価Rhat範囲|",
          "|---|---:|---:|---:|---:|"]
    for g in groups:
        text.append(f'|{g["kind"]}|{g["n"]}|{g["energy_mean"]:.2f} ± {g["energy_seed_sd"]:.2f}|{g["train_last1000_mean"]:.2f}|{g["rhat_min"]:.2f}–{g["rhat_max"]:.2f}|')
    text += ["", "全条件の値は[run_summary.csv](run_summary.csv)、方式別記述統計は[group_summary.csv](group_summary.csv)。",
             "", "![学習曲線と独立評価](training_vs_evaluation.png)", "",
             "## サンプリング診断", "",
             "保存された[time, chain]軸を保って既存のtime_statisticsで診断した。SEMは鎖平均の標準誤差と複数の時間ブロックSEMの最大値。元のsummaryは時刻ブロック内で8鎖を平均しており、鎖間の持続的なずれを十分反映しない。再計算SEMも非平衡バイアスを含む保証はなく、図の誤差棒は記述値である。",
             "", "ESSは鎖内自己相関からの参考値。高いRhatの下では多鎖の有効標本数として解釈しない。Rhat自体も64時点では精度が限られ、低くても収束証明にはならない。",
             "", f"独立評価−学習末尾平均の差は{min(r['eval_minus_train'] for r in completed):.1f}〜{max(r['eval_minus_train'] for r in completed):.1f}。学習前HF−学習後評価の差も評価鎖が未収束なので、物理的な悪化の確定には使わない。",
             "", "![評価値とRhat](energy_and_mixing.png)", "",
             "![独立評価の鎖](evaluation_chains.png)", "",
             "学習前HFは同じseed内で全方式の配列が完全一致することを検証した。独立なHF参照は3本のseed集合であり、16個の独立参照として数えない。",
             "", "## Fourier振幅と除去応答", "",
             "L1ありでは複数の振幅座標が小さくなる。これは疎性項に対する座標の応答であり、物理的な秩序や波数の因果的重要性ではない。振幅と射影・NNの間には再パラメータ化の自由度が残る。",
             "", "![Fourier振幅履歴](fourier_amplitudes.png)", "",
             f"完了したFourier条件の個別除去{len(ablations)}評価も、エネルギーRhat={min(r['rhat'] for r in ablations):.3f}〜{max(r['rhat'] for r in ablations):.3f}で、すべて1.05を超えた。除去後エネルギー差の符号を波数の有用性の証拠として採用しない。候補除去の参照状態の生のエネルギー配列は保存されていないため、その参照Rhatは今回再計算できない。",
             "", "数値は[fourier_ablation.csv](fourier_ablation.csv)。nominal_combined_semは候補側の再計算SEMと参照側の元のblock SEMを合成した参考値で、妥当な信頼区間ではない。位相応答も固定した規約・入力での診断量であり、ゲージ不変な物理観測量とは扱わない。",
             "", "## 構造因子・内部表現", "",
             f"全512標本から6波数のS(q)を再計算し、元summaryと一致することを検証した。{len(completed)}条件×6波数の{len(structure)}診断では、Rhat={min(r['rhat'] for r in structure):.3f}〜{max(r['rhat'] for r in structure):.3f}、1.05以下は{sum(r['rhat']<=1.05 for r in structure)}件。[structure_factor.csv](structure_factor.csv)に鎖別診断を併記。結晶相判定には用いない。",
             "", "学習前後の固定probeと学習後Born probeのQK/Screening/Spin Fourier診断は[attention_fourier_probes.csv](attention_fourier_probes.csv)。各probeは8配置のみ。与えた高周波特徴や供給HF由来の構造と、学習で獲得した構造を区別する必要があり、内部量のピークだけで秩序を主張しない。",
             "", "## 次に必要な確認", "",
             "1. 全18条件の比較を同じ精度の測定に揃える。",
             "2. 固定した最終checkpointで独立鎖を長く回し、burn-in長・開始配置・鎖数を変えてエネルギーとS(q)が一致するか確認する。現在はburn-in256、4 sweepごとに64時点のみ。",
             "3. 鎖間のずれが解消した後に、複数checkpointでのエネルギーの定常性を評価する。学習は2 sweeps/更新、固定学習率0.002、10000更新で終了し、エネルギー収束による停止条件はない。",
             "4. エネルギー測定を安定させた後に同一精度のFourier除去応答を再評価する。",
             "", "今回実施したのは既存保存標本のオフライン解析。新たなVMC・再サンプリング・ジョブ変更は行っていない。",
             "", "## 実行状態と再現性", "",
             "|条件|最終ログstep|解析完了|", "|---|---:|---|"]
    text += [f'|{r["name"]}|{r["last_step"]}|{"完了" if r["complete"] else "学習中：本解析の評価比較には含めない"}|' for r in runs]
    text += ["", "[squeue.txt](squeue.txt)、[diagnostics.json](diagnostics.json)、[input_manifest.json](input_manifest.json)に状態、詳細診断、読込時点のSHA256を保存。未完了runのhistoryは読込後も伸びるため、そのSHA256は当該スナップショットの記録。完了runの入力ファイルは解析終了時にも不変であることを確認した。",
             "", "再実行時は別の新規出力先を指定する。既存の自動collect報告は変更しない。",
             "", "```bash", "JAX_PLATFORMS=cpu research_framework/.venv/bin/python -m research_framework.analysis.analyze_n36_snapshot \\",
             f"  --root {root} \\", "  --output /path/to/a/new/analysis-directory", "```", ""]
    (out / "REPORT_JA.md").write_text("\n".join(text))
    # Completed artifacts should be immutable; a live history is recorded once.
    complete_names = {r["name"] for r in completed}
    checked=0
    for filename, item in manifest.items():
        path=Path(filename)
        if any(name in path.parts for name in complete_names):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"], filename
            checked+=1
    write_json("input_manifest.json", manifest)
    write_json("validation.json", dict(completed_files_unchanged=checked,
               saved_energy_means_reproduced=len(completed),
               structure_factor_values_reproduced=len(structure),
               candidate_energy_means_reproduced=len(ablations),
               identical_hf_arrays_within_seed=True,
               complete_checkpoint_steps_verified=10000,
               training_history_contiguous=True))
    print(json.dumps(dict(output=str(out), completed=len(completed),
                          rhat_range=[min(r["rhat"] for r in completed),max(r["rhat"] for r in completed)],
                          groups=groups),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
