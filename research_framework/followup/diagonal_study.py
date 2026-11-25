"""Six bounded CPU runs testing the missing low-frequency diagonal feature.

Adapters are local to this study's processes. Existing factories, Hamiltonian,
optimizer, Metropolis kernel, checkpoints, and jobs are never modified.
"""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ..paths import ROOT, lattice_source, wannier_source
from ..cluster import _copy_code
from ..config import validate_config
from ..embeddings.periodic import PeriodicEmbedding
from ..experiments.store import RunStore, json_save, csv_save, digest
from .continuous import MatchedHFTask, initial_evidence, MODES
from screening_analysis.converged_window_study import time_statistics

STEPS = (0, 1000, 3000, 6000, 10000)
VARIANTS = ("baseline_plus_diagonal", "fourier_l1_without_diagonal")


class BudgetExpired(RuntimeError):
    pass


class DiagonalEmbedding(PeriodicEmbedding):
    """Keep the 20x32 projection and six scalar controls in both variants."""
    def __init__(self, config, physics, model, twist=False, variant=None):
        super().__init__(config, physics, model, twist)
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        if variant == VARIANTS[0] and (self.learned or self.high or self.onehot or twist):
            raise ValueError("plus-diagonal requires scalar-spin padded baseline")
        if variant == VARIANTS[1] and (not self.learned or self.modes.tolist() != MODES):
            raise ValueError("disabled-diagonal requires the original six Fourier candidates")

    def dimensions(self, base=4):
        return 7 if self.variant == VARIANTS[0] else super().dimensions(base)

    def init(self, key, base_dim, width):
        weight, extra = super().init(key, base_dim, width)
        if self.variant == VARIANTS[1]:
            extra["fourier_amplitude"] = extra["fourier_amplitude"].at[2].set(0.)
        return weight, extra

    def features(self, base, spins, ae, extra=None):
        extra = extra or {}
        if self.variant == VARIANTS[0]:
            # Preserve all original baseline coordinates, spin row, and gains.
            plain = {k: v for k, v in extra.items() if k != "_feature_mask"}
            original = super().features(base, spins, ae, plain)
            theta = 2*jnp.pi*(ae[:, 0, :]@self.reciprocal.T)
            angle = theta[:, 0]-theta[:, 1]
            added = jnp.stack((jnp.cos(angle), jnp.sin(angle)), -1)
            added = added*extra["feature_gain"][jnp.asarray([5, 0])]
            result = jnp.concatenate((original, added), -1)
            return result*extra.get("_feature_mask", jnp.ones(7))
        result = super().features(base, spins, ae, extra)
        # This mask is a fixed object property, never an optimizer parameter.
        return result*jnp.ones(17).at[8:10].set(0.)

    def penalty(self, extra):
        if self.variant == VARIANTS[1]:
            # Same 1/6 L1 coefficient for every retained amplitude. The disabled
            # coordinate has zero data/penalty gradient and stays exactly zero.
            return self.config["sparse_penalty"]*jnp.mean(
                jnp.abs(extra["fourier_amplitude"])*jnp.ones(6).at[2].set(0.))
        return super().penalty(extra)

    def feature_mask(self, kind):
        if self.variant == VARIANTS[0] and kind == "remove_spin":
            return jnp.ones(7).at[4].set(0.)
        return super().feature_mask(kind)


class DiagonalTask(MatchedHFTask):
    def __init__(self, config, variant="identity"):
        super().__init__(config)
        self.variant = variant

    def build(self, folder):
        if self.variant == "identity":
            return super().build(folder)
        from ..wavefunctions import continuous
        original = continuous.make_embedding
        def factory(config, task, physics, model, twist=False):
            return DiagonalEmbedding(config, physics, model, twist, self.variant)
        continuous.make_embedding = factory
        try:
            super().build(folder)
        finally:
            continuous.make_embedding = original


def now_jst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def clock_start(root):
    path = root/"budget_clock.json"
    try:
        json_save(path, dict(start_unix=time.time(), started_at=now_jst(),
                             collection_cutoff_hours=47.5, hard_limit_hours=48))
    except FileExistsError:
        # The first worker owns this atomic creation; wait only for its tiny write.
        for _ in range(20):
            try:
                return json.loads(path.read_text())["start_unix"]
            except json.JSONDecodeError:
                time.sleep(.05)
        raise
    return json.loads(path.read_text())["start_unix"]


def check_time(deadline):
    if time.time() >= deadline:
        raise BudgetExpired("Declared wall-clock budget reached")


def load_checkpoint(path):
    with path.open("rb") as f:
        value = pickle.load(f)  # trusted, locally created research output only
    return value["step"], jax.tree_util.tree_map(jnp.asarray, value["state"])


def history_rows(path):
    data=path.read_bytes()
    lines=data.splitlines()
    if data and not data.endswith(b"\n"):
        lines=lines[:-1]  # a killed writer's uncommitted last record
    return [json.loads(x) for x in lines if x]


class StudyStore(RunStore):
    def __init__(self, path, config, task, deadline, resume=False):
        super().__init__(path, config, resume=resume)
        self.task, self.deadline = task, deadline

    def log(self, row):
        check_time(self.deadline)
        super().log(row)

    def save_checkpoint(self, step, state):
        check_time(self.deadline)
        super().save_checkpoint(step, state)
        if step in STEPS:
            with np.load(self.path/"inputs.npz") as z:
                x, s = z["probe_positions"], z["probe_spins"]
            self.task.diagnostics(self, f"step_{step:07d}", state["params"], x, s)


def verify(root):
    for filename, expected in json.loads((root/"frozen_sha256.json").read_text()).items():
        if digest(Path(filename).read_bytes()) != expected:
            raise ValueError(f"Frozen input/source changed: {filename}")


def prepare(root, previous):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("inputs", "logs", "jobs", "state", "runs", "analysis", "measurement"):
        (root/folder).mkdir()
    source = root/"source"
    project = source/"Screening_topology_gpu"
    for folder in ("research_framework", "periodicwave", "screening_analysis"):
        _copy_code(ROOT/folder, project/folder)
    _copy_code(lattice_source()/"lattice", source/"Lattice_screening/lattice")
    for folder in ("wannier_ph", "vendor"):
        _copy_code(wannier_source()/folder, source/"wannier_ph"/folder)
    hf = root/"inputs/N36_HF_orbitals.txt"
    shutil.copy2(previous/"inputs/N36_HF_orbitals.txt", hf)
    old_plan = json.loads((previous/"plan.json").read_text())
    plan, reused = [], []
    for seed in (101, 202, 303):
        for variant, old_name in zip(VARIANTS, (f"n36_baseline_l10_s{seed}", f"n36_learned_fourier_l10.01_s{seed}")):
            old = previous/old_name
            if not (old/"FOLLOWUP_COMPLETE.json").exists():
                raise ValueError(f"Reference run incomplete: {old}")
            old_config = next(v["config"] for v in old_plan if v["name"] == old_name)
            config = copy.deepcopy(old_config)
            name = f"{variant}_s{seed}"
            config["experiment"].update(name=name, stage="diagonal-48h")
            config["physics"]["hf_path"] = str(hf)
            config = validate_config(config)
            reference = dict(name=old_name, folder=str(old), config=old_config, variant="identity",
                             label="baseline" if variant == VARIANTS[0] else "fourier_l1")
            reused.append(reference)
            plan.append(dict(index=len(plan), name=name, config=config, variant=variant,
                             reference=reference))
    protocol = dict(created_at=now_jst(), objective="Missing low-frequency diagonal versus learned Fourier selection",
        new_training_cases=6, reused_cases=6, seeds=[101, 202, 303], updates=10000,
        variant_definition={VARIANTS[0]: "Original baseline five coordinates, then gain-scaled cos(theta1-theta2), sin(theta1-theta2); 20 padded rows, six gains",
                            VARIANTS[1]: "Original 17-feature Fourier L1 network; candidate index2 initialized zero and statically masked; its L1 gradient excluded; other L1 coefficients unchanged"},
        fixed_probe_steps=list(STEPS), fixed_probe_count=8,
        measurement=dict(chains_per_start=16, starts=["fresh_uniform_all_plus", "saved_training_walkers"],
                         burn_in=2048, thinning=4, initial_times=2048, max_times=8192, chunk_times=256,
                         energy_rhat_max=1.05, structure_rhat_max=1.1, energy_ess_min=400,
                         energy_sem_Ha_per_electron_max=1e-4, start_equivalence_Ha_per_electron=5e-4,
                         maximum_hours_per_model=4, energy_conversion="stored_energy / (N * rs**2)"),
        resources=dict(cpus_per_worker=4, memory_gib_per_worker=24, parallel_workers=6,
                       worker_time_hours=47, watch_cpus=1, watch_memory_gib=4),
        deadlines=dict(training_hours_from_first_worker=28, analysis_hours=46,
                       cancel_own_workers_and_collect_hours=47.5, watchdog_limit_hours=48),
        interpretation="Exploratory three-seed matched study; unconverged or timed-out measurements are reported, not promoted to physical energy rankings.",
        existing_jobs="untouched; no GPU requested")
    json_save(root/"plan.json", plan);json_save(root/"protocol.json", protocol)
    # Include every reused checkpoint/diagnostic actually consumed by workers.
    inputs = list(source.rglob("*"))+[hf, root/"plan.json", root/"protocol.json"]
    for old in reused:
        d = Path(old["folder"])
        inputs += [d/"config.json", d/"initial_state.json", d/"inputs.npz", d/"history.jsonl", Path(old["config"]["physics"]["hf_path"])]
        checkpoints=[d/"checkpoints"/f"step_{step:07d}.pkl" for step in STEPS]
        if not all(p.is_file() for p in checkpoints):
            raise ValueError(f"Missing reused checkpoint: {d}")
        inputs += checkpoints
    json_save(root/"frozen_sha256.json", {str(p): digest(p.read_bytes()) for p in inputs if p.is_file()})
    python = ROOT/"research_framework/.venv/bin/python"
    q = shlex.quote
    common = f'''set -euo pipefail
cd {q(str(project))}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RF_LATTICE_SOURCE={q(str(source/'Lattice_screening'))}
export RF_WANNIER_SOURCE={q(str(source/'wannier_ph'))}
export PWT_SOURCE={q(str(project))}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
'''
    for action, cpus, memory, hours, array in (("worker",4,24,47,"#SBATCH --array=0-5%6\n"), ("watch",1,4,48,"")):
        script = f'''#!/bin/bash
#SBATCH --job-name=diag48-{action}
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}G
#SBATCH --time={hours}:00:00
#SBATCH --output={root}/logs/{action}-%A_%a.out
#SBATCH --error={root}/logs/{action}-%A_%a.err
{array}{common}
exec {q(str(python))} -u -m research_framework.followup.diagonal_study {action} --root {q(str(root))}
'''
        (root/"jobs"/f"{action}.sbatch").write_text(script)
    (root/"README_JA.md").write_text("""# 対角低波数の追加対照（48時間以内を目標）

新規はbaseline＋(1,-1)とFourier L1の(1,-1)無効化、各3seed・10000更新の6条件。
旧baselineとFourier L1の6条件は再学習せず、同じ独立評価で測り直す。
各workerは新規学習1条件と対応する旧1条件のPH軌跡・出力介入・独立評価を担当する。
新規6workerを各CPU4・24GiBで同時実行。GPUは要求しない。
固定checkpointの独立評価は2種類の開始配置×16鎖、burn-in2048、測定2048〜8192時点。
28時間で学習、46時間で解析を打ち切り、47.5時間に専用watcherがこの配列だけを停止して集計する。
短い学習後summaryのエネルギーと長い独立評価は別ファイル。未収束・未完了は明記する。
自動結果はREPORT_JA.md、summary.csv、figures/。進捗はstate/とlogs/。
投入IDはsubmission.json、計算開始と期限はbudget_clock.json。
""")
    return root


def submit(root):
    verify(root)
    with (root/"submission.lock").open("x") as f:
        f.write(now_jst()+"\n")
    jobs = {}
    for action in ("worker", "watch"):
        command = ["sbatch", "--parsable", str(root/"jobs"/f"{action}.sbatch")]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch response: {result.stdout}")
        jobs[action] = job
        json_save(root/f"submitted_{action}.json", dict(job_id=job, command=command))
        json_save(root/"submission.json", jobs, replace=True)
    return jobs


def pilot_gate(root, case, store, task):
    reference = Path(case["reference"]["folder"])
    evidence = initial_evidence(task, case["config"]["experiment"]["seed"])
    old = json.loads((reference/"initial_state.json").read_text())
    for k in ("phase", "logabs"):
        if not np.allclose(evidence[k], old[k], atol=1e-9, rtol=1e-10):
            raise ValueError(f"Initial HF mismatch: {k}")
    if evidence["parameter_count"] != old["parameter_count"]:
        raise ValueError("Nominal parameter counts differ")
    new_steps = [json.loads(x) for x in (store.path/"history.jsonl").read_text().splitlines()]
    old_first = json.loads((reference/"history.jsonl").open().readline())
    if not np.isclose(new_steps[0]["energy"], old_first["energy"], atol=1e-8, rtol=1e-10):
        raise ValueError("Matched first local energies differ")
    _, new = load_checkpoint(store.path/"checkpoints/step_0000000.pkl")
    _, previous = load_checkpoint(reference/"checkpoints/step_0000000.pkl")
    for name in ("positions", "spins"):
        if not np.array_equal(np.asarray(getattr(new["walkers"],name)), np.asarray(getattr(previous["walkers"],name))):
            raise ValueError(f"Initial burned walkers differ: {name}")
    json_save(store.path/"PILOT_GATE.json", dict(passed=True, initial=evidence,
        matched_first_energy=new_steps[0]["energy"], finite_updates=len(new_steps),
        steady_update_seconds=new_steps[-1]["update_seconds"]))


def measurement_statistics(energy_groups, sq_groups, protocol, scale):
    """Time and chain axes stay intact; two starts are separate groups."""
    energies = np.concatenate(energy_groups, axis=1)
    energies_Ha = energies/scale
    e = time_statistics(energies_Ha.real)
    imag = time_statistics(energies_Ha.imag)
    starts = [time_statistics(x.real/scale) for x in energy_groups]
    sf = np.concatenate(sq_groups, axis=1)
    structure = [time_statistics(sf[..., i]) for i in range(sf.shape[-1])]
    bound = abs(starts[0]["mean"]-starts[1]["mean"])+1.96*np.hypot(starts[0]["sem"], starts[1]["sem"])
    passed = (len(energies) >= protocol["initial_times"] and
        max(e["rhat"], *[x["rhat"] for x in starts]) <= protocol["energy_rhat_max"] and
        e["ess"] >= protocol["energy_ess_min"] and
        e["sem"] <= protocol["energy_sem_Ha_per_electron_max"] and
        max(x["rhat"] for x in structure) <= protocol["structure_rhat_max"] and
        bound <= protocol["start_equivalence_Ha_per_electron"] and
        abs(imag["mean"]) <= 3*imag["sem"]+protocol["energy_sem_Ha_per_electron_max"])
    return dict(energy_Ha_per_electron=e, energy_stored_mean=float(energies.real.mean()),
                imaginary_Ha_per_electron=imag, starts=starts, structure_factor=structure,
                start_equivalence_bound_Ha_per_electron=float(bound), sampling_pass=bool(passed),
                time_samples=len(energies), total_chains=energies.shape[1],
                ess_limitation="Within-chain estimate; interpret only with Rhat/start diagnostics")


def measure(root, model, task, state, deadline):
    target = root/"measurement"/model["name"]
    target.mkdir(exist_ok=False)
    m = json.loads((root/"protocol.json").read_text())["measurement"]
    deadline = min(deadline, time.time()+3600*m["maximum_hours_per_model"])
    check_time(deadline)
    params = state["params"]
    seed = model["config"]["experiment"]["seed"]+70000
    key = jax.random.PRNGKey(seed)
    key, fresh_key = jax.random.split(key)
    count = m["chains_per_start"]
    fresh = task.initial_walkers(fresh_key, count)
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(state["walkers"].spins), size=count, replace=False)
    trained = jax.tree_util.tree_map(lambda x:x[subset], state["walkers"])
    starts = [fresh, trained]
    keys = list(jax.random.split(key, 2))
    for i in (0, 1):
        starts[i], keys[i], _ = task.sample(params, starts[i], keys[i], m["burn_in"])
    energy, sf, elapsed = [[], []], [[], []], []
    chunk_e, chunk_x, chunk_s, chunk_sf = [[], []], [[], []], [[], []], [[], []]
    reciprocal = np.linalg.inv(model["config"]["physics"]["lattice"]).T
    n = model["config"]["physics"]["n"]
    scale = n*model["config"]["physics"]["rs"]**2
    status, stats, t = "unconverged_max_samples", None, 0
    def save_chunk(t):
        if not chunk_e[0]:
            return
        arrays={}
        for i in (0,1):
            for name,values in (("local_energy",chunk_e),("positions",chunk_x),
                                ("spins",chunk_s),("structure_factor",chunk_sf)):
                arrays[f"{name}_start{i}"]=np.asarray(values[i])
        np.savez_compressed(target/f"samples_through_{t:05d}.npz",**arrays,axes="time,chain,...; separate start0/start1")
        for values in (chunk_e,chunk_x,chunk_s,chunk_sf):
            for group in values:group.clear()
    try:
        for t in range(1, m["max_times"]+1):
            check_time(deadline)
            before = time.monotonic()
            for i in (0, 1):
                starts[i], keys[i], accept = task.sample(params, starts[i], keys[i], m["thinning"])
                data = starts[i]
                en = np.asarray(task.energy_batch(params, data.positions, data.spins))
                if not np.isfinite(en).all():
                    raise FloatingPointError("Nonfinite fixed-state energy; no samples discarded")
                x, s = np.asarray(data.positions), np.asarray(data.spins)
                theta = 2*np.pi*np.einsum("bnd,qd->bnq", x.reshape(count,n,2)@reciprocal, np.asarray(MODES))
                sq = abs(np.exp(1j*theta).sum(1))**2/n
                energy[i].append(en);sf[i].append(sq)
                for dest,value in ((chunk_e,en),(chunk_x,x),(chunk_s,s),(chunk_sf,sq)):
                    dest[i].append(value)
            elapsed.append(time.monotonic()-before)
            if t % m["chunk_times"] == 0:
                save_chunk(t)
                json_save(target/"progress.json",dict(time_samples=t,updated_at=now_jst(),
                    seconds_per_time=float(np.mean(elapsed[-128:])),deadline_unix=deadline),replace=True)
            if t in (m["initial_times"], 2*m["initial_times"], m["max_times"]):
                stats = measurement_statistics([np.asarray(x) for x in energy], [np.asarray(x) for x in sf], m, scale)
                json_save(target/f"diagnostics_{t:05d}.json", stats)
                print("measurement",model["name"],t,stats["sampling_pass"],flush=True)
                if stats["sampling_pass"]:
                    status = "passed_sampling_protocol"
                    break
    except BudgetExpired:
        status = "unconverged_time_budget"
    except Exception:
        status = "failed_measurement"
        raise
    finally:
        actual_times = min(map(len,energy))
        save_chunk(actual_times)
        if actual_times >= 4:
            stats = measurement_statistics([np.asarray(x[:actual_times]) for x in energy], [np.asarray(x[:actual_times]) for x in sf], m, scale)
            if status != "passed_sampling_protocol":
                stats["sampling_pass"]=False
        json_save(target/"summary.json", dict(status=status, model=model["name"],
            protocol=m, actual_times=actual_times, diagnostics=stats, starts=m["starts"],
            valid_times_by_start=list(map(len,energy)),
            elapsed_time_sample_seconds=float(sum(elapsed)), updated_at=now_jst()))
    return stats


def response(a, b):
    phase = np.angle(np.exp(1j*(b[0]-a[0])))
    phase -= np.angle(np.mean(np.exp(1j*phase)))
    phase = np.angle(np.exp(1j*phase))
    return dict(logabs_centered_rms=float(np.std(b[1]-a[1])),
                phase_centered_rms=float(np.sqrt(np.mean(phase**2))))


def analyze(root, model, task, deadline):
    from ripser import ripser
    from screening_analysis.topology.persistence import symmetrize, rank_weights
    folder = Path(model["folder"])
    output = root/"analysis"/model["name"]
    output.mkdir(exist_ok=False);(output/"diagnostics").mkdir()
    with np.load(folder/"inputs.npz") as z:
        x, s = z["probe_positions"], z["probe_spins"]
    phrows, diagrams = [], {}
    for step in STEPS:
        check_time(deadline)
        checkpoint = folder/"checkpoints"/f"step_{step:07d}.pkl"
        if not checkpoint.exists():
            continue
        _, state = load_checkpoint(checkpoint)
        label = f"step_{step:07d}"
        path = folder/"diagnostics"/(label+"_internal.npz")
        if not path.exists():
            task.diagnostics(SimpleNamespace(path=output), label, state["params"], x, s)
            path = output/"diagnostics"/(label+"_internal.npz")
        with np.load(path) as a:
            graphs = dict(QK=(np.clip(a["layer0_similarity"],-1,1)+1)/2,
                          Screening=a["layer0_aggregation_weights"], Window=a["layer0_window"])
            f=a["features"];f=f/np.maximum(np.linalg.norm(f,axis=-1,keepdims=True),1e-12)
            graphs["FeatureVR"]=(1-np.sqrt(np.maximum(0,2-2*np.clip(np.einsum("bid,bjd->bij",f,f),-1,1)))/2)[:,None]
        for graph, values in graphs.items():
            for variant in ("raw","rank"):
                metrics=[]
                for ci, conf in enumerate(values):
                    for hi, w in enumerate(conf):
                        w=symmetrize(np.clip(w,0,1))
                        if variant=="rank":w=rank_weights(w)
                        distance=1-w;distance[w==0]=np.inf;np.fill_diagonal(distance,0)
                        dg=ripser(distance,distance_matrix=True,maxdim=1,thresh=1.,coeff=2)["dgms"]
                        d=dg[1];finite=np.isfinite(d[:,1]);life=d[finite,1]-d[finite,0]
                        metrics.append((float(life.sum()+np.sum(1-d[~finite,0])),int(np.isinf(dg[0][:,1]).sum())))
                        for di in (0,1):diagrams[f"{label}_{graph}_{variant}_c{ci}h{hi}_H{di}"]=dg[di]
                phrows.append(dict(name=model["name"],label=model["label"],seed=model["config"]["experiment"]["seed"],
                                   step=step,graph=graph,variant=variant,H1_area_0_1=float(np.mean([v[0] for v in metrics])),
                                   H0_components=float(np.mean([v[1] for v in metrics]))))
    csv_save(output/"ph_trajectory.csv", phrows)
    np.savez_compressed(output/"persistence_diagrams.npz",**diagrams)
    final = folder/"checkpoints/step_0010000.pkl"
    actions=[]
    if final.exists():
        check_time(deadline)
        _, state = load_checkpoint(final);params=state["params"]
        evaluate=jax.jit(jax.vmap(task.wave.apply,in_axes=(None,0,0)))
        initial=tuple(map(np.asarray,evaluate(params,jnp.asarray(x),jnp.asarray(s))))
        # Fixed-input causal interventions, with global amplitude/phase removed.
        heads=copy.deepcopy(params);ap=heads["layers"]["attention"][0]
        ap["attn_output"]=jnp.zeros_like(ap["attn_output"])
        value=tuple(map(np.asarray,evaluate(heads,jnp.asarray(x),jnp.asarray(s))))
        actions.append(dict(intervention="remove_all_heads",**response(initial,value)))
        if model["variant"]==VARIANTS[0]:
            masks=[("remove_diagonal",jnp.ones(7).at[5:7].set(0.))]
        elif task.wave.embedding.learned:
            masks=[(f"remove_fourier_{i}",jnp.ones(17).at[4+2*i:6+2*i].set(0.)) for i in range(6)]
        else:masks=[]
        for name,mask in masks:
            pp=copy.deepcopy(params);pp["layers"]["embedding_extra"]["_feature_mask"]=mask
            value=tuple(map(np.asarray,evaluate(pp,jnp.asarray(x),jnp.asarray(s))))
            actions.append(dict(intervention=name,**response(initial,value)))
    json_save(output/"functional_response.json",dict(name=model["name"],responses=actions,
        note="Eight matched uniform probes, not Born-weighted fidelity/energy."))
    return phrows


def worker(root, index):
    jax.config.update("jax_enable_x64",True)
    verify(root)
    started=clock_start(root)
    case=json.loads((root/"plan.json").read_text())[index]
    statefile=root/"state"/f"worker-{index}.json"
    result=dict(index=index,name=case["name"],started_at=now_jst(),status="training")
    json_save(statefile,result,replace=True)
    try:
        check_time(started+28*3600)
        task=DiagonalTask(case["config"],case["variant"])
        store=StudyStore(root/"runs"/case["name"],case["config"],task,started+28*3600)
        json_save(store.path/"study_case.json",case)
        task.run(store,until=2)
        pilot_gate(root,case,store,task)
        result.update(status="training_gate_passed",gate_passed_at=now_jst())
        json_save(statefile,result,replace=True)
        task.run(store,resume=True)
        json_save(store.path/"STUDY_TRAIN_COMPLETE.json",dict(steps=10000,completed_at=now_jst()))
        result.update(status="analysis_and_measurement",training_completed_at=now_jst())
        json_save(statefile,result,replace=True)
        models=[dict(name=case["name"],label=case["variant"],folder=str(store.path),
                     config=case["config"],variant=case["variant"]),case["reference"]]
        result["models"]=[]
        for model in models:
            check_time(started+46*3600)
            mt=task if model["variant"]==case["variant"] else DiagonalTask(model["config"],"identity")
            if mt is not task:mt.build(root/"analysis")
            analyze(root,model,mt,started+46*3600)
            _,checkpoint=load_checkpoint(Path(model["folder"])/"checkpoints/step_0010000.pkl")
            measured=measure(root,model,mt,checkpoint,started+46*3600)
            result["models"].append(dict(name=model["name"],sampling_pass=bool(measured and measured["sampling_pass"])))
            json_save(statefile,result,replace=True)
        result.update(status="completed",completed_at=now_jst())
        json_save(statefile,result,replace=True)
    except Exception as error:
        result.update(status="time_budget" if isinstance(error,BudgetExpired) else "failed",
                      error=repr(error),traceback=traceback.format_exc(),ended_at=now_jst())
        json_save(statefile,result,replace=True)
        raise


def report(root, reason="all_workers_finished"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan=json.loads((root/"plan.json").read_text());states=[];rows=[];phrows=[];functions=[]
    for case in plan:
        path=root/"state"/f'worker-{case["index"]}.json'
        states.append(json.loads(path.read_text()) if path.exists() else dict(index=case["index"],name=case["name"],status="not_started"))
        models=[dict(name=case["name"],label=case["variant"],folder=str(root/"runs"/case["name"]),config=case["config"]),case["reference"]]
        for model in models:
            folder=Path(model["folder"]);history=folder/"history.jsonl"
            rs=history_rows(history) if history.exists() else []
            mpath=root/"measurement"/model["name"]/"summary.json"
            measurement=json.loads(mpath.read_text()) if mpath.exists() else {}
            diagnostics=measurement.get("diagnostics") or {}
            e=diagnostics.get("energy_Ha_per_electron",{})
            rows.append(dict(name=model["name"],label=model["label"],seed=model["config"]["experiment"]["seed"],
                step=rs[-1]["step"] if rs else 0,
                train_last1000=float(np.mean([v["energy"] for v in rs[-1000:]])) if rs else None,
                measurement_status=measurement.get("status","not_completed"),
                energy_Ha_per_electron=e.get("mean"),energy_sem=e.get("sem"),rhat=e.get("rhat"),
                sampling_pass=bool(diagnostics.get("sampling_pass",False))))
            path=root/"analysis"/model["name"]/"ph_trajectory.csv"
            if path.exists():
                import csv
                phrows.extend(list(csv.DictReader(path.open())))
            path=root/"analysis"/model["name"]/"functional_response.json"
            if path.exists():functions.append(json.loads(path.read_text()))
    csv_save(root/"summary.csv",rows,replace=True)
    paired=[]
    for case in plan:
        new=next(r for r in rows if r["name"]==case["name"])
        old=next(r for r in rows if r["name"]==case["reference"]["name"])
        qualified=new["sampling_pass"] and old["sampling_pass"]
        paired.append(dict(variant=case["variant"],seed=case["config"]["experiment"]["seed"],
            new_name=new["name"],reference_name=old["name"],
            training_delta=(new["train_last1000"]-old["train_last1000"]) if new["train_last1000"] is not None else None,
            both_sampling_pass=qualified,
            fixed_energy_delta_Ha_per_electron=(new["energy_Ha_per_electron"]-old["energy_Ha_per_electron"]) if qualified else None,
            independent_combined_sem=float(np.hypot(new["energy_sem"],old["energy_sem"])) if qualified else None))
    csv_save(root/"paired_contrasts.csv",paired,replace=True)
    if phrows:csv_save(root/"ph_trajectory.csv",phrows,replace=True)
    json_save(root/"results.json",dict(updated_at=now_jst(),reason=reason,workers=states,models=rows,
              paired_contrasts=paired,functional_response=functions),replace=True)
    figures=root/"figures";figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    labels=["baseline",VARIANTS[0],"fourier_l1",VARIANTS[1]]
    for ax,label in zip(axes.flat,labels):
        for row in [v for v in rows if v["label"]==label]:
            c=next(c for c in plan if c["config"]["experiment"]["seed"]==row["seed"] and (c["variant"]==label or c["reference"]["label"]==label))
            d=Path(c["reference"]["folder"]) if label in ("baseline","fourier_l1") else root/"runs"/c["name"]
            if not (d/"history.jsonl").exists():continue
            history=history_rows(d/"history.jsonl")
            count=len(history)//100
            if count:ax.plot(np.arange(100,count*100+1,100),np.array([v["energy"] for v in history[:count*100]]).reshape(count,100).mean(1),label=str(row["seed"]))
        ax.set(title=label,xlabel="Update",ylabel="Training energy, stored units");ax.legend()
    fig.tight_layout();fig.savefig(figures/"training.png",dpi=150);plt.close(fig)
    if phrows:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,graph in zip(axes,("QK","Screening")):
            for label in labels:
                for seed in (101,202,303):
                    vs=[v for v in phrows if v["label"]==label and int(v["seed"])==seed and v["graph"]==graph and v["variant"]=="rank"]
                    if vs:ax.plot([int(v["step"]) for v in vs],[float(v["H1_area_0_1"]) for v in vs],".-",label=label if seed==101 else None)
            ax.set(title=graph,xlabel="Update",ylabel="Rank graph integrated beta1");ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(figures/"ph_trajectory.png",dpi=150);plt.close(fig)
    text=["# 対角低波数の追加対照：自動結果", "",f"更新 {now_jst()}。終了理由：{reason}。", "",
          "新規6条件＋既存6条件の比較。測定のsampling_passが偽のエネルギーを物理的な順位に使わない。PH・介入は同じ固定8配置上の関数構造として解釈する。", "",
          "|条件|seed|更新数|学習末尾E（内部単位）|独立評価E（Ha/電子）|Rhat|測定状態|", "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        fmt=lambda x:"—" if x is None else f"{x:.6g}"
        text.append(f'|{r["label"]}|{r["seed"]}|{r["step"]}|{fmt(r["train_last1000"])}|{fmt(r["energy_Ha_per_electron"])}|{fmt(r["rhat"])}|{r["measurement_status"]}|')
    text += ["", "同一seedの対応比較：差は新条件−旧条件。独立評価の差は両方が測定基準を通過した場合だけ表示する。", "",
             "|新条件|seed|学習エネルギー差|独立評価差（Ha/電子）|両測定通過|", "|---|---:|---:|---:|---|"]
    for p in paired:
        fmt=lambda x:"—" if x is None else f"{x:.6g}"
        text.append(f'|{p["variant"]}|{p["seed"]}|{fmt(p["training_delta"])}|{fmt(p["fixed_energy_delta_Ha_per_electron"])}|{p["both_sampling_pass"]}|')
    text += ["", "baselineへの対角成分追加で差が再現するか、学習Fourierから対角成分を除くと他の特徴が補えるかを、この対応比較で評価する。3 seedの探索的実験なので、通常の5%有意差や物理相の認定とは扱わない。",
             "","![学習曲線](figures/training.png)","","PHは[ph_trajectory.csv](ph_trajectory.csv)、個別介入は[results.json](results.json)、対応差は[paired_contrasts.csv](paired_contrasts.csv)。",
             "", "worker状態："]+[f'- {s["name"]}: {s["status"]}' for s in states]
    if phrows:text += ["","![PH軌跡](figures/ph_trajectory.png)"]
    (root/"REPORT_JA.md").write_text("\n".join(text)+"\n")
    return root/"REPORT_JA.md"


def watch(root):
    jobs=json.loads((root/"submission.json").read_text())
    worker_id=jobs["worker"]
    while True:
        # Only this newly created array may be cancelled by the budget watcher.
        status=subprocess.run(["squeue","-h","-j",worker_id,"-o","%i %T"],capture_output=True,text=True)
        path=root/"budget_clock.json"
        start=json.loads(path.read_text())["start_unix"] if path.exists() else None
        json_save(root/"monitor.json",dict(updated_at=now_jst(),worker_array_id=worker_id,
            queue=status.stdout,elapsed_hours=(time.time()-start)/3600 if start else None),replace=True)
        if not status.stdout.strip():
            finished = status.returncode == 0
            if not finished:
                account=subprocess.run(["sacct","-X","-n","-j",worker_id,"--format=State","-P"],capture_output=True,text=True)
                states=[x.split("|")[0].strip().split()[0] for x in account.stdout.splitlines() if x.strip()]
                terminal=("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","BOOT_FAIL","PREEMPTED","DEADLINE")
                finished=bool(states) and all(any(s.startswith(t) for t in terminal) for s in states)
            if finished:
                print(report(root),flush=True);return
        if start and time.time()-start >= 47.5*3600:
            subprocess.run(["scancel",worker_id],check=True)
            json_save(root/"budget_stop.json",dict(stopped_at=now_jst(),cancelled_array=worker_id))
            time.sleep(5)
            print(report(root,"47.5_hour_budget_stop"),flush=True);return
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("command",choices=("prepare","submit","worker","watch","report"))
    p.add_argument("--root",type=Path,required=True);p.add_argument("--previous",type=Path)
    p.add_argument("--index",type=int);args=p.parse_args();root=args.root.resolve()
    if args.command=="prepare":print(prepare(root,args.previous))
    elif args.command=="submit":print(json.dumps(submit(root)))
    elif args.command=="worker":worker(root,args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.command=="watch":watch(root)
    else:print(report(root))


if __name__=="__main__":main()
