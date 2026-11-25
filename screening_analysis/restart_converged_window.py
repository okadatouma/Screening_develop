"""Prepare a separate relaxed-protocol study, retaining checkpoint provenance."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .converged_window_study import check_convergence, sampling_passes


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_restart(original, destination, tolerance):
    original=Path(original).resolve();destination=Path(destination).resolve()
    old=json.loads((original/"plan.json").read_text());plan=copy.deepcopy(old)
    if tolerance <= old["energy_tolerance_Ha_per_electron"]:
        raise ValueError("The new tolerance must be greater than the old tolerance")
    destination.mkdir(parents=True,exist_ok=False)
    plan["energy_tolerance_Ha_per_electron"]=tolerance
    plan["revision"]=dict(original_root=str(original),original_tolerance=old["energy_tolerance_Ha_per_electron"],
        created_at=datetime.now(timezone.utc).isoformat(),purpose="exploratory relaxed precision",
        reuse="first stationary candidate under revised tolerance; otherwise latest saved progress",
        independent_learning_rate_challenge_required=True)
    save(destination/"plan.json",plan)
    shutil.copytree(original/"source-production",destination/"source-production",
        ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("converged_window_study.py","restart_converged_window.py"):
        shutil.copy2(Path(__file__).parent/name,destination/"source-production/screening_analysis"/name)
    (destination/"runs").mkdir();(destination/"logs").mkdir()
    audit=[]
    for case in plan["cases"]:
        source=original/"runs"/case["id"];directory=destination/"runs"/case["id"]
        if not source.exists():continue
        directory.mkdir();save(directory/"case.json",case)
        if (source/"reference").exists():shutil.copytree(source/"reference",directory/"reference")
        history=[];measurements=[];selected=[];candidate=False
        for specpath in sorted(source.glob("stage-[0-9][0-9].json")):
            spec=json.loads(specpath.read_text());training=Path(spec["output"]);measurement=Path(spec["measurement_output"])
            summary=measurement/"summary.json"
            if not summary.exists():break
            result=json.loads(summary.read_text())
            result["sampling_pass"]=sampling_passes(result,plan)
            stage=directory/training.name;stage.mkdir()
            for item in training.iterdir():
                if item.name != "physical_summary.json":(stage/item.name).symlink_to(item.resolve())
            output=directory/measurement.name;output.mkdir()
            for item in measurement.iterdir():
                if item.name != "summary.json":(output/item.name).symlink_to(item.resolve())
            result["reuse_provenance"]=dict(summary=str(summary),sha256=sha(summary),
                original_sampling_pass=json.loads(summary.read_text())["sampling_pass"])
            result["checkpoint"]=str(stage/Path(result["checkpoint"]).name)
            save(output/"summary.json",result)
            measurements.append(str((output/"summary.json").relative_to(directory)))
            history.append(result);selected.append(specpath.name)
            candidate=len(history)>=plan["minimum_stage_count"] and check_convergence(history,plan).get("stationary_candidate",False)
            if candidate:break
        partial=None
        if not candidate:
            next_spec=source/f"stage-{len(history):02d}.json"
            if next_spec.exists():
                spec=json.loads(next_spec.read_text());training=Path(spec["output"])
                # Training writes checkpoints directly; copy and decode to reject an in-flight write.
                for checkpoint in sorted(training.glob("qmcjax_ckpt_*.npz"),reverse=True):
                    raw=checkpoint.read_bytes()
                    try:
                        with np.load(io.BytesIO(raw),allow_pickle=True) as z:
                            for key in z.files:z[key]
                            step=int(z["t"])+1
                    except (OSError,ValueError,EOFError,zipfile.BadZipFile):continue
                    if step <= (history[-1]["step"] if history else 0):break
                    stage=directory/training.name;stage.mkdir()
                    copied=stage/checkpoint.name;copied.write_bytes(raw)
                    if (training/"runtime_config.json").exists():shutil.copy2(training/"runtime_config.json",stage/"runtime_config.json")
                    save(stage/"resume_checkpoint.json",dict(checkpoint=copied.name,sha256=sha(copied),
                        source=str(checkpoint),step=step,target=spec["target"]))
                    if step==spec["target"]:save(stage/"stage_complete.json",dict(target=step))
                    partial=dict(step=step,checkpoint=str(copied),sha256=sha(copied));break
        if history or partial:
            save(directory/"resume_state.json",dict(case=case,protocol_sha256=sha(destination/"plan.json"),
                measurements=measurements,previous=Path(history[-1]["checkpoint"]).parent.name if history else None))
        audit.append(dict(case=case["id"],reused_measurements=len(history),
            previous_step=history[-1]["step"] if history else 0,challenge_next=candidate,partial=partial))
    save(destination/"restart_audit.json",dict(original_root=str(original),cases=audit))
    frozen=destination/"source-production"
    save(destination/"production_source_sha256.json",{str(p.relative_to(frozen)):sha(p) for p in frozen.rglob("*") if p.is_file()})
    return audit


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original",type=Path);parser.add_argument("destination",type=Path)
    parser.add_argument("--tolerance",type=float,required=True);args=parser.parse_args()
    print(json.dumps(prepare_restart(args.original,args.destination,args.tolerance),indent=2))
