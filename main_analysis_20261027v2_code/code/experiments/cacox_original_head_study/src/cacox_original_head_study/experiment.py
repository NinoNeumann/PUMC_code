"""Approved fixed head-comparison protocol with verified per-fit resumption."""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Bundle
from .training import fit
from .checkpoint import save_checkpoint, load_checkpoint
from .evaluation import normalize, assert_complete, assert_paired
from .metrics_core import harrell
from .runtime import stamp, sha
from .storage import write_json, mark_complete, verify_completed, run_lock
from .provenance import verify_inputs, references, freeze_manifest


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def role_path(study, role):
    return study.out / ("controls" if role == "A0_cox_efron" else "candidates") / role


class FoldData:
    def __init__(self, study, fold):
        self.fold = fold
        self.bundle = Bundle(study.bundle_path(fold))
        self.variables = self.bundle.variables
        self.cache = {}

    def pair(self, j, inner=None):
        key = (j, inner)
        if key not in self.cache:
            self.cache[key] = self.bundle.pair(j, inner)
        return self.cache[key]


def predictions(study, fitted, frame, role, fold, j):
    score, survival = fitted.predict(frame)
    q = frame[["patient_id", "Death", "Survival_Days"]].copy()
    q["repeat_id"], q["fold"], q["imputation"], q["score"] = 1, fold, j, score
    for column, values in zip([f"survival_{t}d" for t in study.config["common"]["times"]], survival.T):
        q[column] = values
    definition = "Cox log relative risk" if fitted.spec["kind"] == "cox" else "negative RMST to 1095 days"
    return normalize(q, role, study.config["study_id"], definition)


def evaluate_inner(study, data, role, spec, indices, phase):
    identity = spec.get("candidate_id", spec.get("model_id"))
    out = role_path(study, role) / "inner" / identity / f"fold{data.fold:02d}" / phase
    out.mkdir(parents=True, exist_ok=True)
    if verify_completed(out):
        return pd.read_csv(out / "metrics.csv")
    config = {"role": role, "spec": spec, "common": study.config["common"],
              "variables": data.variables, "imputations": list(indices), "phase": phase}
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Inner fit contract changed")
    write_json(path, config)
    start, clock = stamp(), time.monotonic()
    records = pd.read_csv(out / "metrics.csv").to_dict("records") if (out / "metrics.csv").exists() else []
    done = {(int(r["imputation"]), int(r["inner_fold"])) for r in records}
    outputs = ["config.json", "metrics.csv", "summary.json"]
    for j in indices:
        for k in range(1, 6):
            fit_dir = out / f"fits/mi{j:02d}_inner{k:02d}"
            files = ["result.json", "diagnostics.json", "training_trace.csv", "validation_predictions.csv"]
            outputs.extend(str((fit_dir / name).relative_to(out)) for name in files)
            if (j, k) in done:
                if not verify_completed(fit_dir):
                    raise ValueError("Recorded fit lacks verified completion")
                continue
            if verify_completed(fit_dir):
                result = json.loads((fit_dir / "result.json").read_text())
            else:
                development, validation = data.pair(j, k)
                if (set(development.patient_id) | set(validation.patient_id)) & set(data.bundle.raw_assessment.patient_id):
                    raise ValueError("Outer test patients entered inner fitting")
                seed = study.config["common"]["seed_base"] + (data.fold-1) * 1000 + j * 10000 + k * 100
                context = {"status": "running", "stage": "inner_selection", "role": role,
                           "candidate": identity, "fold": data.fold, "imputation": j, "inner_fold": k,
                           "completed_cells": len(records), "total_cells": len(indices)*5}
                study.status(**context)
                study.record("inner_fit_started", role=role, candidate=identity, fold=data.fold, imputation=j, inner_fold=k)
                fit_clock = time.monotonic()
                fitted, trace, diagnostic = fit(development, data.variables, spec, study.config["common"], seed,
                                                validation=validation,
                                                progress=lambda record: study.status(**context, epoch_progress=record))
                q = predictions(study, fitted, validation, role, data.fold, j)
                c = harrell(validation.Survival_Days, validation.Death, q.score)["estimate"]
                result = {"fold": data.fold, "imputation": j, "inner_fold": k, "candidate_id": identity,
                          "C_index": c, "best_epoch": diagnostic["best_epoch"], "stopped_epoch": diagnostic["stopped_epoch"],
                          "parameters": diagnostic["parameters"], "feature_parameters": diagnostic["feature_parameters"],
                          "head_parameters": diagnostic["head_parameters"], "seed": seed,
                          "initial_feature_sha256": diagnostic["initial_feature_sha256"],
                          "elapsed_seconds": time.monotonic()-fit_clock,
                          "cap_hit": diagnostic["stopped_epoch"] == study.config["common"]["max_epochs"]}
                write_json(fit_dir / "diagnostics.json", diagnostic)
                write_csv(trace, fit_dir / "training_trace.csv")
                write_csv(q, fit_dir / "validation_predictions.csv")
                write_json(fit_dir / "result.json", result)
                mark_complete(fit_dir, files)
                study.record("inner_fit_complete", **result, role=role)
                del fitted
            records.append(result)
            done.add((j,k))
            write_csv(pd.DataFrame(records).sort_values(["imputation","inner_fold"]), out / "metrics.csv")
    frame = pd.DataFrame(records)
    if len(frame) != len(indices)*5 or frame[["imputation","inner_fold"]].duplicated().any():
        raise ValueError("Incomplete inner fit group")
    summary = {"started": start, "completed_at": stamp(), "elapsed_seconds": time.monotonic()-clock,
               "fits": len(frame), "C_index_mean": float(frame.C_index.mean()),
               "C_index_range": [float(frame.C_index.min()),float(frame.C_index.max())]}
    write_json(out / "summary.json", summary)
    mark_complete(out, outputs)
    study.log_experiment(f"{role} / {identity} / 第{data.fold}折 / {phase}", start,
                         "固定所有超参数，仅在内部验证确定训练轮数", spec, out,
                         f"完成{len(frame)}个内部拟合，C-index均值{frame.C_index.mean():.5f}；外层未评价")
    print(f"{stamp()} {role} fold{data.fold} {identity} {phase} C={frame.C_index.mean():.5f}", flush=True)
    return frame


def inner_epochs(study, role, spec, fold):
    identity=spec.get("candidate_id",spec.get("model_id"))
    directory=role_path(study,role)/"inner"/identity/f"fold{fold:02d}"
    phases=["full_m20"]
    q=pd.concat([pd.read_csv(directory/phase/"metrics.csv") for phase in phases],ignore_index=True)
    if len(q)!=100 or q[["imputation","inner_fold"]].duplicated().any():
        raise ValueError("Final refit requires full frozen inner epochs")
    return q


def outer_fold(study, data, role, spec):
    inner=inner_epochs(study,role,spec,data.fold)
    out=role_path(study,role)/"outer"/f"fold{data.fold:02d}"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; outputs=["predictions_all20.csv","fold_metadata.json"]
    start=stamp()
    for j in range(1,21):
        model_out=out/f"imputation{j:02d}"
        files=["model.pt","predictions.csv","training_trace.csv","metadata.json"]
        outputs.extend(str((model_out/name).relative_to(out)) for name in files)
        if verify_completed(model_out):
            q=pd.read_csv(model_out/"predictions.csv",float_precision="round_trip")
        else:
            five=inner[inner.imputation.eq(j)]
            if len(five)!=5: raise ValueError("Missing five inner epoch choices")
            epochs=max(1,int(np.ceil(np.median(five.best_epoch))))
            development,assessment=data.pair(j)
            seed=study.config["common"]["seed_base"]+(data.fold-1)*1000+j
            context={"status":"running","stage":"outer_refit","role":role,"fold":data.fold,"imputation":j,"epochs":epochs}
            study.status(**context)
            clock=time.monotonic()
            fitted,trace,diagnostic=fit(development,data.variables,spec,study.config["common"],seed,
                                       fixed_epochs=epochs,progress=lambda row:study.status(**context,epoch_progress=row))
            q=predictions(study,fitted,assessment,role,data.fold,j)
            audit={"fold":data.fold,"imputation":j,"role":role,"spec":spec,"inner_best_epochs":five.best_epoch.tolist(),
                   "refit_epochs":epochs,"elapsed_seconds":time.monotonic()-clock,"variables":data.variables,
                   "bundle_sha256":sha(data.bundle.directory/"bundle.json"),**diagnostic}
            save_checkpoint(model_out/"model.pt",fitted,audit)
            write_csv(q,model_out/"predictions.csv")
            write_csv(trace,model_out/"training_trace.csv")
            write_json(model_out/"metadata.json",audit)
            mark_complete(model_out,files)
            study.record("outer_model_complete",role=role,fold=data.fold,imputation=j,refit_epochs=epochs,
                         elapsed_seconds=audit["elapsed_seconds"])
            del fitted
        rows.append(q)
    result=pd.concat(rows,ignore_index=True)
    if len(result)!=228*20: raise ValueError("Incomplete outer fold")
    write_csv(result,out/"predictions_all20.csv")
    write_json(out/"fold_metadata.json",{"role":role,"fold":data.fold,"models":20,"spec":spec,"variables":data.variables})
    mark_complete(out,outputs)
    study.log_experiment(f"{role} 第{data.fold}折外层评测",start,"评价已冻结预测头",spec,out,
                         "20个模型及预测均保存，测试患者不用于选型或早停")
    print(f"{stamp()} {role} outer fold{data.fold} complete",flush=True)
    return result


def smoke(study):
    out=study.out/"validation/smoke"
    out.mkdir(parents=True,exist_ok=True)
    data=FoldData(study,1)
    development,validation=data.pair(1,1)
    specs=[study.config["control"],study.config["candidate"]]
    records=[]
    for spec in specs:
        identity=spec.get("candidate_id",spec.get("model_id"))
        clock=time.monotonic()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        seed=study.config["common"]["seed_base"]+10000+100
        fitted,trace,audit=fit(development,data.variables,spec,study.config["common"],seed,validation=validation)
        expected=fitted.predict(validation)
        save_checkpoint(out/f"{identity}.pt",fitted,audit)
        restored,_=load_checkpoint(out/f"{identity}.pt",study.config["common"]["device"])
        for a,b in zip(expected,restored.predict(validation)):
            np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-7)
        write_csv(trace,out/f"{identity}_trace.csv")
        records.append({"identity":identity,"elapsed_seconds":time.monotonic()-clock,
                        "peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                        "validation_C_index":harrell(validation.Survival_Days,validation.Death,expected[0])["estimate"],**audit})
    if len({r["initial_feature_sha256"] for r in records})!=1:
        raise AssertionError("Common feature initialization differs between arms")
    report={"status":"passed","at":stamp(),"outer_evaluated":False,"trials":records,
            "config_sha256":sha(study.config_path),"source_sha256":study.code_hashes()}
    write_json(study.out/"validation/SMOKE.json",report)
    study.record("smoke_passed",trials=len(records))
    return report


def run(study):
    inputs=verify_inputs(study)
    for name in ["VALIDATION.json","SMOKE.json"]:
        proof=json.loads((study.out/"validation"/name).read_text())
        if (proof["status"]!="passed" or proof.get("source_sha256",proof.get("code_sha256"))!=study.code_hashes()
                or proof["config_sha256"]!=sha(study.config_path)):
            raise ValueError(f"Stale or failed gate: {name}")
        if name=="VALIDATION.json" and proof["input_fingerprints"]!=inputs:raise ValueError("Inputs changed")
    freeze_manifest(study,inputs)
    role=study.config["candidate"]["model_id"];spec=study.config["candidate"]
    with run_lock(study.out):
        started=stamp();study.record("run_started",pid=os.getpid())
        try:
            for fold in range(1,6):
                data=FoldData(study,fold)
                evaluate_inner(study,data,role,spec,range(1,21),"full_m20")
                del data
            write_json(study.out/"ALL_EPOCHS_FROZEN.json",dict(at=stamp(),scope="all500 inner fits completed before new outer predictions",spec=spec,config_sha256=sha(study.config_path)))
            frames=references(study);rows=[]
            for fold in range(1,6):
                data=FoldData(study,fold);rows.append(outer_fold(study,data,role,spec));del data
            frame=pd.concat(rows,ignore_index=True);assert_complete(frame);assert_paired(frames["O0_original_cox"],frame)
            write_csv(frame,role_path(study,role)/"predictions_all20.csv");frames[role]=frame
            study.status(status="running",stage="paired_summary")
            from .reporting import report
            summary=report(study,frames)
            study.status(status="complete",started=started,completed_at=stamp(),completed_folds=5,skipped_folds=0,
                         new_inner_fits=500,new_outer_models=100,reused_original_models=100,summary=summary)
            study.record("run_complete",target_met=summary["target_met"],head_gain=summary["delta_vs_original"])
            print(json.dumps(summary,ensure_ascii=False),flush=True)
            return summary
        except Exception as error:
            import traceback
            study.status(status="failed",started=started,error=repr(error),traceback=traceback.format_exc())
            study.record("run_failed",error=repr(error));raise


