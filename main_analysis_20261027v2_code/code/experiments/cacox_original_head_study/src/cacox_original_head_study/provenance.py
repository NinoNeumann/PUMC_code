"""Pin original architecture, original checkpoints and identical upstream inputs."""
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .runtime import sha, stamp
from .storage import verify_completed, write_json, lock_manifest
from .data import Bundle
from .schema29 import CANDIDATES
from .evaluation import normalize, assert_complete, assert_paired, evaluate


def original_fold(study,fold):
    return study.path("reference_original")/f"gpu_runs/repeat01_fold{fold:02d}_attempt01/repeat01_fold{fold:02d}/main"


def verify_inputs(study):
    cfg=study.config;contract=json.loads(study.path("source_contract").read_text());inputs={}
    for key,field in [("source_workbook","source_sha256"),("assignments","assignments_sha256")]:
        path=study.path(key)
        if sha(path)!=contract[field]:raise ValueError("Original workbook/assignments changed")
        inputs[str(path)]=sha(path)
    inputs[str(study.path("source_contract"))]=sha(study.path("source_contract"))
    inputs[str(study.path("reference_metrics"))]=sha(study.path("reference_metrics"))
    source=study.path("reference_original")/"snapshot/src/pmc_rebuttal/network.py"
    if sha(source)!=sha(study.package_root/"src/cacox_original_head_study/original_network.py"):
        raise ValueError("Original network was modified")
    inputs[str(source)]=sha(source)
    for fold,count in enumerate([15,9,8,9,8],1):
        path=study.bundle_path(fold)
        if not verify_completed(path):raise ValueError("Incomplete MICE/Boruta+LASSO bundle")
        bundle=Bundle(path)
        if bundle.meta["candidate_variables"]!=CANDIDATES or len(bundle.variables)!=count:raise ValueError("Wrong predictors")
        if bundle.meta["selection_rule"]!=cfg["protocol"]["selection_rule"]:raise ValueError("Selection rule changed")
        bundle.inner_pairs(1)
        for name in ["bundle.json","COMPLETE.json"]:inputs[str(path/name)]=sha(path/name)
        old=original_fold(study,fold)
        if not verify_completed(old):raise ValueError("Original fold not complete")
        inputs[str(old/"predictions_all20.csv")]=sha(old/"predictions_all20.csv")
        for j in range(1,21):
            d=old/f"imputation_{j:02d}";proof=json.loads((d/"COMPLETE.json").read_text())
            for name in ["cacox.pt","training.json"]:
                if sha(d/name)!=proof["outputs"][name]:raise ValueError("Original model/recipe changed")
                inputs[str(d/name)]=sha(d/name)
            record=json.loads((d/"training.json").read_text())
            if record["seed"]!=cfg["common"]["seed_base"]+(fold-1)*1000+j:raise ValueError("Seed mismatch")
            expected={**cfg["common"],"hidden_dim":cfg["common"]["hidden"]}
            if any(expected[k]!=v for k,v in record["training_config"].items()):raise ValueError("Training config mismatch")
        r=study.path("reference_rcs")/f"repeat01_fold{fold:02d}/predictions_all20.csv"
        inputs[str(r)]=sha(r)
    return inputs


def references(study):
    result={}
    for role in ["R0_rcs_ridge","O0_original_cox"]:
        frames=[];sources=[]
        for fold in range(1,6):
            path=(study.path("reference_rcs")/f"repeat01_fold{fold:02d}/predictions_all20.csv" if role.startswith("R0")
                  else original_fold(study,fold)/"predictions_all20.csv")
            frames.append(pd.read_csv(path,float_precision="round_trip"));sources.append(dict(path=str(path),sha256=sha(path)))
        frame=normalize(pd.concat(frames,ignore_index=True),role,study.config["study_id"],"Cox log relative risk")
        assert_complete(frame)
        out=study.out/"references"/role;out.mkdir(parents=True,exist_ok=True)
        frame.to_csv(out/"predictions_all20.csv",index=False)
        write_json(out/"reference_manifest.json",dict(read_only_sources=sources,normalized_copy_sha256=sha(out/"predictions_all20.csv")))
        result[role]=frame
    assert_paired(*result.values())
    return result


def freeze_manifest(study,inputs):
    env=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
             gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,torch_threads=torch.get_num_threads(),tf32=torch.backends.cuda.matmul.allow_tf32)
    signature=dict(format_version=1,config=study.config,config_sha256=sha(study.config_path),source_sha256=study.code_hashes(),inputs=inputs,environment=env)
    lock_manifest(study.out,signature);write_json(study.out/"config.json",study.config);write_json(study.out/"environment.json",env)


def validate(study):
    from .validation import verify_original
    out=study.out/"validation";out.mkdir(parents=True,exist_ok=True)
    inputs=verify_inputs(study);proof=verify_original(study)
    frames=references(study);rows=[]
    for role in frames:
        _,pool=evaluate(frames[role])
        path=study.path("reference_metrics") if role.startswith("R0") else study.path("reference_original")/"comparison/pooled_metrics.csv"
        expected=pd.read_csv(path);name="RCS + ridge-Cox" if role.startswith("R0") else "CACox"
        expected=expected[expected.model.eq(name)]
        for row in pool.itertuples(index=False):
            value=float(expected.loc[expected.metric.eq(row.metric),"estimate"].iloc[0]);error=abs(value-row.estimate)
            if error>1e-6:raise AssertionError("Archived reference metric changed")
            rows.append(dict(role=role,metric=row.metric,estimate=row.estimate,archived=value,error=error))
    pd.DataFrame(rows).to_csv(out/"reference_metric_equivalence.csv",index=False)
    report=dict(status="passed",at=stamp(),source_sha256=study.code_hashes(),config_sha256=sha(study.config_path),input_fingerprints=inputs,original=proof)
    write_json(out/"VALIDATION.json",report);study.record("validation_passed",original_checkpoints=100)
    return report
