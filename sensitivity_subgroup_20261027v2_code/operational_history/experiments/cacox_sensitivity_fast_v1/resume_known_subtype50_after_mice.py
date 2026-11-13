"""Audited, resource-gated selection-only recovery for known_subtype fold 50."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

import full_controller as frozen
import fast_data
from opportunistic_fold_4x4 import gpu_free_mib, production, resource

BRANCH = "known_subtype"
ORDINAL = 50
FOLD = frozen.fold_path(BRANCH, ORDINAL)
RUN = frozen.RUN
AUDIT = RUN / "handoffs/opportunistic_known_subtype50_abort_review_20261108"
BASELINE = AUDIT / "PARTIAL_AUDIT.json"
STATUS = RUN / "RESUME_KNOWN_SUBTYPE50_STATUS.json"
SIDECAR = RUN / "OPPORTUNISTIC_4X4_known_subtype_50_STATUS.json"
LOGS = RUN / "logs/resume_known_subtype50_after_mice"


def formal_target_reached() -> bool:
    info = frozen.read_json_retry(RUN / "DATA_BATCH_05_STATUS.json")
    if info.get("status") == "complete":
        return True
    if info.get("current_ordinal") is None or info.get("current_branch") is None:
        return False
    current = (int(info["current_ordinal"]), frozen.BRANCHES.index(info["current_branch"]))
    target = (ORDINAL, frozen.BRANCHES.index(BRANCH))
    return current >= target


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(**fields: object) -> None:
    frozen.atomic_json(STATUS, {"at_unix": time.time(), "pid": os.getpid(), **fields})


def check_snapshot() -> dict:
    if not BASELINE.is_file():
        raise RuntimeError("The reviewed partial-fold audit is absent")
    baseline = frozen.read_json_retry(BASELINE)
    if not baseline["source_fingerprint_equal"] or len(baseline["mice"]) != 6:
        raise RuntimeError("Partial-fold audit did not pass")
    old = frozen.read_json_retry(FOLD / "FAST_DATA_STATUS.json")
    if old["source_sha256"] != frozen.fingerprint() or old["source_sha256"] != fast_data.fingerprint():
        raise RuntimeError("Frozen input or science code changed")
    if (FOLD / "FAST_DATA_COMPLETE.json").exists() or (FOLD / "FAST_DATA_RUNNING.lock").exists():
        raise RuntimeError("Fold is already complete or claimed")
    for record in baseline["mice"]:
        part = FOLD / record["partition"]
        if sha(part / "imputation.rds") != record["imputation_sha256"] or sha(part / "mice_chain_diagnostic.csv") != record["diagnostic_sha256"]:
            raise RuntimeError(f"MICE checkpoint changed: {record['partition']}")
        if not record["max_rhat"] < 1.1:
            raise RuntimeError(f"MICE R-hat failed: {record['partition']}")
    for record in baseline["partial_selection"]:
        path = FOLD / record["path"]
        if not path.is_file() or sha(path) != record["sha256"]:
            raise RuntimeError(f"Partial selection changed: {record['path']}")
    if any((FOLD / f"imputation_{i:02d}/combined_selection.csv").exists() for i in range(1, 21)):
        raise RuntimeError("An already-complete selection unit needs separate review")
    prior = frozen.read_json_retry(SIDECAR)
    if prior.get("status") != "aborted" or not prior.get("partial_fold_requires_review"):
        raise RuntimeError("Unexpected former sidecar state")
    return baseline


def preserve_original(baseline: dict) -> None:
    snapshot = AUDIT / "before_selection_only_resume"
    if snapshot.exists():
        raise RuntimeError("Recovery snapshot already exists")
    snapshot.mkdir(parents=True)
    for name in ("FAST_DATA_STATUS.json", "prepared.rds"):
        shutil.copy2(FOLD / name, snapshot / name)
    shutil.copy2(SIDECAR, snapshot / "OPPORTUNISTIC_4X4_known_subtype_50_STATUS.aborted.json")
    for record in baseline["mice"]:
        source = FOLD / record["partition"]
        target = snapshot / record["partition"]
        target.mkdir()
        for name in ("imputation.rds", "mice_chain_diagnostic.csv"):
            shutil.copy2(source / name, target / name)
    for record in baseline["partial_selection"]:
        source = FOLD / record["path"]
        target = snapshot / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for record in baseline["mice"]:
        target = snapshot / record["partition"]
        if sha(target / "imputation.rds") != record["imputation_sha256"]:
            raise RuntimeError("Snapshot MICE hash mismatch")
    for record in baseline["partial_selection"]:
        if sha(snapshot / record["path"]) != record["sha256"]:
            raise RuntimeError("Snapshot partial-selection hash mismatch")


def stop_only_recovery_tree(pid: int) -> list[int]:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    processes = parent.children(recursive=True) + [parent]
    identifiers = [p.pid for p in processes]
    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    return identifiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if STATUS.exists():
        raise RuntimeError("Existing recovery status requires review")
    baseline = check_snapshot()
    if formal_target_reached():
        raise RuntimeError("Formal batch has reached the partial fold")
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "mice": len(baseline["mice"]),
                          "partial_selection": len(baseline["partial_selection"])}), flush=True)
        return
    preserve_original(baseline)
    save(status="waiting_for_window", min_available_gib=2, max_used_pct_exclusive=95,
         original_mice_sha256={x["partition"]: x["imputation_sha256"] for x in baseline["mice"]})
    deadline = time.monotonic() + 4 * 3600
    while time.monotonic() < deadline:
        controller, folds = production()
        free, used = resource()
        gpu = gpu_free_mib()
        cpu = psutil.cpu_percent(interval=1)
        if formal_target_reached():
            raise RuntimeError("Formal batch reached fold 50 before a safe recovery window")
        if (free >= 12 and used < 65 and gpu >= 9000 and cpu < 70 and
            all(f["cuda"] == 0 for f in folds) and sum(f["mice"] for f in folds) <= 8):
            break
        time.sleep(10)
    else:
        save(status="no_safe_window", reason="No safe recovery admission within four hours")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    command = [str(frozen.PY), str(frozen.HERE / "fast_data.py"), BRANCH, str(ORDINAL),
               "--run-name", frozen.RUN_NAME, "--resume-after-mice",
               "--mice-workers", "4", "--selection-workers", "4", "--opportunistic-4x4"]
    previous = frozen.read_json_retry(SIDECAR)
    if previous.get("status") != "aborted":
        raise RuntimeError("Original sidecar status changed before recovery")
    with (LOGS / "fold.out.log").open("x", encoding="utf-8") as out, \
         (LOGS / "fold.err.log").open("x", encoding="utf-8") as err:
        proc = subprocess.Popen(command, cwd=frozen.PROJECT, env=frozen.ENV,
                                stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        frozen.atomic_json(SIDECAR, {"updated_unix": time.time(), "owner_pid": os.getpid(),
                                    "branch": BRANCH, "ordinal": ORDINAL,
                                    "status": "running", "child_pid": proc.pid,
                                    "recovery_status": str(STATUS),
                                    "previous_aborted_status": str(AUDIT / "before_selection_only_resume/OPPORTUNISTIC_4X4_known_subtype_50_STATUS.aborted.json")})
        save(status="running", child_pid=proc.pid, admission_free_gib=free,
             admission_used_pct=used, admission_gpu_free_mib=gpu)
        samples = LOGS / "resources.csv"
        with samples.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unix", "free_gib", "used_pct", "gpu_free_mib", "formal_mice", "formal_cuda"])
            while proc.poll() is None:
                free, used = resource()
                gpu = gpu_free_mib()
                try:
                    _, folds = production()
                    formal_mice = sum(f["mice"] for f in folds)
                    formal_cuda = sum(f["cuda"] for f in folds)
                    reason = None
                except Exception as exc:
                    formal_mice = formal_cuda = 0
                    reason = f"Formal controller unavailable: {exc!r}"
                stage_file = FOLD / "FAST_DATA_STATUS.json"
                active = frozen.read_json_retry(stage_file).get("active_pids", {}) if stage_file.exists() else {}
                extra_cuda = sum(key.startswith("selection_") for key in active)
                writer.writerow([time.time(), round(free, 3), round(used, 3), gpu,
                                 formal_mice, formal_cuda])
                stream.flush()
                if free < 2 or used >= 95 or gpu < 2500 or formal_cuda + extra_cuda > 4:
                    reason = reason or "Recovery resource protection threshold reached"
                if reason:
                    stopped = stop_only_recovery_tree(proc.pid)
                    interrupted = AUDIT / f"resume_interrupted_{int(time.time())}"
                    interrupted.mkdir()
                    lock = FOLD / "FAST_DATA_RUNNING.lock"
                    if lock.exists():
                        lock.replace(interrupted / "FAST_DATA_RUNNING.lock.stopped")
                    claim = RUN / "OPPORTUNISTIC_CUDA_ACTIVE.json"
                    if claim.exists():
                        owner = frozen.read_json_retry(claim).get("pid")
                        if owner in stopped:
                            claim.replace(interrupted / "OPPORTUNISTIC_CUDA_ACTIVE.stopped.json")
                    frozen.atomic_json(SIDECAR, {"updated_unix": time.time(),
                                                "owner_pid": os.getpid(), "branch": BRANCH,
                                                "ordinal": ORDINAL, "status": "aborted",
                                                "reason": reason, "stopped_pids": stopped,
                                                "partial_fold_requires_review": True})
                    save(status="aborted", reason=reason, stopped_pids=stopped)
                    return
                time.sleep(5)
    if proc.returncode:
        frozen.atomic_json(SIDECAR, {"updated_unix": time.time(), "owner_pid": os.getpid(),
                                    "branch": BRANCH, "ordinal": ORDINAL,
                                    "status": "fold_failed", "exit_code": proc.returncode})
        save(status="fold_failed", exit_code=proc.returncode)
        return
    original_source = frozen.read_json_retry(
        AUDIT / "before_selection_only_resume/FAST_DATA_STATUS.json")["source_sha256"]
    if not frozen.check_fast(BRANCH, ORDINAL) or frozen.fingerprint() != original_source:
        raise RuntimeError("Completed recovery failed frozen output audit")
    for record in baseline["mice"]:
        if sha(FOLD / record["partition"] / "imputation.rds") != record["imputation_sha256"]:
            raise RuntimeError("Recovery unexpectedly changed a MICE checkpoint")
    frozen.atomic_json(SIDECAR, {"updated_unix": time.time(), "owner_pid": os.getpid(),
                                "branch": BRANCH, "ordinal": ORDINAL,
                                "status": "complete", "recovered_selection_only": True,
                                "formal_batch_may_reuse": True,
                                "previous_aborted_status": str(AUDIT / "before_selection_only_resume/OPPORTUNISTIC_4X4_known_subtype_50_STATUS.aborted.json")})
    save(status="complete", selection_units=20, mice_reused=6,
         marker=str(FOLD / "FAST_DATA_COMPLETE.json"))
    print("SELECTION_ONLY_RECOVERY_COMPLETE", BRANCH, ORDINAL, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        save(status="failed", error=repr(exc))
        raise
