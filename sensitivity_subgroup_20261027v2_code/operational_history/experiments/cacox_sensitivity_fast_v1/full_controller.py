"""Gate, then run the frozen five-branch CACox sensitivity study with 6+6 data workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from functools import lru_cache
import subprocess
import sys
import time

import psutil

from fast_data import BRANCHES, HERE, LIB, PACKAGE, PROJECT, PY, R, atomic_json, fingerprint, output_files, sha

RUN_NAME = "run_fast_20261106_v1"
RUN = PACKAGE / RUN_NAME
NATIVE = PACKAGE / "run_native_pilot_20261106_v1"
KNOWN_REFERENCE = PACKAGE / "run_native_integrated_reference_20261106_v1"
STATUS = RUN / "FAST_CONTROLLER_STATUS.json"
LOGS = RUN / "logs/fast_controller"
PILOT = RUN / "PILOT_GATE.json"
FINAL = RUN / "FAST_FULL_COMPLETE.json"
ENV = dict(os.environ, CACOX_PORTABLE_ROOT=str(PACKAGE), CACOX_RUN_ROOT=str(RUN),
           CACOX_R_LIBS=str(LIB), CACOX_PYTHON=str(PY), CACOX_RSCRIPT=str(R),
           PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def read_json_retry(path: Path) -> dict:
    """Tolerate a brief Windows sharing violation while another process updates status."""
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def fold_path(branch: str, ordinal: int) -> Path:
    return RUN / "branches" / branch / f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"


def native_path(branch: str) -> Path:
    source = KNOWN_REFERENCE if branch == "known_subtype" else NATIVE
    return source / "branches" / branch / "repeat01_fold01"


def valid_native(branch: str) -> bool:
    fold = native_path(branch)
    return (fold / "selection_frequency_29variables.csv").is_file() and all(
        (fold / f"imputation_{j:02d}/combined_selection.csv").is_file() for j in range(1, 21))


@lru_cache(maxsize=1)
def legacy_fingerprint() -> dict[str, str] | None:
    """Verify old completed folds against the preserved pre-adaptive package."""
    legacy = os.environ.get("CACOX_FAST_LEGACY_PACKAGE")
    if not legacy:
        return None
    root = Path(legacy)
    return {name: sha(root / name) for name in fingerprint()}


@lru_cache(maxsize=1)
def preversion_fingerprint() -> dict[str, str] | None:
    """Verify copied completed folds from the immediately preceding frozen package."""
    previous = os.environ.get("CACOX_FAST_PREVIOUS_PACKAGE")
    if not previous:
        return None
    root = Path(previous)
    return {name: sha(root / name) for name in fingerprint()}


@lru_cache(maxsize=1)
def grandparent_fingerprint() -> dict[str, str] | None:
    """Verify older copied markers through their preserved source package."""
    previous = os.environ.get("CACOX_FAST_GRANDPARENT_PACKAGE")
    if not previous:
        return None
    root = Path(previous)
    return {name: sha(root / name) for name in fingerprint()}


@lru_cache(maxsize=1)
def preversion_complete_marker_hashes() -> dict[str, str]:
    audit = RUN / "PREVERSION_COMPLETE_UNITS.json"
    if not audit.is_file():
        return {}
    return read_json_retry(audit)["marker_sha256"]


@lru_cache(maxsize=1)
def audited_ancestor_fingerprints() -> list[dict[str, str]]:
    """Accept only exact source versions recorded by the pre-handoff audit."""
    audit = RUN / "PREVERSION_COMPLETE_UNITS.json"
    if not audit.is_file():
        return []
    return read_json_retry(audit).get("approved_source_fingerprints", [])


def check_fast(branch: str, ordinal: int) -> bool:
    fold = fold_path(branch, ordinal)
    marker = fold / "FAST_DATA_COMPLETE.json"
    if not marker.is_file():
        return False
    rec = json.loads(marker.read_text(encoding="utf-8"))
    current = fingerprint()
    old = legacy_fingerprint()
    historical = ordinal <= 6 and (branch, ordinal) != ("no_egpa", 6)
    prior = preversion_fingerprint()
    grandparent = grandparent_fingerprint()
    key = f"{branch}/repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"
    copied_prior_ok = (rec["source_sha256"] in
                       [prior, grandparent, *audited_ancestor_fingerprints()] and
                       preversion_complete_marker_hashes().get(key) == sha(marker))
    source_ok = (rec["source_sha256"] == current or
                 (historical and old is not None and rec["source_sha256"] == old) or
                 copied_prior_ok)
    if rec["status"] != "complete" or not source_ok:
        raise RuntimeError(f"Fast data source/signature mismatch: {branch} {ordinal}")
    if any(not (fold / p).is_file() or sha(fold / p) != h
           for p, h in rec["output_sha256"].items()):
        raise RuntimeError(f"Fast data output hash mismatch: {branch} {ordinal}")
    if set(rec["output_sha256"]) != {str(p.relative_to(fold)) for p in output_files(fold)}:
        raise RuntimeError(f"Fast data marker coverage mismatch: {branch} {ordinal}")
    return True


def check_model(branch: str, ordinal: int) -> bool:
    model = fold_path(branch, ordinal) / "cacox_es"
    if not (model / "COMPLETE.json").is_file():
        return False
    proc = subprocess.run([str(PY), str(PACKAGE / "code/verify_prediction.py"), str(model)],
                          env=ENV, cwd=PROJECT, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"Model completion verification failed: {branch} {ordinal}: {proc.stderr}")
    return True


def run_step(name: str, command: list[str], state: dict) -> None:
    state["stage"] = name
    state["step_started_unix"] = time.time()
    atomic_json(STATUS, state)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{name}.out.log").open("a", encoding="utf-8") as out, \
         (LOGS / f"{name}.err.log").open("a", encoding="utf-8") as err:
        proc = subprocess.Popen(command, env=ENV, cwd=PROJECT, stdout=out, stderr=err)
        state["active_pid"] = proc.pid
        atomic_json(STATUS, state)
        rc = proc.wait()
    state["active_pid"] = None
    state.setdefault("steps", []).append({"name": name, "exit_code": rc,
                                           "seconds": time.time() - state["step_started_unix"]})
    atomic_json(STATUS, state)
    if rc:
        raise RuntimeError(f"{name} exited {rc}; inspect {LOGS / (name + '.err.log')}")


def wait_for(path: Path, state: dict, label: str) -> None:
    state["stage"] = label
    atomic_json(STATUS, state)
    while not path.is_file():
        if label == "waiting_native_pilot":
            native_status = NATIVE / "progress.json"
            if native_status.is_file():
                record = read_json_retry(native_status)
                if record["status"] in ("FAILED", "STOPPED"):
                    raise RuntimeError("Native R pilot stopped or failed; preserve logs and inspect")
            owner = NATIVE / "RUNNING.json"
            if owner.is_file():
                pid = read_json_retry(owner)["pid"]
                if not psutil.pid_exists(pid):
                    raise RuntimeError("Native R controller disappeared before pilot completed")
        elif label == "waiting_known_fast_data":
            info = fold_path("known_subtype", 1) / "FAST_DATA_STATUS.json"
            if info.is_file():
                item = json.loads(info.read_text(encoding="utf-8"))
                if item["status"] == "failed" or not psutil.pid_exists(item["pid"]):
                    raise RuntimeError("Known-subtype fast data worker failed or disappeared")
        state["last_wait_check_unix"] = time.time()
        atomic_json(STATUS, state)
        time.sleep(20)


def do_fast_data(branch: str, ordinal: int, state: dict) -> None:
    if not check_fast(branch, ordinal):
        run_step(f"data_{branch}_{ordinal:02d}",
                 [str(PY), str(HERE / "fast_data.py"), branch, str(ordinal),
                  "--run-name", RUN_NAME], state)
    if not check_fast(branch, ordinal):
        raise RuntimeError(f"Fast data fold did not pass final marker audit: {branch} {ordinal}")
    state["data_complete"] += 1
    atomic_json(STATUS, state)


def do_native_compare(branch: str, state: dict) -> None:
    source = KNOWN_REFERENCE if branch == "known_subtype" else NATIVE
    if not valid_native(branch):
        raise RuntimeError(f"Native reference fold incomplete: {branch}")
    audit_path = fold_path(branch, 1) / "NATIVE_EXACT_AUDIT.json"
    if not audit_path.is_file():
        run_step(f"compare_{branch}_01",
                 [str(R), "--vanilla", str(HERE / "compare_fold.R"), str(source), branch, "1"], state)
    audit = read_json_retry(audit_path)
    if audit["pass"] is not True or audit["partitions"] != 6 or audit["votes"] != 20:
        raise RuntimeError(f"Native exact comparison failed: {branch}")


def do_model(branch: str, ordinal: int, state: dict) -> None:
    if not check_model(branch, ordinal):
        run_step(f"model_{branch}_{ordinal:02d}",
                 [str(PY), str(PACKAGE / "code/cacox_fold.py"), branch, str(ordinal),
                  "--device", "cuda", "--threads", "1", "--rscript", str(R)], state)
    if not check_model(branch, ordinal):
        raise RuntimeError(f"CACox model failed output verification: {branch} {ordinal}")
    state["model_complete"] += 1
    atomic_json(STATUS, state)


def main() -> None:
    if not (RUN / "data/INITIALIZED.txt").is_file():
        raise RuntimeError("Fast run root must already be initialized")
    lock = RUN / "FAST_CONTROLLER.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError("Existing fast controller lock requires review")
    source = fingerprint()
    state = {"status": "running", "stage": "preflight", "pid": os.getpid(),
             "started_unix": time.time(), "active_pid": None, "data_complete": 0,
             "model_complete": 0, "planned_branch_folds": 250,
             "source_sha256": source, "steps": []}
    try:
        atomic_json(STATUS, state)
        run_step("preflight", [str(PY), str(PACKAGE / "run.py"), "check", "--device", "cuda",
                               "--run-root", RUN_NAME], state)
        # The first fast data fold was started before this controller, in the same run root.
        if not check_fast("known_subtype", 1):
            wait_for(fold_path("known_subtype", 1) / "FAST_DATA_COMPLETE.json",
                     state, "waiting_known_fast_data")
        do_fast_data("known_subtype", 1, state)
        do_native_compare("known_subtype", state)
        do_model("known_subtype", 1, state)
        # The remaining native references are being produced in a separate, read-only run.
        if not all(valid_native(b) for b in BRANCHES if b != "known_subtype"):
            state["stage"] = "waiting_native_pilot"; atomic_json(STATUS, state)
            while not all(valid_native(b) for b in BRANCHES if b != "known_subtype"):
                record = read_json_retry(NATIVE / "progress.json")
                if record["status"] in ("FAILED", "STOPPED"):
                    raise RuntimeError("Native R pilot failed or stopped; inspect native logs")
                owner = NATIVE / "RUNNING.json"
                if not owner.is_file() or not psutil.pid_exists(read_json_retry(owner)["pid"]):
                    raise RuntimeError("Native R controller disappeared before all references finished")
                state["last_wait_check_unix"] = time.time(); atomic_json(STATUS, state)
                time.sleep(20)
        for branch in BRANCHES:
            if branch == "known_subtype":
                continue
            do_fast_data(branch, 1, state)
            do_native_compare(branch, state)
            do_model(branch, 1, state)
        if fingerprint() != source:
            raise RuntimeError("Frozen input/source changed during pilot")
        pilot = {"status": "passed", "branches": list(BRANCHES),
                 "native_exact_audits": {b: str(fold_path(b, 1) / "NATIVE_EXACT_AUDIT.json")
                                         for b in BRANCHES},
                 "completed_models": 5, "source_sha256": source,
                 "finished_unix": time.time()}
        if PILOT.is_file():
            earlier = read_json_retry(PILOT)
            if earlier.get("status") != "passed" or earlier.get("completed_models") != 5 or \
               earlier.get("source_sha256") not in (source, legacy_fingerprint()):
                raise RuntimeError("Existing pilot gate does not match verified source")
        else:
            atomic_json(PILOT, pilot)
        state["stage"] = "full_250_folds"; atomic_json(STATUS, state)
        for ordinal in range(1, 51):
            for branch in BRANCHES:
                if ordinal == 1:
                    continue
                do_fast_data(branch, ordinal, state)
                do_model(branch, ordinal, state)
        if state["data_complete"] != 250 or state["model_complete"] != 250:
            raise RuntimeError("Full run did not cover all 250 branch-folds")
        if fingerprint() != source:
            raise RuntimeError("Frozen input/source changed during full run")
        run_step("subgroup_original_oof", [str(R), "--vanilla",
                 str(PACKAGE / "code/subgroup_evaluation.R")], state)
        run_step("aggregate_sensitivity", [str(R), "--vanilla",
                 str(PACKAGE / "code/aggregate_sensitivity.R")], state)
        summary = RUN / "results/final_10x5_cacox_es/RUN_STATUS.txt"
        if "SUCCESS COMPLETE_50_OF_50" not in summary.read_text(encoding="utf-8"):
            raise RuntimeError("Formal aggregation did not report complete coverage")
        atomic_json(FINAL, {"status": "complete", "branches": list(BRANCHES),
                            "branch_folds": 250, "source_sha256": source,
                            "pilot_gate": str(PILOT), "finished_unix": time.time()})
        state["status"] = "complete"; state["stage"] = "complete"; atomic_json(STATUS, state)
    except BaseException as exc:
        state["status"] = "failed"; state["error"] = repr(exc); atomic_json(STATUS, state)
        raise
    finally:
        if lock.exists() and lock.read_text(encoding="utf-8") == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
