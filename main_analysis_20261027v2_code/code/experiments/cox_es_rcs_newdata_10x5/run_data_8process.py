"""Bounded two-fold data scheduler; scientific R code and inputs are frozen.

At most two outer folds are in flight and at most eight independent Rscript
processes run at once. Each fold still executes exactly the same prepare,
six MICE partitions, R-hat validation, 20 selectors, summary, and export as
run_data.py. Completed fold bundles are verified and left untouched.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import run_data as original
from run_data_6process import check_completed_folds, check_frozen_inputs

R_PROCESSES = 8
FOLDS_IN_FLIGHT = 2
PER_FOLD_WORKERS = {"impute": 6, "select": 6}
R_LIMIT = threading.BoundedSemaphore(R_PROCESSES)
EVENT_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()
JSON_LOCK = threading.RLock()
AUDIT = original.OUT / "DATA_SCHEDULER_8PROCESS.json"
EVENTS = original.OUT / "DATA_EVENTS.jsonl"
ORIGINAL_RUN_ONE = original.run_one
ORIGINAL_ATOMIC_JSON = original.atomic_json


def locked_atomic_json(path: Path, data: dict) -> None:
    # run_data.py's temporary filename uses the process PID. Two fold threads
    # must not replace the same RUN_STATUS.json.tmp.PID simultaneously.
    with JSON_LOCK:
        ORIGINAL_ATOMIC_JSON(path, data)


def bounded_run_one(stage: str, repeat: int, fold: int, ident: int) -> dict:
    if original.stage_done(stage, repeat, fold, ident):
        return dict(stage=stage, repeat_id=repeat, fold=fold, ident=ident,
                    reused=True, seconds=0)
    with R_LIMIT:
        return ORIGINAL_RUN_ONE(stage, repeat, fold, ident)


def r_command(args: list[str], log: Path) -> None:
    original.await_memory()
    log.parent.mkdir(parents=True, exist_ok=True)
    with R_LIMIT, log.open("w", encoding="utf8") as output:
        result = subprocess.run(args, cwd=original.ROOT, stdout=output,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"R stage failed exit={result.returncode}; see {log}")


def event(stage: str, repeat: int, fold: int, **details: object) -> None:
    row = dict(at=original.now(), stage=stage, repeat_id=repeat, fold=fold, **details)
    with EVENT_LOCK, EVENTS.open("a", encoding="utf8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()


def ready_count() -> int:
    return sum((original.unit(r, k) / "selected_variables.txt").exists()
               for r in range(1, 11) for k in range(1, 6))


def status(stage: str, repeat: int, fold: int) -> None:
    with STATUS_LOCK:
        payload = dict(status="running", scheduler="8process_twofold", stage=stage,
                       repeat_id=repeat, fold=fold, completed_new_folds=ready_count(),
                       controller_pid=os.getpid(), resource=original.memory(), at=original.now())
        original.atomic_json(original.OUT / "RUN_STATUS.json", payload)
        original.atomic_json(original.OUT / "DATA_8PROCESS_STATUS.json", payload)


def process_fold(repeat: int, fold: int) -> None:
    directory = original.unit(repeat, fold)
    bundle = (original.OUT / "bundles" /
              f"repeat{repeat:02d}_fold{fold:02d}" / "main")
    selected_path = directory / "selected_variables.txt"
    if selected_path.exists():
        original.validate_mice(repeat, fold)
        if not all(original.stage_done("select", repeat, fold, mi)
                   for mi in range(1, 21)):
            raise RuntimeError(f"Completed fold has incomplete selectors {repeat}/{fold}")
        if not original.verify_completed(bundle):
            raise RuntimeError(f"Completed fold has no bundle {repeat}/{fold}")
        return

    status("prepare", repeat, fold)
    if not (directory / "prepared.rds").exists():
        r_command([str(original.RSCRIPT), "--vanilla", str(original.SCRIPT),
                   str(original.ROOT), str(original.OUT), "prepare",
                   str(repeat), str(fold)], directory / "logs/prepare.log")
    if not (directory / "prepared.rds").exists():
        raise RuntimeError(f"Missing prepared fold {repeat}/{fold}")

    status("impute", repeat, fold)
    if not all(original.stage_done("impute", repeat, fold, ident)
               for ident in range(6)):
        jobs = original.batch("impute", repeat, fold, range(6))
        event("impute", repeat, fold, jobs=jobs)
    diagnostic = original.validate_mice(repeat, fold)
    original.atomic_json(directory / "MICE_VALIDATION.json", diagnostic)

    status("select", repeat, fold)
    if not all(original.stage_done("select", repeat, fold, ident)
               for ident in range(1, 21)):
        jobs = original.batch("select", repeat, fold, range(1, 21))
        event("select", repeat, fold, jobs=jobs)
    if not selected_path.exists():
        r_command([str(original.RSCRIPT), "--vanilla", str(original.SCRIPT),
                   str(original.ROOT), str(original.OUT), "summary",
                   str(repeat), str(fold)], directory / "logs/summary.log")
    selected = selected_path.read_text(encoding="utf8").splitlines()
    if not selected:
        raise RuntimeError(f"Empty selected variable list {repeat}/{fold}")

    status("export", repeat, fold)
    if not original.verify_completed(bundle):
        r_command([str(original.RSCRIPT), "--vanilla", str(original.HERE / "export_bundle.R"),
                   str(original.ROOT), str(original.OUT), str(repeat), str(fold)],
                  directory / "logs/export_bundle.log")
        original.atomic_json(bundle / "bundle.json", dict(
            m=20, maxit=60, candidate_variables=original.CANDIDATES,
            selected_variables=selected, repeat_id=repeat, fold=fold,
            backend="upstream R MICE 3.19.0, 20x60",
            selection_rule="Boruta non-rejected UNION Cox-LASSO lambda.min; >=10/20"))
        loaded = original.Bundle(bundle)
        if loaded.variables != selected:
            raise ValueError(f"Exported variable mismatch {repeat}/{fold}")
        for mi in (1, 10, 20):
            loaded.inner_pairs(mi)
            loaded.pair(mi)
        files = [str(path.relative_to(bundle)) for path in bundle.rglob("*")
                 if path.is_file() and path.name != "COMPLETE.json"]
        original.mark_complete(bundle, files)
    if not original.verify_completed(bundle):
        raise RuntimeError(f"Bundle verification failed {repeat}/{fold}")
    event("fold_ready", repeat, fold, selected=selected,
          max_Rhat=diagnostic["max_Rhat"])
    status("fold_ready", repeat, fold)


def verify_and_record() -> int:
    manifest = check_frozen_inputs()
    prior = original.OUT / "DATA_SCHEDULER_6PROCESS.json"
    if not prior.exists():
        raise RuntimeError("Missing previous six-process scheduling audit")
    done = check_completed_folds()
    content = dict(at=original.now(), old_data_manifest_sha256=original.sha(
        original.OUT / "DATA_MANIFEST.json"),
        old_data_code_sha256=original.sha(original.HERE / "run_data.py"),
        previous_schedule_sha256=original.sha(prior),
        scheduler_code_sha256=original.sha(Path(__file__)),
        completed_folds_verified_before_resume=done,
        original_workers=manifest["workers"],
        previous_workers=dict(impute=6, select=6),
        new_max_independent_R_processes=R_PROCESSES,
        max_folds_in_flight=FOLDS_IN_FLIGHT,
        per_fold_workers=PER_FOLD_WORKERS,
        scientific_settings_unchanged=True)
    if AUDIT.exists():
        prior_record = json.loads(AUDIT.read_text(encoding="utf8"))
        for key in ("old_data_manifest_sha256", "old_data_code_sha256",
                    "previous_schedule_sha256", "scheduler_code_sha256",
                    "new_max_independent_R_processes", "max_folds_in_flight"):
            if content[key] != prior_record[key]:
                raise RuntimeError(f"Eight-process scheduling audit changed: {key}")
    else:
        original.atomic_json(AUDIT, content)
    return done


def main() -> None:
    if sys.argv[1:] not in (["run"], ["audit"]):
        raise SystemExit("Usage: run_data_8process.py run|audit")
    completed = verify_and_record() if sys.argv[1:] == ["run"] else check_completed_folds()
    if sys.argv[1:] == ["audit"]:
        check_frozen_inputs()
        print(json.dumps(dict(completed_folds=completed,
                              impute_process_limit=R_PROCESSES,
                              folds_in_flight=FOLDS_IN_FLIGHT)))
        return
    lock = original.OUT / "DATA_RUNNING.lock"
    if lock.exists():
        raise RuntimeError(f"Active or stale data lock: {lock}")
    lock.write_text(str(os.getpid()), encoding="ascii")
    original.MAX_WORKERS = PER_FOLD_WORKERS.copy()
    original.run_one = bounded_run_one
    original.atomic_json = locked_atomic_json
    try:
        keys = [(r, k) for r in range(1, 11) for k in range(1, 6)]
        pending = [key for key in keys if not
                   (original.unit(*key) / "selected_variables.txt").exists()]
        for offset in range(0, len(pending), FOLDS_IN_FLIGHT):
            pair = pending[offset:offset + FOLDS_IN_FLIGHT]
            with ThreadPoolExecutor(max_workers=len(pair)) as pool:
                futures = {pool.submit(process_fold, *key): key for key in pair}
                for future in as_completed(futures):
                    future.result()
        if check_completed_folds() != 50:
            raise RuntimeError("Final data fold integrity count is not 50")
        original.atomic_json(original.OUT / "DATA_COMPLETE.json", dict(
            at=original.now(), prepared=50, new_folds=50,
            imputation_partitions=300, selection_MI_jobs=1000,
            scheduler="8process_twofold"))
        original.atomic_json(original.OUT / "RUN_STATUS.json",
                             dict(status="data_complete", at=original.now()))
    except BaseException as error:
        original.atomic_json(original.OUT / "RUN_STATUS.json", dict(
            status="failed", scheduler="8process_twofold", error=repr(error),
            controller_pid=os.getpid(), resource=original.memory(), at=original.now()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    main()
