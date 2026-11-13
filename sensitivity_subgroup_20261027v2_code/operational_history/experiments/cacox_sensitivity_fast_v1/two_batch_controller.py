"""Run two disjoint 50-fold data batches concurrently, then finish frozen models."""
from __future__ import annotations

import os
import subprocess
import time

import psutil

import data_first_controller as phase
import full_controller as frozen


MAX_ACTIVE_BATCHES = 2
ADMISSION_FREE_GIB = 8.0


def count_data_markers() -> int:
    return sum((frozen.fold_path(branch, ordinal) / "FAST_DATA_COMPLETE.json").is_file()
               for ordinal in range(1, 51) for branch in frozen.BRANCHES)


def start_batch(batch: int) -> subprocess.Popen:
    log_dir = frozen.RUN / "logs/data_batches"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"batch_{batch:02d}.out.log").open("a", encoding="utf-8") as out, \
         (log_dir / f"batch_{batch:02d}.err.log").open("a", encoding="utf-8") as err:
        return subprocess.Popen(
            [str(frozen.PY), str(frozen.HERE / "data_batch_worker.py"), str(batch)],
            cwd=frozen.PROJECT, env=frozen.ENV, stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )


def run_data_batches(state: dict) -> None:
    pending = list(range(1, 6))
    active: dict[int, subprocess.Popen] = {}
    errors: list[str] = []
    state["stage"] = "two_data_batches"
    while pending or active:
        used_pct, free_gib = phase.memory_snapshot()
        state["memory_used_pct"] = used_pct
        state["memory_available_gib"] = free_gib
        state["data_complete"] = count_data_markers()
        while pending and len(active) < MAX_ACTIVE_BATCHES and not errors and \
              used_pct < 95.0 and free_gib >= (ADMISSION_FREE_GIB if active else 5.0):
            batch = pending.pop(0)
            active[batch] = start_batch(batch)
            state.setdefault("launched_batches", []).append(batch)
            state["active_batches"] = {str(k): p.pid for k, p in active.items()}
            frozen.atomic_json(frozen.STATUS, state)
            # Let the first batch allocate its R workers before admitting another.
            time.sleep(15)
            used_pct, free_gib = phase.memory_snapshot()
            state["memory_used_pct"] = used_pct
            state["memory_available_gib"] = free_gib
        for batch, proc in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del active[batch]
            marker = frozen.RUN / f"DATA_BATCH_{batch:02d}_COMPLETE.json"
            if rc or not marker.is_file():
                errors.append(f"Batch {batch} exited {rc} without a valid completion marker")
            else:
                rec = frozen.read_json_retry(marker)
                if rec.get("status") != "complete" or rec.get("units") != 50 or \
                   rec.get("source_sha256") != state["source_sha256"]:
                    errors.append(f"Batch {batch} completion marker failed source/coverage audit")
                else:
                    state.setdefault("completed_batches", []).append(batch)
            state["active_batches"] = {str(k): p.pid for k, p in active.items()}
            state["data_complete"] = count_data_markers()
            frozen.atomic_json(frozen.STATUS, state)
        if errors and not active:
            raise RuntimeError("; ".join(errors))
        if pending or active:
            time.sleep(10)


def main() -> None:
    lock = frozen.RUN / "FAST_CONTROLLER.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError("Existing controller lock requires reviewed handoff")
    source = frozen.fingerprint()
    state = {"status": "running", "schedule": "two_50_fold_batches",
             "stage": "preflight", "pid": os.getpid(), "started_unix": time.time(),
             "data_complete": count_data_markers(), "model_complete": 0,
             "planned_branch_folds": 250, "active_pid": None, "active_batches": {},
             "source_sha256": source, "steps": [],
             "memory_policy": {"max_used_pct_exclusive": 95.0,
                               "min_available_gib_inclusive": 5.0,
                               "second_batch_admission_gib": ADMISSION_FREE_GIB,
                               "per_fold_concurrent_mice_workers": 6,
                               "per_fold_concurrent_selection_workers": 2}}
    try:
        frozen.atomic_json(frozen.STATUS, state)
        phase.require_pilot_and_inputs(source)
        frozen.run_step("two_batch_preflight", [str(frozen.PY), str(frozen.PACKAGE / "run.py"),
                                                "check", "--device", "cuda", "--run-root",
                                                frozen.RUN_NAME], state)
        run_data_batches(state)
        if state["data_complete"] != 250 or frozen.fingerprint() != source:
            raise RuntimeError("Data batches did not cover 250 folds with frozen input/source")
        for ordinal in range(1, 51):
            for branch in frozen.BRANCHES:
                if not frozen.check_fast(branch, ordinal):
                    raise RuntimeError(f"Data completion audit failed: {branch} {ordinal}")
        frozen.atomic_json(frozen.RUN / "DATA_FIRST_DATA_COMPLETE.json", {
            "status": "complete", "schedule": state["schedule"], "branch_folds": 250,
            "source_sha256": source, "finished_unix": time.time(),
            "memory_policy": state["memory_policy"],
        })
        state["stage"] = "model_phase_250_folds"
        frozen.atomic_json(frozen.STATUS, state)
        for ordinal in range(1, 51):
            for branch in frozen.BRANCHES:
                frozen.do_model(branch, ordinal, state)
        if state["model_complete"] != 250 or frozen.fingerprint() != source:
            raise RuntimeError("Model phase did not cover 250 folds with frozen input/source")
        frozen.run_step("subgroup_original_oof", [str(frozen.R), "--vanilla",
                                                 str(frozen.PACKAGE / "code/subgroup_evaluation.R")], state)
        frozen.run_step("aggregate_sensitivity", [str(frozen.R), "--vanilla",
                                                 str(frozen.PACKAGE / "code/aggregate_sensitivity.R")], state)
        summary = frozen.RUN / "results/final_10x5_cacox_es/RUN_STATUS.txt"
        if "SUCCESS COMPLETE_50_OF_50" not in summary.read_text(encoding="utf-8"):
            raise RuntimeError("Formal aggregation did not report complete coverage")
        frozen.atomic_json(frozen.FINAL, {
            "status": "complete", "schedule": state["schedule"],
            "branches": list(frozen.BRANCHES), "branch_folds": 250,
            "source_sha256": source, "pilot_gate": str(frozen.PILOT),
            "finished_unix": time.time(),
        })
        state["status"] = "complete"
        state["stage"] = "complete"
        frozen.atomic_json(frozen.STATUS, state)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        frozen.atomic_json(frozen.STATUS, state)
        raise
    finally:
        if lock.is_file() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
