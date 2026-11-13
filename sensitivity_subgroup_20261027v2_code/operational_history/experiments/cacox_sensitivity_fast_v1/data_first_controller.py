"""Resume the frozen CACox study by completing data folds before missing models."""
from __future__ import annotations

import json
import os
import time

import psutil

import full_controller as frozen


MAX_MEMORY_USED_PCT = 95.0
MIN_AVAILABLE_GIB = 5.0
MEMORY_RECHECK_SECONDS = 30


def memory_snapshot() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    return 100.0 * (vm.total - vm.available) / vm.total, vm.available / 2**30


def wait_for_memory(branch: str, ordinal: int, state: dict) -> None:
    while True:
        used_pct, free_gib = memory_snapshot()
        state["memory_used_pct"] = used_pct
        state["memory_available_gib"] = free_gib
        if used_pct < MAX_MEMORY_USED_PCT and free_gib >= MIN_AVAILABLE_GIB:
            return
        state["stage"] = f"waiting_for_memory_{branch}_{ordinal:02d}"
        state["last_memory_check_unix"] = time.time()
        frozen.atomic_json(frozen.STATUS, state)
        time.sleep(MEMORY_RECHECK_SECONDS)


def require_pilot_and_inputs(source: dict[str, str]) -> None:
    if not (frozen.RUN / "data/INITIALIZED.txt").is_file():
        raise RuntimeError("Frozen run root is not initialized")
    pilot = frozen.read_json_retry(frozen.PILOT)
    if pilot.get("status") != "passed" or pilot.get("completed_models") != 5 or \
       pilot.get("source_sha256") not in (source, frozen.legacy_fingerprint()):
        raise RuntimeError("Existing first-fold pilot gate does not match frozen source")
    for branch in frozen.BRANCHES:
        if not frozen.check_fast(branch, 1) or not frozen.check_model(branch, 1):
            raise RuntimeError(f"First-fold pilot artifact invalid: {branch}")


def main() -> None:
    lock = frozen.RUN / "FAST_CONTROLLER.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError("An existing controller lock requires review before handoff")

    source = frozen.fingerprint()
    state = {
        "status": "running", "schedule": "data_first", "stage": "preflight",
        "pid": os.getpid(), "started_unix": time.time(), "active_pid": None,
        "data_complete": 0, "model_complete": 0, "planned_branch_folds": 250,
        "memory_policy": {"max_used_pct_exclusive": MAX_MEMORY_USED_PCT,
                          "min_available_gib_inclusive": MIN_AVAILABLE_GIB},
        "source_sha256": source, "steps": [],
    }
    try:
        frozen.atomic_json(frozen.STATUS, state)
        require_pilot_and_inputs(source)
        frozen.run_step("data_first_preflight", [str(frozen.PY), str(frozen.PACKAGE / "run.py"),
                                                 "check", "--device", "cuda", "--run-root",
                                                 frozen.RUN_NAME], state)

        state["stage"] = "data_first_250_folds"
        frozen.atomic_json(frozen.STATUS, state)
        for ordinal in range(1, 51):
            for branch in frozen.BRANCHES:
                if not frozen.check_fast(branch, ordinal):
                    wait_for_memory(branch, ordinal, state)
                frozen.do_fast_data(branch, ordinal, state)
        if state["data_complete"] != 250:
            raise RuntimeError("Data-first phase did not verify all 250 folds")
        if frozen.fingerprint() != source:
            raise RuntimeError("Frozen package input/source changed during data phase")
        frozen.atomic_json(frozen.RUN / "DATA_FIRST_DATA_COMPLETE.json", {
            "status": "complete", "branch_folds": 250, "source_sha256": source,
            "finished_unix": time.time(), "memory_policy": state["memory_policy"],
        })

        state["stage"] = "model_phase_250_folds"
        frozen.atomic_json(frozen.STATUS, state)
        for ordinal in range(1, 51):
            for branch in frozen.BRANCHES:
                frozen.do_model(branch, ordinal, state)
        if state["model_complete"] != 250:
            raise RuntimeError("Model phase did not verify all 250 folds")
        if frozen.fingerprint() != source:
            raise RuntimeError("Frozen package input/source changed during model phase")

        frozen.run_step("subgroup_original_oof", [str(frozen.R), "--vanilla",
                                                 str(frozen.PACKAGE / "code/subgroup_evaluation.R")], state)
        frozen.run_step("aggregate_sensitivity", [str(frozen.R), "--vanilla",
                                                 str(frozen.PACKAGE / "code/aggregate_sensitivity.R")], state)
        summary = frozen.RUN / "results/final_10x5_cacox_es/RUN_STATUS.txt"
        if "SUCCESS COMPLETE_50_OF_50" not in summary.read_text(encoding="utf-8"):
            raise RuntimeError("Formal aggregation did not report complete coverage")
        frozen.atomic_json(frozen.FINAL, {
            "status": "complete", "schedule": "data_first", "branches": list(frozen.BRANCHES),
            "branch_folds": 250, "source_sha256": source,
            "pilot_gate": str(frozen.PILOT), "finished_unix": time.time(),
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
        if lock.exists() and lock.read_text(encoding="utf-8") == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
