"""Bounded, single-fold test of filling a low-CPU window without stopping production."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

import full_controller as frozen


BRANCH = "renal_observed"
ORDINAL = 31
STATUS = frozen.RUN / "OPPORTUNISTIC_FOLD_TEST_STATUS.json"
SAMPLES = frozen.RUN / "OPPORTUNISTIC_FOLD_TEST_RESOURCES.csv"
LOGS = frozen.RUN / "logs/opportunistic_fold_test"
MAX_WAIT_SECONDS = 5400
MIN_ADMISSION_GIB = 12.0
MIN_RUNNING_GIB = 7.0


def record(**fields: object) -> None:
    frozen.atomic_json(STATUS, {"updated_unix": time.time(), **fields})


def stage(batch: int) -> tuple[str, int]:
    info = frozen.read_json_retry(frozen.RUN / f"DATA_BATCH_{batch:02d}_STATUS.json")
    if info.get("status") != "running" or not psutil.pid_exists(info["pid"]):
        raise RuntimeError(f"Production batch {batch} is not running")
    ordinal = int(info["current_ordinal"])
    fold = frozen.fold_path(info["current_branch"], ordinal)
    state = frozen.read_json_retry(fold / "FAST_DATA_STATUS.json")
    return str(state["stage"]), len(state.get("active_pids", {}))


def resource() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    return vm.available / 2**30, 100 * (vm.total-vm.available) / vm.total


def stop_only_test_tree(pid: int) -> list[int]:
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    ids = [p.pid for p in children] + [parent.pid]
    for proc in reversed(children):
        try: proc.terminate()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(children, timeout=10)
    try: parent.terminate()
    except psutil.NoSuchProcess: pass
    psutil.wait_procs([parent], timeout=10)
    for proc in children + [parent]:
        try:
            if proc.is_running(): proc.kill()
        except psutil.NoSuchProcess: pass
    return ids


def main() -> None:
    if STATUS.exists():
        raise RuntimeError("Existing test status requires review")
    if frozen.FINAL.exists():
        raise RuntimeError("Production is already complete")
    target = frozen.fold_path(BRANCH, ORDINAL)
    if target.exists() or (frozen.RUN / "DATA_BATCH_04_STATUS.json").exists():
        raise RuntimeError("Reserved future fold is no longer unclaimed")
    source = frozen.fingerprint()
    if not frozen.PILOT.exists() or not (frozen.RUN / "RESUME_HANDOFF_COMPLETE.json").exists():
        raise RuntimeError("Pilot or recovery gate is missing")
    LOGS.mkdir(parents=True, exist_ok=True)
    record(status="waiting_for_low_cpu_window", branch=BRANCH, ordinal=ORDINAL,
           max_mice_workers=2, max_selection_workers=1,
           admission={"cpu_pct_below": 50, "available_gib_at_least": MIN_ADMISSION_GIB,
                      "one_production_fold_active_workers_at_most": 2},
           source_sha256=source)
    started = time.monotonic()
    while time.monotonic() - started < MAX_WAIT_SECONDS:
        controller = frozen.read_json_retry(frozen.STATUS)
        if controller.get("status") != "running" or not psutil.pid_exists(controller["pid"]):
            raise RuntimeError("Production controller stopped during test admission")
        if target.exists() or (frozen.RUN / "DATA_BATCH_04_STATUS.json").exists():
            raise RuntimeError("Reserved fold was claimed before the test")
        two = [stage(2), stage(3)]
        free_gib, used_pct = resource()
        cpu_pct = psutil.cpu_percent(interval=3)
        if (free_gib >= MIN_ADMISSION_GIB and used_pct < 65 and cpu_pct < 50 and
                min(active for _, active in two) <= 2):
            break
        time.sleep(10)
    else:
        record(status="no_safe_admission_window", waited_seconds=time.monotonic()-started)
        return

    baseline_data = sum((frozen.fold_path(b, i) / "FAST_DATA_COMPLETE.json").is_file()
                        for i in range(1, 51) for b in frozen.BRANCHES)
    command = [str(frozen.PY), str(frozen.HERE / "fast_data.py"), BRANCH, str(ORDINAL),
               "--run-name", frozen.RUN_NAME, "--mice-workers", "2",
               "--selection-workers", "1"]
    with (LOGS / "fold.out.log").open("w", encoding="utf-8") as out, \
         (LOGS / "fold.err.log").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(command, cwd=frozen.PROJECT, env=frozen.ENV,
                                stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        record(status="running", branch=BRANCH, ordinal=ORDINAL, pid=proc.pid,
               admission_cpu_pct=cpu_pct, admission_free_gib=free_gib,
               production_data_before=baseline_data, source_sha256=source,
               started_unix=time.time())
        fields = ["unix", "cpu_pct", "free_gib", "used_pct", "test_stage",
                  "batch2_stage", "batch3_stage"]
        min_free = free_gib
        max_used = used_pct
        with SAMPLES.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            test_started = time.monotonic()
            while proc.poll() is None:
                free, used = resource()
                min_free, max_used = min(min_free, free), max(max_used, used)
                state_file = target / "FAST_DATA_STATUS.json"
                test_stage = frozen.read_json_retry(state_file).get("stage") if state_file.exists() else "starting"
                try: b2, _ = stage(2)
                except (OSError, RuntimeError, ValueError): b2 = "unavailable"
                try: b3, _ = stage(3)
                except (OSError, RuntimeError, ValueError): b3 = "unavailable"
                writer.writerow({"unix": time.time(), "cpu_pct": psutil.cpu_percent(interval=.2),
                                 "free_gib": round(free, 3), "used_pct": round(used, 3),
                                 "test_stage": test_stage, "batch2_stage": b2,
                                 "batch3_stage": b3})
                stream.flush()
                if free < MIN_RUNNING_GIB or used >= 90:
                    stopped = stop_only_test_tree(proc.pid)
                    archive = LOGS / "resource_abort_preserved"
                    archive.mkdir(exist_ok=True)
                    lock = target / "FAST_DATA_RUNNING.lock"
                    if lock.exists():
                        lock.replace(archive / "FAST_DATA_RUNNING.lock.stopped")
                    state_copy = target / "FAST_DATA_STATUS.json"
                    if state_copy.exists():
                        (archive / "FAST_DATA_STATUS.at_abort.json").write_bytes(state_copy.read_bytes())
                    record(status="resource_abort", branch=BRANCH, ordinal=ORDINAL,
                           min_free_gib=min_free, max_used_pct=max_used,
                           stopped_test_pids=stopped,
                           production_uninterrupted=True,
                           partial_fold_requires_audit_before_reuse=True)
                    return
                time.sleep(5)
    if proc.returncode:
        record(status="fold_failed", branch=BRANCH, ordinal=ORDINAL,
               exit_code=proc.returncode, min_free_gib=min_free,
               max_used_pct=max_used, error_log=str(LOGS / "fold.err.log"))
        return
    if frozen.fingerprint() != source or not frozen.check_fast(BRANCH, ORDINAL):
        raise RuntimeError("Completed opportunistic fold failed source/output hash audit")
    production_data_after = sum((frozen.fold_path(b, i) / "FAST_DATA_COMPLETE.json").is_file()
                                for i in range(1, 51) for b in frozen.BRANCHES)
    record(status="complete", branch=BRANCH, ordinal=ORDINAL,
           completed_unix=time.time(), elapsed_seconds=time.monotonic()-test_started,
           min_free_gib=min_free, max_used_pct=max_used,
           production_data_before=baseline_data, production_data_after=production_data_after,
           target_complete_marker=str(target / "FAST_DATA_COMPLETE.json"),
           future_batch_may_reuse=True, source_sha256=source)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record(status="failed", error=repr(exc))
        raise
