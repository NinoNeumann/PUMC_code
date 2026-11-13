"""Prepare one disjoint 50-unit batch using the existing frozen data-fold runner."""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import psutil

import full_controller as frozen


MICE_WORKERS = 6
SELECTION_WORKERS = 2


def units(batch: int):
    for index in range((batch - 1) * 50, batch * 50):
        ordinal, branch_index = divmod(index, len(frozen.BRANCHES))
        yield frozen.BRANCHES[branch_index], ordinal + 1


def available_memory() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) * 100.0 / vm.total, vm.available / 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int, choices=range(1, 6))
    args = parser.parse_args()
    batch = args.batch
    status_path = frozen.RUN / f"DATA_BATCH_{batch:02d}_STATUS.json"
    complete_path = frozen.RUN / f"DATA_BATCH_{batch:02d}_COMPLETE.json"
    lock = frozen.RUN / f"DATA_BATCH_{batch:02d}.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {"status": "running", "batch": batch, "pid": os.getpid(),
             "started_unix": time.time(), "completed_units": 0, "reused_units": 0,
             "active_pid": None, "current_branch": None, "current_ordinal": None,
             "source_sha256": frozen.fingerprint(),
             "mice_workers": MICE_WORKERS,
             "selection_workers": SELECTION_WORKERS,
             "memory_policy": {"max_used_pct_exclusive": 95.0,
                               "min_available_gib_inclusive": 5.0}}
    try:
        frozen.atomic_json(status_path, state)
        if complete_path.is_file():
            prior = frozen.read_json_retry(complete_path)
            if prior.get("status") != "complete" or prior.get("source_sha256") != state["source_sha256"]:
                raise RuntimeError("Existing batch completion marker does not match frozen input")
        log_dir = frozen.RUN / "logs/data_batches" / f"batch_{batch:02d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        for branch, ordinal in units(batch):
            state["current_branch"] = branch
            state["current_ordinal"] = ordinal
            state["stage"] = "checking"
            frozen.atomic_json(status_path, state)
            extra_paths = (frozen.RUN / "OPPORTUNISTIC_4X4_STATUS.json",
                           frozen.RUN / f"OPPORTUNISTIC_4X4_{branch}_{ordinal:02d}_STATUS.json")
            for extra_status in extra_paths:
                if not extra_status.is_file():
                    continue
                extra = frozen.read_json_retry(extra_status)
                if (extra.get("branch"), extra.get("ordinal")) == (branch, ordinal):
                    while extra.get("status") == "running":
                        owner = extra.get("owner_pid")
                        if owner is None or not psutil.pid_exists(int(owner)):
                            raise RuntimeError(f"Extra fold owner disappeared: {branch} {ordinal}")
                        state["stage"] = "waiting_for_extra_fold"
                        frozen.atomic_json(status_path, state)
                        time.sleep(5)
                        extra = frozen.read_json_retry(extra_status)
                    if extra.get("status") not in ("complete", "waiting_for_window", "not_claimed"):
                        raise RuntimeError(f"Extra fold requires review before reuse: {branch} {ordinal}: {extra.get('status')}")
            if frozen.check_fast(branch, ordinal):
                state["reused_units"] += 1
            else:
                while True:
                    used_pct, free_gib = available_memory()
                    state["memory_used_pct"] = used_pct
                    state["memory_available_gib"] = free_gib
                    if used_pct < 95.0 and free_gib >= 5.0:
                        break
                    state["stage"] = "waiting_for_memory"
                    frozen.atomic_json(status_path, state)
                    time.sleep(30)
                state["stage"] = "data"
                frozen.atomic_json(status_path, state)
                name = f"{branch}_{ordinal:02d}"
                with (log_dir / f"{name}.out.log").open("a", encoding="utf-8") as out, \
                     (log_dir / f"{name}.err.log").open("a", encoding="utf-8") as err:
                    # All six partitions and 20 votes run; only overlap is bounded.
                    proc = subprocess.Popen(
                        [str(frozen.PY), str(frozen.HERE / "fast_data.py"), branch,
                         str(ordinal), "--run-name", frozen.RUN_NAME,
                         "--mice-workers", str(MICE_WORKERS),
                         "--selection-workers", str(SELECTION_WORKERS)],
                        cwd=frozen.PROJECT, env=frozen.ENV, stdout=out, stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    state["active_pid"] = proc.pid
                    frozen.atomic_json(status_path, state)
                    rc = proc.wait()
                state["active_pid"] = None
                frozen.atomic_json(status_path, state)
                if rc:
                    raise RuntimeError(f"Data fold {branch} {ordinal} exited {rc}; see {log_dir / (name + '.err.log')}")
                if not frozen.check_fast(branch, ordinal):
                    raise RuntimeError(f"Data fold marker failed audit: {branch} {ordinal}")
            state["completed_units"] += 1
            state["last_completed_unix"] = time.time()
            frozen.atomic_json(status_path, state)
        if frozen.fingerprint() != state["source_sha256"]:
            raise RuntimeError("Frozen input/source changed during data batch")
        frozen.atomic_json(complete_path, {
            "status": "complete", "batch": batch, "units": 50,
            "source_sha256": state["source_sha256"], "finished_unix": time.time(),
        })
        state["status"] = "complete"
        state["stage"] = "complete"
        frozen.atomic_json(status_path, state)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        frozen.atomic_json(status_path, state)
        raise
    finally:
        if lock.is_file() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
