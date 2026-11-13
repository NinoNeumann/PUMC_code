"""Run one future fold with 4 MICE/4 CUDA workers only in a measured spare-capacity window."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time

import psutil

import full_controller as frozen
from fast_data import selection_mutex
from opportunistic_fold_test import stop_only_test_tree


BRANCH = os.environ.get("CACOX_OPPORTUNISTIC_BRANCH", "egfr_observed")
ORDINAL = int(os.environ.get("CACOX_OPPORTUNISTIC_ORDINAL", "41"))
if BRANCH not in frozen.BRANCHES or ORDINAL not in range(1, 51):
    raise ValueError("Invalid opportunistic branch/ordinal")
TAG = "" if (BRANCH, ORDINAL) == ("egfr_observed", 41) else f"_{BRANCH}_{ORDINAL:02d}"
STATUS = frozen.RUN / f"OPPORTUNISTIC_4X4{TAG}_STATUS.json"
SAMPLES = frozen.RUN / f"OPPORTUNISTIC_4X4{TAG}_RESOURCES.csv"
LOGS = frozen.RUN / "logs" / f"opportunistic_4x4{TAG}"
MAX_WAIT_SECONDS = 4 * 3600


def formal_target_reached() -> bool:
    """Return whether the active formal batch has reached this fold."""
    batch = (ORDINAL - 1) // 10 + 1
    path = frozen.RUN / f"DATA_BATCH_{batch:02d}_STATUS.json"
    if not path.exists():
        return False
    info = frozen.read_json_retry(path)
    if info.get("status") == "complete":
        return True
    current_ordinal = info.get("current_ordinal")
    current_branch = info.get("current_branch")
    if current_ordinal is None or current_branch is None:
        return False
    current_key = (int(current_ordinal), frozen.BRANCHES.index(current_branch))
    target_key = (ORDINAL, frozen.BRANCHES.index(BRANCH))
    return current_key >= target_key


def record(**fields: object) -> None:
    frozen.atomic_json(STATUS, {"updated_unix": time.time(), "owner_pid": os.getpid(),
                                "branch": BRANCH, "ordinal": ORDINAL, **fields})


def gpu_free_mib() -> int:
    for attempt in range(3):
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10, check=True)
            return int(proc.stdout.strip().splitlines()[0].strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            if attempt == 2:
                return 0
            time.sleep(1)
    raise AssertionError("unreachable")


def resource() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    return vm.available / 2**30, 100 * (vm.total - vm.available) / vm.total


def production() -> tuple[dict, list[dict]]:
    controller = frozen.read_json_retry(frozen.STATUS)
    if controller.get("status") != "running" or not psutil.pid_exists(int(controller["pid"])):
        raise RuntimeError("Formal controller is not running")
    folds = []
    for batch in controller.get("active_batches", {}):
        info = frozen.read_json_retry(frozen.RUN / f"DATA_BATCH_{int(batch):02d}_STATUS.json")
        if info.get("status") != "running" or not psutil.pid_exists(int(info["pid"])):
            continue
        if info.get("current_branch") is None:
            continue
        fold = frozen.fold_path(info["current_branch"], int(info["current_ordinal"]))
        stage_file = fold / "FAST_DATA_STATUS.json"
        if not stage_file.exists():
            continue
        stage = frozen.read_json_retry(stage_file)
        active = stage.get("active_pids", {})
        folds.append({"batch": int(batch), "stage": stage.get("stage"),
                      "mice": sum(key.startswith("mice_") for key in active),
                      "cuda": sum(key.startswith("selection_") for key in active),
                      "active": len(active)})
    return controller, folds


def abort(proc: subprocess.Popen, target, reason: str, min_free: float,
          max_used: float, min_gpu: int) -> None:
    stopped = stop_only_test_tree(proc.pid)
    archive = LOGS / "abort_preserved"
    archive.mkdir(exist_ok=True)
    lock = target / "FAST_DATA_RUNNING.lock"
    if lock.exists():
        lock.replace(archive / "FAST_DATA_RUNNING.lock.stopped")
    stage = target / "FAST_DATA_STATUS.json"
    if stage.exists():
        (archive / "FAST_DATA_STATUS.at_abort.json").write_bytes(stage.read_bytes())
    with selection_mutex(frozen.RUN):
        cuda_claim = frozen.RUN / "OPPORTUNISTIC_CUDA_ACTIVE.json"
        if cuda_claim.exists():
            owner = frozen.read_json_retry(cuda_claim).get("pid")
            if owner in stopped:
                cuda_claim.replace(archive / "OPPORTUNISTIC_CUDA_ACTIVE.stopped.json")
    record(status="aborted", reason=reason, stopped_extra_pids=stopped,
           min_free_gib=min_free, max_used_pct=max_used, min_gpu_free_mib=min_gpu,
           partial_fold_requires_review=True, formal_batches_untouched=True)


def main() -> None:
    if STATUS.exists():
        raise RuntimeError("Existing 4x4 status requires review")
    if frozen.FINAL.exists():
        raise RuntimeError("Formal run is already complete")
    target = frozen.fold_path(BRANCH, ORDINAL)
    if target.exists() or formal_target_reached():
        raise RuntimeError("Target fold is already claimed by a formal batch")
    source = frozen.fingerprint()
    if not frozen.PILOT.exists() or not (frozen.RUN / "ADAPTIVE_PROTOCOL_V4.json").exists():
        raise RuntimeError("Versioned pilot or adaptive audit is absent")
    LOGS.mkdir(parents=True, exist_ok=True)
    record(status="waiting_for_window", source_sha256=source,
           mice_workers=4, selection_workers=4,
           admission={"min_free_gib": 16, "max_used_pct_exclusive": 50,
                      "min_gpu_free_mib": 9000, "max_cpu_pct": 55,
                      "formal_mice_workers_at_most": 8,
                      "formal_cuda_workers": 0})
    waiting_since = time.monotonic()
    while time.monotonic() - waiting_since < MAX_WAIT_SECONDS:
        if target.exists() or formal_target_reached():
            record(status="not_claimed", reason="Formal batch claimed target first")
            return
        controller, folds = production()
        free, used = resource()
        cpu = psutil.cpu_percent(interval=1)
        gpu = gpu_free_mib()
        if (1 <= len(folds) <= 2 and all(f["cuda"] == 0 for f in folds) and
            sum(f["mice"] for f in folds) <= 8 and min(f["active"] for f in folds) <= 2 and
            free >= 16 and used < 50 and gpu >= 9000 and cpu < 55):
            break
        time.sleep(10)
    else:
        record(status="not_claimed", reason="No safe resource window within four hours",
               waited_seconds=time.monotonic() - waiting_since)
        return

    if target.exists() or formal_target_reached():
        record(status="not_claimed", reason="Formal batch claimed target at admission")
        return

    before = sum((frozen.fold_path(b, i) / "FAST_DATA_COMPLETE.json").exists()
                 for i in range(1, 51) for b in frozen.BRANCHES)
    command = [str(frozen.PY), str(frozen.HERE / "fast_data.py"), BRANCH, str(ORDINAL),
               "--run-name", frozen.RUN_NAME, "--mice-workers", "4",
               "--selection-workers", "4", "--opportunistic-4x4"]
    # Reserve in the status read by the already-running formal batch worker
    # before starting the extra child. Its owner PID is this sidecar process.
    record(status="running", child_pid=None, started_unix=time.time(),
           admission_cpu_pct=cpu, admission_free_gib=free,
           admission_used_pct=used, admission_gpu_free_mib=gpu,
           production_data_before=before, source_sha256=source)
    with (LOGS / "fold.out.log").open("x", encoding="utf-8") as out, \
         (LOGS / "fold.err.log").open("x", encoding="utf-8") as err:
        proc = subprocess.Popen(command, cwd=frozen.PROJECT, env=frozen.ENV,
                                stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        record(status="running", child_pid=proc.pid, started_unix=time.time(),
               admission_cpu_pct=cpu, admission_free_gib=free,
               admission_used_pct=used, admission_gpu_free_mib=gpu,
               production_data_before=before, source_sha256=source)
        fields = ("unix", "cpu_pct", "free_gib", "used_pct", "gpu_free_mib",
                  "extra_stage", "extra_mice", "extra_cuda", "formal_mice", "formal_cuda")
        min_free, max_used, min_gpu = free, used, gpu
        started = time.monotonic()
        with SAMPLES.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while proc.poll() is None:
                free, used = resource()
                gpu = gpu_free_mib()
                min_free, max_used, min_gpu = min(min_free, free), max(max_used, used), min(min_gpu, gpu)
                stage_file = target / "FAST_DATA_STATUS.json"
                stage = frozen.read_json_retry(stage_file) if stage_file.exists() else {}
                active = stage.get("active_pids", {})
                try:
                    controller, folds = production()
                    formal_mice = sum(f["mice"] for f in folds)
                    formal_cuda = sum(f["cuda"] for f in folds)
                    controller_ok = True
                except (OSError, ValueError, RuntimeError):
                    formal_mice = formal_cuda = 0
                    controller_ok = False
                extra_mice = sum(k.startswith("mice_") for k in active)
                extra_cuda = sum(k.startswith("selection_") for k in active)
                writer.writerow({"unix": time.time(), "cpu_pct": psutil.cpu_percent(interval=.2),
                                 "free_gib": round(free, 3), "used_pct": round(used, 3),
                                 "gpu_free_mib": gpu, "extra_stage": stage.get("stage", "starting"),
                                 "extra_mice": extra_mice, "extra_cuda": extra_cuda,
                                 "formal_mice": formal_mice, "formal_cuda": formal_cuda})
                stream.flush()
                if (target / "FAST_DATA_COMPLETE.json").exists():
                    time.sleep(1)
                    continue
                reason = None
                if not controller_ok:
                    reason = "Formal controller stopped"
                elif free < 2 or used >= 95 or gpu < 2500:
                    reason = "Resource protection threshold reached"
                elif formal_mice + extra_mice > 16:
                    reason = "Global MICE concurrency exceeded 16"
                elif formal_cuda + extra_cuda > 4:
                    reason = "Global CUDA concurrency exceeded 4"
                if reason:
                    abort(proc, target, reason, min_free, max_used, min_gpu)
                    return
                time.sleep(5)
    if proc.returncode:
        record(status="fold_failed", exit_code=proc.returncode,
               error_log=str(LOGS / "fold.err.log"),
               min_free_gib=min_free, max_used_pct=max_used,
               min_gpu_free_mib=min_gpu)
        return
    if frozen.fingerprint() != source or not frozen.check_fast(BRANCH, ORDINAL):
        raise RuntimeError("Completed extra fold failed frozen source/output hash audit")
    after = sum((frozen.fold_path(b, i) / "FAST_DATA_COMPLETE.json").exists()
                for i in range(1, 51) for b in frozen.BRANCHES)
    record(status="complete", elapsed_seconds=time.monotonic() - started,
           completed_unix=time.time(), production_data_before=before,
           production_data_after=after, min_free_gib=min_free,
           max_used_pct=max_used, min_gpu_free_mib=min_gpu,
           future_batch_may_reuse=True, target_complete_marker=str(target / "FAST_DATA_COMPLETE.json"),
           source_sha256=source)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record(status="failed", error=repr(exc))
        raise
