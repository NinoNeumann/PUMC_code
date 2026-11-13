"""Resume two data batches with six MICE and four selection workers per fold."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import time

import psutil

import full_controller as frozen
from data_batch_worker import MICE_WORKERS, SELECTION_WORKERS
from handoff_data_first import audit_existing_outputs


OLD_CONTROLLER = 9964
OLD_BATCHES = {1: 18800, 2: 39200}
IN_FLIGHT = (("known_subtype", 10), ("renal_observed", 11))
AUDIT = frozen.RUN / "TWO_BATCH_PHASE_SPLIT_HANDOFF.json"
ARCHIVE = frozen.RUN / "handoffs/two_batch_phase_split_20261107_v1"
REASON = "Split phase concurrency to raise MICE CPU use without changing data outputs"


def record(status: str, **fields: object) -> None:
    frozen.atomic_json(AUDIT, {"status": status, "updated_unix": time.time(), **fields})


def require_stopped(pid: int, script: str) -> None:
    try:
        command = " ".join(psutil.Process(pid).cmdline()).replace("\\", "/")
    except psutil.NoSuchProcess:
        return
    if script in command:
        raise RuntimeError(f"Previous process is still active: {pid} {script}")


def wait_for_fold(branch: str, ordinal: int) -> None:
    fold = frozen.fold_path(branch, ordinal)
    marker = fold / "FAST_DATA_COMPLETE.json"
    status_path = fold / "FAST_DATA_STATUS.json"
    while not marker.is_file():
        status = frozen.read_json_retry(status_path)
        if status.get("status") == "failed":
            raise RuntimeError(f"In-flight fold failed: {branch} {ordinal}: {status.get('error')}")
        if not psutil.pid_exists(status["pid"]):
            raise RuntimeError(f"In-flight fold disappeared: {branch} {ordinal}")
        time.sleep(10)
    if not frozen.check_fast(branch, ordinal):
        raise RuntimeError(f"In-flight fold output hash audit failed: {branch} {ordinal}")


def main() -> None:
    require_stopped(OLD_CONTROLLER, "two_batch_controller.py")
    for batch, pid in OLD_BATCHES.items():
        require_stopped(pid, f"data_batch_worker.py {batch}")
    record("waiting_for_in_flight_folds", folds=list(IN_FLIGHT), reason=REASON)
    for branch, ordinal in IN_FLIGHT:
        wait_for_fold(branch, ordinal)
    data_count, model_count = audit_existing_outputs()
    locks = [(frozen.RUN / "FAST_CONTROLLER.lock", str(OLD_CONTROLLER)),
             *[(frozen.RUN / f"DATA_BATCH_{batch:02d}.lock", str(pid))
               for batch, pid in OLD_BATCHES.items()]]
    for path, owner in locks:
        if not path.is_file() or path.read_text(encoding="utf-8").strip() != owner:
            raise RuntimeError(f"Stopped process lock mismatch: {path}")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for path, _ in locks:
        path.replace(ARCHIVE / f"{path.name}.stopped")
    frozen.STATUS.replace(ARCHIVE / "FAST_CONTROLLER_STATUS.stopped.json")
    for batch in OLD_BATCHES:
        path = frozen.RUN / f"DATA_BATCH_{batch:02d}_STATUS.json"
        if path.is_file():
            path.replace(ARCHIVE / f"{path.name}.stopped")
    script = Path(__file__).with_name("two_batch_controller.py")
    env = dict(os.environ, CACOX_FAST_PACKAGE=str(frozen.PACKAGE),
               PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    out_path = frozen.RUN / "logs/two_batch_phase_split_controller.out.log"
    err_path = frozen.RUN / "logs/two_batch_phase_split_controller.err.log"
    with out_path.open("a", encoding="utf-8") as out, err_path.open("a", encoding="utf-8") as err:
        launcher = subprocess.Popen([str(frozen.PY), str(script)], cwd=frozen.PROJECT,
                                    env=env, stdout=out, stderr=err,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
    record("launched", launcher_pid=launcher.pid, preserved_data_folds=data_count,
           preserved_models=model_count, script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    for _ in range(30):
        if frozen.STATUS.is_file():
            state = frozen.read_json_retry(frozen.STATUS)
            pid = state.get("pid")
            if state.get("schedule") == "two_50_fold_batches" and pid and psutil.pid_exists(pid):
                cmd = " ".join(psutil.Process(pid).cmdline()).replace("\\", "/")
                if "experiments/cacox_sensitivity_fast_v1/two_batch_controller.py" not in cmd:
                    raise RuntimeError("Published PID does not match two-batch controller")
                policy = state.get("memory_policy", {})
                if policy.get("per_fold_concurrent_mice_workers") != MICE_WORKERS or \
                   policy.get("per_fold_concurrent_selection_workers") != SELECTION_WORKERS:
                    raise RuntimeError("Phase-specific concurrency policy did not load")
                record("running", controller_pid=pid, launcher_pid=launcher.pid,
                       preserved_data_folds=data_count, preserved_models=model_count,
                       script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                       memory_policy=policy)
                return
        if launcher.poll() is not None:
            raise RuntimeError(f"New controller exited before status; see {err_path}")
        time.sleep(1)
    raise RuntimeError("New controller did not publish status in 30 seconds")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("failed", error=repr(exc))
        raise
