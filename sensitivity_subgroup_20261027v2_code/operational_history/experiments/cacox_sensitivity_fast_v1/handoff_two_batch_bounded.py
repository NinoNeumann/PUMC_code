"""Resume two data batches with four internal workers after measured RAM pressure."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import time

import psutil

import full_controller as frozen
from handoff_data_first import audit_existing_outputs


OLD_CONTROLLER_PID = 13108
OLD_BATCH_PIDS = {1: 51312, 2: 5000}
AUDIT = frozen.RUN / "TWO_BATCH_BOUNDED_HANDOFF.json"
ARCHIVE = frozen.RUN / "handoffs/two_batch_bounded_20261107_v1"


def record(status: str, **fields: object) -> None:
    frozen.atomic_json(AUDIT, {"status": status, "updated_unix": time.time(), **fields})


def require_stopped(pid: int, name: str) -> None:
    try:
        cmd = " ".join(psutil.Process(pid).cmdline()).replace("\\", "/")
    except psutil.NoSuchProcess:
        return
    if name in cmd:
        raise RuntimeError(f"Old process still running: {pid} {name}")


def main() -> None:
    require_stopped(OLD_CONTROLLER_PID, "two_batch_controller.py")
    for batch, pid in OLD_BATCH_PIDS.items():
        require_stopped(pid, f"data_batch_worker.py {batch}")
    fold = frozen.fold_path("egfr_observed", 10)
    status_path = fold / "FAST_DATA_STATUS.json"
    marker = fold / "FAST_DATA_COMPLETE.json"
    record("waiting_for_in_flight_data", branch="egfr_observed", ordinal=10,
           reason="Two 6-worker CUDA selections reduced free RAM to 4.2 GiB")
    while not marker.is_file():
        status = frozen.read_json_retry(status_path)
        if status.get("status") == "failed":
            raise RuntimeError(f"In-flight fold failed: {status.get('error')}")
        if not psutil.pid_exists(status["pid"]):
            raise RuntimeError("In-flight fold process disappeared without completion marker")
        time.sleep(10)
    if not frozen.check_fast("egfr_observed", 10) or not frozen.check_fast("complete_case", 11):
        raise RuntimeError("In-flight or second-batch fold failed completion hash audit")
    data_count, model_count = audit_existing_outputs()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    locks = [(frozen.RUN / "FAST_CONTROLLER.lock", str(OLD_CONTROLLER_PID)),
             *[(frozen.RUN / f"DATA_BATCH_{batch:02d}.lock", str(pid))
               for batch, pid in OLD_BATCH_PIDS.items()]]
    for path, owner in locks:
        if not path.is_file() or path.read_text(encoding="utf-8").strip() != owner:
            raise RuntimeError(f"Stale lock owner mismatch: {path}")
    for path, _ in locks:
        path.replace(ARCHIVE / f"{path.name}.stopped")
    frozen.STATUS.replace(ARCHIVE / "FAST_CONTROLLER_STATUS.stopped.json")
    for batch in OLD_BATCH_PIDS:
        path = frozen.RUN / f"DATA_BATCH_{batch:02d}_STATUS.json"
        if path.is_file():
            path.replace(ARCHIVE / f"{path.name}.stopped")
    env = dict(os.environ, CACOX_FAST_PACKAGE=str(frozen.PACKAGE),
               PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    script = Path(__file__).with_name("two_batch_controller.py")
    out_path = frozen.RUN / "logs/two_batch_bounded_controller.out.log"
    err_path = frozen.RUN / "logs/two_batch_bounded_controller.err.log"
    with out_path.open("a", encoding="utf-8") as out, err_path.open("a", encoding="utf-8") as err:
        launcher = subprocess.Popen([str(frozen.PY), str(script)], cwd=frozen.PROJECT,
                                    env=env, stdout=out, stderr=err,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
    record("launched", launcher_pid=launcher.pid, preserved_data_folds=data_count,
           preserved_models=model_count, script=str(script),
           script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    for _ in range(30):
        if frozen.STATUS.is_file():
            status = frozen.read_json_retry(frozen.STATUS)
            pid = status.get("pid")
            if status.get("schedule") == "two_50_fold_batches" and pid and psutil.pid_exists(pid):
                cmd = " ".join(psutil.Process(pid).cmdline()).replace("\\", "/")
                if "experiments/cacox_sensitivity_fast_v1/two_batch_controller.py" not in cmd:
                    raise RuntimeError("New status PID belongs to another process")
                if status.get("memory_policy", {}).get("per_fold_concurrent_workers") != 4:
                    raise RuntimeError("New controller did not load four-worker resource policy")
                record("running", controller_pid=pid, launcher_pid=launcher.pid,
                       preserved_data_folds=data_count, preserved_models=model_count,
                       script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                       memory_policy=status["memory_policy"])
                return
        if launcher.poll() is not None:
            raise RuntimeError(f"New controller exited before status publication; see {err_path}")
        time.sleep(1)
    raise RuntimeError("New controller did not publish status in 30 seconds")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("failed", error=repr(exc))
        raise
