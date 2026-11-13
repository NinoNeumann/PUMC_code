"""Wait for the orphaned fold, audit preserved work, then start two data batches."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import time

import psutil

import full_controller as frozen
from handoff_data_first import audit_existing_outputs


OLD_PID = 20536
IN_FLIGHT_BRANCH = "no_egpa"
IN_FLIGHT_ORDINAL = 9
HANDOFF = frozen.RUN / "TWO_BATCH_HANDOFF.json"
ARCHIVE = frozen.RUN / "handoffs/two_batch_20261107_v1"


def record(status: str, **fields: object) -> None:
    frozen.atomic_json(HANDOFF, {"status": status, "updated_unix": time.time(), **fields})


def main() -> None:
    try:
        command = " ".join(psutil.Process(OLD_PID).cmdline()).replace("\\", "/")
    except psutil.NoSuchProcess:
        command = ""
    if "experiments/cacox_sensitivity_fast_v1/data_first_controller.py" in command:
        raise RuntimeError("Old data-first controller is still running")
    fold = frozen.fold_path(IN_FLIGHT_BRANCH, IN_FLIGHT_ORDINAL)
    status_path = fold / "FAST_DATA_STATUS.json"
    marker = fold / "FAST_DATA_COMPLETE.json"
    record("waiting_for_in_flight_data", branch=IN_FLIGHT_BRANCH,
           ordinal=IN_FLIGHT_ORDINAL, stopped_controller_pid=OLD_PID)
    while not marker.is_file():
        status = frozen.read_json_retry(status_path)
        if status.get("status") == "failed":
            raise RuntimeError(f"In-flight data failed: {status.get('error')}")
        if not psutil.pid_exists(status["pid"]):
            raise RuntimeError("In-flight data process disappeared without completion marker")
        time.sleep(10)
    if not frozen.check_fast(IN_FLIGHT_BRANCH, IN_FLIGHT_ORDINAL):
        raise RuntimeError("In-flight data completion hash audit failed")
    data_count, model_count = audit_existing_outputs()
    lock = frozen.RUN / "FAST_CONTROLLER.lock"
    if not lock.is_file() or lock.read_text(encoding="utf-8").strip() != str(OLD_PID):
        raise RuntimeError("Stopped controller lock changed; refusing handoff")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    lock.replace(ARCHIVE / "FAST_CONTROLLER.lock.stopped")
    frozen.STATUS.replace(ARCHIVE / "FAST_CONTROLLER_STATUS.stopped.json")
    env = dict(os.environ, CACOX_FAST_PACKAGE=str(frozen.PACKAGE),
               PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    script = Path(__file__).with_name("two_batch_controller.py")
    out_path = frozen.RUN / "logs/two_batch_controller.out.log"
    err_path = frozen.RUN / "logs/two_batch_controller.err.log"
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
            actual_pid = status.get("pid")
            if status.get("schedule") == "two_50_fold_batches" and actual_pid and \
               psutil.pid_exists(actual_pid):
                cmd = " ".join(psutil.Process(actual_pid).cmdline()).replace("\\", "/")
                if "experiments/cacox_sensitivity_fast_v1/two_batch_controller.py" not in cmd:
                    raise RuntimeError("New status PID does not belong to two-batch controller")
                record("running", controller_pid=actual_pid, launcher_pid=launcher.pid,
                       preserved_data_folds=data_count, preserved_models=model_count,
                       script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                       memory_policy=status["memory_policy"])
                return
        if launcher.poll() is not None:
            raise RuntimeError(f"Two-batch controller exited before publishing status; see {err_path}")
        time.sleep(1)
    raise RuntimeError("Two-batch controller did not publish status within 30 seconds")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("failed", error=repr(exc))
        raise
