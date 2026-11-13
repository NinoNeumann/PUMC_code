"""Safely hand an in-flight data fold from the stopped controller to data-first."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

import full_controller as frozen


OLD_CONTROLLER_PID = 1704
IN_FLIGHT_BRANCH = "no_egpa"
IN_FLIGHT_ORDINAL = 8
AUDIT = frozen.RUN / "DATA_FIRST_HANDOFF.json"
HANDOFF_DIR = frozen.RUN / "handoffs/data_first_20261107_v1"


def record(status: str, **details: object) -> None:
    frozen.atomic_json(AUDIT, {"status": status, "updated_unix": time.time(), **details})


def stopped_controller() -> bool:
    try:
        cmd = " ".join(psutil.Process(OLD_CONTROLLER_PID).cmdline())
    except psutil.NoSuchProcess:
        return True
    return "experiments/cacox_sensitivity_fast_v1/full_controller.py" not in cmd.replace("\\", "/")


def wait_for_in_flight_data() -> None:
    fold = frozen.fold_path(IN_FLIGHT_BRANCH, IN_FLIGHT_ORDINAL)
    marker = fold / "FAST_DATA_COMPLETE.json"
    status_file = fold / "FAST_DATA_STATUS.json"
    record("waiting_for_in_flight_data", branch=IN_FLIGHT_BRANCH, ordinal=IN_FLIGHT_ORDINAL)
    while not marker.is_file():
        if status_file.is_file():
            try:
                status = frozen.read_json_retry(status_file)
            except (json.JSONDecodeError, PermissionError):
                time.sleep(2)
                continue
            if status.get("status") == "failed":
                raise RuntimeError(f"In-flight data worker failed: {status.get('error')}")
            pid = status.get("pid")
            if pid and not psutil.pid_exists(pid):
                raise RuntimeError("In-flight data worker disappeared without completion marker")
        time.sleep(10)
    if not frozen.check_fast(IN_FLIGHT_BRANCH, IN_FLIGHT_ORDINAL):
        raise RuntimeError("In-flight data fold did not pass complete-marker hash audit")


def audit_existing_outputs() -> tuple[int, int]:
    data_count = model_count = 0
    for ordinal in range(1, 51):
        for branch in frozen.BRANCHES:
            fold = frozen.fold_path(branch, ordinal)
            if (fold / "FAST_DATA_COMPLETE.json").is_file():
                if not frozen.check_fast(branch, ordinal):
                    raise RuntimeError(f"Data marker failed audit: {branch} {ordinal}")
                data_count += 1
            if (fold / "cacox_es/COMPLETE.json").is_file():
                if not frozen.check_model(branch, ordinal):
                    raise RuntimeError(f"Model marker failed audit: {branch} {ordinal}")
                model_count += 1
    if data_count < model_count:
        raise RuntimeError("More complete models than complete data folds")
    return data_count, model_count


def main() -> None:
    if not stopped_controller():
        raise RuntimeError("Old controller is still running; handoff refused")
    wait_for_in_flight_data()
    data_count, model_count = audit_existing_outputs()
    lock = frozen.RUN / "FAST_CONTROLLER.lock"
    if not lock.is_file() or lock.read_text(encoding="utf-8").strip() != str(OLD_CONTROLLER_PID):
        raise RuntimeError("Stale controller lock content changed; handoff refused")
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    lock.replace(HANDOFF_DIR / "FAST_CONTROLLER.lock.stopped")
    frozen.STATUS.replace(HANDOFF_DIR / "FAST_CONTROLLER_STATUS.stopped.json")

    out_path = frozen.RUN / "logs/data_first_controller.out.log"
    err_path = frozen.RUN / "logs/data_first_controller.err.log"
    env = dict(os.environ, CACOX_FAST_PACKAGE=str(frozen.PACKAGE),
               PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    script = Path(__file__).with_name("data_first_controller.py")
    with out_path.open("a", encoding="utf-8") as out, err_path.open("a", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [str(frozen.PY), str(script)], cwd=frozen.PROJECT, env=env,
            stdout=out, stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    record("launched", controller_pid=proc.pid, preserved_data_folds=data_count,
           preserved_models=model_count, script=str(script),
           script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
           stopped_controller_pid=OLD_CONTROLLER_PID,
           memory_policy={"max_used_pct_exclusive": 95.0,
                          "min_available_gib_inclusive": 5.0})
    time.sleep(5)
    if proc.poll() is not None:
        raise RuntimeError(f"Data-first controller exited immediately: {proc.returncode}; see {err_path}")
    status = frozen.read_json_retry(frozen.STATUS)
    actual_pid = status.get("pid")
    if status.get("schedule") != "data_first" or not actual_pid or not psutil.pid_exists(actual_pid):
        raise RuntimeError("Data-first controller did not publish matching status")
    actual_command = " ".join(psutil.Process(actual_pid).cmdline()).replace("\\", "/")
    if "experiments/cacox_sensitivity_fast_v1/data_first_controller.py" not in actual_command:
        raise RuntimeError("Published PID does not belong to the data-first controller")
    record("running", controller_pid=actual_pid, launcher_pid=proc.pid,
           preserved_data_folds=data_count,
           preserved_models=model_count, script=str(script),
           script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
           stopped_controller_pid=OLD_CONTROLLER_PID,
           memory_policy=status["memory_policy"])


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("failed", error=repr(exc))
        raise
