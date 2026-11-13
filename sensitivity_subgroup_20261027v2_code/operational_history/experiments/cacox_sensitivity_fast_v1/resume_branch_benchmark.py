"""Finish an isolated branch benchmark after its resource guard, preserving production."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import psutil

from benchmark_branch_parallel import DEST, PACKAGE, PROJECT, PYTHON, RLIB, RSCRIPT, RUN, compare


STATUS = DEST / "BRANCH_MODEL_RESUME_STATUS.json"
PRODUCTION = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive80_20261107_v1\run_fast_20261106_v1\FAST_CONTROLLER_STATUS.json")


def save(value: dict) -> None:
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def terminate_tree(proc: subprocess.Popen) -> None:
    if not psutil.pid_exists(proc.pid):
        return
    owner = psutil.Process(proc.pid)
    children = owner.children(recursive=True)
    for item in reversed(children):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        owner.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs([*children, owner], timeout=15)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass


def main() -> None:
    model = RUN / "branches/egfr_observed/repeat02_fold01/cacox_es"
    if (model / "RUNNING.lock").exists() or (model / "COMPLETE.json").exists():
        raise RuntimeError("Expected an incomplete benchmark model without a live lock")
    state = {"status": "waiting_for_production_model", "started_unix": time.time(),
             "pid": psutil.Process().pid, "production_stage_required": "model_egfr_observed_08"}
    save(state)
    wait_start = time.monotonic()
    while True:
        production = json.loads(PRODUCTION.read_text(encoding="utf-8"))
        if production["status"] != "running" or not psutil.pid_exists(production["pid"]):
            raise RuntimeError("Production controller stopped while benchmark waited")
        if production["stage"] == "model_egfr_observed_08" and psutil.virtual_memory().available > 12*2**30:
            break
        if time.monotonic()-wait_start > 3600:
            raise TimeoutError("No suitable production model window within 60 minutes")
        time.sleep(5)
    state["wait_seconds"] = time.monotonic()-wait_start
    state["status"] = "running_model"
    save(state)
    env = dict(__import__("os").environ, CACOX_PORTABLE_ROOT=str(PACKAGE),
               CACOX_RUN_ROOT=str(RUN), CACOX_R_LIBS=str(RLIB), CACOX_PYTHON=str(PYTHON),
               CACOX_RSCRIPT=str(RSCRIPT), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(PYTHON), str(PACKAGE / "code/cacox_fold.py"), "egfr_observed", "6",
               "--device", "cuda", "--threads", "1", "--rscript", str(RSCRIPT)]
    start = time.monotonic()
    with (DEST / "egfr_observed_model_resume.log").open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        state["worker_pid"] = proc.pid
        save(state)
        try:
            minimum = float("inf")
            while proc.poll() is None:
                ram = psutil.virtual_memory().available / 2**30
                minimum = min(minimum, ram)
                state.update(elapsed_seconds=time.monotonic()-start, min_free_ram_gib=minimum)
                save(state)
                if ram < 7:
                    raise RuntimeError(f"Benchmark resume resource guard: free RAM {ram:.2f} GiB")
                time.sleep(10)
            if proc.wait() != 0:
                raise RuntimeError(f"Benchmark model exited {proc.returncode}")
            result = compare("egfr_observed")
            if not (result["selection_20_of_20_hash_equal"] and result["prediction_sha256_equal"]):
                raise AssertionError("Completed benchmark differs from frozen reference")
            state.update(status="complete", active_seconds=time.monotonic()-start,
                         comparison=result, finished_unix=time.time())
            save(state)
        except BaseException as error:
            terminate_tree(proc)
            state.update(status="failed", error=repr(error), finished_unix=time.time())
            save(state)
            raise


if __name__ == "__main__":
    main()
