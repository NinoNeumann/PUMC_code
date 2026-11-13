"""Isolated branch-level CACox benchmark without stopping the production controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive80_20261107_v1")
DEST = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_branch_parallel_benchmark_20261107_v1")
PACKAGE = DEST / "package"
RUN_NAME = "run_fast_20261106_v1"
RUN = PACKAGE / RUN_NAME
ORIGINAL = SOURCE / RUN_NAME
BRANCHES = ("complete_case", "egfr_observed")
ORDINAL = 6
PYTHON = PROJECT / ".venv-gpu/Scripts/python.exe"
RSCRIPT = Path(r"C:\Users\NinoNeumann\AppData\Local\Programs\R\R-4.4.2\bin\Rscript.exe")
RLIB = PROJECT / ".r_library_selector_windows"
STATUS = DEST / "BRANCH_BENCHMARK_STATUS.json"


def fold(root: Path, branch: str) -> Path:
    return root / "branches" / branch / "repeat02_fold01"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(value: dict) -> None:
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATUS)


def environment() -> dict[str, str]:
    return dict(os.environ, CACOX_FAST_PACKAGE=str(PACKAGE), CACOX_PORTABLE_ROOT=str(PACKAGE),
                CACOX_RUN_ROOT=str(RUN), CACOX_R_LIBS=str(RLIB), CACOX_PYTHON=str(PYTHON),
                CACOX_RSCRIPT=str(RSCRIPT), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1")


def worker(branch: str) -> None:
    env = environment()
    commands = (
        ("data", [str(PYTHON), str(PROJECT / "experiments/cacox_sensitivity_fast_v1/benchmark_fast_data_3.py"),
                  branch, str(ORDINAL), "--run-name", RUN_NAME]),
        ("model", [str(PYTHON), str(PACKAGE / "code/cacox_fold.py"), branch,
                   str(ORDINAL), "--device", "cuda", "--threads", "1",
                   "--rscript", str(RSCRIPT)]),
    )
    for name, command in commands:
        with (DEST / f"{branch}_{name}.log").open("w", encoding="utf-8") as output:
            subprocess.run(command, cwd=PROJECT, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)


def compare(branch: str) -> dict:
    reference = fold(ORIGINAL, branch)
    produced = fold(RUN, branch)
    original_data = json.loads((reference / "FAST_DATA_COMPLETE.json").read_text())
    new_data = json.loads((produced / "FAST_DATA_COMPLETE.json").read_text())
    original_model = json.loads((reference / "cacox_es/COMPLETE.json").read_text())
    new_model = json.loads((produced / "cacox_es/COMPLETE.json").read_text())
    selection_hashes = [
        sha(reference / f"imputation_{mi:02d}/combined_selection.csv") ==
        sha(produced / f"imputation_{mi:02d}/combined_selection.csv")
        for mi in range(1, 21)
    ]
    original_predictions = reference / "cacox_es/predictions_all20.csv"
    new_predictions = produced / "cacox_es/predictions_all20.csv"
    old = pd.read_csv(original_predictions, float_precision="round_trip")
    new = pd.read_csv(new_predictions, float_precision="round_trip")
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        raise AssertionError(f"Prediction structure differs: {branch}")
    numeric = old.select_dtypes(include="number").columns
    numeric_diff = np.abs(old[numeric].to_numpy() - new[numeric].to_numpy())
    return {
        "branch": branch, "ordinal": ORDINAL, "new_data_complete": new_data["status"] == "complete",
        "new_model_complete": new_model["status"] == "complete",
        "reference_data_seconds": original_data["elapsed_seconds"],
        "reference_model_seconds": original_model["elapsed_this_invocation_seconds"],
        "new_data_seconds": new_data["elapsed_seconds"],
        "new_model_seconds": new_model["elapsed_this_invocation_seconds"],
        "selection_20_of_20_hash_equal": all(selection_hashes),
        "selection_hash_equal_count": sum(selection_hashes),
        "prediction_rows": len(new),
        "prediction_numeric_max_abs": float(numeric_diff.max()),
        "prediction_numeric_exact": bool(np.all(numeric_diff == 0)),
        "prediction_text_exact": old.drop(columns=numeric).equals(new.drop(columns=numeric)),
        "prediction_sha256_equal": sha(original_predictions) == sha(new_predictions),
    }


def gpu() -> tuple[float | None, float | None, float | None]:
    try:
        command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        utilization, used, total = (float(item.strip()) for item in result.stdout.splitlines()[0].split(",")[:3])
        return utilization, used / 1024, total / 1024
    except Exception:
        return None, None, None


def terminate_benchmark_tree(processes: dict[str, subprocess.Popen]) -> None:
    targets = []
    for process in processes.values():
        if psutil.pid_exists(process.pid):
            owner = psutil.Process(process.pid)
            targets.extend(owner.children(recursive=True))
            targets.append(owner)
    unique = {item.pid: item for item in targets}
    for item in reversed(list(unique.values())):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(unique.values()), timeout=15)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass


def parent() -> None:
    if DEST.exists():
        raise FileExistsError(f"Benchmark destination must be new: {DEST}")
    DEST.mkdir(parents=True)
    for name in ("code", "input", "vendor"):
        shutil.copytree(SOURCE / name, PACKAGE / name)
    shutil.copytree(ORIGINAL / "data", RUN / "data")
    baseline = {}
    for branch in BRANCHES:
        source_fold = fold(ORIGINAL, branch)
        baseline[branch] = {
            "data_marker_sha256": sha(source_fold / "FAST_DATA_COMPLETE.json"),
            "model_prediction_sha256": sha(source_fold / "cacox_es/predictions_all20.csv"),
        }
    status = {"status": "running", "started_unix": time.time(), "pid": os.getpid(),
              "source": str(SOURCE), "destination": str(DEST), "baseline": baseline,
              "branches": list(BRANCHES), "ordinal": ORDINAL, "workers": {}}
    save(status)
    processes = {}
    streams = {}
    try:
        status["stage"] = "waiting_for_production_model_window"
        save(status)
        wait_start = time.monotonic()
        production_status = (ORIGINAL / "FAST_CONTROLLER_STATUS.json")
        while True:
            production = json.loads(production_status.read_text(encoding="utf-8"))
            ram = psutil.virtual_memory().available / 2**30
            if production["status"] != "running" or not psutil.pid_exists(production["pid"]):
                raise RuntimeError("Production controller is not healthy while benchmark waits")
            if production["stage"].startswith("model_") and ram > 12:
                status["production_stage_at_launch"] = production["stage"]
                status["wait_production_model_seconds"] = time.monotonic()-wait_start
                save(status)
                break
            if time.monotonic()-wait_start > 1800:
                raise TimeoutError("No safe production model window within 30 minutes")
            time.sleep(5)
        start = time.monotonic()
        for branch in BRANCHES:
            streams[branch] = (DEST / f"{branch}_worker.log").open("w", encoding="utf-8")
            process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--worker", branch],
                                       cwd=PROJECT, env=environment(), stdout=streams[branch],
                                       stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            processes[branch] = process
            status["workers"][branch] = process.pid
            save(status)
        status["stage"] = "parallel_branch_pipelines"
        min_ram, peak_gpu, peak_gpu_util, samples = float("inf"), 0.0, 0.0, 0
        while any(proc.poll() is None for proc in processes.values()):
            ram = psutil.virtual_memory().available / 2**30
            gpu_util, used, total = gpu()
            min_ram = min(min_ram, ram)
            if used is not None:
                peak_gpu = max(peak_gpu, used)
                peak_gpu_util = max(peak_gpu_util, gpu_util)
            samples += 1
            status.update(elapsed_seconds=time.monotonic() - start, min_free_ram_gib=min_ram,
                          peak_gpu_used_gib=peak_gpu, peak_gpu_utilization_percent=peak_gpu_util,
                          samples=samples)
            save(status)
            if ram < 8 or (used is not None and total is not None and total-used < 1.5):
                raise RuntimeError(f"Benchmark resource guard: RAM={ram:.2f} GiB, GPU={used}/{total} GiB")
            time.sleep(10)
        exits = {branch: proc.wait() for branch, proc in processes.items()}
        if any(value != 0 for value in exits.values()):
            raise RuntimeError(f"Branch worker exits: {exits}")
        comparisons = {branch: compare(branch) for branch in BRANCHES}
        serial = sum(x["reference_data_seconds"] + x["reference_model_seconds"]
                     for x in comparisons.values())
        wall = time.monotonic() - start
        status.update(status="complete", wall_seconds=wall, reference_serial_seconds=serial,
                      observed_speedup=serial / wall, comparisons=comparisons,
                      finished_unix=time.time())
        save(status)
        if not all(x["selection_20_of_20_hash_equal"] and x["prediction_numeric_exact"]
                   and x["prediction_text_exact"] and x["prediction_sha256_equal"]
                   for x in comparisons.values()):
            raise AssertionError("Branch benchmark output did not match reference")
    except BaseException as error:
        terminate_benchmark_tree(processes)
        status.update(status="failed", error=repr(error), finished_unix=time.time())
        save(status)
        raise
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=BRANCHES)
    args = parser.parse_args()
    worker(args.worker) if args.worker else parent()
