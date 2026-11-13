"""Isolated two-fold model-concurrency benchmark; never writes to the active run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import psutil


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive80_20261107_v1")
DEST = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_parallel_model_benchmark_20261107_v1")
PACKAGE = DEST / "package"
RUN_NAME = "run_fast_20261106_v1"
ORIGINAL = SOURCE / RUN_NAME
RUN = PACKAGE / RUN_NAME
FOLDS = (("egfr_observed", 6), ("renal_observed", 6))
PYTHON = PROJECT / ".venv-gpu/Scripts/python.exe"
RSCRIPT = Path(r"C:\Users\NinoNeumann\AppData\Local\Programs\R\R-4.4.2\bin\Rscript.exe")
RLIB = PROJECT / ".r_library_selector_windows"
STATUS = DEST / "BENCHMARK_STATUS.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(record: dict) -> None:
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def location(branch: str, ordinal: int) -> Path:
    return Path("branches") / branch / f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"


def clone() -> dict:
    if DEST.exists():
        raise FileExistsError(f"Benchmark directory must be new: {DEST}")
    DEST.mkdir(parents=True)
    for name in ("code", "input", "vendor"):
        shutil.copytree(SOURCE / name, PACKAGE / name)
    shutil.copytree(ORIGINAL / "data", RUN / "data")
    baseline = {}
    for branch, ordinal in FOLDS:
        relative = location(branch, ordinal)
        src = ORIGINAL / relative
        if not (src / "FAST_DATA_COMPLETE.json").is_file():
            raise FileNotFoundError(src / "FAST_DATA_COMPLETE.json")
        report = json.loads((src / "cacox_es/COMPLETE.json").read_text(encoding="utf-8"))
        baseline[branch] = {
            "ordinal": ordinal,
            "serial_seconds": report["elapsed_this_invocation_seconds"],
            "prediction_sha256": sha(src / "cacox_es/predictions_all20.csv"),
            "data_marker_sha256": sha(src / "FAST_DATA_COMPLETE.json"),
        }
        # The model directory is deliberately absent: both copied folds retrain fully.
        shutil.copytree(src, RUN / relative,
                        ignore=shutil.ignore_patterns("cacox_es", "cuda_server_worker_*", "neural_inputs"))
        if sha(RUN / relative / "FAST_DATA_COMPLETE.json") != baseline[branch]["data_marker_sha256"]:
            raise RuntimeError(f"Copied data marker changed for {branch}")
    return baseline


def gpu() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        used, total = (float(x.strip()) for x in result.stdout.splitlines()[0].split(",")[:2])
        return used / 1024, total / 1024
    except Exception:
        return None, None


def compare(branch: str, ordinal: int) -> dict:
    relative = location(branch, ordinal) / "cacox_es"
    new = RUN / relative
    old = ORIGINAL / relative
    new_report = json.loads((new / "COMPLETE.json").read_text(encoding="utf-8"))
    a = pd.read_csv(old / "predictions_all20.csv", float_precision="round_trip")
    b = pd.read_csv(new / "predictions_all20.csv", float_precision="round_trip")
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise AssertionError(f"Prediction table structure changed: {branch}")
    numeric = a.select_dtypes(include="number").columns
    exact = (a[numeric].to_numpy() == b[numeric].to_numpy()).all()
    max_abs = float(abs(a[numeric].to_numpy() - b[numeric].to_numpy()).max())
    text_equal = a.drop(columns=numeric).equals(b.drop(columns=numeric))
    return {
        "complete": new_report["status"] == "complete" and new_report["completed_stages"] == 12,
        "rows": len(b), "numeric_exact": bool(exact), "numeric_max_abs": max_abs,
        "text_exact": bool(text_equal),
        "prediction_sha256_equal": sha(new / "predictions_all20.csv") == sha(old / "predictions_all20.csv"),
        "new_elapsed_seconds": new_report["elapsed_this_invocation_seconds"],
    }


def main() -> None:
    baseline = clone()
    record = {"status": "running", "source": str(SOURCE), "destination": str(DEST),
              "folds": baseline, "started_unix": time.time(), "pid": os.getpid(), "workers": {}}
    save(record)
    env = dict(os.environ, CACOX_PORTABLE_ROOT=str(PACKAGE), CACOX_RUN_ROOT=str(RUN),
               CACOX_R_LIBS=str(RLIB), CACOX_PYTHON=str(PYTHON), CACOX_RSCRIPT=str(RSCRIPT),
               PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    processes = {}
    logs = {}
    try:
        # Run the one-off R export and encoding in sequence. The measured concurrent
        # interval below then isolates model training rather than simultaneous MICE
        # loading, which has a much larger transient RAM peak.
        preparation_start = time.monotonic()
        for branch, ordinal in FOLDS:
            record["stage"] = f"preparing_{branch}"
            save(record)
            prep_log = DEST / f"{branch}_prepare.log"
            command = [str(PYTHON), str(PACKAGE / "code/cacox_fold.py"), branch, str(ordinal),
                       "--device", "cuda", "--threads", "1", "--rscript", str(RSCRIPT),
                       "--prepare-only"]
            with prep_log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=PROJECT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if psutil.virtual_memory().available / 2**30 < 8:
                raise RuntimeError("Less than 8 GiB RAM available after sequential preparation")
        record["preparation_seconds"] = time.monotonic() - preparation_start
        record["stage"] = "parallel_models"
        save(record)
        start = time.monotonic()
        for branch, ordinal in FOLDS:
            log_path = DEST / f"{branch}_model.log"
            logs[branch] = log_path.open("w", encoding="utf-8")
            command = [str(PYTHON), str(PACKAGE / "code/cacox_fold.py"), branch, str(ordinal),
                       "--device", "cuda", "--threads", "1", "--rscript", str(RSCRIPT)]
            proc = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=logs[branch],
                                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            processes[branch] = proc
            record["workers"][branch] = proc.pid
            save(record)
        peak_gpu = 0.0
        min_ram = float("inf")
        samples = 0
        while any(proc.poll() is None for proc in processes.values()):
            ram = psutil.virtual_memory().available / 2**30
            used, total = gpu()
            min_ram = min(min_ram, ram)
            if used is not None:
                peak_gpu = max(peak_gpu, used)
            samples += 1
            record.update(elapsed_seconds=time.monotonic() - start, min_free_ram_gib=min_ram,
                          peak_total_gpu_gib=peak_gpu, samples=samples)
            save(record)
            # Abort only benchmark workers before they can starve the production run.
            if ram < 8 or (used is not None and total is not None and total - used < 1.5):
                raise RuntimeError(f"Benchmark resource guard: free RAM={ram:.2f} GiB, GPU={used}/{total} GiB")
            time.sleep(10)
        codes = {branch: proc.wait() for branch, proc in processes.items()}
        if any(code != 0 for code in codes.values()):
            raise RuntimeError(f"Benchmark model process exit codes: {codes}")
        elapsed = time.monotonic() - start
        comparisons = {branch: compare(branch, ordinal) for branch, ordinal in FOLDS}
        serial = sum(item["serial_seconds"] for item in baseline.values())
        record.update(status="complete", wall_seconds=elapsed, reference_serial_seconds=serial,
                      throughput_speedup_model_interval=serial / elapsed,
                      total_wall_seconds=record["preparation_seconds"] + elapsed,
                      comparisons=comparisons,
                      finished_unix=time.time())
        save(record)
        if not all(x["complete"] and x["numeric_exact"] and x["text_exact"]
                   for x in comparisons.values()):
            raise AssertionError("Benchmark output differs from production reference")
    except BaseException as error:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        record.update(status="failed", error=repr(error), finished_unix=time.time())
        save(record)
        raise
    finally:
        for stream in logs.values():
            stream.close()


if __name__ == "__main__":
    main()
