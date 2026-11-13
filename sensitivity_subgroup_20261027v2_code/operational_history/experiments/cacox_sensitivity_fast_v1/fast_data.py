"""Run one frozen CACox sensitivity data fold with 6 R MICE and 6 CUDA votes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PACKAGE = Path(os.environ.get(
    "CACOX_FAST_PACKAGE",
    PROJECT / "results/cacox_sensitivity_newdata_smoke/20261105_v1/cacox_es_sensitivity_portable_20261104",
)).resolve()
R = Path(r"C:\Users\NinoNeumann\AppData\Local\Programs\R\R-4.4.2\bin\Rscript.exe")
PY = PROJECT / ".venv-gpu/Scripts/python.exe"
LIB = PROJECT / ".r_library_selector_windows"
BRANCHES = ("complete_case", "egfr_observed", "renal_observed", "known_subtype", "no_egpa")
LOCK = threading.RLock()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint() -> dict[str, str]:
    paths = [*PACKAGE.joinpath("input").glob("*"),
             *PACKAGE.joinpath("code").glob("*.py"),
             *PACKAGE.joinpath("code").glob("*.R"),
             *PACKAGE.joinpath("vendor/r").glob("*.R"),
             *PACKAGE.joinpath("vendor/model").rglob("*.py")]
    return {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(paths) if p.is_file()}


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json_retry(path: Path) -> dict:
    for attempt in range(8):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 7:
                raise
            time.sleep(0.25)
    raise AssertionError("unreachable")


@contextmanager
def selection_mutex(run: Path):
    """Serialize entry to formal/extra CUDA selection across Windows processes."""
    path = run / "SELECTION_ADMISSION.lock"
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.2)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def formal_cuda_active(run: Path) -> bool:
    for batch in range(1, 6):
        status_path = run / f"DATA_BATCH_{batch:02d}_STATUS.json"
        if not status_path.exists():
            continue
        info = read_json_retry(status_path)
        if info.get("status") != "running" or info.get("current_branch") is None:
            continue
        ordinal = int(info["current_ordinal"])
        name = f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"
        other = run / "branches" / info["current_branch"] / name / "FAST_DATA_STATUS.json"
        if other.exists() and read_json_retry(other).get("stage") == "cuda_selection_6":
            return True
    return False


def output_files(fold: Path) -> list[Path]:
    paths = [fold / "prepared.rds", fold / "selection_frequency_29variables.csv",
             fold / "all_selection_results.csv"]
    paths += [fold / ("outer" if i == 0 else f"inner_{i:02d}") / "imputation.rds"
              for i in range(6)]
    for j in range(1, 21):
        q = fold / f"imputation_{j:02d}"
        paths += [q / name for name in ("boruta.rds", "lasso.rds", "combined_selection.csv")]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=BRANCHES)
    parser.add_argument("ordinal", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6,
                        help="Maximum concurrent R MICE and CUDA selection workers; all six partitions and 20 votes still run")
    parser.add_argument("--mice-workers", type=int, choices=range(1, 7),
                        help="Override concurrent MICE partitions without changing their count")
    parser.add_argument("--selection-workers", type=int, choices=range(1, 7),
                        help="Override concurrent selection workers without changing 20 votes")
    parser.add_argument("--resume-after-mice", action="store_true",
                        help="Resume only a reviewed fold after its saved, audited MICE")
    parser.add_argument("--opportunistic-4x4", action="store_true",
                        help="Wait for the formal batches to release CUDA before an extra fold selects variables")
    args = parser.parse_args()
    mice_workers = args.mice_workers or args.workers
    selection_workers = args.selection_workers or args.workers
    if args.ordinal not in range(1, 51) or not args.run_name.startswith("run_fast_") or \
       not args.run_name.replace("_", "").isalnum():
        parser.error("Use ordinal 1..50 and a run_fast_* directory name")
    reviewed_adaptive_folds = {("egfr_observed", 23), ("known_subtype", 24),
                               ("known_subtype", 35),
                               ("known_subtype", 50)}  # Audited six original MICE partitions; selection was interrupted.
    if args.resume_after_mice and (args.branch, args.ordinal) not in reviewed_adaptive_folds:
        parser.error("The reviewed after-MICE resume is not authorized for this fold")
    run = PACKAGE / args.run_name
    run.resolve().relative_to(PACKAGE.resolve())
    if not (run / "data/INITIALIZED.txt").is_file():
        raise RuntimeError("Initialize the new run root using the frozen package entry first")
    fold_name = f"repeat{(args.ordinal-1)//5+1:02d}_fold{(args.ordinal-1)%5+1:02d}"
    fold = run / "branches" / args.branch / fold_name
    fold.mkdir(parents=True, exist_ok=True)
    logs = run / "logs/fast_data" / args.branch / fold_name
    logs.mkdir(parents=True, exist_ok=True)
    marker = fold / "FAST_DATA_COMPLETE.json"
    source = fingerprint()
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["source_sha256"] != source or \
           any(not (fold / key).is_file() or sha(fold / key) != digest
               for key, digest in previous["output_sha256"].items()):
            raise RuntimeError("Previously completed fast data fold failed hash audit")
        print("FAST_DATA_REUSED", args.branch, args.ordinal, flush=True)
        return
    lock = fold / "FAST_DATA_RUNNING.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"Existing fold lock requires review: {lock}")
    env = dict(os.environ, CACOX_PORTABLE_ROOT=str(PACKAGE), CACOX_RUN_ROOT=str(run),
               CACOX_R_LIBS=str(LIB), CACOX_PYTHON=str(PY), CACOX_RSCRIPT=str(R),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    state = {"status": "running", "branch": args.branch, "ordinal": args.ordinal,
             "pid": os.getpid(), "started_unix": time.time(), "stage": "starting",
             "active_pids": {}, "timing_seconds": {}, "source_sha256": source}
    status_file = fold / "FAST_DATA_STATUS.json"

    def save() -> None:
        with LOCK:
            atomic_json(status_file, state)

    def execute(name: str, command: list[str]) -> dict:
        if psutil.virtual_memory().available < 5 * 2**30:
            raise RuntimeError("Available system RAM below 5 GiB before launching a data worker")
        start = time.monotonic()
        with (logs / f"{name}.out.log").open("a", encoding="utf-8") as out, \
             (logs / f"{name}.err.log").open("a", encoding="utf-8") as err:
            proc = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=out, stderr=err)
            with LOCK:
                state["active_pids"][name] = proc.pid
                save()
            code = proc.wait()
        result = {"stage": name, "exit_code": code, "seconds": time.monotonic() - start}
        with LOCK:
            state["active_pids"].pop(name, None)
            state["timing_seconds"][name] = result["seconds"]
            save()
        if code != 0:
            raise RuntimeError(f"{name} exited {code}; see {logs / (name + '.err.log')}")
        return result

    try:
        save()
        if args.resume_after_mice:
            if not (fold / "prepared.rds").is_file() or any(
                not (fold / ("outer" if i == 0 else f"inner_{i:02d}") / "imputation.rds").is_file()
                for i in range(6)
            ):
                raise RuntimeError("Reviewed MICE resume lacks a prepared fold or partition")
            state["stage"] = "reviewed_adaptive_mice_resume"; save()
        else:
            state["stage"] = "prepare"; save()
            execute("prepare", [str(R), "--vanilla", str(PACKAGE / "code/data_stage.R"),
                                args.branch, str(args.ordinal), "prepare"])
            state["stage"] = "mice_6"; save()
            with ThreadPoolExecutor(max_workers=mice_workers) as pool:
                futures = [pool.submit(execute, f"mice_{i:02d}",
                           [str(R), "--vanilla", str(HERE / "mice_worker.R"),
                            args.branch, str(args.ordinal), str(i)]) for i in range(6)]
                for future in futures: future.result()
        state["stage"] = "mice_quality_gate"; save()
        execute("mice_quality_gate", [str(R), "--vanilla", str(HERE / "verify_data.R"),
                                      args.branch, str(args.ordinal), "mice"])
        claim = run / "OPPORTUNISTIC_CUDA_ACTIVE.json"
        state["stage"] = "waiting_for_cuda_slot"; save()
        while True:
            with selection_mutex(run):
                if claim.exists():
                    owner = read_json_retry(claim).get("pid")
                    if owner is None or not psutil.pid_exists(int(owner)):
                        raise RuntimeError("Stale opportunistic CUDA claim requires review")
                    ready = False
                elif args.opportunistic_4x4:
                    controller = read_json_retry(run / "FAST_CONTROLLER_STATUS.json")
                    if controller.get("status") != "running":
                        raise RuntimeError("Formal controller stopped before extra CUDA selection")
                    ready = (psutil.virtual_memory().available >= 8 * 2**30 and
                             not formal_cuda_active(run))
                    if ready:
                        atomic_json(claim, {"pid": os.getpid(), "branch": args.branch,
                                            "ordinal": args.ordinal, "started_unix": time.time()})
                else:
                    ready = True
                if ready:
                    state["stage"] = "cuda_selection_6"; save()
                    break
            time.sleep(5)
        try:
            if args.branch == "complete_case":
                # selection 1 creates the shared deterministic complete_case_lasso.rds.
                execute("selection_shared_lasso_vote_01", [str(R), "--vanilla",
                        str(HERE / "select_worker.R"), args.branch, str(args.ordinal), "0", "1"])
                votes = list(range(2, 21))
            else:
                votes = list(range(1, 21))
            chunks = [votes[k::selection_workers] for k in range(selection_workers)]
            with ThreadPoolExecutor(max_workers=selection_workers) as pool:
                futures = [pool.submit(execute, f"selection_worker_{k:02d}",
                           [str(R), "--vanilla", str(HERE / "select_worker.R"),
                            args.branch, str(args.ordinal), str(k + 1), *map(str, chunk)])
                           for k, chunk in enumerate(chunks) if chunk]
                for future in futures: future.result()
        finally:
            if args.opportunistic_4x4:
                with selection_mutex(run):
                    if claim.exists() and read_json_retry(claim).get("pid") == os.getpid():
                        claim.unlink()
        state["stage"] = "summary"; save()
        execute("summary", [str(R), "--vanilla", str(PACKAGE / "code/data_stage.R"),
                            args.branch, str(args.ordinal), "summary"])
        state["stage"] = "final_data_audit"; save()
        execute("final_data_audit", [str(R), "--vanilla", str(HERE / "verify_data.R"),
                                     args.branch, str(args.ordinal), "final"])
        if fingerprint() != source:
            raise RuntimeError("Frozen package input or source changed during the fold")
        files = output_files(fold)
        if not all(path.is_file() for path in files):
            raise RuntimeError("A required data or screening artifact is missing")
        record = {"status": "complete", "branch": args.branch, "ordinal": args.ordinal,
                  "source_sha256": source,
                  "output_sha256": {str(path.relative_to(fold)): sha(path) for path in files},
                  "timing_seconds": state["timing_seconds"],
                  "elapsed_seconds": time.time() - state["started_unix"]}
        atomic_json(marker, record)
        state["status"] = "complete"; state["stage"] = "complete"; save()
        print("FAST_DATA_COMPLETE", args.branch, args.ordinal,
              round(record["elapsed_seconds"], 3), flush=True)
    except BaseException as exc:
        state["status"] = "failed"; state["error"] = repr(exc); save()
        raise
    finally:
        if lock.exists() and lock.read_text(encoding="utf-8") == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
