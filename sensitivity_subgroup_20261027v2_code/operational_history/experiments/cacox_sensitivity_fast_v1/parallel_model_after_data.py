"""Add one guarded model worker alongside the formal serial model phase.

Run only after all 250 data folds are audited. The formal controller keeps
ownership of its normal model pass; this worker trains distant folds so the
two controllers cannot write the same fold concurrently.
"""
from __future__ import annotations

import os
import subprocess
import time

import psutil

import full_controller as frozen
from overlap_model_during_data import resources, stop_tree


STATUS = frozen.RUN / os.environ.get("CACOX_EXTRA_MODEL_STATUS", "PARALLEL_MODEL_STATUS.json")
LOCK = frozen.RUN / os.environ.get("CACOX_EXTRA_MODEL_LOCK", "PARALLEL_MODEL.lock")
LOGS = frozen.RUN / "logs" / os.environ.get("CACOX_EXTRA_MODEL_LOG_DIR", "parallel_models")
DATA_DONE = frozen.RUN / "DATA_FIRST_DATA_COMPLETE.json"
OVERLAP_LOCK = frozen.RUN / "OVERLAP_MODEL.lock"
PREVIOUS_LOCK = frozen.RUN / os.environ.get("CACOX_EXTRA_MODEL_PREVIOUS_LOCK", "OVERLAP_MODEL.lock")
PREVIOUS_STATUS = frozen.RUN / os.environ.get("CACOX_EXTRA_MODEL_PREVIOUS_STATUS", "OVERLAP_MODEL_STATUS.json")
MAX_ORDINAL = int(os.environ.get("CACOX_EXTRA_MODEL_MAX_ORDINAL", "50"))
MIN_ORDINAL = int(os.environ.get("CACOX_EXTRA_MODEL_MIN_ORDINAL", "41"))
ONLY_BRANCH = os.environ.get("CACOX_EXTRA_MODEL_ONLY_BRANCH")
FORMAL_ORDINAL_GAP = int(os.environ.get("CACOX_EXTRA_MODEL_FORMAL_ORDINAL_GAP", "2"))


def formal_model_pid() -> int | None:
    if not DATA_DONE.is_file():
        return None
    formal = frozen.read_json_retry(frozen.STATUS)
    if formal.get("status") != "running" or not str(formal.get("stage", "")).startswith("model_"):
        return None
    pid = formal.get("active_pid")
    return pid if isinstance(pid, int) and psutil.pid_exists(pid) else None


def candidates():
    # The formal pass starts near ordinal 8 and proceeds in ascending order.
    # These verified late folds are disjoint for the duration of this worker.
    for ordinal in range(MAX_ORDINAL, MIN_ORDINAL - 1, -1):
        for branch in reversed(frozen.BRANCHES):
            if ONLY_BRANCH is None or branch == ONLY_BRANCH:
                yield branch, ordinal


def formal_ordinal() -> int | None:
    formal = frozen.read_json_retry(frozen.STATUS)
    stage = str(formal.get("stage", ""))
    if not stage.startswith("model_"):
        return None
    try:
        return int(stage.rsplit("_", 1)[1])
    except ValueError:
        return None


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {"status": "waiting_for_data", "pid": os.getpid(),
             "started_unix": time.time(), "active_pid": None, "completed": [],
             "candidate_ordinal_range": [MAX_ORDINAL, MIN_ORDINAL],
             "only_branch": ONLY_BRANCH, "formal_ordinal_gap": FORMAL_ORDINAL_GAP,
             "resource_guard": {"start_ram_gib": 12.0, "start_ram_pct_below": 65.0,
                                "start_gpu_mib": 6000.0, "running_ram_gib": 5.0,
                                "running_ram_pct_below": 90.0, "running_gpu_mib": 2000.0}}
    try:
        frozen.atomic_json(STATUS, state)
        source = frozen.fingerprint()
        while not DATA_DONE.is_file() or OVERLAP_LOCK.exists() or PREVIOUS_LOCK.exists():
            formal = frozen.read_json_retry(frozen.STATUS)
            if formal.get("status") != "running" or not psutil.pid_exists(formal.get("pid", -1)):
                raise RuntimeError("Formal controller stopped before data/model handoff")
            state.update(updated_unix=time.time(), data_marker=DATA_DONE.is_file(),
                         overlap_active=OVERLAP_LOCK.exists(),
                         previous_active=PREVIOUS_LOCK.exists())
            frozen.atomic_json(STATUS, state)
            time.sleep(10)
        previous = frozen.read_json_retry(PREVIOUS_STATUS)
        if previous.get("status") not in ("complete", "data_phase_finished"):
            raise RuntimeError("Previous model sidecar did not complete cleanly")
        data = frozen.read_json_retry(DATA_DONE)
        formal = frozen.read_json_retry(frozen.STATUS)
        if data.get("status") != "complete" or data.get("branch_folds") != 250 or \
           data.get("source_sha256") != source or formal.get("source_sha256") != source:
            raise RuntimeError("Data completion or frozen source does not match formal controller")
        for branch, ordinal in candidates():
            current_formal_ordinal = formal_ordinal()
            if current_formal_ordinal is not None and current_formal_ordinal >= ordinal - FORMAL_ORDINAL_GAP:
                state.update(status="caught_up_with_formal", phase="stopped",
                             formal_ordinal=current_formal_ordinal)
                frozen.atomic_json(STATUS, state)
                return
            if frozen.check_model(branch, ordinal):
                continue
            if not frozen.check_fast(branch, ordinal):
                raise RuntimeError(f"Missing verified data: {branch} {ordinal}")
            model = frozen.fold_path(branch, ordinal) / "cacox_es"
            if (model / "RUNNING.lock").exists():
                raise RuntimeError(f"Unexpected existing model lock: {branch} {ordinal}")
            while True:
                formal_pid = formal_model_pid()
                formal = frozen.read_json_retry(frozen.STATUS)
                if formal.get("status") != "running":
                    state.update(status="formal_stopped", phase="stopped")
                    frozen.atomic_json(STATUS, state)
                    return
                current_formal_ordinal = formal_ordinal()
                if current_formal_ordinal is not None and current_formal_ordinal >= ordinal - FORMAL_ORDINAL_GAP:
                    state.update(status="caught_up_with_formal", phase="stopped",
                                 formal_ordinal=current_formal_ordinal)
                    frozen.atomic_json(STATUS, state)
                    return
                ram, pct, gpu = resources()
                state.update(status="running", phase="waiting_for_resources", branch=branch,
                             ordinal=ordinal, formal_model_pid=formal_pid,
                             free_ram_gib=ram, ram_pct=pct, free_gpu_mib=gpu,
                             updated_unix=time.time())
                frozen.atomic_json(STATUS, state)
                if formal_pid and not OVERLAP_LOCK.exists() and ram >= 12 and pct < 65 and \
                   gpu is not None and gpu >= 6000:
                    break
                time.sleep(10)
            name = f"{branch}_{ordinal:02d}"
            with (LOGS / f"{name}.out.log").open("a", encoding="utf-8") as out, \
                 (LOGS / f"{name}.err.log").open("a", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    [str(frozen.PY), str(frozen.PACKAGE / "code/cacox_fold.py"), branch,
                     str(ordinal), "--device", "cuda", "--threads", "1", "--rscript", str(frozen.R)],
                    cwd=frozen.PROJECT, env=frozen.ENV, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                state.update(phase="training", active_pid=proc.pid,
                             step_started_unix=time.time())
                frozen.atomic_json(STATUS, state)
                while proc.poll() is None:
                    ram, pct, gpu = resources()
                    state.update(free_ram_gib=ram, ram_pct=pct, free_gpu_mib=gpu,
                                 updated_unix=time.time())
                    frozen.atomic_json(STATUS, state)
                    current_formal_ordinal = formal_ordinal()
                    if current_formal_ordinal is not None and current_formal_ordinal >= ordinal:
                        stop_tree(proc)
                        state.update(status="caught_up_with_formal", phase="stopped",
                                     active_pid=None, formal_ordinal=current_formal_ordinal)
                        frozen.atomic_json(STATUS, state)
                        return
                    if ram < 5 or pct >= 90 or gpu is None or gpu < 2000:
                        stop_tree(proc)
                        raise RuntimeError(f"Additional model guard at {name}: RAM={ram:.2f} GiB, "
                                           f"RAM used={pct:.1f}%, GPU free={gpu} MiB")
                    time.sleep(5)
                if proc.returncode != 0:
                    raise RuntimeError(f"Additional model {name} exited {proc.returncode}; inspect logs")
            if frozen.fingerprint() != source or not frozen.check_model(branch, ordinal):
                raise RuntimeError(f"Additional model {name} failed source/output verification")
            state["completed"].append({"branch": branch, "ordinal": ordinal,
                                       "elapsed_seconds": time.time() - state["step_started_unix"]})
            state.update(active_pid=None, phase="verified", updated_unix=time.time())
            frozen.atomic_json(STATUS, state)
        state.update(status="complete", phase="complete", updated_unix=time.time())
        frozen.atomic_json(STATUS, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), active_pid=None,
                     updated_unix=time.time())
        frozen.atomic_json(STATUS, state)
        raise
    finally:
        if LOCK.is_file() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK.unlink()


if __name__ == "__main__":
    main()
