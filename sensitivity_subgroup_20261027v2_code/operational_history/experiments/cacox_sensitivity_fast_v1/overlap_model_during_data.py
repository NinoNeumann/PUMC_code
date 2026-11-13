"""Train verified CACox folds during the existing data phase, one at a time.

This scheduler changes no scientific inputs or model options. It exits when the
formal controller leaves its data phase; the formal model pass then verifies
and reuses any completed folds.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

import psutil

import full_controller as frozen


STATUS = frozen.RUN / "OVERLAP_MODEL_STATUS.json"
LOCK = frozen.RUN / "OVERLAP_MODEL.lock"
LOGS = frozen.RUN / "logs" / "overlap_models"
MIN_START_RAM_GIB = 10.0
MAX_START_RAM_PCT = 70.0
MIN_START_GPU_MIB = 8000.0
MIN_RUNNING_RAM_GIB = 5.0
MAX_RUNNING_RAM_PCT = 90.0
MIN_RUNNING_GPU_MIB = 2500.0


def resources() -> tuple[float, float, float | None]:
    vm = psutil.virtual_memory()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        gpu_free = float(out.splitlines()[0].split(",")[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        gpu_free = None
    return vm.available / 2**30, vm.percent, gpu_free


def formal_data_phase() -> bool:
    record = frozen.read_json_retry(frozen.STATUS)
    if record.get("status") != "running" or record.get("stage") != "two_data_batches":
        return False
    pid = record.get("pid")
    return isinstance(pid, int) and psutil.pid_exists(pid)


def stop_tree(proc: subprocess.Popen) -> None:
    try:
        root = psutil.Process(proc.pid)
        children = root.children(recursive=True)
        for child in children:
            child.terminate()
        root.terminate()
        _, alive = psutil.wait_procs([*children, root], timeout=15)
        for child in alive:
            child.kill()
    except psutil.NoSuchProcess:
        pass


def candidates():
    # Formal training begins at ordinal 8; these folds cannot collide before
    # the data phase ends. Each candidate must have a validated data marker.
    for ordinal in range(40, 50):
        for branch in frozen.BRANCHES:
            yield branch, ordinal


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {"status": "running", "pid": os.getpid(), "started_unix": time.time(),
             "phase": "preflight", "completed": [], "active_pid": None,
             "resource_guard": {"start_ram_gib": MIN_START_RAM_GIB,
                                "start_ram_pct_below": MAX_START_RAM_PCT,
                                "start_gpu_mib": MIN_START_GPU_MIB,
                                "running_ram_gib": MIN_RUNNING_RAM_GIB,
                                "running_ram_pct_below": MAX_RUNNING_RAM_PCT,
                                "running_gpu_mib": MIN_RUNNING_GPU_MIB}}
    try:
        frozen.atomic_json(STATUS, state)
        source = frozen.fingerprint()
        formal = frozen.read_json_retry(frozen.STATUS)
        if source != formal.get("source_sha256") or not formal_data_phase():
            raise RuntimeError("Formal data controller or frozen source did not pass preflight")
        for branch, ordinal in candidates():
            if not formal_data_phase():
                state.update(status="data_phase_finished", phase="stopped")
                break
            if not frozen.check_fast(branch, ordinal):
                continue
            if frozen.check_model(branch, ordinal):
                continue
            model = frozen.fold_path(branch, ordinal) / "cacox_es"
            if (model / "RUNNING.lock").exists():
                raise RuntimeError(f"Existing model lock requires review: {branch} {ordinal}")
            while True:
                if not formal_data_phase():
                    state.update(status="data_phase_finished", phase="stopped")
                    break
                ram, pct, gpu = resources()
                state.update(phase="waiting_for_resources", free_ram_gib=ram,
                             ram_pct=pct, free_gpu_mib=gpu, updated_unix=time.time())
                frozen.atomic_json(STATUS, state)
                if ram >= MIN_START_RAM_GIB and pct < MAX_START_RAM_PCT and \
                   gpu is not None and gpu >= MIN_START_GPU_MIB:
                    break
                time.sleep(10)
            if state["status"] != "running":
                break
            name = f"{branch}_{ordinal:02d}"
            with (LOGS / f"{name}.out.log").open("a", encoding="utf-8") as out, \
                 (LOGS / f"{name}.err.log").open("a", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    [str(frozen.PY), str(frozen.PACKAGE / "code/cacox_fold.py"), branch,
                     str(ordinal), "--device", "cuda", "--threads", "1", "--rscript", str(frozen.R)],
                    cwd=frozen.PROJECT, env=frozen.ENV, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                state.update(phase="training", branch=branch, ordinal=ordinal,
                             active_pid=proc.pid, step_started_unix=time.time())
                frozen.atomic_json(STATUS, state)
                while proc.poll() is None:
                    ram, pct, gpu = resources()
                    state.update(free_ram_gib=ram, ram_pct=pct, free_gpu_mib=gpu,
                                 updated_unix=time.time())
                    frozen.atomic_json(STATUS, state)
                    if ram < MIN_RUNNING_RAM_GIB or pct >= MAX_RUNNING_RAM_PCT or \
                       gpu is None or gpu < MIN_RUNNING_GPU_MIB:
                        stop_tree(proc)
                        raise RuntimeError(f"Model resource guard at {name}: RAM={ram:.2f} GiB, "
                                           f"RAM used={pct:.1f}%, GPU free={gpu} MiB")
                    time.sleep(5)
                if proc.returncode != 0:
                    raise RuntimeError(f"Model {name} exited {proc.returncode}; inspect logs")
            if frozen.fingerprint() != source or not frozen.check_model(branch, ordinal):
                raise RuntimeError(f"Model {name} failed frozen-source or output verification")
            state["completed"].append({"branch": branch, "ordinal": ordinal,
                                       "elapsed_seconds": time.time() - state["step_started_unix"]})
            state.update(active_pid=None, phase="verified", updated_unix=time.time())
            frozen.atomic_json(STATUS, state)
        if state["status"] == "running":
            state.update(status="complete", phase="complete")
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
