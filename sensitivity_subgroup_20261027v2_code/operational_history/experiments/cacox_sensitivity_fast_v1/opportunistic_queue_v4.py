"""Fill unused resources with one future v4 data fold at a time."""
from __future__ import annotations

import json
import os
import subprocess
import time

import psutil

import full_controller as frozen


STATUS = frozen.RUN / "OPPORTUNISTIC_QUEUE_V4_STATUS.json"
LOCK = frozen.RUN / "OPPORTUNISTIC_QUEUE_V4.lock"
LOGS = frozen.RUN / "logs/opportunistic_queue_v4"
MIN_LEAD_UNITS = 8


def save(**fields: object) -> None:
    frozen.atomic_json(STATUS, {"updated_unix": time.time(), "pid": os.getpid(), **fields})


def formal() -> dict:
    info = frozen.read_json_retry(frozen.STATUS)
    if info.get("status") == "complete":
        return info
    if info.get("status") != "running" or not psutil.pid_exists(int(info["pid"])):
        raise RuntimeError("Formal controller failed or disappeared")
    return info


def next_target():
    """Prefer the farthest unclaimed fold in active batch five."""
    batches = {int(k) for k in formal().get("active_batches", {})}
    if 5 not in batches:
        return None
    batch_path = frozen.RUN / "DATA_BATCH_05_STATUS.json"
    batch = frozen.read_json_retry(batch_path)
    if batch.get("status") != "running" or not psutil.pid_exists(int(batch["pid"])):
        return None
    branch = batch.get("current_branch")
    ordinal = batch.get("current_ordinal")
    if branch is None or ordinal is None:
        return None
    cursor = (int(ordinal) - 41) * len(frozen.BRANCHES) + frozen.BRANCHES.index(branch)
    for number in range(50, 40, -1):
        for candidate in reversed(frozen.BRANCHES):
            index = (number - 41) * len(frozen.BRANCHES) + frozen.BRANCHES.index(candidate)
            if index - cursor < MIN_LEAD_UNITS:
                continue
            fold = frozen.fold_path(candidate, number)
            status = frozen.RUN / f"OPPORTUNISTIC_4X4_{candidate}_{number:02d}_STATUS.json"
            if fold.exists() or status.exists():
                continue
            return candidate, number, status
    return None


def main() -> None:
    if frozen.PACKAGE.name != "cacox_sensitivity_portable_adaptive_20261108_v4":
        raise RuntimeError("Expected the audited v4 package")
    if STATUS.exists() or LOCK.exists():
        raise RuntimeError("Existing queue state requires review")
    protocol = frozen.read_json_retry(frozen.RUN / "ADAPTIVE_PROTOCOL_V4.json")
    if protocol.get("status") != "ready_for_formal_controller":
        raise RuntimeError("v4 recovery audit is not ready")
    formal()
    LOGS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    completed = []
    source = frozen.fingerprint()
    try:
        while (formal().get("status") == "running" and
               formal().get("stage") == "two_data_batches"):
            choice = next_target()
            if choice is None:
                save(status="complete", reason="No distant unclaimed fold remains",
                     completed=completed, source_sha256=source)
                return
            branch, ordinal, status_path = choice
            env = dict(frozen.ENV, CACOX_OPPORTUNISTIC_BRANCH=branch,
                       CACOX_OPPORTUNISTIC_ORDINAL=str(ordinal))
            sidecar_log = LOGS / f"{branch}_{ordinal:02d}"
            with sidecar_log.with_suffix(".out.log").open("x", encoding="utf-8") as out, \
                 sidecar_log.with_suffix(".err.log").open("x", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    [str(frozen.PY), str(frozen.HERE / "opportunistic_fold_4x4.py")],
                    cwd=frozen.PROJECT, env=env, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                save(status="running", current_branch=branch, current_ordinal=ordinal,
                     sidecar_pid=proc.pid, completed=completed, source_sha256=source)
                rc = proc.wait()
            if not status_path.exists():
                raise RuntimeError(f"Extra fold {branch} {ordinal} produced no status; rc={rc}")
            result = frozen.read_json_retry(status_path)
            if rc != 0 or result.get("status") in ("failed", "fold_failed", "aborted"):
                save(status="failed", current_branch=branch, current_ordinal=ordinal,
                     sidecar_status=result.get("status"), sidecar_error=result.get("error"),
                     completed=completed, source_sha256=source)
                return
            if result.get("status") == "not_claimed":
                reason = result.get("reason", "")
                if "No safe resource window" in reason:
                    save(status="no_safe_window", current_branch=branch,
                         current_ordinal=ordinal, completed=completed,
                         source_sha256=source)
                    return
                continue
            if result.get("status") != "complete":
                raise RuntimeError(f"Unexpected extra-fold status: {result.get('status')}")
            if frozen.fingerprint() != source or not frozen.check_fast(branch, ordinal):
                raise RuntimeError(f"Extra fold {branch} {ordinal} failed source/output audit")
            completed.append({"branch": branch, "ordinal": ordinal,
                              "elapsed_seconds": result.get("elapsed_seconds")})
            save(status="running", completed=completed, source_sha256=source)
        save(status="complete", reason="Formal data stage ended", completed=completed,
             source_sha256=source)
    except BaseException as exc:
        save(status="failed", error=repr(exc), completed=completed,
             source_sha256=source)
        raise
    finally:
        if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK.unlink()


if __name__ == "__main__":
    main()
