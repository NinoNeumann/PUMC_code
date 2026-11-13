"""Provenance locks and atomic output; immutable reference cache is never resumed."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(100):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(min(.01 * (attempt + 1), .05))


def fingerprint(files, settings):
    return {"settings": settings, "files": {
        str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}


def lock_manifest(directory, signature):
    p = Path(directory) / "manifest.json"
    if p.exists():
        if json.loads(p.read_text()) != signature:
            raise ValueError(f"Input/code/config changed: {p}; use a new output directory")
    else:
        write_json(p, signature)


@contextlib.contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / "RUNNING.lock"
    try:
        handle = p.open("x")
    except FileExistsError:
        raise RuntimeError(f"Run lock exists: {p}; inspect its process before resuming") from None
    with handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        p.unlink()


def verify_completed(directory):
    p = Path(directory) / "COMPLETE.json"
    if not p.exists():
        return False
    obj = json.loads(p.read_text())
    for name, expected in obj["outputs"].items():
        f = Path(directory) / name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Completed output changed or missing: {f}")
    return True


def mark_complete(directory, output_names):
    directory = Path(directory)
    write_json(directory / "COMPLETE.json", {"outputs": {
        str(n): hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in output_names}})
