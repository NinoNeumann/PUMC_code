"""Resume the frozen data stage with six concurrent independent R jobs.

Only scheduling changes: the original R code, seeds, MICE settings, selection,
input hashes, and output format remain those of run_data.py. The original data
manifest is retained, and the scheduling change has its own audit record.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import run_data as original

WORKERS = {"impute": 6, "select": 6}
AUDIT = original.OUT / "DATA_SCHEDULER_6PROCESS.json"


def check_frozen_inputs() -> dict:
    manifest = json.loads((original.OUT / "DATA_MANIFEST.json").read_text(encoding="utf8"))
    if manifest["workers"] != {"impute": 4, "select": 6}:
        raise RuntimeError("Unexpected original data scheduling manifest")
    for name, expected in manifest["source_SHA256"].items():
        path = original.ROOT / Path(name)
        if not path.is_file() or original.sha(path) != expected:
            raise RuntimeError(f"Frozen source/input changed: {path}")
    return manifest


def check_completed_folds() -> int:
    ready = 0
    for repeat in range(1, 11):
        for fold in range(1, 6):
            directory = original.unit(repeat, fold)
            bundle = original.OUT / "bundles" / f"repeat{repeat:02d}_fold{fold:02d}" / "main"
            selected = directory / "selected_variables.txt"
            if not selected.exists():
                continue
            original.validate_mice(repeat, fold)
            if not all(original.stage_done("select", repeat, fold, mi) for mi in range(1, 21)):
                raise RuntimeError(f"Incomplete selection in completed fold {repeat}/{fold}")
            if not original.verify_completed(bundle):
                raise RuntimeError(f"Missing completed bundle {repeat}/{fold}")
            ready += 1
    return ready


def main() -> None:
    if sys.argv[1:] != ["run"]:
        raise SystemExit("Usage: run_data_6process.py run")
    check_frozen_inputs()
    completed = check_completed_folds()
    content = {
        "original_manifest_sha256": original.sha(original.OUT / "DATA_MANIFEST.json"),
        "original_data_code_sha256": original.sha(original.HERE / "run_data.py"),
        "scheduler_code_sha256": original.sha(Path(__file__)),
        "original_workers": {"impute": 4, "select": 6},
        "new_workers": WORKERS,
        "completed_folds_verified_before_resume": completed,
        "scientific_settings_unchanged": True,
        "at": original.now(),
    }
    if AUDIT.exists():
        prior = json.loads(AUDIT.read_text(encoding="utf8"))
        for key in ("original_manifest_sha256", "original_data_code_sha256", "scheduler_code_sha256", "new_workers"):
            if prior[key] != content[key]:
                raise RuntimeError(f"Scheduling audit changed: {key}")
    else:
        original.atomic_json(AUDIT, content)
    original.MAX_WORKERS = WORKERS.copy()
    # The frozen entry point performs the same validations and resumes by
    # skipping completed units. Its manifest remains tied to the original code.
    sys.argv = [str(original.HERE / "run_data.py"), "run"]
    original.main()


if __name__ == "__main__":
    main()
