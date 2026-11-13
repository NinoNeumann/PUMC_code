"""Version the reviewed second 80-round MICE exception without changing prior results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import time


BASE = Path(r"C:\Users\NinoNeumann\CodexResults")
PREVIOUS = BASE / "cacox_sensitivity_portable_adaptive80_20261107_v1"
CURRENT = BASE / "cacox_sensitivity_portable_adaptive80_20261107_v2"
RUN_NAME = "run_fast_20261106_v1"
PREV_RUN = PREVIOUS / RUN_NAME
RUN = CURRENT / RUN_NAME
RECOVERY = RUN / "handoffs/adaptive80_egfr_observed_23_20261107"
FOLD = RUN / "branches/egfr_observed/repeat05_fold03"
PART = FOLD / "inner_04"
EXPECTED_60_HASH = "9e579a78951d4caa424f6d8b43c2623967af00833fc35c43602e31abf6b99e1c"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    assert PREV_RUN.is_dir() and RUN.is_dir()
    assert (PREV_RUN / "FAST_CONTROLLER.lock").read_text().strip() == "51256"
    assert (PREV_RUN / "DATA_BATCH_02.lock").read_text().strip() == "47624"
    assert sha(PREV_RUN / "branches/egfr_observed/repeat05_fold03/inner_04/imputation.rds") == EXPECTED_60_HASH
    assert sha(PART / "imputation.rds") == EXPECTED_60_HASH
    assert not (FOLD / "FAST_DATA_COMPLETE.json").exists()
    assert not (PART / "adaptive/iter_060_original").exists()
    assert not (RUN / "PREVERSION_COMPLETE_UNITS.json").exists()

    markers = {}
    for marker in sorted((RUN / "branches").glob("*/repeat*_fold*/FAST_DATA_COMPLETE.json")):
        relative = marker.parent.relative_to(RUN / "branches").as_posix()
        prior_marker = PREV_RUN / marker.relative_to(RUN)
        assert sha(marker) == sha(prior_marker)
        markers[relative] = sha(marker)
    assert len(markers) == 80, len(markers)

    expected = ("imputation.rds", "mice_chain_diagnostic.csv", "continuous_imputation_trace.png")
    for name in expected:
        assert sha(PART / name) == sha(PREV_RUN / PART.relative_to(RUN) / name)
        assert sha(PART / "adaptive/iter_080" / name) == sha(
            PREV_RUN / PART.relative_to(RUN) / "adaptive/iter_080" / name
        )
    audit80 = list(csv.DictReader((PART / "adaptive/iter_080/ADAPTIVE_AUDIT.csv").open(newline="")))[0]
    assert audit80["branch"] == "egfr_observed" and audit80["ordinal"] == "23"
    assert audit80["partition"] == "inner_04" and audit80["target_iteration"] == "80"
    assert float(audit80["max_Rhat"]) <= 1.1 and int(audit80["logged_events"]) == 0

    RECOVERY.mkdir(parents=True, exist_ok=True)
    write_json(RUN / "PREVERSION_COMPLETE_UNITS.json", {
        "previous_package": str(PREVIOUS), "count": len(markers),
        "marker_sha256": markers, "copied_unmodified": True,
    })
    saved60 = PART / "adaptive/iter_060_original"
    saved60.mkdir(parents=True, exist_ok=False)
    original_hashes = {}
    promoted_hashes = {}
    for name in expected:
        shutil.copy2(PART / name, saved60 / name)
        original_hashes[name] = sha(saved60 / name)
        shutil.copy2(PART / "adaptive/iter_080" / name, PART / name)
        promoted_hashes[name] = sha(PART / name)

    manifest_path = CURRENT / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_manifest = json.loads((PREVIOUS / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest == previous_manifest
    old_pipeline_hash = previous_manifest["files"]["vendor/r/pipeline.R"]
    assert sha(PREVIOUS / "vendor/r/pipeline.R") == old_pipeline_hash
    new_pipeline_hash = sha(CURRENT / "vendor/r/pipeline.R")
    assert new_pipeline_hash != old_pipeline_hash
    manifest["files"]["vendor/r/pipeline.R"] = new_pipeline_hash
    manifest["adaptive_protocol"] = {
        "scope": ["no_egpa repeat02_fold01 inner_05",
                  "egfr_observed repeat05_fold03 inner_04"],
        "mice_chains": 20, "original_iterations": 60,
        "continued_iterations": [70, 80], "use_iterations": 80,
        "Rhat_gate": 1.1, "previous_package": str(PREVIOUS),
    }
    write_json(manifest_path, manifest)

    hashes_path = RUN / "data/SOURCE_HASHES.csv"
    with hashes_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    target = [row for row in rows if row["path"] == "vendor/r/pipeline.R"]
    assert len(target) == 1 and target[0]["sha256"] == old_pipeline_hash
    target[0]["sha256"] = new_pipeline_hash
    target[0]["bytes"] = str((CURRENT / "vendor/r/pipeline.R").stat().st_size)
    with hashes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    for name in ("FAST_CONTROLLER.lock", "DATA_BATCH_02.lock", "FAST_CONTROLLER_STATUS.json",
                 "DATA_BATCH_01_STATUS.json", "DATA_BATCH_02_STATUS.json",
                 "DATA_BATCH_03_STATUS.json", "DATA_BATCH_01_COMPLETE.json"):
        path = RUN / name
        if path.exists():
            path.replace(RECOVERY / (name + ".previous"))
    failed_status = FOLD / "FAST_DATA_STATUS.json"
    failed_status.replace(RECOVERY / "egfr_observed_23_FAST_DATA_STATUS.failed.json")
    for name in ("mice_quality_gate.out.log", "mice_quality_gate.err.log"):
        path = RUN / "logs/fast_data/egfr_observed/repeat05_fold03" / name
        if path.exists():
            path.replace(RECOVERY / (name + ".failed"))

    write_json(RUN / "ADAPTIVE_PROTOCOL_V2.json", {
        "status": "versioned", "previous_package": str(PREVIOUS),
        "current_package": str(CURRENT), "preversion_completed_data": len(markers),
        "preversion_completed_models": 39,
        "second_exception": "egfr_observed repeat05_fold03 inner_04",
        "Rhat_60": 1.10253568721231, "Rhat_70": 1.065710626112,
        "Rhat_80": float(audit80["max_Rhat"]), "logged_events_80": 0,
        "original_60_hashes": original_hashes,
        "promoted_80_hashes": promoted_hashes,
        "pipeline_hash_before": old_pipeline_hash,
        "pipeline_hash_after": new_pipeline_hash,
        "scientific_changes": "Only this failed MICE partition uses its verified 80-round continuation; prior completed folds are unchanged",
        "versioned_unix": time.time(),
    })
    print(json.dumps({"status": "versioned", "copied_complete": len(markers),
                      "rhat80": float(audit80["max_Rhat"])}))


if __name__ == "__main__":
    main()
