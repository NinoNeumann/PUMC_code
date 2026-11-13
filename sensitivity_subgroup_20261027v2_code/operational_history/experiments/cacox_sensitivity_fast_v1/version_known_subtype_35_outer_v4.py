"""Version the one reviewed 70-iteration partition after a complete v3 audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import time


BASE = Path(r"C:\Users\NinoNeumann\CodexResults")
OLD = BASE / "cacox_sensitivity_portable_adaptive80_20261108_v3"
NEW = BASE / "cacox_sensitivity_portable_adaptive_20261108_v4"
RUN_NAME = "run_fast_20261106_v1"
OLD_RUN = OLD / RUN_NAME
RUN = NEW / RUN_NAME
RECOVERY = RUN / "handoffs/adaptive_known_subtype35_outer_v4_20261108"
SOURCE_AUDIT = OLD_RUN / "handoffs/adaptive_known_subtype35_outer_20261108/V4_PREHANDOFF_AUDIT.json"
FOLD = RUN / "branches/known_subtype/repeat07_fold05"
PART = FOLD / "outer"
FILES = ("imputation.rds", "mice_chain_diagnostic.csv", "continuous_imputation_trace.png")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    assert OLD.is_dir() and NEW.is_dir()
    assert not RECOVERY.exists()
    prior = read(SOURCE_AUDIT)
    assert prior["status"] == "passed" and prior["data_count"] == 175
    assert prior["model_count"] == 39 and len(prior["approved_source_fingerprints"]) == 4
    assert read(OLD_RUN / "FAST_CONTROLLER_STATUS.json")["status"] == "failed"
    assert read(OLD_RUN / "DATA_BATCH_03_COMPLETE.json")["status"] == "complete"
    assert read(OLD_RUN / "DATA_BATCH_04_STATUS.json")["status"] == "failed"
    assert not (FOLD / "FAST_DATA_COMPLETE.json").exists()

    old_manifest = read(OLD / "PACKAGE_MANIFEST.json")
    copied_manifest = read(NEW / "PACKAGE_MANIFEST.json")
    assert copied_manifest == old_manifest
    changed = "vendor/r/pipeline.R"
    for name, digest in old_manifest["files"].items():
        assert sha(OLD / name) == digest, name
        if name != changed:
            assert sha(NEW / name) == digest, name
        else:
            assert sha(NEW / name) != digest
    for key, digest in prior["data_marker_sha256"].items():
        assert sha(RUN / "branches" / key / "FAST_DATA_COMPLETE.json") == digest, key
    for key, digest in prior["model_marker_sha256"].items():
        assert sha(RUN / "branches" / key / "cacox_es/COMPLETE.json") == digest, key

    checkpoint = read(OLD_RUN / "handoffs/adaptive_known_subtype35_outer_20261108/CHECKPOINT_REVIEW.json")
    assert checkpoint["continued_Rhat"] < 1.1 and checkpoint["logged_events"] == 0
    assert sha(PART / "imputation.rds") == checkpoint["original_imputation_sha256"]
    assert sha(PART / "adaptive/iter_070/imputation.rds") == checkpoint["continued_imputation_sha256"]
    RECOVERY.mkdir(parents=True)
    write(RECOVERY / "V4_PREHANDOFF_AUDIT.json", prior)
    write(RECOVERY / "CHECKPOINT_REVIEW.json", checkpoint)
    saved = PART / "adaptive/iter_060_original"
    saved.mkdir()
    original_hashes, promoted_hashes = {}, {}
    for name in FILES:
        assert sha(PART / name) == sha(OLD_RUN / PART.relative_to(RUN) / name)
        assert sha(PART / "adaptive/iter_070" / name) == sha(
            OLD_RUN / PART.relative_to(RUN) / "adaptive/iter_070" / name)
        shutil.copy2(PART / name, saved / name)
        original_hashes[name] = sha(saved / name)
        shutil.copy2(PART / "adaptive/iter_070" / name, PART / name)
        promoted_hashes[name] = sha(PART / name)

    manifest = copied_manifest
    manifest["files"][changed] = sha(NEW / changed)
    manifest["adaptive_protocol"] = {
        "scope": ["no_egpa repeat02_fold01 inner_05:80",
                  "egfr_observed repeat05_fold03 inner_04:80",
                  "known_subtype repeat05_fold04 inner_02:80",
                  "known_subtype repeat07_fold05 outer:70"],
        "mice_chains": 20, "original_iterations": 60,
        "Rhat_gate_exclusive": 1.1, "previous_package": str(OLD),
    }
    write(NEW / "PACKAGE_MANIFEST.json", manifest)
    source_table = RUN / "data/SOURCE_HASHES.csv"
    with source_table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["path"] == changed]
    assert len(matches) == 1 and matches[0]["sha256"] == old_manifest["files"][changed]
    matches[0]["sha256"] = manifest["files"][changed]
    matches[0]["bytes"] = str((NEW / changed).stat().st_size)
    with source_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    previous_audit = RUN / "PREVERSION_COMPLETE_UNITS.json"
    shutil.copy2(previous_audit, RECOVERY / "PREVERSION_COMPLETE_UNITS.v3.json")
    write(previous_audit, {
        "previous_package": str(OLD),
        "count": prior["data_count"], "marker_sha256": prior["data_marker_sha256"],
        "model_count": prior["model_count"],
        "model_marker_sha256": prior["model_marker_sha256"],
        "approved_source_fingerprints": prior["approved_source_fingerprints"],
        "copied_unmodified_before_prepared_path_rebase": True,
    })
    for name in ("FAST_CONTROLLER_STATUS.json", *(
        f"DATA_BATCH_{i:02d}_STATUS.json" for i in range(1, 6)), *(
        f"DATA_BATCH_{i:02d}_COMPLETE.json" for i in range(1, 6))):
        path = RUN / name
        if path.exists():
            path.replace(RECOVERY / (name + ".v3"))
    for name in ("FAST_CONTROLLER.lock", *(
        f"DATA_BATCH_{i:02d}.lock" for i in range(1, 6))):
        assert not (RUN / name).exists(), name
    (FOLD / "FAST_DATA_STATUS.json").replace(RECOVERY / "failed_fold_status.v3.json")
    for name in ("mice_quality_gate.out.log", "mice_quality_gate.err.log"):
        path = RUN / "logs/fast_data/known_subtype/repeat07_fold05" / name
        if path.exists():
            path.replace(RECOVERY / (name + ".v3"))
    write(RUN / "ADAPTIVE_PROTOCOL_V4.json", {
        "status": "versioned_pending_path_rebase", "previous_package": str(OLD),
        "current_package": str(NEW), "preversion_data": prior["data_count"],
        "preversion_models": prior["model_count"],
        "fourth_exception": "known_subtype repeat07_fold05 outer",
        "Rhat_60": checkpoint["original_Rhat"],
        "Rhat_70": checkpoint["continued_Rhat"], "logged_events_70": 0,
        "original_60_sha256": original_hashes,
        "promoted_70_sha256": promoted_hashes,
        "changed_scientific_source": [changed],
        "scientific_change": "Only failed outer mids uses its 70-round continuation",
        "versioned_unix": time.time(),
    })
    print(json.dumps({"status": "versioned_pending_path_rebase",
                      "data": prior["data_count"], "models": prior["model_count"]}))


if __name__ == "__main__":
    main()
