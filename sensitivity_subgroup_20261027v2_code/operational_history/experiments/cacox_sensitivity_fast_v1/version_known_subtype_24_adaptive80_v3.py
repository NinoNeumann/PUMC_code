"""Audit and version the third reviewed MICE continuation in the copied v3 package."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import time


BASE = Path(r"C:\Users\NinoNeumann\CodexResults")
PREVIOUS = BASE / "cacox_sensitivity_portable_adaptive80_20261107_v2"
CURRENT = BASE / "cacox_sensitivity_portable_adaptive80_20261108_v3"
RUN_NAME = "run_fast_20261106_v1"
OLD_RUN = PREVIOUS / RUN_NAME
RUN = CURRENT / RUN_NAME
RECOVERY = RUN / "handoffs/adaptive80_known_subtype_24_20261108"
FOLD = RUN / "branches/known_subtype/repeat05_fold04"
PART = FOLD / "inner_02"
EXPECTED = ("imputation.rds", "mice_chain_diagnostic.csv", "continuous_imputation_trace.png")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    assert PREVIOUS.is_dir() and CURRENT.is_dir()
    assert read(OLD_RUN / "FAST_CONTROLLER_STATUS.json")["status"] == "failed"
    assert read(OLD_RUN / "DATA_BATCH_02_COMPLETE.json")["status"] == "complete"
    assert read(OLD_RUN / "DATA_BATCH_03_STATUS.json")["status"] == "failed"
    assert not (FOLD / "FAST_DATA_COMPLETE.json").exists()
    assert not RECOVERY.exists()

    prior_manifest = read(PREVIOUS / "PACKAGE_MANIFEST.json")
    copied_manifest = read(CURRENT / "PACKAGE_MANIFEST.json")
    assert copied_manifest == prior_manifest
    changed_sources = {"vendor/r/pipeline.R", "code/cacox_fold.py"}
    for name, digest in prior_manifest["files"].items():
        assert sha(PREVIOUS / name) == digest, name
        copied_hash = sha(CURRENT / name)
        assert (copied_hash != digest) == (name in changed_sources), name

    markers = {}
    for marker in sorted((RUN / "branches").glob("*/repeat*_fold*/FAST_DATA_COMPLETE.json")):
        old = OLD_RUN / marker.relative_to(RUN)
        assert sha(marker) == sha(old)
        markers[marker.parent.relative_to(RUN / "branches").as_posix()] = sha(marker)
    assert len(markers) == 119, len(markers)
    models = {}
    for marker in sorted((RUN / "branches").glob("*/repeat*_fold*/cacox_es/COMPLETE.json")):
        assert sha(marker) == sha(OLD_RUN / marker.relative_to(RUN))
        models[marker.parent.parent.relative_to(RUN / "branches").as_posix()] = sha(marker)
    assert len(models) == 39, len(models)

    old_part = OLD_RUN / PART.relative_to(RUN)
    for name in EXPECTED:
        assert sha(PART / name) == sha(old_part / name)
        assert sha(PART / "adaptive/iter_080" / name) == sha(old_part / "adaptive/iter_080" / name)
    with (PART / "adaptive/iter_080/ADAPTIVE_AUDIT.csv").open(newline="", encoding="utf-8") as stream:
        audit80 = list(csv.DictReader(stream))[0]
    assert audit80["branch"] == "known_subtype" and audit80["ordinal"] == "24"
    assert audit80["partition"] == "inner_02" and audit80["target_iteration"] == "80"
    assert float(audit80["max_Rhat"]) <= 1.1 and int(audit80["logged_events"]) == 0
    with (PART / "mice_chain_diagnostic.csv").open(newline="", encoding="utf-8") as stream:
        rhat60 = max(float(row["classical_Rhat"]) for row in csv.DictReader(stream)
                     if row["classical_Rhat"])
    assert abs(rhat60 - 1.1162831985484) < 1e-12

    RECOVERY.mkdir(parents=True)
    shutil.copy2(RUN / "PREVERSION_COMPLETE_UNITS.json",
                 RECOVERY / "PREVERSION_COMPLETE_UNITS.v2.json")
    write(RUN / "PREVERSION_COMPLETE_UNITS.json", {
        "previous_package": str(PREVIOUS),
        "grandparent_package": str(BASE / "cacox_sensitivity_portable_adaptive80_20261107_v1"),
        "count": len(markers), "marker_sha256": markers,
        "model_count": len(models), "model_marker_sha256": models,
        "copied_unmodified": True,
    })
    original_hashes, promoted_hashes = {}, {}
    saved60 = PART / "adaptive/iter_060_original"
    saved60.mkdir(parents=True)
    for name in EXPECTED:
        shutil.copy2(PART / name, saved60 / name)
        original_hashes[name] = sha(saved60 / name)
        shutil.copy2(PART / "adaptive/iter_080" / name, PART / name)
        promoted_hashes[name] = sha(PART / name)

    manifest = copied_manifest
    for name in changed_sources:
        manifest["files"][name] = sha(CURRENT / name)
    manifest["adaptive_protocol"] = {
        "scope": ["no_egpa repeat02_fold01 inner_05",
                  "egfr_observed repeat05_fold03 inner_04",
                  "known_subtype repeat05_fold04 inner_02"],
        "mice_chains": 20, "original_iterations": 60,
        "continued_iterations": [70, 80], "use_iterations": 80,
        "Rhat_gate": 1.1, "previous_package": str(PREVIOUS),
        "prediction_metadata": "Uses the actual exported MICE iteration count",
    }
    write(CURRENT / "PACKAGE_MANIFEST.json", manifest)
    hashes_path = RUN / "data/SOURCE_HASHES.csv"
    with hashes_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for name in changed_sources:
        row = [r for r in rows if r["path"] == name]
        # SOURCE_HASHES tracks R data-stage inputs; Python model files are
        # frozen by PACKAGE_MANIFEST and the model's own source signature.
        assert len(row) <= 1
        if row:
            assert row[0]["sha256"] == prior_manifest["files"][name]
            row[0]["sha256"] = manifest["files"][name]
            row[0]["bytes"] = str((CURRENT / name).stat().st_size)
    with hashes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    for name in ("FAST_CONTROLLER_STATUS.json", "DATA_BATCH_01_STATUS.json",
                 "DATA_BATCH_02_STATUS.json", "DATA_BATCH_03_STATUS.json",
                 "DATA_BATCH_01_COMPLETE.json", "DATA_BATCH_02_COMPLETE.json"):
        path = RUN / name
        if path.exists():
            path.replace(RECOVERY / (name + ".previous"))
    for name in ("FAST_CONTROLLER.lock", "DATA_BATCH_02.lock", "DATA_BATCH_03.lock"):
        path = RUN / name
        if path.exists():
            raise RuntimeError("Unexpected live lock in cloned package: " + str(path))
    failed_status = FOLD / "FAST_DATA_STATUS.json"
    failed_status.replace(RECOVERY / "known_subtype_24_FAST_DATA_STATUS.failed.json")
    for name in ("mice_quality_gate.out.log", "mice_quality_gate.err.log"):
        path = RUN / "logs/fast_data/known_subtype/repeat05_fold04" / name
        if path.exists():
            path.replace(RECOVERY / (name + ".failed"))
    write(RUN / "ADAPTIVE_PROTOCOL_V3.json", {
        "status": "versioned", "previous_package": str(PREVIOUS),
        "current_package": str(CURRENT),
        "preversion_completed_data": len(markers), "preversion_completed_models": len(models),
        "third_exception": "known_subtype repeat05_fold04 inner_02",
        "Rhat_60": rhat60, "Rhat_70": 1.0622904102248,
        "Rhat_80": float(audit80["max_Rhat"]), "logged_events_80": 0,
        "original_60_hashes": original_hashes, "promoted_80_hashes": promoted_hashes,
        "changed_source_files": sorted(changed_sources),
        "scientific_changes": "Only the failed partition uses its reviewed 80-round continuation; CACox prediction metadata now reports the actual exported iteration count",
        "versioned_unix": time.time(),
    })
    print(json.dumps({"status": "versioned", "data": len(markers), "models": len(models),
                      "rhat80": float(audit80["max_Rhat"])}))


if __name__ == "__main__":
    main()
