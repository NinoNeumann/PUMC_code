"""Finish v3 handoff after confirming SOURCE_HASHES omits Python model code."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from version_known_subtype_24_adaptive80_v3 import (
    CURRENT, PREVIOUS, OLD_RUN, RUN, RECOVERY, FOLD, PART, EXPECTED, read, sha, write,
)


def main() -> None:
    assert RECOVERY.is_dir()
    assert read(RECOVERY / "PREVERSION_COMPLETE_UNITS.v2.json")["count"] == 80
    preversion = read(RUN / "PREVERSION_COMPLETE_UNITS.json")
    assert preversion["count"] == 119 and preversion["model_count"] == 39
    assert not (RUN / "ADAPTIVE_PROTOCOL_V3.json").exists()
    prior_manifest = read(PREVIOUS / "PACKAGE_MANIFEST.json")
    manifest = read(CURRENT / "PACKAGE_MANIFEST.json")
    changed = {"vendor/r/pipeline.R", "code/cacox_fold.py"}
    for name in changed:
        assert prior_manifest["files"][name] == sha(PREVIOUS / name)
        assert manifest["files"][name] == sha(CURRENT / name)
        assert manifest["files"][name] != prior_manifest["files"][name]
    assert len(manifest["adaptive_protocol"]["scope"]) == 3
    saved60 = PART / "adaptive/iter_060_original"
    original_hashes, promoted_hashes = {}, {}
    for name in EXPECTED:
        assert sha(saved60 / name) == sha(OLD_RUN / PART.relative_to(RUN) / name)
        assert sha(PART / name) == sha(PART / "adaptive/iter_080" / name)
        original_hashes[name] = sha(saved60 / name)
        promoted_hashes[name] = sha(PART / name)

    path = RUN / "data/SOURCE_HASHES.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    r_pipeline = [row for row in rows if row["path"] == "vendor/r/pipeline.R"]
    assert len(r_pipeline) == 1
    assert r_pipeline[0]["sha256"] == prior_manifest["files"]["vendor/r/pipeline.R"]
    assert not [row for row in rows if row["path"] == "code/cacox_fold.py"]
    r_pipeline[0]["sha256"] = manifest["files"]["vendor/r/pipeline.R"]
    r_pipeline[0]["bytes"] = str((CURRENT / "vendor/r/pipeline.R").stat().st_size)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    for name in ("FAST_CONTROLLER_STATUS.json", "DATA_BATCH_01_STATUS.json",
                 "DATA_BATCH_02_STATUS.json", "DATA_BATCH_03_STATUS.json",
                 "DATA_BATCH_01_COMPLETE.json", "DATA_BATCH_02_COMPLETE.json"):
        item = RUN / name
        if item.exists():
            item.replace(RECOVERY / (name + ".previous"))
    for name in ("FAST_CONTROLLER.lock", "DATA_BATCH_02.lock", "DATA_BATCH_03.lock"):
        assert not (RUN / name).exists(), name
    (FOLD / "FAST_DATA_STATUS.json").replace(
        RECOVERY / "known_subtype_24_FAST_DATA_STATUS.failed.json")
    for name in ("mice_quality_gate.out.log", "mice_quality_gate.err.log"):
        item = RUN / "logs/fast_data/known_subtype/repeat05_fold04" / name
        if item.exists():
            item.replace(RECOVERY / (name + ".failed"))
    with (PART / "adaptive/iter_080/ADAPTIVE_AUDIT.csv").open(newline="", encoding="utf-8") as stream:
        audit80 = list(csv.DictReader(stream))[0]
    assert float(audit80["max_Rhat"]) <= 1.1 and int(audit80["logged_events"]) == 0
    write(RUN / "ADAPTIVE_PROTOCOL_V3.json", {
        "status": "versioned", "previous_package": str(PREVIOUS),
        "current_package": str(CURRENT),
        "preversion_completed_data": 119, "preversion_completed_models": 39,
        "third_exception": "known_subtype repeat05_fold04 inner_02",
        "Rhat_60": 1.1162831985484, "Rhat_70": 1.0622904102248,
        "Rhat_80": float(audit80["max_Rhat"]), "logged_events_80": 0,
        "original_60_hashes": original_hashes, "promoted_80_hashes": promoted_hashes,
        "changed_source_files": sorted(changed),
        "source_table_note": "Python model file tracked by PACKAGE_MANIFEST, not R data SOURCE_HASHES",
        "scientific_changes": "Only the failed partition uses its reviewed 80-round continuation; prediction metadata reports actual exported iteration count",
        "versioned_unix": time.time(),
    })
    print(json.dumps({"status": "versioned", "data": 119, "models": 39,
                      "rhat80": float(audit80["max_Rhat"])}))


if __name__ == "__main__":
    main()
