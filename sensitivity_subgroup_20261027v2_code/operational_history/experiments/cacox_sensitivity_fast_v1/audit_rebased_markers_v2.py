"""Record path-only prepared.rds rebases in copied completion metadata."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil


RUN = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive80_20261107_v2\run_fast_20261106_v1")
PREVIOUS = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive80_20261107_v1\run_fast_20261106_v1")
RECOVERY = RUN / "handoffs/adaptive80_egfr_observed_23_20261107"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    audit_path = RECOVERY / "PREPARED_REBASE_AUDIT.csv"
    rows = list(csv.DictReader(audit_path.open(encoding="utf-8", newline="")))
    assert len(rows) == 42, len(rows)
    preversion_path = RUN / "PREVERSION_COMPLETE_UNITS.json"
    preversion = json.loads(preversion_path.read_text(encoding="utf-8"))
    original_marker_hashes = dict(preversion["marker_sha256"])
    backup_dir = RECOVERY / "completed_markers_before_rebase"
    backup_dir.mkdir(parents=True, exist_ok=False)
    changed = 0
    for row in rows:
        branch, ordinal = row["branch"], int(row["ordinal"])
        name = f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"
        key = f"{branch}/{name}"
        fold = RUN / "branches" / key
        prepared = fold / "prepared.rds"
        old_prepared = PREVIOUS / "branches" / key / "prepared.rds"
        assert sha(prepared) == row["rebased_sha256"]
        assert sha(old_prepared) == row["original_sha256"]
        marker = fold / "FAST_DATA_COMPLETE.json"
        if not marker.exists():
            assert key == "egfr_observed/repeat05_fold03"
            continue
        assert sha(marker) == original_marker_hashes[key]
        record = json.loads(marker.read_text(encoding="utf-8"))
        assert record["output_sha256"]["prepared.rds"] == row["original_sha256"]
        shutil.copy2(marker, backup_dir / f"{branch}_{name}.json")
        record["output_sha256"]["prepared.rds"] = row["rebased_sha256"]
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        preversion["marker_sha256"][key] = sha(marker)
        changed += 1
    assert changed == 41, changed
    preversion["original_marker_sha256"] = original_marker_hashes
    preversion["prepared_path_rebases"] = changed
    preversion["marker_metadata_rewritten_only_in_copy"] = True
    preversion_path.write_text(json.dumps(preversion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "audited", "rebased_prepared": len(rows),
                      "rewritten_copied_completion_markers": changed}))


if __name__ == "__main__":
    main()
