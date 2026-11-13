"""Re-sign copied v4 markers after auditing prepared-path metadata changes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from version_known_subtype_35_outer_v4 import OLD_RUN, RUN, RECOVERY, sha, read, write


def main() -> None:
    with (RECOVERY / "PREPARED_REBASE_AUDIT.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    preversion_path = RUN / "PREVERSION_COMPLETE_UNITS.json"
    preversion = read(preversion_path)
    originals = dict(preversion["marker_sha256"])
    assert len(originals) == 175
    backup = RECOVERY / "completed_markers_before_rebase"
    backup.mkdir()
    count = 0
    for row in rows:
        branch, ordinal = row["branch"], int(row["ordinal"])
        name = f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"
        key = f"{branch}/{name}"
        fold = RUN / "branches" / key
        assert sha(fold / "prepared.rds") == row["rebased_sha256"]
        assert sha(OLD_RUN / "branches" / key / "prepared.rds") == row["original_sha256"]
        marker = fold / "FAST_DATA_COMPLETE.json"
        if not marker.exists():
            assert key == "known_subtype/repeat07_fold05"
            continue
        assert sha(marker) == originals[key]
        record = read(marker)
        assert record["output_sha256"]["prepared.rds"] == row["original_sha256"]
        shutil.copy2(marker, backup / f"{branch}_{name}.json")
        record["output_sha256"]["prepared.rds"] = row["rebased_sha256"]
        write(marker, record)
        preversion["marker_sha256"][key] = sha(marker)
        count += 1
    assert len(rows) == 137, len(rows)
    assert count == 136, count
    preversion["original_marker_sha256"] = originals
    preversion["prepared_path_rebases"] = count
    preversion["marker_metadata_rewritten_only_in_copy"] = True
    write(preversion_path, preversion)
    protocol = RUN / "ADAPTIVE_PROTOCOL_V4.json"
    state = read(protocol)
    assert state["status"] == "versioned_pending_path_rebase"
    state["status"] = "versioned_pending_failed_fold_resume"
    state["rebased_prepared"] = len(rows)
    state["updated_copied_markers"] = count
    write(protocol, state)
    print(json.dumps({"status": "audited", "rebased_prepared": len(rows),
                      "updated_copied_markers": count}))


if __name__ == "__main__":
    main()
