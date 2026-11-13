"""Re-sign only copied v3 completion markers whose prepared path was rebased."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from version_known_subtype_24_adaptive80_v3 import OLD_RUN, RUN, RECOVERY, sha, read, write


def main() -> None:
    with (RECOVERY / "PREPARED_REBASE_AUDIT.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 81, len(rows)
    preversion_path = RUN / "PREVERSION_COMPLETE_UNITS.json"
    preversion = read(preversion_path)
    original_marker_hashes = dict(preversion["marker_sha256"])
    assert len(original_marker_hashes) == 119
    backup_dir = RECOVERY / "completed_markers_before_rebase"
    backup_dir.mkdir(parents=True, exist_ok=False)
    changed = 0
    for row in rows:
        branch, ordinal = row["branch"], int(row["ordinal"])
        name = f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"
        key = f"{branch}/{name}"
        fold = RUN / "branches" / key
        prepared = fold / "prepared.rds"
        old_prepared = OLD_RUN / "branches" / key / "prepared.rds"
        assert sha(prepared) == row["rebased_sha256"]
        assert sha(old_prepared) == row["original_sha256"]
        marker = fold / "FAST_DATA_COMPLETE.json"
        if not marker.exists():
            assert key == "known_subtype/repeat05_fold04"
            continue
        assert sha(marker) == original_marker_hashes[key]
        record = read(marker)
        assert record["output_sha256"]["prepared.rds"] == row["original_sha256"]
        shutil.copy2(marker, backup_dir / f"{branch}_{name}.json")
        record["output_sha256"]["prepared.rds"] = row["rebased_sha256"]
        write(marker, record)
        preversion["marker_sha256"][key] = sha(marker)
        changed += 1
    assert changed == 80, changed
    preversion["original_marker_sha256"] = original_marker_hashes
    preversion["prepared_path_rebases"] = changed
    preversion["marker_metadata_rewritten_only_in_copy"] = True
    write(preversion_path, preversion)
    print(json.dumps({"status": "audited", "rebased_prepared": len(rows),
                      "rewritten_copied_completion_markers": changed}))


if __name__ == "__main__":
    main()
