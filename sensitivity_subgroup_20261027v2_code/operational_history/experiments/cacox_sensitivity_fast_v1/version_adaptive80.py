"""Version the portable package manifest for the authorized 80-iteration exception."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil


ORIGINAL = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_20261104")
ADAPTIVE = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive80_20261107_v1")
CHANGED = ("vendor/r/pipeline.R", "code/export_neural.R")
RUN = "run_fast_20261106_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    original_manifest = ORIGINAL / "PACKAGE_MANIFEST.json"
    adaptive_manifest = ADAPTIVE / "PACKAGE_MANIFEST.json"
    assert digest(original_manifest) == digest(adaptive_manifest)
    old = json.loads(original_manifest.read_text(encoding="utf-8"))
    new = json.loads(adaptive_manifest.read_text(encoding="utf-8"))
    assert old == new

    old_hashes = ORIGINAL / RUN / "data/SOURCE_HASHES.csv"
    new_hashes = ADAPTIVE / RUN / "data/SOURCE_HASHES.csv"
    assert digest(old_hashes) == digest(new_hashes)
    archive = ADAPTIVE / RUN / "recoveries/adaptive80_version_20261107"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(adaptive_manifest, archive / "PACKAGE_MANIFEST_before.json")
    shutil.copy2(new_hashes, archive / "SOURCE_HASHES_before.csv")

    changes = {}
    for name in CHANGED:
        assert digest(ORIGINAL / name) == old["files"][name]
        assert digest(ADAPTIVE / name) != old["files"][name]
        new_digest = digest(ADAPTIVE / name)
        new["files"][name] = new_digest
        changes[name] = {"before": old["files"][name], "after": new_digest}

    new["adaptive_protocol"] = {
        "scope": "no_egpa repeat02_fold01 inner_05 only",
        "mice_chains": 20,
        "original_iterations": 60,
        "continued_iterations": [70, 80],
        "use_iterations": 80,
        "Rhat_gate": 1.1,
        "original_package": str(ORIGINAL),
    }
    adaptive_manifest.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")

    with new_hashes.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(sum(row["path"] == name for row in rows) == 1 for name in CHANGED)
    for row in rows:
        if row["path"] in CHANGED:
            path = ADAPTIVE / row["path"]
            row["bytes"] = str(path.stat().st_size)
            row["sha256"] = changes[row["path"]]["after"]
    with new_hashes.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "status": "versioned",
        "original_package": str(ORIGINAL),
        "adaptive_package": str(ADAPTIVE),
        "changed_files": changes,
        "original_manifest_sha256": digest(original_manifest),
        "original_source_hashes_sha256": digest(old_hashes),
        "original_source_and_completed_results_preserved": True,
        "scientific_change": "Only the R MICE iteration gate and export metadata accept the one authorized 80-iteration partition",
    }
    (ADAPTIVE / RUN / "ADAPTIVE_PROTOCOL.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "versioned", "changed_files": list(changes)}))


if __name__ == "__main__":
    main()
