"""Read-only audit of v3 before copying the single reviewed 70-round exception."""
from __future__ import annotations

import json
from pathlib import Path
import time

import full_controller as frozen
from fast_data import sha


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    run = frozen.RUN
    assert frozen.PACKAGE.name == "cacox_sensitivity_portable_adaptive80_20261108_v3"
    assert read(run / "FAST_CONTROLLER_STATUS.json")["status"] == "failed"
    assert read(run / "DATA_BATCH_03_COMPLETE.json")["status"] == "complete"
    assert read(run / "DATA_BATCH_04_STATUS.json")["status"] == "failed"
    for name in ("FAST_CONTROLLER.lock", "DATA_BATCH_03.lock", "DATA_BATCH_04.lock"):
        assert not (run / name).exists(), name
    failed = frozen.fold_path("known_subtype", 35)
    assert not (failed / "FAST_DATA_COMPLETE.json").exists()
    assert not (failed / "FAST_DATA_RUNNING.lock").exists()

    manifest = read(frozen.PACKAGE / "PACKAGE_MANIFEST.json")
    for name, digest in manifest["files"].items():
        assert sha(frozen.PACKAGE / name) == digest, name
    source = frozen.fingerprint()
    records, models, sources = {}, {}, {}
    for ordinal in range(1, 51):
        for branch in frozen.BRANCHES:
            fold = frozen.fold_path(branch, ordinal)
            marker = fold / "FAST_DATA_COMPLETE.json"
            if marker.is_file():
                assert frozen.check_fast(branch, ordinal)
                key = f"{branch}/{fold.name}"
                records[key] = sha(marker)
                s = read(marker)["source_sha256"]
                sources[json.dumps(s, sort_keys=True)] = s
            model = fold / "cacox_es/COMPLETE.json"
            if model.is_file():
                assert frozen.check_model(branch, ordinal)
                models[f"{branch}/{fold.name}"] = sha(model)
    assert len(records) == 175, len(records)
    assert len(models) == 39, len(models)
    part = failed / "outer"
    audit70 = (part / "adaptive/iter_070/ADAPTIVE_AUDIT.csv").read_text(encoding="utf-8")
    assert "1.07516620707022" in audit70
    assert sha(part / "imputation.rds") == read(
        run / "handoffs/adaptive_known_subtype35_outer_20261108/CHECKPOINT_REVIEW.json"
    )["original_imputation_sha256"]
    assert sha(part / "adaptive/iter_070/imputation.rds") == read(
        run / "handoffs/adaptive_known_subtype35_outer_20261108/CHECKPOINT_REVIEW.json"
    )["continued_imputation_sha256"]

    record = {
        "status": "passed", "audited_unix": time.time(),
        "package": str(frozen.PACKAGE), "source_sha256": source,
        "data_count": len(records), "model_count": len(models),
        "data_marker_sha256": records, "model_marker_sha256": models,
        "approved_source_fingerprints": list(sources.values()),
        "reviewed_exception": "known_subtype repeat07_fold05 outer 60->70",
        "Rhat_60": 1.10375207663824, "Rhat_70": 1.07516620707022,
    }
    path = run / "handoffs/adaptive_known_subtype35_outer_20261108/V4_PREHANDOFF_AUDIT.json"
    assert not path.exists()
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "data": len(records),
                      "models": len(models), "source_versions": len(sources)}))


if __name__ == "__main__":
    main()
