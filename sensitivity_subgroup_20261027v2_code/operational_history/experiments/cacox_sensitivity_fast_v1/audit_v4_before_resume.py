"""Check every copied v4 data/model unit before resuming the failed fold."""
from __future__ import annotations

import json
from pathlib import Path
import time

import full_controller as frozen
from fast_data import sha


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    assert frozen.PACKAGE.name == "cacox_sensitivity_portable_adaptive_20261108_v4"
    manifest = read(frozen.PACKAGE / "PACKAGE_MANIFEST.json")
    for name, digest in manifest["files"].items():
        assert sha(frozen.PACKAGE / name) == digest, name
    pre = read(frozen.RUN / "PREVERSION_COMPLETE_UNITS.json")
    assert pre["count"] == 175 and pre["model_count"] == 39
    assert pre["prepared_path_rebases"] == 136
    assert len(pre["approved_source_fingerprints"]) == 4
    data = model = 0
    for ordinal in range(1, 51):
        for branch in frozen.BRANCHES:
            fold = frozen.fold_path(branch, ordinal)
            if (fold / "FAST_DATA_COMPLETE.json").is_file():
                assert frozen.check_fast(branch, ordinal)
                data += 1
            if (fold / "cacox_es/COMPLETE.json").is_file():
                assert frozen.check_model(branch, ordinal)
                model += 1
    assert data == 175 and model == 39, (data, model)
    failed = frozen.fold_path("known_subtype", 35)
    assert not (failed / "FAST_DATA_COMPLETE.json").exists()
    assert not (failed / "FAST_DATA_RUNNING.lock").exists()
    assert (failed / "outer/adaptive/iter_060_original/imputation.rds").is_file()
    assert sha(failed / "outer/imputation.rds") == read(
        frozen.RUN / "handoffs/adaptive_known_subtype35_outer_v4_20261108/CHECKPOINT_REVIEW.json"
    )["continued_imputation_sha256"]
    for name in ("outer", *(f"inner_{i:02d}" for i in range(1, 6))):
        assert (failed / name / "imputation.rds").is_file()
    record = {"status": "passed", "audited_unix": time.time(),
              "data_count": data, "model_count": model,
              "failed_fold": "known_subtype repeat07_fold05",
              "only_failed_partition_promoted_to_iteration": 70,
              "source_sha256": frozen.fingerprint()}
    path = frozen.RUN / "handoffs/adaptive_known_subtype35_outer_v4_20261108/V4_BEFORE_RESUME_AUDIT.json"
    assert not path.exists()
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "data": data, "models": model}))


if __name__ == "__main__":
    main()
