"""Verify the copied portable run before switching its logical path to C:."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from fast_data import PACKAGE as OLD, PROJECT, PY, fingerprint, output_files, sha


NEW = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_20261104")
RUN = "run_fast_20261106_v1"


def main() -> None:
    source = fingerprint()
    for relative, expected in source.items():
        assert sha(NEW / relative) == expected, f"Source hash mismatch: {relative}"

    data_markers = sorted((OLD / RUN / "branches").glob("*/repeat*_fold*/FAST_DATA_COMPLETE.json"))
    model_markers = sorted((OLD / RUN / "branches").glob("*/repeat*_fold*/cacox_es/COMPLETE.json"))
    assert len(data_markers) == 26, len(data_markers)
    assert len(model_markers) == 25, len(model_markers)

    for marker in data_markers:
        relative = marker.relative_to(OLD)
        new_marker = NEW / relative
        assert sha(marker) == sha(new_marker), f"Data marker copy mismatch: {relative}"
        record = json.loads(new_marker.read_text(encoding="utf-8"))
        assert record["source_sha256"] == source
        fold = new_marker.parent
        assert set(record["output_sha256"]) == {str(p.relative_to(fold)) for p in output_files(fold)}
        for path, expected in record["output_sha256"].items():
            assert sha(fold / path) == expected, f"Data artifact mismatch: {relative} {path}"

    for marker in model_markers:
        relative = marker.relative_to(OLD)
        new_marker = NEW / relative
        assert sha(marker) == sha(new_marker), f"Model marker copy mismatch: {relative}"
        result = subprocess.run(
            [str(PY), str(NEW / "code/verify_prediction.py"), str(new_marker.parent)],
            cwd=PROJECT, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Model artifact mismatch: {relative}\n{result.stdout}\n{result.stderr}"

    print(json.dumps({"source_files": len(source), "data_folds": len(data_markers),
                      "model_folds": len(model_markers), "status": "verified"}))


if __name__ == "__main__":
    main()
