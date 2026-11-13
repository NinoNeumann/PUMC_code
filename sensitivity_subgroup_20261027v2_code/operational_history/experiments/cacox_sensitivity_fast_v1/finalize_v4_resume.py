"""Audit the recovered fold and untouched copied units before formal restart."""
from __future__ import annotations

import csv
import json
import time

import full_controller as frozen
from fast_data import sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    run = frozen.RUN
    archive = run / "handoffs/adaptive_known_subtype35_outer_v4_20261108"
    before = read(archive / "V4_BEFORE_RESUME_AUDIT.json")
    assert before["status"] == "passed" and before["data_count"] == 175
    assert frozen.fingerprint() == before["source_sha256"]
    pre = read(run / "PREVERSION_COMPLETE_UNITS.json")
    for key, digest in pre["marker_sha256"].items():
        assert sha(run / "branches" / key / "FAST_DATA_COMPLETE.json") == digest, key
    for key, digest in pre["model_marker_sha256"].items():
        assert sha(run / "branches" / key / "cacox_es/COMPLETE.json") == digest, key
    fold = frozen.fold_path("known_subtype", 35)
    assert read(fold / "FAST_DATA_STATUS.json")["status"] == "complete"
    assert not (fold / "FAST_DATA_RUNNING.lock").exists()
    assert frozen.check_fast("known_subtype", 35)
    assert len(list(fold.glob("imputation_*/combined_selection.csv"))) == 20
    all_rhat = []
    for part in ("outer", *(f"inner_{i:02d}" for i in range(1, 6))):
        with (fold / part / "mice_chain_diagnostic.csv").open(newline="", encoding="utf-8") as stream:
            all_rhat.extend(float(row["classical_Rhat"]) for row in csv.DictReader(stream)
                            if row["classical_Rhat"])
    assert max(all_rhat) < 1.1
    assert len(pre["marker_sha256"]) + 1 == 176
    protocol_path = run / "ADAPTIVE_PROTOCOL_V4.json"
    protocol = read(protocol_path)
    assert protocol["status"] == "versioned_pending_failed_fold_resume"
    protocol["status"] = "ready_for_formal_controller"
    protocol["recovered_fold_marker_sha256"] = sha(fold / "FAST_DATA_COMPLETE.json")
    protocol["recovered_fold_max_Rhat"] = max(all_rhat)
    protocol["recovered_fold_selection_count"] = 20
    protocol["ready_unix"] = time.time()
    write(protocol_path, protocol)
    write(archive / "V4_RESUME_COMPLETE_AUDIT.json", {
        "status": "passed", "data_count": 176, "model_count": 39,
        "recovered_fold": "known_subtype repeat07_fold05",
        "Rhat_max": max(all_rhat), "selection_count": 20,
        "marker_sha256": protocol["recovered_fold_marker_sha256"],
        "audited_unix": time.time(),
    })
    print(json.dumps({"status": "ready", "data": 176, "models": 39,
                      "max_Rhat": max(all_rhat)}))


if __name__ == "__main__":
    main()
