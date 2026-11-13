"""Independent structural and arithmetic audit of the renal subgroup tables."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: verify_main_renal_subgroups.py PACKAGE_ROOT OUTPUT_DIR R_SCRIPT")
    root, output, r_script = map(Path, sys.argv[1:])
    pooled_path = output / "renal_subgroup_pooled.csv"
    each_path = output / "renal_subgroup_each_imputation.csv"
    folds_path = output / "renal_subgroup_each_fold.csv"
    pooled = {r["population"]: r for r in rows(pooled_path)}
    each = rows(each_path)
    folds = rows(folds_path)
    assert set(pooled) == {
        "renal_involvement_no", "renal_involvement_yes",
        "renal_involvement_yes_minus_no",
    }
    assert len(each) == 20 and {int(r["imputation"]) for r in each} == set(range(1, 21))
    assert len(folds) == 2000
    for group in ("renal_involvement_no", "renal_involvement_yes"):
        subset = [r for r in folds if r["population"] == group]
        assert len(subset) == 1000
        for mi in range(1, 21):
            one = [r for r in subset if int(r["imputation"]) == mi]
            assert len(one) == 50
            assert {(int(r["repeat_id"]), int(r["fold"])) for r in one} == {
                (repeat, fold) for repeat in range(1, 11) for fold in range(1, 6)
            }
            assert all(float(r["comparable_pairs"]) > 0 for r in one)
            assert all(0 <= float(r["C"]) <= 1 for r in one)
            assert close(sum(float(r["weight"]) for r in one), 1)
    for r in each:
        assert close(float(r["difference_yes_minus_no"]), float(r["C_yes"]) - float(r["C_no"]))
        assert close(float(r["variance_difference"]), float(r["variance_yes"]) + float(r["variance_no"]))
    metrics = {
        "renal_involvement_no": ("C_no", "variance_no", 423, 37),
        "renal_involvement_yes": ("C_yes", "variance_yes", 436, 61),
        "renal_involvement_yes_minus_no": ("difference_yes_minus_no", "variance_difference", 859, 98),
    }
    for group, (value, variance, n, events) in metrics.items():
        p = pooled[group]
        q = [float(r[value]) for r in each]
        u = [float(r[variance]) for r in each]
        within = statistics.mean(u)
        between = statistics.variance(q)
        total = within + 1.05 * between
        assert int(p["patients"]) == n and int(p["events"]) == events
        assert (int(p["repeats"]), int(p["outer_folds"]), int(p["imputations"])) == (10, 50, 20)
        assert close(float(p["estimate"]), statistics.mean(q))
        assert close(float(p["within_variance"]), within)
        assert close(float(p["between_variance"]), between)
        assert close(float(p["total_variance"]), total)
        assert close(float(p["SE"]) ** 2, total)
    oof = root / "input" / "main_oof_cacox.csv.gz"
    expected = (root / "input" / "main_oof_cacox.csv.gz.sha256").read_text().strip()
    assert sha256(oof) == expected
    files = [pooled_path, each_path, folds_path, output / "METHODS.txt"]
    audit = {
        "status": "passed",
        "model": "original_full_cohort_CACox_ES_frozen_OOF",
        "contrast": "renal_involvement_yes_minus_no",
        "oof_sha256": expected,
        "raw_typed_sha256": sha256(root / "input" / "raw_typed.rds"),
        "evaluation_r_sha256": sha256(root / "code" / "evaluation.R"),
        "analysis_script_sha256": sha256(r_script),
        "files_sha256": {p.name: sha256(p) for p in files},
        "checks": [
            "50 outer folds for each of 20 imputations and both renal groups",
            "positive comparable pairs and within-imputation fold weights sum to one",
            "raw group and event counts; disjoint groups checked by R evaluator",
            "yes-minus-no per-imputation differences and sum of conditional variances",
            "Rubin pooled means, within/between/total variances and SE",
            "original compressed OOF SHA256 matches archived sidecar",
        ],
    }
    (output / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RENAL_SUBGROUP_AUDIT_PASSED")


if __name__ == "__main__":
    main()
