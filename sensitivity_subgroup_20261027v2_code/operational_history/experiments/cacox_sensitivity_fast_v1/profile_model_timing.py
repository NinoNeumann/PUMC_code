"""Read-only timing summary for completed CACox sensitivity folds."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import statistics


ROOT = Path(r"C:\Users\NinoNeumann\CodexResults\cacox_sensitivity_portable_adaptive_20261108_v4\run_fast_20261106_v1")
BRANCHES = ("complete_case", "egfr_observed", "renal_observed", "known_subtype", "no_egpa")
STAGE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO stage=(\w+) status=")


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "median_seconds": round(statistics.median(values), 1),
            "mean_seconds": round(statistics.mean(values), 1),
            "min_seconds": round(min(values), 1), "max_seconds": round(max(values), 1)}


def timed_stages(log: Path) -> dict[str, float] | None:
    if not log.is_file():
        return None
    entries = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STAGE_RE.match(line)
        if match:
            entries.append((datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f"), match.group(2)))
    if not entries or entries[-1][1] != "complete":
        return None
    starts = [i for i, (_, stage) in enumerate(entries) if stage == "exporting_raw_partitions"]
    if not starts:
        return None
    entries = entries[starts[-1]:]
    timing = defaultdict(float)
    for (start, stage), (end, _) in zip(entries, entries[1:]):
        timing[stage] += (end - start).total_seconds()
    return dict(timing)


def main() -> None:
    rows = []
    for branch in BRANCHES:
        for ordinal in range(1, 51):
            name = f"repeat{(ordinal-1)//5+1:02d}_fold{(ordinal-1)%5+1:02d}"
            model = ROOT / "branches" / branch / name / "cacox_es"
            complete = model / "COMPLETE.json"
            if not complete.is_file():
                continue
            report = json.loads(complete.read_text(encoding="utf-8"))
            timing = timed_stages(model / "logs" / "fold.log")
            rows.append({"branch": branch, "ordinal": ordinal,
                         "elapsed_seconds": report.get("elapsed_this_invocation_seconds"),
                         "completed_at": report.get("completed_at"), "stages": timing})
    result = {"root": str(ROOT), "completed_models": len(rows),
              "by_branch": {branch: summary([r["elapsed_seconds"] for r in rows
                                             if r["branch"] == branch and r["elapsed_seconds"] is not None])
                            for branch in BRANCHES},
              "stage_coverage": sum(r["stages"] is not None for r in rows),
              "rows": rows}
    for branch in BRANCHES:
        subset = [r for r in rows if r["branch"] == branch and r["stages"]]
        result.setdefault("stages_by_branch", {})[branch] = {
            key: summary([r["stages"].get(key, 0.0) for r in subset])
            for key in ("exporting_raw_partitions", "encoding_and_isolation",
                        "i20_inner01", "i20_inner02", "i20_inner03", "i20_inner04", "i20_inner05",
                        "select_i20", "es_inner01", "es_inner02", "es_inner03", "es_inner04",
                        "es_inner05", "select_es", "i20_outer", "es_outer", "replay_outer",
                        "writing_final_predictions")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
