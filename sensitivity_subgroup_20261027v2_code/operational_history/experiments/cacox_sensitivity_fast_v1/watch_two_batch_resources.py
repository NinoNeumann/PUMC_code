"""Write bounded two-batch CPU, RAM, and GPU samples for the scheduling audit."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import subprocess
import time

import psutil

import full_controller as frozen


def stage(batch: int) -> str:
    path = frozen.RUN / f"DATA_BATCH_{batch:02d}_STATUS.json"
    if not path.is_file():
        return "missing"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        return f"{info.get('status')}:{info.get('current_branch')}:{info.get('current_ordinal')}:{info.get('stage')}"
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def gpu() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=5, check=True,
        )
        return tuple(part.strip() for part in result.stdout.splitlines()[0].split(",", 1))
    except (OSError, subprocess.SubprocessError, IndexError):
        return "", ""


def main() -> None:
    path = frozen.RUN / "TWO_BATCH_RESOURCE_SAMPLES.csv"
    fields = ["utc", "cpu_pct", "ram_used_pct", "ram_available_gib",
              "gpu_util_pct", "gpu_memory_mib", "batch_1", "batch_2"]
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        for _ in range(360):
            vm = psutil.virtual_memory()
            gpu_util, gpu_mem = gpu()
            writer.writerow({
                "utc": datetime.now(timezone.utc).isoformat(),
                "cpu_pct": psutil.cpu_percent(interval=0.1),
                "ram_used_pct": round((vm.total - vm.available) * 100 / vm.total, 3),
                "ram_available_gib": round(vm.available / 2**30, 3),
                "gpu_util_pct": gpu_util,
                "gpu_memory_mib": gpu_mem,
                "batch_1": stage(1), "batch_2": stage(2),
            })
            stream.flush()
            time.sleep(5)


if __name__ == "__main__":
    main()
