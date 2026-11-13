"""Runtime controls, immutable configuration and audit logging."""
import hashlib
import json
import os
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from .storage import write_json


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Study:
    def __init__(self, project, config):
        self.project = Path(project).resolve()
        self.config_path = Path(config).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.package_root = Path(__file__).resolve().parents[2]
        self.out = (self.project / self.config["output"]).resolve()
        if not self.out.is_relative_to(self.project / "results/cacox_original_head_study"):
            raise ValueError("Output must stay inside the dedicated study directory")
        c = self.config["common"]
        if (c["folds"], c["imputations"], c["max_epochs"], c["patience"], c["rmst_horizon"]) != (5,20,3000,150,1095):
            raise ValueError("Approved fold/MI/epoch/horizon contract changed")
        required=(c["learning_rate"],c["dropout"],c["weight_decay"],c["hidden"],c["num_blocks"],c["num_heads"],c["num_cross_layers"])
        if required != (7e-6,.15,.01,96,2,4,2):
            raise ValueError("Fixed original-CACox training parameters changed")
        if (self.config["protocol"]["ranking_loss"] or self.config["protocol"]["spline_branch"]
                or self.config["protocol"]["search_enabled"] or self.config["candidates"]
                or self.config["candidate"]["bins"]!=6):
            raise ValueError("Original architecture/no-search contract changed")

    def path(self, key):
        return self.project / self.config[key]

    def bundle_path(self, fold):
        return self.path("bundles") / f"repeat01_fold{fold:02d}/main"

    def record(self, event, **payload):
        self.out.mkdir(parents=True, exist_ok=True)
        row = {"time": stamp(), "event": event, **payload}
        with (self.out / "stage_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def status(self, **payload):
        write_json(self.out / "RUN_STATUS.json", {"updated_at": stamp(), "controller_pid":os.getpid(), **payload})

    def log_experiment(self, title, start, purpose, parameters, location, explanation):
        with (self.out / "EXPERIMENTS.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n## {title}\n\n- 时间：{start} → {stamp()}。\n"
                         f"- 目的：{purpose}。\n- 参数：`{json.dumps(parameters, ensure_ascii=False, sort_keys=True)}`。\n"
                         f"- 结果位置：`{location}`。\n- 结果说明：{explanation}。\n")

    def code_hashes(self):
        files=[p for directory in ("src","tools","tests") for p in (self.package_root/directory).rglob("*.py")]
        files += [self.package_root/"pyproject.toml",self.package_root/"REFERENCE_COMPONENTS.json"]
        return {str(p.relative_to(self.package_root)):sha(p) for p in sorted(files)}
