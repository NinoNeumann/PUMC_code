"""One production CACox ES sensitivity fold, with explicit patient isolation.

--prepare-only exports and encodes all six partitions without training. Completed
training stages are resumed through cacox_cpu_train's signed checkpoint API.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
import psutil

import cacox_cpu_core as c
import cacox_cpu_train as t

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
_run_setting = os.environ.get("CACOX_RUN_ROOT", os.environ.get("CACOX_OUTPUT_ROOT", "run"))
RUN_ROOT = (ROOT / _run_setting).resolve()
if not RUN_ROOT.is_relative_to(ROOT):
    raise ValueError("CACOX_RUN_ROOT must be inside the portable package for movable integrity records")
BRANCHES = ("complete_case", "egfr_observed", "renal_observed", "known_subtype", "no_egpa")
PARTITIONS = ("outer", "inner01", "inner02", "inner03", "inner04", "inner05")
ENCODER_SOURCE = c.SOURCE_ROOT
sys.path.insert(0, str(ENCODER_SOURCE))
_encoder_module = importlib.import_module("cacox_encoder.numeric_encoder")
_schema_module = importlib.import_module("cacox_encoder.schema29")
Encoder = _encoder_module.Encoder
ENCODER_FILE = Path(_encoder_module.__file__).resolve()
SCHEMA_FILE = Path(_schema_module.__file__).resolve()


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def discover_rscript(requested=None):
    candidate = requested or os.environ.get("CACOX_RSCRIPT") or shutil.which("Rscript")
    if not candidate:
        raise FileNotFoundError("Rscript was not found. Set CACOX_RSCRIPT, pass --rscript, or add R/bin to PATH.")
    discovered = shutil.which(candidate) or (str(Path(candidate).resolve()) if Path(candidate).is_file() else None)
    if discovered is None:
        raise FileNotFoundError(f"Configured Rscript does not exist: {candidate}")
    return discovered


def identity_hash(value):
    return hashlib.sha256(json.dumps(t._jsonable(value), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def verify_files(root, files):
    for relative, expected in files.items():
        path = (Path(root) / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
            raise ValueError(f"Artifact identifier must stay relative to its root: {relative}")
        if not path.is_file() or c.sha_file(path) != expected:
            raise ValueError(f"Missing or changed signed artifact: {path}")


def write_npz(path, group):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    values = {k: v for k, v in group.items() if k != "metadata"}
    values["metadata_json"] = np.array(json.dumps(group["metadata"], ensure_ascii=False, sort_keys=True))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


class Progress:
    def __init__(self, out, branch, ordinal, logger):
        self.path = out / "progress.json"
        self.logger = logger
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = dict(branch=branch, ordinal=ordinal, pid=os.getpid(), status="running",
                          stage="initializing", started_at=stamp(), stage_started_at=stamp())
        self.stage_start = time.monotonic()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def update(self, stage, **values):
        with self.lock:
            if stage != self.state["stage"]:
                self.stage_start = time.monotonic()
                self.state["stage_started_at"] = stamp()
            self.state.update(stage=stage, **values)
            self._write()
        self.logger.info("stage=%s status=%s", stage, self.state["status"])

    def _write(self):
        self.state.update(updated_at=stamp(), stage_elapsed_seconds=time.monotonic() - self.stage_start)
        t.write_json(self.path, self.state)

    def _heartbeat(self):
        while not self.stop.wait(30):
            with self.lock:
                self._write()
                stage = self.state["stage"]
                elapsed = self.state["stage_elapsed_seconds"]
            self.logger.info("heartbeat stage=%s elapsed_seconds=%.1f", stage, elapsed)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def acquire_lock(path):
    if path.exists():
        old = int(path.read_text().strip())
        if psutil.pid_exists(old):
            raise RuntimeError(f"CACox fold is already owned by live PID {old}: {path}")
        path.unlink()
    with path.open("x") as stream:
        stream.write(str(os.getpid()))


def read_export_metadata(neural_root, branch, ordinal):
    result = {}
    repeat_id, fold = (ordinal - 1) // 5 + 1, (ordinal - 1) % 5 + 1
    expected_views = 1 if branch == "complete_case" else 20
    for index, part in enumerate(PARTITIONS):
        directory = neural_root / part
        metadata = t.read_json(directory / "metadata.json")
        expected = dict(branch=branch, ordinal=ordinal, repeat_id=repeat_id, fold=fold,
                        inner_fold=index, n_views=expected_views)
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(f"Export identity mismatch: {part} {key}")
        if metadata.get("path_base") != "package_root":
            raise ValueError("Portable export metadata must use path_base=package_root")
        verify_files(ROOT, metadata["output_sha256"])
        verify_files(ROOT, metadata["input_sha256"])
        variables = (directory / "selected_variables.txt").read_text(encoding="utf-8").splitlines()
        if variables != metadata["variables"] or not variables or len(set(variables)) != len(variables):
            raise ValueError("Selected-variable contract mismatch")
        if len(metadata["views"]) != expected_views:
            raise ValueError("Incorrect exported view count")
        if metadata["columns"] != [*variables, "Death", "Survival_Days", "patient_id"]:
            raise ValueError("Raw columns differ from the requested clinical export contract")
        result[part] = metadata
    if any(m["variables"] != result["outer"]["variables"] for m in result.values()):
        raise ValueError("Selected variables differ between outer and inner partitions")
    return result


def read_clinical_csv(path, metadata, side):
    categories = {v: "string" for v in metadata["variables"] if v in _schema_module.LEVELS}
    frame = pd.read_csv(path, dtype=categories, float_precision="round_trip")
    if frame.columns.tolist() != metadata["columns"] or len(frame) != metadata[f"{side}_count"]:
        raise ValueError(f"CSV row/column contract mismatch: {path}")
    if frame.isna().any().any() or frame.patient_id.duplicated().any():
        raise ValueError(f"Missing value or duplicate patient: {path}")
    if not frame.Death.isin([0, 1]).all() or not (frame.Survival_Days > 0).all():
        raise ValueError(f"Invalid outcome: {path}")
    numeric = [v for v in metadata["variables"] if v in _schema_module.CONTINUOUS]
    if not np.isfinite(frame[[*numeric, "Death", "Survival_Days", "patient_id"]].to_numpy(float)).all():
        raise ValueError(f"Nonfinite numeric input: {path}")
    ids = frame.patient_id.to_numpy(np.int64)
    if not np.array_equal(ids, frame.patient_id.to_numpy()):
        raise ValueError("Patient IDs must remain integral")
    return frame


def encode_partition(directory, metadata, destination):
    variables = metadata["variables"]
    train_arrays, val_arrays, encoders = [], [], []
    columns = None
    group = {}
    for view_index, view in enumerate(metadata["views"], 1):
        if view["imputation"] != view_index:
            raise ValueError("MI export index/order mismatch")
        train = read_clinical_csv(directory / view["train_file"], metadata, "train")
        val = read_clinical_csv(directory / view["val_file"], metadata, "val")
        for side, frame in (("train", train), ("val", val)):
            values = dict(ids=frame.patient_id.to_numpy(np.int64), time=frame.Survival_Days.to_numpy(np.float64),
                          event=frame.Death.to_numpy(bool))
            for field, value in values.items():
                name = f"{side}_{field}"
                if view_index == 1:
                    group[name] = value
                else:
                    np.testing.assert_array_equal(group[name], value,
                                                  err_msg="IDs/outcomes differ between completed views")
        encoder = Encoder(variables, "robust")
        x = encoder.fit_transform(train)
        vx = encoder.transform(val)
        if columns is None:
            columns = list(encoder.columns)
        elif columns != encoder.columns:
            raise ValueError("Encoded feature columns differ between views")
        state = dict(view=view_index, variables=list(encoder.variables), scale=encoder.scale,
                     centers=dict(encoder.centers), scales=dict(encoder.scales), columns=list(encoder.columns),
                     fitted_partition="training_only")
        # The plain JSON state must suffice to replay the untouched source Encoder.
        restored = Encoder(state["variables"], state["scale"])
        restored.centers, restored.scales, restored.columns = state["centers"], state["scales"], state["columns"]
        np.testing.assert_array_equal(restored.transform(train), x)
        np.testing.assert_array_equal(restored.transform(val), vx)
        train_arrays.append(x)
        val_arrays.append(vx)
        encoders.append(state)
    group.update(x=np.stack(train_arrays), vx=np.stack(val_arrays))
    group["metadata"] = dict(metadata, raw_columns=metadata["columns"], columns=columns,
                             variables=variables, encoder="archived_Encoder_robust_training_only",
                             encoder_source_sha256=c.sha_file(ENCODER_FILE))
    group = c.validate_group(group)
    destination.mkdir(parents=True, exist_ok=True)
    write_npz(destination / "group.npz", group)
    t.write_json(destination / "encoders.json", encoders)
    return group


def validate_fold_isolation(groups):
    outer = groups["outer"]
    development = set(outer["train_ids"].tolist())
    assessment = set(outer["val_ids"].tolist())
    if development & assessment:
        raise ValueError("Outer development/test overlap")
    expected_outcomes = {
        int(patient): (float(duration), bool(event))
        for patient, duration, event in zip(outer["train_ids"], outer["train_time"], outer["train_event"])
    }
    validation_ids = []
    inner_sizes = []
    for k in range(1, 6):
        g = groups[f"inner{k:02d}"]
        train = set(g["train_ids"].tolist())
        val = set(g["val_ids"].tolist())
        if train & val or (train | val) != development or ((train | val) & assessment):
            raise ValueError(f"Inner fold {k} does not partition outer development exclusively")
        if g["metadata"]["variables"] != outer["metadata"]["variables"]:
            raise ValueError("Variable list differs across partitions")
        if g["metadata"]["columns"] != outer["metadata"]["columns"]:
            raise ValueError("Encoded design differs across partitions")
        if g["x"].shape[0] != outer["x"].shape[0]:
            raise ValueError("View count differs across partitions")
        for side in ("train", "val"):
            for patient, duration, event in zip(g[f"{side}_ids"], g[f"{side}_time"], g[f"{side}_event"]):
                if expected_outcomes[int(patient)] != (float(duration), bool(event)):
                    raise ValueError("Patient outcome differs between inner and outer partitions")
        validation_ids.extend(g["val_ids"].tolist())
        inner_sizes.append(dict(inner_fold=k, train=len(train), val=len(val)))
    if len(validation_ids) != len(set(validation_ids)) or set(validation_ids) != development:
        raise ValueError("Five inner validation folds must cover outer development exactly once")
    return dict(passed=True, outer_train=len(development), outer_val=len(assessment),
                inner=inner_sizes, genuine_views=outer["x"].shape[0],
                clinical_variables=len(outer["metadata"]["variables"]),
                encoded_features=outer["x"].shape[2], outer_test_used_in_any_inner_partition=False,
                inner_validation_exactly_once=True, outcome_consistency=True)


def prepare_groups(neural_root, out, branch, ordinal):
    exports = read_export_metadata(neural_root, branch, ordinal)
    source_files = [Path(__file__), CODE / "export_neural.R", CODE / "common.R", ENCODER_FILE, SCHEMA_FILE]
    identity = dict(branch=branch, ordinal=ordinal,
                    source_sha256={c.package_id(path): c.sha_file(path) for path in source_files},
                    metadata_sha256={part: c.sha_file(neural_root / part / "metadata.json") for part in PARTITIONS},
                    preprocessing=dict(scale="robust", float_precision="round_trip", per_view_training_fit=True,
                                       complete_case_views=1, imputed_views=20),
                    numpy=np.__version__, pandas=pd.__version__)
    signature = identity_hash(identity)
    marker = out / "PREPARED.json"
    if marker.exists():
        record = t.read_json(marker)
        if record["signature"] != signature:
            raise ValueError("Encoded inputs belong to different sources or exports; use a new versioned run")
        verify_files(out, record["files"])
        groups = {part: c.load_group_npz(out / "encoded" / part / "group.npz") for part in PARTITIONS}
        isolation = validate_fold_isolation(groups)
        if isolation != record["isolation"]:
            raise ValueError("Changed encoded patient-isolation audit")
        return groups, record
    groups = {part: encode_partition(neural_root / part, exports[part], out / "encoded" / part)
              for part in PARTITIONS}
    isolation = validate_fold_isolation(groups)
    t.write_json(out / "isolation.json", isolation)
    files = {path.relative_to(out).as_posix(): c.sha_file(path)
             for path in sorted((out / "encoded").rglob("*")) if path.is_file()}
    files["isolation.json"] = c.sha_file(out / "isolation.json")
    record = dict(signature=signature, identity=identity, files=files, isolation=isolation,
                  group_sha256={part: c.group_digest(group) for part, group in groups.items()})
    t.write_json(marker, record)
    return groups, record


def verify_stage(directory):
    record = t.read_json(directory / "COMPLETE.json")
    if record["manifest_sha256"] != c.sha_file(directory / "manifest.json"):
        raise ValueError(f"Changed stage manifest: {directory}")
    verify_files(directory, record["files"])


def freeze_selection(value, destination):
    if destination.exists() and t.read_json(destination) != t._jsonable(value):
        raise ValueError(f"Frozen internal selection changed: {destination}")
    t.write_json(destination, value)


def save_final_predictions(out, group, replay, branch, repeat_id, fold):
    scores, probabilities = replay["selected_val"], replay["survival"]
    views, n = scores.shape
    if probabilities.shape != (views, n, 3) or views != group["x"].shape[0]:
        raise ValueError("Final prediction dimensions differ from genuine views")
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite final predictions")
    if (probabilities < 0).any() or (probabilities > 1).any() or (np.diff(probabilities, axis=2) > 0).any():
        raise ValueError("Invalid survival probabilities")
    frames = []
    for mi in range(1, 21):
        source = 0 if branch == "complete_case" else mi - 1
        frames.append(pd.DataFrame(dict(
            branch=branch, repeat_id=repeat_id, fold=fold, imputation=mi,
            patient_id=group["val_ids"], Death=group["val_event"].astype(int), Survival_Days=group["val_time"],
            score_E_selected=scores[source], survival_365d=probabilities[source, :, 0],
            survival_730d=probabilities[source, :, 1], survival_1095d=probabilities[source, :, 2],
            genuine_views=views,
            imputation_kind=("complete_case_single_view_replicated_for_pairing" if views == 1
                             else group["metadata"]["imputation_kind"]))))
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 20 * n or result[["imputation", "patient_id"]].duplicated().any():
        raise ValueError("Invalid final MI/patient pairing")
    path = out / "predictions_all20.csv"
    temporary = path.with_suffix(".csv.tmp")
    result.to_csv(temporary, index=False, float_format="%.17g")
    reread = pd.read_csv(temporary, float_precision="round_trip")
    for column in ("score_E_selected", "survival_365d", "survival_730d", "survival_1095d", "Survival_Days"):
        np.testing.assert_array_equal(reread[column].to_numpy(), result[column].to_numpy())
    temporary.replace(path)
    return len(result)


def run_training(groups, preparation, out, args, progress):
    options = t.RunOptions(threads=args.threads, neural_device=args.device)
    args.device = options.neural_device
    options.validate()
    c.configure(options.threads)
    source_files = [Path(__file__), CODE / "export_neural.R", CODE / "common.R", CODE / "cacox_cpu_core.py",
                    CODE / "cacox_cpu_train.py", ENCODER_FILE, SCHEMA_FILE]
    identity = dict(branch=args.branch, ordinal=args.ordinal, preparation_signature=preparation["signature"],
                    prepared_sha256=c.sha_file(out / "PREPARED.json"), options=asdict(options),
                    source_sha256={c.package_id(path): c.sha_file(path) for path in source_files},
                    source=c.provenance(args.device), group_sha256=preparation["group_sha256"],
                    exporter="code/export_neural.R", rscript="Rscript (installation path excluded from scientific signature)",
                    pandas=pd.__version__, production=True)
    signed = dict(signature=identity_hash(identity), identity=t._jsonable(identity))
    manifest = out / "manifest.json"
    if manifest.exists() and t.read_json(manifest) != signed:
        raise ValueError("Top-level fold signature differs from current code, data, versions or options")
    t.write_json(manifest, signed)
    stages = out / "stages"
    i20_inner = [stages / f"i20_inner{k:02d}" for k in range(1, 6)]
    es_inner = [stages / f"es_inner{k:02d}" for k in range(1, 6)]
    i20_outer, es_outer = stages / "i20_outer", stages / "es_outer"
    stage_dirs = [*i20_inner, *es_inner, i20_outer, es_outer]
    completed = out / "COMPLETE.json"
    if completed.exists():
        report = t.read_json(completed)
        if report["signature"] != signed["signature"]:
            raise ValueError("Completed fold signature differs")
        verify_files(out, report["files"])
        for directory in stage_dirs:
            verify_stage(directory)
        progress.update("complete", status="complete", reused=True)
        return report
    fold, repeat_id = (args.ordinal - 1) % 5 + 1, (args.ordinal - 1) // 5 + 1
    started = time.monotonic()
    for k, directory in enumerate(i20_inner, 1):
        progress.update(f"i20_inner{k:02d}", inner_fold=k, completed_stages=k - 1)
        t.fit_i20_inner(groups[f"inner{k:02d}"], directory, fold, k, options)
        verify_stage(directory)
    progress.update("select_i20", completed_stages=5)
    i20_choice = t.select_i20(i20_inner)
    freeze_selection(i20_choice, out / "i20_selection.json")
    for k, directory in enumerate(es_inner, 1):
        progress.update(f"es_inner{k:02d}", inner_fold=k, completed_stages=5 + k - 1)
        group = t.attach_i20_inner(groups[f"inner{k:02d}"], i20_inner[k - 1], i20_choice)
        t.fit_es_inner(group, directory, fold, k, options)
        verify_stage(directory)
    progress.update("select_es", completed_stages=10)
    es_choice = t.select_es(es_inner)
    freeze_selection(es_choice, out / "es_selection.json")
    progress.update("i20_outer", inner_fold=None, completed_stages=10)
    outer = groups["outer"]
    t.fit_i20_outer(outer, i20_outer, fold, i20_choice, options)
    verify_stage(i20_outer)
    outer_es = t.attach_i20_outer(outer, i20_outer)
    progress.update("es_outer", completed_stages=11)
    t.fit_es_outer(outer_es, es_outer, fold, es_choice, options)
    verify_stage(es_outer)
    progress.update("replay_outer", completed_stages=12)
    replay_train, replay_val = t.replay_i20(outer, i20_outer, verify=True, atol=0., rtol=0.)
    np.testing.assert_array_equal(replay_train, outer_es["I20_train"])
    np.testing.assert_array_equal(replay_val, outer_es["I20_val"])
    replay = t.replay_es(outer_es, es_outer, verify=True, atol=0., rtol=0.)
    t.write_json(out / "replay_verification.json", dict(passed=True, atol=0., rtol=0.,
                 i20_train_shape=replay_train.shape, i20_val_shape=replay_val.shape,
                 es_val_shape=replay["selected_val"].shape, survival_shape=replay["survival"].shape,
                 neural_device=args.device, signature=signed["signature"]))
    progress.update("writing_final_predictions", completed_stages=12)
    rows = save_final_predictions(out, outer, replay, args.branch, repeat_id, fold)
    artifacts = [manifest, out / "PREPARED.json", out / "predictions_all20.csv", out / "isolation.json",
                 out / "i20_selection.json", out / "es_selection.json", out / "replay_verification.json",
                 *[directory / "COMPLETE.json" for directory in stage_dirs]]
    report = dict(status="complete", branch=args.branch, ordinal=args.ordinal, repeat_id=repeat_id, fold=fold,
                  signature=signed["signature"], production=True, device=args.device,
                  genuine_views=outer["x"].shape[0], paired_imputation_indices=20, prediction_rows=rows,
                  i20_choice=i20_choice["choice"], es_candidate=es_choice["candidate"],
                  completed_stages=12, replay_passed=True, elapsed_this_invocation_seconds=time.monotonic() - started,
                  completed_at=stamp(), files={p.relative_to(out).as_posix(): c.sha_file(p) for p in artifacts})
    t.write_json(completed, report)
    progress.update("complete", status="complete", completed_stages=12, prediction_rows=rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=BRANCHES)
    parser.add_argument("ordinal", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rscript", help="Rscript executable; otherwise CACOX_RSCRIPT or PATH")
    args = parser.parse_args()
    args.device = c.resolve_device(args.device)
    args.rscript = discover_rscript(args.rscript)
    if args.ordinal not in range(1, 51) or args.threads != 1:
        parser.error("ordinal must be 1..50 and this production wrapper uses --threads 1")
    repeat_id, fold = (args.ordinal - 1) // 5 + 1, (args.ordinal - 1) % 5 + 1
    folder = RUN_ROOT / "branches" / args.branch / f"repeat{repeat_id:02d}_fold{fold:02d}"
    out = folder / "cacox_es"
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    logger = logging.getLogger("cacox_fold")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(logs / "fold.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    lock = out / "RUNNING.lock"
    acquire_lock(lock)
    progress = Progress(out, args.branch, args.ordinal, logger)
    try:
        progress.update("exporting_raw_partitions")
        # Exactly one exporter invocation per fold command; individual ML stages
        # consume encoded arrays and never call R or repeat raw preprocessing.
        command = [args.rscript, "--vanilla", str(CODE / "export_neural.R"), args.branch, str(args.ordinal), "all"]
        with (logs / "raw_export.log").open("a", encoding="utf-8") as stream:
            env = dict(os.environ, CACOX_PORTABLE_ROOT=str(ROOT), CACOX_RUN_ROOT=str(RUN_ROOT),
                       CACOX_RSCRIPT=args.rscript, CACOX_PYTHON=sys.executable)
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        progress.update("encoding_and_isolation")
        groups, preparation = prepare_groups(folder / "neural_inputs", out, args.branch, args.ordinal)
        if args.prepare_only:
            progress.update("prepared", status="prepared", training_started=False,
                            genuine_views=groups["outer"]["x"].shape[0], isolation=preparation["isolation"])
            logger.info("PREPARE_ONLY_PASSED %s", json.dumps(preparation["isolation"], ensure_ascii=False))
            return
        report = run_training(groups, preparation, out, args, progress)
        logger.info("CACOX_FOLD_COMPLETE branch=%s ordinal=%d rows=%d", args.branch, args.ordinal, report["prediction_rows"])
    except BaseException as error:
        progress.update("failed", status="failed", error=repr(error))
        logger.exception("CACox fold did not complete")
        raise
    finally:
        progress.close()
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
