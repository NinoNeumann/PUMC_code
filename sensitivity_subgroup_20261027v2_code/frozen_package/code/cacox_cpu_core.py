"""Portable scientific primitives for the frozen CACox ES procedure.

Four self-contained source modules are bundled under vendor/model. No historical
runner, configuration selector, or result loader is imported. File hashes use
package-relative identifiers and do not depend on installation paths.

Float32 networks can run on CPU, Apple MPS or NVIDIA CUDA. All losses, trees,
mixing and calibration stay on CPU. Original Cox loss stays float32; shared
losses stay float64. CUDA support requires a target-machine
smoke test; the package-build machine has no NVIDIA hardware. Cross-device
bitwise training or in-place resumption of historical Mac runs is not promised.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import platform
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "vendor" / "model"
SOURCE_FILES = {
    "original": "original_network.py",
    "tabm": "tabm_backend.py",
    "additive": "additive_model.py",
    "pair": "pair_loss.py",
}
_MODULES: dict[str, Any] = {}

ORIGINAL_COMMON = dict(
    num_blocks=2, dropout=.15, num_cross_layers=2, num_heads=4,
    learning_rate=7e-6, weight_decay=.01, max_epochs=3000, patience=150,
    gradient_clip=100., hidden=96, seed_base=20261022,
)
SHARED_COMMON = dict(
    blocks=2, width=64, dropout=.15, weight_decay=.0003,
    gradient_clip=5., max_epochs=3000, patience=150,
)
ALPHA = (.5, .75, .5, .5, .5)
NAM_NAMES = ("A8_NLL", "A8_C", "A16_NLL", "A16_C")
FIT_NAMES = ("P_C", "C_J_SP", *NAM_NAMES)
CANDIDATES = [dict(candidate="C0_ref17", source="C_J_SP", weight17=1.)] + [
    dict(candidate=f"{name}_w{int(weight * 100):03d}", source=name, weight17=weight)
    for name in NAM_NAMES for weight in (.75, .5, .25, 0.)
]
CANDIDATE_ORDER = [c["candidate"] for c in CANDIDATES]
TREE_CANDIDATES = [
    dict(depth=d, leaf=l, trees=n)
    for d in (1, 2) for l in (25, 50) for n in (0, 75, 150, 300, 600)
]


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_id(path: str | Path) -> str:
    """Stable POSIX identifier for a file inside this movable package."""
    return Path(path).resolve().relative_to(PACKAGE_ROOT).as_posix()


def resolve_device(requested: str = "auto") -> str:
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("neural device must be auto, cpu, mps or cuda")
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable. Install a CUDA-enabled PyTorch build and NVIDIA driver, or explicitly choose CPU.")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS requested but unavailable. Choose CPU explicitly.")
    return requested


def device_identity(device: str) -> dict:
    result = dict(neural_device=device, system=platform.system(), machine=platform.machine())
    if device == "cuda":
        result.update(cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      device_name=torch.cuda.get_device_name(0),
                      compute_capability=list(torch.cuda.get_device_capability(0)))
    return result


def source_module(name: str):
    if name not in _MODULES:
        path = SOURCE_ROOT / SOURCE_FILES[name]
        spec = importlib.util.spec_from_file_location(f"_cacox_cpu_verified_{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load scientific source: {path}")
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _MODULES[name] = module
    return _MODULES[name]


def provenance(neural_device="cpu") -> dict:
    import tabm
    import xgboost
    vendor = json.loads((SOURCE_ROOT / "SOURCES.json").read_text(encoding="utf-8"))
    for relative, record in vendor["vendored"].items():
        if sha_file(SOURCE_ROOT / relative) != record["sha256"]:
            raise ValueError(f"Changed vendored scientific source: {relative}")
    return dict(
        source_root="vendor/model",
        source_sha256={f"vendor/model/{p}": sha_file(SOURCE_ROOT / p)
                       for p in [*vendor["vendored"], "SOURCES.json"]},
        adapted_code_sha256=sha_file(__file__),
        torch=torch.__version__, xgboost=xgboost.__version__,
        tabm=getattr(tabm, "__version__", "0.0.3 (package metadata)"),
        numpy=np.__version__, neural_device=neural_device,
        execution_identity=device_identity(neural_device),
        original_cox_loss_device="cpu", original_cox_loss_dtype="float32",
        shared_loss_device="cpu", tree_device="cpu", mixing_calibration_device="cpu",
        execution_flags=dict(deterministic_algorithms=True, allow_tf32=False,
                             cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        precision="float32 networks and tree gradients; float64 shared loss/mixing/calibration",
        differences=[
            f"{neural_device} neural execution with CPU tree/loss/mixing; no cross-device bitwise training equivalence claimed",
            "Changing neural device, hardware identity or scientific package versions requires a new run; moving the unchanged package directory does not",
            "n_views is explicit; complete-case uses one genuine view and eight neural members",
            "Twenty complete-case selection runs are not imputation views",
            "Original Cox cumulative-risk/tie approximation is intentionally retained",
            "Original float32 Cox loss and duration ordering use CPU on every neural device, avoiding the nondeterministic CUDA float cumsum kernel without changing the formula",
        ],
    )


def configure(threads: int = 1) -> None:
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_group(group: dict, require_i20: bool = False) -> dict:
    g = dict(group)
    for key in ("x", "vx"):
        a = np.asarray(g[key])
        if a.dtype != np.float32 or a.ndim != 3 or not np.isfinite(a).all():
            raise ValueError(f"{key} must be finite float32 [views, patients, features]")
        g[key] = a
    M, n, p = g["x"].shape
    if M not in (1, 20) or min(n, p) < 1:
        raise ValueError("Use exactly one complete-case view or twenty MI views")
    if g["vx"].shape[0] != M or g["vx"].shape[2] != p:
        raise ValueError("Train/query view or feature dimensions differ")
    for side, size in (("train", n), ("val", g["vx"].shape[1])):
        for field, dtype in (("time", float), ("event", bool)):
            a = np.asarray(g[f"{side}_{field}"], dtype=dtype)
            if a.shape != (size,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {side}_{field}")
            g[f"{side}_{field}"] = a
        ids = np.asarray(g[f"{side}_ids"])
        if ids.shape != (size,) or len(set(ids.tolist())) != size:
            raise ValueError(f"Invalid or duplicate {side}_ids")
        g[f"{side}_ids"] = ids
    if set(g["train_ids"].tolist()) & set(g["val_ids"].tolist()):
        raise ValueError("Train/validation patient overlap")
    if not g["train_event"].any():
        raise ValueError("No training events")
    metadata = dict(g.get("metadata", {}))
    if len(metadata.get("columns", [])) != p or not metadata.get("variables"):
        raise ValueError("metadata.columns and metadata.variables are required")
    source_module("additive").groups_from_metadata(metadata)
    g["metadata"] = metadata
    if require_i20:
        for key, size in (("I20_train", n), ("I20_val", g["vx"].shape[1])):
            a = np.asarray(g[key], dtype=np.float64)
            if a.shape != (M, size) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {key}")
            g[key] = a
    return g


def group_digest(g: dict, include_i20: bool = False) -> str:
    h = hashlib.sha256()
    keys = ["x", "vx", "train_time", "train_event", "val_time", "val_event",
            "train_ids", "val_ids"]
    if include_i20:
        keys += ["I20_train", "I20_val"]
    for key in keys:
        a = np.ascontiguousarray(g[key])
        h.update(key.encode())
        h.update(str((a.shape, a.dtype)).encode())
        if a.dtype.hasobject:
            h.update(json.dumps(a.tolist(), ensure_ascii=False).encode())
        else:
            h.update(a.tobytes())
    h.update(json.dumps(g["metadata"], sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def pair_indices(time, event):
    t, e = np.asarray(time, float), np.asarray(event, bool)
    return np.where(e[:, None] & ((t[:, None] < t[None, :]) |
                    ((t[:, None] == t[None, :]) & ~e[None, :])))


def concordance(scores, time, event) -> np.ndarray:
    s = np.asarray(scores, np.float64)
    i, j = pair_indices(time, event)
    if not len(i):
        raise ValueError("No comparable validation pairs")
    return ((s[..., i] > s[..., j]) + .5 * (s[..., i] == s[..., j])).mean(-1)


def validation_count(member_logits, time, event) -> tuple[int, float]:
    # Preserve the original FP32 -> numpy FP64 -> mean8 realization.
    a = member_logits.detach().double().cpu().numpy().mean(2)
    i, j = pair_indices(time, event)
    if not len(i):
        raise ValueError("No comparable validation pairs")
    twice = 2 * (a[:, i] > a[:, j]).astype(np.int64) + (a[:, i] == a[:, j])
    count = int(twice.sum())
    return count, count / (2 * a.shape[0] * len(i))


def original_loss(time, event, device="cpu"):
    # CUDA floating cumsum has no strict deterministic implementation in the
    # supported PyTorch build. Keep the original FP32 formula on CPU, including
    # ordering, and preserve autograd through score.to('cpu'). No detach/cast.
    t = torch.as_tensor(np.asarray(time), dtype=torch.float32, device="cpu")
    order = t.argsort(descending=True)
    e = torch.as_tensor(np.asarray(event), dtype=torch.float32, device="cpu")[order]
    fn = source_module("original").cox_ph_loss_sorted
    return lambda score: fn(score.reshape(-1).to("cpu")[order], e)


def make_original(features: int, seed: int, device="cpu"):
    seed_all(seed)
    c = ORIGINAL_COMMON
    return source_module("original").build_network(
        features, c["hidden"], c["num_blocks"], c["dropout"],
        c["num_cross_layers"], c["num_heads"]
    ).to(device)


def make_shared(features: int, metadata: dict, name: str, seed: int, device="cpu"):
    if name in ("P_C", "C_J_SP"):
        return source_module("tabm").make(
            features, dict(members=8, learning_rate=.001), seed, device=device)
    if name not in NAM_NAMES:
        raise ValueError(f"Unknown source {name}")
    seed_all(seed)
    groups = source_module("additive").groups_from_metadata(metadata)
    width = 8 if name.startswith("A8_") else 16
    return source_module("additive").GroupedAdditiveEnsemble(
        features, groups, width, .15, 8).to(device)


def predict_members(net, x: np.ndarray) -> np.ndarray:
    M, n, p = x.shape
    net.eval()
    with torch.no_grad():
        device = next(net.parameters()).device
        value = net(x_num=torch.as_tensor(x.reshape(M*n, p), device=device), x_cat=None).squeeze(-1)
    # MPS cannot hold float64. Cast only after transfer, matching saved precision.
    return value.cpu().double().numpy().reshape(M, n, 8)


def normalization(i20_train, nn_train) -> list[dict]:
    return [dict(i20_mean=float(a.mean()), i20_sd=max(float(a.std()), 1e-8),
                 kernel_mean=float(b.mean()), kernel_sd=max(float(b.std()), 1e-8))
            for a, b in zip(i20_train, nn_train)]


def combine_source(i20, nn, stats: list[dict], fold: int):
    a = ALPHA[fold-1]
    return np.stack([(1-a)*(u-s["i20_mean"])/s["i20_sd"] +
                     a*(v-s["kernel_mean"])/s["kernel_sd"]
                     for u, v, s in zip(i20, nn, stats)])


def combine_torch(nn_mean, i20, stats, fold):
    values = {key: torch.tensor([s[key] for s in stats], dtype=torch.float64)[:, None]
              for key in ("i20_mean", "i20_sd", "kernel_mean", "kernel_sd")}
    a = ALPHA[fold-1]
    return ((1-a)*(i20-values["i20_mean"])/values["i20_sd"] +
            a*(nn_mean-values["kernel_mean"])/values["kernel_sd"])


def blend_sources(source_train: dict, source_val: dict):
    control = np.asarray(source_train["C_J_SP"], np.float64)
    query = np.asarray(source_val["C_J_SP"], np.float64)
    aa, vv, all_stats = [control], [query], {}
    for name in NAM_NAMES:
        other = np.asarray(source_train[name], np.float64)
        qother = np.asarray(source_val[name], np.float64)
        stats = [dict(control_mean=float(a.mean()), control_sd=max(float(a.std()), 1e-8),
                      other_mean=float(b.mean()), other_sd=max(float(b.std()), 1e-8))
                 for a, b in zip(control, other)]
        all_stats[name] = stats
        aligned = np.stack([(v-s["other_mean"])/s["other_sd"]*s["control_sd"]+
                            s["control_mean"] for v, s in zip(other, stats)])
        qaligned = np.stack([(v-s["other_mean"])/s["other_sd"]*s["control_sd"]+
                             s["control_mean"] for v, s in zip(qother, stats)])
        for w in (.75, .5, .25, 0.):
            aa.append(aligned if w == 0. else w*control+(1-w)*aligned)
            vv.append(qaligned if w == 0. else w*query+(1-w)*qaligned)
    return np.stack(aa), np.stack(vv), all_stats


def breslow(time, event, score):
    # Same torch.float64 formula as cuda_breslow, on CPU.
    t = torch.as_tensor(np.asarray(time), dtype=torch.float64)
    e = torch.as_tensor(np.asarray(event), dtype=torch.bool)
    s = torch.as_tensor(np.asarray(score), dtype=torch.float64)
    times = torch.unique(t[e], sorted=True)
    if not len(times):
        raise ValueError("No events for baseline calibration")
    shift = torch.quantile(s, .5)
    risk = torch.exp(s-shift)
    increments = (((t[None, :] == times[:, None]) & e[None, :]).sum(1) /
                  ((t[None, :] >= times[:, None])*risk[None, :]).sum(1))
    return dict(event_times=times, cumulative_hazard=increments.cumsum(0), shift=shift)


def survival(score, baseline, horizons=(365, 730, 1095)):
    s = torch.as_tensor(np.asarray(score), dtype=torch.float64)
    idx = torch.searchsorted(baseline["event_times"], torch.tensor(horizons, dtype=torch.float64), right=True)
    ch = torch.cat([torch.zeros(1, dtype=torch.float64), baseline["cumulative_hazard"]])[idx]
    return torch.exp(-torch.exp(s-baseline["shift"])[:, None]*ch[None, :]).numpy()


def fit_rank(x, time, event, offset, vx, voffset, depth, leaf, counts, seed):
    """Original first-order hist boosting; explicit CPU; unit working Hessian."""
    import xgboost as xgb
    xx, vv = np.asarray(x, np.float32), np.asarray(vx, np.float32)
    oo, ov = np.asarray(offset, np.float32), np.asarray(voffset, np.float32)
    left, right = pair_indices(time, event)
    if not len(left):
        raise ValueError("No training survival-comparable pairs")
    i, j = torch.as_tensor(left), torch.as_tensor(right)
    data = xgb.QuantileDMatrix(xx, label=np.zeros(len(xx), np.float32), base_margin=oo, max_bin=32)
    found = {0: (np.asarray(offset).copy(), np.asarray(voffset).copy())} if 0 in counts else {}
    booster = xgb.Booster(dict(
        device="cpu", tree_method="hist", max_depth=int(depth), min_child_weight=int(leaf),
        eta=.03, subsample=.8, sampling_method="uniform", reg_lambda=0., reg_alpha=0.,
        gamma=0., max_bin=32, seed=int(seed), nthread=1, disable_default_eval_metric=1), [data])
    for iteration in range(max(counts)):
        pred = oo if iteration == 0 else booster.inplace_predict(xx, predict_type="margin", base_margin=oo)
        score = torch.from_numpy(pred)
        w = torch.sigmoid(score[j]-score[i])
        grad = torch.zeros_like(score)
        grad.scatter_add_(0, j, w)
        grad.scatter_add_(0, i, -w)
        grad = (grad*(len(score)/len(i))).contiguous()
        booster.boost(data, iteration, grad=grad.numpy(), hess=np.ones_like(pred))
        n = iteration+1
        if n in counts:
            found[n] = (
                booster.inplace_predict(xx, predict_type="margin", base_margin=oo).astype(float),
                booster.inplace_predict(vv, predict_type="margin", base_margin=ov).astype(float))
    return booster, found


def load_group_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        g = {k: z[k].copy() for k in z.files if k not in ("metadata", "metadata_json")}
        key = "metadata_json" if "metadata_json" in z.files else "metadata"
        if key in z.files:
            g["metadata"] = json.loads(str(z[key].item()))
        else:
            g["metadata"] = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return validate_group(g, require_i20="I20_train" in g)
