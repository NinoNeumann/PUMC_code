"""Staged, restartable portable fitting for CACox -> I20 -> E_selected.

The caller owns patient restrictions, folds, variable selection and imputation.
Each group contains x/vx [M,n,p], train/val time/event/ids and metadata with
variables/columns. M is one for complete cases or twenty for actual MI.

Usage order for each outer fold:
  fit_i20_inner on five inner groups; select_i20;
  attach_i20_inner + fit_es_inner on those five groups; select_es;
  fit_i20_outer on the outer development/test group;
  attach_i20_outer + fit_es_outer; replay_i20/replay_es for verification.

No outer test outcome is used for fitting, calibration, epoch selection or
candidate selection. The returned outer C is reporting only.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

import cacox_cpu_core as c


@dataclass(frozen=True)
class RunOptions:
    threads: int = 1
    neural_device: str = "auto"
    original_max_epochs: int = 3000
    shared_max_epochs: int = 3000
    joint_max_epochs: int = 500
    patience: int = 150
    tree_counts: tuple[int, ...] = (0, 75, 150, 300, 600)
    smoke_test: bool = False

    def __post_init__(self):
        object.__setattr__(self, "neural_device", c.resolve_device(self.neural_device))

    def validate(self):
        if not self.smoke_test:
            actual = asdict(self)
            expected = asdict(RunOptions(threads=self.threads, neural_device=self.neural_device))
            if actual != expected:
                raise ValueError("Scientific budgets may only be reduced for explicit smoke tests")
        if min(self.threads, self.original_max_epochs, self.shared_max_epochs,
               self.joint_max_epochs, self.patience) < 1:
            raise ValueError("Invalid execution options")
        if not self.tree_counts or min(self.tree_counts) < 0:
            raise ValueError("Invalid tree counts")
        c.resolve_device(self.neural_device)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix+".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(p)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def _trace(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _start(group, outdir, stage, fold, inner_fold, options, extra=None):
    options.validate()
    if fold not in range(1, 6) or (inner_fold is not None and inner_fold not in range(1, 6)):
        raise ValueError("fold/inner_fold must be one through five")
    g = c.validate_group(group, require_i20=stage.startswith("es_"))
    c.configure(options.threads)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    signed = _jsonable(dict(
        stage=stage, fold=fold, inner_fold=inner_fold,
        group_sha256=c.group_digest(g, include_i20=stage.startswith("es_")),
        options=asdict(options), extra=extra,
        source=c.provenance(options.neural_device), train_code_sha256=c.sha_file(__file__),
    ))
    manifest = out/"manifest.json"
    if manifest.exists() and read_json(manifest) != signed:
        raise ValueError(f"Existing stage belongs to different inputs/settings: {out}")
    write_json(manifest, signed)
    completed = out/"COMPLETE.json"
    if completed.exists():
        report = read_json(completed)
        if report["manifest_sha256"] != c.sha_file(manifest):
            raise ValueError("Changed stage manifest")
        for relative, expected in report["files"].items():
            if c.sha_file(out/relative) != expected:
                raise ValueError(f"Changed completed artifact: {out/relative}")
        return g, out, read_json(out/"summary.json")
    return g, out, None


def _finish(out, summary):
    write_json(out/"summary.json", summary)
    files = {p.relative_to(out).as_posix(): c.sha_file(p)
             for p in sorted(out.rglob("*"))
             if p.is_file() and p.name not in ("COMPLETE.json",) and not p.name.endswith(".tmp")}
    write_json(out/"COMPLETE.json", dict(manifest_sha256=c.sha_file(out/"manifest.json"), files=files))
    return _jsonable(summary)


def _fit_original(x, time_values, event, vx, val_time, val_event, seed, options,
                  fixed_epochs=None):
    if fixed_epochs is not None and not 1 <= fixed_epochs <= options.original_max_epochs:
        raise ValueError("Invalid original CACox frozen epoch count")
    started = time.perf_counter()
    device = options.neural_device
    net = c.make_original(x.shape[1], seed, device)
    X = torch.as_tensor(x, device=device)
    loss_fn = c.original_loss(time_values, event, device)
    val_fn = c.original_loss(val_time, val_event, device) if fixed_epochs is None else None
    VX = torch.as_tensor(vx, device=device) if val_fn is not None else None
    common = c.ORIGINAL_COMMON
    opt = torch.optim.Adam(net.parameters(), lr=common["learning_rate"],
                           weight_decay=common["weight_decay"])
    best, best_epoch, stale, state = float("inf"), 0, 0, None
    trace = []
    for epoch in range(1, (fixed_epochs or options.original_max_epochs)+1):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(X))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite original Cox loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), common["gradient_clip"], error_if_nonfinite=True)
        opt.step()
        row = dict(epoch=epoch, train_loss=float(loss.detach()))
        if val_fn is not None:
            net.eval()
            with torch.no_grad():
                value = float(val_fn(net(VX)))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite original validation loss")
            row["validation_loss"] = value
            if value < best:
                best, best_epoch, stale, state = value, epoch, 0, _state(net)
            else:
                stale += 1
        trace.append(row)
        if val_fn is not None and stale >= options.patience:
            break
    if val_fn is not None:
        net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        a = net(X).cpu().numpy().astype(float).reshape(-1)
        v = net(torch.as_tensor(vx, device=device)).cpu().numpy().astype(float).reshape(-1)
    audit = dict(seed=seed, best_epoch=best_epoch if val_fn is not None else epoch,
                 stopped_epoch=epoch, fixed_epochs_no_validation_forward=val_fn is None,
                 seconds=time.perf_counter()-started, common=common, neural_device=device,
                 loss_device="cpu", loss_dtype="float32", duration_order_device="cpu",
                 training_patients=len(x), training_events=int(np.asarray(event).sum()),
                 loss="original float32 cumulative Cox risk approximation; original ties retained")
    return net, a, v, trace, audit


def fit_i20_inner(group, outdir, fold, inner_fold, options=None):
    options = options or RunOptions()
    g, out, done = _start(group, outdir, "i20_inner", fold, inner_fold, options)
    if done is not None:
        return done
    M, n, p = g["x"].shape
    candidates = [dict(depth=d, leaf=l, trees=t)
                  for d in (1, 2) for l in (25, 50) for t in options.tree_counts]
    aa = np.empty((len(candidates), M, n), float)
    vv = np.empty((len(candidates), M, g["vx"].shape[1]), float)
    epochs = []
    for j in range(M):
        d = out/f"view{j+1:02d}"
        d.mkdir(exist_ok=True)
        seed = c.ORIGINAL_COMMON["seed_base"]+(fold-1)*1000+(j+1)*10000+inner_fold*100
        net, a, v, trace, audit = _fit_original(
            g["x"][j], g["train_time"], g["train_event"], g["vx"][j],
            g["val_time"], g["val_event"], seed, options)
        epochs.append(audit["best_epoch"])
        _save_torch(d/"cacox.pt", dict(kind="original", state=_state(net), features=p,
                                      seed=seed, metadata=g["metadata"], audit=audit))
        _trace(d/"original_training_trace.csv", trace)
        write_json(d/"original_audit.json", audit)
        np.savez_compressed(d/"original_predictions.npz", train=a, val=v)
        del net
        for depth in (1, 2):
            for leaf in (25, 50):
                booster, found = c.fit_rank(
                    g["x"][j], g["train_time"], g["train_event"], .5*a,
                    g["vx"][j], .5*v, depth, leaf, options.tree_counts,
                    100000+fold*1000+(j+1)*10+inner_fold)
                booster.save_model(d/f"tree_d{depth}_l{leaf}.ubj")
                for idx, candidate in enumerate(candidates):
                    if candidate["depth"] == depth and candidate["leaf"] == leaf:
                        aa[idx, j], vv[idx, j] = found[candidate["trees"]]
    metric = c.concordance(vv, g["val_time"], g["val_event"])
    np.savez_compressed(out/"grid.npz", train=aa, val=vv, C_index=metric,
                        train_ids=g["train_ids"], val_ids=g["val_ids"])
    return _finish(out, dict(stage="i20_inner", fold=fold, inner_fold=inner_fold,
                            n_views=M, candidates=candidates, best_epochs=epochs,
                            mean_C=metric.mean(1).tolist(), production=not options.smoke_test))


def select_i20(inner_dirs, destination=None):
    if len(inner_dirs) != 5:
        raise ValueError("Exactly five inner groups are required")
    summaries = [read_json(Path(d)/"summary.json") for d in inner_dirs]
    if {s["inner_fold"] for s in summaries} != set(range(1, 6)):
        raise ValueError("Distinct inner folds 1..5 required")
    candidates = summaries[0]["candidates"]
    if any(s["candidates"] != candidates or s["n_views"] != summaries[0]["n_views"] for s in summaries):
        raise ValueError("Inconsistent I20 candidate/view definitions")
    means = np.mean([s["mean_C"] for s in summaries], axis=0)
    chosen = min(range(len(candidates)), key=lambda i: (
        -means[i], candidates[i]["trees"], candidates[i]["depth"], -candidates[i]["leaf"]))
    epochs = np.asarray([s["best_epochs"] for s in summaries], int)
    result = dict(choice=candidates[chosen], candidate_index=chosen,
                  inner_C=float(means[chosen]), all_candidate_C=means.tolist(),
                  original_inner_epochs=epochs.tolist(),
                  original_outer_epochs=np.median(epochs, axis=0).astype(int).tolist(),
                  n_views=summaries[0]["n_views"],
                  rule="mean across genuine views within inner fold, then mean five folds; fewer trees, shallower, larger leaf",
                  outer_outcomes_used=False)
    if destination is not None:
        write_json(destination, result)
    return result


def attach_i20_inner(group, inner_dir, selected):
    summary = read_json(Path(inner_dir)/"summary.json")
    index = summary["candidates"].index(selected["choice"])
    with np.load(Path(inner_dir)/"grid.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(group["train_ids"], z["train_ids"])
        np.testing.assert_array_equal(group["val_ids"], z["val_ids"])
        return dict(group, I20_train=z["train"][index].copy(), I20_val=z["val"][index].copy())


def fit_i20_outer(group, outdir, fold, selected, options=None):
    options = options or RunOptions()
    g, out, done = _start(group, outdir, "i20_outer", fold, None, options, selected)
    if done is not None:
        return done
    M, n, p = g["x"].shape
    if selected["n_views"] != M or len(selected["original_outer_epochs"]) != M:
        raise ValueError("Frozen I20 views differ from outer group")
    aa, vv, ca, cv = [], [], [], []
    choice = selected["choice"]
    for j in range(M):
        d = out/f"view{j+1:02d}"
        d.mkdir(exist_ok=True)
        seed = c.ORIGINAL_COMMON["seed_base"]+(fold-1)*1000+j+1
        net, a, v, trace, audit = _fit_original(
            g["x"][j], g["train_time"], g["train_event"], g["vx"][j],
            g["val_time"], g["val_event"], seed, options,
            fixed_epochs=selected["original_outer_epochs"][j])
        _save_torch(d/"cacox.pt", dict(kind="original", state=_state(net), features=p,
                                      seed=seed, metadata=g["metadata"], audit=audit))
        _trace(d/"original_training_trace.csv", trace)
        write_json(d/"original_audit.json", audit)
        ca.append(a); cv.append(v)
        del net
        if choice["trees"]:
            booster, found = c.fit_rank(
                g["x"][j], g["train_time"], g["train_event"], .5*a,
                g["vx"][j], .5*v, choice["depth"], choice["leaf"], [choice["trees"]],
                100000+fold*1000+(j+1)*10+9)
            booster.save_model(d/"tree.ubj")
            a, v = found[choice["trees"]]
        else:
            a, v = .5*a, .5*v
        aa.append(a); vv.append(v)
    np.savez_compressed(out/"predictions.npz", I20_train=np.stack(aa), I20_val=np.stack(vv),
                        CACox_train=np.stack(ca), CACox_val=np.stack(cv),
                        train_ids=g["train_ids"], val_ids=g["val_ids"])
    return _finish(out, dict(stage="i20_outer", fold=fold, n_views=M,
                            choice=choice, selected=selected, production=not options.smoke_test))


def attach_i20_outer(group, outer_dir):
    with np.load(Path(outer_dir)/"predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(group["train_ids"], z["train_ids"])
        np.testing.assert_array_equal(group["val_ids"], z["val_ids"])
        return dict(group, I20_train=z["I20_train"].copy(), I20_val=z["I20_val"].copy())


def _fit_shared(g, name, fold, inner_fold, options, warm=None, fixed_epochs=None):
    M, n, p = g["x"].shape
    seed = 1010000+fold*1000+(9 if inner_fold is None else inner_fold)*100
    joint = name == "C_J_SP"
    maximum = options.joint_max_epochs if joint else options.shared_max_epochs
    if fixed_epochs is not None and not (0 if joint else 1) <= fixed_epochs <= maximum:
        raise ValueError("Invalid frozen shared-network epoch count")
    if joint and warm is None:
        raise ValueError("C_J_SP requires the P_C warmstart from the same group")
    started = time.perf_counter()
    device = options.neural_device
    net = c.make_shared(p, g["metadata"], name, seed, device)
    stats = None
    if joint:
        net.load_state_dict(warm["state"])
        stats = c.normalization(g["I20_train"], warm["train_members"].mean(2))
    X = torch.as_tensor(g["x"].reshape(M*n, p), device=device)
    pair_loss = c.source_module("pair").MeanRiskPairLoss(g["train_time"], g["train_event"], "cpu")
    train_cox = c.source_module("tabm").Efron(g["train_time"], g["train_event"], device="cpu")
    tree = torch.as_tensor(g["I20_train"], dtype=torch.float64)
    if fixed_epochs is None:
        nv = g["vx"].shape[1]
        VX = torch.as_tensor(g["vx"].reshape(M*nv, p), device=device)
        val_cox = c.source_module("tabm").Efron(g["val_time"], g["val_event"], device="cpu")
        vtree = torch.as_tensor(g["I20_val"], dtype=torch.float64)
    opt = torch.optim.AdamW(net.parameters(), lr=.0001 if joint else .001,
                            weight_decay=c.SHARED_COMMON["weight_decay"])
    monitor = "NLL" if name.endswith("_NLL") else "C"
    best = float("inf") if monitor == "NLL" else -1
    best_epoch, stale, best_state = 0, 0, _state(net) if joint else None
    trace = []

    def logits(tensor, rows):
        # Transfer preserves autograd. No float64 tensor is ever moved to MPS.
        return net(x_num=tensor, x_cat=None).squeeze(-1).reshape(M, rows, 8).to("cpu")

    def objective(eta):
        if joint:
            combined = c.combine_torch(eta.double().mean(2), tree, stats, fold)
            return pair_loss(combined.unsqueeze(2))
        return pair_loss(eta)

    def evaluate(row, initial=False):
        nonlocal best, best_epoch, stale, best_state
        net.eval()
        with torch.no_grad():
            eta = logits(VX, nv)
            if joint:
                score = c.combine_torch(eta.double().mean(2), vtree, stats, fold)
                count, value_C = c.validation_count(score.unsqueeze(2), g["val_time"], g["val_event"])
                value = float(val_cox(score.T))
            else:
                value = float(val_cox(eta.double().mean(2).T))
                count, value_C = c.validation_count(eta, g["val_time"], g["val_event"])
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite shared validation Efron")
        metric = value if monitor == "NLL" else count
        better = metric < best if monitor == "NLL" else metric > best
        if better:
            best, best_epoch, stale, best_state = metric, row["epoch"], 0, _state(net)
        elif not initial:
            stale += 1
        row.update(validation_Efron=value, validation_C=value_C,
                   validation_twice_wins=count, best_epoch=best_epoch, stale=stale)

    if joint:
        net.eval()
        with torch.no_grad():
            initial_loss = float(objective(logits(X, n)))
        row = dict(epoch=0, training_loss=initial_loss, initial_eval_not_optimizer_step=True)
        if fixed_epochs is None:
            evaluate(row, True)
        trace.append(row)
    epochs_to_run = maximum if fixed_epochs is None else fixed_epochs
    for epoch in range(1, epochs_to_run+1):
        net.train()
        opt.zero_grad(set_to_none=True)
        eta = logits(X, n)
        loss = objective(eta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite shared ranking objective")
        # Source code computes the Efron diagnostic even for pure pair training.
        diagnostic = None if joint else float(train_cox(eta.permute(1, 0, 2).reshape(n, M*8)).detach())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        row = dict(epoch=epoch, training_loss=float(loss.detach()),
                   gradient_norm_before_clip=float(norm), initial_eval_not_optimizer_step=False)
        if diagnostic is not None:
            row["training_member_Efron_diagnostic"] = diagnostic
        if fixed_epochs is None:
            evaluate(row)
        trace.append(row)
        if fixed_epochs is None and stale >= options.patience:
            break
    stopped_epoch = trace[-1]["epoch"]
    if fixed_epochs is None:
        net.load_state_dict(best_state)
    net.eval()
    a, v = c.predict_members(net, g["x"]), c.predict_members(net, g["vx"])
    audit = dict(name=name, seed=seed, monitor="combined_C" if joint else monitor,
                 loss="softplus comparable-pair loss on final combined score" if joint else "softplus comparable-pair loss on mean eight-member score",
                 n_views=M, members=8, independent_patients=n, batch_rows=M*n,
                 neural_device=device, shared_loss_device="cpu", shared_loss_dtype="float64",
                 best_epoch=best_epoch if fixed_epochs is None else fixed_epochs,
                 stopped_epoch=stopped_epoch, seconds=time.perf_counter()-started,
                 fixed_epochs_no_validation_forward=fixed_epochs is not None,
                 normalization=stats, original_hyperparameters=c.SHARED_COMMON,
                 learning_rate=.0001 if joint else .001,
                 parameters=sum(p.numel() for p in net.parameters()))
    return net, a, v, trace, audit


def _fit_es(g, out, fold, inner_fold, options, budgets=None, selected=None):
    M = g["x"].shape[0]
    source_a, source_v, epochs = {}, {}, {}
    warm = None
    for name in c.FIT_NAMES:
        frozen_epochs = None if budgets is None else int(budgets[name])
        net, a, v, trace, audit = _fit_shared(
            g, name, fold, inner_fold, options, warm=warm, fixed_epochs=frozen_epochs)
        d = out/"fits"/name
        d.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(kind="shared", name=name, state=_state(net), features=g["x"].shape[2],
                          metadata=g["metadata"], seed=audit["seed"], audit=audit,
                          normalization=audit["normalization"])
        _save_torch(d/"neural.pt", checkpoint)
        np.savez_compressed(d/"predictions.npz", train_members=a, val_members=v)
        _trace(d/"training_trace.csv", trace)
        write_json(d/"audit.json", audit)
        epochs[name] = audit["best_epoch"]
        if name == "P_C":
            warm = dict(state=checkpoint["state"], train_members=a)
        else:
            stats = audit["normalization"] if name == "C_J_SP" else c.normalization(g["I20_train"], a.mean(2))
            source_a[name] = c.combine_source(g["I20_train"], a.mean(2), stats, fold)
            source_v[name] = c.combine_source(g["I20_val"], v.mean(2), stats, fold)
        del net
    aa, vv, alignment = c.blend_sources(source_a, source_v)
    metric = c.concordance(vv, g["val_time"], g["val_event"])
    np.savez_compressed(out/"risks.npz", train=aa, val=vv, C_index=metric,
                        train_ids=g["train_ids"], val_ids=g["val_ids"])
    summary = dict(stage="es_inner" if inner_fold is not None else "es_outer", fold=fold,
                   inner_fold=inner_fold, n_views=M, members=8,
                   candidates=c.CANDIDATE_ORDER, best_epochs=epochs,
                   mean_C=metric.mean(1).tolist(), training_alignment=alignment,
                   source_alpha=c.ALPHA[fold-1], production=not options.smoke_test)
    if selected is not None:
        index = c.CANDIDATE_ORDER.index(selected["candidate"])
        baselines = [c.breslow(g["train_time"], g["train_event"], a) for a in aa[index]]
        surv = np.stack([c.survival(v, b) for v, b in zip(vv[index], baselines)])
        _save_torch(out/"baselines.pt", baselines)
        np.savez_compressed(out/"selected_predictions.npz", train=aa[index], val=vv[index],
                            survival=surv, horizons=np.array([365, 730, 1095]),
                            train_ids=g["train_ids"], val_ids=g["val_ids"])
        summary.update(selected=selected, candidate=selected["candidate"],
                       outer_C_per_view=metric[index].tolist())
    return _finish(out, summary)


def fit_es_inner(group, outdir, fold, inner_fold, options=None):
    options = options or RunOptions()
    g, out, done = _start(group, outdir, "es_inner", fold, inner_fold, options)
    return done if done is not None else _fit_es(g, out, fold, inner_fold, options)


def select_es(inner_dirs, destination=None):
    if len(inner_dirs) != 5:
        raise ValueError("Exactly five inner groups are required")
    summaries = [read_json(Path(d)/"summary.json") for d in inner_dirs]
    if {s["inner_fold"] for s in summaries} != set(range(1, 6)):
        raise ValueError("Distinct inner folds 1..5 required")
    if any(s["candidates"] != c.CANDIDATE_ORDER or s["n_views"] != summaries[0]["n_views"] for s in summaries):
        raise ValueError("Inconsistent ES candidates/views")
    means = np.mean([s["mean_C"] for s in summaries], axis=0)
    index = next(i for i, value in enumerate(means) if value >= means.max()-1e-12)
    epochs = {name: [s["best_epochs"][name] for s in summaries] for name in c.FIT_NAMES}
    result = dict(candidate=c.CANDIDATE_ORDER[index], candidate_index=index,
                  inner_C=float(means[index]), all_candidate_C=means.tolist(),
                  inner_epochs=epochs,
                  epochs={name: int(np.ceil(np.median(values))) for name, values in epochs.items()},
                  n_views=summaries[0]["n_views"], outer_outcomes_used=False,
                  rule="mean genuine views within inner fold, then mean five folds; 1e-12 tie and fixed candidate order")
    if destination is not None:
        write_json(destination, result)
    return result


def fit_es_outer(group, outdir, fold, selected, options=None):
    options = options or RunOptions()
    g, out, done = _start(group, outdir, "es_outer", fold, None, options, selected)
    if selected["n_views"] != g["x"].shape[0]:
        raise ValueError("Frozen ES views differ from outer group")
    return done if done is not None else _fit_es(g, out, fold, None, options, selected["epochs"], selected)


def _assert_prediction(actual, expected, atol=0., rtol=0.):
    if atol == 0. and rtol == 0.:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def replay_i20(group, outdir, verify=True, atol=0., rtol=0.):
    """Recompute saved inner grids or outer I20 risk without training."""
    import xgboost as xgb
    g = c.validate_group(group)
    out = Path(outdir)
    summary = read_json(out/"summary.json")
    inner = summary["stage"] == "i20_inner"
    if not inner and summary["stage"] != "i20_outer":
        raise ValueError("Not an I20 stage")
    candidates = summary["candidates"] if inner else [summary["choice"]]
    M, n, _ = g["x"].shape
    aa = np.empty((len(candidates), M, n), float)
    vv = np.empty((len(candidates), M, g["vx"].shape[1]), float)
    for j in range(M):
        d = out/f"view{j+1:02d}"
        obj = torch.load(d/"cacox.pt", map_location="cpu", weights_only=False)
        device = obj["audit"].get("neural_device", "cpu")
        net = c.make_original(obj["features"], obj["seed"], device)
        net.load_state_dict(obj["state"]); net.eval()
        with torch.no_grad():
            a = net(torch.as_tensor(g["x"][j], device=device)).cpu().numpy().astype(float).reshape(-1)
            v = net(torch.as_tensor(g["vx"][j], device=device)).cpu().numpy().astype(float).reshape(-1)
        boosters = {}
        for index, choice in enumerate(candidates):
            if choice["trees"] == 0:
                aa[index, j], vv[index, j] = .5*a, .5*v
                continue
            filename = f'tree_d{choice["depth"]}_l{choice["leaf"]}.ubj' if inner else "tree.ubj"
            if filename not in boosters:
                boosters[filename] = xgb.Booster()
                boosters[filename].load_model(d/filename)
                boosters[filename].set_param(dict(device="cpu", nthread=1))
            booster = boosters[filename]
            aa[index, j] = booster.inplace_predict(
                g["x"][j], predict_type="margin", base_margin=np.asarray(.5*a, np.float32),
                iteration_range=(0, choice["trees"])).astype(float)
            vv[index, j] = booster.inplace_predict(
                g["vx"][j], predict_type="margin", base_margin=np.asarray(.5*v, np.float32),
                iteration_range=(0, choice["trees"])).astype(float)
    if verify:
        with np.load(out/("grid.npz" if inner else "predictions.npz"), allow_pickle=False) as z:
            expected_a = z["train"] if inner else z["I20_train"][None]
            expected_v = z["val"] if inner else z["I20_val"][None]
            _assert_prediction(aa, expected_a, atol, rtol)
            _assert_prediction(vv, expected_v, atol, rtol)
    return (aa, vv) if inner else (aa[0], vv[0])


def replay_es(group, outdir, verify=True, atol=0., rtol=0.):
    """Load each source weight, recompute all 17 mixtures and optional survival."""
    g = c.validate_group(group, require_i20=True)
    out = Path(outdir)
    summary = read_json(out/"summary.json")
    fold = summary["fold"]
    source_a, source_v = {}, {}
    for name in ("C_J_SP", *c.NAM_NAMES):
        d = out/"fits"/name
        obj = torch.load(d/"neural.pt", map_location="cpu", weights_only=False)
        device = obj["audit"].get("neural_device", "cpu")
        net = c.make_shared(obj["features"], obj["metadata"], name, obj["seed"], device)
        net.load_state_dict(obj["state"])
        a, v = c.predict_members(net, g["x"]), c.predict_members(net, g["vx"])
        if verify:
            with np.load(d/"predictions.npz", allow_pickle=False) as z:
                _assert_prediction(a, z["train_members"], atol, rtol)
                _assert_prediction(v, z["val_members"], atol, rtol)
        stats = obj["normalization"] if name == "C_J_SP" else c.normalization(g["I20_train"], a.mean(2))
        source_a[name] = c.combine_source(g["I20_train"], a.mean(2), stats, fold)
        source_v[name] = c.combine_source(g["I20_val"], v.mean(2), stats, fold)
    aa, vv, _ = c.blend_sources(source_a, source_v)
    if verify:
        with np.load(out/"risks.npz", allow_pickle=False) as z:
            _assert_prediction(aa, z["train"], atol, rtol)
            _assert_prediction(vv, z["val"], atol, rtol)
    result = dict(train=aa, val=vv)
    if "candidate" in summary:
        index = c.CANDIDATE_ORDER.index(summary["candidate"])
        baselines = torch.load(out/"baselines.pt", map_location="cpu", weights_only=False)
        surv = np.stack([c.survival(v, b) for v, b in zip(vv[index], baselines)])
        if verify:
            with np.load(out/"selected_predictions.npz", allow_pickle=False) as z:
                _assert_prediction(vv[index], z["val"], atol, rtol)
                _assert_prediction(surv, z["survival"], atol, rtol)
        result.update(selected_train=aa[index], selected_val=vv[index], survival=surv)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["i20-inner", "i20-outer", "es-inner", "es-outer", "replay-i20", "replay-es"])
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--selection")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--neural-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    group = c.load_group_npz(args.group)
    options = RunOptions(threads=args.threads, neural_device=args.neural_device)
    selected = read_json(args.selection) if args.selection else None
    if args.stage == "i20-inner":
        result = fit_i20_inner(group, args.output, args.fold, args.inner_fold, options)
    elif args.stage == "i20-outer":
        result = fit_i20_outer(group, args.output, args.fold, selected, options)
    elif args.stage == "es-inner":
        result = fit_es_inner(group, args.output, args.fold, args.inner_fold, options)
    elif args.stage == "es-outer":
        result = fit_es_outer(group, args.output, args.fold, selected, options)
    elif args.stage == "replay-i20":
        a, v = replay_i20(group, args.output)
        result = dict(replay="passed", train_shape=a.shape, val_shape=v.shape)
    else:
        result = replay_es(group, args.output)
        result = dict(replay="passed", shapes={k: v.shape for k, v in result.items()})
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
