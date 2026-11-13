"""Small synthetic parity and end-to-end serialization checks; never clinical fits."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import xgboost

import cacox_cpu_core as c
import cacox_cpu_train as t


def synthetic_group(M):
    rng = np.random.RandomState(1047)
    n, nv, p = 36, 18, 3
    x = rng.normal(size=(M, n, p)).astype(np.float32)
    vx = rng.normal(size=(M, nv, p)).astype(np.float32)
    return dict(x=x, vx=vx, train_time=np.arange(1, n+1, dtype=float)*40,
                val_time=np.arange(1, nv+1, dtype=float)*60,
                train_event=np.arange(n) % 3 != 0, val_event=np.arange(nv) % 3 != 0,
                train_ids=np.arange(n), val_ids=np.arange(1000, 1000+nv),
                metadata=dict(variables=["Age", "Scr", "CRP"], columns=["Age", "Scr", "CRP"],
                              data="synthetic software validation", n_views=M))


def source_cpu_tree_parity():
    """Execute the source CPU branch without importing its unrelated CuPy code."""
    source_path = c.SOURCE_ROOT/"reference/tree_functions.py"
    tree = ast.parse(source_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in {"sync", "comparable_pairs", "rank_gradient", "fit_rank"}]
    namespace = dict(np=np, torch=torch, xgb=xgboost, cp=None, json=json, time=time)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
    g = synthetic_group(1)
    a = np.linspace(-.7, .8, len(g["train_time"]))
    v = np.linspace(-.6, .9, len(g["val_time"]))
    args = (g["x"][0], g["train_time"], g["train_event"], a,
            g["vx"][0], v, 2, 5, (0, 1, 3), 1873)
    _, expected, _ = namespace["fit_rank"](*args, device="cpu")
    _, actual = c.fit_rank(*args)
    for count in expected:
        for a, b in zip(expected[count], actual[count]):
            np.testing.assert_array_equal(a, b)
    return dict(source_cpu_tree_branch="bitwise predictions equal", counts=[0, 1, 3])


def original_loss_device_route():
    """Routing check only: no CUDA hardware or CUDA allocation is required."""
    group = synthetic_group(1)
    expected_value, expected_grad = None, None
    for neural_device in ("cpu", "mps", "cuda"):
        scores = torch.linspace(-.8, .9, len(group["train_time"]), requires_grad=True)
        value = c.original_loss(group["train_time"], group["train_event"], neural_device)(scores)
        value.backward()
        assert value.device.type == "cpu" and value.dtype == torch.float32
        if expected_value is None:
            expected_value, expected_grad = value.detach(), scores.grad.clone()
        else:
            torch.testing.assert_close(value.detach(), expected_value, rtol=0, atol=0)
            torch.testing.assert_close(scores.grad, expected_grad, rtol=0, atol=0)
    return dict(routes=["cpu", "mps", "cuda"], loss_device="cpu", loss_dtype="float32",
                loss_and_gradient="bitwise equal", cuda_hardware_test=False)


def source_training_parity():
    """Compare unchanged source training loops, changing only CUDA allocation.

    The original fit functions are compiled from their AST, without importing
    any historical runner. CUDA allocations and synchronize are redirected to
    CPU; all loss/optimizer statements remain the original source statements.
    """
    import pandas as pd

    class TorchCPU:
        cuda = SimpleNamespace(synchronize=lambda: None)
        def __getattr__(self, key):
            return getattr(torch, key)
        @staticmethod
        def tensor(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return torch.tensor(*args, **kwargs)

    def function(relative, name, namespace):
        sources = json.loads((c.SOURCE_ROOT/"SOURCES.json").read_text(encoding="utf-8"))
        path = c.SOURCE_ROOT/sources["reference_function_files"][relative]
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]

    def equal_state(a, b):
        assert a.state_dict().keys() == b.state_dict().keys()
        for key in a.state_dict():
            np.testing.assert_array_equal(a.state_dict()[key].detach().numpy(),
                                          b.state_dict()[key].detach().numpy())

    g = synthetic_group(20)
    g["I20_train"] = np.linspace(-1, 1, len(g["train_time"]))[None].repeat(20, 0)
    g["I20_val"] = np.linspace(-.8, .8, len(g["val_time"]))[None].repeat(20, 0)
    options = t.RunOptions(neural_device="cpu", original_max_epochs=2, shared_max_epochs=2,
                           joint_max_epochs=2, patience=2, tree_counts=(0, 1), smoke_test=True)
    seed = 1010000+1000+100
    source_pair = lambda tt, ee, device: c.source_module("pair").MeanRiskPairLoss(tt, ee, "cpu")
    shared_b = SimpleNamespace(
        make=lambda p, config, ss: c.make_shared(p, g["metadata"], "P_C", ss),
        tensor=lambda x: torch.as_tensor(x),
        Efron=lambda tt, ee: c.source_module("tabm").Efron(tt, ee, device="cpu"),
        output=lambda net, x: net(x_num=x, x_cat=None).squeeze(-1))
    parent = SimpleNamespace(b=shared_b, initial_digest=lambda net: "validation-only")
    class ValidationC:
        def __init__(self, group):
            self.group = group
            self.denominator = 2*group["x"].shape[0]*len(c.pair_indices(group["val_time"], group["val_event"])[0])
        def __call__(self, eta):
            return c.validation_count(eta, self.group["val_time"], self.group["val_event"])
    previous = SimpleNamespace(ValidationC=ValidationC)
    common = dict(c.SHARED_COMMON, max_epochs=2, patience=2)
    config = dict(config_id="c0", consistency=0., members=8, learning_rate=.001)
    namespace = dict(np=np, torch=TorchCPU(), copy=__import__("copy"), time=time,
                     parent=parent, previous=previous,
                     prototype=SimpleNamespace(MeanRiskPairLoss=source_pair),
                     COMMON=common, CONFIG=config, seed=lambda f, k: seed,
                     m=SimpleNamespace(stamp=lambda: "validation-only"))
    ref_fit = function("experiments/cacox_shared_rank_objective/backend.py", "fit", namespace)
    reference, _, _ = ref_fit(g, dict(rank_weight=1., monitor="C"), 1, 1, fixed_epochs=2)
    actual, a, _, _, _ = t._fit_shared(g, "P_C", 1, 1, options, fixed_epochs=2)
    equal_state(reference, actual)
    warm = dict(state=t._state(actual), train_members=a)
    names = ["P_C"]
    reference, _, ref_audit = ref_fit(g, dict(rank_weight=1., monitor="C"), 1, 1)
    actual, _, _, _, actual_audit = t._fit_shared(g, "P_C", 1, 1, options)
    equal_state(reference, actual)
    assert ref_audit["best_epoch"] == actual_audit["best_epoch"]

    for name in c.NAM_NAMES:
        width, monitor = (8 if name.startswith("A8_") else 16), name.split("_")[1]
        condition = dict(condition_id=name, rank_weight=1., monitor=monitor, width=width)
        ns = dict(namespace, legacy=SimpleNamespace(prototype=SimpleNamespace(MeanRiskPairLoss=source_pair)),
                  CONDITIONS=[condition],
                  training_spec=lambda group, cond: dict(features=group["x"].shape[2],
                                                          width=cond["width"]),
                  make_from_spec=lambda spec, ss, name=name: c.make_shared(
                      spec["features"], g["metadata"], name, ss))
        ref_fit = function("experiments/cacox_grouped_additive/backend.py", "fit", ns)
        reference, _, _ = ref_fit(g, condition, 1, 1, fixed_epochs=2)
        actual, _, _, _, _ = t._fit_shared(g, name, 1, 1, options, fixed_epochs=2)
        equal_state(reference, actual)
        reference, _, ref_audit = ref_fit(g, condition, 1, 1)
        actual, _, _, _, actual_audit = t._fit_shared(g, name, 1, 1, options)
        equal_state(reference, actual)
        assert ref_audit["best_epoch"] == actual_audit["best_epoch"]
        names.append(name)

    stats = c.normalization(g["I20_train"], warm["train_members"].mean(2))
    ns = dict(namespace,
              legacy=SimpleNamespace(prototype=SimpleNamespace(MeanRiskPairLoss=source_pair), previous=previous),
              bounded=SimpleNamespace(MeanRiskSigmoid=lambda *args: None),
              source=lambda group, f, k, w: (dict(state=warm["state"]), stats, {}),
              combine_torch=c.combine_torch,
              COMMON=common, CONFIG=dict(config, learning_rate=.0001))
    ref_fit = function("experiments/cacox_joint_score_finetune/backend.py", "fit", ns)
    condition = dict(condition_id="C_J_SP", warmstart="C", target="J_SP", temperature=1.)
    reference, _, _ = ref_fit(g, condition, 1, 1, fixed_epochs=2)
    actual, _, _, _, _ = t._fit_shared(g, "C_J_SP", 1, 1, options, warm=warm, fixed_epochs=2)
    equal_state(reference, actual)
    reference, _, ref_audit = ref_fit(g, condition, 1, 1)
    actual, _, _, _, actual_audit = t._fit_shared(g, "C_J_SP", 1, 1, options, warm=warm)
    equal_state(reference, actual)
    assert ref_audit["best_epoch"] == actual_audit["best_epoch"]
    names.append("C_J_SP")

    class IdentityEncoder:
        def __init__(self, variables, scale):
            self.variables = variables
        def fit_transform(self, frame):
            return frame[self.variables].to_numpy(np.float32)
        transform = fit_transform
    frame = pd.DataFrame(g["x"][0], columns=g["metadata"]["columns"])
    frame["Survival_Days"] = g["train_time"]
    frame["Death"] = g["train_event"]
    frame["patient_id"] = g["train_ids"]
    ns = dict(np=np, pd=pd, torch=torch, Encoder=IdentityEncoder,
              make_model=lambda p, spec, common, ss: c.make_original(p, ss),
              feature_digest=lambda net: "validation-only", breslow=lambda *args: None,
              FittedModel=lambda net, *args: SimpleNamespace(net=net),
              original_loss=lambda frame, device: c.original_loss(frame.Survival_Days, frame.Death, device))
    ref_fit = function("experiments/cacox_original_head_study/src/cacox_original_head_study/training.py", "fit", ns)
    reference, _, _ = ref_fit(frame, g["metadata"]["variables"], dict(kind="cox"),
                               dict(c.ORIGINAL_COMMON, scale="robust", device="cpu"), 971,
                               fixed_epochs=2)
    actual, _, _, _, _ = t._fit_original(
        g["x"][0], g["train_time"], g["train_event"], g["vx"][0],
        g["val_time"], g["val_event"], 971, options, fixed_epochs=2)
    equal_state(reference.net, actual)
    return dict(source_training_loops="all state tensors bitwise equal after two optimizer steps",
                shared_validation_monitors="all six early-stop best states and epoch choices match source loops",
                sources=["original_CACox", *names], n_views=20,
                redirection="CPU tensor allocation and CUDA synchronization only")


def run(output, neural_device="cpu"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    c.configure(1)
    neural_device = c.resolve_device(neural_device)
    result = dict(neural_device=neural_device, original_cox_routing=original_loss_device_route(),
                  tree=source_cpu_tree_parity(),
                  training=source_training_parity(), views={})
    options = t.RunOptions(neural_device=neural_device, original_max_epochs=2,
                           shared_max_epochs=2, joint_max_epochs=2, patience=2,
                           tree_counts=(0, 1, 2), smoke_test=True)
    for M in (1, 20):
        start = time.perf_counter()
        g = synthetic_group(M)
        path = root/f"M{M}"
        inner = path/"i20_inner"
        isum = t.fit_i20_inner(g, inner, 1, 1, options)
        t.replay_i20(g, inner)
        choice = dict(depth=1, leaf=25, trees=2)
        selected_i = dict(choice=choice, n_views=M, original_outer_epochs=isum["best_epochs"])
        eg = t.attach_i20_inner(g, inner, selected_i)
        e_inner = path/"es_inner"
        esum = t.fit_es_inner(eg, e_inner, 1, 1, options)
        t.replay_es(eg, e_inner)
        # Use a predetermined candidate, not synthetic outer outcome selection.
        selected_e = dict(candidate="A16_C_w050", n_views=M, epochs=esum["best_epochs"])
        outer = path/"i20_outer"
        t.fit_i20_outer(g, outer, 1, selected_i, options)
        t.replay_i20(g, outer)
        og = t.attach_i20_outer(g, outer)
        e_outer = path/"es_outer"
        t.fit_es_outer(og, e_outer, 1, selected_e, options)
        replay = t.replay_es(og, e_outer)
        np.testing.assert_equal(replay["survival"].shape, (M, len(g["val_time"]), 3))
        assert np.isfinite(replay["survival"]).all()
        assert ((replay["survival"] >= 0) & (replay["survival"] <= 1)).all()
        assert (np.diff(replay["survival"], axis=2) <= 1e-14).all()
        # Resume validates every recorded artifact before returning cached output.
        again = t.fit_es_outer(og, e_outer, 1, selected_e, options)
        assert again["candidate"] == selected_e["candidate"]
        result["views"][str(M)] = dict(
            seconds=time.perf_counter()-start, all_saved_replay="bitwise equal",
            risk_shape=list(replay["val"].shape), survival_shape=list(replay["survival"].shape),
            resume="passed", production=False)
        print(json.dumps(result["views"][str(M)]), flush=True)
    t.write_json(root/"VALIDATION.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neural-device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.neural_device), indent=2))
