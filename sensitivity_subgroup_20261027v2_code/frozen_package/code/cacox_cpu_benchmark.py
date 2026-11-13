"""Synthetic CPU/accelerator forward, loss, gradient and short-step benchmark.

Not a clinical fit. Shared ranking operations remain CPU float64, including
autograd; only float32 networks/inputs use Apple MPS or NVIDIA CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

import cacox_cpu_core as c
import cacox_cpu_train as t


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def compare(output, accelerator="auto"):
    accelerator = c.resolve_device(accelerator)
    if accelerator == "cpu":
        raise RuntimeError("This comparison requires MPS or CUDA; run cacox_cpu_validate.py for CPU validation")
    c.configure(1)
    rng = np.random.RandomState(561)
    n, p, M = 600, 12, 20
    x = rng.normal(size=(M, n, p)).astype(np.float32)
    times = np.arange(1, n+1)*2.
    events = np.arange(n) % 5 == 0
    metadata = dict(variables=[f"v{i}" for i in range(p)], columns=[f"v{i}" for i in range(p)])
    pair = c.source_module("pair").MeanRiskPairLoss(times, events, "cpu")
    records = {}
    for name in ("original_CACox", "P_C", "C_J_SP", "A16_C"):
        original = name == "original_CACox"
        shape = (n, p) if original else (M*n, p)
        raw_x = x[0] if original else x.reshape(shape)
        cpu = c.make_original(p, 457) if original else c.make_shared(p, metadata, name, 457)
        mps = c.make_original(p, 457, accelerator) if original else c.make_shared(p, metadata, name, 457, accelerator)
        mps.load_state_dict(cpu.state_dict())
        i20 = torch.as_tensor(rng.normal(size=(M, n)), dtype=torch.float64)
        stats = [dict(i20_mean=0., i20_sd=1., kernel_mean=0., kernel_sd=1.) for _ in range(M)]

        def objective(net, device):
            X = torch.as_tensor(raw_x, device=device)
            if original:
                eta = net(X).reshape(-1)
                return eta.to("cpu"), c.original_loss(times, events, device)(eta)
            eta = net(x_num=X, x_cat=None).squeeze(-1).reshape(M, n, 8).to("cpu")
            if name == "C_J_SP":
                value = c.combine_torch(eta.double().mean(2), i20, stats, 1).unsqueeze(2)
                return eta, pair(value)
            return eta, pair(eta)

        cpu.eval(); mps.eval()
        eta_c, loss_c = objective(cpu, "cpu")
        loss_c.backward()
        eta_m, loss_m = objective(mps, accelerator)
        loss_m.backward()
        sync(accelerator)
        output_error = float(np.max(np.abs(eta_c.detach().numpy()-eta_m.detach().numpy())))
        loss_error = abs(float(loss_c.detach().cpu())-float(loss_m.detach().cpu()))
        gradient_error = max(float((a.grad-b.grad.cpu()).abs().max())
                             for a, b in zip(cpu.parameters(), mps.parameters()) if a.grad is not None)
        if output_error > 2e-4 or loss_error > 2e-5 or gradient_error > 2e-3:
            raise AssertionError(f"CPU/{accelerator} numerical differences too large: {name}: "
                                 f"{output_error}, {loss_error}, {gradient_error}")
        elapsed = {}
        for device, net in (("cpu", cpu), (accelerator, mps)):
            net.train()
            optimizer = (torch.optim.Adam(net.parameters(), lr=7e-6, weight_decay=.01) if original
                         else torch.optim.AdamW(net.parameters(), lr=.0001 if name == "C_J_SP" else .001,
                                                weight_decay=.0003))
            values = []
            for step in range(3):
                sync(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                _, loss = objective(net, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 100. if original else 5., error_if_nonfinite=True)
                optimizer.step()
                sync(device)
                values.append(time.perf_counter()-start)
            elapsed[device] = dict(warm_step_seconds=values[0], mean_next_two_seconds=float(np.mean(values[1:])))
        records[name] = dict(output_max_abs_error=output_error, loss_abs_error=loss_error,
                             gradient_max_abs_error=gradient_error, timings=elapsed,
                             accelerator_speedup=elapsed["cpu"]["mean_next_two_seconds"]/elapsed[accelerator]["mean_next_two_seconds"])
        print(json.dumps({name: records[name]}), flush=True)
        del cpu, mps
        if accelerator == "mps":
            torch.mps.empty_cache()
        elif accelerator == "cuda":
            torch.cuda.empty_cache()
    result = dict(synthetic=True, accelerator=accelerator, patients=n, encoded_features=p, MI_views=M,
                  CPU_threads=1, shared_loss_device="cpu", shared_loss_dtype="float64",
                  tolerance=dict(output_atol=2e-4, loss_atol=2e-5, gradient_atol=2e-3),
                  results=records)
    t.write_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(compare(args.output, args.device), indent=2))
