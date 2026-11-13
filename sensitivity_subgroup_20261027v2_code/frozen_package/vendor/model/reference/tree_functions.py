"""Verbatim selected functions for validation only; no historical imports.
Source: experiments/cacox_cuda_pilot/cuda_models.py
"""

def sync(device='cuda'):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
        cp.cuda.get_current_stream().synchronize()

def comparable_pairs(t, e, device):
    t = torch.as_tensor(np.asarray(t), device=device, dtype=torch.float64)
    e = torch.as_tensor(np.asarray(e), device=device, dtype=torch.bool)
    return torch.where(e[:, None] & ((t[:, None] < t[None, :]) |
                       ((t[:, None] == t[None, :]) & ~e[None, :])))

def rank_gradient(score, pair):
    i, j = pair
    w = torch.sigmoid(score[j] - score[i])
    # scatter_add preserves all censoring-compatible comparisons, not all row pairs.
    g = torch.zeros_like(score)
    g.scatter_add_(0, j, w)
    g.scatter_add_(0, i, -w)
    return g * (len(score) / len(i))

def fit_rank(x, t, e, offset, vx, voffset, depth, leaf, counts, seed, device='cuda'):
    """First-order boosting with unit working Hessian; GPU split/predict + GPU gradient."""
    gpu = str(device).startswith('cuda')
    xp = cp if gpu else np
    xx, vv = xp.asarray(x, dtype=xp.float32), xp.asarray(vx, dtype=xp.float32)
    oo, ov = xp.asarray(offset, dtype=xp.float32), xp.asarray(voffset, dtype=xp.float32)
    pairs = comparable_pairs(t, e, device)
    train = xgb.QuantileDMatrix(xx, label=xp.zeros(len(x), dtype=xp.float32), base_margin=oo, max_bin=32)
    found = {}
    if 0 in counts:
        found[0] = (np.asarray(offset).copy(), np.asarray(voffset).copy())
    telemetry = dict(gradient_calls=0, cuda_gradient_calls=0, fit_device=device,
                     prediction_device=device, working_hessian='ones; not exact pairwise Hessian')

    def objective(pred, _):
        telemetry['gradient_calls'] += 1
        if gpu:
            if not isinstance(pred, cp.ndarray):
                raise RuntimeError('XGBoost custom objective fell back to host predictions')
            score = torch.utils.dlpack.from_dlpack(pred)
            if not score.is_cuda:
                raise RuntimeError('Ranking gradient is not on CUDA')
            telemetry['cuda_gradient_calls'] += 1
        else:
            score = torch.from_numpy(pred)
        g = rank_gradient(score, pairs).contiguous()
        if gpu:
            return cp.from_dlpack(g), cp.from_dlpack(torch.ones_like(score))
        return g.numpy(), np.ones_like(pred)

    sync(device)
    start = time.perf_counter()
    model = xgb.Booster(dict(device=device, tree_method='hist', max_depth=int(depth),
                          min_child_weight=int(leaf), eta=.03, subsample=.8,
                          sampling_method='uniform', reg_lambda=0., reg_alpha=0.,
                          gamma=0., max_bin=32, seed=int(seed), nthread=1,
                          disable_default_eval_metric=1), [train])
    for iteration in range(max(counts)):
        # xgb.train's Python fobj wrapper currently materializes predictions on
        # the host. Explicit iterations retain scores/gradients on the device.
        pred = oo if iteration == 0 else model.inplace_predict(xx,predict_type='margin',base_margin=oo)
        grad,hess = objective(pred,train)
        model.boost(train,iteration,grad=grad,hess=hess)
        n = iteration+1
        if n in counts:
            a = model.inplace_predict(xx,predict_type='margin',base_margin=oo)
            b = model.inplace_predict(vv,predict_type='margin',base_margin=ov)
            if gpu and (not isinstance(a,cp.ndarray) or not isinstance(b,cp.ndarray)):
                raise RuntimeError('Ranking inference is not CUDA')
            found[n] = (cp.asnumpy(a).astype(float),cp.asnumpy(b).astype(float)) if gpu else (a.astype(float),b.astype(float))
    sync(device)
    telemetry['fit_seconds'] = time.perf_counter() - start
    telemetry['booster_config'] = json.loads(model.save_config())
    start = time.perf_counter()
    prediction = model.inplace_predict(vv, predict_type='margin', base_margin=ov)
    sync(device)
    telemetry['inference_seconds'] = time.perf_counter() - start
    if gpu:
        assert telemetry['cuda_gradient_calls'] == max(counts)
    return model, found, telemetry
