"""CUDA tree kernels. Data splitting and feature selection are external and frozen."""
import os
import json
import time
from pathlib import Path
import numpy as np
import torch

# Make CUDA runtime libraries bundled with PyTorch visible to CuPy on Windows.
if os.name == 'nt':
    _dll_handle = os.add_dll_directory(str(Path(torch.__file__).parent / 'lib'))
import cupy as cp
import xgboost as xgb


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


class HistogramSurvivalForest:
    """Batched shallow log-rank survival trees, computed on selected torch device.

    Bootstrap multiplicities weight log-rank and KM; min leaf counts unique in-bag
    patients. Candidate features are sampled once per node. Histogram thresholds
    differ from scikit-survival's exhaustive split search. No missing-value logic
    is needed because the external frozen R MICE cache is complete.
    """
    def __init__(self, depth, leaf, seed, trees=300, bins=32, device='cuda'):
        self.depth, self.leaf, self.seed = int(depth), int(leaf), int(seed)
        self.trees, self.bins, self.device = int(trees), int(bins), str(device)

    def _tensor(self, x, dtype=torch.float64):
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _bin(self, x):
        x = self._tensor(x)
        return torch.stack([torch.bucketize(x[:, f].contiguous(), self.cuts[:, f].contiguous(), right=False)
                            for f in range(x.shape[1])], dim=1)

    @staticmethod
    def logrank(left_risk, left_events, risk, events):
        y = risk.clamp_min(1)
        u = (left_events - left_risk / y * events).sum(-1)
        v = (left_risk * (risk-left_risk).clamp_min(0) * events *
             (risk-events).clamp_min(0) / (y*y*(risk-1).clamp_min(1))).sum(-1)
        return torch.where(v > 1e-12, u*u/v.clamp_min(1e-12), torch.full_like(v, -torch.inf))

    def fit(self, x, time_values, event, horizons=(365, 730, 1095)):
        sync(self.device)
        start = time.perf_counter()
        xx = self._tensor(x)
        t = self._tensor(np.asarray(time_values))
        e = self._tensor(np.asarray(event), torch.bool)
        n, p = xx.shape
        q = torch.arange(1, self.bins, device=self.device, dtype=torch.float64)/self.bins
        self.cuts = torch.quantile(xx, q, dim=0)
        xb = self._bin(xx)
        self.event_times = torch.unique(t[e], sorted=True)
        E = len(self.event_times)
        if not E:
            raise ValueError('No training events')
        pos = torch.searchsorted(self.event_times, t, right=True)-1
        live_time = pos >= 0
        safe_pos = pos.clamp_min(0)
        hindex = torch.searchsorted(self.event_times, self._tensor(horizons), right=True)-1
        self.horizons = list(horizons)
        nodes = 2**(self.depth+1)-1
        self.feature = torch.full((self.trees,nodes), -1, device=self.device, dtype=torch.long)
        self.threshold = torch.zeros_like(self.feature)
        self.survival = torch.ones((self.trees,nodes,len(horizons)), device=self.device, dtype=torch.float64)
        # Same random plan on CPU and CUDA, independent of outcome values and device.
        rng = np.random.RandomState(self.seed)
        boot = rng.randint(0,n,size=(self.trees,n))
        weights = np.zeros((self.trees,n), dtype=np.float64)
        for tree in range(self.trees):
            weights[tree] = np.bincount(boot[tree], minlength=n)
        weights = self._tensor(weights)
        route = torch.zeros((self.trees,n), device=self.device, dtype=torch.long)
        active_t = torch.arange(self.trees,device=self.device)
        active_n = torch.zeros_like(active_t)
        F = max(1, int(np.sqrt(p)))
        tree_axis = torch.arange(self.trees,device=self.device)[:,None]
        for level in range(self.depth+1):
            A = len(active_t)
            if A == 0:
                break
            w = weights[active_t] * (route[active_t] == active_n[:,None])
            F = 1 if level == self.depth else max(1,int(np.sqrt(p)))
            features = self._tensor(np.stack([rng.choice(p,F,replace=False) for _ in range(A)]),torch.long)
            bx = xb[:,features].permute(1,0,2)
            # Prefix histograms: node x feature x value-bin x event-time interval.
            axis = torch.arange(A*F,device=self.device).reshape(A,1,F)
            indices = (((axis*self.bins+bx)*E)+safe_pos[None,:,None]).reshape(-1)
            ww = w[:,:,None].expand(-1,-1,F)
            H = torch.zeros(A*F*self.bins*E,device=self.device,dtype=torch.float64)
            H.scatter_add_(0,indices,(ww*live_time[None,:,None]).reshape(-1))
            H = H.reshape(A,F,self.bins,E)
            D = torch.zeros_like(H).reshape(-1)
            D.scatter_add_(0,indices,(ww*e[None,:,None]).reshape(-1))
            D = D.reshape(A,F,self.bins,E)
            total_intervals = H[:,0].sum(1)
            total_events = D[:,0].sum(1)
            total_risk = total_intervals.flip(-1).cumsum(-1).flip(-1)
            km = (1-total_events/total_risk.clamp_min(1)).clamp(0,1).cumprod(-1)
            hs = km[:,hindex.clamp_min(0)]
            hs[:,hindex<0] = 1.
            self.survival[active_t,active_n] = hs
            if level == self.depth:
                break
            left_risk = H.cumsum(2).flip(-1).cumsum(-1).flip(-1)
            left_events = D.cumsum(2)
            stat = self.logrank(left_risk,left_events,total_risk[:,None,None,:],total_events[:,None,None,:])
            unique_counts = torch.zeros(A*F*self.bins,device=self.device,dtype=torch.float64)
            unique_counts.scatter_add_(0,(axis*self.bins+bx).reshape(-1),((w>0)[:,:,None].expand(-1,-1,F)).reshape(-1).to(torch.float64))
            unique_counts = unique_counts.reshape(A,F,self.bins).cumsum(-1)
            valid = (unique_counts>=self.leaf)&((unique_counts[:,:,-1:]-unique_counts)>=self.leaf)
            stat = stat.masked_fill(~valid,-torch.inf)
            # Stable explicit tie policy across device reduction roundoff.
            stat = torch.round(stat*1e10)/1e10
            best, arg = stat.reshape(A,-1).max(1)
            split = torch.isfinite(best)
            sf = features.gather(1,(arg//self.bins)[:,None]).squeeze(1)
            sb = arg % self.bins
            self.feature[active_t[split],active_n[split]] = sf[split]
            self.threshold[active_t[split],active_n[split]] = sb[split]
            # Each tree/patient pair belongs to one active node: no duplicate writes.
            choose = split[:,None] & (route[active_t]==active_n[:,None])
            rr = xb[:,sf].T>sb[:,None]
            aa, pp = torch.where(choose)
            route[active_t[aa],pp] = 2*active_n[aa]+1+rr[aa,pp].to(torch.long)
            ts, ns = active_t[split],active_n[split]
            active_t = ts.repeat_interleave(2)
            active_n = (2*ns[:,None]+self._tensor([1,2],torch.long)[None,:]).reshape(-1)
            del H,D,left_risk,left_events,stat
        sync(self.device)
        self.fit_seconds = time.perf_counter()-start
        self.audit = dict(device=self.device,split='log-rank',leaf='weighted Kaplan-Meier',bins=self.bins,
                          unique_patient_min_leaf=True,bootstrap=True,train_only_quantiles=True,trees=self.trees)
        return self

    def predict_survival(self,x):
        xb = self._bin(x)
        n = len(xb)
        route = torch.zeros((self.trees,n),device=self.device,dtype=torch.long)
        axis = torch.arange(self.trees,device=self.device)[:,None]
        for _ in range(self.depth):
            f = self.feature[axis,route]
            b = self.threshold[axis,route]
            right = xb[torch.arange(n,device=self.device)[None,:],f.clamp_min(0)]>b
            route = torch.where(f>=0,2*route+1+right.to(torch.long),route)
        return self.survival[axis,route].mean(0)

    def state(self):
        return dict(settings=dict(depth=self.depth,leaf=self.leaf,seed=self.seed,trees=self.trees,bins=self.bins),
                    cuts=self.cuts.cpu(),feature=self.feature.cpu(),threshold=self.threshold.cpu(),
                    survival=self.survival.cpu(),event_times=self.event_times.cpu(),horizons=self.horizons)

    @classmethod
    def restore(cls,state,device='cuda'):
        obj = cls(**state['settings'],device=device)
        for key in ['cuts','feature','threshold','survival','event_times']:
            setattr(obj,key,state[key].to(device))
        obj.horizons=state['horizons']
        return obj


def cuda_breslow(t,e,score,horizons=(365,730,1095)):
    score = torch.as_tensor(score,device='cuda',dtype=torch.float64)
    t = torch.as_tensor(np.asarray(t),device='cuda',dtype=torch.float64)
    e = torch.as_tensor(np.asarray(e),device='cuda',dtype=torch.bool)
    times = torch.unique(t[e],sorted=True)
    shift = torch.quantile(score,.5)
    risk = torch.exp(score-shift)
    increments = ((t[None,:]==times[:,None])&e[None,:]).sum(1)/((t[None,:]>=times[:,None])*risk[None,:]).sum(1)
    return dict(event_times=times,cumulative_hazard=increments.cumsum(0),shift=shift)


def cuda_survival(score,baseline,horizons=(365,730,1095)):
    score = torch.as_tensor(score,device='cuda',dtype=torch.float64)
    idx = torch.searchsorted(baseline['event_times'],torch.tensor(horizons,device='cuda',dtype=torch.float64),right=True)
    ch = torch.cat([torch.zeros(1,device='cuda',dtype=torch.float64),baseline['cumulative_hazard']])[idx]
    return torch.exp(-torch.exp(score-baseline['shift'])[:,None]*ch[None,:])
