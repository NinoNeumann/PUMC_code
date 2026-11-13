# Shared mean-risk ranking study — iteration15

Registered clinical protocol: results/cacox_autonomous_optimization/20261031_v1/ITERATION15_PLAN.md.
Fixed data, partitions,20 R MICE views, Boruta/LASSO lists and per-view arrays.
Compact shared-MI TabM8 x2x64,dropout.15,AdamW.001/WD.0003,clip5,max3000,
patience150. This is not the original Transformer nor discarded splinev2.

H is equal weighted member-Efron and mean-member-risk pair softplus;P is pure
pair softplus. Both NLL and C inner-stop monitors are fixed in advance,all same
seed0. Control NLL0 and C0 remain read-only. Mean8 and20 separate risk sets
are preserved;no averaging acrossMI or changing the metric pooling definition.
100 internal and20 outer newsharedNN weights,ten procedures each100 MI outputs.
All fivefold choices and epoch budgets freeze before new outer training.

Backend and preflight are now gated by the completed full20MI preflight:
14 real internal fits (4 control,8 candidate,2 repeat),CUDA mathematical
oracle,all saved-model/member-view replay,common monitor trace and controls
all passed. These are extra validation fits,not formal100 fits or outer tests.
Source files cannot change after this gate without a recorded fresh validation.

## Execution benchmark registered before measurement

Use24 full20MI tasks [A,B,C,D,D,C,B,A]x3 for each of1/2/3 workers:
A=f1inner1 H_NLL;B=f4inner1 H_C;C=f1inner1 P_NLL;D=f4inner1 P_C.
Each worker receives identical counts of all four types in each concurrency
case (6 each at1,3 each at2,2 each at3).Warmup is3 epochs perworker,not timed.
Check all losses,gradients,monitor counts,member risks andC exactly agree
with serial and repeated tasks,then choose fastest passing concurrency.
Planned total72 fullbenchmark fits+6warmups,reported separately from formal.
No outer performance or parameter selection occurs in the benchmark.
Physical27.772GiB/available4GiB/ownedRSS8GiB/max3 guards remain unchanged.

```powershell
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_shared_rank_objective/preflight.py
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_shared_rank_objective/benchmark.py run
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_shared_rank_objective/runner.py prepare
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_shared_rank_objective/runner.py launch
```

Preflight is already complete;do not repeat into its existing directory.
Formal runner/report is implemented and compiled;not launched/frozen at this
README revision. Completed benchmark:154.691/124.009/106.722seconds for1/2/3,
all72 fullfits passed exact trace/memberrisk/C equality;select3.6warmups separate.
Stored internal-state pipeline check independently recounted240 candidateC
rows and six normalized prediction frames,no new fits/outer predictions.
Perfit neural.pt/audit/trace/predictions are saved;perMI wrappers give the
actual selected/new/source weight path andSHA,training statistics,Encoder and
Breslow. All120new and60source states/memberrisks,600inputviews,6000innergrid
values and1000outer risks/probabilities are replayed before final completion.
No recover command:inspect actual PIDs/logs and immutable artifacts before any
authorized controlled continuation;never remove locks or bypass failed gates.
Pure-rank Breslow survival calibration is a postfit convention,not a guarantee
of PH likelihood correctness.Conditional IF intervals do not cover retraining,
selection/imputation or repeated outer-cohort development bias.Fixed lists and
inheritedI20 dependence remain;no independent fullynested validation claim.
