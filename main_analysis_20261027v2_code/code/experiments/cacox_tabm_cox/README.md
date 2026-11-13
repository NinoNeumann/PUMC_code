# Compact TabM-Cox / frozen I20 fusion

This isolated experiment preserves the existing data, selection, encoder, R MICE,
and patient memberships. It uses installed official TabM 0.0.3 (pinned source hash),
a scalar exact-Efron Cox head and matched single/8 shared-parameter members.
It is **not** the old Weibull/WAS M4 and is not discarded CACox v2.

Run from project root, GPU environment:

```powershell
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_tabm_cox/check_loss_cpu.py
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_tabm_cox/preflight.py
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_tabm_cox/runner.py benchmark
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_tabm_cox/runner.py prepare
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_tabm_cox/runner.py launch
```

These operations refuse to overwrite completed gates. CPU mathematical check is
stored in the separately completed compatibility directory. Formal outputs are
`results/cacox_tabm_cox/20261031_v1`. Every experiment records timestamp, purpose,
configuration, artifact location, outcomes and limitations in EXPERIMENTS.md.
Read config.json / manifest.json before interpreting a model.

All 500 internal units fit all four configs (k1/k8 × lr .0003/.001). Train loss is
mean member Efron; stop on the mean-risk validation Efron (max3000/patience150).
Select LR/NN fusion weight by averaging 20 MI within inner fold then five folds.
Fixed outer epochs use ceil median of100 best internal epochs. All choices freeze
before any outer score. Four arms each have100 outer outputs; actual distinct
new outer NN fits depend on selected models/sharing/control fallback, at most400.
Original I20 NN/tree and kernel weights are read-only.

CUDA FP32 networks/FP64 stable exact Efron. No new preprocessing, embeddings,
bins, patient deletions, imputation or filtering. No RCS prediction branch.
Three workers maximum, choose fastest measured 1/2/3. Preserve physical guard
27.772GiB,4GiB reserve,8GiB study RSS. GPU sample peak is not average utilization.

Completion requires root COMPLETE/FINAL_INTEGRITY,2000 internal NN,100 outer
units ×4 predictions,5fold20MI0skip, all saved member risk and probability replay,
independent concordance and source/config/input signature checks. Root status
alone is insufficient. Do not change frozen code, knobs, membership, or inputs.

No `recover` command is implemented. If interrupted, inspect actual processes
and errors first; never launch duplicate runners, delete locks or blindly rerun
`run` after outer work starts. Preserve completed results; independently verify
source and every completed fit and archive stale lock/logs before a separately
reviewed recovery wrapper. Mathematical/hash/repeated errors require diagnosis.

This is continued development on the same cohort, with inherited internal
selection dependence, repeated outer viewing and correlated MI. Conditional CI
is not independent unbiased confirmation. Training loss/gap improvement alone
is not a C-index improvement. Finite grids/budgets are not global optima.
