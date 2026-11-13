# Iteration20: original-variable grouped additive risk ensemble

Protocol: `results/cacox_autonomous_optimization/20261031_v1/ITERATION20_PLAN.md`.
The20 MI arrays, Encoders, original variable lists and inner/outer memberships
are immutable. Each original variable's complete dummy block enters one small
two-hidden-layer ReLU/dropout subnetwork. There are8 separately parameterized
additive members. Width8/16 and NLL/C internal stopping are the four conditions.
Padded dimensions are internal and masked; no clinical column is removed.

The objective is the same15 mean-MI mean-member-risk pair softplus, using the
original patients' comparable pairs separately in each MI. Original15 TabM
P_NLL/P_C controls are read-only, with their own explicitly distinct config.
No RCS risk branch, discarded CACoxv2, new trees/kernels or sensitivity analysis.

Commands: `preflight.py`, `benchmark.py run`, `runner.py prepare`, `runner.py launch`.
Run only after the saved synthetic proof,14 full20MI gates and normalized
1/2-process throughput comparison pass. All30 budgets and10 choices freeze
before any new outer fit. Saved-model CUDA replay, independent metric review,
source/input signatures and actual visual inspection precede reporting completion.
The physical RAM/available-memory/owned-RSS protections remain unchanged.

This is a local pure-ranking NAM-inspired adaptation, not a literal paper
reproduction or independent confirmation. Fixed-list/inherited-I20 dependence,
related MI and repeatedly inspected same-cohort outer outcomes remain limitations.
