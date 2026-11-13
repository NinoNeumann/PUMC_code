# Iteration21: heterogeneous fixed-model convex risk mixtures

Protocol: `results/cacox_autonomous_optimization/20261031_v1/ITERATION21_PLAN.md`.
Sources are fixed17 C_J_SP_I20 and all four fixed20 NAM-I20 procedures. Only
training-score mean/SD alignment and a finite weight are new. No source weights,
data, MICE arrays, Encoders, variable lists, splits, trees or kernels are changed.
This is inference and internal model selection, with zero new model fits.

17 fixed candidates and E_selected each produce100 related MI outputs. All five
outer-fold choices freeze after25 internal groups, before any new outer mixture.
w17=1 preserves source17 scores/probabilities exactly; w17=0 preserves NAM ranking,
while its recalibrated survival probabilities need not equal native NAM probabilities.

Commands: `check_math.py`, `preflight.py`, `benchmark.py run`, `runner.py prepare`,
`runner.py launch`. Never overwrite signed code, source artifacts or remove a lock.
Final saved-weight CUDA replay, independent full metrics review and visual QA are
required before reporting completion. The plots show internal weight responses,
not new neural training loss curves. GPU occupancy can be intermittent in this
cached-inference workflow. Physical RAM/available-memory/owned-RSS guards persist.

Fixed-list/inherited-I20 dependence, reuse of internal validation for early stopping
and mixture selection, repeated same-cohort outer development, and related MI remain
limitations. This is not independent unbiased confirmation or a guarantee of.76.
