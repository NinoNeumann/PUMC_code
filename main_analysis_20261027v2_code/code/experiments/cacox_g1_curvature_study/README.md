# G1 curvature and internal-imputation-budget study

Three independent groups, not a combined change:

1. **G1_original**: read-only completed 5-fold x 20-MI evaluation.
2. **H1_curvature**: keep G1's original per-fold depth/leaf/round counts and
   original 3-MI internal budget. Change only the tree optimization to use a
   positive diagonal upper bound on the pairwise softplus Hessian. Fit 75
   internal diagnostic models and 100 outer models. No new selection.
3. **I20_selection**: unchanged original CUDA G1 learner and original grid;
   choose using all 5 inner folds x 20 MI per outer fold (500 cells). Reuse 75
   immutable original internal grids and fit the other 425 x 4 maximum-600-round
   trees. Freeze all choices before outer work. Evaluate 100 outer models.

The loss is L = n / P * sum softplus(F_j - F_i) over censor-compatible pairs.
Let w_ij = n/P * sigmoid(d_ij) * (1-sigmoid(d_ij)). Its full Hessian is a
weighted graph Laplacian H. The local bound B=2*diag(H)+1e-6*I satisfies
B-H positive semidefinite. It is a bound on the current Hessian, not a claim
of a global quadratic bound or guaranteed monotonic training/test improvement.

H1 supplies g/M and B/M to the booster, M=max(1,max(B)). Both derivatives have
the same scale, leaving the unregularized fixed-leaf Newton ratio unchanged.
Every working weight is positive and <=1. Consequently the unchanged minimum
child weight of 25 conservatively requires at least 25 sampled training rows.
This is a curvature-mass constraint, not exactly the old sample-count rule; it
can require more rows. Actual leaf populations are audited. No neural gradient,
learning rate, dropout, architecture, or weight is changed.

Original seed rule, offset .5, eta .03, subsample .8, hist32, L1/L2/gamma0,
all 1140 patients, clinical variables, Boruta+LASSO lists, R MICE and splits
stay fixed. No outer-dependent budget, tuning, fold-specific rescue or new
sensitivity analysis. H1 and I20 choices are never combined automatically.

Prechecks: independent gradient/full-Hessian/PSD and finite differences,
translation invariance, native one-tree leaf update, actual leaf count,
CUDA execution, repeated complete fits and exact saved replay. Final checks:
all inner/outer counts, selected internal saved predictions, all 200 new outer
model/combination replays, paired outcomes and immutable provenance hashes.
I20 may reuse an outer G1 only if its entire frozen configuration matches.

Result directory: results/cacox_g1_curvature_study/20261031_v1.
Entry: .venv-gpu/Scripts/python.exe experiments/cacox_g1_curvature_study/runner.py
prepare, then launch. Reporting dependency: tabulate==0.9.0.

Same-cohort development with repeatedly inspected outer outcomes. Original
fixed-list and early-stopping-cache internal dependencies remain. MI models
are correlated. Conditional intervals do not remove development-selection bias.
