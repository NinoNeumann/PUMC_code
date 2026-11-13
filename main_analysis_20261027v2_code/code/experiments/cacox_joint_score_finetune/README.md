# Fixed-I20 final-combination neural fine-tuning:iteration17

Protocol:results/cacox_autonomous_optimization/20261031_v1/ITERATION17_PLAN.md.
No changes to1140patients/149events/29candidates/BorutaLASSO15,9,8,9,8,
fixed5folds/inner memberships/RMICE20/maxit60/eachMI originalEncoder/arrays.
No newtree/kernel,I20 updates,RCS riskbranch or abandonedCACoxv2.

Warmstarts both frozen15 P_NLL/P_C. Each has N_SP (continue pureNNsoftplus),
J_SP (final-combined softplus),J_S1/J_S01(final-combined sigmoid tau1/.1).
TabM8members/2x64/dropout.15 unchanged,AdamW reset lr.0001/WD.0003/clip5,
atmost500additionalsteps and150strict-stale. All monitor actual combinedC,
including epoch0 as legal fallback. NNalpha fixed.5/.75/.5/.5/.5.
Combine uses training-only sourceNN/I20 meanSD,frozen throughout finetune
AND inference. Never re-estimate finalNNscale or alter input preprocessing.
In-sample source tree may invite compensation overfit;this is a hypothesis test.

200inner/40outer newstates,zero-budget states optimizersteps0 explicitlycounted;
50/10readonly source states.10000innerC rows;18roles x100 correlatedMI outputs.
All40new/10sourcebudgets and10choices frozen before any newouter. All20MI
separate risksets/pairs;do not treat20n as independentpatients. OriginalC metric.

17full preflight calls plus4zero-step controls passed;independentCPUCUDA
value/gradient,sourceidentity,repeat,saved/reordered/singleMI prediction checks.
Balanced8types perworker:serial8fits44.245sec;2workers16fits55.768sec,
5.531/3.485sec perfit;all traces/gradients/memberrisks/combinedC identical.
Select2,not compare unequal work byrawwall.24benchmarkfits+3threeepochwarmups
separate fromformal200. Physical27.772GiB/available4GiB/ownedRSS8GiB guard,
conservative2GiB perworker reservation;max2. Keep user'sotherapps running.

Beforeformal freeze,report fixes12exploratory procedural Ccontrasts:
2warmstarts x NN/I20 x(J_SP-N_SP,J_S1-J_SP,J_S01-J_SP). ConditionalIF
linear differences,Holmwithin12;9I20roles vs2baselines=18primaryCtests
separateHolm. Doesnot correct past repeatedouterlook. No addedp goal.

Run preflight/benchmark only in fresh directories,do not overwrite gates.
Standalone runner/report explicitly adapted from signed16,source unchanged.
```powershell
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_joint_score_finetune/runner.py prepare
.venv-gpu/Scripts/python.exe -X utf8 experiments/cacox_joint_score_finetune/runner.py launch
```
Freeze only after read-only plumbingprobe. Do not edit signed files duringrun.
Complete replay of240new/60source states,600inputs,allC/choices/wrappers/probs;
independentreview and actual3figure inspection required before resultclaims.
ConditionalpatientIF excludes retraining/selection/imputation/optimism;fixedlist,
inheritedI20internal dependence,relatedMI and samecohortcontinueddevelopment
remain. Boundedloss does not guarantee discreteC monotonicity or PH calibration.
