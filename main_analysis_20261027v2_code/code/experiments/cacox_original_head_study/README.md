# Fixed original-CACox survival-head comparison

This study uses the exact original CrossAttentionCox architecture, without a spline branch or a v2 model. All training hyperparameters are fixed to the evaluated original configuration. The new head replaces only the last linear layer and uses its associated censored likelihood.

Protocol and commands: `docs/cacox_original_head_study.md` in the project root. Configuration: `configs/study_v1.json` here. The study runs input/checkpoint/math validation, a real-data smoke fit, 500 inner early-stopping fits, and 100 new outer models. Original Cox and matched RCS predictions remain read-only references.
