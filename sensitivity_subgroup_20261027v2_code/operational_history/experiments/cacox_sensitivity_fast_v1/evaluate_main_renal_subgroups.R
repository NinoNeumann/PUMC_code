#!/usr/bin/env Rscript
# Read-only renal-involvement subgroup evaluation of frozen main CACox ES OOF scores.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop('Usage: Rscript evaluate_main_renal_subgroups.R PACKAGE_ROOT OUTPUT_DIR')
}
root <- normalizePath(args[[1L]], mustWork = TRUE)
output <- args[[2L]]
source(file.path(root, 'code', 'evaluation.R'), encoding = 'UTF-8')
if (!requireNamespace('survival', quietly = TRUE)) stop('survival package required')

# This helper checks the archived main-OOF SHA256, all 50 folds and 20 imputations,
# and patient/outcome agreement against the original typed cohort.
env <- list(root = root, python = Sys.getenv('CACOX_PYTHON', 'python'))
inputs <- portable_read_inputs(env)
raw <- inputs$raw
main <- inputs$main
renal <- as.character(raw$Renal_Involvement)
if (!identical(sort(unique(renal[!is.na(renal)])), c('no', 'yes'))) {
  stop('Unexpected raw Renal_Involvement codes')
}
group_by_id <- setNames(renal, as.character(raw$.patient_id))
group <- group_by_id[as.character(main$patient_id)]
if (sum(is.na(group)) != sum(is.na(raw$Renal_Involvement)) * 200L) {
  stop('Renal subgroup mapping or repeated row count mismatch')
}
yes <- main[group %in% 'yes', , drop = FALSE]
no <- main[group %in% 'no', , drop = FALSE]
yes_ids <- unique(yes$patient_id)
no_ids <- unique(no$patient_id)
if (length(intersect(yes_ids, no_ids)) || length(yes_ids) != 436L ||
    length(no_ids) != 423L || sum(raw$Death[renal == 'yes'], na.rm = TRUE) != 61L ||
    sum(raw$Death[renal == 'no'], na.rm = TRUE) != 37L) {
  stop('Unexpected or overlapping original renal subgroup membership')
}

yes_fit <- evaluation_population(yes, 'renal_involvement_yes')
no_fit <- evaluation_population(no, 'renal_involvement_no')
a <- yes_fit$each[order(yes_fit$each$imputation), , drop = FALSE]
b <- no_fit$each[order(no_fit$each$imputation), , drop = FALSE]
if (!identical(a$imputation, b$imputation) ||
    any(a$patients != 436L) || any(a$events != 61L) ||
    any(b$patients != 423L) || any(b$events != 37L)) {
  stop('Imputation indices or repeated subgroup counts mismatch')
}

# Groups have disjoint patient IDs: their conditional patient-influence
# covariance is zero. This is a between-subgroup contrast, not matched pairs.
contrast <- evaluation_pool(a$C - b$C, a$variance + b$variance, test = TRUE)
contrast$population <- 'renal_involvement_yes_minus_no'
contrast$model <- 'original_main_CACox_ES'
contrast$patients <- 859L
contrast$events <- 98L
contrast$repeats <- 10L
contrast$outer_folds <- 50L
contrast$imputations <- 20L
contrast <- contrast[, names(yes_fit$pooled)]
pooled <- rbind(no_fit$pooled, yes_fit$pooled, contrast)
pooled <- pooled[, c('population', 'model', 'patients', 'events', 'repeats',
  'outer_folds', 'imputations', 'estimate', 'SE', 'CI95_lower',
  'CI95_upper', 'p_Rubin', 'within_variance', 'between_variance',
  'total_variance', 'df')]

each <- data.frame(imputation = a$imputation, C_yes = a$C,
  variance_yes = a$variance, C_no = b$C, variance_no = b$variance,
  difference_yes_minus_no = a$C - b$C,
  variance_difference = a$variance + b$variance,
  comparable_pairs_yes = a$comparable_pairs,
  comparable_pairs_no = b$comparable_pairs)
folds <- rbind(no_fit$folds, yes_fit$folds)
dir.create(output, recursive = TRUE, showWarnings = FALSE)
write.csv(pooled, file.path(output, 'renal_subgroup_pooled.csv'), row.names = FALSE, na = '')
write.csv(each, file.path(output, 'renal_subgroup_each_imputation.csv'), row.names = FALSE)
write.csv(folds, file.path(output, 'renal_subgroup_each_fold.csv'), row.names = FALSE)
writeLines(c(
  'Input: 20261027v2 full-cohort original main CACox ES predictions, 10 repeats x 5 outer folds x 20 imputations.',
  paste0('Archived main OOF SHA256: ', inputs$main_sha256),
  'Renal groups come only from original raw_typed.rds Renal_Involvement: yes=436 patients/61 events; no=423/37; missing=281 excluded.',
  'The same original overall CACox ES model OOF scores are evaluated within each group. No subgroup selection, imputation or model refitting occurs.',
  'Harrell C-index uses comparable survival pairs only within each outer validation fold and weights by the number of comparable pairs.',
  'Patient influence is aggregated across all ten repeats before conditional variance is computed.',
  'The contrast is C(yes) minus C(no). Groups have disjoint patients; conditional contrast variance is the sum of their influence variances.',
  'Twenty imputation estimates and variances are combined by Rubin rules, with t-based 95% intervals and exploratory two-sided Wald p for the contrast.',
  'This is an unpaired between-subgroup discrimination comparison, not a patient-paired model comparison or a model-by-renal-involvement interaction test.',
  'Inference conditions on the frozen splits and fitted models; it omits full model-development and imputation-model-fitting uncertainty.'
), file.path(output, 'METHODS.txt'))
print(pooled[, c('population', 'patients', 'events', 'estimate',
  'CI95_lower', 'CI95_upper', 'p_Rubin')], row.names = FALSE)
