#!/usr/bin/env Rscript
# Read-only evaluation of MPA versus GPA discrimination using frozen OOF scores.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop('Usage: Rscript evaluate_known_subtype_mpa_gpa.R PACKAGE_ROOT OUTPUT_DIR')
root <- normalizePath(args[[1L]], mustWork = TRUE)
output <- args[[2L]]
source(file.path(root, 'code', 'evaluation.R'), encoding = 'UTF-8')
if (!requireNamespace('survival', quietly = TRUE)) stop('survival package required')
raw <- readRDS(file.path(root, 'input', 'raw_typed.rds'))
id_to_type <- setNames(as.character(raw$Type), as.character(raw$.patient_id))
fields <- c('repeat_id', 'fold', 'imputation', 'patient_id', 'Death', 'Survival_Days', 'score')

read_known <- function() {
  run <- file.path(root, 'run_fast_20261106_v1')
  fold_names <- sprintf('repeat%02d_fold%02d', rep(1:10, each = 5), rep(1:5, 10))
  pieces <- lapply(fold_names, function(fold) {
    model_dir <- file.path(run, 'branches', 'known_subtype', fold, 'cacox_es')
    if (!file.exists(file.path(model_dir, 'COMPLETE.json'))) stop('Missing model completion: ', fold)
    p <- file.path(model_dir, 'predictions_all20.csv')
    z <- read.csv(p, check.names = FALSE)
    if (!identical(unique(z$branch), 'known_subtype')) stop('Unexpected prediction branch: ', fold)
    z$score <- z$score_E_selected
    z[fields]
  })
  evaluation_validate(do.call(rbind, pieces), 1:50, 'known_subtype')
}

read_original <- function() {
  p <- file.path(root, 'input', 'main_oof_cacox.csv.gz')
  z <- read.csv(gzfile(p), check.names = FALSE)
  z$score <- z$score_E_selected
  evaluation_validate(z[fields], 1:50, 'original_main')
}

evaluate <- function(d, model_label) {
  patient_type <- id_to_type[as.character(d$patient_id)]
  if (any(!as.character(d$patient_id) %in% names(id_to_type)))
    stop('Patient without raw Type mapping')
  mpa <- evaluation_population(d[patient_type %in% 'MPA', , drop = FALSE], 'MPA')
  gpa <- evaluation_population(d[patient_type %in% 'GPA', , drop = FALSE], 'GPA')
  if (mpa$pooled$patients != 434L || mpa$pooled$events != 71L ||
      gpa$pooled$patients != 583L || gpa$pooled$events != 62L)
    stop('Unexpected MPA/GPA cohort counts')
  a <- mpa$each[order(mpa$each$imputation), ]
  b <- gpa$each[order(gpa$each$imputation), ]
  if (!identical(a$imputation, b$imputation)) stop('Imputation index mismatch')
  # Distinct original patients: conditional IJ covariance is zero.
  diff <- evaluation_pool(b$C - a$C, b$variance + a$variance, test = TRUE)
  data.frame(model = model_label, contrast = 'GPA_minus_MPA',
    MPA_patients = 434L, MPA_events = 71L, GPA_patients = 583L, GPA_events = 62L,
    C_MPA = mpa$pooled$estimate, C_GPA = gpa$pooled$estimate,
    C_MPA_CI95_lower = mpa$pooled$CI95_lower,
    C_MPA_CI95_upper = mpa$pooled$CI95_upper,
    C_GPA_CI95_lower = gpa$pooled$CI95_lower,
    C_GPA_CI95_upper = gpa$pooled$CI95_upper,
    diff, row.names = NULL, check.names = FALSE)
}

known <- evaluate(read_known(), 'known_subtype_refitted_CACox_ES')
original <- evaluate(read_original(), 'original_main_CACox_ES')
result <- rbind(known, original)
dir.create(output, recursive = TRUE, showWarnings = FALSE)
write.csv(result, file.path(output, 'MPA_vs_GPA_C_index.csv'), row.names = FALSE, na = '')
writeLines(c(
  'MPA versus GPA C-index contrast from existing 10x5x20 CACox ES out-of-fold predictions; no model retraining.',
  'Direction: GPA minus MPA. These are different, non-overlapping patients, so this is not a patient-paired contrast.',
  'Original raw Type defines subgroups. Missing Type and EGPA excluded. Within-fold comparable survival pairs only.',
  'Each imputation C-index uses patient-clustered influence across repeated folds. Conditional contrast variance is the sum of disjoint-group influence variances.',
  'Twenty imputation-specific contrasts use Rubin variance and t intervals. Inference conditions on fitted models and frozen splits, and is not a formal model-by-subtype interaction test.'
), file.path(output, 'METHODS.txt'))
print(result[, c('model', 'C_MPA', 'C_GPA', 'estimate', 'CI95_lower', 'CI95_upper', 'p_Rubin')])
