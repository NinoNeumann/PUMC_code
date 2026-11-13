#!/usr/bin/env Rscript
# Exact data-stage comparison for one branch and outer fold.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: compare_fold.R reference_run branch ordinal")
reference_run <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
branch <- args[[2L]]; ordinal <- as.integer(args[[3L]])
root <- Sys.getenv("CACOX_PORTABLE_ROOT")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
if (!branch %in% branch_names || is.na(ordinal) || !ordinal %in% 1:50)
  stop("Invalid branch or ordinal")
fold <- fold_name(ordinal)
old <- file.path(reference_run, "branches", branch, fold)
new <- fold_output_path(branch, ordinal)
stopifnot(dir.exists(old), dir.exists(new))
verify_data_sources()
po <- readRDS(file.path(old, "prepared.rds"))
pn <- readRDS(file.path(new, "prepared.rds"))
co <- po$config; cn <- pn$config
co$output_dir <- NULL; cn$output_dir <- NULL
oco <- po$original_config; ocn <- pn$original_config
oco$output_dir <- NULL; ocn$output_dir <- NULL
prepared_equal <- identical(po$data, pn$data) && identical(co, cn) &&
  identical(oco, ocn) && identical(po$candidates, pn$candidates)
parts <- lapply(0:5, function(id) {
  name <- if (id == 0L) "outer" else sprintf("inner_%02d", id)
  x <- readRDS(file.path(old, name, "imputation.rds"))
  y <- readRDS(file.path(new, name, "imputation.rds"))
  mx <- x$mids; my <- y$mids
  fx <- mx$formulas; fy <- my$formulas
  mx$formulas <- NULL; my$formulas <- NULL
  diag <- if (is.null(x$mids)) TRUE else identical(
    read.csv(file.path(old, name, "mice_chain_diagnostic.csv"), check.names = FALSE),
    read.csv(file.path(new, name, "mice_chain_diagnostic.csv"), check.names = FALSE))
  rhat <- if (is.null(y$mids)) NA_real_ else suppressWarnings(max(
    read.csv(file.path(new, name, "mice_chain_diagnostic.csv"))$classical_Rhat, na.rm = TRUE))
  data.frame(partition = name, prepared_equal = prepared_equal,
    base_equal = identical(x$development, y$development) &&
      identical(x$assessment, y$assessment) && identical(x$model_vars, y$model_vars) &&
      identical(x$kind, y$kind),
    mids_fields_equal = identical(mx, my),
    formula_text_equal = identical(lapply(fx, deparse), lapply(fy, deparse)),
    completed_20_equal = all(vapply(1:20, function(j)
      identical(sensitivity_complete(x, j), sensitivity_complete(y, j)), logical(1))),
    diagnostic_equal = diag,
    methods_equal = identical(read.csv(file.path(old, name, "imputation_methods.csv"), check.names = FALSE),
                              read.csv(file.path(new, name, "imputation_methods.csv"), check.names = FALSE)),
    Rhat = rhat, Rhat_pass = is.na(rhat) || (is.finite(rhat) && rhat <= 1.1))
})
parts <- do.call(rbind, parts)
write.csv(parts, file.path(new, "NATIVE_MICE_COMPARE.csv"), row.names = FALSE)
votes <- do.call(rbind, lapply(1:20, function(j) {
  a <- file.path(old, sprintf("imputation_%02d", j))
  b <- file.path(new, sprintf("imputation_%02d", j))
  x <- readRDS(file.path(a, "boruta.rds")); y <- readRDS(file.path(b, "boruta.rds"))
  boruta <- identical(x$table, y$table) && identical(x$history, y$history) &&
    identical(x$thresholds, y$thresholds) &&
    identical(x$design_column_mapping, y$design_column_mapping)
  x <- readRDS(file.path(a, "lasso.rds")); y <- readRDS(file.path(b, "lasso.rds"))
  lasso <- identical(x$table, y$table) && identical(x$coefficients, y$coefficients) &&
    identical(x$cv, y$cv) && identical(x$lambda_1se, y$lambda_1se) &&
    identical(x$lambda_min, y$lambda_min)
  combined <- identical(read.csv(file.path(a, "combined_selection.csv"), check.names = FALSE),
                        read.csv(file.path(b, "combined_selection.csv"), check.names = FALSE))
  data.frame(imputation = j, boruta_equal = boruta, lasso_equal = lasso,
             combined_equal = combined)
}))
write.csv(votes, file.path(new, "NATIVE_SELECTION_COMPARE.csv"), row.names = FALSE)
tables <- c("all_selection_results.csv", "selection_frequency_29variables.csv",
            "candidate_estimability.csv", "cohort_counts.csv", "inner_assignments.csv")
tables_equal <- setNames(vapply(tables, function(name)
  identical(read.csv(file.path(old, name), check.names = FALSE),
            read.csv(file.path(new, name), check.names = FALSE)), logical(1)), tables)
pass <- all(parts$prepared_equal & parts$base_equal & parts$mids_fields_equal &
  parts$formula_text_equal & parts$completed_20_equal & parts$diagnostic_equal &
  parts$methods_equal & parts$Rhat_pass) &&
  all(votes$boruta_equal & votes$lasso_equal & votes$combined_equal) && all(tables_equal)
jsonlite::write_json(list(pass = pass, branch = branch, ordinal = ordinal,
  partitions = nrow(parts), votes = nrow(votes),
  all_completed_data_equal = all(parts$completed_20_equal),
  all_selection_equal = all(votes$boruta_equal & votes$lasso_equal & votes$combined_equal),
  fold_tables_equal = as.list(tables_equal)),
  file.path(new, "NATIVE_EXACT_AUDIT.json"), auto_unbox = TRUE, pretty = TRUE)
if (!pass) stop("Native R comparison failed: ", branch, " ", ordinal)
cat("NATIVE_EXACT_MATCH", branch, ordinal, "\n")
