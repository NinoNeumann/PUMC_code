#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: verify_data.R branch ordinal mice|final")
branch <- args[[1L]]; ordinal <- as.integer(args[[2L]]); phase <- args[[3L]]
root <- Sys.getenv("CACOX_PORTABLE_ROOT")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
if (!branch %in% branch_names || is.na(ordinal) || !ordinal %in% 1:50 ||
    !phase %in% c("mice", "final")) stop("Invalid verification request")
verify_data_sources()
out <- fold_output_path(branch, ordinal)
sensitivity_check_mi(out)
if (phase == "final") {
  input <- get_fold_input(branch, ordinal, validate = TRUE)
  stopifnot(file.exists(file.path(out, "selection_frequency_29variables.csv")),
            all(file.exists(file.path(out, sprintf("imputation_%02d/combined_selection.csv", 1:20)))))
}
cat("DATA_VERIFIED", branch, ordinal, phase, "\n")
