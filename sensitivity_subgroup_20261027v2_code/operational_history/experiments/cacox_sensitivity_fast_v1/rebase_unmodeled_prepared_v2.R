#!/usr/bin/env Rscript
# Rebind path-only metadata in the copied v2 package, preserving the v1 source.
root <- Sys.getenv("CACOX_PORTABLE_ROOT", "")
if (!nzchar(root) || !grepl("adaptive80_20261107_v2$", root))
  stop("Expected the new v2 package root")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
verify_data_sources()
archive <- file.path(run_root, "handoffs/adaptive80_egfr_observed_23_20261107/prepared_original")
dir.create(archive, recursive = TRUE, showWarnings = FALSE)
records <- list()
for (branch in branch_names) for (ordinal in 1:50) {
  fold <- fold_output_path(branch, ordinal)
  path <- file.path(fold, "prepared.rds")
  if (!file.exists(path) || file.exists(file.path(fold, "cacox_es/COMPLETE.json"))) next
  prior <- readRDS(path)
  cfg <- sensitivity_fold_cfg(branch, ordinal)
  old_config <- prior$config
  old_original <- prior$original_config
  prior$config$output_dir <- cfg$output_dir
  prior$original_config$output_dir <- cfg$output_dir
  stopifnot(identical(old_config[names(old_config) != "output_dir"],
                      prior$config[names(prior$config) != "output_dir"]),
            identical(old_original[names(old_original) != "output_dir"],
                      prior$original_config[names(prior$original_config) != "output_dir"]),
            identical(prior$original_config, cfg))
  prior$signature <- sensitivity_signature(prior$config)
  validate_prepared_membership(branch, ordinal, prior)
  rel <- sprintf("%s_repeat%02d_fold%02d", branch,
                 (ordinal-1L)%/%5L+1L, (ordinal-1L)%%5L+1L)
  backup <- file.path(archive, paste0(rel, ".rds"))
  if (file.exists(backup)) stop("Prepared backup exists: ", backup)
  stopifnot(file.copy(path, backup, overwrite = FALSE))
  before <- unname(sha256_files(path))
  cox29_atomic_rds(prior, path)
  after <- unname(sha256_files(path))
  check <- readRDS(path)
  stopifnot(identical(check$data, prior$data),
            identical(check$candidates, prior$candidates),
            identical(check$config, prior$config),
            identical(check$signature, sensitivity_signature(check$config)))
  records[[length(records)+1L]] <- data.frame(branch=branch, ordinal=ordinal,
    old_path=old_config$output_dir, new_path=cfg$output_dir,
    original_sha256=before, rebased_sha256=after)
  cat(sprintf("REBASED %s %02d\n", branch, ordinal)); flush.console()
}
if (!length(records)) stop("No unmodeled prepared folds were available to rebase")
sensitivity_write_csv(do.call(rbind, records),
  file.path(run_root, "handoffs/adaptive80_egfr_observed_23_20261107/PREPARED_REBASE_AUDIT.csv"))
