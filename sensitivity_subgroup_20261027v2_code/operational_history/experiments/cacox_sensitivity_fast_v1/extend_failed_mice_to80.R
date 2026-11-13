#!/usr/bin/env Rscript
# Authorized continuation for no_egpa repeat02_fold01 inner_05 only.
# The original 60-iteration MICE object is never overwritten.
root <- Sys.getenv("CACOX_PORTABLE_ROOT", "")
if (!nzchar(root)) stop("CACOX_PORTABLE_ROOT is required")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
verify_data_sources()

fold <- fold_output_path("no_egpa", 6L)
part <- file.path(fold, "inner_05")
original <- file.path(part, "imputation.rds")
stopifnot(file.exists(original))
bundle <- readRDS(original)
prior <- bundle$mids
stopifnot(!is.null(prior), prior$m == 20L, prior$iteration == 60L,
          is.null(prior$loggedEvents))

for (target in c(70L, 80L)) {
  dest_dir <- file.path(part, "adaptive", sprintf("iter_%03d", target))
  if (file.exists(dest_dir)) stop("Adaptive output already exists: ", dest_dir)
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  before <- prior
  started <- proc.time()[["elapsed"]]
  after <- mice::mice.mids(before, maxit = 10L, printFlag = FALSE)
  stopifnot(after$m == 20L, after$iteration == target,
            identical(after$ignore, before$ignore),
            identical(after$data, before$data),
            identical(after$method, before$method),
            identical(after$predictorMatrix, before$predictorMatrix))
  diagnostic <- diagnose29_mice(after, dest_dir)
  extended <- bundle
  extended$mids <- after
  cox29_atomic_rds(extended, file.path(dest_dir, "imputation.rds"))
  summary <- data.frame(
    branch = "no_egpa", ordinal = 6L, partition = "inner_05",
    source_iteration = before$iteration, target_iteration = target,
    max_Rhat = max(diagnostic$classical_Rhat, na.rm = TRUE),
    logged_events = if (is.null(after$loggedEvents)) 0L else nrow(after$loggedEvents),
    elapsed_seconds = proc.time()[["elapsed"]] - started
  )
  sensitivity_write_csv(summary, file.path(dest_dir, "ADAPTIVE_AUDIT.csv"))
  cat(sprintf("ADAPTIVE_MICE target=%d max_Rhat=%.12f logged_events=%d seconds=%.2f\n",
              target, summary$max_Rhat, summary$logged_events, summary$elapsed_seconds))
  flush.console()
  if (summary$logged_events != 0L) stop("MICE continuation logged events")
  prior <- after
}
if (max(diagnostic$classical_Rhat, na.rm = TRUE) > 1.1)
  stop("80-iteration MICE R-hat remains above 1.1; preserve outputs and stop")
