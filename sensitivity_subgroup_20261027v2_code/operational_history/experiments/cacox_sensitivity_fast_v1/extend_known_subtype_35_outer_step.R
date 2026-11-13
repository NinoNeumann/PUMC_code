#!/usr/bin/env Rscript
# Continue only known_subtype repeat07_fold05 outer from its saved mids.
# One invocation makes one immutable ten-iteration checkpoint.
root <- Sys.getenv("CACOX_PORTABLE_ROOT", "")
if (!nzchar(root)) stop("CACOX_PORTABLE_ROOT is required")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
verify_data_sources()

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript ... target_iteration")
target <- suppressWarnings(as.integer(args[[1L]]))
if (is.na(target) || target < 70L || target %% 10L != 0L)
  stop("Target must be a multiple of ten, starting at 70")

fold <- fold_output_path("known_subtype", 35L)
part <- file.path(fold, "outer")
source_file <- if (target == 70L) file.path(part, "imputation.rds") else
  file.path(part, "adaptive", sprintf("iter_%03d", target - 10L), "imputation.rds")
if (!file.exists(source_file)) stop("Missing previous mids checkpoint: ", source_file)
dest <- file.path(part, "adaptive", sprintf("iter_%03d", target))
if (file.exists(dest)) stop("Checkpoint already exists: ", dest)

bundle <- readRDS(source_file)
before <- bundle$mids
if (is.null(before) || before$m != 20L || before$iteration != target - 10L ||
    !is.null(before$loggedEvents)) stop("Previous mids failed state audit")
if (target == 70L) {
  original <- read.csv(file.path(part, "mice_chain_diagnostic.csv"))
  if (!isTRUE(all.equal(max(original$classical_Rhat, na.rm = TRUE),
                        1.10375207663824, tolerance = 1e-12)))
    stop("Original R-hat changed")
}

dir.create(dest, recursive = TRUE, showWarnings = FALSE)
started <- proc.time()[["elapsed"]]
after <- mice::mice.mids(before, maxit = 10L, printFlag = FALSE)
stopifnot(after$m == 20L, after$iteration == target,
          identical(after$ignore, before$ignore),
          identical(after$data, before$data),
          identical(after$method, before$method),
          identical(after$predictorMatrix, before$predictorMatrix))
diagnostic <- diagnose29_mice(after, dest)
extended <- bundle
extended$mids <- after
cox29_atomic_rds(extended, file.path(dest, "imputation.rds"))
rhat <- max(diagnostic$classical_Rhat, na.rm = TRUE)
events <- if (is.null(after$loggedEvents)) 0L else nrow(after$loggedEvents)
summary <- data.frame(branch = "known_subtype", ordinal = 35L,
                      partition = "outer", source_iteration = before$iteration,
                      target_iteration = target, max_Rhat = rhat,
                      logged_events = events,
                      elapsed_seconds = proc.time()[["elapsed"]] - started)
sensitivity_write_csv(summary, file.path(dest, "ADAPTIVE_AUDIT.csv"))
cat(sprintf("ADAPTIVE_MICE target=%d max_Rhat=%.12f logged_events=%d seconds=%.2f\n",
            target, rhat, events, summary$elapsed_seconds))
if (events != 0L) stop("Continuation produced logged events")
