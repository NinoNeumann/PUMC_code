#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: mice_worker.R branch ordinal partition_id")
branch <- args[[1L]]; ordinal <- as.integer(args[[2L]]); id <- as.integer(args[[3L]])
root <- Sys.getenv("CACOX_PORTABLE_ROOT")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
if (!branch %in% branch_names || is.na(ordinal) || !ordinal %in% 1:50 ||
    is.na(id) || !id %in% 0:5) stop("Invalid fold or partition")
verify_data_sources()
# Status telemetry is isolated; the frozen scientific functions are unchanged.
sensitivity_status <- function(cfg, stage, message = "") {
  sensitivity_write_csv(data.frame(
    branch = cfg$branch, ordinal = cfg$ordinal,
    repeat_id = cfg$repeat_id, fold = cfg$fold_id, stage = stage,
    message = message,
    time = format(Sys.time(), tz = "Asia/Shanghai", usetz = TRUE)
  ), file.path(cfg$output_dir, sprintf("status_partition_%02d.csv", id)))
}
start <- proc.time()[["elapsed"]]
sensitivity_stage(branch, ordinal, "impute", id)
cat(sprintf("PARTITION_COMPLETE %s %d %d seconds=%.6f\n",
            branch, ordinal, id, proc.time()[["elapsed"]] - start))
