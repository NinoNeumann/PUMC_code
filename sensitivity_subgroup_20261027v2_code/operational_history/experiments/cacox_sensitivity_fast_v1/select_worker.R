#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) stop("Usage: select_worker.R branch ordinal worker_id vote_ids...")
branch <- args[[1L]]; ordinal <- as.integer(args[[2L]])
worker_id <- as.integer(args[[3L]]); votes <- as.integer(args[-(1:3)])
root <- Sys.getenv("CACOX_PORTABLE_ROOT")
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", arg[[1L]]), winslash = "/", mustWork = TRUE))
adapter_dir <- file.path(dirname(here), "cuda_boruta_alignment_v1")
source(file.path(root, "code/common.R"), encoding = "UTF-8")
source(file.path(adapter_dir, "r_cuda_adapter.R"), encoding = "UTF-8")
if (!branch %in% branch_names || is.na(ordinal) || !ordinal %in% 1:50 ||
    is.na(worker_id) || anyNA(votes) || any(!votes %in% 1:20) || anyDuplicated(votes))
  stop("Invalid selection job")
verify_data_sources()
output <- fold_output_path(branch, ordinal)
sensitivity_check_mi(output)
install_cuda_boruta(root)
# Status telemetry is the only cross-vote write in the frozen stage.
sensitivity_status <- function(cfg, stage, message = "") {
  sensitivity_write_csv(data.frame(
    branch = cfg$branch, ordinal = cfg$ordinal,
    repeat_id = cfg$repeat_id, fold = cfg$fold_id, stage = stage,
    message = message,
    time = format(Sys.time(), tz = "Asia/Shanghai", usetz = TRUE)
  ), file.path(cfg$output_dir, sprintf("status_select_%02d.csv", current_vote)))
}
server_dir <- file.path(output, sprintf("cuda_server_worker_%02d_pid_%d", worker_id, Sys.getpid()))
start_cuda_boruta_server(adapter_dir, server_dir, Sys.getenv("CACOX_PYTHON"))
tryCatch({
  for (j in votes) {
    current_vote <- j
    start <- proc.time()[["elapsed"]]
    sensitivity_stage(branch, ordinal, "select", j)
    q <- file.path(output, sprintf("imputation_%02d", j))
    stopifnot(file.exists(file.path(q, "boruta.rds")),
              file.exists(file.path(q, "lasso.rds")),
              file.exists(file.path(q, "combined_selection.csv")))
    cat(sprintf("SELECTION_COMPLETE %s %d %d seconds=%.6f\n",
                branch, ordinal, j, proc.time()[["elapsed"]] - start))
    flush.console()
  }
}, finally = stop_cuda_boruta_server())
