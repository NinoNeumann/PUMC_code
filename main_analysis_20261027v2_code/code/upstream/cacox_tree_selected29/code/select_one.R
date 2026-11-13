args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[[1]], winslash='/', mustWork=TRUE)
fold <- as.integer(args[[2]])
j <- as.integer(args[[3]])
stopifnot(fold %in% 1:5, j %in% 1:20)
project <- 'E:/CodexPorjects/PUMC-rebuttal'
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
stopifnot(requireNamespace('ranger',quietly=TRUE),requireNamespace('glmnet',quietly=TRUE),requireNamespace('survival',quietly=TRUE))
source(file.path(root,'code/leaf_sum_prediction.R'),encoding='UTF-8')
source(file.path(root,'code/selection29_fast.R'),encoding='UTF-8')
risk_audit_enabled <- FALSE
risk_audit_rows <- list()
cfg <- readRDS(file.path(root,'upstream_cache/cv10_20261027v2/data/config.rds'))
base <- file.path(root,sprintf('upstream_cache/cv10_20261027v2/repeat01_fold%02d',fold))
cfg$output_dir <- base
cfg$seed <- 20261027L + (fold-1L)*1000L
complete29 <- function(bundle,j) {
  mids <- bundle$mids
  stopifnot(mids$m == 20, mids$iteration == 60)
  z <- mids$data
  for(v in names(mids$imp)) {
    imp <- mids$imp[[v]]
    if(nrow(imp) == 0L) next
    ids <- as.integer(rownames(imp))
    stopifnot(all(mids$where[ids,v]))
    z[ids,v] <- imp[[j]]
  }
  stopifnot(!anyNA(z))
  n <- nrow(bundle$development)
  d <- bundle$development; a <- bundle$assessment
  for(v in names(z)) {
    d[[v]] <- z[seq_len(n),v]
    a[[v]] <- z[n+seq_len(nrow(a)),v]
  }
  list(dev=d,assessment=a)
}
ds <- complete29(readRDS(file.path(base,'outer/imputation.rds')),j)
work <- file.path(root,sprintf('selection/repeat01_fold%02d/imputation_%02d',fold,j))
dir.create(work,recursive=TRUE,showWarnings=FALSE)
started <- proc.time()[['elapsed']]
lasso <- lasso29(ds$dev,cfg,j)
lasso_seconds <- proc.time()[['elapsed']] - started
started <- proc.time()[['elapsed']]
boruta <- boruta29(ds$dev,cfg,j,work)
boruta_seconds <- proc.time()[['elapsed']] - started
stopifnot(identical(boruta$table$variable,lasso$table$variable))
z <- data.frame(imputation=j,boruta$table,
  lasso_lambda_min=lasso$table$lasso_min,lasso_lambda_1se=lasso$table$lasso_1se)
z$union_selected <- z$boruta_non_rejected | z$lasso_lambda_min
reference <- read.csv(file.path(base,sprintf('imputation_%02d/combined_selection.csv',j)))
fields <- c('variable','boruta_non_rejected',
  'lasso_lambda_min','lasso_lambda_1se','union_selected')
same_flags <- identical(z[fields],reference[fields])
raw_boruta_match <- identical(z[c('boruta_decision','hits','rounds')],
  reference[c('boruta_decision','hits','rounds')])
importance_diff <- max(abs(z$median_importance-reference$median_importance),na.rm=TRUE)
write.csv(z,file.path(work,'combined_selection.csv'),row.names=FALSE)
write.csv(data.frame(fold=fold,imputation=j,lasso_seconds=lasso_seconds,
  boruta_seconds=boruta_seconds,selection_flags_match_upstream=same_flags,
  raw_boruta_match_upstream=raw_boruta_match,
  max_importance_difference=importance_diff),
  file.path(work,'verification.csv'),row.names=FALSE)
cat(sprintf('SELECTION_COMPLETE fold=%d mi=%d lasso=%.3f boruta=%.3f selection_match=%s raw_match=%s diff=%.3g\n',
  fold,j,lasso_seconds,boruta_seconds,same_flags,raw_boruta_match,importance_diff))
