# Refit the upstream RCS + ridge-Cox definition on the exact local selected
# variables and the same archived R MICE datasets used by CACox.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
fold <- as.integer(args[[1L]])
mode <- args[[2L]]
stopifnot(fold %in% 1:5, mode %in% c('probe', 'full'))
project <- normalizePath('E:/CodexPorjects/PUMC-rebuttal', winslash='/', mustWork=TRUE)
root <- file.path(project, 'results/cacox_vs_rcs_selected29_optimization_20261028_v1')
selected_root <- file.path(project, 'results/cacox_tree_selected29_integrated_20261028_v1')
code <- file.path(root, 'windows_rcs_ridge_code')
.libPaths(c(file.path(project, '.r_library_selector_windows'), .libPaths()))
for (pkg in c('survival'))
  if (!requireNamespace(pkg, quietly=TRUE)) stop('Missing R package: ', pkg)
for (name in c('data_imputation_functions.R', 'selection29_functions.R',
               'cox29_reliable_fit.R', 'cox29_fitting_functions.R',
               'time_auc.R', 'ridge_functions.R'))
  source(file.path(code, name), encoding='UTF-8')
# The cached mids needs only completed values for fitting. Reconstruct those
# exactly as the verified CACox export does, without rerunning MICE.
complete29 <- function(bundle, j) {
  mids <- bundle$mids
  stopifnot(mids$m == 20L, mids$iteration == 60L, j %in% 1:20)
  z <- mids$data
  for (v in names(mids$imp)) {
    imp <- mids$imp[[v]]
    if (nrow(imp) == 0L) next
    ids <- as.integer(rownames(imp))
    stopifnot(all(mids$where[ids, v]))
    z[ids, v] <- imp[[j]]
  }
  n <- nrow(bundle$development)
  dev <- bundle$development; assessment <- bundle$assessment
  for (v in names(z)) {
    dev[[v]] <- z[seq_len(n), v]
    assessment[[v]] <- z[n + seq_len(nrow(assessment)), v]
    observed <- !is.na(mids$data[[v]])
    stopifnot(identical(as.character(z[[v]][observed]),
                        as.character(mids$data[[v]][observed])))
  }
  stopifnot(!anyNA(z))
  list(dev=dev,assessment=assessment)
}
ridge_grid <- 10^seq(-4, 4, by=1)
key <- sprintf('repeat01_fold%02d', fold)
source_fold <- file.path(selected_root, 'upstream_cache/cv10_20261027v2', key)
out <- file.path(root, 'matched_rcs_ridge', key)
dir.create(out, recursive=TRUE, showWarnings=FALSE)
cfg <- readRDS(file.path(selected_root, 'upstream_cache/cv10_20261027v2/data/config.rds'))
selected <- readLines(file.path(selected_root, 'selection', key, 'selected_variables.txt'), warn=FALSE)
stopifnot(length(cfg$model_vars)==29L, length(selected)>0L,
          identical(selected, cfg$model_vars[cfg$model_vars %in% selected]))
freq <- read.csv(file.path(selected_root, 'selection', key,
                           'selection_frequency_29variables.csv'))
stopifnot(identical(selected, as.character(freq$variable[freq$union_selected>=10])))
cfg$model_vars <- selected
cfg$continuous_vars <- intersect(cfg$continuous_vars, selected)
prepared <- readRDS(file.path(source_fold, 'prepared.rds'))
spec <- make_spec(prepared$data$dev, cfg)
outer <- readRDS(file.path(source_fold, 'outer/imputation.rds'))
stopifnot(outer$mids$m==20L, outer$mids$iteration==60L)
write_csv(data.frame(variable=selected),file.path(out,'selected_variables.csv'))
if (mode == 'probe') {
  ds <- complete29(outer, 1L)
  fit <- ridge_fit(ds$dev, spec, 1, list(fold=fold, imputation=1L, phase='probe'))
  pred <- cox29_predict(fit, ds$assessment)
  stopifnot(length(pred$lp)==nrow(ds$assessment), all(is.finite(pred$lp)))
  cat(sprintf('PROBE_OK fold=%d variables=%d assessment=%d\n',fold,length(selected),length(pred$lp)))
  quit(save='no')
}
tuning <- ridge_tune(source_fold, cfg, file.path(out,'tuning'))
write_csv(data.frame(lambda=tuning$lambda,inner_mean_C=tuning$mean_inner_C),
          file.path(out,'selected_lambda.csv'))
predictions <- list()
for (j in 1:20) {
  ds <- complete29(outer, j)
  fit <- ridge_fit(ds$dev, spec, tuning$lambda,
                   list(fold=fold, imputation=j, phase='outer'))
  pred <- cox29_predict(fit, ds$assessment)
  z <- data.frame(repeat_id=1L,fold=fold,imputation=j,
                  patient_id=ds$assessment$.patient_id,
                  Death=ds$assessment$Death,
                  Survival_Days=ds$assessment$Survival_Days,
                  score_rcs_ridge=pred$lp)
  for (i in seq_along(c(365,730,1095)))
    z[[paste0('survival_',c(365,730,1095)[i],'d')]] <- pred$survival[,i]
  predictions[[j]] <- z
  write_csv(z,file.path(out,sprintf('prediction_%02d.csv',j)))
  cat(sprintf('RCS_RIDGE fold=%d MI=%02d lambda=%.6g\n',fold,j,tuning$lambda))
  flush.console()
}
all <- do.call(rbind,predictions)
stopifnot(nrow(all)==228L*20L,all(table(all$imputation)==228L))
write_csv(all,file.path(out,'predictions_all20.csv'))
writeLines(format(Sys.time(),tz='Asia/Shanghai',usetz=TRUE),file.path(out,'COMPLETE.txt'))
