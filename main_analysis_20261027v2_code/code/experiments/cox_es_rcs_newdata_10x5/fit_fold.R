# Refit the archived RCS + ridge-Cox model on one frozen repeated outer fold.
# The old first repeat is audited and reused by run.py; this script handles 2:10.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
project <- normalizePath(args[[1]],winslash='/',mustWork=TRUE)
input <- normalizePath(args[[2]],winslash='/',mustWork=TRUE)
output <- normalizePath(args[[3]],winslash='/',mustWork=TRUE)
key <- args[[4]]
stopifnot(grepl('^repeat(0[1-9]|10)_fold0[1-5]$',key))
repeat_id <- as.integer(substr(key,7,8))
fold <- as.integer(substr(key,14,15))
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
stopifnot(requireNamespace('survival',quietly=TRUE))
code <- file.path(project,'results/cacox_vs_rcs_selected29_optimization_20261028_v1/windows_rcs_ridge_code')
for(name in c('data_imputation_functions.R','selection29_functions.R',
              'cox29_reliable_fit.R','cox29_fitting_functions.R',
              'time_auc.R','ridge_functions.R'))
  source(file.path(code,name),encoding='UTF-8')

# Reproduce the complete29 implementation used by run_matched_rcs_ridge.R.
complete29 <- function(bundle,j) {
  mids <- bundle$mids
  stopifnot(mids$m==20L,mids$iteration==60L,j %in% 1:20)
  z <- mids$data
  for(v in names(mids$imp)) {
    imp <- mids$imp[[v]]
    if(nrow(imp)==0L) next
    ids <- as.integer(rownames(imp))
    stopifnot(all(mids$where[ids,v]))
    z[ids,v] <- imp[[j]]
  }
  n <- nrow(bundle$development)
  dev <- bundle$development;assessment <- bundle$assessment
  for(v in names(z)) {
    dev[[v]] <- z[seq_len(n),v]
    assessment[[v]] <- z[n+seq_len(nrow(assessment)),v]
    observed <- !is.na(mids$data[[v]])
    stopifnot(identical(as.character(z[[v]][observed]),
                        as.character(mids$data[[v]][observed])))
  }
  stopifnot(!anyNA(z))
  list(dev=dev,assessment=assessment)
}

source_fold <- file.path(input,key)
target <- file.path(output,'folds',key)
dir.create(target,recursive=TRUE,showWarnings=FALSE)
stopifnot(file.exists(file.path(source_fold,'MICE_VALIDATION.json')))
cfg <- readRDS(file.path(input,'input/config.rds'))
selected <- readLines(file.path(source_fold,'selected_variables.txt'),warn=FALSE)
freq <- read.csv(file.path(source_fold,'selection_frequency_29variables.csv'))
stopifnot(length(cfg$model_vars)==29L,length(selected)>0L,
          identical(selected,cfg$model_vars[cfg$model_vars %in% selected]),
          identical(selected,as.character(freq$variable[freq$union_selected>=10])))
cfg$model_vars <- selected
cfg$continuous_vars <- intersect(cfg$continuous_vars,selected)
prepared <- readRDS(file.path(source_fold,'prepared.rds'))
stopifnot(prepared$repeat_id==repeat_id,prepared$fold==fold,
          nrow(prepared$data$dev)==912L,nrow(prepared$data$assessment)==228L)
outer <- readRDS(file.path(source_fold,'outer/imputation.rds'))
stopifnot(outer$mids$m==20L,outer$mids$iteration==60L,
          identical(outer$development$.patient_id,prepared$data$dev$.patient_id),
          identical(outer$assessment$.patient_id,prepared$data$assessment$.patient_id))
spec <- make_spec(prepared$data$dev,cfg)
ridge_grid <- 10^seq(-4,4,by=1)
tuning <- ridge_tune(source_fold,cfg,file.path(target,'tuning'))
write_csv(data.frame(lambda=tuning$lambda,inner_mean_C=tuning$mean_inner_C),
          file.path(target,'selected_lambda.csv'))
rows <- vector('list',20L)
for(j in 1:20) {
  prediction <- file.path(target,sprintf('prediction_%02d.csv',j))
  if(file.exists(prediction)) {
    rows[[j]] <- read.csv(prediction)
  } else {
    ds <- complete29(outer,j)
    fit <- ridge_fit(ds$dev,spec,tuning$lambda,
      list(repeat_id=repeat_id,fold=fold,imputation=j,phase='outer'))
    model_path <- file.path(target,sprintf('model_%02d.rds',j))
    model_tmp <- paste0(model_path,'.tmp.',Sys.getpid())
    saveRDS(fit,model_tmp,compress=TRUE)
    if(!file.rename(model_tmp,model_path))stop('Atomic fitted model rename failed')
    pred <- cox29_predict(fit,ds$assessment)
    z <- data.frame(repeat_id=repeat_id,fold=fold,imputation=j,
      patient_id=ds$assessment$.patient_id,Death=ds$assessment$Death,
      Survival_Days=ds$assessment$Survival_Days,score_rcs_ridge=pred$lp)
    for(i in seq_along(c(365,730,1095)))
      z[[paste0('survival_',c(365,730,1095)[i],'d')]] <- pred$survival[,i]
    stopifnot(nrow(z)==228L,!anyDuplicated(z$patient_id),
              all(is.finite(z$score_rcs_ridge)),
              all(as.matrix(z[,grep('^survival_',names(z)),drop=FALSE])>=0 &
                  as.matrix(z[,grep('^survival_',names(z)),drop=FALSE])<=1))
    tmp <- paste0(prediction,'.tmp.',Sys.getpid())
    write_csv(z,tmp)
    if(!file.rename(tmp,prediction))stop('Atomic prediction rename failed')
    rows[[j]] <- z
  }
  cat(sprintf('RCS_RIDGE %s MI=%02d lambda=%.8g\n',key,j,tuning$lambda))
  flush.console()
}
all <- do.call(rbind,rows)
stopifnot(nrow(all)==4560L,all(table(all$imputation)==228L),
          length(unique(all$patient_id))==228L)
tmp <- file.path(target,paste0('predictions_all20.csv.tmp.',Sys.getpid()))
write_csv(all,tmp)
if(!file.rename(tmp,file.path(target,'predictions_all20.csv')))
  stop('Atomic fold prediction rename failed')
writeLines(format(Sys.time(),tz='Asia/Shanghai',usetz=TRUE),file.path(target,'COMPLETE.txt'))
