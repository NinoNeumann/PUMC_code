# Reconstruct frozen RCS + ridge-Cox survival curves for 0–1095 d IBS.
# No tuning, splitting, screening, or imputation is performed here.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
project <- normalizePath(args[[1]],winslash='/',mustWork=TRUE)
input <- normalizePath(args[[2]],winslash='/',mustWork=TRUE)
output <- normalizePath(args[[3]],winslash='/',mustWork=FALSE)
key <- args[[4]]
stopifnot(grepl('^repeat(0[1-9]|10)_fold0[1-5]$',key))
repeat_id <- as.integer(substr(key,7,8)); fold <- as.integer(substr(key,14,15))
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
stopifnot(requireNamespace('survival',quietly=TRUE))
code <- file.path(project,'results/cacox_vs_rcs_selected29_optimization_20261028_v1/windows_rcs_ridge_code')
for(name in c('data_imputation_functions.R','selection29_functions.R',
              'cox29_reliable_fit.R','cox29_fitting_functions.R',
              'time_auc.R','ridge_functions.R'))
  source(file.path(code,name),encoding='UTF-8')

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
  }
  stopifnot(!anyNA(z))
  list(dev=dev,assessment=assessment)
}

km_censor <- function(time,event) {
  tt <- sort(unique(time))
  nr <- vapply(tt,function(t)sum(time>=t),numeric(1))
  dc <- vapply(tt,function(t)sum(time==t & event==0),numeric(1))
  list(times=tt,surv=c(1,cumprod(1-dc/nr)))
}

if(repeat_id==1L) {
  oldroot <- file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1')
  foldpath <- file.path(oldroot,'upstream_cache/cv10_20261027v2',key)
  selectionpath <- file.path(oldroot,'selection',key)
} else {
  foldpath <- file.path(input,key)
  selectionpath <- foldpath
}
original <- if(repeat_id==1L) {
  file.path(project,'results/cacox_vs_rcs_selected29_optimization_20261028_v1/matched_rcs_ridge',key)
} else {
  file.path(project,'results/rcs_ridge_10x5/20261102_v1/folds',key)
}
cfg <- readRDS(file.path(project,
  'results/cacox_tree_selected29_integrated_20261028_v1/upstream_cache/cv10_20261027v2/data/config.rds'))
selected <- readLines(file.path(selectionpath,'selected_variables.txt'),warn=FALSE)
stopifnot(length(cfg$model_vars)==29L,length(selected)>0L,
          identical(selected,cfg$model_vars[cfg$model_vars %in% selected]))
cfg$model_vars <- selected
cfg$continuous_vars <- intersect(cfg$continuous_vars,selected)
prepared <- readRDS(file.path(foldpath,'prepared.rds'))
stopifnot(nrow(prepared$data$dev)==912L,nrow(prepared$data$assessment)==228L)
outer <- readRDS(file.path(foldpath,'outer/imputation.rds'))
stopifnot(outer$mids$m==20L,outer$mids$iteration==60L)
spec <- make_spec(prepared$data$dev,cfg)
lambda <- read.csv(file.path(original,'selected_lambda.csv'))$lambda[[1]]
stopifnot(is.finite(lambda),lambda>0)
bundle_root <- if(repeat_id==1L) {
  file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/bundles')
} else {
  file.path(project,'results/cox_e_selected_10x5/20261101_v1/bundles')
}
rawdev <- read.csv(file.path(bundle_root,key,'main/raw_development.csv'))
stopifnot(nrow(rawdev)==912L)
G <- km_censor(rawdev$Survival_Days,rawdev$Death)
grid <- 0:1095
gg <- G$surv[findInterval(grid,G$times)+1L]
stopifnot(all(gg>0))
archive <- read.csv(file.path(original,'predictions_all20.csv'))
stopifnot(nrow(archive)==4560L)

rows <- vector('list',20L)
for(j in 1:20) {
  ds <- complete29(outer,j)
  fit <- ridge_fit(ds$dev,spec,lambda,
    list(repeat_id=repeat_id,fold=fold,imputation=j,phase='IBS_reconstruction'))
  pred <- cox29_predict(fit,ds$assessment,times=grid)
  S <- pred$survival
  stopifnot(nrow(S)==228L,ncol(S)==1096L,all(is.finite(S)),
            all(S>=0 & S<=1))
  old <- archive[archive$imputation==j,]
  matchidx <- match(ds$assessment$.patient_id,old$patient_id)
  stopifnot(length(matchidx)==228L,all(!is.na(matchidx)))
  old <- old[matchidx,]
  stopifnot(identical(as.integer(ds$assessment$Death),as.integer(old$Death)),
            max(abs(ds$assessment$Survival_Days-old$Survival_Days))==0)
  times <- c(365,730,1095)
  sdiff <- max(abs(S[,times+1L]-as.matrix(old[,paste0('survival_',times,'d')])))
  lpdiff <- max(abs(pred$lp-old$score_rcs_ridge))
  if(sdiff>1e-7 || lpdiff>1e-7)
    stop(sprintf('Three-point archived prediction mismatch %s MI%02d S=%.9g LP=%.9g',key,j,sdiff,lpdiff))
  t <- as.numeric(ds$assessment$Survival_Days)
  e <- as.integer(ds$assessment$Death)
  g_before <- G$surv[findInterval(t,G$times,left.open=TRUE)+1L]
  stopifnot(all(is.finite(g_before)),all(g_before[e==1]>0))
  dead <- outer(t,grid,'<=') & e==1L
  alive <- outer(t,grid,'>')
  terms <- (S*S/g_before)*dead + ((1-S)*(1-S)/rep(gg,each=length(t)))*alive
  brier <- colMeans(terms)
  ibs <- sum((brier[-length(brier)]+brier[-1L])/2)/1095
  rows[[j]] <- data.frame(repeat_id=repeat_id,fold=fold,imputation=j,
    model='rcs_ridge_cox',IBS_0_1095d_daily=ibs,
    Brier_365d=brier[366],Brier_730d=brier[731],Brier_1095d=brier[1096],
    max_S_3point_diff=sdiff,max_LP_diff=lpdiff,lambda=lambda)
  cat(sprintf('IBS_RCS %s MI=%02d %.9f archive_diff=%.3g\n',key,j,ibs,sdiff))
  flush.console()
}
result <- do.call(rbind,rows)
dir.create(output,recursive=TRUE,showWarnings=FALSE)
tmp <- file.path(output,paste0('rcs_ibs.csv.tmp.',Sys.getpid()))
write.csv(result,tmp,row.names=FALSE)
if(!file.rename(tmp,file.path(output,'rcs_ibs.csv')))stop('Atomic write failed')
