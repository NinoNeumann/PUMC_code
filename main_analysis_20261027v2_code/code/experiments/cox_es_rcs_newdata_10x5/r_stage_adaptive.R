# Portable wrapper around the archived 29-variable upstream R functions.
# Arguments: project, output root, mode, repeat, fold, optional MI or partition.
argv <- commandArgs(trailingOnly=TRUE)
stopifnot(length(argv) >= 5L)
stopifnot(!is.na(Sys.setlocale('LC_CTYPE','English_United States.utf8')))
project <- normalizePath(argv[[1]], winslash='/', mustWork=TRUE)
root <- normalizePath(argv[[2]], winslash='/', mustWork=TRUE)
mode <- argv[[3]]
r <- as.integer(argv[[4]]); k <- as.integer(argv[[5]])
id <- if(length(argv)>=6L) as.integer(argv[[6]]) else NA_integer_
stopifnot(r %in% 1:10, k %in% 1:5)
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
original <- file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1')
cache <- file.path(root,'input')
source(file.path(original,'upstream_source/data_imputation_functions.R'),encoding='UTF-8')
source(file.path(original,'code/leaf_sum_prediction.R'),encoding='UTF-8')
source(file.path(original,'code/selection29_fast.R'),encoding='UTF-8')
risk_audit_enabled <- FALSE
risk_audit_rows <- list()
effective_mice <- function(part) {
  base <- file.path(out,part)
  for(iter in c(120L,100L,80L)) {
    candidate <- file.path(base,'adaptive',sprintf('iter_%03d',iter),'imputation.rds')
    if(file.exists(candidate)) return(candidate)
  }
  file.path(base,'imputation.rds')
}
atomic_rds <- function(value,dest) {
  tmp <- paste0(dest,'.tmp.',Sys.getpid())
  on.exit(if(file.exists(tmp)) unlink(tmp))
  saveRDS(value,tmp,compress=TRUE)
  if(!file.rename(tmp,dest)) stop('Atomic RDS rename failed: ',dest)
}
atomic_csv <- function(value,dest) {
  tmp <- paste0(dest,'.tmp.',Sys.getpid())
  on.exit(if(file.exists(tmp)) unlink(tmp))
  write_csv(value,tmp)
  if(!file.rename(tmp,dest)) stop('Atomic CSV rename failed: ',dest)
}
out <- file.path(root,sprintf('repeat%02d_fold%02d',r,k))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
cfg <- readRDS(file.path(cache,'config.rds'))
cfg$repeat_id <- r; cfg$fold_id <- k
cfg$seed <- 20261027L+(r-1L)*10000L+(k-1L)*1000L
cfg$output_dir <- out
cfg$outer_repeats <- 10L
data <- readRDS(file.path(cache,'raw_typed.rds'))
stopifnot(nrow(data)==1140L, sum(data$Death)==149L,
          length(cfg$model_vars)==29L, cfg$m==20L, cfg$maxit==60L,
          cfg$selection_threshold==10L)

# The original initialization uses this exact stratification seed for repeat r.
assignment <- stratified_folds(data$Death,5L,20261027L+r)
prior <- read.csv(file.path(cache,'outer_assignments.csv'))
if(r<=2L) {
  old <- prior[prior$repeat_id==r,]
  stopifnot(identical(as.integer(old$patient_id),data$.patient_id),
            identical(as.integer(old$assessment_fold),assignment))
}

if(mode=='prepare') {
  p <- file.path(out,'prepared.rds')
  if(file.exists(p)) quit(save='no')
  dev <- data[assignment!=k,]; assessment <- data[assignment==k,]
  inner <- stratified_folds(dev$Death,5L,cfg$seed)
  atomic_rds(list(data=list(dev=dev,assessment=assessment,inner_fold=inner),
                  repeat_id=r,fold=k,seed=cfg$seed),p)
  write_csv(data.frame(patient_id=dev$.patient_id,Death=dev$Death,
                       inner_fold=inner),file.path(out,'inner_assignments.csv'))
  cat(sprintf('PREPARED repeat=%d fold=%d dev=%d assessment=%d\n',r,k,nrow(dev),nrow(assessment)))
  quit(save='no')
}

p <- readRDS(file.path(out,'prepared.rds'))
stopifnot(p$repeat_id==r,p$fold==k,p$seed==cfg$seed)
dev <- p$data$dev; assessment <- p$data$assessment; inner <- p$data$inner_fold
if(mode=='impute') {
  stopifnot(id %in% 0:5)
  target <- file.path(out,if(id==0L)'outer' else sprintf('inner_%02d',id))
  dir.create(target,showWarnings=FALSE)
  dest <- file.path(target,'imputation.rds')
  if(file.exists(dest)) quit(save='no')
  tr <- if(id==0L)dev else dev[inner!=id,]
  va <- if(id==0L)assessment else dev[inner==id,]
  t0 <- proc.time()[['elapsed']]
  mi <- impute_pair(tr,va,cfg,cfg$seed+id*100L)
  stopifnot(!is.null(mi$mids),mi$mids$m==20L,mi$mids$iteration==60L,
            identical(mi$mids$ignore,c(rep(FALSE,nrow(tr)),rep(TRUE,nrow(va)))),
            identical(names(mi$mids$data),cfg$model_vars),
            !any(c('Death','Survival_Days','.patient_id')%in%names(mi$mids$data)))
  bundle <- list(mids=mi$mids,development=tr,assessment=va)
  for(j in 1:20) invisible(complete29(bundle,j))
  write_csv(data.frame(variable=names(mi$method),method=mi$method),
            file.path(target,'imputation_methods.csv'))
  if(!is.null(mi$mids$loggedEvents))
    write_csv(mi$mids$loggedEvents,file.path(target,'mice_logged_events.csv'))
  diagnose29_mice(mi$mids,target)
  atomic_rds(bundle,dest)
  cat(sprintf('IMPUTED repeat=%d fold=%d partition=%d seconds=%.3f events=%d\n',
              r,k,id,proc.time()[['elapsed']]-t0,
              if(is.null(mi$mids$loggedEvents))0 else nrow(mi$mids$loggedEvents)))
  quit(save='no')
}

if(mode=='select') {
  stopifnot(id %in% 1:20)
  for(part in c('outer',sprintf('inner_%02d',1:5))) {
    path <- effective_mice(part)
    b <- readRDS(path)
    stopifnot(b$mids$m==20L,b$mids$iteration %in% c(60L,80L,100L,120L))
    if(!is.null(b$mids$loggedEvents)) stop('MICE logged events: ',part)
    tab <- read.csv(file.path(dirname(path),'mice_chain_diagnostic.csv'))
    if(any(tab$classical_Rhat>1.1,na.rm=TRUE)) stop('MICE Rhat>1.1: ',part)
  }
  target <- file.path(out,sprintf('imputation_%02d',id))
  dir.create(target,showWarnings=FALSE)
  if(file.exists(file.path(target,'combined_selection.csv'))) quit(save='no')
  ds <- complete29(readRDS(effective_mice('outer')),id)
  t0 <- proc.time()[['elapsed']]
  lasso <- lasso29(ds$dev,cfg,id); lasso_seconds <- proc.time()[['elapsed']]-t0
  t0 <- proc.time()[['elapsed']]
  boruta <- boruta29(ds$dev,cfg,id,target); boruta_seconds <- proc.time()[['elapsed']]-t0
  stopifnot(identical(boruta$table$variable,lasso$table$variable))
  z <- data.frame(imputation=id,boruta$table,
       lasso_lambda_min=lasso$table$lasso_min,lasso_lambda_1se=lasso$table$lasso_1se)
  z$union_selected <- z$boruta_non_rejected | z$lasso_lambda_min
  atomic_csv(z,file.path(target,'combined_selection.csv'))
  write_csv(lasso$coefficients,file.path(target,'lasso_coefficients.csv'))
  write_csv(lasso$cv,file.path(target,'lasso_nested_cv.csv'))
  write_csv(data.frame(lasso_seconds=lasso_seconds,boruta_seconds=boruta_seconds),
            file.path(target,'timing.csv'))
  cat(sprintf('SELECTED repeat=%d fold=%d mi=%d lasso=%.3f boruta=%.3f\n',
              r,k,id,lasso_seconds,boruta_seconds))
  quit(save='no')
}

if(mode=='summary') {
  paths <- file.path(out,sprintf('imputation_%02d/combined_selection.csv',1:20))
  stopifnot(all(file.exists(paths)))
  z <- do.call(rbind,lapply(paths,read.csv))
  stopifnot(nrow(z)==29L*20L,all(table(z$variable)==20L))
  freq <- do.call(rbind,lapply(cfg$model_vars,function(v) {
    q <- z[z$variable==v,]
    data.frame(variable=v,boruta_non_rejected=sum(q$boruta_non_rejected),
               lasso_lambda_min=sum(q$lasso_lambda_min),
               union_selected=sum(q$union_selected),
               included=sum(q$union_selected)>=10L)
  }))
  selected <- freq$variable[freq$included]
  stopifnot(length(selected)>0L)
  write_csv(z,file.path(out,'all_selection_results.csv'))
  write_csv(freq,file.path(out,'selection_frequency_29variables.csv'))
  writeLines(selected,file.path(out,'selected_variables.txt'))
  cat(sprintf('SUMMARY repeat=%d fold=%d selected=%d\n',r,k,length(selected)))
  quit(save='no')
}
stop('Unknown mode: ',mode)
