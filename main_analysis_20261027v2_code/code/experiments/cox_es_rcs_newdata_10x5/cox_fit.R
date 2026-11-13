# One ordinary Cox refit on one selected, imputed outer development set.
argv <- commandArgs(trailingOnly=TRUE)
stopifnot(length(argv)==5L)
stopifnot(!is.na(Sys.setlocale('LC_CTYPE','English_United States.utf8')))
project <- normalizePath(argv[[1]],winslash='/',mustWork=TRUE)
root <- normalizePath(argv[[2]],winslash='/',mustWork=TRUE)
r <- as.integer(argv[[3]]); k <- as.integer(argv[[4]]); j <- as.integer(argv[[5]])
stopifnot(r%in%2:10,k%in%1:5,j%in%1:20)
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
stopifnot(requireNamespace('survival',quietly=TRUE))
code_dir <- file.path(project,'experiments/ordinary_cox_matched_comparison/upstream_code')
source(file.path(code_dir,'cox29_reliable_fit.R'),encoding='UTF-8')
source(file.path(code_dir,'cox29_fitting_functions.R'),encoding='UTF-8')
fold_dir <- file.path(root,sprintf('repeat%02d_fold%02d',r,k))
selected <- readLines(file.path(fold_dir,'selected_variables.txt'),warn=FALSE)
cfg <- readRDS(file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/upstream_cache/cv10_20261027v2/data/config.rds'))
stopifnot(length(selected)>0L,all(selected%in%cfg$model_vars),
          identical(selected,cfg$model_vars[cfg$model_vars%in%selected]))
spec <- list(vars=selected,continuous=intersect(cfg$continuous_vars,selected),levels=cfg$factor_levels)
mi <- readRDS(file.path(fold_dir,'outer/imputation.rds'))
stopifnot(mi$mids$m==20L,mi$mids$iteration==60L)
z <- mice::complete(mi$mids,j)
nd <- nrow(mi$development)
dev <- mi$development; assessment <- mi$assessment
for(v in names(z)) {
  dev[[v]] <- z[seq_len(nd),v]
  assessment[[v]] <- z[nd+seq_len(nrow(assessment)),v]
}
stopifnot(nrow(dev)==912L,nrow(assessment)==228L,
          !anyNA(dev[,selected,drop=FALSE]),!anyNA(assessment[,selected,drop=FALSE]),
          !length(intersect(dev$.patient_id,assessment$.patient_id)))
dest <- file.path(root,'ordinary_cox',sprintf('repeat%02d_fold%02d',r,k),sprintf('mi%02d',j))
dir.create(dest,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(dest,'COMPLETE.txt')))quit(save='no')
t0 <- proc.time()[['elapsed']]
object <- cox29_fit_checkpoint(dev,spec,'ordinary_cox',
  list(repeat_id=r,fold=k,imputation=j,branch='repeated_plain_cox'),file.path(dest,'model'))
stopifnot(!inherits(object$fit,'coxph.penal'),is.null(object$fit$penalty),object$model=='ordinary_cox')
pred <- cox29_predict(object,assessment)
fd <- cox29_frame(assessment,spec,'ordinary_cox')
direct <- as.numeric(predict(object$fit,newdata=fd,type='lp',reference='zero'))
error <- max(abs(direct-pred$lp));stopifnot(error<1e-7)
q <- data.frame(repeat_id=r,fold=k,imputation=j,patient_id=assessment$.patient_id,
                Death=assessment$Death,Survival_Days=assessment$Survival_Days,score=pred$lp)
for(ix in seq_along(c(365,730,1095)))
  q[[paste0('survival_',c(365,730,1095)[ix],'d')]] <- pred$survival[,ix]
tmp <- file.path(dest,paste0('prediction.csv.tmp.',Sys.getpid()))
write.csv(q,tmp,row.names=FALSE)
if(!file.rename(tmp,file.path(dest,'prediction.csv')))stop('Atomic prediction rename failed')
write.csv(data.frame(repeat_id=r,fold=k,imputation=j,variables=length(selected),
          coefficients=length(coef(object$fit)),events=sum(dev$Death),
          numerical_status=object$numerical_status,no_penalty=TRUE,
          linear_continuous=TRUE,max_prediction_error=error,
          elapsed_seconds=proc.time()[['elapsed']]-t0),
          file.path(dest,'fit_diagnostics.csv'),row.names=FALSE)
writeLines('complete',file.path(dest,'COMPLETE.txt'))
cat(sprintf('COX_COMPLETE repeat=%d fold=%d mi=%d variables=%d seconds=%.3f\n',
            r,k,j,length(selected),proc.time()[['elapsed']]-t0))
