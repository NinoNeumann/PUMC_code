# Fit/evaluate one fold using immutable existing imputation/selection caches.
source('/Users/asklepios/Documents/ChatGPT/aav文章修改/cv10_20261027v2_rcs_ridge/code/common.R',encoding='UTF-8')
a<-commandArgs(trailingOnly=TRUE); i<-as.integer(a[1]); verify_ridge(i)
cfg<-cfg_ridge(i);out<-cfg$output_dir;src<-file.path(source_dir,fold_name(i))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
freq<-read.csv(file.path(src,'selection_frequency_29variables.csv'))
selected<-cfg$model_vars[cfg$model_vars %in% freq$variable[freq$union_selected>=10]]
if(!length(selected))stop('No selected variables')
fitcfg<-cfg;fitcfg$model_vars<-selected;fitcfg$continuous_vars<-intersect(cfg$continuous_vars,selected)
prepared<-readRDS(file.path(src,'prepared.rds'))
spec<-make_spec(prepared$data$dev,fitcfg)
q<-file.path(out,'models');dir.create(q,showWarnings=FALSE)
write_csv(freq,file.path(out,'selection_frequency_29variables.csv'))
write_csv(data.frame(variable=selected),file.path(q,'selected_variables.csv'))
cox29_atomic_rds(spec,file.path(q,'fixed_design.rds'))
status_ridge(i,'ridge_tuning')
tuning<-ridge_tune(src,fitcfg,file.path(out,'tuning'))
status_ridge(i,'fitting')
old<-readRDS(file.path(src,'models/models_all20.rds'))
stopifnot(identical(old$spec,spec),setequal(names(old$models),as.character(1:20)))
bundle<-readRDS(file.path(src,'outer/imputation.rds'))
pred<-mods_all<-list();status<-list();hrs<-phs<-list();diag_errors<-character()
for(j in 1:20) {
  ds<-complete29(bundle,j)
  ordinary<-old$models[[as.character(j)]]$ordinary_cox
  stopifnot(ordinary$model=='ordinary_cox',identical(ordinary$spec,spec))
  cache<-file.path(q,'checkpoints',sprintf('imputation_%02d.rds',j))
  if(file.exists(cache)) saved<-readRDS(cache) else {
    obj<-tryCatch(ridge_fit(ds$dev,spec,tuning$lambda,
      list(ordinal=i,repeat_id=cfg$repeat_id,fold=cfg$fold_id,imputation=j)),error=identity)
    saved<-list(success=!inherits(obj,'error'),result=if(!inherits(obj,'error'))obj else NULL,
      error=if(inherits(obj,'error'))conditionMessage(obj) else '')
    cox29_atomic_rds(saved,cache)
  }
  ridge_obj<-saved$result; ok<-saved$success; err<-saved$error
  if(ok) {
    mods<-list(ordinary_cox=ordinary,rcs_ridge_cox=ridge_obj)
    pp<-tryCatch(lapply(mods,cox29_predict,d=ds$assessment),error=identity)
    if(inherits(pp,'error')){ok<-FALSE;err<-paste('Prediction failed:',conditionMessage(pp))} else {
      z<-data.frame(imputation=j,patient_id=ds$assessment$.patient_id,repeat_id=cfg$repeat_id,
        fold=cfg$fold_id,Death=ds$assessment$Death,Survival_Days=ds$assessment$Survival_Days,
        score_cox=pp$ordinary_cox$lp,score_rcs_ridge=pp$rcs_ridge_cox$lp)
      for(n in names(pp))for(t in seq_len(ncol(pp[[n]]$survival)))
        z[[paste0(n,'_',colnames(pp[[n]]$survival)[t])]]<-pp[[n]]$survival[,t]
      pred[[as.character(j)]]<-z;mods_all[[as.character(j)]]<-mods
      for(n in names(mods)) {
        h<-tryCatch(cox29_HR(mods[[n]],j),error=identity)
        if(inherits(h,'error'))diag_errors<-c(diag_errors,paste(j,n,'HR:',conditionMessage(h))) else hrs[[length(hrs)+1L]]<-h
        phs[[length(phs)+1L]]<-cox29_PH(mods[[n]],j)
      }
    }
  }
  status[[length(status)+1L]]<-data.frame(imputation=j,model='ordinary_cox',success=TRUE,
    numerical_status='reused_original_valid_fit',error='')
  status[[length(status)+1L]]<-data.frame(imputation=j,model='rcs_ridge_cox',success=ok,
    numerical_status=if(ok)ridge_obj$numerical_status else 'unresolved',error=err)
  write_csv(do.call(rbind,status),file.path(q,'fit_status.csv'))
  cat('Completed ridge fit/prediction attempt',j,'of20\n');flush.console()
}
# Preserve the original zero-event category audit; no replacement imputations.
zero<-file.path(src,'models/zero_event_categories.csv')
if(file.exists(zero))file.copy(zero,file.path(q,'zero_event_categories.csv'),overwrite=TRUE)
if(length(pred))write_csv(do.call(rbind,pred),file.path(q,'paired_predictions_available_imputations_ONLY.csv'))
if(length(pred)!=20) {
  writeLines('Both model predictions must exist for all20 original imputations. No substitutions.',file.path(q,'INCOMPLETE.txt'))
  stop('Incomplete paired20 ridge fits; see fit_status.csv')
}
cox29_atomic_rds(list(models=mods_all,spec=spec,config=fitcfg,tuning=tuning),file.path(q,'models_all20.rds'))
write_csv(do.call(rbind,pred),file.path(q,'predictions_all20.csv'))
writeLines(format(Sys.time(),tz='Asia/Shanghai'),file.path(q,'FITS_COMPLETE.txt'))
if(length(phs))write_csv(do.call(rbind,phs),file.path(q,'PH_checks_each_imputation.csv'))
if(!length(diag_errors)) {
  hr_result<-tryCatch({
    hp<-cox29_pool_HR(do.call(rbind,hrs))
    hp$interval_scope<-'Approximate conditional intervals; ridge uses penalised covariance; excludes variable-selection and penalty-tuning uncertainty; not causal'
    write_csv(hp,file.path(q,'HR_contrasts_Rubin_pooled.csv'))
    cox29_forest(hp,q,paste('Ordinary Cox vs RCS-Ridge-Cox: fold',i))
    TRUE
  },error=identity)
  if(inherits(hr_result,'error'))diag_errors<-c(diag_errors,conditionMessage(hr_result))
}
if(length(diag_errors))writeLines(diag_errors,file.path(q,'DIAGNOSTIC_ERRORS.txt'))
writeLines('Ridge HR intervals condition on the tuned penalty and selected variables; ridge PH checks are exploratory joint-block and GLOBAL diagnostics, not individual-variable tests. Convergence does not establish proportional hazards.',file.path(q,'INFERENCE_NOTES.txt'))
status_ridge(i,'evaluation')
evaluate29_predictions(do.call(rbind,pred),file.path(out,'evaluation'),
  sprintf('Fold %d: ordinary Cox vs RCS-Ridge-Cox',i))
writeLines(format(Sys.time(),tz='Asia/Shanghai'),file.path(out,'EVALUATION_COMPLETE.txt'))
status_ridge(i,'complete')
