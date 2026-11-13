# Usage: Rscript code/stage.R prepare|impute|select|summary|fit|evaluate repeat fold [id]
source('/Users/asklepios/Documents/ChatGPT/aav文章修改/cv10_20261027v2/code/common.R',encoding='UTF-8')
verify29();a<-commandArgs(trailingOnly=TRUE);mode<-a[1];r<-as.integer(a[2]);k<-as.integer(a[3]);id<-if(length(a)>3)as.integer(a[4]) else 0L
cfg<-cfg29(r,k);out<-cfg$output_dir;dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(mode=='prepare'){
 if(file.exists(file.path(out,'prepared.rds')))quit(save='no')
 d<-readRDS(file.path(run_dir,'data/raw_typed.rds'));a<-read.csv(file.path(run_dir,'data/outer_assignments.csv'));a<-a[a$repeat_id==r,]
 stopifnot(identical(as.integer(a$patient_id),d$.patient_id),all(a$Death==d$Death))
 dev<-d[a$assessment_fold!=k,];assessment<-d[a$assessment_fold==k,]
 dat<-list(dev=dev,assessment=assessment,inner_fold=stratified_folds(dev$Death,5L,cfg$seed))
 cox29_atomic_rds(list(data=dat,config=cfg,input_md5=unname(tools::md5sum(cfg$input_file))),file.path(out,'prepared.rds'))
 write_csv(data.frame(patient_id=dev$.patient_id,Death=dev$Death,inner_fold=dat$inner_fold),file.path(out,'inner_assignments.csv'))
 write_status29(r,k,'prepared');quit(save='no')
}
p<-readRDS(file.path(out,'prepared.rds'));stopifnot(identical(p$config,cfg));dat<-p$data
if(mode=='impute'){
 stopifnot(id%in%0:5);q<-file.path(out,if(id==0)'outer' else sprintf('inner_%02d',id));dir.create(q,showWarnings=FALSE)
 if(file.exists(file.path(q,'imputation.rds')))quit(save='no')
 tr<-if(id==0)dat$dev else dat$dev[dat$inner_fold!=id,];va<-if(id==0)dat$assessment else dat$dev[dat$inner_fold==id,]
 cat('MICE partition',id,'development',nrow(tr),'assessment',nrow(va),'m20 maxit60\n');flush.console()
 mi<-impute_pair(tr,va,cfg,cfg$seed+id*100L)
 stopifnot(!is.null(mi$mids),mi$mids$m==20,mi$mids$iteration==60,
  identical(mi$mids$ignore,c(rep(FALSE,nrow(tr)),rep(TRUE,nrow(va)))),identical(names(mi$mids$data),cfg$model_vars),!any(c('Death','Survival_Days','.patient_id')%in%names(mi$mids$data)))
 bundle<-list(mids=mi$mids,development=tr,assessment=va,input_md5=p$input_md5)
 for(j in 1:20)invisible(complete29(bundle,j))
 write_csv(data.frame(variable=names(mi$method),method=mi$method),file.path(q,'imputation_methods.csv'))
 if(!is.null(mi$mids$loggedEvents))write_csv(mi$mids$loggedEvents,file.path(q,'mice_logged_events.csv'))
 if(length(mi$constant_notes))writeLines(mi$constant_notes,file.path(q,'constant_notes.txt'))
 diagnose29_mice(mi$mids,q)
 cox29_atomic_rds(bundle,file.path(q,'imputation.rds'))
 cat('Completed MICE partition',id,'; logged events:',if(is.null(mi$mids$loggedEvents))0 else nrow(mi$mids$loggedEvents),'\n');quit(save='no')
}
if(mode=='select'){
 stopifnot(id%in%1:20);check_mi29(out);q<-file.path(out,sprintf('imputation_%02d',id));dir.create(q,showWarnings=FALSE)
 if(file.exists(file.path(q,'combined_selection.csv')))quit(save='no')
 ds<-complete29(readRDS(file.path(out,'outer/imputation.rds')),id)
 if(file.exists(file.path(q,'lasso.rds')))lasso<-readRDS(file.path(q,'lasso.rds')) else {lasso<-lasso29(ds$dev,cfg,id);cox29_atomic_rds(lasso,file.path(q,'lasso.rds'))}
 if(file.exists(file.path(q,'boruta.rds')))boruta<-readRDS(file.path(q,'boruta.rds')) else {boruta<-boruta29(ds$dev,cfg,id,q);cox29_atomic_rds(boruta,file.path(q,'boruta.rds'))}
 stopifnot(identical(boruta$table$variable,lasso$table$variable))
 z<-data.frame(imputation=id,boruta$table,lasso_lambda_min=lasso$table$lasso_min,lasso_lambda_1se=lasso$table$lasso_1se)
 z$union_selected<-z$boruta_non_rejected|z$lasso_lambda_min
 write_csv(z,file.path(q,'combined_selection.csv'));write_csv(lasso$coefficients,file.path(q,'lasso_coefficients.csv'));write_csv(lasso$cv,file.path(q,'lasso_nested_cv.csv'))
 write_csv(boruta$design_column_mapping,file.path(q,'design_column_mapping.csv'));plot29_boruta(boruta,id,q)
 cat('Selected',id,'Boruta',sum(z$boruta_non_rejected),'LASSO',sum(z$lasso_lambda_min),'union',sum(z$union_selected),'\n');quit(save='no')
}
if(mode=='summary'){
 check_mi29(out);paths<-file.path(out,sprintf('imputation_%02d/combined_selection.csv',1:20));stopifnot(all(file.exists(paths)))
 all<-do.call(rbind,lapply(paths,read.csv));stopifnot(nrow(all)==29*20,all(table(all$variable)==20))
 freq<-do.call(rbind,lapply(cfg$model_vars,function(v){z<-all[all$variable==v,];data.frame(variable=v,boruta_non_rejected=sum(z$boruta_non_rejected),lasso_lambda_min=sum(z$lasso_lambda_min),union_selected=sum(z$union_selected),included=sum(z$union_selected)>=10)}))
 write_csv(all,file.path(out,'all_selection_results.csv'));write_csv(freq,file.path(out,'selection_frequency_29variables.csv'))
 png(file.path(out,'selection_frequency.png'),width=2100,height=1700,res=160);par(mar=c(5,17,3,1));o<-order(freq$union_selected);barplot(freq$union_selected[o],names.arg=freq$variable[o],horiz=TRUE,las=1,col=ifelse(freq$included[o],'#176A73','#b9c7ca'),xlim=c(0,20),main='Boruta non-rejected UNION LASSO lambda.min',xlab='Selected in 20 imputations');abline(v=10,lty=2);dev.off();quit(save='no')
}
if(mode=='fit'){
 check_mi29(out);q<-file.path(out,'models');dir.create(q,showWarnings=FALSE)
 freq<-read.csv(file.path(out,'selection_frequency_29variables.csv'));selected<-cfg$model_vars[cfg$model_vars%in%freq$variable[freq$union_selected>=10]]
 if(!length(selected))stop('No selected variables; no silent alternative model')
 fitcfg<-cfg;fitcfg$model_vars<-selected;fitcfg$continuous_vars<-intersect(cfg$continuous_vars,selected);spec<-make_spec(dat$dev,fitcfg)
 cox29_atomic_rds(spec,file.path(q,'fixed_design.rds'));write_csv(data.frame(variable=selected),file.path(q,'selected_variables.csv'))
 b<-readRDS(file.path(out,'outer/imputation.rds'));status<-pred<-hr<-ph<-zero<-list();allmods<-list()
 for(j in 1:20){
  ds<-complete29(b,j);mods<-list()
  for(v in setdiff(selected,cfg$continuous_vars))for(lev in cfg$factor_levels[[v]]){
   ix<-which(as.character(ds$dev[[v]])==lev);events<-sum(ds$dev$Death[ix]);if(events==0)zero[[length(zero)+1]]<-data.frame(imputation=j,variable=v,level=lev,N=length(ix),deaths=events)
  }
  for(n in c('ordinary_cox','rcs_cox')){
   target<-file.path(q,'checkpoints',sprintf('imputation_%02d',j),n)
   obj<-tryCatch(cox29_fit_checkpoint(ds$dev,spec,n,list(repeat_id=r,fold=k,imputation=j,branch='main29'),target),error=identity)
   ok<-!inherits(obj,'error');status[[length(status)+1]]<-data.frame(imputation=j,model=n,success=ok,numerical_status=if(ok)obj$numerical_status else 'unresolved',error=if(ok)'' else conditionMessage(obj));mods[[n]]<-obj
  }
  # Continue all20 original imputations even if an earlier model failed.
  if(!any(vapply(mods,inherits,logical(1),'error'))){
   pp<-lapply(mods,cox29_predict,d=ds$assessment);allmods[[as.character(j)]]<-mods
   z<-data.frame(imputation=j,patient_id=ds$assessment$.patient_id,repeat_id=r,fold=k,Death=ds$assessment$Death,Survival_Days=ds$assessment$Survival_Days,score_cox=pp$ordinary_cox$lp,score_rcs=pp$rcs_cox$lp)
   for(n in names(pp))for(t in seq_len(ncol(pp[[n]]$survival)))z[[paste0(n,'_',colnames(pp[[n]]$survival)[t])]]<-pp[[n]]$survival[,t]
   pred[[as.character(j)]]<-z
   for(n in names(mods)){hr[[length(hr)+1]]<-cox29_HR(mods[[n]],j);ph[[length(ph)+1]]<-cox29_PH(mods[[n]],j)}
  }
  write_csv(do.call(rbind,status),file.path(q,'fit_status.csv'));cat('Completed fit attempts for imputation',j,'\n');flush.console()
 }
 if(length(zero))write_csv(do.call(rbind,zero),file.path(q,'zero_event_categories.csv')) else write_csv(data.frame(imputation=integer(),variable=character(),level=character(),N=integer(),deaths=integer()),file.path(q,'zero_event_categories.csv'))
 if(length(pred))write_csv(do.call(rbind,pred),file.path(q,'paired_predictions_available_imputations_ONLY.csv'))
 if(length(pred)!=20){writeLines('Not all20 original imputations produced both valid models. No substitutions, no pooled20 result. See fit_status.csv and checkpoints.',file.path(q,'INCOMPLETE.txt'));quit(save='no',status=2L)}
 cox29_atomic_rds(list(models=allmods,spec=spec,config=fitcfg),file.path(q,'models_all20.rds'))
 write_csv(do.call(rbind,pred),file.path(q,'predictions_all20.csv'))
 hp<-cox29_pool_HR(do.call(rbind,hr));write_csv(hp,file.path(q,'HR_contrasts_Rubin_pooled.csv'));write_csv(do.call(rbind,ph),file.path(q,'PH_checks_each_imputation.csv'))
 cox29_forest(hp,q,sprintf('Development-fold models: repeat %d fold %d',r,k));writeLines(format(Sys.time(),tz='Asia/Shanghai'),file.path(q,'FITS_COMPLETE.txt'));quit(save='no')
}
if(mode=='evaluate'){
 path<-file.path(out,'models/predictions_all20.csv');if(!file.exists(path))stop('No complete paired20 prediction set')
 evaluate29_predictions(read.csv(path),file.path(out,'evaluation'),sprintf('Repeat %d fold %d: 29 candidates',r,k))
 writeLines(format(Sys.time(),tz='Asia/Shanghai'),file.path(out,'EVALUATION_COMPLETE.txt'));quit(save='no')
}
stop('Unknown stage')
