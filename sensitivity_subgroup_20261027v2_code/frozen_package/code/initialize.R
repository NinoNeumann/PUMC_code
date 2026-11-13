#!/usr/bin/env Rscript
args0<-commandArgs(FALSE);script<-sub('^--file=','',grep('^--file=',args0,value=TRUE)[1L])
script<-gsub('~+~',' ',script,fixed=TRUE) # Rscript encodes spaces in --file.
source(file.path(dirname(normalizePath(script,winslash='/',mustWork=TRUE)),'common.R'),encoding='UTF-8')
args<-commandArgs(trailingOnly=TRUE);mode<-if(length(args))args[[1]]else'prepare'
stopifnot(mode %in% c('prepare','verify'))
if(mode=='verify'||file.exists(file.path(state_data_dir,'INITIALIZED.txt'))){
  verify_data_sources();cat('DATA_SOURCES_VERIFIED\n');quit(save='no')
}
raw<-readRDS(file.path(input_dir,'raw_typed.rds'));cfg<-readRDS(file.path(input_dir,'config.rds'))
assignments<-read.csv(file.path(input_dir,'outer_assignments.csv'))
stopifnot(nrow(raw)==1140L,sum(raw$Death)==149L,length(cfg$model_vars)==29L,
  !anyDuplicated(raw$.patient_id),cfg$m==20L,cfg$maxit==60L,cfg$selection_threshold==10L,
  cfg$outer_folds==5L,cfg$outer_repeats==10L)
for(r in 1:10){
  a<-assignments[assignments$repeat_id==r,];a<-a[match(raw$.patient_id,a$patient_id),]
  stopifnot(nrow(a)==1140L,!anyNA(a$assessment_fold),!anyDuplicated(a$patient_id),
    identical(as.integer(a$Death),as.integer(raw$Death)),
    identical(a$assessment_fold,stratified_folds(raw$Death,5L,20261027L+r)))
}
dir.create(state_data_dir,recursive=TRUE,showWarnings=FALSE)
plan<-list();cohorts<-list()
for(branch in branch_names){
  d<-sensitivity_cohort(raw,branch,cfg)
  cohorts[[branch]]<-data.frame(branch=branch,N=nrow(d),deaths=sum(d$Death),excluded=nrow(raw)-nrow(d),
    candidate_count=length(cfg$model_vars),missing_candidate_cells=sum(is.na(d[,cfg$model_vars])))
  for(i in 1:50){
    c<-sensitivity_fold_cfg(branch,i);a<-assignments[assignments$repeat_id==c$repeat_id,]
    a<-a[match(d$.patient_id,a$patient_id),];test<-a$assessment_fold==c$fold_id
    stopifnot(!anyNA(test))
    plan[[length(plan)+1L]]<-data.frame(branch=branch,ordinal=i,repeat_id=c$repeat_id,fold=c$fold_id,
      seed=c$seed,development_N=sum(!test),development_deaths=sum(d$Death[!test]),assessment_N=sum(test),
      assessment_deaths=sum(d$Death[test]),input_mode='fresh_restricted_preprocessing',
      output_path=fold_output_path(branch,i),imputation_kind=if(branch=='complete_case')'none_20_selection_votes'else'MICE_20x60')
  }
}
plan<-do.call(rbind,plan);stopifnot(nrow(plan)==250L,all(plan$assessment_deaths>0L),all(plan$development_deaths>=5L))
write_csv(plan,file.path(state_data_dir,'FOLD_PLAN.csv'))
write_csv(do.call(rbind,cohorts),file.path(state_data_dir,'COHORT_COUNTS.csv'))
write_csv(hash_table(c(data_source_files(),file.path(state_data_dir,c('FOLD_PLAN.csv','COHORT_COUNTS.csv')))),
  file.path(state_data_dir,'SOURCE_HASHES.csv'))
cox29_atomic_rds(list(packages=data_packages(),R_version=as.character(getRversion()),folds=250L,
  reused_data_folds=0L,new_data_folds=250L,path_base='package_root',run_root=run_dir,
  all50_outer_assignments_exact=TRUE,created=format(Sys.time(),tz='UTC',usetz=TRUE)),
  file.path(state_data_dir,'preparation_manifest.rds'))
writeLines('Fresh 5-branch x 50-fold plan. No model training.',file.path(state_data_dir,'INITIALIZED.txt'))
cat('INITIALIZED folds=250 cached=0 model_training_started=FALSE\n')
