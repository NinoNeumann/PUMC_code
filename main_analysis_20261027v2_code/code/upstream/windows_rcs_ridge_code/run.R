# status | run [first_fold] [last_fold]; default range is 1..10.
source('/Users/asklepios/Documents/ChatGPT/aav文章修改/cv10_20261027v2_rcs_ridge/code/common.R',encoding='UTF-8')
a<-commandArgs(trailingOnly=TRUE);mode<-if(length(a))a[1] else 'status'
progress<-function()do.call(rbind,lapply(1:10,function(i){
  p<-file.path(run_dir,fold_name(i));f<-file.path(p,'status.csv')
  s<-if(file.exists(f))read.csv(f)$stage else 'not_started'
  st<-file.path(p,'models/fit_status.csv');z<-if(file.exists(st))read.csv(st) else NULL
  data.frame(ordinal=i,stage=s,ridge_fit_attempts=if(is.null(z))0 else sum(z$model=='rcs_ridge_cox'),
    ridge_fit_successes=if(is.null(z))0 else sum(z$model=='rcs_ridge_cox' & z$success),
    evaluation=file.exists(file.path(p,'EVALUATION_COMPLETE.txt')))
}))
if(mode=='status'){print(progress(),row.names=FALSE);quit(save='no')}
if(mode!='run')stop('Use status or run [first] [last]')
first<-if(length(a)>=2)as.integer(a[2])else 1L
last<-if(length(a)>=3)as.integer(a[3])else 10L
stopifnot(first%in%1:10,last%in%1:10,first<=last);verify_ridge()
lock<-file.path(run_dir,'RUNNING.lock')
if(!dir.create(lock,showWarnings=FALSE))stop('Existing lock: check original process; do not start duplicate run')
writeLines(as.character(Sys.getpid()),file.path(lock,'pid.txt'))
for(i in first:last) {
  p<-file.path(run_dir,fold_name(i));dir.create(p,recursive=TRUE,showWarnings=FALSE)
  if(file.exists(file.path(p,'EVALUATION_COMPLETE.txt')))next
  # Never automatically retry an already failed fold.
  if(file.exists(file.path(p,'status.csv')) && read.csv(file.path(p,'status.csv'))$stage=='failed')next
  cat('START',i,format(Sys.time(),tz='Asia/Shanghai'),'\n');flush.console()
  rc<-system2(file.path(R.home('bin'),'Rscript'),c(shQuote(file.path(code_dir,'stage.R')),i),
    stdout=file.path(p,'run.log'),stderr=file.path(p,'run.log'))
  if(rc!=0) {
    msg<-paste('Exit',rc,'see',file.path(p,'run.log'));status_ridge(i,'failed',msg)
    error_path<-file.path(run_dir,'run_errors.csv')
    rows<-data.frame(ordinal=i,error=msg,time=format(Sys.time(),tz='Asia/Shanghai'))
    if(file.exists(error_path))rows<-rbind(read.csv(error_path),rows)
    write_csv(rows,error_path)
  }
  write_csv(progress(),file.path(run_dir,'progress_10_folds.csv'))
  cat('END',i,format(Sys.time(),tz='Asia/Shanghai'),'\n');flush.console()
}
if(last==10L) {
  rc<-system2(file.path(R.home('bin'),'Rscript'),shQuote(file.path(code_dir,'aggregate.R')),
    stdout=file.path(run_dir,'aggregate.log'),stderr=file.path(run_dir,'aggregate.log'))
  if(rc!=0)writeLines(paste('Aggregation exit',rc),file.path(run_dir,'AGGREGATION_FAILED.txt'))
}
writeLines(paste('Requested fold range',first,last,'finished',format(Sys.time(),tz='Asia/Shanghai')),
  file.path(run_dir,sprintf('QUEUE_%02d_%02d_FINISHED.txt',first,last)))
unlink(lock,recursive=TRUE)
