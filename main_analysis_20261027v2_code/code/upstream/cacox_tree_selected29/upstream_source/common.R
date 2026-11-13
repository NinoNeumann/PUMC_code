project_dir<-'/Users/asklepios/Documents/ChatGPT/aav文章修改'
run_dir<-file.path(project_dir,'cv10_20261027v2')
code_dir<-file.path(run_dir,'code')
.libPaths(c(file.path(project_dir,'.Rlib'),.libPaths()))
for(p in c('readxl','mice','survival','ranger','glmnet'))if(!requireNamespace(p,quietly=TRUE))stop('Missing package: ',p)
source(file.path(code_dir,'data_imputation_functions.R'),encoding='UTF-8')
source(file.path(code_dir,'selection29_functions.R'),encoding='UTF-8')
source(file.path(code_dir,'cox29_reliable_fit.R'),encoding='UTF-8')
source(file.path(code_dir,'cox29_fitting_functions.R'),encoding='UTF-8')
source(file.path(code_dir,'time_auc.R'),encoding='UTF-8')
cfg29<-function(r,k){
 stopifnot(r%in%1:2,k%in%1:5)
 cfg<-readRDS(file.path(run_dir,'data/config.rds'))
 cfg$repeat_id<-as.integer(r);cfg$fold_id<-as.integer(k)
 cfg$seed<-20261027L+(r-1L)*10000L+(k-1L)*1000L
 cfg$output_dir<-file.path(run_dir,sprintf('repeat%02d_fold%02d',r,k));cfg
}
signature29<-function(){
 files<-c(sort(list.files(code_dir,pattern='\\.R$',full.names=TRUE)),file.path(run_dir,'data',c('input_snapshot.xlsx','raw_typed.rds','config.rds','outer_assignments.csv')))
 list(files=tools::md5sum(files),packages=vapply(c('readxl','mice','survival','ranger','glmnet'),function(p)as.character(utils::packageVersion(p)),character(1)))
}
verify29<-function(){path<-file.path(run_dir,'run_manifest.rds');if(!file.exists(path))stop('Initialize and freeze manifest before running');if(!identical(signature29(),readRDS(path)))stop('Input/code/package manifest mismatch; no stale cache reuse')}
write_status29<-function(r,k,stage,message=''){
 path<-file.path(run_dir,sprintf('repeat%02d_fold%02d',r,k));dir.create(path,recursive=TRUE,showWarnings=FALSE)
 write_csv(data.frame(repeat_id=r,fold=k,stage=stage,message=message,time=format(Sys.time(),tz='Asia/Shanghai',usetz=TRUE)),file.path(path,'status.csv'))
}
check_mi29<-function(out){
 for(id in 0:5){q<-file.path(out,if(id==0)'outer' else sprintf('inner_%02d',id));b<-readRDS(file.path(q,'imputation.rds'))
  stopifnot(b$mids$m==20,b$mids$iteration==60)
  if(!is.null(b$mids$loggedEvents))stop('MICE logged events require review: ',q)
  ds<-read.csv(file.path(q,'mice_chain_diagnostic.csv'))
  if(any(ds$classical_Rhat>1.1,na.rm=TRUE))stop('MICE chain diagnostic >1.1 requires review: ',q)
 }
}
