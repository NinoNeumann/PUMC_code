#!/usr/bin/env Rscript
args0<-commandArgs(FALSE);script<-sub('^--file=','',grep('^--file=',args0,value=TRUE)[1L])
script<-gsub('~+~',' ',script,fixed=TRUE) # Rscript encodes spaces in --file.
source(file.path(dirname(normalizePath(script,winslash='/',mustWork=TRUE)),'common.R'),encoding='UTF-8')
options(digits=17)

neural_json <- function(x) {
  if(is.null(x))return('null')
  string <- function(z)encodeString(as.character(z),quote='"',na.encode=FALSE)
  if(is.list(x)) {
    z<-vapply(x,neural_json,character(1))
    if(!is.null(names(x)))return(paste0('{',paste(paste0(vapply(names(x),string,character(1)),':',z),collapse=','),'}'))
    return(paste0('[',paste(z,collapse=','),']'))
  }
  if(length(x)!=1L)return(paste0('[',paste(vapply(as.list(x),neural_json,character(1)),collapse=','),']'))
  if(is.na(x))return('null')
  if(is.logical(x))return(if(x)'true'else'false')
  if(is.numeric(x)) {
    if(!is.finite(x))stop('Nonfinite value in export metadata')
    return(sprintf('%.17g',x))
  }
  string(x)
}
neural_write_text <- function(value,path) {
  tmp<-tempfile('.writing_',tmpdir=dirname(path));on.exit(unlink(tmp),add=TRUE)
  writeLines(value,tmp,useBytes=TRUE)
  portable_commit(tmp,path)
}
neural_csv_frame <- function(frame,variables) {
  keep<-c(variables,'Death','Survival_Days','.patient_id')
  stopifnot(all(keep%in%names(frame)),!anyNA(frame[,keep,drop=FALSE]))
  out<-lapply(frame[,keep,drop=FALSE],function(x) {
    if(is.numeric(x)) {
      stopifnot(all(is.finite(x)))
      sprintf('%.17g',x)
    } else as.character(x)
  })
  names(out)[names(out)=='.patient_id']<-'patient_id'
  as.data.frame(out,stringsAsFactors=FALSE,check.names=FALSE)
}
neural_write_csv <- function(frame,variables,path) {
  data<-neural_csv_frame(frame,variables)
  tmp<-tempfile('.writing_',tmpdir=dirname(path));on.exit(unlink(tmp),add=TRUE)
  utils::write.table(data,tmp,row.names=FALSE,col.names=TRUE,sep=',',quote=TRUE,
    qmethod='double',na='',fileEncoding='UTF-8')
  # Verify the exact 17-digit text. R's decimal reader can differ by one ULP
  # from the correctly rounded value, so its numeric check allows that bound.
  back<-utils::read.csv(tmp,colClasses='character',na.strings=character(),check.names=FALSE,
    stringsAsFactors=FALSE,fileEncoding='UTF-8')
  stopifnot(identical(names(back),names(data)),nrow(back)==nrow(frame))
  for(v in names(data)) {
    stopifnot(identical(back[[v]],data[[v]]))
    source_name<-if(v=='patient_id')'.patient_id'else v
    if(is.numeric(frame[[source_name]])) {
      original<-as.numeric(frame[[source_name]]);parsed<-as.numeric(back[[v]])
      stopifnot(all(abs(parsed-original)<=.Machine$double.eps*pmax(1,abs(original))))
    }
  }
  portable_commit(tmp,path)
}
neural_check_view <- function(actual,expected,variables) {
  cols<-c(variables,'Death','Survival_Days','.patient_id')
  stopifnot(nrow(actual)==nrow(expected),!anyDuplicated(actual$.patient_id),
    identical(actual$.patient_id,expected$.patient_id),!anyNA(actual[,cols,drop=FALSE]),
    identical(actual$Death,expected$Death),identical(actual$Survival_Days,expected$Survival_Days))
  for(v in variables) {
    observed<-!is.na(expected[[v]])
    stopifnot(identical(as.character(actual[[v]][observed]),as.character(expected[[v]][observed])))
  }
  invisible(TRUE)
}
export_neural_partition <- function(inp,partition) {
  stopifnot(partition%in%0:5)
  source_name<-if(partition==0L)'outer'else sprintf('inner_%02d',partition)
  target_name<-if(partition==0L)'outer'else sprintf('inner%02d',partition)
  out<-file.path(inp$output_path,'neural_inputs',target_name)
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  bundle_path<-file.path(inp$path,source_name,'imputation.rds')
  inputs<-c(bundle_path,inp$metadata_path,file.path(inp$path,'selection_frequency_29variables.csv'),
    file.path(analysis_code_dir,c('common.R','export_neural.R')))
  hashes<-sha256_files(inputs)
  n_views<-if(inp$branch=='complete_case')1L else 20L
  identity<-list(branch=inp$branch,ordinal=inp$ordinal,partition=partition,
    n_views=n_views,variables=inp$selected_vars,input_sha256=hashes,format='clinical_csv_17_significant_digits_v1')
  signature<-cox29_signature(identity)
  metadata_rds<-file.path(out,'metadata.rds')
  if(file.exists(metadata_rds)) {
    old<-readRDS(metadata_rds)
    stopifnot(identical(old$signature,signature),file.exists(file.path(out,'metadata.json')))
    verify_hash_table(old$output_hashes)
    cat('NEURAL_EXPORT_REUSED',inp$branch,fold_name(inp$ordinal),target_name,'views',n_views,'\n')
    return(invisible(old$metadata))
  }
  bundle<-readRDS(bundle_path)
  expected_train<-if(partition==0L)inp$prepared$data$dev else
    inp$prepared$data$dev[inp$prepared$data$inner_fold!=partition,,drop=FALSE]
  expected_val<-if(partition==0L)inp$prepared$data$assessment else
    inp$prepared$data$dev[inp$prepared$data$inner_fold==partition,,drop=FALSE]
  stopifnot(length(intersect(expected_train$.patient_id,expected_val$.patient_id))==0L,
    identical(bundle$development,expected_train),identical(bundle$assessment,expected_val))
  files<-character();views<-vector('list',n_views)
  for(j in seq_len(n_views)) {
    pair<-complete_bundle(bundle,j)
    neural_check_view(pair$dev,expected_train,inp$selected_vars)
    neural_check_view(pair$assessment,expected_val,inp$selected_vars)
    train_name<-sprintf('train_mi%02d.csv',j);val_name<-sprintf('val_mi%02d.csv',j)
    train_path<-file.path(out,train_name);val_path<-file.path(out,val_name)
    neural_write_csv(pair$dev,inp$selected_vars,train_path)
    neural_write_csv(pair$assessment,inp$selected_vars,val_path)
    files<-c(files,train_path,val_path)
    views[[j]]<-list(imputation=j,train_file=train_name,val_file=val_name)
  }
  selected_path<-file.path(out,'selected_variables.txt')
  neural_write_text(inp$selected_vars,selected_path);files<-c(files,selected_path)
  output_hashes<-data.frame(path=files,bytes=file.info(files)$size,sha256=unname(sha256_files(files)),stringsAsFactors=FALSE)
  metadata<-list(branch=inp$branch,ordinal=inp$ordinal,repeat_id=inp$repeat_id,fold=inp$fold,
    inner_fold=partition,partition=target_name,n_views=n_views,variables=inp$selected_vars,
    columns=c(inp$selected_vars,'Death','Survival_Days','patient_id'),
    train_count=nrow(expected_train),val_count=nrow(expected_val),n_train=nrow(expected_train),n_val=nrow(expected_val),
    train_deaths=sum(expected_train$Death),val_deaths=sum(expected_val$Death),
    imputation_kind=if(n_views==1L)'complete_case_no_imputation' else
      sprintf('MICE_20x%d', bundle$mids$iteration),
    numeric_format='%.17g',categorical_encoding='original_text_no_encoding',
    source_input_path=inp$path,source_bundle_path=bundle_path,source_metadata_path=inp$metadata_path,
    cache_reused=inp$cache_reused,input_hash=unname(hashes[[relative_path(bundle_path)]]),
    input_sha256=as.list(hashes),signature=signature,signature_algorithm='MD5_of_serialized_identity',path_base='package_root',
    views=views,output_sha256=as.list(setNames(output_hashes$sha256,output_hashes$path)))
  neural_write_text(neural_json(metadata),file.path(out,'metadata.json'))
  json_path<-file.path(out,'metadata.json')
  output_hashes<-rbind(output_hashes,data.frame(path=json_path,bytes=file.info(json_path)$size,
    sha256=unname(sha256_files(json_path)),stringsAsFactors=FALSE))
  cox29_atomic_rds(list(signature=signature,identity=identity,metadata=metadata,output_hashes=output_hashes),metadata_rds)
  cat('NEURAL_EXPORT_COMPLETE',inp$branch,fold_name(inp$ordinal),target_name,
    'views',n_views,'train',nrow(expected_train),'val',nrow(expected_val),'\n')
  invisible(metadata)
}
args<-commandArgs(trailingOnly=TRUE)
if(length(args)<2L)stop('Usage: export_neural.R branch ordinal [0|1|2|3|4|5|all]')
branch<-args[[1]];ordinal<-as.integer(args[[2]]);part<-if(length(args)>=3L)args[[3]]else'all'
partitions<-if(part=='all')0:5 else as.integer(part)
stopifnot(length(partitions)>0L,all(partitions%in%0:5))
inp<-get_fold_input(branch,ordinal)
for(partition in partitions)export_neural_partition(inp,partition)
