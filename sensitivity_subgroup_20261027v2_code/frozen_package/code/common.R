# Portable CACox sensitivity data preparation. All persisted paths are package-relative.
portable_root <- function() {
  root <- Sys.getenv('CACOX_PORTABLE_ROOT', '')
  if (!nzchar(root)) {
    src <- tryCatch(sys.frame(1L)$ofile, error=function(e) NULL)
    if (is.null(src)) {
      arg <- grep('^--file=', commandArgs(FALSE), value=TRUE)
      if (!length(arg)) stop('Set CACOX_PORTABLE_ROOT when sourcing common.R interactively')
      src <- sub('^--file=', '', arg[[1L]])
    }
    root <- dirname(dirname(normalizePath(src, winslash='/', mustWork=TRUE)))
  }
  normalizePath(root, winslash='/', mustWork=TRUE)
}
analysis_root <- portable_root()
setwd(analysis_root)
analysis_code_dir <- 'code'
code_dir <- 'vendor/r'
input_dir <- 'input'
run_root <- Sys.getenv('CACOX_RUN_ROOT', Sys.getenv('CACOX_OUTPUT_ROOT', file.path(analysis_root, 'run')))
if (!grepl('^(/|[A-Za-z]:[/\\\\])', run_root)) run_root <- file.path(analysis_root, run_root)
dir.create(run_root, recursive=TRUE, showWarnings=FALSE)
run_root <- normalizePath(run_root, winslash='/', mustWork=TRUE)
prefix <- paste0(analysis_root, '/')
if (!startsWith(run_root, prefix)) stop('CACOX_RUN_ROOT must be a directory inside the portable package')
run_dir <- substring(run_root, nchar(prefix)+1L)
state_data_dir <- file.path(run_dir, 'data')
extra_lib <- Sys.getenv('CACOX_R_LIBS', '')
if (nzchar(extra_lib)) .libPaths(c(extra_lib, .libPaths()))
for (p in c('mice','survival','ranger','glmnet'))
  if (!requireNamespace(p, quietly=TRUE)) stop('Missing R package: ', p)
for (f in c('data_imputation_functions.R','selection29_functions.R','checkpoint_utils.R'))
  source(file.path(code_dir,f), encoding='UTF-8')
branch_names <- c('complete_case','egfr_observed','renal_observed','known_subtype','no_egpa')
fold_name <- function(i) sprintf('repeat%02d_fold%02d',(i-1L)%/%5L+1L,(i-1L)%%5L+1L)
fold_output_path <- function(branch,ordinal) {
  stopifnot(branch %in% branch_names,length(ordinal)==1L,ordinal %in% 1:50)
  file.path(run_dir,'branches',branch,fold_name(ordinal))
}
fold_input_path <- fold_output_path
sensitivity_cfg <- function(branch,ordinal) {
  cfg <- readRDS(file.path(input_dir,'config.rds'))
  cfg$repeat_id <- as.integer((ordinal-1L)%/%5L+1L)
  cfg$fold_id <- as.integer((ordinal-1L)%%5L+1L)
  cfg$seed <- 20261027L+(cfg$repeat_id-1L)*10000L+(cfg$fold_id-1L)*1000L
  cfg$output_dir <- fold_output_path(branch,ordinal)
  cfg
}
relative_path <- function(path) {
  z <- normalizePath(path,winslash='/',mustWork=TRUE)
  if (any(!startsWith(z,prefix))) stop('Provenance file is outside package root')
  substring(z,nchar(prefix)+1L)
}
sha256_files <- function(paths) {
  stopifnot(length(paths)>0L,all(file.exists(paths)))
  paths <- relative_path(paths)
  py <- Sys.getenv('CACOX_PYTHON','')
  if (!nzchar(py)) py <- unname(Sys.which(c('python','python3'))[nzchar(Sys.which(c('python','python3')))][1L])
  if (is.na(py)||!nzchar(py)) stop('Set CACOX_PYTHON to the Python executable')
  a<-tempfile(fileext='.csv');b<-tempfile(fileext='.csv');on.exit(unlink(c(a,b)),add=TRUE)
  write.csv(data.frame(path=paths),a,row.names=FALSE,fileEncoding='UTF-8')
  status<-system2(py,c(shQuote(file.path(analysis_root,'code/portable_hash.py')),shQuote(a),shQuote(b),'--root',shQuote(analysis_root)))
  if(status!=0L||!file.exists(b))stop('portable_hash.py failed')
  z<-read.csv(b,stringsAsFactors=FALSE,check.names=FALSE,fileEncoding='UTF-8')
  stopifnot(identical(as.character(z$path),paths),all(grepl('^[0-9a-f]{64}$',z$sha256)))
  setNames(as.character(z$sha256),paths)
}
hash_table <- function(paths) {
  paths<-relative_path(paths)
  data.frame(path=paths,bytes=file.info(paths)$size,sha256=unname(sha256_files(paths)),stringsAsFactors=FALSE)
}
verify_hash_table <- function(tab) {
  stopifnot(nrow(tab)>0L,all(file.exists(tab$path)),all(file.info(tab$path)$size==tab$bytes))
  got<-unname(sha256_files(tab$path))
  if(!identical(got,as.character(tab$sha256)))stop('SHA-256 mismatch: ',paste(tab$path[got!=tab$sha256],collapse=', '))
  invisible(TRUE)
}
data_source_files <- function() c(file.path(input_dir,c('raw_typed.rds','config.rds','outer_assignments.csv')),
  file.path('code',c('common.R','data_stage.R','data_fold.R','initialize.R','export_neural.R','portable_hash.py')),
  list.files(code_dir,pattern='\\.R$',full.names=TRUE))
data_code_hashes <- function() {
  paths<-c(file.path('code','common.R'),list.files(code_dir,pattern='\\.R$',full.names=TRUE))
  setNames(unname(tools::md5sum(paths)),paths)
}
data_packages <- function() vapply(c('mice','survival','ranger','glmnet'),function(p)as.character(utils::packageVersion(p)),character(1))
verify_data_sources <- function() {
  path<-file.path(state_data_dir,'SOURCE_HASHES.csv')
  if(!file.exists(path))stop('Run Rscript code/initialize.R first')
  verify_hash_table(read.csv(path,stringsAsFactors=FALSE))
  saved<-readRDS(file.path(state_data_dir,'preparation_manifest.rds'))
  if(!identical(data_packages(),saved$packages))stop('R package versions changed; use a fresh run directory')
  invisible(TRUE)
}
source(file.path(code_dir,'pipeline.R'),encoding='UTF-8')
complete_bundle <- function(bundle,j) sensitivity_complete(bundle,as.integer(j))
validate_prepared_membership <- function(branch,ordinal,prepared) {
  cfg<-sensitivity_fold_cfg(branch,ordinal)
  raw<-readRDS(file.path(input_dir,'raw_typed.rds'))
  assignments<-read.csv(file.path(input_dir,'outer_assignments.csv'))
  d<-sensitivity_cohort(raw,branch,cfg)
  a<-assignments[assignments$repeat_id==cfg$repeat_id,]
  a<-a[match(d$.patient_id,a$patient_id),]
  stopifnot(!anyNA(a$assessment_fold),identical(as.integer(a$Death),as.integer(d$Death)))
  dev<-d[a$assessment_fold!=cfg$fold_id,,drop=FALSE];val<-d[a$assessment_fold==cfg$fold_id,,drop=FALSE]
  active<-sensitivity_active_config(dev,val,cfg)
  stopifnot(identical(active$dev,prepared$data$dev),identical(active$assessment,prepared$data$assessment),
    identical(stratified_folds(dev$Death,5L,cfg$seed),prepared$data$inner_fold),
    identical(active$candidates,prepared$candidates),identical(active$config,prepared$config))
  invisible(TRUE)
}
read_selected_vars <- function(path,prepared) {
  f<-file.path(path,'selection_frequency_29variables.csv')
  if(!file.exists(f))return(character())
  z<-read.csv(f,stringsAsFactors=FALSE)
  stopifnot(nrow(z)==29L,!anyDuplicated(z$variable),all(z$included==(z$union_selected>=10L)))
  out<-prepared$config$model_vars[prepared$config$model_vars %in% z$variable[z$included]]
  if(!length(out))stop('No variables selected: ',path)
  out
}
get_fold_input <- function(branch,ordinal,validate=TRUE,require_ready=TRUE) {
  path<-fold_input_path(branch,ordinal);metadata_path<-file.path(path,'prepared.rds')
  p<-readRDS(metadata_path)
  if(validate){verify_data_sources();validate_prepared_membership(branch,ordinal,p)}
  selected<-read_selected_vars(path,p)
  if(require_ready){
    stopifnot(length(selected)>0L,all(file.exists(file.path(path,c('outer',sprintf('inner_%02d',1:5)),'imputation.rds'))),
      all(file.exists(file.path(path,sprintf('imputation_%02d/combined_selection.csv',1:20)))))
    if(validate)sensitivity_check_mi(path)
  }
  list(path=path,prepared=p,cfg=p$config,selected_vars=selected,cache_reused=FALSE,
    branch=branch,ordinal=as.integer(ordinal),repeat_id=as.integer((ordinal-1L)%/%5L+1L),
    fold=as.integer((ordinal-1L)%%5L+1L),output_path=path,metadata_path=metadata_path)
}
data_status <- function() do.call(rbind,lapply(branch_names,function(b)do.call(rbind,lapply(1:50,function(i){
  p<-fold_input_path(b,i)
  data.frame(branch=b,ordinal=i,prepared=file.exists(file.path(p,'prepared.rds')),
    imputation_partitions=sum(file.exists(file.path(p,c('outer',sprintf('inner_%02d',1:5)),'imputation.rds'))),
    selection_replicates=sum(file.exists(file.path(p,sprintf('imputation_%02d/combined_selection.csv',1:20)))),
    summary=file.exists(file.path(p,'selection_frequency_29variables.csv')))
}))))
run_data_stage <- function(branch,ordinal,stage,id=0L,chunk_rounds=Inf) {
  stopifnot(stage %in% c('prepare','impute','select','summary'))
  verify_data_sources()
  sensitivity_stage(branch,as.integer(ordinal),stage,as.integer(id),chunk_rounds)
  invisible(TRUE)
}
