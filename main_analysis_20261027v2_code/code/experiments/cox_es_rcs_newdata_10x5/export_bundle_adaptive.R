# Export the exact R MICE completion into the existing Python Bundle contract.
argv <- commandArgs(trailingOnly=TRUE)
stopifnot(length(argv)==4L)
stopifnot(!is.na(Sys.setlocale('LC_CTYPE','English_United States.utf8')))
project <- normalizePath(argv[[1]],winslash='/',mustWork=TRUE)
root <- normalizePath(argv[[2]],winslash='/',mustWork=TRUE)
r <- as.integer(argv[[3]]); k <- as.integer(argv[[4]])
stopifnot(r%in%1:10,k%in%1:5)
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
stopifnot(requireNamespace('mice',quietly=TRUE))
source(file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/upstream_source/data_imputation_functions.R'),encoding='UTF-8')
source(file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/code/selection29_fast.R'),encoding='UTF-8')
fold <- file.path(root,sprintf('repeat%02d_fold%02d',r,k))
dest <- file.path(root,'bundles',sprintf('repeat%02d_fold%02d',r,k),'main')
dir.create(dest,recursive=TRUE,showWarnings=FALSE)
cfg <- readRDS(file.path(root,'input/config.rds'))
selected <- readLines(file.path(fold,'selected_variables.txt'),warn=FALSE)
stopifnot(length(selected)>0L,identical(selected,cfg$model_vars[cfg$model_vars%in%selected]))
effective_mice <- function(part) {
  base <- file.path(fold,part)
  for(iter in c(120L,100L,80L)) {
    candidate <- file.path(base,'adaptive',sprintf('iter_%03d',iter),'imputation.rds')
    if(file.exists(candidate)) return(candidate)
  }
  file.path(base,'imputation.rds')
}
for(part in c('outer',sprintf('inner_%02d',1:5))) {
  path <- effective_mice(part)
  bundle <- readRDS(path)
  stopifnot(bundle$mids$m==20L,bundle$mids$iteration %in% c(60L,80L,100L,120L))
  tab <- read.csv(file.path(dirname(path),'mice_chain_diagnostic.csv'))
  stopifnot(!any(tab$classical_Rhat>1.1,na.rm=TRUE),
            is.null(bundle$mids$loggedEvents))
  dir.create(file.path(dest,part),showWarnings=FALSE)
  if(part=='outer') {
    rawd <- bundle$development; rawa <- bundle$assessment
    names(rawd)[names(rawd)=='.patient_id'] <- 'patient_id'
    names(rawa)[names(rawa)=='.patient_id'] <- 'patient_id'
    write.csv(rawd,file.path(dest,'raw_development.csv'),row.names=FALSE,na='')
    write.csv(rawa,file.path(dest,'raw_assessment.csv'),row.names=FALSE,na='')
  }
  for(j in 1:20) {
    ds <- complete29(bundle,j)
    names(ds$dev)[names(ds$dev)=='.patient_id'] <- 'patient_id'
    names(ds$assessment)[names(ds$assessment)=='.patient_id'] <- 'patient_id'
    write.csv(ds$dev,file.path(dest,part,sprintf('development_%02d.csv',j)),row.names=FALSE,na='')
    write.csv(ds$assessment,file.path(dest,part,sprintf('assessment_%02d.csv',j)),row.names=FALSE,na='')
  }
  cat(sprintf('EXPORTED repeat=%d fold=%d partition=%s\n',r,k,part))
}
