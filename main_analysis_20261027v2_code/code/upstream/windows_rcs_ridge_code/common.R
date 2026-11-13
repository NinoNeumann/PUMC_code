# Separate output version. Never writes into the completed unpenalised run.
project_dir <- '/Users/asklepios/Documents/ChatGPT/aav文章修改'
source_dir <- file.path(project_dir, 'cv10_20261027v2')
run_dir <- file.path(project_dir, 'cv10_20261027v2_rcs_ridge')
code_dir <- file.path(run_dir, 'code')
.libPaths(c(file.path(project_dir, '.Rlib'), .libPaths()))
for (p in c('mice', 'survival')) if (!requireNamespace(p, quietly=TRUE)) stop('Missing package: ', p)
for (f in c('data_imputation_functions.R', 'selection29_functions.R',
            'cox29_reliable_fit.R', 'cox29_fitting_functions.R', 'time_auc.R',
            'ridge_functions.R')) source(file.path(code_dir, f), encoding='UTF-8')
ridge_grid <- 10^seq(-4, 4, by=1)
fold_name <- function(i) sprintf('repeat%02d_fold%02d', (i-1L)%/%5L+1L, (i-1L)%%5L+1L)
cfg_ridge <- function(i) {
  stopifnot(i %in% 1:10)
  cfg <- readRDS(file.path(source_dir, 'data/config.rds'))
  cfg$output_dir <- file.path(run_dir, fold_name(i))
  cfg$repeat_id <- (i-1L)%/%5L+1L; cfg$fold_id <- (i-1L)%%5L+1L
  cfg
}
input_files <- function(i) file.path(source_dir, fold_name(i), c(
  'prepared.rds', 'outer/imputation.rds', sprintf('inner_%02d/imputation.rds', 1:5),
  'selection_frequency_29variables.csv', 'models/models_all20.rds',
  'models/predictions_all20.csv'))
code_signature <- function() tools::md5sum(sort(list.files(code_dir, pattern='[.]R$', full.names=TRUE)))
verify_ridge <- function(i=NULL) {
  f <- file.path(run_dir, 'run_manifest.rds')
  if (!file.exists(f)) stop('Run initialize.R once before computation')
  m <- readRDS(f)
  stopifnot(identical(m$code, code_signature()), identical(m$grid, ridge_grid),
    identical(m$packages, vapply(c('survival','mice'), function(p) as.character(utils::packageVersion(p)), character(1))))
  if (!is.null(i)) {
    files <- c(file.path(source_dir, 'data/config.rds'), input_files(i))
    if (!identical(m$input[files], tools::md5sum(files))) stop('Source cache changed; refuse reuse')
  }
}
status_ridge <- function(i, stage, message='') {
  p <- file.path(run_dir, fold_name(i)); dir.create(p, recursive=TRUE, showWarnings=FALSE)
  write_csv(data.frame(ordinal=i, stage=stage, message=message,
    time=format(Sys.time(), tz='Asia/Shanghai', usetz=TRUE)), file.path(p,'status.csv'))
}
