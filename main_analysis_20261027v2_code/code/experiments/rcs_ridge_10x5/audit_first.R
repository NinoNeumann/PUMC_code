args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
project <- normalizePath(args[[1]],winslash='/',mustWork=TRUE)
input <- normalizePath(args[[2]],winslash='/',mustWork=TRUE)
old <- file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/upstream_cache/cv10_20261027v2')
for(fold in 1:5) {
  key <- sprintf('repeat01_fold%02d',fold)
  a <- readRDS(file.path(input,key,'prepared.rds'))
  b <- readRDS(file.path(old,key,'prepared.rds'))
  stopifnot(identical(a$data$dev,b$data$dev),
            identical(a$data$assessment,b$data$assessment),
            identical(a$data$inner_fold,b$data$inner_fold))
  cat(sprintf('FIRST_REPEAT_INPUT_IDENTICAL fold=%d\n',fold))
}
