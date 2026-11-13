args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2L)
root <- normalizePath(args[[1]], winslash='/', mustWork=TRUE)
fold <- as.integer(args[[2]])
stopifnot(fold %in% 1:5)
base <- file.path(root, sprintf('upstream_cache/cv10_20261027v2/repeat01_fold%02d', fold))
dest <- file.path(root, sprintf('bundles/repeat01_fold%02d/main', fold))
dir.create(dest, recursive=TRUE, showWarnings=FALSE)
cfg <- readRDS(file.path(root, 'upstream_cache/cv10_20261027v2/data/config.rds'))
complete_without_mice <- function(bundle, j) {
  mids <- bundle$mids
  stopifnot(mids$m == 20, mids$iteration == 60, j %in% 1:20)
  z <- mids$data
  for (v in names(mids$imp)) {
    imp <- mids$imp[[v]]
    if (nrow(imp) == 0L) next
    ids <- as.integer(rownames(imp))
    stopifnot(length(ids) == nrow(imp), all(mids$where[ids, v]))
    z[ids, v] <- imp[[j]]
  }
  stopifnot(!anyNA(z))
  n <- nrow(bundle$development)
  d <- bundle$development
  a <- bundle$assessment
  for (v in names(z)) {
    d[[v]] <- z[seq_len(n), v]
    a[[v]] <- z[n + seq_len(nrow(a)), v]
    observed <- !is.na(mids$data[[v]])
    stopifnot(identical(as.character(z[[v]][observed]), as.character(mids$data[[v]][observed])))
  }
  names(d)[names(d) == '.patient_id'] <- 'patient_id'
  names(a)[names(a) == '.patient_id'] <- 'patient_id'
  list(dev=d, assessment=a)
}
for (part in c('outer',sprintf('inner_%02d',1:5))) {
  b <- readRDS(file.path(base, part, 'imputation.rds'))
  target <- file.path(dest, part)
  dir.create(target, recursive=TRUE, showWarnings=FALSE)
  if (part == 'outer') {
    raw_d <- b$development; raw_a <- b$assessment
    names(raw_d)[names(raw_d) == '.patient_id'] <- 'patient_id'
    names(raw_a)[names(raw_a) == '.patient_id'] <- 'patient_id'
    write.csv(raw_d,file.path(dest,'raw_development.csv'),row.names=FALSE,na='')
    write.csv(raw_a,file.path(dest,'raw_assessment.csv'),row.names=FALSE,na='')
  }
  for (j in 1:20) {
    q <- complete_without_mice(b,j)
    write.csv(q$dev,file.path(target,sprintf('development_%02d.csv',j)),row.names=FALSE,na='')
    write.csv(q$assessment,file.path(target,sprintf('assessment_%02d.csv',j)),row.names=FALSE,na='')
  }
  cat(sprintf('EXPORTED fold=%d part=%s\n',fold,part))
}
