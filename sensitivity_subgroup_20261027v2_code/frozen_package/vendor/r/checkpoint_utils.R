# Filesystem utilities only; no Cox or RCS model fitting is included.
portable_commit <- function(tmp,path) {
  # Windows file.rename cannot replace an existing file.
  if (.Platform$OS.type == 'windows' && file.exists(path)) {
    if (!unlink(path)==0L) stop('Cannot replace file: ',path)
  }
  if (!file.rename(tmp,path)) stop('Cannot commit file: ',path)
  invisible(path)
}
cox29_atomic_rds <- function(object,path) {
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  tmp<-tempfile('.writing_',tmpdir=dirname(path));on.exit(unlink(tmp),add=TRUE)
  saveRDS(object,tmp);portable_commit(tmp,path)
}
cox29_signature <- function(object) {
  tmp<-tempfile();on.exit(unlink(tmp));saveRDS(object,tmp,compress=FALSE)
  unname(tools::md5sum(tmp))
}
