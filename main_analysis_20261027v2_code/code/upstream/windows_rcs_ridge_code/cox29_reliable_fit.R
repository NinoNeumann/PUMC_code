# Numerical safeguards only. No penalty, variable deletion, or change of model.
cox29_atomic_rds <- function(object, path) {
 dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
 tmp<-tempfile('.writing_',tmpdir=dirname(path));on.exit(unlink(tmp),add=TRUE)
 saveRDS(object,tmp);if(!file.rename(tmp,path))stop('Cannot commit file: ',path)
}
cox29_signature <- function(object) {
 tmp<-tempfile();on.exit(unlink(tmp));saveRDS(object,tmp,compress=FALSE)
 unname(tools::md5sum(tmp))
}
