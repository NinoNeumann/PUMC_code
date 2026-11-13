root <- Sys.getenv('CACOX_PORTABLE_ROOT',getwd())
lib <- Sys.getenv('CACOX_R_LIBS',file.path(root,'.Rlib'))
if(dir.exists(lib)).libPaths(c(lib,.libPaths()))
packages <- c('mice','survival','ranger','glmnet')
missing <- packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing))stop('Missing R packages: ',paste(missing,collapse=', '),'; run python run.py install-r')
out <- Sys.getenv('CACOX_RUN_ROOT',file.path(root,'run'))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(data.frame(package=packages,version=vapply(packages,function(p)as.character(utils::packageVersion(p)),character(1))),
          file.path(out,'environment_R.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo_R.txt'))
cat('R environment verified:',R.version.string,'\n')
