# Run once after reviewing the new code. Only fingerprints saved input caches.
source('/Users/asklepios/Documents/ChatGPT/aav文章修改/cv10_20261027v2_rcs_ridge/code/common.R',encoding='UTF-8')
path <- file.path(run_dir,'run_manifest.rds')
if(file.exists(path))stop('Manifest exists; do not overwrite an initialized run')
files <- unique(c(file.path(source_dir,'data/config.rds'),unlist(lapply(1:10,input_files))))
if(!all(file.exists(files)))stop('Missing source caches: ',paste(files[!file.exists(files)],collapse=', '))
for(i in 1:10) {
  p<-file.path(source_dir,fold_name(i))
  stopifnot(file.exists(file.path(p,'models/FITS_COMPLETE.txt')),
    file.exists(file.path(p,'EVALUATION_COMPLETE.txt')))
  status<-read.csv(file.path(p,'models/fit_status.csv'))
  stopifnot(nrow(status)==40,all(status$success))
}
manifest <- list(code=code_signature(),input=tools::md5sum(files),grid=ridge_grid,
  packages=vapply(c('survival','mice'),function(p)as.character(utils::packageVersion(p)),character(1)),
  created=format(Sys.time(),tz='Asia/Shanghai',usetz=TRUE),
  models=c('ordinary_cox','rcs_ridge_cox'),source=source_dir)
cox29_atomic_rds(manifest,path)
write_csv(data.frame(path=names(manifest$input),md5=unname(manifest$input)),file.path(run_dir,'source_input_manifest.csv'))
writeLines(capture.output(sessionInfo()),file.path(run_dir,'sessionInfo.txt'))
cat('Initialized separate ridge run. No fits have been started.\n')
