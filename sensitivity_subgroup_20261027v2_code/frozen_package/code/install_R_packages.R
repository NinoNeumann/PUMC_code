# Install into the extracted package; do not update an existing working library.
root <- Sys.getenv('CACOX_PORTABLE_ROOT', getwd())
lib <- Sys.getenv('CACOX_R_LIBS',file.path(root,'.Rlib'))
dir.create(lib,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(lib,.libPaths()))
packages <- c('mice','survival','ranger','glmnet')
missing <- packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,lib=lib,repos='https://cloud.r-project.org',Ncpus=2L)
if(!all(vapply(packages,requireNamespace,logical(1),quietly=TRUE)))stop('Required R packages are still missing')
reference <- c(mice='3.19.0',survival='3.8.3',ranger='0.18.0',glmnet='5.1')
versions <- vapply(packages,function(p)as.character(utils::packageVersion(p)),character(1))
report <- data.frame(package=packages,installed=versions,reference=reference[packages],
                     matches_reference=unname(versions)==unname(reference[packages]))
write.csv(report,file.path(root,'R_install_versions.csv'),row.names=FALSE)
print(report,row.names=FALSE)
cat('R packages ready. Actual versions will be frozen by data initialization.\n')
if(any(!report$matches_reference))cat('Version differences from the Mac reference are recorded; exact cross-environment numerical identity is not claimed.\n')
