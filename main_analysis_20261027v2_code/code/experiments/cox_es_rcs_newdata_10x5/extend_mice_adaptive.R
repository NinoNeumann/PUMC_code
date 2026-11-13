# Continue the same saved 20 MICE chains for one predeclared 20-iteration step.
# Original 60-iteration artifacts stay read-only. The output path is versioned.
argv <- commandArgs(trailingOnly=TRUE)
stopifnot(length(argv)==6L)
stopifnot(!is.na(Sys.setlocale('LC_CTYPE','English_United States.utf8')))
project <- normalizePath(argv[[1]],winslash='/',mustWork=TRUE)
root <- normalizePath(argv[[2]],winslash='/',mustWork=TRUE)
r <- as.integer(argv[[3]]); k <- as.integer(argv[[4]])
id <- as.integer(argv[[5]]); target <- as.integer(argv[[6]])
stopifnot(r %in% 1:10,k %in% 1:5,id %in% 0:5,target %in% c(80L,100L,120L))
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
source(file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/upstream_source/data_imputation_functions.R'),encoding='UTF-8')
source(file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1/code/selection29_fast.R'),encoding='UTF-8')
part <- file.path(root,sprintf('repeat%02d_fold%02d',r,k),
                  if(id==0L)'outer' else sprintf('inner_%02d',id))
source_path <- if(target==80L) file.path(part,'imputation.rds') else
  file.path(part,'adaptive',sprintf('iter_%03d',target-20L),'imputation.rds')
dest_dir <- file.path(part,'adaptive',sprintf('iter_%03d',target))
dest <- file.path(dest_dir,'imputation.rds')
stopifnot(file.exists(source_path),!file.exists(dest_dir))
before <- readRDS(source_path)
old <- before$mids
stopifnot(old$m==20L,old$iteration==target-20L,is.null(old$loggedEvents))
start <- proc.time()[['elapsed']]
after <- mice::mice.mids(old,maxit=20L,printFlag=FALSE)
stopifnot(after$m==20L,after$iteration==target,
          identical(after$ignore,old$ignore),identical(after$data,old$data),
          identical(after$method,old$method),
          identical(after$predictorMatrix,old$predictorMatrix),
          is.null(after$loggedEvents))
dir.create(dest_dir,recursive=TRUE,showWarnings=FALSE)
diagnostics <- diagnose29_mice(after,dest_dir)
before$mids <- after
temp <- paste0(dest,'.tmp.',Sys.getpid())
saveRDS(before,temp,compress=TRUE)
if(!file.rename(temp,dest)) stop('Could not atomically save extended MICE object')
cat(sprintf('ADAPTIVE_MICE repeat=%d fold=%d part=%d iterations=%d max_Rhat=%.8f seconds=%.2f\n',
            r,k,id,target,max(diagnostics$classical_Rhat,na.rm=TRUE),
            proc.time()[['elapsed']]-start))
