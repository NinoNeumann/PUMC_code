# Freeze and type the supplied 20261027v2 workbook using the archived 29-variable schema.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
stopifnot(!is.na(Sys.setlocale('LC_CTYPE','English_United States.utf8')))
project <- normalizePath(args[[1]],winslash='/',mustWork=TRUE)
out <- normalizePath(args[[2]],winslash='/',mustWork=TRUE)
.libPaths(c(file.path(project,'.r_library_selector_windows'),.libPaths()))
original <- file.path(project,'results/cacox_tree_selected29_integrated_20261028_v1')
source(file.path(original,'upstream_source/data_imputation_functions.R'),encoding='UTF-8')
cfg <- readRDS(file.path(original,'upstream_cache/cv10_20261027v2/data/config.rds'))
input <- file.path(out,'input','20261027v2.xlsx')
stopifnot(file.exists(input))
raw <- read.csv(file.path(out,'input','20261027v2_text.csv'),
                colClasses='character',na.strings=NULL,check.names=FALSE,
                fileEncoding='UTF-8')
renamed <- c(death='Death',survialday='Survival_Days',gender='Gender',
             age='Age',Antibody1='Antibody')
stopifnot(nrow(raw)==1140L,ncol(raw)==31L,all(names(renamed)%in%names(raw)))
names(raw)[match(names(renamed),names(raw))] <- unname(renamed)
stopifnot(identical(names(raw),c('Death','Survival_Days',cfg$model_vars)))
cfg$input_file <- input
typed <- prepare_data(raw,cfg)
stopifnot(nrow(typed)==1140L,sum(typed$Death)==149L,
          identical(typed$.patient_id,seq_len(1140L)))
saveRDS(cfg,file.path(out,'input','config.rds'),compress=TRUE)
saveRDS(typed,file.path(out,'input','raw_typed.rds'),compress=TRUE)

# Provenance check: the conversion must reproduce the archived typed data
# when applied to the previous 20261023 workbook.
previous <- read.csv(file.path(out,'input','previous_text_for_audit.csv'),
                     colClasses='character',na.strings=NULL,check.names=FALSE,
                     fileEncoding='UTF-8')
names(previous)[match(c('death','survialday','gender','age'),names(previous))] <-
  c('Death','Survival_Days','Gender','Age')
prior_typed <- prepare_data(previous,cfg)
archived <- readRDS(file.path(original,'upstream_cache/cv10_20261027v2/data/raw_typed.rds'))
stopifnot(identical(prior_typed,archived))
same <- vapply(setdiff(names(typed),'Antibody'),function(n)
  identical(typed[[n]],archived[[n]]),logical(1))
stopifnot(all(same))
changed <- sum(as.character(typed$Antibody)!=as.character(archived$Antibody),na.rm=TRUE)
cat(sprintf('INPUT_READY n=%d deaths=%d predictors=%d antibody_changed=%d other_columns_equal=%s\n',
    nrow(typed),sum(typed$Death),length(cfg$model_vars),changed,all(same)))
