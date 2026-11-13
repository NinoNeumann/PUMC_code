#!/usr/bin/env Rscript
args0<-commandArgs(FALSE);script<-sub('^--file=','',grep('^--file=',args0,value=TRUE)[1L])
script<-gsub('~+~',' ',script,fixed=TRUE) # Rscript encodes spaces in --file.
source(file.path(dirname(normalizePath(script,winslash='/',mustWork=TRUE)),'common.R'),encoding='UTF-8')
a<-commandArgs(trailingOnly=TRUE)
if(length(a)!=2L)stop('Usage: data_fold.R branch ordinal')
b<-a[[1]];i<-as.integer(a[[2]])
verify_data_sources()
sensitivity_stage(b,i,'prepare')
for(j in 0:5)sensitivity_stage(b,i,'impute',j)
for(j in 1:20)sensitivity_stage(b,i,'select',j)
sensitivity_stage(b,i,'summary')
invisible(get_fold_input(b,i,validate=TRUE))
cat('DATA_FOLD_COMPLETE',b,fold_name(i),'\n')
