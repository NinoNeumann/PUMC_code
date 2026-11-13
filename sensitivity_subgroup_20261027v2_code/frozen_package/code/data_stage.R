#!/usr/bin/env Rscript
args0<-commandArgs(FALSE);script<-sub('^--file=','',grep('^--file=',args0,value=TRUE)[1L])
script<-gsub('~+~',' ',script,fixed=TRUE) # Rscript encodes spaces in --file.
source(file.path(dirname(normalizePath(script,winslash='/',mustWork=TRUE)),'common.R'),encoding='UTF-8')
args<-commandArgs(trailingOnly=TRUE)
if(length(args)==0L||identical(args[[1]],'status')) {
  z<-data_status()
  print(aggregate(cbind(prepared,imputation_partitions,selection_replicates,summary)~branch,z,sum),row.names=FALSE)
} else {
  if(length(args)<3L)stop('Usage: data_stage.R branch ordinal prepare|impute|select|summary [id] [chunk_rounds]')
  run_data_stage(args[[1]],as.integer(args[[2]]),args[[3]],
    if(length(args)>=4L)as.integer(args[[4]])else 0L,
    if(length(args)>=5L)as.numeric(args[[5]])else Inf)
}
