source('/Users/asklepios/Documents/ChatGPT/aav文章修改/cv10_20261027v2_rcs_ridge/code/common.R',encoding='UTF-8')
verify_ridge();pred<-coverage<-perC<-perA<-list()
out<-file.path(run_dir,'summary');dir.create(out,showWarnings=FALSE)
for(i in 1:10) {
  p<-file.path(run_dir,fold_name(i));f<-file.path(p,'models/predictions_all20.csv')
  good<-file.exists(f)&&file.exists(file.path(p,'EVALUATION_COMPLETE.txt'))
  coverage[[i]]<-data.frame(ordinal=i,complete_paired20=good)
  if(good) {
    d<-read.csv(f);stopifnot(setequal(d$imputation,1:20));pred[[as.character(i)]]<-d
    perC[[as.character(i)]]<-data.frame(ordinal=i,read.csv(file.path(p,'evaluation/C_index_pooled_and_difference.csv')))
    perA[[as.character(i)]]<-data.frame(ordinal=i,read.csv(file.path(p,'evaluation/AUC_pooled_and_difference.csv')))
  }
}
coverage<-do.call(rbind,coverage);write_csv(coverage,file.path(out,'fold_coverage.csv'))
if(!length(pred))stop('No complete evaluated folds')
write_csv(do.call(rbind,perC),file.path(out,'C_index_by_fold.csv'))
write_csv(do.call(rbind,perA),file.path(out,'AUC_by_fold.csv'))
complete<-all(coverage$complete_paired20)
dest<-file.path(out,if(complete)'complete_10fold'else'exploratory_successful_fold_subset')
d<-do.call(rbind,pred)
if(complete)stopifnot(nrow(d)==1140*2*20)
evaluate29_predictions(d,dest,'Ordinary Cox vs RCS-Ridge-Cox: repeated five-fold CV')
writeLines(c(paste('Available folds:',sum(coverage$complete_paired20),'of10'),
  paste('Missing:',paste(coverage$ordinal[!coverage$complete_paired20],collapse=',')),
  'C-index: within-fold comparable-pair weighting, patient-clustered IJ covariance; Rubin pooling over20 MI.',
  'AUC: within-fold N-weighted cumulative/dynamic IPCW, paired patient-clustered jackknife; Rubin pooling.',
  'Conditional intervals exclude model development, variable selection and ridge tuning uncertainty.',
  'Incomplete-fold results are exploratory, potentially biased by selective failures.',
  'Ordinary Cox retained from original valid fits; RCS-Ridge-Cox newly tuned/fitted. No bootstrap or optimism correction.'),
  file.path(out,'METHODS_AND_LIMITATIONS.txt'))
writeLines(format(Sys.time(),tz='Asia/Shanghai'),file.path(run_dir,
  if(complete)'ALL_10_FOLDS_COMPLETE.txt'else'INCOMPLETE.txt'))
