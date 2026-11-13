#!/usr/bin/env Rscript
# CACox ES only. Formal aggregation requires all five branches and all 50 folds.
script_arg<-grep('^--file=',commandArgs(FALSE),value=TRUE)
if(length(script_arg)!=1L)stop('Run this script with Rscript')
script_path<-gsub('~+~',' ',sub('^--file=','',script_arg),fixed=TRUE)
script_root<-dirname(dirname(normalizePath(script_path,mustWork=TRUE)))
source(file.path(script_root,'code','evaluation.R'),encoding='UTF-8')
opts<-portable_options(c('analysis-root','run-root','output'))
env<-portable_environment(opts,script_root)
output<-portable_path(portable_option(opts,'output','results/final_10x5_cacox_es'),env$run)
dir.create(output,recursive=TRUE,showWarnings=FALSE)
status_path<-file.path(output,'RUN_STATUS.txt')
writeLines('VALIDATING_ALL_FIVE_BRANCHES_AND_50_FOLDS',status_path)

main<-function() {
  if(!requireNamespace('survival',quietly=TRUE))stop('Install the survival R package')
  input<-portable_read_inputs(env);raw<-input$raw;base<-input$main
  branches<-names(input$masks);ordinals<-1:50
  write.csv(input$counts,file.path(output,'cohort_counts.csv'),row.names=FALSE)
  fold_name<-function(i)sprintf('repeat%02d_fold%02d',(i-1L)%/%5L+1L,(i-1L)%%5L+1L)
  status<-do.call(rbind,lapply(branches,function(b)do.call(rbind,lapply(ordinals,function(i) {
    rel<-file.path('branches',b,fold_name(i),'cacox_es')
    folder<-file.path(env$run,rel)
    data.frame(branch=b,ordinal=i,relative_folder=gsub('\\\\','/',rel),
      prediction_exists=file.exists(file.path(folder,'predictions_all20.csv')),
      complete_exists=file.exists(file.path(folder,'COMPLETE.json')))
  }))))
  write.csv(status,file.path(output,'requested_input_status.csv'),row.names=FALSE)
  if(any(!status$prediction_exists|!status$complete_exists))
    stop('Incomplete formal coverage: all 250 completed branch-folds are required. No partial analysis is generated.')
  predictions<-list();provenance<-list();counter<-0L
  for(branch in branches) {
    ids<-raw$.patient_id[input$masks[[branch]]]
    parts<-lapply(ordinals,function(i) {
      rel<-file.path('branches',branch,fold_name(i),'cacox_es')
      folder<-file.path(env$run,rel);path<-file.path(folder,'predictions_all20.csv')
      marker<-file.path(folder,'COMPLETE.json')
      # The helper verifies the COMPLETE signature/manifest and SHA256 files.
      portable_helper(env,folder,script='verify_prediction.py')
      guard<-unname(tools::md5sum(c(path,marker)))
      z<-read.csv(path,check.names=FALSE)
      if(!identical(guard,unname(tools::md5sum(c(path,marker)))))
        stop('A completed prediction or marker changed while it was being read: ',rel)
      required<-c('repeat_id','fold','imputation','patient_id','Death','Survival_Days','score_E_selected')
      if(!all(required %in% names(z)))stop('Missing prediction columns: ',rel)
      z<-z[required];z$score<-z$score_E_selected;z$score_E_selected<-NULL
      actual<-unique((z$repeat_id-1L)*5L+z$fold)
      if(length(actual)!=1L||actual!=i)stop('Wrong outer fold in prediction file: ',rel)
      counter<<-counter+1L
      provenance[[counter]]<<-data.frame(branch=branch,ordinal=i,
        relative_prediction=gsub('\\\\','/',file.path(rel,'predictions_all20.csv')),
        bytes=file.info(path)$size,complete_and_sha256_verified=TRUE,
        read_guard_md5=guard[1],complete_guard_md5=guard[2])
      z
    })
    z<-evaluation_validate(do.call(rbind,parts),ordinals,branch)
    if(!setequal(z$patient_id,ids))stop('Restricted cohort membership mismatch: ',branch)
    ref<-base[base$patient_id %in% ids,,drop=FALSE]
    j<-match(evaluation_key(z),evaluation_key(ref))
    if(nrow(z)!=nrow(ref)||anyNA(j)||any(z$Death!=ref$Death[j])||
       any(z$Survival_Days!=ref$Survival_Days[j]))
      stop('Predictions do not match the original frozen assessment patients/folds/outcomes: ',branch)
    if(branch=='complete_case') {
      one<-z[z$imputation==1L,];j<-match(evaluation_key(z,FALSE),evaluation_key(one,FALSE))
      if(anyNA(j)||max(abs(z$score-one$score[j]))>1e-12)
        stop('Complete-case predictions must duplicate one real fitted prediction over 20 pairing labels')
    }
    predictions[[branch]]<-z
  }
  comparisons<-list();case_mix<-list()
  full_C<-evaluation_population(base,'full')$pooled$estimate
  for(branch in branches) {
    after<-predictions[[branch]];before<-base[base$patient_id %in% unique(after$patient_id),]
    fit<-evaluation_compare(before,after,ordinals,paste0(branch,':refitted_minus_original_same_patients'),
      list(model='CACox_ES',comparison_type='same_algorithm_same_patients',
        branch_before='original_full_model',branch_after=branch,analysis_scope='COMPLETE_50_OF_50'))
    comparisons[[branch]]<-fit
    a<-fit$pooled$estimate[fit$pooled$metric=='C_after'];b<-fit$pooled$estimate[fit$pooled$metric=='C_before']
    case_mix[[branch]]<-data.frame(branch=branch,model='CACox_ES',patients=length(unique(after$patient_id)),
      C_original_full=full_C,C_original_retained=b,C_refitted_retained=a,
      original_retained_minus_original_full=b-full_C,refitted_retained_minus_original_full=a-full_C,
      interpretation='Descriptive case-mix contrast only; no CI or p')
  }
  common<-raw$.patient_id[input$masks$no_egpa]
  known<-predictions$known_subtype
  known<-known[known$patient_id %in% common,,drop=FALSE]
  comparisons[['EGPA_exclusion']]<-evaluation_compare(known,predictions$no_egpa,ordinals,
    'no_egpa_minus_known_subtype_on_common_MPA_GPA',
    list(model='CACox_ES',comparison_type='EGPA_exclusion_on_common_MPA_GPA',
      branch_before='known_subtype',branch_after='no_egpa',analysis_scope='COMPLETE_50_OF_50'))
  pooled<-do.call(rbind,lapply(comparisons,`[[`,'pooled'))
  write.csv(do.call(rbind,lapply(comparisons,`[[`,'each')),file.path(output,'C_index_each_imputation.csv'),row.names=FALSE,na='')
  write.csv(pooled,file.path(output,'C_index_pooled_and_difference.csv'),row.names=FALSE,na='')
  write.csv(do.call(rbind,lapply(comparisons,`[[`,'folds')),file.path(output,'C_index_fold_coverage.csv'),row.names=FALSE)
  write.csv(do.call(rbind,case_mix),file.path(output,'full_cohort_case_mix_descriptive.csv'),row.names=FALSE)
  write.csv(do.call(rbind,provenance),file.path(output,'prediction_provenance.csv'),row.names=FALSE)
  write.csv(data.frame(relative_input='input/main_oof_cacox.csv.gz',sha256=input$main_sha256),
    file.path(output,'original_oof_provenance.csv'),row.names=FALSE)
  lines<-c('CACox ES sensitivity: complete 10 repeats x 5 folds in all five branches.',
    'All 250 completion markers, prediction SHA256 values, patients, folds, imputation labels and outcomes were checked.',
    'Five primary differences: restricted-cohort-refitted CACox ES minus original full-cohort CACox ES, on identical held-out patients.',
    'EGPA contrast: no_egpa minus known_subtype, both evaluated on common MPA/GPA patients; both development cohorts exclude missing Type.',
    evaluation_methods(),'','Paired differences:')
  for(i in which(pooled$metric=='difference_after_minus_before')) {
    z<-pooled[i,]
    lines<-c(lines,sprintf('%s: n=%d, events=%d, delta=%.10f, 95%% CI %.10f to %.10f, Rubin p=%s',
      z$comparison,z$patients,z$events,z$estimate,z$CI95_lower,z$CI95_upper,
      if(is.na(z$p_Rubin))'undefined (zero variance)' else format(z$p_Rubin,digits=10)))
  }
  writeLines(lines,file.path(output,'METHODS_AND_RESULTS.txt'),useBytes=TRUE)
  writeLines(capture.output(sessionInfo()),file.path(output,'sessionInfo.txt'))
  writeLines('SUCCESS COMPLETE_50_OF_50 model CACox_ES branches 5 comparisons 6',status_path)
  cat('AGGREGATION_SUCCESS: CACox ES, all five branches x 50 folds, six paired comparisons.\n')
}
tryCatch(main(),error=function(e) {
  writeLines(paste('FAILED_NO_NEW_INFERENTIAL_REPORT:',conditionMessage(e)),status_path)
  stop(e)
})
