#!/usr/bin/env Rscript
# Frozen original CACox ES OOF evaluation. No selection, imputation or training.
script_arg<-grep('^--file=',commandArgs(FALSE),value=TRUE)
if(length(script_arg)!=1L)stop('Run this script with Rscript')
script_path<-gsub('~+~',' ',sub('^--file=','',script_arg),fixed=TRUE)
script_root<-dirname(dirname(normalizePath(script_path,mustWork=TRUE)))
source(file.path(script_root,'code','evaluation.R'),encoding='UTF-8')
opts<-portable_options(c('analysis-root','run-root','output'),flags='validate')
env<-portable_environment(opts,script_root)
output<-portable_path(portable_option(opts,'output','results/subgroup_original_oof'),env$run)
dir.create(output,recursive=TRUE,showWarnings=FALSE)
status_path<-file.path(output,'RUN_STATUS.txt');writeLines('VALIDATING_ORIGINAL_OOF',status_path)

main<-function() {
  if(!requireNamespace('survival',quietly=TRUE))stop('Install the survival R package')
  input<-portable_read_inputs(env);raw<-input$raw;d<-input$main
  populations<-c('full','MPA','GPA');results<-list()
  expected_n<-c(full=1140L,MPA=434L,GPA=583L);expected_events<-c(full=149L,MPA=71L,GPA=62L)
  for(pop in populations) {
    ids<-if(pop=='full')raw$.patient_id else raw$.patient_id[!is.na(raw$Type)&as.character(raw$Type)==pop]
    z<-d[d$patient_id %in% ids,,drop=FALSE]
    fit<-evaluation_population(z,pop)
    if(fit$pooled$patients!=expected_n[pop]||fit$pooled$events!=expected_events[pop])stop('Unexpected subgroup count: ',pop)
    results[[pop]]<-fit
  }
  pooled<-do.call(rbind,lapply(results,`[[`,'pooled'))
  write.csv(pooled,file.path(output,'C_index_pooled.csv'),row.names=FALSE,na='')
  write.csv(do.call(rbind,lapply(results,`[[`,'each')),file.path(output,'C_index_each_imputation.csv'),row.names=FALSE)
  write.csv(do.call(rbind,lapply(results,`[[`,'folds')),file.path(output,'C_index_fold_imputation.csv'),row.names=FALSE)
  desc<-do.call(rbind,lapply(c('MPA','GPA'),function(pop)data.frame(population=pop,
    C_subgroup=results[[pop]]$pooled$estimate,C_full=results$full$pooled$estimate,
    subgroup_minus_full=results[[pop]]$pooled$estimate-results$full$pooled$estimate,
    interpretation='Descriptive case-mix difference; no CI or p; not an interaction test')))
  write.csv(desc,file.path(output,'subgroup_vs_full_descriptive.csv'),row.names=FALSE)
  write.csv(data.frame(relative_input='input/main_oof_cacox.csv.gz',sha256=input$main_sha256),
    file.path(output,'input_provenance.csv'),row.names=FALSE)
  lines<-c('Original CACox ES OOF: full cohort, known MPA and known GPA. No refitting or feature selection.',
    'Original 10 repeats x 5 folds x 20 imputation labels are preserved.',
    'Subgroups use original raw Type; missing Type and EGPA are excluded from MPA/GPA subgroup evaluation.',
    evaluation_methods(),'','Results:')
  for(i in seq_len(nrow(pooled))) {
    z<-pooled[i,]
    lines<-c(lines,sprintf('%s: n=%d, events=%d, C=%.10f, 95%% CI %.10f to %.10f',
      z$population,z$patients,z$events,z$estimate,z$CI95_lower,z$CI95_upper))
  }
  if(isTRUE(opts$validate)) {
    # Reference values from the independently checked original full and subgroup
    # calculations. These constants are summary statistics, not patient data.
    reference<-data.frame(population=populations,
      estimate=c(.749726596296937,.693895330418805,.774579367625566),
      SE=c(.0179486645226525,.0306938619542166,.0266516607854502),
      CI95_lower=c(.714547332502763,.633736208227426,.722342450651338),
      CI95_upper=c(.784905860091112,.754054452610184,.826816284599794),
      within_variance=c(.000317210796348846,.000934379921939371,.000700602095703194),
      between_variance=c(4.70834456940901e-6,7.36499021441483e-6,9.24659706619974e-6),
      total_variance=c(.000322154558146726,.000942113161664506,.000710311022622704))
    checks<-list();ncheck<-0L
    add<-function(name,difference,tolerance=1e-10) {
      ncheck<<-ncheck+1L
      checks[[ncheck]]<<-data.frame(check=name,absolute_difference=difference,tolerance=tolerance,
        passed=is.finite(difference)&&difference<=tolerance)
    }
    for(pop in populations)for(field in names(reference)[-1L]) {
      a<-pooled[pooled$population==pop,field];b<-reference[reference$population==pop,field]
      add(paste('original_reference',pop,field,sep=':'),abs(a-b))
    }
    ids<-raw$.patient_id[input$masks$no_egpa];z<-d[d$patient_id %in% ids,,drop=FALSE]
    identity<-evaluation_compare(z,z,1:50,'identical_frozen_predictions')
    delta<-identity$pooled[identity$pooled$metric=='difference_after_minus_before',]
    add('identical_predictions_zero_difference_and_variance',max(abs(unlist(delta[c('estimate','SE','CI95_lower','CI95_upper','total_variance')]))),0)
    add('zero_variance_does_not_invent_p',as.numeric(!is.na(delta$p_Rubin)),0)
    one<-z[z$imputation==1L,,drop=FALSE];copy<-one;copy$repeat_id<-copy$repeat_id+10L
    a<-evaluation_clustered_C(one);b<-evaluation_clustered_C(rbind(one,copy))
    add('duplicate_repeats_C_invariant',abs(a$C-b$C),1e-12)
    add('duplicate_repeats_variance_invariant',abs(a$variance-b$variance),1e-12)
    add('duplicate_repeats_influence_invariant',max(abs(a$influence-b$influence[names(a$influence)])),1e-12)
    # A single real fitted view copied to 20 pairing labels must retain its
    # original patient variance; the labels must not multiply effective n.
    copied<-do.call(rbind,lapply(1:20,function(j) {v<-one;v$imputation<-j;v}))
    copied_fit<-evaluation_population(copied,'single_view_repeated_labels')$pooled
    add('one_view_20_labels_within_variance_unchanged',abs(copied_fit$within_variance-a$variance),1e-12)
    add('one_view_20_labels_no_between_variance',abs(copied_fit$between_variance),0)
    add('one_view_20_labels_total_variance_unchanged',abs(copied_fit$total_variance-a$variance),1e-12)
    rejected<-inherits(try(evaluation_validate(rbind(z,z[1,]),1:50,'intentional duplicate'),silent=TRUE),'try-error')
    add('duplicate_within_repeat_rejected',as.numeric(!rejected),0)
    rejected<-inherits(try(evaluation_validate(z[z$fold!=5L,],1:50,'intentional missing fold'),silent=TRUE),'try-error')
    add('missing_fold_rejected',as.numeric(!rejected),0)
    checks<-do.call(rbind,checks)
    write.csv(checks,file.path(output,'validation_checks.csv'),row.names=FALSE)
    write.csv(reference,file.path(output,'reference_summary_statistics.csv'),row.names=FALSE)
    if(!all(checks$passed))stop('Numerical/statistical validation failed')
    lines<-c(lines,'',sprintf('Validation: all %d checks passed; maximum absolute discrepancy %.6g.',nrow(checks),max(checks$absolute_difference)),
      'Validation uses original frozen predictions and synthetic duplicate/identity comparisons; it does not claim any newly trained sensitivity result.')
  }
  writeLines(lines,file.path(output,'METHODS_AND_RESULTS.txt'),useBytes=TRUE)
  writeLines(capture.output(sessionInfo()),file.path(output,'sessionInfo.txt'))
  writeLines(if(isTRUE(opts$validate))'SUCCESS FROZEN_OOF_10X5 VALIDATION_PASSED NO_REFITTING' else 'SUCCESS FROZEN_OOF_10X5 NO_REFITTING',status_path)
  cat('SUBGROUP_SUCCESS: CACox ES original OOF full/MPA/GPA; no retraining.\n')
}
tryCatch(main(),error=function(e) {
  writeLines(paste('FAILED:',conditionMessage(e)),status_path)
  stop(e)
})
