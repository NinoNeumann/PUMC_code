# Conditional inference for frozen and refitted CACox ES predictions.
# Pure evaluation helpers: no model fitting.
evaluation_key <- function(d, include_imputation=TRUE) {
  fields<-c('repeat_id','fold',if(include_imputation)'imputation','patient_id')
  do.call(paste,c(d[fields],sep='\r'))
}

evaluation_validate <- function(d,ordinals,label='predictions') {
  columns<-c('repeat_id','fold','imputation','patient_id','Death','Survival_Days','score')
  if(!all(columns %in% names(d)))stop(label,': missing required columns')
  d<-d[columns]
  if(!nrow(d)||anyNA(d)||!all(vapply(d,is.numeric,logical(1)))||
     !all(is.finite(as.matrix(d))))stop(label,': empty/missing/nonfinite/non-numeric values')
  if(!setequal(unique(d$imputation),1:20)||any(!d$Death %in% 0:1)||any(d$Survival_Days<0))
    stop(label,': invalid imputations or outcomes')
  if(any(d$repeat_id!=as.integer(d$repeat_id))||any(d$repeat_id<1L)||any(!d$fold %in% 1:5)||
     any(d$imputation!=as.integer(d$imputation)))stop(label,': invalid fold/repeat/MI identifiers')
  actual<-sort(unique((d$repeat_id-1L)*5L+d$fold))
  if(!identical(as.integer(actual),as.integer(sort(ordinals))))stop(label,': unexpected outer-fold coverage')
  if(anyDuplicated(evaluation_key(d))||
     anyDuplicated(d[c('repeat_id','imputation','patient_id')]))stop(label,': duplicate patients within a repeat/imputation')
  nrepeats<-length(unique(d$repeat_id))
  if(any(table(d$patient_id)!=nrepeats*20L))stop(label,': patients missing from repeats or imputations')
  ref<-d[d$imputation==1L,,drop=FALSE];ref_keys<-evaluation_key(ref,FALSE)
  for(j in 2:20) {
    z<-d[d$imputation==j,,drop=FALSE];i<-match(ref_keys,evaluation_key(z,FALSE))
    if(nrow(z)!=nrow(ref)||anyNA(i)||any(z$Death[i]!=ref$Death)||any(z$Survival_Days[i]!=ref$Survival_Days))
      stop(label,': imputation membership or outcome mismatch')
  }
  one<-ref[!duplicated(ref$patient_id),]
  i<-match(ref$patient_id,one$patient_id)
  if(any(ref$Death!=one$Death[i])||any(ref$Survival_Days!=one$Survival_Days[i]))
    stop(label,': repeated patient outcomes differ')
  d[order(d$imputation,d$repeat_id,d$fold,d$patient_id),,drop=FALSE]
}

evaluation_clustered_C <- function(d) {
  parts<-split(d,interaction(d$repeat_id,d$fold,drop=TRUE,lex.order=TRUE))
  fits<-lapply(parts,function(z)survival::concordance(
    survival::Surv(Survival_Days,Death)~score,data=z,reverse=TRUE,timewt='n',influence=2))
  counts<-Reduce('+',lapply(fits,function(z)as.numeric(z$count)))
  den<-sum(counts[1:3]);if(!is.finite(den)||den<=0)stop('No comparable survival pairs')
  value<-(counts[1]+.5*counts[3])/den
  ids<-sort(unique(as.character(d$patient_id)));inf<-setNames(numeric(length(ids)),ids)
  detail<-vector('list',length(parts))
  for(k in seq_along(parts)) {
    z<-parts[[k]];m<-fits[[k]]$influence
    if(!is.matrix(m)||nrow(m)!=nrow(z)||ncol(m)<3L)stop('Unexpected concordance influences')
    contribution<-(m[,1]+.5*m[,3]-value*rowSums(m[,1:3,drop=FALSE]))/den
    who<-as.character(z$patient_id);inf[who]<-inf[who]+contribution
    npairs<-sum(fits[[k]]$count[1:3])
    if(!is.finite(npairs)||npairs<=0)stop('A required outer test fold has no comparable survival pairs')
    detail[[k]]<-data.frame(repeat_id=z$repeat_id[1],fold=z$fold[1],patients=nrow(z),
      events=sum(z$Death),comparable_pairs=npairs,
      C=(fits[[k]]$count[1]+.5*fits[[k]]$count[3])/npairs,weight=npairs/den)
  }
  list(C=unname(value),influence=inf,variance=sum(inf^2),patients=length(ids),
    events=sum(d$Death[!duplicated(d$patient_id)]),prediction_rows=nrow(d),
    comparable_pairs=unname(den),folds=do.call(rbind,detail))
}

evaluation_joint <- function(before,after) {
  if(!setequal(names(before$influence),names(after$influence)))stop('Primary comparison requires exactly matched patients')
  b<-before$influence;a<-after$influence[names(b)]
  data.frame(C_before=before$C,C_after=after$C,difference_after_minus_before=after$C-before$C,
    variance_before=sum(b^2),variance_after=sum(a^2),covariance=sum(a*b),
    variance_difference=sum((a-b)^2),patients=after$patients,events=after$events,
    prediction_rows=after$prediction_rows,comparable_pairs=after$comparable_pairs)
}

evaluation_pool <- function(Q,U,test=FALSE) {
  if(length(Q)!=20L||length(U)!=20L||any(!is.finite(Q))||any(!is.finite(U))||any(U<0))
    stop('Rubin pooling requires 20 finite estimates and nonnegative variances')
  within<-mean(U);between<-var(Q);total<-within+1.05*between
  df<-if(between==0)Inf else 19*(1+within/(1.05*between))^2
  se<-sqrt(total);margin<-qt(.975,df)*se
  data.frame(estimate=mean(Q),SE=se,CI95_lower=mean(Q)-margin,CI95_upper=mean(Q)+margin,
    within_variance=within,between_variance=between,total_variance=total,df=df,
    p_Rubin=if(test&&se>0)2*pt(-abs(mean(Q)/se),df) else NA_real_)
}

evaluation_compare <- function(before,after,ordinals,comparison,metadata=list()) {
  before<-evaluation_validate(before,ordinals,paste(comparison,'before'))
  after<-evaluation_validate(after,ordinals,paste(comparison,'after'))
  i<-match(evaluation_key(after),evaluation_key(before))
  if(nrow(before)!=nrow(after)||anyNA(i)||any(before$Death[i]!=after$Death)||
     any(before$Survival_Days[i]!=after$Survival_Days))stop('Comparison is not exactly paired: ',comparison)
  before<-before[i,,drop=FALSE]
  each<-details<-vector('list',20L)
  for(j in 1:20) {
    b<-evaluation_clustered_C(before[before$imputation==j,,drop=FALSE])
    a<-evaluation_clustered_C(after[after$imputation==j,,drop=FALSE])
    if(b$comparable_pairs!=a$comparable_pairs)stop('Matched comparable-pair mismatch')
    each[[j]]<-data.frame(comparison=comparison,imputation=j,evaluation_joint(b,a))
    details[[j]]<-rbind(data.frame(comparison=comparison,imputation=j,side='before',b$folds),
      data.frame(comparison=comparison,imputation=j,side='after',a$folds))
  }
  each<-do.call(rbind,each);details<-do.call(rbind,details)
  defs<-list(C_before=c('C_before','variance_before'),C_after=c('C_after','variance_after'),
    difference_after_minus_before=c('difference_after_minus_before','variance_difference'))
  pooled<-do.call(rbind,lapply(names(defs),function(metric) {
    x<-defs[[metric]]
    data.frame(comparison=comparison,metric=metric,patients=each$patients[1],events=each$events[1],
      repeats=length(unique(after$repeat_id)),outer_folds=length(ordinals),imputations=20L,
      evaluation_pool(each[[x[1]]],each[[x[2]]],metric=='difference_after_minus_before'))
  }))
  for(n in names(metadata)) {
    each[[n]]<-metadata[[n]];pooled[[n]]<-metadata[[n]];details[[n]]<-metadata[[n]]
  }
  list(each=each,pooled=pooled,folds=details)
}

evaluation_population <- function(d,population) {
  d<-evaluation_validate(d,1:50,population)
  each<-folds<-vector('list',20L)
  for(j in 1:20) {
    fit<-evaluation_clustered_C(d[d$imputation==j,,drop=FALSE])
    each[[j]]<-data.frame(population=population,imputation=j,C=fit$C,
      variance=fit$variance,patients=fit$patients,events=fit$events,
      prediction_rows=fit$prediction_rows,comparable_pairs=fit$comparable_pairs)
    folds[[j]]<-data.frame(population=population,imputation=j,fit$folds)
  }
  each<-do.call(rbind,each)
  pooled<-data.frame(population=population,model='CACox_ES',patients=each$patients[1],
    events=each$events[1],repeats=10L,outer_folds=50L,imputations=20L,
    evaluation_pool(each$C,each$variance))
  list(each=each,pooled=pooled,folds=do.call(rbind,folds))
}

portable_options <- function(allowed,flags=character()) {
  argv<-commandArgs(trailingOnly=TRUE);opts<-list();k<-1L
  while(k<=length(argv)) {
    arg<-argv[k]
    if(!startsWith(arg,'--'))stop('Expected a --option, got: ',arg)
    if(grepl('=',arg,fixed=TRUE)) {
      bits<-strsplit(sub('^--','',arg),'=',fixed=TRUE)[[1]]
      name<-bits[1];value<-paste(bits[-1],collapse='=')
    } else {
      name<-sub('^--','',arg)
      if(name %in% flags)value<-TRUE else {
        k<-k+1L;if(k>length(argv))stop('Missing value for --',name)
        value<-argv[k]
      }
    }
    if(!name %in% c(allowed,flags))stop('Unknown option: --',name)
    if(!is.null(opts[[name]]))stop('Duplicate option: --',name)
    opts[[name]]<-value;k<-k+1L
  }
  opts
}
portable_option <- function(opts,name,default)if(is.null(opts[[name]]))default else opts[[name]]
portable_absolute <- function(path)grepl('^(/|[A-Za-z]:[/\\\\]|\\\\\\\\)',path)
portable_path <- function(path,root)if(portable_absolute(path))path else file.path(root,path)
portable_environment <- function(opts,script_root) {
  root<-normalizePath(portable_option(opts,'analysis-root',Sys.getenv('CACOX_PORTABLE_ROOT',script_root)),mustWork=TRUE)
  lib<-portable_path(Sys.getenv('CACOX_R_LIBS',file.path(root,'.Rlib')),root)
  if(dir.exists(lib)).libPaths(c(lib,.libPaths()))
  run<-portable_path(portable_option(opts,'run-root',Sys.getenv('CACOX_RUN_ROOT','run')),root)
  dir.create(run,recursive=TRUE,showWarnings=FALSE)
  run<-normalizePath(run,mustWork=TRUE)
  python<-Sys.getenv('CACOX_PYTHON','python')
  list(root=root,run=run,python=python)
}
portable_helper <- function(env,args,script='portable_hash.py') {
  helper<-file.path(env$root,'code',script)
  if(!file.exists(helper))stop('Required integrity helper is missing: ',helper)
  ans<-system2(env$python,args=vapply(c(helper,args),shQuote,character(1)),stdout=TRUE,stderr=TRUE)
  status<-attr(ans,'status');if(is.null(status))status<-0L
  if(status!=0L)stop('Integrity helper failed (exit ',status,'): ',paste(tail(ans,5L),collapse=' | '))
  invisible(ans)
}
portable_read_inputs <- function(env) {
  file<-file.path(env$root,'input','main_oof_cacox.csv.gz')
  sidecar<-paste0(file,'.sha256')
  if(!file.exists(sidecar))stop('Missing original-OOF SHA256 sidecar')
  expected<-trimws(readLines(sidecar,warn=FALSE))
  if(length(expected)!=1L||!grepl('^[0-9a-fA-F]{64}$',expected))stop('Invalid original-OOF SHA256')
  portable_helper(env,c('--verify-sha',file,'--expected',expected))
  con<-gzfile(file,'rt');on.exit(close(con),add=TRUE)
  main<-read.csv(con,check.names=FALSE)
  required<-c('repeat_id','fold','imputation','patient_id','Death','Survival_Days','score_E_selected')
  if(!identical(names(main),required))stop('Unexpected original-OOF columns')
  main$score<-main$score_E_selected;main$score_E_selected<-NULL
  main<-evaluation_validate(main,1:50,'original full-cohort OOF')
  raw<-readRDS(file.path(env$root,'input','raw_typed.rds'))
  cfg<-readRDS(file.path(env$root,'input','config.rds'))
  if(nrow(raw)!=1140L||sum(raw$Death)!=149L||anyDuplicated(raw$.patient_id)||
     length(cfg$model_vars)!=29L||nrow(main)!=228000L)stop('Unexpected original cohort or model variables')
  ri<-match(main$patient_id,raw$.patient_id)
  if(anyNA(ri)||!setequal(main$patient_id,raw$.patient_id)||any(main$Death!=raw$Death[ri])||
     any(main$Survival_Days!=raw$Survival_Days[ri]))stop('Original-OOF/raw membership or outcome mismatch')
  mask<-list(complete_case=complete.cases(raw[,cfg$model_vars,drop=FALSE]),
    egfr_observed=!is.na(raw$eGFR),renal_observed=!is.na(raw$Renal_Involvement),
    known_subtype=!is.na(raw$Type),
    no_egpa=!is.na(raw$Type)&as.character(raw$Type) %in% c('MPA','GPA'))
  counts<-do.call(rbind,lapply(names(mask),function(b)data.frame(branch=b,
    patients=sum(mask[[b]]),events=sum(raw$Death[mask[[b]]]))))
  if(!identical(as.integer(counts$patients),c(488L,756L,859L,1110L,1017L))||
     !identical(as.integer(counts$events),c(47L,102L,98L,141L,133L)))stop('Restricted-cohort counts mismatch')
  list(main=main,raw=raw,config=cfg,masks=mask,counts=counts,main_sha256=expected)
}

evaluation_methods <- function()c(
  'Only CACox ES and Harrell C-index are evaluated. No other performance metric is computed.',
  'Only within-fold comparable survival pairs are ranked; global C is the comparable-pair-weighted ratio.',
  'Global ratio influences are summed for each original patient across repeats before computing variance and paired covariance.',
  'Twenty paired imputation estimates use Rubin variance: mean within variance + 1.05*between variance, with Rubin t intervals.',
  'Complete-case fitted predictions have one real view and are duplicated over 20 labels only for pairing with original OOF predictions.',
  'Subgroup/full-cohort contrasts are descriptive case-mix contrasts, not tests of interaction.',
  'Inference is approximate and conditional on fitted models, selected procedures and frozen splits. It does not incorporate full model-development or imputation-model-fitting uncertainty.',
  'These internal subgroup and sensitivity comparisons are exploratory, unadjusted for multiplicity, and are not external validation.')
