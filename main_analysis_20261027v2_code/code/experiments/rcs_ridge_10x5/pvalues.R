# Patient-clustered, within-MI inference across the same 50 outer folds.
# Rubin pooling follows the upstream ridge_functions.R convention.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
project <- normalizePath(args[[1]],winslash='/',mustWork=TRUE)
out <- normalizePath(args[[2]],winslash='/',mustWork=TRUE)
source(file.path(project,
  'results/cacox_vs_rcs_selected29_optimization_20261028_v1/upstream_rcs_ridge_code/ridge_functions.R'))
data <- read.csv(file.path(out,'final/paired_scores.csv'))
stopifnot(nrow(data)==228000L,
          !anyDuplicated(data[c('repeat_id','fold','imputation','patient_id')]),
          all(table(data$repeat_id,data$fold,data$imputation)==228L))
models <- c('ordinary_cox','E_selected','rcs_ridge_cox')
pairs <- list(c('rcs_ridge_cox','ordinary_cox'),
              c('rcs_ridge_cox','E_selected'))

one <- function(d,column) {
  d$score <- d[[column]]
  parts <- split(d,interaction(d$repeat_id,d$fold,drop=TRUE))
  stopifnot(length(parts)==50L,all(vapply(parts,nrow,integer(1))==228L))
  fits <- lapply(parts,function(z)survival::concordance(
    survival::Surv(Survival_Days,Death)~score,data=z,reverse=TRUE,
    timewt='n',influence=2))
  counts <- Reduce('+',lapply(fits,function(f)as.numeric(f$count)))
  den <- sum(counts[1:3]);stopifnot(den>0)
  value <- (counts[1]+counts[3]/2)/den
  ids <- sort(unique(d$patient_id));stopifnot(length(ids)==1140L)
  inf <- setNames(numeric(length(ids)),as.character(ids))
  for(k in seq_along(parts)) {
    z <- parts[[k]];m <- fits[[k]]$influence
    stopifnot(is.matrix(m),nrow(m)==228L,ncol(m)>=3L)
    v <- (m[,1]+.5*m[,3]-value*rowSums(m[,1:3,drop=FALSE]))/den
    names(v) <- as.character(z$patient_id)
    inf[names(v)] <- inf[names(v)]+v
  }
  list(C=value,influence=inf)
}

pool <- function(Q,U) {
  stopifnot(length(Q)==20L,length(U)==20L,all(is.finite(Q)),all(is.finite(U)),all(U>=0))
  qbar <- mean(Q);within <- mean(U);between <- var(Q)
  total <- within+1.05*between
  df <- if(between==0) Inf else 19*(1+within/(1.05*between))^2
  se <- sqrt(total)
  data.frame(estimate=qbar,SE=se,CI95_lower=qbar-qt(.975,df)*se,
    CI95_upper=qbar+qt(.975,df)*se,within_variance=within,
    between_variance=between,total_variance=total,df=df,
    p_Rubin=if(se>0)2*pt(-abs(qbar/se),df) else NA_real_)
}

estimates <- matrix(NA_real_,20L,length(models),dimnames=list(NULL,models))
variances <- estimates
influences <- array(NA_real_,c(1140L,20L,length(models)),
  dimnames=list(NULL,NULL,models))
for(j in 1:20) {
  d <- data[data$imputation==j,,drop=FALSE]
  stopifnot(nrow(d)==11400L)
  for(model in models) {
    z <- one(d,paste0('score_',model))
    estimates[j,model] <- z$C
    variances[j,model] <- sum(z$influence^2)
    influences[,j,model] <- z$influence
  }
  cat(sprintf('MI=%02d Cox=%.7f E_selected=%.7f RCS=%.7f\n',j,
    estimates[j,'ordinary_cox'],estimates[j,'E_selected'],estimates[j,'rcs_ridge_cox']))
  flush.console()
}
rows <- list()
for(model in models) {
  z <- pool(estimates[,model],variances[,model]);z$model <- model
  z$conditional_SE_no_Rubin <- NA_real_
  z$conditional_CI95_lower_no_Rubin <- NA_real_
  z$conditional_CI95_upper_no_Rubin <- NA_real_
  z$p_conditional_normal_no_Rubin <- NA_real_
  rows[[length(rows)+1L]] <- z
}
for(pair in pairs) {
  left <- pair[[1]];right <- pair[[2]]
  delta <- influences[,,left]-influences[,,right]
  z <- pool(estimates[,left]-estimates[,right],colSums(delta^2))
  z$model <- paste0(left,'_minus_',right)
  se <- sqrt(sum(rowMeans(delta)^2))
  z$conditional_SE_no_Rubin <- se
  z$conditional_CI95_lower_no_Rubin <- z$estimate-qnorm(.975)*se
  z$conditional_CI95_upper_no_Rubin <- z$estimate+qnorm(.975)*se
  z$p_conditional_normal_no_Rubin <- 2*pnorm(-abs(z$estimate/se))
  rows[[length(rows)+1L]] <- z
}
result <- do.call(rbind,rows)
write.csv(result,file.path(out,'final/cindex_and_pvalues.csv'),row.names=FALSE)
cat('COMPLETE 50 folds, 20 imputations, 1000 models per method\n')
