# Fold-first Rubin summary for the frozen E_selected and RCS + ridge-Cox study.
# The overall test retains patient-clustered influences across repeated folds.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
project <- normalizePath(args[[1]],winslash='/',mustWork=TRUE)
output <- normalizePath(args[[2]],winslash='/',mustWork=FALSE)
source_dir <- file.path(project,'results/rcs_ridge_10x5/20261102_v1/final')
stopifnot(file.exists(file.path(source_dir,'COMPLETE.json')))
dir.create(output,recursive=TRUE,showWarnings=FALSE)
data <- read.csv(file.path(source_dir,'paired_scores.csv'))
keys <- c('repeat_id','fold','imputation','patient_id')
stopifnot(nrow(data)==228000L,!anyDuplicated(data[keys]),
          all(table(data$repeat_id,data$fold,data$imputation)==228L),
          all(is.finite(data$score_E_selected)),
          all(is.finite(data$score_rcs_ridge_cox)))

pool <- function(q,u) {
  stopifnot(length(q)==20L,length(u)==20L,all(is.finite(q)),
            all(is.finite(u)),all(u>=0))
  mean_q <- mean(q); within <- mean(u); between <- var(q)
  total <- within+1.05*between
  df <- if(between==0) Inf else 19*(1+within/(1.05*between))^2
  se <- sqrt(total)
  c(estimate=mean_q,SE=se,CI95_lower=mean_q-qt(.975,df)*se,
    CI95_upper=mean_q+qt(.975,df)*se,
    within_variance=within,between_variance=between,
    total_variance=total,df=df,
    p_Rubin=if(se>0)2*pt(-abs(mean_q/se),df) else NA_real_)
}

fit_one <- function(d,column) {
  d$score <- d[[column]]
  z <- survival::concordance(survival::Surv(Survival_Days,Death)~score,
                              data=d,reverse=TRUE,timewt='n',influence=2)
  counts <- as.numeric(z$count)
  den <- sum(counts[1:3]);stopifnot(den>0)
  C <- (counts[1]+counts[3]/2)/den
  m <- z$influence
  stopifnot(is.matrix(m),nrow(m)==228L,ncol(m)>=3L)
  local <- (m[,1]+.5*m[,3]-C*rowSums(m[,1:3,drop=FALSE]))/den
  list(C=C,pairs=den,variance=sum(local^2),counts=m)
}

mi_rows <- vector('list',1000L)
patients <- sort(unique(data$patient_id))
stopifnot(length(patients)==1140L)
global_q <- matrix(NA_real_,20L,3L,
                   dimnames=list(NULL,c('E_selected','rcs_ridge_cox','E_minus_RCS')))
global_u <- global_q
global_delta_inf <- matrix(NA_real_,1140L,20L)
expected_pairs <- matrix(NA_real_,10L,5L)
index <- 0L
for(j in 1:20) {
  dj <- data[data$imputation==j,,drop=FALSE]
  pair_total <- 0
  numerator <- c(E_selected=0,rcs_ridge_cox=0)
  raw <- list(E_selected=matrix(0,1140L,3L),
              rcs_ridge_cox=matrix(0,1140L,3L))
  for(r in 1:10) for(f in 1:5) {
    z <- dj[dj$repeat_id==r & dj$fold==f,,drop=FALSE]
    z <- z[order(z$patient_id),,drop=FALSE]
    stopifnot(nrow(z)==228L)
    e <- fit_one(z,'score_E_selected')
    b <- fit_one(z,'score_rcs_ridge_cox')
    stopifnot(e$pairs==b$pairs)
    if(j==1) expected_pairs[r,f] <- e$pairs
    else stopifnot(e$pairs==expected_pairs[r,f])
    ids <- match(z$patient_id,patients)
    stopifnot(!anyNA(ids))
    raw$E_selected[ids,] <- raw$E_selected[ids,]+e$counts[,1:3,drop=FALSE]
    raw$rcs_ridge_cox[ids,] <- raw$rcs_ridge_cox[ids,]+b$counts[,1:3,drop=FALSE]
    pair_total <- pair_total+e$pairs
    numerator['E_selected'] <- numerator['E_selected']+e$C*e$pairs
    numerator['rcs_ridge_cox'] <- numerator['rcs_ridge_cox']+b$C*b$pairs
    # Paired within-MI variance: score influences use the same 228 patients.
    local_e <- (e$counts[,1]+.5*e$counts[,3]-e$C*rowSums(e$counts[,1:3,drop=FALSE]))/e$pairs
    local_b <- (b$counts[,1]+.5*b$counts[,3]-b$C*rowSums(b$counts[,1:3,drop=FALSE]))/b$pairs
    index <- index+1L
    mi_rows[[index]] <- data.frame(repeat_id=r,fold=f,imputation=j,
      comparable_pairs=e$pairs,C_E_selected=e$C,C_rcs_ridge_cox=b$C,
      difference_E_minus_RCS=e$C-b$C,
      variance_E_selected=e$variance,variance_rcs_ridge_cox=b$variance,
      variance_difference=sum((local_e-local_b)^2))
  }
  global_q[j,1:2] <- numerator/pair_total
  global_q[j,3] <- global_q[j,1]-global_q[j,2]
  ginf <- lapply(names(raw),function(model) {
    m <- raw[[model]]
    (m[,1]+.5*m[,3]-global_q[j,model]*rowSums(m))/pair_total
  })
  names(ginf) <- names(raw)
  global_delta_inf[,j] <- ginf$E_selected-ginf$rcs_ridge_cox
  global_u[j,] <- c(sum(ginf$E_selected^2),sum(ginf$rcs_ridge_cox^2),
                    sum(global_delta_inf[,j]^2))
  cat(sprintf('MI %02d / 20 complete\n',j));flush.console()
}
mi <- do.call(rbind,mi_rows)
stopifnot(nrow(mi)==1000L)
saved <- read.csv(file.path(source_dir,'model_cindex.csv'))
for(model in c('E_selected','rcs_ridge_cox')) {
  x <- mi[,c('repeat_id','fold','imputation',paste0('C_',model))]
  names(x)[4] <- 'computed'
  y <- saved[saved$model==model,c('repeat_id','fold','imputation','C_index')]
  check <- merge(x,y,by=c('repeat_id','fold','imputation'))
  stopifnot(nrow(check)==1000L,
            max(abs(check$computed-check$C_index))<1e-12)
}

fold_rows <- vector('list',50L)
index <- 0L
for(r in 1:10) for(f in 1:5) {
  z <- mi[mi$repeat_id==r & mi$fold==f,,drop=FALSE]
  z <- z[order(z$imputation),,drop=FALSE]
  stopifnot(nrow(z)==20L,identical(z$imputation,1:20))
  e <- pool(z$C_E_selected,z$variance_E_selected)
  b <- pool(z$C_rcs_ridge_cox,z$variance_rcs_ridge_cox)
  d <- pool(z$difference_E_minus_RCS,z$variance_difference)
  index <- index+1L
  fold_rows[[index]] <- data.frame(repeat_id=r,fold=f,imputations=20L,
    comparable_pairs=z$comparable_pairs[1],
    C_E_selected=e['estimate'],E_CI95_lower=e['CI95_lower'],
    E_CI95_upper=e['CI95_upper'],
    C_rcs_ridge_cox=b['estimate'],RCS_CI95_lower=b['CI95_lower'],
    RCS_CI95_upper=b['CI95_upper'],
    difference_E_minus_RCS=d['estimate'],
    difference_CI95_lower=d['CI95_lower'],
    difference_CI95_upper=d['CI95_upper'],fold_Rubin_p=d['p_Rubin'])
}
folds <- do.call(rbind,fold_rows)
stopifnot(nrow(folds)==50L)
weights <- folds$comparable_pairs/sum(folds$comparable_pairs)
for(model in c('E_selected','rcs_ridge_cox')) {
  stopifnot(abs(sum(weights*folds[[paste0('C_',model)]])-
                mean(global_q[,model]))<1e-12)
}
summary <- lapply(colnames(global_q),function(model) {
  z <- as.data.frame(as.list(pool(global_q[,model],global_u[,model])))
  z$model <- model
  z
})
summary <- do.call(rbind,summary)
summary$p_conditional_no_Rubin <- NA_real_
se_cond <- sqrt(sum(rowMeans(global_delta_inf)^2))
summary$p_conditional_no_Rubin[summary$model=='E_minus_RCS'] <-
  2*pnorm(-abs(mean(global_q[,'E_minus_RCS'])/se_cond))
historic <- read.csv(file.path(source_dir,'cindex_and_pvalues.csv'))
for(model in c('E_selected','rcs_ridge_cox')) {
  z <- summary[summary$model==model,]
  h <- historic[historic$model==model,]
  stopifnot(abs(z$estimate-h$estimate)<1e-12,
            abs(z$SE-h$SE)<1e-12)
}
z <- summary[summary$model=='E_minus_RCS',]
h <- historic[historic$model=='rcs_ridge_cox_minus_E_selected',]
stopifnot(abs(z$estimate+h$estimate)<1e-12,
          abs(z$SE-h$SE)<1e-12,
          abs(z$p_Rubin-h$p_Rubin)<1e-12,
          abs(z$p_conditional_no_Rubin-h$p_conditional_normal_no_Rubin)<1e-12)
write.csv(mi,file.path(output,'fold_imputation_cindex.csv'),row.names=FALSE)
write.csv(folds,file.path(output,'fold_pooled_cindex.csv'),row.names=FALSE)
write.csv(summary,file.path(output,'overall_comparison.csv'),row.names=FALSE)
cat('COMPLETE: 50 fold-level Rubin summaries; historical overall inference verified\n')
