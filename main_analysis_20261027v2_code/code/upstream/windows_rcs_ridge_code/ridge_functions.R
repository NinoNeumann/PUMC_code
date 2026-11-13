# Ridge on all nonconstant expanded columns, including spline linear terms
# and dummy columns. Penalty: events * lambda / 2 * sum(var(X_j) * beta_j^2).
# survival::ridge(scale=TRUE) keeps coefficients in original design units.
ridge_fit <- function(d, spec, lambda, context=list()) {
  x <- cox29_x(d, spec, 'rcs_ridge_cox')
  stopifnot(all(is.finite(x)), lambda > 0, sum(d$Death) > 0,
            all(d$Survival_Days > 0))
  variances <- apply(x, 2, var)
  active <- is.finite(variances) & variances > 0
  if (!any(active)) stop('No varying design columns')
  fd <- data.frame(time=d$Survival_Days, event=d$Death)
  fd$X <- I(x[, active, drop=FALSE])
  theta <- sum(d$Death) * lambda
  ridge <- survival::ridge
  # Fix theta as a literal so saved fits do not depend on a mutable tuning value.
  formula <- as.formula(sprintf('survival::Surv(time,event) ~ ridge(X, theta=%.17g, scale=TRUE)', theta))
  environment(formula) <- asNamespace('survival')
  run <- function(eps, iter) {
    warns <- character()
    f <- tryCatch(withCallingHandlers(survival::coxph(formula, data=fd,
      ties='efron', x=TRUE, y=TRUE, model=TRUE, singular.ok=FALSE,
      control=survival::coxph.control(eps=eps, iter.max=iter)),
      warning=function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart('muffleWarning') }), error=identity)
    good <- !inherits(f,'error') && all(is.finite(coef(f))) &&
      all(is.finite(vcov(f))) && all(diag(vcov(f)) >= 0) &&
      !any(grepl('infinite|converg|ran out', warns, ignore.case=TRUE))
    list(fit=if(!inherits(f,'error')) f else NULL, good=good, warnings=warns,
         error=if(inherits(f,'error')) conditionMessage(f) else '')
  }
  attempts <- list(original=run(1e-9,100L)); accepted <- attempts$original
  numerical <- 'original_accepted'
  if (!accepted$good) {
    attempts$tighter <- run(1e-11,300L); attempts$confirmation <- run(1e-12,300L)
    a <- attempts$tighter; b <- attempts$confirmation
    stable <- a$good && b$good
    if (stable) {
      delta <- coef(a$fit)-coef(b$fit)
      stable <- max(abs(delta)/pmax(1,sqrt(diag(vcov(b$fit))))) < 1e-6 &&
        max(abs(as.numeric(x[,active,drop=FALSE] %*% delta))) < 1e-5 &&
        abs(tail(a$fit$loglik,1)-tail(b$fit$loglik,1)) < 1e-7 &&
        max(abs(vcov(a$fit)-vcov(b$fit))/pmax(1,abs(vcov(b$fit)))) < 1e-5
    }
    if (!stable) stop(paste('Persistent ridge failure:', paste(unique(unlist(
      lapply(attempts,function(z)c(z$error,z$warnings)))),collapse=' | ')))
    accepted <- b; numerical <- 'numerical_retry_accepted'
  }
  fit <- accepted$fit
  stopifnot(ncol(fit$x)==sum(active), max(abs(unname(fit$x)-unname(x[,active,drop=FALSE]))) < 1e-8)
  beta <- setNames(numeric(ncol(x)),colnames(x)); beta[active] <- coef(fit)
  V <- matrix(0,ncol(x),ncol(x),dimnames=list(colnames(x),colnames(x)))
  V[active,active] <- vcov(fit)
  list(fit=fit, spec=spec, model='rcs_ridge_cox', beta=beta, variance=V,
       active=active, zero_variance_columns=colnames(x)[!active],
       lambda=lambda, theta=theta, numerical_status=numerical, fit_context=context,
       audit=lapply(attempts,function(z)z[c('good','warnings','error')]))
}

cox29_predict <- function(obj,d,times=c(365,730,1095)) {
  x <- cox29_x(d,obj$spec,obj$model)
  beta <- if(obj$model=='rcs_ridge_cox') obj$beta else coef(obj$fit)
  stopifnot(identical(colnames(x),names(beta)))
  lp <- as.numeric(x %*% beta)
  bh <- survival::basehaz(obj$fit,centered=TRUE)
  ch <- c(0,bh$hazard)[findInterval(times,bh$time)+1L]
  center <- sum(obj$fit$means * coef(obj$fit))
  S <- exp(-outer(exp(lp-center),ch))
  stopifnot(all(is.finite(lp)),all(is.finite(S)),all(S>=0 & S<=1))
  colnames(S) <- paste0('survival_',times,'d')
  list(lp=lp,survival=S)
}

ridge_tune <- function(source_fold, spec_cfg, out) {
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  rows <- list()
  for(k in 1:5) {
    bundle <- readRDS(file.path(source_fold,sprintf('inner_%02d/imputation.rds',k)))
    # Knots from this inner training partition only, not outer validation.
    spec <- make_spec(bundle$development,spec_cfg)
    for(j in 1:20) {
      checkpoint <- file.path(out,sprintf('inner%02d_imp%02d.rds',k,j))
      if(file.exists(checkpoint)) z <- readRDS(checkpoint) else {
        ds <- complete29(bundle,j)
        z <- do.call(rbind,lapply(ridge_grid,function(lambda) {
          a <- tryCatch({
            obj <- ridge_fit(ds$dev,spec,lambda)
            x <- cox29_x(ds$assessment,spec,'rcs_ridge_cox')
            score <- as.numeric(x %*% obj$beta)
            cc <- survival::concordance(survival::Surv(ds$assessment$Survival_Days,
              ds$assessment$Death) ~ score,reverse=TRUE,timewt='n')$concordance
            if(!is.finite(cc)) stop('Nonfinite inner C-index')
            list(C=cc,error='')
          },error=function(e)list(C=NA_real_,error=conditionMessage(e)))
          data.frame(inner_fold=k,imputation=j,lambda=lambda,C_index=a$C,error=a$error)
        }))
        cox29_atomic_rds(z,checkpoint)
      }
      rows[[length(rows)+1L]] <- z
      cat('Ridge tuning inner',k,'imputation',j,'complete\n');flush.console()
    }
  }
  tab <- do.call(rbind,rows)
  sums <- do.call(rbind,lapply(ridge_grid,function(l) {
    z <- tab[tab$lambda==l,]; valid <- nrow(z)==100L && all(is.finite(z$C_index))
    data.frame(lambda=l,valid_cells=sum(is.finite(z$C_index)),eligible=valid,
               mean_inner_C=if(valid)mean(z$C_index) else NA_real_)
  }))
  write_csv(tab,file.path(out,'ridge_inner_C_all.csv'))
  write_csv(sums,file.path(out,'ridge_lambda_summary.csv'))
  if(!any(sums$eligible))stop('No lambda valid in all 5 x 20 inner fits; see tuning errors')
  best <- max(sums$mean_inner_C,na.rm=TRUE)
  lambda <- max(sums$lambda[sums$eligible & !is.na(sums$mean_inner_C) & sums$mean_inner_C >= best-1e-12])
  result <- list(lambda=lambda,grid=ridge_grid,mean_inner_C=best,
    boundary=lambda %in% range(ridge_grid),
    method='Maximum mean Harrell C over 5 inner folds x 20 MI, ties choose larger lambda; selected outer-development variables held fixed')
  cox29_atomic_rds(result,file.path(out,'selected_lambda.rds'))
  write_csv(data.frame(lambda=lambda,mean_inner_C=best,boundary=result$boundary),file.path(out,'selected_lambda.csv'))
  if(result$boundary)writeLines('Selected lambda is on fixed grid boundary; review, no automatic grid expansion.',file.path(out,'BOUNDARY_WARNING.txt'))
  result
}

# Use the same contrasts for both models. Ridge intervals are approximate and
# conditional on the selected variables and tuned penalty (not unbiased effects).
cox29_HR <- function(obj,j) {
  spec <- obj$spec
  b <- if(obj$model=='rcs_ridge_cox')obj$beta else coef(obj$fit)
  V <- if(obj$model=='rcs_ridge_cox')obj$variance else vcov(obj$fit)
  rows <- list()
  for(v in spec$vars) {
    contrasts <- if(v %in% spec$continuous)list(spec$iqrs[[v]]) else
      lapply(spec$levels[[v]][-1L],function(x)c(spec$levels[[v]][1L],x))
    for(ct in contrasts) {
      lo<-hi<-spec$reference;lo[[v]]<-ct[1];hi[[v]]<-ct[2]
      dx<-as.numeric(cox29_x(hi,spec,obj$model)-cox29_x(lo,spec,obj$model))
      if(obj$model=='rcs_ridge_cox' && any(abs(dx[!obj$active])>0))
        stop('HR contrast not estimable: zero-variance training columns for ',v)
      desc<-if(v %in% spec$continuous)sprintf('%s: %.3g vs %.3g',v,as.numeric(ct[2]),as.numeric(ct[1])) else
        if(v=='Gender')'Gender: female vs male' else paste0(v,': ',ct[2],' vs ',ct[1])
      rows[[length(rows)+1L]]<-data.frame(imputation=j,model=obj$model,variable=v,contrast=desc,
        reference_value=as.character(ct[1]),comparison_value=as.character(ct[2]),
        logHR=sum(dx*b),variance_logHR=as.numeric(t(dx)%*%V%*%dx))
    }
  }
  do.call(rbind,rows)
}

cox29_PH <- function(obj,j) {
  penalised <- obj$model=='rcs_ridge_cox'
  # The ridge matrix is one penalty term. terms=FALSE misaligns effective
  # degrees of freedom with expanded columns in survival::cox.zph.
  z <- tryCatch(survival::cox.zph(obj$fit,transform='km',terms=TRUE,global=TRUE),error=identity)
  note <- if(penalised)'Exploratory joint penalised-predictor-block and GLOBAL PH diagnostics; fixed tuned penalty; no individual-variable PH inference' else ''
  if(inherits(z,'error'))return(data.frame(imputation=j,model=obj$model,variable='ERROR',chisq=NA,df=NA,p_value=NA,note=conditionMessage(z)))
  nm<-rownames(z$table)
  if(penalised)nm[nm!='GLOBAL']<-'ALL_RIDGE_PREDICTORS_JOINT'
  data.frame(imputation=j,model=obj$model,variable=nm,chisq=z$table[,'chisq'],
    df=z$table[,'df'],p_value=z$table[,'p'],note=note)
}

# Dimension-safe stratified C-index: calculate per-stratum patient influences,
# then aggregate by patient. Adds between-stratum denominator contributions;
# not an independence assumption for repeated observations of one patient.
cox29_paired_C <- function(d) {
  one <- function(score) {
    dd <- d;dd$score <- score
    parts <- split(dd,interaction(dd$repeat_id,dd$fold,drop=TRUE))
    fits <- lapply(parts,function(z)survival::concordance(
      survival::Surv(Survival_Days,Death)~score,data=z,reverse=TRUE,
      timewt='n',influence=2))
    count <- Reduce('+',lapply(fits,function(f)as.numeric(f$count)))
    den <- sum(count[1:3]);if(den<=0)stop('No comparable pairs')
    C <- (count[1]+count[3]/2)/den
    # influence=2 provides per-row contributions to pair counts. For reverse
    # scoring the package has already oriented concordant/discordant counts.
    ids<-sort(unique(dd$patient_id));inf<-setNames(numeric(length(ids)),ids)
    for(k in seq_along(parts)) {
      z<-parts[[k]];m<-fits[[k]]$influence
      stopifnot(is.matrix(m),nrow(m)==nrow(z),ncol(m)>=3)
      v<-(m[,1]+.5*m[,3]-C*rowSums(m[,1:3,drop=FALSE]))/den
      inf[as.character(z$patient_id)]<-inf[as.character(z$patient_id)]+v
    }
    list(C=C,influence=inf)
  }
  a<-one(d$score_cox);b<-one(d$score_rcs_ridge)
  data.frame(C_cox=a$C,C_rcs_ridge=b$C,difference_rcs_ridge_minus_cox=b$C-a$C,
    variance_cox=sum(a$influence^2),variance_rcs_ridge=sum(b$influence^2),
    covariance_cox_rcs_ridge=sum(a$influence*b$influence),
    variance_difference=sum((b$influence-a$influence)^2),
    patients=length(unique(d$patient_id)),prediction_rows=nrow(d))
}
