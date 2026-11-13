matrix29 <- function(d, cfg) {
  f <- reformulate(cfg$model_vars)
  mm <- model.matrix(f, data = d)
  group <- attr(terms(f), "term.labels")[attr(mm, "assign")[-1L]]
  x <- mm[, -1L, drop = FALSE]
  stopifnot(!anyNA(x), all(is.finite(x)), setequal(group, cfg$model_vars))
  list(x = x, group = group)
}

# Lightweight classical between/within-chain diagnostic, without rstan.
# This is a screen for trends in MICE chain summaries, not proof of valid MI.
diagnose29_mice <- function(mids, directory) {
  rows <- list()
  for (parameter in c("mean", "variance")) {
    a <- if (parameter == "mean") mids$chainMean else mids$chainVar
    for (v in names(mids$method)[nzchar(mids$method)]) {
      z <- a[v, , , drop = TRUE]
      z <- z[(floor(nrow(z) / 2) + 1L):nrow(z), , drop = FALSE]
      n <- nrow(z); W <- mean(apply(z, 2, var)); B <- n * var(colMeans(z))
      rhat <- if (!is.finite(W) || W <= 0) NA_real_ else sqrt(((n - 1) / n * W + B / n) / W)
      rows[[length(rows) + 1L]] <- data.frame(variable = v, parameter = parameter,
        iterations_used = n, chains = ncol(z), classical_Rhat = rhat,
        within_chain_variance = W, between_chain_variance = B)
    }
  }
  tab <- do.call(rbind, rows)
  write_csv(tab, file.path(directory, "mice_chain_diagnostic.csv"))
  continuous <- intersect(c("eGFR", "Hb", "PLT", "CRP", "ESR", "Age"), dimnames(mids$chainMean)[[1]])
  continuous <- continuous[vapply(continuous, function(v) any(is.finite(mids$chainMean[v,,])), logical(1))]
  if (!length(continuous)) return(invisible(tab))
  png(file.path(directory, "continuous_imputation_trace.png"), width = 2200, height = 1500, res = 180)
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
  for (v in continuous) {
    matplot(mids$chainMean[v, , ], type = "l", lty = 1,
      col = adjustcolor(rep(c("#176A73", "#A16D2F", "#677CAA", "#8E779C"), length.out = mids$m), .55),
      xlab = "MICE iteration", ylab = "Mean of imputed values", main = v)
  }
  dev.off()
  invisible(tab)
}
complete29 <- function(bundle, j) {
  z <- mice::complete(bundle$mids, j)
  nd <- nrow(bundle$development)
  d <- bundle$development; a <- bundle$assessment
  for (v in names(z)) {
    d[[v]] <- z[seq_len(nd), v]
    a[[v]] <- z[nd + seq_len(nrow(a)), v]
    observed <- !is.na(bundle$mids$data[[v]])
    stopifnot(identical(as.character(z[[v]][observed]), as.character(bundle$mids$data[[v]][observed])))
  }
  stopifnot(!anyNA(z))
  list(dev = d, assessment = a)
}

# Joint permutation of a categorical variable's dummy columns preserves legal
# category combinations. Each clinical variable has one real importance and one
# shadow importance, irrespective of the number of design columns.
boruta29 <- function(d, cfg, j, directory) {
  mm <- matrix29(d, cfg); x <- mm$x; group <- mm$group
  vars <- cfg$model_vars; p <- length(vars)
  decision <- hits <- integer(p)
  history <- matrix(NA_real_, cfg$rounds, p, dimnames = list(NULL, vars))
  thresholds <- numeric(cfg$rounds)
  cindex <- function(score) survival::concordance(
    survival::Surv(d$Survival_Days, d$Death) ~ score, reverse = TRUE)$concordance
  actual <- 0L
  for (iter in seq_len(cfg$rounds)) {
    active <- which(decision >= 0L)
    if (!any(decision == 0L)) break
    active_names <- vars[active]
    real_cols <- which(group %in% active_names)
    real <- x[, real_cols, drop = FALSE]; real_group <- group[real_cols]
    blocks <- lapply(active_names, function(v) which(real_group == v))
    shadow <- real
    seed <- cfg$seed + 200000L + j * 100L + iter
    set.seed(seed)
    for (cols in blocks) shadow[, cols] <- real[sample.int(nrow(real)), cols, drop = FALSE]
    joined <- cbind(real, shadow)
    colnames(joined) <- paste0("X", seq_len(ncol(joined)))
    all_blocks <- c(blocks, lapply(blocks, function(cols) cols + ncol(real)))
    # As in the old BorutaPy-style auto tree rule, p is the active clinical count.
    ntree <- max(1L, as.integer(floor(100 * sqrt(2 * length(active)) / cfg$depth)))
    fd <- data.frame(time = d$Survival_Days, event = d$Death, joined)
    rf <- ranger::ranger(survival::Surv(time, event) ~ ., data = fd,
      num.trees = ntree, max.depth = cfg$depth, min.node.size = cfg$minleaf,
      splitrule = "logrank", num.threads = 1L, seed = seed, write.forest = TRUE)
    risk <- make_leaf_risk(rf, d, j)
    baseline <- cindex(risk(joined))
    set.seed(seed + 1000000L)
    importance <- vapply(all_blocks, function(cols) {
      # Batch the five prediction calls for this block; the permutation
      # importance definition is unchanged and memory stays bounded.
      batches <- lapply(seq_len(cfg$permutation_repeats), function(repetition) {
        permuted <- joined
        permuted[, cols] <- joined[sample.int(nrow(joined)), cols, drop = FALSE]
        permuted
      })
      permuted_scores <- matrix(risk(do.call(rbind, batches)), nrow = nrow(joined))
      mean(baseline - apply(permuted_scores, 2, cindex))
    }, numeric(1))
    real_imp <- importance[seq_along(active)]
    threshold <- unname(quantile(importance[-seq_along(active)], cfg$perc / 100, type = 7))
    history[iter, active] <- real_imp; thresholds[iter] <- threshold
    hits[active] <- hits[active] + as.integer(real_imp > threshold)
    if (iter > 4L) {
      accept_p <- pbinom(hits[active] - 1L, iter, .5, lower.tail = FALSE)
      reject_p <- pbinom(hits[active], iter, .5)
      accept <- p.adjust(accept_p, "BH") <= cfg$alpha & accept_p <= cfg$alpha / iter
      reject <- p.adjust(reject_p, "BH") <= cfg$alpha & reject_p <= cfg$alpha / iter
      decision[active[decision[active] == 0L & accept]] <- 1L
      decision[active[decision[active] == 0L & reject]] <- -1L
    }
    actual <- iter
    cat(sprintf("imp %02d round %02d: confirmed=%d unresolved=%d rejected=%d\n",
      j, iter, sum(decision == 1L), sum(decision == 0L), sum(decision == -1L)))
    flush.console()
  }
  history <- history[seq_len(actual), , drop = FALSE]; thresholds <- thresholds[seq_len(actual)]
  med <- apply(history, 2, median, na.rm = TRUE)
  status <- ifelse(decision == 1L, "Confirmed", ifelse(decision == -1L, "Rejected",
    ifelse(med > median(thresholds), "Tentative", "Undecided")))
  list(table = data.frame(variable = vars, boruta_decision = status,
    boruta_non_rejected = status != "Rejected", hits = hits,
    median_importance = med, rounds = actual), history = history, thresholds = thresholds,
    design_column_mapping = data.frame(column = colnames(x), variable = group))
}

plot29_boruta <- function(boruta, j, directory) {
  png(file.path(directory, "boruta_results.png"), width = 2500, height = 2300, res = 180)
  par(mar = c(5, 18, 4, 1))
  ord <- order(boruta$table$median_importance)
  cols <- c(Confirmed = "#398D6D", Tentative = "#D5A740", Undecided = "#A7ADB3", Rejected = "#C87979")
  boxplot(lapply(ord, function(k) boruta$history[, k]), horizontal = TRUE,
    names = boruta$table$variable[ord], las = 1, outline = FALSE, cex.axis = .75,
    col = cols[boruta$table$boruta_decision[ord]],
    xlab = "Grouped permutation importance (development Harrell C decrease)",
    main = sprintf("Survival Boruta: 29 candidates, imputation %02d", j))
  abline(v = median(boruta$thresholds), lty = 2, col = "#346A9E")
  legend("bottomright", legend = c(names(cols), "Median of round-specific shadow thresholds"),
    pch = c(rep(15, 4), NA), lty = c(rep(NA, 4), 2),
    col = c(cols, "#346A9E"), bty = "n", cex = .75)
  dev.off()
}

lasso29 <- function(d, cfg, j) {
  # Common absolute lambda grid is prespecified, avoiding lambda-path selection
  # from inner-validation outcomes. Each inner fold has its own MICE fit.
  C <- matrix(NA_real_, 5L, length(cfg$lambda))
  for (k in 1:5) {
    inner <- readRDS(file.path(cfg$output_dir, sprintf("inner_%02d/imputation.rds", k)))
    stopifnot(inner$mids$iteration == cfg$maxit, inner$mids$m == cfg$m)
    ds <- complete29(inner, j)
    tr <- matrix29(ds$dev, cfg)$x; va <- matrix29(ds$assessment, cfg)$x
    stopifnot(identical(colnames(tr), colnames(va)))
    fit <- glmnet::glmnet(tr, survival::Surv(ds$dev$Survival_Days, ds$dev$Death),
      family = "cox", alpha = 1, lambda = cfg$lambda, standardize = TRUE,
      cox.ties = "efron", control = list(maxit = 100000L))
    stopifnot(fit$jerr == 0L, length(fit$lambda) == length(cfg$lambda))
    pr <- predict(fit, newx = va, s = cfg$lambda, type = "link")
    C[k, ] <- apply(pr, 2, function(score) survival::concordance(
      survival::Surv(ds$assessment$Survival_Days, ds$assessment$Death) ~ score,
      reverse = TRUE, timewt = "n")$concordance)
  }
  mean_C <- colMeans(C); se_C <- apply(C, 2, sd) / sqrt(5)
  best <- which.max(mean_C)
  # Descending lambda: the first eligible index is the largest penalty.
  one_se <- which(mean_C >= mean_C[best] - se_C[best])[1L]
  mm <- matrix29(d, cfg)
  fit <- glmnet::glmnet(mm$x, survival::Surv(d$Survival_Days, d$Death),
    family = "cox", alpha = 1, lambda = cfg$lambda, standardize = TRUE,
    cox.ties = "efron", control = list(maxit = 100000L))
  stopifnot(fit$jerr == 0L)
  b1 <- as.numeric(coef(fit, s = cfg$lambda[one_se]))
  bm <- as.numeric(coef(fit, s = cfg$lambda[best]))
  table <- data.frame(variable = cfg$model_vars,
    lasso_1se = vapply(cfg$model_vars, function(v) any(abs(b1[mm$group == v]) > 1e-10), logical(1)),
    lasso_min = vapply(cfg$model_vars, function(v) any(abs(bm[mm$group == v]) > 1e-10), logical(1)))
  list(table = table, coefficients = data.frame(column = colnames(mm$x),
      variable = mm$group, beta_1se = b1, beta_min = bm),
    cv = data.frame(lambda = cfg$lambda, mean_C = mean_C, SE_C = se_C, t(C)),
    lambda_1se = cfg$lambda[one_se], lambda_min = cfg$lambda[best],
    best_grid_endpoint = best %in% c(1L, length(cfg$lambda)),
    inner_C_at_1se = mean_C[one_se], inner_C_at_min = mean_C[best], fit = fit)
}
