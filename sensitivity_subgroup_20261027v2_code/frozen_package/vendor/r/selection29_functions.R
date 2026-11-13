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

