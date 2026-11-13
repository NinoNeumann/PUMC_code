# Shared design, pooling and plotting helpers; ridge_functions.R supplies fitting diagnostics.
cox29_frame <- function(d, spec, model) {
  out <- data.frame(time = d$Survival_Days, event = d$Death)
  for (v in spec$vars) {
    if (v %in% spec$continuous) {
      if (model == "rcs_ridge_cox") {
        a <- rcs_basis(as.numeric(d[[v]]), spec$knots[[v]])
        colnames(a) <- c("linear", paste0("rcs", seq_len(ncol(a) - 1L)))
        out[[v]] <- I(a)
      } else out[[v]] <- as.numeric(d[[v]])
    } else out[[v]] <- factor(as.character(d[[v]]), levels = spec$levels[[v]])
  }
  out
}

cox29_x <- function(d, spec, model) {
  # Contrasts/reference rows need no actual outcomes.
  if (!"Survival_Days" %in% names(d)) d$Survival_Days <- rep(1, nrow(d))
  if (!"Death" %in% names(d)) d$Death <- rep(0, nrow(d))
  fd <- cox29_frame(d, spec, model)
  model.matrix(reformulate(spec$vars), data = fd)[, -1L, drop = FALSE]
}

cox29_pool_scalar <- function(Q, U) {
  stopifnot(length(Q) == 20L, length(U) == 20L, all(is.finite(Q)), all(is.finite(U)), all(U >= 0))
  z <- mice::pool.scalar(Q, U, n = Inf)
  se <- sqrt(z$t); df <- z$df
  if (!is.finite(df) && !is.infinite(df)) df <- Inf
  cut <- qt(.975, df = df)
  data.frame(estimate = z$qbar, SE = se, lower = z$qbar - cut * se,
    upper = z$qbar + cut * se, df = df, within_variance = z$ubar,
    between_variance = z$b, total_variance = z$t,
    p_value = if (se > 0) 2 * pt(-abs(z$qbar / se), df) else NA_real_)
}

cox29_pool_metrics <- function(tab) {
  defs <- list(C_cox = c("C_cox", "variance_cox"), C_rcs_ridge = c("C_rcs_ridge", "variance_rcs_ridge"),
    difference_rcs_ridge_minus_cox = c("difference_rcs_ridge_minus_cox", "variance_difference"))
  out <- do.call(rbind, lapply(names(defs), function(n) {
    z <- cox29_pool_scalar(tab[[defs[[n]][1]]], tab[[defs[[n]][2]]])
    if (n != "difference_rcs_ridge_minus_cox") {
      z$lower <- max(0, z$lower); z$upper <- min(1, z$upper); z$p_value <- NA_real_
    }
    data.frame(metric = n, z, interval_scope = "Approximate Rubin + paired patient-clustered IJ; conditional on selected fitted models and splits")
  }))
  rownames(out) <- NULL
  out
}

cox29_pool_HR <- function(tab) {
  keys <- unique(tab[c("model", "variable", "contrast", "reference_value", "comparison_value")])
  do.call(rbind, lapply(seq_len(nrow(keys)), function(i) {
    z <- tab[tab$model == keys$model[i] & tab$contrast == keys$contrast[i], ]
    p <- cox29_pool_scalar(z$logHR, z$variance_logHR)
    data.frame(keys[i, ], logHR = p$estimate, SE_logHR = p$SE,
      HR = exp(p$estimate), CI_lower = exp(p$lower), CI_upper = exp(p$upper),
      df = p$df, within_variance = p$within_variance, between_variance = p$between_variance,
      interval_scope = "Rubin pooled contrast, conditional on selected variables; not selection-adjusted or causal")
  }))
}

cox29_forest <- function(tab, out, title) {
  a <- tab[tab$model == "ordinary_cox", ]; b <- tab[tab$model == "rcs_ridge_cox", ]
  stopifnot(identical(a$contrast, b$contrast))
  draw <- function() {
    n <- nrow(a); y <- rev(seq_len(n)); yr <- c(.3, n + .8)
    cols <- c("#176A73", "#B76B24")
    layout(matrix(1:3, 1, 3), widths = c(4.6, 4, 3.1))
    par(oma = c(4.4, 0, 4, 0), family = "sans")
    par(mar = c(3.2, .8, 1.5, .3)); plot.new(); plot.window(c(0, 1), yr)
    text(0, n + .65, "Predictor and contrast", adj = 0, font = 2, cex = .9)
    text(0, y, a$contrast, adj = 0, cex = .78)
    par(mar = c(3.2, .2, 1.5, .2))
    lim <- range(c(a$CI_lower, a$CI_upper, b$CI_lower, b$CI_upper, 1))
    stopifnot(all(is.finite(lim)), all(lim > 0))
    lim <- exp(log(lim) + c(-.12, .12))
    plot(NA, xlim = lim, ylim = yr, log = "x", axes = FALSE, xlab = "", ylab = "")
    axis(1, cex.axis = .8); abline(v = 1, lty = 2, col = "#87939D"); abline(h = y, col = "#E9EFF2")
    segments(a$CI_lower, y + .14, a$CI_upper, y + .14, col = cols[1], lwd = 1.7)
    points(a$HR, y + .14, pch = 16, col = cols[1])
    segments(b$CI_lower, y - .14, b$CI_upper, y - .14, col = cols[2], lwd = 1.7)
    points(b$HR, y - .14, pch = 17, col = cols[2])
    par(mar = c(3.2, .2, 1.5, .2)); plot.new(); plot.window(c(0, 1), yr)
    text(.02, n + .65, "HR (95% pooled CI)", adj = 0, font = 2, cex = .9)
    text(.02, y + .14, sprintf("%.2f (%.2f, %.2f)", a$HR, a$CI_lower, a$CI_upper), adj = 0, col = cols[1], cex = .8)
    text(.02, y - .14, sprintf("%.2f (%.2f, %.2f)", b$HR, b$CI_lower, b$CI_upper), adj = 0, col = cols[2], cex = .8)
    mtext(title, side = 3, outer = TRUE, line = 2.1, font = 2, cex = 1.2)
    mtext("Ordinary Cox: teal circles     RCS-Ridge-Cox: orange triangles | 20 imputations, Rubin pooling", side = 3, outer = TRUE, line = .6, cex = .9)
    mtext("Continuous contrasts: observed development Q75 vs Q25. RCS contrasts are not constant per-unit HRs.", side = 1, outer = TRUE, line = 1.4, cex = .8)
    mtext("Approximate conditional intervals; selection/tuning uncertainty and non-proportional hazards are not corrected.", side = 1, outer = TRUE, line = 2.8, cex = .8)
  }
  h <- max(8, .48 * nrow(a) + 3.5)
  png(file.path(out, "forest_pooled_two_cox.png"), width = 3000, height = round(h * 180), res = 180)
  draw(); dev.off()
  pdf(file.path(out, "forest_pooled_two_cox.pdf"), width = 16.67, height = h)
  draw(); dev.off()
}

