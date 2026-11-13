# Restricted-cohort CACox data preparation and variable selection only.
# All tuning and variable selection are repeated inside each development fold.
# The original 29 candidates, grouped Boruta settings, LASSO grid and >=10/20
# union vote are preserved; non-estimable constants are recorded explicitly.

sensitivity_write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile(".writing_", tmpdir = dirname(path))
  on.exit(unlink(tmp), add = TRUE)
  write_csv(x, tmp)
  portable_commit(tmp, path)
}

sensitivity_fold_cfg <- function(branch, ordinal) {
  if (!exists("sensitivity_cfg", mode = "function"))
    stop("common.R must provide sensitivity_cfg(branch, ordinal)")
  cfg <- sensitivity_cfg(branch, ordinal)
  stopifnot(cfg$m == 20L, cfg$maxit == 60L, cfg$selection_threshold == 10L,
    cfg$outer_folds == 5L, ordinal >= 1L, ordinal <= 50L)
  cfg$branch <- branch; cfg$ordinal <- as.integer(ordinal)
  cfg
}

sensitivity_cohort <- function(d, branch, cfg) {
  keep <- switch(branch,
    baseline = rep(TRUE, nrow(d)),
    complete_case = complete.cases(d[, cfg$model_vars, drop = FALSE]),
    egfr_observed = !is.na(d$eGFR),
    renal_observed = !is.na(d$Renal_Involvement),
    known_subtype = !is.na(d$Type),
    no_egpa = !is.na(d$Type) & as.character(d$Type) %in% c("MPA", "GPA"),
    mpa = !is.na(d$Type) & as.character(d$Type) == "MPA",
    gpa = !is.na(d$Type) & as.character(d$Type) == "GPA",
    stop("Unknown sensitivity branch: ", branch))
  stopifnot(length(keep) == nrow(d), !anyNA(keep))
  d[keep, , drop = FALSE]
}

sensitivity_status <- function(cfg, stage, message = "") {
  sensitivity_write_csv(data.frame(branch = cfg$branch, ordinal = cfg$ordinal,
    repeat_id = cfg$repeat_id, fold = cfg$fold_id, stage = stage,
    message = message, time = format(Sys.time(), tz = "Asia/Shanghai", usetz = TRUE)),
    file.path(cfg$output_dir, "status.csv"))
}

sensitivity_signature <- function(cfg) {
  hashes <- data_code_hashes()
  # The first six ordinals were prepared under the preserved original
  # pipeline. Keep their signed prepared.rds identities valid in this
  # versioned continuation; new ordinals use the current source hash.
  if (cfg$ordinal <= 6L) {
    key <- names(hashes)[grepl("(^|/)pipeline[.]R$", names(hashes))]
    stopifnot(length(key) == 1L)
    hashes[[key]] <- "5c56938123e21dbc33c6c9cd2c055ac0"
  }
  cox29_signature(list(cfg = cfg, pipeline = hashes,
    packages = vapply(c("mice", "survival", "ranger", "glmnet"),
      function(p) as.character(utils::packageVersion(p)), character(1))))
}

sensitivity_active_config <- function(dev, assessment, cfg) {
  rows <- lapply(cfg$model_vars, function(v) {
    observed <- dev[[v]][!is.na(dev[[v]])]
    if (!length(observed)) stop("No development observations for ", v)
    nv <- length(unique(observed))
    data.frame(variable = v, active = nv >= 2L, development_observed_values = nv,
      reason = if (nv < 2L) "Constant in development; not estimable" else "")
  })
  tab <- do.call(rbind, rows)
  cfg$candidate_vars <- cfg$model_vars
  cfg$model_vars <- tab$variable[tab$active]
  cfg$continuous_vars <- intersect(cfg$continuous_vars, cfg$model_vars)
  for (v in intersect(names(cfg$factor_levels), cfg$model_vars)) {
    observed <- unique(as.character(dev[[v]][!is.na(dev[[v]])]))
    lev <- cfg$factor_levels[[v]][cfg$factor_levels[[v]] %in% observed]
    unseen <- setdiff(unique(as.character(assessment[[v]][!is.na(assessment[[v]])])), lev)
    if (length(unseen)) stop("Assessment-only level of ", v, ": ", paste(unseen, collapse = ", "),
      "; cannot encode an assessment-only category without training observations")
    cfg$factor_levels[[v]] <- lev
    for (which in c("dev", "assessment")) {
      z <- get(which)
      z[[v]] <- factor(as.character(z[[v]]), levels = lev, ordered = v %in% cfg$ordered_vars)
      assign(which, z)
    }
  }
  list(dev = dev, assessment = assessment, config = cfg, candidates = tab)
}

sensitivity_complete <- function(bundle, j) {
  stopifnot(j %in% seq_len(20L))
  if (is.null(bundle$mids)) {
    stopifnot(bundle$kind %in% c("complete_case", "already_complete"),
      !anyNA(bundle$development[, bundle$model_vars, drop = FALSE]),
      !anyNA(bundle$assessment[, bundle$model_vars, drop = FALSE]))
    return(list(dev = bundle$development, assessment = bundle$assessment))
  }
  complete29(bundle, j)
}

sensitivity_check_mi <- function(out) {
  for (id in 0:5) {
    q <- file.path(out, if (id == 0L) "outer" else sprintf("inner_%02d", id))
    b <- readRDS(file.path(q, "imputation.rds"))
    if (is.null(b$mids)) {
      invisible(sensitivity_complete(b, 1L)); next
    }
    # Versioned exceptions continued from their saved 60-iteration mids states
    # only where the R-hat gate failed. Other partitions use 60.
    adaptive_80 <- (identical(basename(out), "repeat02_fold01") &&
      identical(basename(dirname(out)), "no_egpa") && id == 5L) ||
      (identical(basename(out), "repeat05_fold03") &&
       identical(basename(dirname(out)), "egfr_observed") && id == 4L) ||
      (identical(basename(out), "repeat05_fold04") &&
       identical(basename(dirname(out)), "known_subtype") && id == 2L)
    adaptive_70 <- identical(basename(out), "repeat07_fold05") &&
      identical(basename(dirname(out)), "known_subtype") && id == 0L
    expected_iteration <- if (adaptive_70) 70L else if (adaptive_80) 80L else 60L
    stopifnot(b$mids$m == 20L, b$mids$iteration == expected_iteration)
    if (!is.null(b$mids$loggedEvents)) stop("MICE logged events require review: ", q)
    ds <- read.csv(file.path(q, "mice_chain_diagnostic.csv"))
    if (nrow(ds) && any(ds$classical_Rhat > 1.1, na.rm = TRUE))
      stop("MICE chain diagnostic >1.1 requires review: ", q)
  }
  invisible(TRUE)
}

# Same lambda.min rule as lasso29, with resumable per-inner-fold checkpoints.
# Complete-case data have twenty stochastic Boruta selections, but the identical
# deterministic LASSO calculation is safely reused across selection replicates.
sensitivity_lasso <- function(d, cfg, j, directory) {
  is_cca <- identical(cfg$branch, "complete_case")
  shared <- file.path(cfg$output_dir, "complete_case_lasso.rds")
  if (is_cca && file.exists(shared)) return(readRDS(shared))
  path <- file.path(directory, "lasso_inner_checkpoint.rds")
  sig <- cox29_signature(list(cfg = cfg, j = if (is_cca) 1L else j,
    data = d, code = data_code_hashes()))
  C <- matrix(NA_real_, 5L, length(cfg$lambda))
  if (file.exists(path)) {
    old <- readRDS(path); stopifnot(identical(old$signature, sig)); C <- old$C
  }
  for (k in 1:5) {
    if (all(is.finite(C[k, ]))) next
    inner <- readRDS(file.path(cfg$output_dir, sprintf("inner_%02d/imputation.rds", k)))
    ds <- sensitivity_complete(inner, j)
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
    stopifnot(all(is.finite(C[k, ])))
    cox29_atomic_rds(list(signature = sig, C = C), path)
  }
  mean_C <- colMeans(C); se_C <- apply(C, 2, sd) / sqrt(5)
  best <- which.max(mean_C)
  one_se <- which(mean_C >= mean_C[best] - se_C[best])[1L]
  mm <- matrix29(d, cfg)
  fit <- glmnet::glmnet(mm$x, survival::Surv(d$Survival_Days, d$Death),
    family = "cox", alpha = 1, lambda = cfg$lambda, standardize = TRUE,
    cox.ties = "efron", control = list(maxit = 100000L))
  stopifnot(fit$jerr == 0L)
  b1 <- as.numeric(coef(fit, s = cfg$lambda[one_se]))
  bm <- as.numeric(coef(fit, s = cfg$lambda[best]))
  tab <- data.frame(variable = cfg$model_vars,
    lasso_1se = vapply(cfg$model_vars, function(v) any(abs(b1[mm$group == v]) > 1e-10), logical(1)),
    lasso_min = vapply(cfg$model_vars, function(v) any(abs(bm[mm$group == v]) > 1e-10), logical(1)))
  ans <- list(table = tab, coefficients = data.frame(column = colnames(mm$x), variable = mm$group,
    beta_1se = b1, beta_min = bm), cv = data.frame(lambda = cfg$lambda, mean_C = mean_C,
    SE_C = se_C, t(C)), lambda_1se = cfg$lambda[one_se], lambda_min = cfg$lambda[best],
    best_grid_endpoint = best %in% c(1L, length(cfg$lambda)), inner_C_at_1se = mean_C[one_se],
    inner_C_at_min = mean_C[best], fit = fit)
  if (is_cca) cox29_atomic_rds(ans, shared)
  ans
}

# Exact grouped boruta29 computation, checkpointed after each completed round.
# chunk_rounds limits work per process without changing seeds or algorithm.
sensitivity_boruta <- function(d, cfg, j, directory, chunk_rounds = Inf) {
  mm <- matrix29(d, cfg); x <- mm$x; group <- mm$group
  vars <- cfg$model_vars; p <- length(vars)
  checkpoint <- file.path(directory, "boruta_round_checkpoint.rds")
  sig <- cox29_signature(list(data = d, cfg = cfg, j = j,
    code = data_code_hashes()))
  decision <- hits <- integer(p)
  history <- matrix(NA_real_, cfg$rounds, p, dimnames = list(NULL, vars))
  thresholds <- numeric(cfg$rounds); actual <- 0L
  if (file.exists(checkpoint)) {
    old <- readRDS(checkpoint); stopifnot(identical(old$signature, sig))
    decision <- old$decision; hits <- old$hits; history <- old$history
    thresholds <- old$thresholds; actual <- old$actual
  }
  cindex <- function(score) survival::concordance(
    survival::Surv(d$Survival_Days, d$Death) ~ score, reverse = TRUE)$concordance
  initial <- actual
  while (actual < cfg$rounds && any(decision == 0L) && actual - initial < chunk_rounds) {
    iter <- actual + 1L; active <- which(decision >= 0L); active_names <- vars[active]
    real_cols <- which(group %in% active_names)
    real <- x[, real_cols, drop = FALSE]; real_group <- group[real_cols]
    blocks <- lapply(active_names, function(v) which(real_group == v))
    shadow <- real
    seed <- cfg$seed + 200000L + j * 100L + iter
    set.seed(seed)
    for (cols in blocks) shadow[, cols] <- real[sample.int(nrow(real)), cols, drop = FALSE]
    joined <- cbind(real, shadow); colnames(joined) <- paste0("X", seq_len(ncol(joined)))
    all_blocks <- c(blocks, lapply(blocks, function(cols) cols + ncol(real)))
    ntree <- max(1L, as.integer(floor(100 * sqrt(2 * length(active)) / cfg$depth)))
    fd <- data.frame(time = d$Survival_Days, event = d$Death, joined)
    rf <- ranger::ranger(survival::Surv(time, event) ~ ., data = fd,
      num.trees = ntree, max.depth = cfg$depth, min.node.size = cfg$minleaf,
      splitrule = "logrank", num.threads = 1L, seed = seed, write.forest = TRUE)
    risk <- function(newx) rowSums(predict(rf, data = as.data.frame(newx), num.threads = 1L)$chf)
    baseline <- cindex(risk(joined)); set.seed(seed + 1000000L)
    importance <- vapply(all_blocks, function(cols) {
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
    cox29_atomic_rds(list(signature = sig, decision = decision, hits = hits,
      history = history, thresholds = thresholds, actual = actual), checkpoint)
    cat(sprintf("%s fold %02d selection %02d round %02d: confirmed=%d unresolved=%d rejected=%d\n",
      cfg$branch, cfg$ordinal, j, iter, sum(decision == 1L), sum(decision == 0L), sum(decision == -1L)))
    flush.console()
  }
  if (actual < cfg$rounds && any(decision == 0L)) return(list(complete = FALSE, rounds = actual))
  history <- history[seq_len(actual), , drop = FALSE]; thresholds <- thresholds[seq_len(actual)]
  med <- apply(history, 2, median, na.rm = TRUE)
  status <- ifelse(decision == 1L, "Confirmed", ifelse(decision == -1L, "Rejected",
    ifelse(med > median(thresholds), "Tentative", "Undecided")))
  list(complete = TRUE, table = data.frame(variable = vars, boruta_decision = status,
    boruta_non_rejected = status != "Rejected", hits = hits, median_importance = med,
    rounds = actual), history = history, thresholds = thresholds,
    design_column_mapping = data.frame(column = colnames(x), variable = group))
}

sensitivity_stage <- function(branch, ordinal, stage, id = 0L, chunk_rounds = Inf) {
  cfg <- sensitivity_fold_cfg(branch, ordinal)
  out <- cfg$output_dir; dir.create(out, recursive = TRUE, showWarnings = FALSE)
  tryCatch(sensitivity_stage_impl(cfg, stage, id, chunk_rounds), error = function(e) {
    sensitivity_status(cfg, paste0(stage, "_failed"), conditionMessage(e))
    q <- file.path(out, "failures"); dir.create(q, showWarnings = FALSE)
    cox29_atomic_rds(list(branch = branch, ordinal = ordinal, stage = stage, id = id,
      error = conditionMessage(e), timestamp = Sys.time(), session = sessionInfo()),
      file.path(q, sprintf("%s_%02d.rds", stage, id)))
    stop(e)
  })
}

sensitivity_stage_impl <- function(original_cfg, stage, id, chunk_rounds) {
  out <- original_cfg$output_dir
  prepare_path <- file.path(out, "prepared.rds")
  if (stage == "prepare") {
    if (file.exists(prepare_path)) {
      old <- readRDS(prepare_path)
      stopifnot(identical(old$original_config, original_cfg), identical(old$signature, sensitivity_signature(old$config)))
      return(invisible(TRUE))
    }
    d <- readRDS(file.path(input_dir, "raw_typed.rds"))
    assignments <- read.csv(file.path(input_dir, "outer_assignments.csv"))
    assignments <- assignments[assignments$repeat_id == original_cfg$repeat_id, ]
    stopifnot(!anyDuplicated(assignments$patient_id), setequal(assignments$patient_id, d$.patient_id))
    d <- sensitivity_cohort(d, original_cfg$branch, original_cfg)
    a <- assignments[match(d$.patient_id, assignments$patient_id), ]
    stopifnot(all(a$Death == d$Death), !anyNA(a$assessment_fold))
    dev <- d[a$assessment_fold != original_cfg$fold_id, , drop = FALSE]
    assessment <- d[a$assessment_fold == original_cfg$fold_id, , drop = FALSE]
    stopifnot(nrow(assessment) > 0L, sum(assessment$Death) > 0L)
    active <- sensitivity_active_config(dev, assessment, original_cfg)
    cfg <- active$config
    dat <- list(dev = active$dev, assessment = active$assessment,
      inner_fold = stratified_folds(dev$Death, 5L, cfg$seed))
    cox29_atomic_rds(list(data = dat, config = cfg, original_config = original_cfg,
      candidates = active$candidates, signature = sensitivity_signature(cfg)), prepare_path)
    sensitivity_write_csv(active$candidates, file.path(out, "candidate_estimability.csv"))
    sensitivity_write_csv(data.frame(patient_id = dev$.patient_id, Death = dev$Death,
      inner_fold = dat$inner_fold), file.path(out, "inner_assignments.csv"))
    sensitivity_write_csv(data.frame(branch = cfg$branch, ordinal = cfg$ordinal,
      development_N = nrow(dev), development_deaths = sum(dev$Death),
      assessment_N = nrow(assessment), assessment_deaths = sum(assessment$Death)),
      file.path(out, "cohort_counts.csv"))
    sensitivity_status(cfg, "prepared"); return(invisible(TRUE))
  }
  p <- readRDS(prepare_path)
  stopifnot(identical(p$original_config, original_cfg), identical(p$signature, sensitivity_signature(p$config)))
  cfg <- p$config; dat <- p$data
  if (stage == "impute") {
    stopifnot(id %in% 0:5)
    q <- file.path(out, if (id == 0L) "outer" else sprintf("inner_%02d", id))
    dir.create(q, showWarnings = FALSE)
    if (file.exists(file.path(q, "imputation.rds"))) return(invisible(TRUE))
    tr <- if (id == 0L) dat$dev else dat$dev[dat$inner_fold != id, , drop = FALSE]
    va <- if (id == 0L) dat$assessment else dat$dev[dat$inner_fold == id, , drop = FALSE]
    is_cca <- identical(cfg$branch, "complete_case")
    if (is_cca) {
      stopifnot(!anyNA(tr[, cfg$candidate_vars, drop = FALSE]), !anyNA(va[, cfg$candidate_vars, drop = FALSE]))
      bundle <- list(mids = NULL, development = tr, assessment = va, model_vars = cfg$model_vars,
        kind = "complete_case", note = "No imputation; selection replicate is stochastic Boruta only")
      sensitivity_write_csv(data.frame(variable = cfg$model_vars, method = "none_complete_case"),
        file.path(q, "imputation_methods.csv"))
    } else {
      cat(cfg$branch, "fold", cfg$ordinal, "MICE partition", id, "development", nrow(tr),
        "assessment", nrow(va), "m20 maxit60\n"); flush.console()
      mi <- impute_pair(tr, va, cfg, cfg$seed + id * 100L)
      if (!is.null(mi$mids)) stopifnot(mi$mids$m == 20L, mi$mids$iteration == 60L,
        identical(mi$mids$ignore, c(rep(FALSE, nrow(tr)), rep(TRUE, nrow(va)))),
        identical(names(mi$mids$data), cfg$model_vars),
        !any(c("Death", "Survival_Days", ".patient_id") %in% names(mi$mids$data)))
      bundle <- list(mids = mi$mids, development = if (is.null(mi$mids)) mi$dev[[1L]] else tr,
        assessment = if (is.null(mi$mids)) mi$assessment[[1L]] else va, model_vars = cfg$model_vars,
        kind = if (is.null(mi$mids)) "already_complete" else "multiple_imputation")
      sensitivity_write_csv(data.frame(variable = names(mi$method), method = mi$method), file.path(q, "imputation_methods.csv"))
      if (!is.null(mi$mids$loggedEvents)) sensitivity_write_csv(mi$mids$loggedEvents, file.path(q, "mice_logged_events.csv"))
      if (length(mi$constant_notes)) writeLines(mi$constant_notes, file.path(q, "constant_notes.txt"))
      if (!is.null(mi$mids)) diagnose29_mice(mi$mids, q)
    }
    for (j in 1:20) invisible(sensitivity_complete(bundle, j))
    cox29_atomic_rds(bundle, file.path(q, "imputation.rds"))
    sensitivity_status(cfg, paste0("imputed_partition_", id)); return(invisible(TRUE))
  }
  if (stage == "select") {
    stopifnot(id %in% 1:20); sensitivity_check_mi(out)
    q <- file.path(out, sprintf("imputation_%02d", id)); dir.create(q, showWarnings = FALSE)
    if (file.exists(file.path(q, "combined_selection.csv"))) return(invisible(TRUE))
    ds <- sensitivity_complete(readRDS(file.path(out, "outer", "imputation.rds")), id)
    if (file.exists(file.path(q, "lasso.rds"))) lasso <- readRDS(file.path(q, "lasso.rds")) else {
      lasso <- sensitivity_lasso(ds$dev, cfg, id, q); cox29_atomic_rds(lasso, file.path(q, "lasso.rds"))
    }
    if (file.exists(file.path(q, "boruta.rds"))) boruta <- readRDS(file.path(q, "boruta.rds")) else {
      boruta <- sensitivity_boruta(ds$dev, cfg, id, q, chunk_rounds)
      if (!boruta$complete) return(invisible(FALSE))
      cox29_atomic_rds(boruta, file.path(q, "boruta.rds"))
    }
    stopifnot(identical(boruta$table$variable, lasso$table$variable))
    z <- data.frame(imputation = id, boruta$table, lasso_lambda_min = lasso$table$lasso_min,
      lasso_lambda_1se = lasso$table$lasso_1se, candidate_active = TRUE, exclusion_reason = "")
    inactive <- p$candidates[!p$candidates$active, ]
    if (nrow(inactive)) z <- rbind(z, data.frame(imputation = id, variable = inactive$variable,
      boruta_decision = "Not_estimable", boruta_non_rejected = FALSE, hits = 0L,
      median_importance = NA_real_, rounds = 0L, lasso_lambda_min = FALSE,
      lasso_lambda_1se = FALSE, candidate_active = FALSE, exclusion_reason = inactive$reason))
    z <- z[match(cfg$candidate_vars, z$variable), ]
    z$union_selected <- z$boruta_non_rejected | z$lasso_lambda_min
    sensitivity_write_csv(z, file.path(q, "combined_selection.csv"))
    sensitivity_write_csv(lasso$coefficients, file.path(q, "lasso_coefficients.csv"))
    sensitivity_write_csv(lasso$cv, file.path(q, "lasso_nested_cv.csv"))
    sensitivity_write_csv(boruta$design_column_mapping, file.path(q, "design_column_mapping.csv"))
    plot29_boruta(boruta, id, q)
    sensitivity_status(cfg, paste0("selected_", id)); return(invisible(TRUE))
  }
  if (stage == "summary") {
    sensitivity_check_mi(out)
    paths <- file.path(out, sprintf("imputation_%02d/combined_selection.csv", 1:20))
    stopifnot(all(file.exists(paths)))
    all <- do.call(rbind, lapply(paths, read.csv)); stopifnot(nrow(all) == 29L * 20L,
      all(table(all$variable) == 20L))
    freq <- do.call(rbind, lapply(cfg$candidate_vars, function(v) {
      z <- all[all$variable == v, ]
      data.frame(variable = v, candidate_active = all(z$candidate_active),
        boruta_non_rejected = sum(z$boruta_non_rejected), lasso_lambda_min = sum(z$lasso_lambda_min),
        union_selected = sum(z$union_selected), included = sum(z$union_selected) >= 10L)
    }))
    sensitivity_write_csv(all, file.path(out, "all_selection_results.csv"))
    sensitivity_write_csv(freq, file.path(out, "selection_frequency_29variables.csv"))
    sensitivity_status(cfg, "selection_summarized"); return(invisible(TRUE))
  }
  stop("Unknown data stage: ", stage)
}
