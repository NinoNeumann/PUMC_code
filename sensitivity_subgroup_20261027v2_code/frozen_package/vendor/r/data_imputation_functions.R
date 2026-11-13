# Exact stratified folds and outcome-blind MICE used by the source analysis.
stratified_folds <- function(event, k, seed) {
  if (length(unique(event)) != 2L || min(table(event)) < k)
    stop("Too few events/non-events for the requested number of folds.")
  set.seed(seed)
  fold <- integer(length(event))
  n <- length(event)
  target_size <- rep(n %/% k, k)
  if (n %% k) target_size[sample.int(k, n %% k)] <- target_size[1L] + 1L
  death_ids <- sample(which(event == 1L))
  survivor_ids <- sample(which(event == 0L))
  death_target <- rep(length(death_ids) %/% k, k)
  if (length(death_ids) %% k)
    death_target[sample.int(k, length(death_ids) %% k)] <-
      length(death_ids) %/% k + 1L
  survivor_target <- target_size - death_target
  if (any(survivor_target < 0L) || sum(survivor_target) != length(survivor_ids))
    stop("Could not make balanced stratified folds.")
  fold[death_ids] <- rep(seq_len(k), death_target)
  fold[survivor_ids] <- rep(seq_len(k), survivor_target)
  fold
}

write_csv <- function(x, path) {
  utils::write.csv(x, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
}

# MICE estimates are based on development rows only (ignore = TRUE for assessment).
# Initialization also uses only development observations; initial global constant/
# collinearity pruning is disabled because it would inspect assessment rows.
# Outcomes, IDs, FFS are deliberately absent from the imputation input.
# This is an outcome-blind prediction pipeline, not an outcome-assisted MI
# analysis intended for unbiased etiological regression coefficients.
impute_pair <- function(dev, assessment, cfg, seed) {
  vars <- unique(c(cfg$model_vars, cfg$auxiliary_vars))
  if (any(vars %in% c("Death", "Survival_Days", "Five_Factor_Score", ".patient_id")))
    stop("Outcomes, ID and FFS cannot be imputation variables in this pipeline.")
  nd <- nrow(dev)
  both <- rbind(dev[, vars, drop = FALSE], assessment[, vars, drop = FALSE])
  rownames(both) <- NULL
  method <- setNames(rep("", length(vars)), vars)
  pred <- matrix(1L, length(vars), length(vars), dimnames = list(vars, vars))
  diag(pred) <- 0L
  constant_notes <- character()
  for (v in vars) {
    observed <- dev[[v]][!is.na(dev[[v]])]
    if (!length(observed)) stop("No development observations for ", v)
    if (length(unique(observed)) < 2L) {
      pred[v, ] <- 0L
      pred[, v] <- 0L
      both[[v]][is.na(both[[v]])] <- observed[1L]
      constant_notes <- c(constant_notes, paste(v, "constant in development; missing values filled with that value"))
    } else if (anyNA(both[[v]])) {
      method[v] <- if (is.ordered(both[[v]])) "polr" else if (is.factor(both[[v]])) {
        if (nlevels(both[[v]]) == 2L) "logreg" else "polyreg"
      } else "pmm"
    }
  }
  if (!anyNA(both)) {
    completed <- rep(list(both), cfg$m)
    mids <- NULL
  } else {
    initial <- both
    set.seed(seed)
    for (v in vars) {
      miss <- which(is.na(initial[[v]]))
      donors <- dev[[v]][!is.na(dev[[v]])]
      if (length(miss)) initial[[v]][miss] <- donors[sample.int(length(donors), length(miss), replace = TRUE)]
    }
    mids <- mice::mice(
      both, m = cfg$m, maxit = cfg$maxit, method = method,
      predictorMatrix = pred, ignore = seq_len(nrow(both)) > nd,
      data.init = initial, seed = seed, printFlag = FALSE,
      remove.constant = FALSE, remove.collinear = FALSE,
      donors = cfg$pmm_donors, polr.to.loggedEvents = TRUE
    )
    completed <- lapply(seq_len(cfg$m), function(j) mice::complete(mids, j))
  }
  dev_out <- assessment_out <- vector("list", cfg$m)
  for (j in seq_len(cfg$m)) {
    dd <- dev; aa <- assessment
    for (v in vars) {
      dd[[v]] <- completed[[j]][seq_len(nd), v]
      aa[[v]] <- completed[[j]][nd + seq_len(nrow(assessment)), v]
    }
    if (anyNA(dd[, vars, drop = FALSE]) || anyNA(aa[, vars, drop = FALSE]))
      stop("Incomplete MICE output. Inspect imputation logs.")
    # Original FFS and observed outcome values are never changed.
    dev_out[[j]] <- dd; assessment_out[[j]] <- aa
  }
  list(dev = dev_out, assessment = assessment_out, mids = mids,
       constant_notes = constant_notes, method = method, predictor_matrix = pred)
}

