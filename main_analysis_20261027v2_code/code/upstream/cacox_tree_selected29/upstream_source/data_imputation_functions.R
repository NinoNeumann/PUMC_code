# AAV common data preparation, outcome-blind MICE and RCS basis utilities.
# Used by the isolated 20261027v2 29-predictor pipeline.
# Survival_Days in days; Death 1=death, 0=censored.
# Raw Excel is preserved.

check_packages <- function() {
  need <- c("readxl", "mice", "survival")
  missing <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Install required packages: ", paste(missing, collapse = ", "))
}

clean_text <- function(x) {
  z <- trimws(as.character(x))
  z[tolower(z) %in% c("", "na", "n/a", "nan", "null", "none", "unknown",
                     "unkown", "未知", "缺失")] <- NA_character_
  z
}

as_numeric_checked <- function(x, name) {
  z <- clean_text(x)
  out <- suppressWarnings(as.numeric(z))
  if (any(!is.na(z) & is.na(out))) stop("Non-numeric values in ", name)
  if (any(!is.finite(out[!is.na(out)]))) stop("Non-finite values in ", name)
  out
}

prepare_data <- function(raw, cfg, outcomes = TRUE) {
  d <- as.data.frame(raw, stringsAsFactors = FALSE)
  needed <- unique(c(cfg$model_vars, cfg$auxiliary_vars,
                     if (outcomes) c("Death", "Survival_Days")))
  absent <- setdiff(needed, names(d))
  if (length(absent)) stop("Missing columns: ", paste(absent, collapse = ", "))
  for (v in needed) d[[v]] <- clean_text(d[[v]])
  for (v in cfg$continuous_vars) d[[v]] <- as_numeric_checked(d[[v]], v)
  for (v in names(cfg$factor_levels)) {
    if (!v %in% needed) next
    z <- d[[v]]
    if (v == "Type") z <- toupper(z) else z <- tolower(z)
    lev <- cfg$factor_levels[[v]]
    bad <- setdiff(unique(z[!is.na(z)]), lev)
    if (length(bad)) stop("Unexpected category in ", v, ": ", paste(bad, collapse = ", "))
    d[[v]] <- factor(z, levels = lev, ordered = v %in% cfg$ordered_vars)
  }
  typed <- union(cfg$continuous_vars, names(cfg$factor_levels))
  if (length(setdiff(c(cfg$model_vars, cfg$auxiliary_vars), typed)))
    stop("Every model/auxiliary variable needs a type in the configuration.")
  if (outcomes) {
    d$Death <- as_numeric_checked(d$Death, "Death")
    d$Survival_Days <- as_numeric_checked(d$Survival_Days, "Survival_Days")
    if (anyNA(d$Death) || anyNA(d$Survival_Days)) stop("Outcomes must not be missing.")
    if (any(!d$Death %in% c(0, 1))) stop("Death must be coded 0/1.")
    if (any(d$Survival_Days <= 0)) stop("Survival_Days must be positive.")
  }
  if ("Five_Factor_Score" %in% names(d))
    d$Five_Factor_Score <- as_numeric_checked(d$Five_Factor_Score, "Five_Factor_Score")
  if (!".patient_id" %in% names(d)) d$.patient_id <- seq_len(nrow(d))
  if (anyDuplicated(d$.patient_id)) stop("Patient ID must identify a unique row.")
  d
}

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

save_imputation <- function(mi, directory, export_csv = TRUE) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  saveRDS(mi, file.path(directory, "imputation_bundle.rds"), compress = TRUE)
  write_csv(data.frame(variable = names(mi$method), method = mi$method),
            file.path(directory, "imputation_methods.csv"))
  if (length(mi$constant_notes)) writeLines(mi$constant_notes, file.path(directory, "constant_variables.txt"))
  if (!is.null(mi$mids)) {
    if (!is.null(mi$mids$loggedEvents)) write_csv(mi$mids$loggedEvents, file.path(directory, "mice_logged_events.csv"))
    # Means/variances by iteration and chain retained for convergence review.
    saveRDS(list(mean = mi$mids$chainMean, variance = mi$mids$chainVar),
            file.path(directory, "mice_chain_diagnostics.rds"))
    tryCatch({
      grDevices::pdf(file.path(directory, "mice_trace.pdf"), width = 9, height = 7)
      tryCatch(print(plot(mi$mids)), finally = grDevices::dev.off())
    }, error = function(e) warning("Trace plot unavailable: ", conditionMessage(e)))
  }
  if (export_csv) for (j in seq_along(mi$dev)) {
    write_csv(mi$dev[[j]], file.path(directory, sprintf("development_imp_%02d.csv", j)))
    if (nrow(mi$assessment[[j]]))
      write_csv(mi$assessment[[j]], file.path(directory, sprintf("assessment_imp_%02d.csv", j)))
  }
}

# Restricted cubic spline basis (Harrell parameterisation, linear tails).
# K knots produce K-1 columns: one linear and K-2 nonlinear terms.
rcs_basis <- function(x, knots) {
  k <- length(knots)
  if (k < 3L || any(diff(knots) <= 0)) stop("RCS knots must be distinct and increasing.")
  pos3 <- function(a) pmax(a, 0)^3
  nonlinear <- vapply(seq_len(k - 2L), function(j) {
    (pos3(x - knots[j]) - pos3(x - knots[k - 1L]) *
       (knots[k] - knots[j]) / (knots[k] - knots[k - 1L]) +
       pos3(x - knots[k]) * (knots[k - 1L] - knots[j]) /
       (knots[k] - knots[k - 1L])) / (knots[k] - knots[1L])^2
  }, numeric(length(x)))
  matrix(cbind(x, nonlinear), nrow = length(x), ncol = k - 1L)
}

make_spec <- function(raw_dev, cfg) {
  knots <- list(); reference <- list(); iqrs <- list()
  for (v in cfg$model_vars) {
    if (v %in% cfg$continuous_vars) {
      x <- raw_dev[[v]][!is.na(raw_dev[[v]])]
      q <- as.numeric(stats::quantile(x, cfg$knot_probs, type = 7))
      if (length(unique(q)) != length(q))
        stop("Repeated knots for ", v, ". Prespecify an appropriate alternative; no silent fallback.")
      knots[[v]] <- q
      reference[[v]] <- stats::median(x)
      iqrs[[v]] <- as.numeric(stats::quantile(x, c(.25, .75), type = 7))
    } else reference[[v]] <- cfg$factor_levels[[v]][1L]
  }
  list(vars = cfg$model_vars, continuous = intersect(cfg$model_vars, cfg$continuous_vars),
       levels = cfg$factor_levels, knots = knots,
       reference = as.data.frame(reference, stringsAsFactors = FALSE), iqrs = iqrs)
}

