# Scalar survival risk only. The forest and Boruta sampling/decisions are unchanged.
# Sum_t mean_tree CHF(tree,leaf,t) = mean_tree sum_t CHF(tree,leaf,t).
# Floating point addition order changes; verified separately, never called bit-exact.
make_leaf_risk <- function(rf, d, j) {
  stopifnot(identical(rf$treetype, "Survival"), length(rf$forest$chf) == rf$num.trees)
  state_before <- get(".Random.seed", envir = globalenv())
  leaf_sums <- lapply(rf$forest$chf, function(tree) vapply(tree, sum, numeric(1)))
  time_count <- length(rf$unique.death.times)
  stopifnot(time_count > 0L)
  stopifnot(identical(state_before, get(".Random.seed", envir = globalenv())),
            all(vapply(leaf_sums, function(z) all(is.finite(z)), logical(1))))
  function(newx) {
    before <- if (isTRUE(risk_audit_enabled)) get(".Random.seed", envir = globalenv()) else NULL
    started <- proc.time()[["elapsed"]]
    # Exactly one predict call with default seed=NULL, matching the original's
    # runif(1) consumption; cache preparation and scalar arithmetic use no RNG.
    nodes <- predict(rf, data = as.data.frame(newx), type = "terminalNodes", num.threads = 1L)$predictions
    stopifnot(is.matrix(nodes), nrow(nodes) == nrow(newx), ncol(nodes) == rf$num.trees)
    values <- matrix(0, nrow(nodes), ncol(nodes))
    for (tree in seq_len(ncol(nodes))) {
      ids <- nodes[, tree] + 1L # ranger node IDs are zero-based.
      if (any(ids < 1L | ids > length(leaf_sums[[tree]]))) stop("Terminal node outside saved CHF")
      values[, tree] <- leaf_sums[[tree]][ids]
    }
    scores <- rowMeans(values)
    # Nonnegative CHF sums allow a conservative roundoff screen. If adjacent
    # scores could have had their order/tie changed by reassociation, reconstruct
    # just those curves using the ORIGINAL time/tree summation order. This is a
    # numerical safety check, not score rounding or a new clinical threshold.
    bound <- 8 * .Machine$double.eps * (rf$num.trees + time_count + 4) * pmax(1, abs(scores))
    stopifnot(all(is.finite(scores)))
    group_size <- if(length(scores) %% nrow(d) == 0L) nrow(d) else length(scores)
    refine <- integer()
    for (block in seq_len(length(scores) / group_size)) {
      rows <- seq_len(group_size) + (block - 1L) * group_size
      ord <- rows[order(scores[rows])]
      left <- ord[-length(ord)];right <- ord[-1L]
      near <- which(scores[right] - scores[left] <= bound[left] + bound[right])
      if (!length(near)) next
      groups <- split(near, cumsum(c(1L, as.integer(diff(near) > 1L))))
      for (group in groups) {
        ids <- ord[c(group, tail(group, 1L) + 1L)]
        # Identical terminal paths have identical original CHF/risk. No
        # reconstruction is needed even for a large group of tied observations.
        if (any(nodes[ids, , drop = FALSE] != rep(nodes[ids[1L], ], each = length(ids))))
          refine <- c(refine, ids)
      }
    }
    refine <- unique(refine)
    if (length(refine)) {
      curve <- matrix(0, length(refine), time_count)
      for (tree in seq_len(ncol(nodes))) {
        increments <- vapply(nodes[refine, tree] + 1L,
          function(id) rf$forest$chf[[tree]][[id]], numeric(time_count))
        # Same double additions in ascending tree order as ranger's
        # ForestSurvival::predictInternal, then divide and R rowSums.
        curve <- curve + t(matrix(increments, nrow = time_count, ncol = length(refine)))
      }
      scores[refine] <- rowSums(curve / rf$num.trees)
    }
    optimized_seconds <- proc.time()[["elapsed"]] - started
    if (isTRUE(risk_audit_enabled)) {
      after <- get(".Random.seed", envir = globalenv())
      assign(".Random.seed", before, envir = globalenv())
      started <- proc.time()[["elapsed"]]
      original <- rowSums(predict(rf, data = as.data.frame(newx), num.threads = 1L)$chf)
      original_seconds <- proc.time()[["elapsed"]] - started
      rng_equal <- identical(after, get(".Random.seed", envir = globalenv()))
      assign(".Random.seed", after, envir = globalenv())
      # Each original prediction is either the n-row baseline or five stacked
      # n-row permutations. Compare rankings and C within each original block.
      stopifnot(length(scores) %% nrow(d) == 0L)
      fast <- matrix(scores, nrow(d)); old <- matrix(original, nrow(d))
      rank_equal <- all(vapply(seq_len(ncol(fast)), function(k)
        identical(rank(fast[, k], ties.method = "min"), rank(old[, k], ties.method = "min")), logical(1)))
      cfun <- function(score) survival::concordance(
        survival::Surv(d$Survival_Days, d$Death) ~ score, reverse = TRUE)$concordance
      max_c_difference <- max(abs(apply(fast, 2, cfun) - apply(old, 2, cfun)))
      row <- data.frame(imputation = j, call = length(risk_audit_rows) + 1L,
        prediction_rows = length(scores), optimized_prediction_seconds = optimized_seconds,
        exact_refinement_rows = length(refine),
        original_prediction_seconds = original_seconds,
        max_abs_risk_difference = max(abs(scores - original)),
        max_relative_risk_difference = max(abs(scores - original) / pmax(1, abs(original))),
        rankings_equal = rank_equal, cindex_max_abs_difference = max_c_difference,
        RNG_state_equal = rng_equal)
      risk_audit_rows[[length(risk_audit_rows) + 1L]] <<- row
      if (!rng_equal || !rank_equal || max_c_difference != 0) {
        utils::write.csv(do.call(rbind,risk_audit_rows),
          file.path(out,sprintf("imputation_%02d",j),"risk_prediction_audit_FAILED.csv"),row.names=FALSE)
        stop("Scalar risk audit failed: RNG, rankings or Harrell C changed")
      }
    }
    scores
  }
}
