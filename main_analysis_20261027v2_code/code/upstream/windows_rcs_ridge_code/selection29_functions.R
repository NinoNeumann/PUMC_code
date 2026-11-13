# Reuse the completed outcome-blind imputation datasets; no reselection here.
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
