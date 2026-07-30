args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

suppressPackageStartupMessages(library(mgcv))
options(digits = 17)

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
design_path <- file.path(
  reference_dir,
  "mgcv_tensor_design_reference.csv"
)
penalty_path <- file.path(
  reference_dir,
  "mgcv_tensor_penalty_reference.csv"
)

first_design <- matrix(
  c(
    1.0, -0.5, 0.2,
    1.0, -0.1, 0.7,
    1.0, 0.3, -0.4,
    1.0, 0.8, 0.5,
    1.0, 1.2, -0.2
  ),
  nrow = 5,
  byrow = TRUE
)
second_design <- matrix(
  c(
    0.8, 0.2,
    0.6, 0.4,
    0.5, 0.5,
    0.3, 0.7,
    0.1, 0.9
  ),
  nrow = 5,
  byrow = TRUE
)
first_difference <- matrix(c(1, -2, 1), nrow = 1)
second_difference <- matrix(c(1, -1), nrow = 1)
first_penalty <- crossprod(first_difference)
second_penalty <- crossprod(second_difference)

tensor_design <- tensor.prod.model.matrix(
  list(first_design, second_design)
)
colnames(tensor_design) <- paste0(
  "coefficient_",
  seq_len(ncol(tensor_design))
)
design_reference <- data.frame(
  observation = seq_len(nrow(tensor_design)),
  tensor_design,
  check.names = FALSE
)

tensor_penalties <- tensor.prod.penalties(
  list(first_penalty, second_penalty)
)
penalty_reference <- do.call(
  rbind,
  lapply(
    seq_along(tensor_penalties),
    function(index) {
      penalty <- tensor_penalties[[index]]
      data.frame(
        penalty = index,
        row = rep(seq_len(nrow(penalty)), each = ncol(penalty)),
        column = rep(seq_len(ncol(penalty)), nrow(penalty)),
        value = as.vector(t(penalty))
      )
    }
  )
)

write_reference <- function(value, path) {
  connection <- file(path, open = "wb")
  on.exit(close(connection))
  write.csv(value, connection, row.names = FALSE, na = "")
}

check_reference <- function(actual, path, label) {
  temporary_path <- tempfile(fileext = ".csv")
  write_reference(actual, temporary_path)
  generated <- read.csv(
    temporary_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  expected <- read.csv(
    path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!identical(names(generated), names(expected)) ||
      nrow(generated) != nrow(expected)) {
    stop(label, " reference dimensions or columns differ")
  }
  for (column in names(generated)) {
    if (is.numeric(generated[[column]]) &&
        is.numeric(expected[[column]])) {
      difference <- abs(
        generated[[column]] - expected[[column]]
      )
      allowed <- 1e-12 * (1 + abs(expected[[column]]))
      if (any(difference > allowed)) {
        stop(label, " numeric parity differs for ", column)
      }
    } else if (!identical(
      generated[[column]],
      expected[[column]]
    )) {
      stop(label, " reference values differ for ", column)
    }
  }
}

if (check_only) {
  check_reference(design_reference, design_path, "Tensor design")
  check_reference(penalty_reference, penalty_path, "Tensor penalty")
  message(
    "mgcv tensor-product reference checks passed with mgcv ",
    packageVersion("mgcv")
  )
} else {
  write_reference(design_reference, design_path)
  write_reference(penalty_reference, penalty_path)
  message(
    "Wrote mgcv tensor-product references with mgcv ",
    packageVersion("mgcv")
  )
}
