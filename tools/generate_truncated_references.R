local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.tr))

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)

evaluate_case <- function(
    case,
    family,
    type,
    lower,
    upper,
    y,
    mu,
    sigma,
    probability) {
  par <- switch(
    type,
    left = lower,
    right = upper,
    both = c(lower, upper)
  )
  density <- trun.d(par = par, family = family, type = type)
  probability_function <- trun.p(par = par, family = family, type = type)
  quantile_function <- trun.q(par = par, family = family, type = type)
  family_constructor <- trun(
    par = par,
    family = family,
    type = type,
    local = TRUE
  )
  family_object <- family_constructor()

  parameter_arguments <- list(mu = mu)
  if (family == "NO") {
    parameter_arguments$sigma <- sigma
  }
  arguments_at <- function(index) {
    lapply(parameter_arguments, function(value) value[index])
  }
  density_values <- vapply(
    seq_along(y),
    function(index) {
      do.call(
        density,
        c(list(x = y[index]), arguments_at(index), list(log = TRUE))
      )
    },
    numeric(1)
  )
  cdf_values <- vapply(
    seq_along(y),
    function(index) {
      do.call(
        probability_function,
        c(list(q = y[index]), arguments_at(index))
      )
    },
    numeric(1)
  )
  quantile_values <- vapply(
    seq_along(y),
    function(index) {
      do.call(
        quantile_function,
        c(list(p = probability[index]), arguments_at(index))
      )
    },
    numeric(1)
  )
  score_mu <- vapply(
    seq_along(y),
    function(index) {
      as.numeric(do.call(
        family_object$dldm,
        c(list(y = y[index]), arguments_at(index))
      ))
    },
    numeric(1)
  )

  if (family == "NO") {
    score_sigma <- vapply(
      seq_along(y),
      function(index) {
        as.numeric(do.call(
          family_object$dldd,
          c(list(y = y[index]), arguments_at(index))
        ))
      },
      numeric(1)
    )
    second_mu <- vapply(
      sigma,
      family_object$d2ldm2,
      numeric(1)
    )
    second_sigma <- vapply(
      sigma,
      family_object$d2ldd2,
      numeric(1)
    )
    second_cross <- vapply(
      y,
      family_object$d2ldmdd,
      numeric(1)
    )
  } else {
    score_sigma <- rep(NA_real_, length(y))
    second_mu <- vapply(
      mu,
      family_object$d2ldm2,
      numeric(1)
    )
    second_sigma <- rep(NA_real_, length(y))
    second_cross <- rep(NA_real_, length(y))
  }

  data.frame(
    case = case,
    family = family,
    type = type,
    lower = if (is.na(lower)) "" else format(lower, scientific = FALSE),
    upper = if (is.na(upper)) "" else format(upper, scientific = FALSE),
    y = y,
    mu = mu,
    sigma = sigma,
    probability = probability,
    log_density = density_values,
    cdf = cdf_values,
    quantile = quantile_values,
    dldmu = score_mu,
    dldsigma = score_sigma,
    d2ldmu2 = second_mu,
    d2ldsigma2 = second_sigma,
    d2ldmudsigma = second_cross,
    gamlss_tr_version = as.character(packageVersion("gamlss.tr")),
    check.names = FALSE
  )
}

references <- rbind(
  evaluate_case(
    "normal_left", "NO", "left", 0, NA,
    c(0.1, 0.8, 2.1),
    c(-0.5, 0.4, 1.1),
    c(0.6, 1.0, 1.4),
    c(0.1, 0.5, 0.9)
  ),
  evaluate_case(
    "normal_right", "NO", "right", NA, 1.5,
    c(-1.0, 0.2, 1.4),
    c(-0.6, 0.5, 1.0),
    c(0.7, 1.1, 0.9),
    c(0.15, 0.55, 0.85)
  ),
  evaluate_case(
    "normal_both", "NO", "both", -1, 2,
    c(-0.8, 0.5, 1.7),
    c(-0.3, 0.4, 1.2),
    c(0.8, 1.3, 0.7),
    c(0.2, 0.6, 0.8)
  ),
  evaluate_case(
    "poisson_left", "PO", "left", 0, NA,
    c(1, 2, 5),
    c(0.7, 2.5, 5.0),
    c(NA, NA, NA),
    c(0.1, 0.5, 0.9)
  ),
  evaluate_case(
    "poisson_right", "PO", "right", NA, 6,
    c(0, 2, 5),
    c(0.8, 2.7, 5.5),
    c(NA, NA, NA),
    c(0.15, 0.55, 0.85)
  ),
  evaluate_case(
    "poisson_both", "PO", "both", 0, 6,
    c(1, 3, 5),
    c(0.9, 3.2, 6.0),
    c(NA, NA, NA),
    c(0.2, 0.6, 0.8)
  )
)

reference_connection <- file(
  file.path(reference_dir, "truncated_reference.csv"),
  open = "wb"
)
write.csv(
  references,
  reference_connection,
  row.names = FALSE,
  na = ""
)
close(reference_connection)

message(
  "Wrote truncated-family references from gamlss.tr ",
  packageVersion("gamlss.tr")
)
