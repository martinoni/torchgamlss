args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.tr))

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
reference_path <- file.path(reference_dir, "truncated_reference.csv")

evaluate_case <- function(
    case,
    family,
    type,
    lower,
    upper,
    y,
    mu,
    sigma,
    probability,
    varying = FALSE) {
  par <- switch(
    type,
    left = lower,
    right = upper,
    both = if (varying) cbind(lower, upper) else c(lower, upper)
  )
  density <- trun.d(
    par = par,
    family = family,
    type = type,
    varying = varying
  )
  probability_function <- trun.p(
    par = par,
    family = family,
    type = type,
    varying = varying
  )
  quantile_function <- trun.q(
    par = par,
    family = family,
    type = type,
    varying = varying
  )
  family_constructor <- trun(
    par = par,
    family = family,
    type = type,
    local = TRUE,
    varying = varying
  )
  family_object <- family_constructor()

  parameter_arguments <- list(mu = mu)
  if (family == "NO") {
    parameter_arguments$sigma <- sigma
  }
  arguments_at <- function(index) {
    lapply(parameter_arguments, function(value) value[index])
  }
  if (varying) {
    density_values <- do.call(
      density,
      c(list(x = y), parameter_arguments, list(log = TRUE))
    )
    cdf_values <- do.call(
      probability_function,
      c(list(q = y), parameter_arguments)
    )
    quantile_values <- do.call(
      quantile_function,
      c(list(p = probability), parameter_arguments)
    )
    score_mu <- as.numeric(do.call(
      family_object$dldm,
      c(list(y = y), parameter_arguments)
    ))
  } else {
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
  }

  if (family == "NO") {
    if (varying) {
      score_sigma <- as.numeric(do.call(
        family_object$dldd,
        c(list(y = y), parameter_arguments)
      ))
    } else {
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
    }
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
    varying = varying,
    lower = ifelse(
      is.na(lower),
      "",
      format(lower, scientific = FALSE, trim = TRUE)
    ),
    upper = ifelse(
      is.na(upper),
      "",
      format(upper, scientific = FALSE, trim = TRUE)
    ),
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
  ),
  evaluate_case(
    "normal_varying_left", "NO", "left",
    c(-0.5, 0.2, 1.0), NA,
    c(-0.2, 0.8, 1.7),
    c(-0.6, 0.4, 1.2),
    c(0.7, 1.0, 0.8),
    c(0.1, 0.5, 0.9),
    varying = TRUE
  ),
  evaluate_case(
    "normal_varying_right", "NO", "right",
    NA, c(0.2, 1.3, 2.4),
    c(-0.8, 0.6, 2.0),
    c(-0.4, 0.7, 1.5),
    c(0.9, 0.8, 1.1),
    c(0.15, 0.55, 0.85),
    varying = TRUE
  ),
  evaluate_case(
    "normal_varying_both", "NO", "both",
    c(-1.2, -0.1, 0.8), c(0.4, 1.4, 2.8),
    c(-0.5, 0.7, 2.2),
    c(-0.4, 0.5, 1.7),
    c(0.8, 1.0, 0.7),
    c(0.2, 0.6, 0.8),
    varying = TRUE
  ),
  evaluate_case(
    "poisson_varying_left", "PO", "left",
    c(0, 1, 3), NA,
    c(1, 3, 6),
    c(0.9, 2.8, 5.5),
    c(NA, NA, NA),
    c(0.1, 0.5, 0.9),
    varying = TRUE
  ),
  evaluate_case(
    "poisson_varying_right", "PO", "right",
    NA, c(3, 6, 9),
    c(0, 4, 8),
    c(0.8, 3.7, 7.0),
    c(NA, NA, NA),
    c(0.15, 0.55, 0.85),
    varying = TRUE
  ),
  evaluate_case(
    "poisson_varying_both", "PO", "both",
    c(0, 1, 3), c(4, 7, 10),
    c(1, 5, 8),
    c(1.1, 4.2, 7.5),
    c(NA, NA, NA),
    c(0.2, 0.6, 0.8),
    varying = TRUE
  )
)

write_references <- function(data, path) {
  reference_connection <- file(path, open = "wb")
  on.exit(close(reference_connection))
  write.csv(
    data,
    reference_connection,
    row.names = FALSE,
    na = ""
  )
}

if (check_only) {
  temporary_path <- tempfile(fileext = ".csv")
  write_references(references, temporary_path)
  actual <- read.csv(
    temporary_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  expected <- read.csv(
    reference_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!identical(names(actual), names(expected)) ||
      nrow(actual) != nrow(expected)) {
    stop("Truncated reference dimensions or columns differ")
  }
  for (column in names(actual)) {
    if (!identical(is.na(actual[[column]]), is.na(expected[[column]]))) {
      stop("Truncated reference NA pattern differs for ", column)
    }
    if (is.numeric(actual[[column]]) && is.numeric(expected[[column]])) {
      present <- !is.na(actual[[column]])
      difference <- abs(actual[[column]][present] - expected[[column]][present])
      allowed <- 1e-7 * (1 + abs(expected[[column]][present]))
      if (any(difference > allowed)) {
        stop("Truncated numeric parity differs for ", column)
      }
    } else if (!identical(actual[[column]], expected[[column]])) {
      stop("Truncated reference values differ for ", column)
    }
  }
  message(
    "Truncated-family R parity checks passed with gamlss.tr ",
    packageVersion("gamlss.tr")
  )
} else {
  write_references(references, reference_path)
  message(
    "Wrote truncated-family references from gamlss.tr ",
    packageVersion("gamlss.tr")
  )
}
