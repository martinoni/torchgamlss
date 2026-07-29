args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.tr))

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
reference_path <- file.path(reference_dir, "truncated_reference.csv")

parameter_order <- c("mu", "sigma", "nu", "tau")
score_methods <- c(
  mu = "dldm",
  sigma = "dldd",
  nu = "dldv",
  tau = "dldt"
)
second_methods <- c(
  mu_mu = "d2ldm2",
  sigma_sigma = "d2ldd2",
  nu_nu = "d2ldv2",
  tau_tau = "d2ldt2",
  mu_sigma = "d2ldmdd",
  mu_nu = "d2ldmdv",
  mu_tau = "d2ldmdt",
  sigma_nu = "d2ldddv",
  sigma_tau = "d2ldddt",
  nu_tau = "d2ldvdt"
)

evaluate_case <- function(
    case,
    family,
    type,
    lower,
    upper,
    y,
    probability,
    parameters,
    varying = FALSE) {
  observation_count <- length(y)
  if (!identical(names(parameters), parameter_order[
    parameter_order %in% names(parameters)
  ])) {
    stop("Parameters must follow mu, sigma, nu, tau order for ", case)
  }
  if (any(vapply(
    parameters,
    function(value) length(value) != observation_count,
    logical(1)
  ))) {
    stop("Every parameter must have one value per observation for ", case)
  }

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

  row_arguments <- function(arguments, index) {
    lapply(arguments, function(value) {
      if (length(value) == 1) value else value[index]
    })
  }
  normalize_result <- function(value, label) {
    value <- as.numeric(value)
    if (length(value) == 1) {
      return(rep(value, observation_count))
    }
    if (length(value) != observation_count) {
      stop(label, " returned ", length(value), " values for ", case)
    }
    value
  }
  evaluate_callable <- function(callable, arguments, label) {
    if (varying) {
      return(normalize_result(do.call(callable, arguments), label))
    }
    vapply(
      seq_len(observation_count),
      function(index) {
        as.numeric(do.call(callable, row_arguments(arguments, index)))
      },
      numeric(1)
    )
  }
  evaluate_family_method <- function(method, required_parameters) {
    if (!all(required_parameters %in% names(parameters))) {
      return(rep(NA_real_, observation_count))
    }
    callable <- family_object[[method]]
    if (!is.function(callable)) {
      return(rep(NA_real_, observation_count))
    }
    arguments <- c(list(y = y), parameters)
    formal_names <- names(formals(callable))
    if (!"..." %in% formal_names) {
      arguments <- arguments[names(arguments) %in% formal_names]
    }
    evaluate_callable(callable, arguments, method)
  }

  density_values <- evaluate_callable(
    density,
    c(list(x = y), parameters, list(log = TRUE)),
    "density"
  )
  cdf_values <- evaluate_callable(
    probability_function,
    c(list(q = y), parameters),
    "CDF"
  )
  quantile_values <- evaluate_callable(
    quantile_function,
    c(list(p = probability), parameters),
    "quantile"
  )

  scores <- lapply(parameter_order, function(parameter) {
    evaluate_family_method(score_methods[[parameter]], parameter)
  })
  names(scores) <- parameter_order

  second_parameter_pairs <- list(
    mu_mu = c("mu"),
    sigma_sigma = c("sigma"),
    nu_nu = c("nu"),
    tau_tau = c("tau"),
    mu_sigma = c("mu", "sigma"),
    mu_nu = c("mu", "nu"),
    mu_tau = c("mu", "tau"),
    sigma_nu = c("sigma", "nu"),
    sigma_tau = c("sigma", "tau"),
    nu_tau = c("nu", "tau")
  )
  second <- lapply(names(second_methods), function(pair) {
    evaluate_family_method(
      second_methods[[pair]],
      second_parameter_pairs[[pair]]
    )
  })
  names(second) <- names(second_methods)

  parameter_column <- function(parameter) {
    if (parameter %in% names(parameters)) {
      parameters[[parameter]]
    } else {
      rep(NA_real_, observation_count)
    }
  }
  bound_column <- function(bound) {
    if (all(is.na(bound))) {
      rep("", observation_count)
    } else {
      format(bound, scientific = FALSE, trim = TRUE)
    }
  }

  data.frame(
    case = case,
    family = family,
    type = type,
    varying = varying,
    lower = bound_column(lower),
    upper = bound_column(upper),
    y = y,
    mu = parameter_column("mu"),
    sigma = parameter_column("sigma"),
    nu = parameter_column("nu"),
    tau = parameter_column("tau"),
    probability = probability,
    log_density = density_values,
    cdf = cdf_values,
    quantile = quantile_values,
    dldmu = scores$mu,
    dldsigma = scores$sigma,
    dldnu = scores$nu,
    dldtau = scores$tau,
    d2ldmu2 = second$mu_mu,
    d2ldsigma2 = second$sigma_sigma,
    d2ldnu2 = second$nu_nu,
    d2ldtau2 = second$tau_tau,
    d2ldmudsigma = second$mu_sigma,
    d2ldmudnu = second$mu_nu,
    d2ldmudtau = second$mu_tau,
    d2ldsigmadnu = second$sigma_nu,
    d2ldsigmadtau = second$sigma_tau,
    d2ldnudtau = second$nu_tau,
    gamlss_tr_version = as.character(packageVersion("gamlss.tr")),
    check.names = FALSE
  )
}

references <- rbind(
  evaluate_case(
    "normal_left", "NO", "left", 0, NA,
    c(0.1, 0.8, 2.1), c(0.1, 0.5, 0.9),
    list(
      mu = c(-0.5, 0.4, 1.1),
      sigma = c(0.6, 1.0, 1.4)
    )
  ),
  evaluate_case(
    "normal_right", "NO", "right", NA, 1.5,
    c(-1.0, 0.2, 1.4), c(0.15, 0.55, 0.85),
    list(
      mu = c(-0.6, 0.5, 1.0),
      sigma = c(0.7, 1.1, 0.9)
    )
  ),
  evaluate_case(
    "normal_both", "NO", "both", -1, 2,
    c(-0.8, 0.5, 1.7), c(0.2, 0.6, 0.8),
    list(
      mu = c(-0.3, 0.4, 1.2),
      sigma = c(0.8, 1.3, 0.7)
    )
  ),
  evaluate_case(
    "poisson_left", "PO", "left", 0, NA,
    c(1, 2, 5), c(0.1, 0.5, 0.9),
    list(mu = c(0.7, 2.5, 5.0))
  ),
  evaluate_case(
    "poisson_right", "PO", "right", NA, 6,
    c(0, 2, 5), c(0.15, 0.55, 0.85),
    list(mu = c(0.8, 2.7, 5.5))
  ),
  evaluate_case(
    "poisson_both", "PO", "both", 0, 6,
    c(1, 3, 5), c(0.2, 0.6, 0.8),
    list(mu = c(0.9, 3.2, 6.0))
  ),
  evaluate_case(
    "normal_varying_left", "NO", "left",
    c(-0.5, 0.2, 1.0), NA,
    c(-0.2, 0.8, 1.7), c(0.1, 0.5, 0.9),
    list(
      mu = c(-0.6, 0.4, 1.2),
      sigma = c(0.7, 1.0, 0.8)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "normal_varying_right", "NO", "right",
    NA, c(0.2, 1.3, 2.4),
    c(-0.8, 0.6, 2.0), c(0.15, 0.55, 0.85),
    list(
      mu = c(-0.4, 0.7, 1.5),
      sigma = c(0.9, 0.8, 1.1)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "normal_varying_both", "NO", "both",
    c(-1.2, -0.1, 0.8), c(0.4, 1.4, 2.8),
    c(-0.5, 0.7, 2.2), c(0.2, 0.6, 0.8),
    list(
      mu = c(-0.4, 0.5, 1.7),
      sigma = c(0.8, 1.0, 0.7)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "poisson_varying_left", "PO", "left",
    c(0, 1, 3), NA,
    c(1, 3, 6), c(0.1, 0.5, 0.9),
    list(mu = c(0.9, 2.8, 5.5)),
    varying = TRUE
  ),
  evaluate_case(
    "poisson_varying_right", "PO", "right",
    NA, c(3, 6, 9),
    c(0, 4, 8), c(0.15, 0.55, 0.85),
    list(mu = c(0.8, 3.7, 7.0)),
    varying = TRUE
  ),
  evaluate_case(
    "poisson_varying_both", "PO", "both",
    c(0, 1, 3), c(4, 7, 10),
    c(1, 5, 8), c(0.2, 0.6, 0.8),
    list(mu = c(1.1, 4.2, 7.5)),
    varying = TRUE
  ),
  evaluate_case(
    "gamma_both", "GA", "both", 0.2, 5.0,
    c(0.4, 1.5, 4.2), c(0.2, 0.6, 0.85),
    list(
      mu = c(0.8, 1.8, 3.2),
      sigma = c(0.35, 0.6, 0.9)
    )
  ),
  evaluate_case(
    "gamma_varying_both", "GA", "both",
    c(0.1, 0.4, 0.8), c(1.5, 3.0, 6.0),
    c(0.3, 1.2, 4.5), c(0.15, 0.5, 0.9),
    list(
      mu = c(0.7, 1.6, 3.8),
      sigma = c(0.3, 0.55, 0.8)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "nbi_both", "NBI", "both", 0, 10,
    c(1, 4, 8), c(0.2, 0.6, 0.85),
    list(
      mu = c(1.2, 4.0, 7.0),
      sigma = c(0.2, 0.5, 0.9)
    )
  ),
  evaluate_case(
    "nbi_varying_both", "NBI", "both",
    c(0, 1, 3), c(5, 9, 14),
    c(1, 5, 11), c(0.15, 0.55, 0.9),
    list(
      mu = c(1.0, 4.8, 9.5),
      sigma = c(0.25, 0.6, 1.0)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "beta_both", "BE", "both", 0.05, 0.95,
    c(0.12, 0.5, 0.88), c(0.2, 0.6, 0.85),
    list(
      mu = c(0.2, 0.55, 0.8),
      sigma = c(0.25, 0.4, 0.6)
    )
  ),
  evaluate_case(
    "beta_varying_both", "BE", "both",
    c(0.02, 0.1, 0.2), c(0.55, 0.8, 0.98),
    c(0.1, 0.45, 0.9), c(0.15, 0.5, 0.9),
    list(
      mu = c(0.18, 0.48, 0.82),
      sigma = c(0.2, 0.35, 0.55)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "bccg_both", "BCCG", "both", 0.4, 5.0,
    c(0.7, 2.0, 4.0), c(0.2, 0.6, 0.85),
    list(
      mu = c(1.0, 2.2, 3.5),
      sigma = c(0.2, 0.35, 0.5),
      nu = c(-0.5, 0.3, 1.0)
    )
  ),
  evaluate_case(
    "bccg_varying_both", "BCCG", "both",
    c(0.3, 0.7, 1.2), c(1.5, 3.5, 6.0),
    c(0.8, 2.0, 4.5), c(0.15, 0.5, 0.9),
    list(
      mu = c(0.9, 2.1, 4.0),
      sigma = c(0.25, 0.4, 0.45),
      nu = c(-0.8, 0.0, 0.7)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "tf_both", "TF", "both", -2.0, 3.0,
    c(-1.2, 0.5, 2.2), c(0.2, 0.6, 0.85),
    list(
      mu = c(-0.5, 0.4, 1.0),
      sigma = c(0.7, 1.0, 1.3),
      nu = c(4.0, 7.0, 12.0)
    )
  ),
  evaluate_case(
    "tf_varying_both", "TF", "both",
    c(-2.5, -1.0, 0.0), c(0.5, 2.0, 4.0),
    c(-1.0, 0.8, 2.8), c(0.15, 0.5, 0.9),
    list(
      mu = c(-0.7, 0.5, 1.7),
      sigma = c(0.8, 0.9, 1.1),
      nu = c(5.0, 9.0, 15.0)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "pe_both", "PE", "both", -2.0, 3.0,
    c(-1.1, 0.4, 2.3), c(0.2, 0.6, 0.85),
    list(
      mu = c(-0.4, 0.5, 1.1),
      sigma = c(0.7, 1.0, 1.2),
      nu = c(1.2, 2.0, 3.0)
    )
  ),
  evaluate_case(
    "pe_varying_both", "PE", "both",
    c(-2.5, -1.0, 0.0), c(0.7, 2.2, 4.0),
    c(-1.2, 0.6, 2.9), c(0.15, 0.5, 0.9),
    list(
      mu = c(-0.6, 0.4, 1.8),
      sigma = c(0.8, 0.9, 1.0),
      nu = c(1.4, 2.3, 3.5)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "bct_both", "BCT", "both", 0.4, 6.0,
    c(0.8, 2.2, 4.8), c(0.2, 0.6, 0.85),
    list(
      mu = c(1.1, 2.4, 4.0),
      sigma = c(0.2, 0.35, 0.5),
      nu = c(-0.5, 0.2, 0.8),
      tau = c(4.0, 7.0, 12.0)
    )
  ),
  evaluate_case(
    "bct_varying_both", "BCT", "both",
    c(0.3, 0.8, 1.5), c(1.8, 4.0, 7.0),
    c(0.9, 2.5, 5.5), c(0.15, 0.5, 0.9),
    list(
      mu = c(1.0, 2.6, 4.8),
      sigma = c(0.25, 0.4, 0.45),
      nu = c(-0.7, 0.0, 0.6),
      tau = c(5.0, 8.0, 14.0)
    ),
    varying = TRUE
  ),
  evaluate_case(
    "bcpe_both", "BCPE", "both", 0.4, 6.0,
    c(0.8, 2.2, 4.8), c(0.2, 0.6, 0.85),
    list(
      mu = c(1.1, 2.4, 4.0),
      sigma = c(0.2, 0.35, 0.5),
      nu = c(-0.5, 0.2, 0.8),
      tau = c(1.2, 2.0, 3.0)
    )
  ),
  evaluate_case(
    "bcpe_varying_both", "BCPE", "both",
    c(0.3, 0.8, 1.5), c(1.8, 4.0, 7.0),
    c(0.9, 2.5, 5.5), c(0.15, 0.5, 0.9),
    list(
      mu = c(1.0, 2.6, 4.8),
      sigma = c(0.25, 0.4, 0.45),
      nu = c(-0.7, 0.0, 0.6),
      tau = c(1.4, 2.2, 3.5)
    ),
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
