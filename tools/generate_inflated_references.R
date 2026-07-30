args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(gamlss.inf))
suppressPackageStartupMessages(library(gamlss))

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
family_reference_path <- file.path(
  reference_dir,
  "inflated_family_reference.csv"
)
generic_reference_path <- file.path(
  reference_dir,
  "point_mass_reference.csv"
)
fit_data_path <- file.path(reference_dir, "inflated_fit_data.csv")
fit_reference_path <- file.path(
  reference_dir,
  "inflated_fit_reference.csv"
)

call_family_method <- function(callable, arguments, row_count) {
  if (is.null(callable)) {
    return(rep(NA_real_, row_count))
  }
  formal_names <- names(formals(callable))
  if (!"..." %in% formal_names) {
    arguments <- arguments[names(arguments) %in% formal_names]
  }
  vapply(
    seq_len(row_count),
    function(index) {
      row_arguments <- lapply(
        arguments,
        function(argument) {
          if (length(argument) == row_count) argument[index]
          else argument
        }
      )
      value <- as.numeric(do.call(callable, row_arguments))
      if (length(value) == 0) 0 else value[[1]]
    },
    numeric(1)
  )
}

initial_values <- function(family, y, parameter) {
  initial <- family[[paste0(parameter, ".initial")]]
  if (is.null(initial)) {
    return(rep(NA_real_, length(y)))
  }
  environment <- new.env(parent = globalenv())
  environment$y <- y
  eval(initial, envir = environment)
  as.numeric(get(parameter, envir = environment))
}

empty_parameter <- function(row_count) {
  rep(NA_real_, row_count)
}

evaluate_family <- function(code, y, mu, sigma, probability, nu = NULL,
                            tau = NULL) {
  family <- get(code, envir = asNamespace("gamlss.dist"))()
  row_count <- length(y)
  supplied <- list(mu = mu, sigma = sigma, nu = nu, tau = tau)
  parameter_names <- names(family$parameters)
  parameters <- supplied[parameter_names]
  arguments <- c(list(y = y), parameters)
  distribution_arguments <- c(list(x = y), parameters, list(log = TRUE))
  cdf_arguments <- c(list(q = y), parameters)
  quantile_arguments <- c(list(p = probability), parameters)

  density <- get(
    paste0("d", code),
    envir = asNamespace("gamlss.dist")
  )
  cdf <- get(
    paste0("p", code),
    envir = asNamespace("gamlss.dist")
  )
  quantile <- get(
    paste0("q", code),
    envir = asNamespace("gamlss.dist")
  )

  derivative <- function(name) {
    call_family_method(family[[name]], arguments, row_count)
  }
  parameter_column <- function(name) {
    if (name %in% parameter_names) parameters[[name]]
    else empty_parameter(row_count)
  }
  link_column <- function(name) {
    if (!name %in% parameter_names) {
      return(empty_parameter(row_count))
    }
    as.numeric(family[[paste0(name, ".linkfun")]](parameters[[name]]))
  }
  initial_column <- function(name) {
    if (!name %in% parameter_names) {
      return(empty_parameter(row_count))
    }
    initial_values(family, y, name)
  }

  data.frame(
    family = code,
    y = y,
    mu = parameter_column("mu"),
    sigma = parameter_column("sigma"),
    nu = parameter_column("nu"),
    tau = parameter_column("tau"),
    probability = probability,
    eta_mu = link_column("mu"),
    eta_sigma = link_column("sigma"),
    eta_nu = link_column("nu"),
    eta_tau = link_column("tau"),
    log_density = call_family_method(
      density,
      distribution_arguments,
      row_count
    ),
    cdf = call_family_method(cdf, cdf_arguments, row_count),
    quantile = call_family_method(
      quantile,
      quantile_arguments,
      row_count
    ),
    dldmu = derivative("dldm"),
    dldsigma = derivative("dldd"),
    dldnu = derivative("dldv"),
    dldtau = derivative("dldt"),
    d2ldmu2 = derivative("d2ldm2"),
    d2ldsigma2 = derivative("d2ldd2"),
    d2ldnu2 = derivative("d2ldv2"),
    d2ldtau2 = derivative("d2ldt2"),
    d2ldmudsigma = derivative("d2ldmdd"),
    d2ldmunu = derivative("d2ldmdv"),
    d2ldmutau = derivative("d2ldmdt"),
    d2ldsigmanu = derivative("d2ldddv"),
    d2ldsigmatau = derivative("d2ldddt"),
    d2ldnutau = derivative("d2ldvdt"),
    initial_mu = initial_column("mu"),
    initial_sigma = initial_column("sigma"),
    initial_nu = initial_column("nu"),
    initial_tau = initial_column("tau"),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
    check.names = FALSE
  )
}

family_references <- rbind(
  evaluate_family(
    "ZIP",
    c(0, 1, 3, 7),
    c(0.4, 1.2, 3.5, 6.0),
    c(0.05, 0.2, 0.4, 0.7),
    c(0.05, 0.3, 0.7, 0.95)
  ),
  evaluate_family(
    "ZINBI",
    c(0, 1, 3, 7),
    c(0.6, 1.5, 3.2, 6.5),
    c(0.15, 0.35, 0.8, 1.4),
    c(0.04, 0.25, 0.65, 0.93),
    nu = c(0.08, 0.2, 0.45, 0.65)
  ),
  evaluate_family(
    "BEZI",
    c(0, 0.1, 0.45, 0.9),
    c(0.2, 0.35, 0.6, 0.8),
    c(0.7, 2.0, 5.0, 12.0),
    c(0.03, 0.3, 0.7, 0.96),
    nu = c(0.08, 0.2, 0.35, 0.6)
  ),
  evaluate_family(
    "BEOI",
    c(0.1, 0.45, 0.9, 1),
    c(0.2, 0.4, 0.65, 0.85),
    c(0.8, 2.5, 6.0, 10.0),
    c(0.04, 0.35, 0.72, 0.95),
    nu = c(0.12, 0.25, 0.4, 0.7)
  ),
  evaluate_family(
    "BEINF",
    c(0, 0.2, 0.75, 1),
    c(0.2, 0.4, 0.65, 0.8),
    c(0.25, 0.4, 0.55, 0.7),
    c(0.03, 0.3, 0.7, 0.97),
    nu = c(0.08, 0.2, 0.5, 1.2),
    tau = c(0.12, 0.35, 0.7, 1.5)
  ),
  evaluate_family(
    "BEINF0",
    c(0, 0.2, 0.75, 0.95),
    c(0.2, 0.4, 0.65, 0.8),
    c(0.25, 0.4, 0.55, 0.7),
    c(0.03, 0.3, 0.7, 0.97),
    nu = c(0.08, 0.2, 0.5, 1.2)
  ),
  evaluate_family(
    "BEINF1",
    c(0.05, 0.2, 0.75, 1),
    c(0.2, 0.4, 0.65, 0.8),
    c(0.25, 0.4, 0.55, 0.7),
    c(0.03, 0.3, 0.7, 0.97),
    nu = c(0.08, 0.2, 0.5, 1.2)
  )
)

evaluate_generic <- function(case, type, y, mu, sigma, probability,
                             xi0 = NULL, xi1 = NULL, uniforms) {
  density <- Inf0to1.d("BE", type.of.Inflation = type)
  cdf <- Inf0to1.p("BE", type.of.Inflation = type)
  quantile <- Inf0to1.q("BE", type.of.Inflation = type)
  row_count <- length(y)
  parameters <- list(mu = mu, sigma = sigma)
  if (!is.null(xi0)) parameters$xi0 <- xi0
  if (!is.null(xi1)) parameters$xi1 <- xi1
  cdf_value <- call_family_method(
    cdf,
    c(list(q = y), parameters),
    row_count
  )
  mass_zero <- rep(0, row_count)
  mass_one <- rep(0, row_count)
  if (type == "Zero") mass_zero <- xi0
  if (type == "One") mass_one <- xi1
  if (type == "Zero&One") {
    denominator <- 1 + xi0 + xi1
    mass_zero <- xi0 / denominator
    mass_one <- xi1 / denominator
  }
  jump <- ifelse(y == 0, mass_zero, ifelse(y == 1, mass_one, 0))
  cdf_left <- cdf_value - jump
  randomized_probability <- cdf_left + uniforms * jump

  data.frame(
    case = case,
    type = type,
    y = y,
    mu = mu,
    sigma = sigma,
    xi0 = if (is.null(xi0)) empty_parameter(row_count) else xi0,
    xi1 = if (is.null(xi1)) empty_parameter(row_count) else xi1,
    probability = probability,
    uniform = uniforms,
    log_density = call_family_method(
      density,
      c(list(x = y), parameters, list(log = TRUE)),
      row_count
    ),
    cdf = cdf_value,
    cdf_left = cdf_left,
    quantile = call_family_method(
      quantile,
      c(list(p = probability), parameters),
      row_count
    ),
    randomized_probability = randomized_probability,
    randomized_residual = qnorm(randomized_probability),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
    gamlss_inf_version = as.character(packageVersion("gamlss.inf")),
    check.names = FALSE
  )
}

generic_references <- rbind(
  evaluate_generic(
    "beta_zero_probability",
    "Zero",
    c(0, 0.15, 0.55, 0.9),
    c(0.2, 0.35, 0.6, 0.8),
    c(0.25, 0.4, 0.55, 0.7),
    c(0.03, 0.3, 0.7, 0.96),
    xi0 = c(0.08, 0.2, 0.35, 0.6),
    uniforms = c(0.13, 0.37, 0.61, 0.89)
  ),
  evaluate_generic(
    "beta_one_probability",
    "One",
    c(0.1, 0.35, 0.8, 1),
    c(0.2, 0.4, 0.65, 0.85),
    c(0.25, 0.4, 0.55, 0.7),
    c(0.04, 0.32, 0.73, 0.95),
    xi1 = c(0.1, 0.25, 0.4, 0.7),
    uniforms = c(0.17, 0.41, 0.67, 0.91)
  ),
  evaluate_generic(
    "beta_zero_one_odds",
    "Zero&One",
    c(0, 0.2, 0.75, 1),
    c(0.2, 0.4, 0.65, 0.8),
    c(0.25, 0.4, 0.55, 0.7),
    c(0.02, 0.3, 0.7, 0.98),
    xi0 = c(0.08, 0.2, 0.5, 1.2),
    xi1 = c(0.12, 0.35, 0.7, 1.5),
    uniforms = c(0.11, 0.43, 0.71, 0.93)
  )
)

interior_beta <- seq(0.05, 0.95, length.out = 20)
fit_responses <- list(
  ZIP = c(
    rep(0, 35),
    rep(1, 15),
    rep(2, 20),
    rep(3, 12),
    rep(5, 8),
    rep(8, 5)
  ),
  ZINBI = c(
    rep(0, 40),
    rep(1, 12),
    rep(2, 10),
    rep(4, 10),
    rep(8, 8),
    rep(15, 5)
  ),
  BEZI = c(rep(0, 20), rep(interior_beta, each = 4)),
  BEOI = c(rep(interior_beta, each = 4), rep(1, 20)),
  BEINF = c(
    rep(0, 15),
    rep(interior_beta, each = 3),
    rep(1, 20)
  ),
  BEINF0 = c(rep(0, 20), rep(interior_beta, each = 4)),
  BEINF1 = c(rep(interior_beta, each = 4), rep(1, 20))
)
fit_data <- do.call(
  rbind,
  lapply(
    names(fit_responses),
    function(code) {
      response <- fit_responses[[code]]
      data.frame(
        family = code,
        observation = seq_along(response),
        y = response
      )
    }
  )
)
row.names(fit_data) <- NULL

fit_family <- function(code) {
  family <- get(code, envir = asNamespace("gamlss.dist"))()
  data <- fit_data[fit_data$family == code, , drop = FALSE]
  arguments <- list(
    formula = y ~ 1,
    family = family,
    data = data,
    trace = FALSE,
    control = gamlss.control(
      n.cyc = 500,
      c.crit = 1e-10,
      trace = FALSE
    )
  )
  for (parameter in setdiff(names(family$parameters), "mu")) {
    arguments[[paste0(parameter, ".formula")]] <- ~ 1
  }
  fit <- do.call(gamlss, arguments)
  coefficient <- function(parameter) {
    if (!parameter %in% names(family$parameters)) return(NA_real_)
    as.numeric(fit[[paste0(parameter, ".coefficients")]][[1]])
  }
  fitted_parameter <- function(parameter) {
    if (!parameter %in% names(family$parameters)) return(NA_real_)
    as.numeric(fitted(fit, parameter)[[1]])
  }
  data.frame(
    family = code,
    observation_count = nrow(data),
    converged = fit$converged,
    iterations = fit$iter,
    global_deviance = fit$G.deviance,
    coefficient_mu = coefficient("mu"),
    coefficient_sigma = coefficient("sigma"),
    coefficient_nu = coefficient("nu"),
    coefficient_tau = coefficient("tau"),
    fitted_mu = fitted_parameter("mu"),
    fitted_sigma = fitted_parameter("sigma"),
    fitted_nu = fitted_parameter("nu"),
    fitted_tau = fitted_parameter("tau"),
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
    check.names = FALSE
  )
}

fit_references <- do.call(
  rbind,
  lapply(names(fit_responses), fit_family)
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

check_reference <- function(actual, path, label) {
  temporary_path <- tempfile(fileext = ".csv")
  write_references(actual, temporary_path)
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
    if (!identical(
      is.na(generated[[column]]),
      is.na(expected[[column]])
    )) {
      stop(label, " reference NA pattern differs for ", column)
    }
    if (is.numeric(generated[[column]]) &&
        is.numeric(expected[[column]])) {
      present <- !is.na(generated[[column]])
      difference <- abs(
        generated[[column]][present] - expected[[column]][present]
      )
      allowed <- 1e-7 * (
        1 + abs(expected[[column]][present])
      )
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
  check_reference(
    family_references,
    family_reference_path,
    "Inflated-family"
  )
  check_reference(
    generic_references,
    generic_reference_path,
    "Generic point-mass"
  )
  check_reference(fit_data, fit_data_path, "Inflated fit data")
  check_reference(
    fit_references,
    fit_reference_path,
    "Inflated fit"
  )
  message(
    "Inflated and point-mass R parity checks passed with gamlss.dist ",
    packageVersion("gamlss.dist"),
    " and gamlss.inf ",
    packageVersion("gamlss.inf")
  )
} else {
  write_references(family_references, family_reference_path)
  write_references(generic_references, generic_reference_path)
  write_references(fit_data, fit_data_path)
  write_references(fit_references, fit_reference_path)
  message(
    "Wrote inflated and point-mass references from gamlss.dist ",
    packageVersion("gamlss.dist"),
    " and gamlss.inf ",
    packageVersion("gamlss.inf")
  )
}
