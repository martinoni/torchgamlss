args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.cens))
suppressPackageStartupMessages(library(survival))

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
family_reference_path <- file.path(
  reference_dir,
  "survival_family_reference.csv"
)
censored_reference_path <- file.path(
  reference_dir,
  "censored_reference.csv"
)

call_family_method <- function(callable, arguments) {
  formal_names <- names(formals(callable))
  if (!"..." %in% formal_names) {
    arguments <- arguments[names(arguments) %in% formal_names]
  }
  as.numeric(do.call(callable, arguments))
}

call_distribution_function <- function(
    callable,
    value,
    parameters,
    extra = list()) {
  value_name <- names(formals(callable))[1]
  arguments <- c(
    setNames(list(value), value_name),
    parameters,
    extra
  )
  call_family_method(callable, arguments)
}

call_distribution_rows <- function(
    callable,
    values,
    parameters,
    extra = list()) {
  vapply(
    seq_along(values),
    function(index) {
      row_parameters <- lapply(
        parameters,
        function(parameter) parameter[index]
      )
      call_distribution_function(
        callable,
        values[index],
        row_parameters,
        extra
      )
    },
    numeric(1)
  )
}

call_response_rows <- function(
    callable,
    response,
    parameters,
    extra = list()) {
  row_count <- if (is.matrix(response)) nrow(response) else length(response)
  vapply(
    seq_len(row_count),
    function(index) {
      row_response <- if (is.matrix(response)) {
        response[index, , drop = FALSE]
      } else {
        response[index]
      }
      row_parameters <- lapply(
        parameters,
        function(parameter) parameter[index]
      )
      call_distribution_function(
        callable,
        row_response,
        row_parameters,
        extra
      )
    },
    numeric(1)
  )
}

call_optional_family_method <- function(
    family_object,
    method,
    arguments) {
  callable <- family_object[[method]]
  if (is.null(callable)) {
    return(rep(NA_real_, length(arguments$y)))
  }
  call_family_method(callable, arguments)
}

call_optional_response_rows <- function(
    family_object,
    method,
    response,
    parameters) {
  callable <- family_object[[method]]
  if (is.null(callable)) {
    row_count <- if (is.matrix(response)) nrow(response) else length(response)
    return(rep(NA_real_, row_count))
  }
  call_response_rows(callable, response, parameters)
}

call_parameter_rows <- function(callable, parameters, response = NULL) {
  row_count <- length(parameters[[1]])
  vapply(
    seq_len(row_count),
    function(index) {
      row_parameters <- lapply(
        parameters,
        function(parameter) parameter[index]
      )
      arguments <- row_parameters
      if (!is.null(response)) {
        arguments <- c(list(y = response[index]), arguments)
      }
      call_family_method(callable, arguments)
    },
    numeric(1)
  )
}

call_optional_parameter_rows <- function(
    family_object,
    method,
    parameters,
    response = NULL) {
  callable <- family_object[[method]]
  if (is.null(callable)) {
    return(rep(NA_real_, length(parameters[[1]])))
  }
  call_parameter_rows(callable, parameters, response)
}

evaluate_family <- function(
    family,
    y,
    mu,
    sigma,
    probability,
    nu = rep(NA_real_, length(y))) {
  family_object <- get(family)()
  density <- get(paste0("d", family))
  distribution <- get(paste0("p", family))
  quantile <- get(paste0("q", family))
  parameters <- list(mu = mu, sigma = sigma)
  if (family == "GG") {
    parameters$nu <- nu
  }
  arguments <- c(list(y = y), parameters)
  log_y <- log(y)

  if (family == "WEI") {
    variance_log_y <- var(log_y)
    initial_sigma <- rep(
      1.283 / sqrt(variance_log_y),
      length(y)
    )
    initial_mu <- exp(log_y + 0.5772 / initial_sigma)
    initial_nu <- rep(NA_real_, length(y))
  } else if (family == "LOGNO") {
    initial_mu <- (log_y + mean(log_y)) / 2
    initial_sigma <- rep(sd(log_y), length(y))
    initial_nu <- rep(NA_real_, length(y))
  } else if (family == "IG") {
    initial_mu <- (y + mean(y)) / 2
    initial_sigma <- rep(sd(y) / mean(y)^1.5, length(y))
    initial_nu <- rep(NA_real_, length(y))
  } else {
    initial_mu <- (y + mean(y)) / 2
    initial_sigma <- rep(1, length(y))
    initial_nu <- rep(1, length(y))
  }

  log_density <- call_distribution_rows(
    density,
    y,
    parameters,
    list(log = TRUE)
  )
  cdf <- call_distribution_rows(distribution, y, parameters)
  survival <- call_distribution_rows(
    distribution,
    y,
    parameters,
    list(lower.tail = FALSE)
  )

  data.frame(
    family = family,
    y = y,
    mu = mu,
    sigma = sigma,
    nu = nu,
    probability = probability,
    log_density = log_density,
    cdf = cdf,
    survival = survival,
    hazard = exp(log_density) / survival,
    cumulative_hazard = -call_distribution_rows(
      distribution,
      y,
      parameters,
      list(lower.tail = FALSE, log.p = TRUE)
    ),
    quantile = call_distribution_rows(
      quantile,
      probability,
      parameters
    ),
    dldmu = call_family_method(family_object$dldm, arguments),
    dldsigma = call_family_method(family_object$dldd, arguments),
    dldnu = call_optional_family_method(
      family_object,
      "dldv",
      arguments
    ),
    d2ldmu2 = call_family_method(family_object$d2ldm2, arguments),
    d2ldsigma2 = call_family_method(family_object$d2ldd2, arguments),
    d2ldnu2 = call_optional_family_method(
      family_object,
      "d2ldv2",
      arguments
    ),
    d2ldmudsigma = call_family_method(
      family_object$d2ldmdd,
      arguments
    ),
    d2ldmudnu = call_optional_family_method(
      family_object,
      "d2ldmdv",
      arguments
    ),
    d2ldsigmadnu = call_optional_family_method(
      family_object,
      "d2ldddv",
      arguments
    ),
    mean = call_family_method(family_object$mean, parameters),
    variance = call_family_method(family_object$variance, parameters),
    initial_mu = initial_mu,
    initial_sigma = initial_sigma,
    initial_nu = initial_nu,
    gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
    check.names = FALSE
  )
}

family_references <- rbind(
  evaluate_family(
    "WEI",
    c(0.25, 0.8, 2.0, 5.5),
    c(0.6, 1.2, 2.8, 4.5),
    c(0.7, 1.2, 2.0, 3.5),
    c(0.1, 0.35, 0.7, 0.9)
  ),
  evaluate_family(
    "LOGNO",
    c(0.2, 0.9, 2.5, 7.0),
    c(-0.8, -0.1, 0.8, 1.5),
    c(0.35, 0.6, 0.9, 1.2),
    c(0.1, 0.35, 0.7, 0.9)
  ),
  evaluate_family(
    "IG",
    c(0.2, 0.9, 2.5, 6.0),
    c(0.5, 1.2, 2.8, 4.0),
    c(0.5, 0.7, 0.4, 0.3),
    c(0.1, 0.35, 0.7, 0.9)
  ),
  evaluate_family(
    "GG",
    c(0.2, 0.9, 2.5, 6.0),
    c(0.6, 1.1, 2.5, 4.5),
    c(0.45, 0.7, 0.55, 0.4),
    c(0.1, 0.35, 0.7, 0.9),
    c(-0.8, -0.35, 0.7, 1.4)
  )
)

evaluate_censored_case <- function(
    case,
    family,
    type,
    response,
    mu,
    sigma,
    nu = rep(NA_real_, length(mu))) {
  density <- cens.d(family = family, type = type)
  family_constructor <- cens(
    family = family,
    type = type,
    local = FALSE
  )
  family_object <- family_constructor()
  observed <- response[, 1]
  raw_status <- response[, "status"]
  status <- if (type == "left") {
    ifelse(raw_status == 1, 1, 2)
  } else {
    raw_status
  }
  upper <- ifelse(status == 3, response[, 2], NA_real_)
  parameters <- list(mu = mu, sigma = sigma)
  if (family == "GG") {
    parameters$nu <- nu
  }
  base_family <- get(family)()

  data.frame(
    case = case,
    family = family,
    type = type,
    status = status,
    observed = observed,
    upper = upper,
    mu = mu,
    sigma = sigma,
    nu = nu,
    log_likelihood = call_response_rows(
      density,
      response,
      parameters,
      list(log = TRUE)
    ),
    dldmu = call_response_rows(
      family_object$dldm,
      response,
      parameters
    ),
    dldsigma = call_response_rows(
      family_object$dldd,
      response,
      parameters
    ),
    dldnu = call_optional_response_rows(
      family_object,
      "dldv",
      response,
      parameters
    ),
    d2ldmu2 = call_parameter_rows(
      base_family$d2ldm2,
      parameters,
      observed
    ),
    d2ldsigma2 = call_parameter_rows(
      base_family$d2ldd2,
      parameters,
      observed
    ),
    d2ldnu2 = call_optional_parameter_rows(
      base_family,
      "d2ldv2",
      parameters,
      observed
    ),
    d2ldmudsigma = call_parameter_rows(
      base_family$d2ldmdd,
      parameters,
      observed
    ),
    d2ldmudnu = call_optional_parameter_rows(
      base_family,
      "d2ldmdv",
      parameters,
      observed
    ),
    d2ldsigmadnu = call_optional_parameter_rows(
      base_family,
      "d2ldddv",
      parameters,
      observed
    ),
    gamlss_cens_version = as.character(packageVersion("gamlss.cens")),
    check.names = FALSE
  )
}

censored_references <- rbind(
  evaluate_censored_case(
    "weibull_right",
    "WEI",
    "right",
    Surv(
      c(0.8, 1.5, 3.0, 5.0),
      c(1, 0, 1, 0),
      type = "right"
    ),
    c(1.0, 2.0, 3.5, 4.0),
    c(1.2, 1.5, 0.9, 2.0)
  ),
  evaluate_censored_case(
    "weibull_left",
    "WEI",
    "left",
    Surv(
      c(0.4, 1.1, 2.2, 4.0),
      c(0, 1, 0, 1),
      type = "left"
    ),
    c(0.8, 1.4, 2.8, 3.5),
    c(0.9, 1.3, 1.7, 2.2)
  ),
  evaluate_censored_case(
    "weibull_interval",
    "WEI",
    "interval",
    Surv(
      c(0.7, 1.2, NA, 2.5),
      c(0.7, Inf, 1.0, 3.8),
      type = "interval2"
    ),
    c(0.9, 1.8, 1.4, 3.2),
    c(1.1, 1.5, 0.8, 2.0)
  ),
  evaluate_censored_case(
    "lognormal_right",
    "LOGNO",
    "right",
    Surv(
      c(0.5, 1.4, 3.0, 6.0),
      c(1, 0, 1, 0),
      type = "right"
    ),
    c(-0.4, 0.2, 0.9, 1.4),
    c(0.5, 0.7, 0.9, 1.1)
  ),
  evaluate_censored_case(
    "lognormal_left",
    "LOGNO",
    "left",
    Surv(
      c(0.3, 1.0, 2.0, 5.0),
      c(0, 1, 0, 1),
      type = "left"
    ),
    c(-0.7, 0.05, 0.6, 1.3),
    c(0.4, 0.6, 0.8, 1.0)
  ),
  evaluate_censored_case(
    "lognormal_interval",
    "LOGNO",
    "interval",
    Surv(
      c(0.4, 1.1, NA, 2.8),
      c(0.4, Inf, 0.9, 4.2),
      type = "interval2"
    ),
    c(-0.5, 0.1, -0.1, 1.0),
    c(0.45, 0.65, 0.55, 0.9)
  ),
  evaluate_censored_case(
    "inverse_gaussian_right",
    "IG",
    "right",
    Surv(
      c(0.4, 1.2, 2.7, 5.0),
      c(1, 0, 1, 0),
      type = "right"
    ),
    c(0.7, 1.6, 2.5, 3.8),
    c(0.5, 0.65, 0.45, 0.35)
  ),
  evaluate_censored_case(
    "inverse_gaussian_left",
    "IG",
    "left",
    Surv(
      c(0.25, 0.9, 2.1, 4.2),
      c(0, 1, 0, 1),
      type = "left"
    ),
    c(0.6, 1.2, 2.3, 3.5),
    c(0.55, 0.7, 0.5, 0.4)
  ),
  evaluate_censored_case(
    "inverse_gaussian_interval",
    "IG",
    "interval",
    Surv(
      c(0.3, 1.0, NA, 2.4),
      c(0.5, Inf, 0.8, 3.5),
      type = "interval2"
    ),
    c(0.7, 1.5, 1.1, 3.0),
    c(0.5, 0.6, 0.55, 0.4)
  ),
  evaluate_censored_case(
    "generalized_gamma_right",
    "GG",
    "right",
    Surv(
      c(0.35, 1.1, 2.8, 5.5),
      c(1, 0, 1, 0),
      type = "right"
    ),
    c(0.65, 1.4, 2.6, 4.0),
    c(0.45, 0.65, 0.55, 0.4),
    c(-0.7, -0.3, 0.8, 1.2)
  ),
  evaluate_censored_case(
    "generalized_gamma_left",
    "GG",
    "left",
    Surv(
      c(0.25, 0.85, 2.0, 4.5),
      c(0, 1, 0, 1),
      type = "left"
    ),
    c(0.55, 1.15, 2.2, 3.7),
    c(0.4, 0.7, 0.5, 0.45),
    c(-0.6, -0.25, 0.65, 1.1)
  ),
  evaluate_censored_case(
    "generalized_gamma_interval",
    "GG",
    "interval",
    Surv(
      c(0.3, 0.95, NA, 2.6),
      c(0.55, Inf, 0.75, 3.9),
      type = "interval2"
    ),
    c(0.6, 1.3, 1.0, 3.2),
    c(0.45, 0.65, 0.5, 0.4),
    c(-0.75, -0.35, 0.75, 1.3)
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
    "Survival-family"
  )
  check_reference(
    censored_references,
    censored_reference_path,
    "Censored-family"
  )
  message(
    "Survival and censoring R parity checks passed with gamlss.cens ",
    packageVersion("gamlss.cens")
  )
} else {
  write_references(family_references, family_reference_path)
  write_references(censored_references, censored_reference_path)
  message(
    "Wrote survival and censoring references from gamlss.cens ",
    packageVersion("gamlss.cens")
  )
}
