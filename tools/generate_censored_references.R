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

evaluate_family <- function(
    family,
    y,
    mu,
    sigma,
    probability) {
  family_object <- get(family)()
  density <- get(paste0("d", family))
  distribution <- get(paste0("p", family))
  quantile <- get(paste0("q", family))
  arguments <- list(y = y, mu = mu, sigma = sigma)
  log_y <- log(y)

  if (family == "WEI") {
    variance_log_y <- var(log_y)
    initial_sigma <- rep(
      1.283 / sqrt(variance_log_y),
      length(y)
    )
    initial_mu <- exp(log_y + 0.5772 / initial_sigma)
    mean <- mu * gamma(1 / sigma + 1)
    variance <- mu^2 * (
      gamma(2 / sigma + 1) - gamma(1 / sigma + 1)^2
    )
  } else {
    initial_mu <- (log_y + mean(log_y)) / 2
    initial_sigma <- rep(sd(log_y), length(y))
    mean <- exp(mu + sigma^2 / 2)
    variance <- exp(2 * mu + sigma^2) * (exp(sigma^2) - 1)
  }

  data.frame(
    family = family,
    y = y,
    mu = mu,
    sigma = sigma,
    probability = probability,
    log_density = density(y, mu, sigma, log = TRUE),
    cdf = distribution(y, mu, sigma),
    survival = distribution(y, mu, sigma, lower.tail = FALSE),
    hazard = exp(density(y, mu, sigma, log = TRUE)) /
      distribution(y, mu, sigma, lower.tail = FALSE),
    cumulative_hazard = -distribution(
      y,
      mu,
      sigma,
      lower.tail = FALSE,
      log.p = TRUE
    ),
    quantile = quantile(probability, mu, sigma),
    dldmu = call_family_method(family_object$dldm, arguments),
    dldsigma = call_family_method(family_object$dldd, arguments),
    d2ldmu2 = call_family_method(family_object$d2ldm2, arguments),
    d2ldsigma2 = call_family_method(family_object$d2ldd2, arguments),
    d2ldmudsigma = call_family_method(
      family_object$d2ldmdd,
      arguments
    ),
    mean = mean,
    variance = variance,
    initial_mu = initial_mu,
    initial_sigma = initial_sigma,
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
  )
)

evaluate_censored_case <- function(
    case,
    family,
    type,
    response,
    mu,
    sigma) {
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
  score_arguments <- list(
    y = response,
    mu = mu,
    sigma = sigma
  )
  base_arguments <- list(
    y = observed,
    mu = mu,
    sigma = sigma
  )
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
    log_likelihood = density(
      response,
      mu = mu,
      sigma = sigma,
      log = TRUE
    ),
    dldmu = call_family_method(
      family_object$dldm,
      score_arguments
    ),
    dldsigma = call_family_method(
      family_object$dldd,
      score_arguments
    ),
    d2ldmu2 = call_family_method(
      base_family$d2ldm2,
      base_arguments
    ),
    d2ldsigma2 = call_family_method(
      base_family$d2ldd2,
      base_arguments
    ),
    d2ldmudsigma = call_family_method(
      base_family$d2ldmdd,
      base_arguments
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
