args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages(library(gamlss.mx))

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
reference_path <- file.path(reference_dir, "mixture_reference.csv")
fit_data_path <- file.path(reference_dir, "mixture_fit_data.csv")
fit_reference_path <- file.path(reference_dir, "mixture_fit_reference.csv")
fit_posterior_path <- file.path(
  reference_dir,
  "mixture_fit_posterior_reference.csv"
)

component_density <- function(family, y, mu, sigma) {
  switch(
    family,
    NO = dNO(y, mu = mu, sigma = sigma),
    GA = dGA(y, mu = mu, sigma = sigma),
    PO = dPO(y, mu = mu),
    stop("Unsupported mixture reference family: ", family)
  )
}

component_moments <- function(family, mu, sigma) {
  variance <- switch(
    family,
    NO = sigma^2,
    GA = (sigma * mu)^2,
    PO = mu,
    stop("Unsupported mixture reference family: ", family)
  )
  list(mean = mu, variance = variance)
}

evaluate_case <- function(
    case,
    family_1,
    family_2,
    y,
    mu_1,
    sigma_1,
    mu_2,
    sigma_2,
    pi_1) {
  pi_2 <- 1 - pi_1
  density_1 <- component_density(family_1, y, mu_1, sigma_1)
  density_2 <- component_density(family_2, y, mu_2, sigma_2)
  density <- vapply(seq_along(y), function(index) {
    dMX(
      y = y[index],
      mu = list(mu_1[index], mu_2[index]),
      sigma = list(sigma_1[index], sigma_2[index]),
      pi = list(pi_1[index], pi_2[index]),
      family = list(family_1, family_2)
    )
  }, numeric(1))
  cdf <- vapply(seq_along(y), function(index) {
    pMX(
      q = y[index],
      mu = list(mu_1[index], mu_2[index]),
      sigma = list(sigma_1[index], sigma_2[index]),
      pi = list(pi_1[index], pi_2[index]),
      family = list(family_1, family_2)
    )
  }, numeric(1))
  moments_1 <- component_moments(family_1, mu_1, sigma_1)
  moments_2 <- component_moments(family_2, mu_2, sigma_2)
  mixture_mean <- pi_1 * moments_1$mean + pi_2 * moments_2$mean
  second_moment <- pi_1 * (
    moments_1$variance + moments_1$mean^2
  ) + pi_2 * (
    moments_2$variance + moments_2$mean^2
  )

  data.frame(
    case = case,
    family_1 = family_1,
    family_2 = family_2,
    y = y,
    component_1_mu = mu_1,
    component_1_sigma = sigma_1,
    component_2_mu = mu_2,
    component_2_sigma = sigma_2,
    mixing_1 = log(pi_1 / pi_2),
    pi_1 = pi_1,
    pi_2 = pi_2,
    log_density = log(density),
    cdf = cdf,
    posterior_1 = pi_1 * density_1 / density,
    posterior_2 = pi_2 * density_2 / density,
    mean = mixture_mean,
    variance = second_moment - mixture_mean^2,
    gamlss_mx_version = as.character(packageVersion("gamlss.mx"))
  )
}

references <- rbind(
  evaluate_case(
    "normal_normal",
    "NO",
    "NO",
    y = c(-2.0, -0.2, 1.3, 4.0),
    mu_1 = c(-1.0, -0.5, 0.0, 0.5),
    sigma_1 = c(0.7, 0.9, 1.1, 1.3),
    mu_2 = c(2.5, 2.8, 3.1, 3.4),
    sigma_2 = c(1.2, 1.0, 0.8, 0.6),
    pi_1 = c(0.2, 0.4, 0.65, 0.85)
  ),
  evaluate_case(
    "gamma_gamma",
    "GA",
    "GA",
    y = c(0.2, 0.8, 2.5, 8.0),
    mu_1 = c(0.8, 1.0, 1.4, 2.0),
    sigma_1 = c(0.25, 0.35, 0.45, 0.55),
    mu_2 = c(3.0, 3.5, 4.0, 5.0),
    sigma_2 = c(0.6, 0.5, 0.4, 0.3),
    pi_1 = c(0.15, 0.35, 0.6, 0.8)
  ),
  evaluate_case(
    "poisson_poisson",
    "PO",
    "PO",
    y = c(0, 2, 5, 12),
    mu_1 = c(0.8, 1.5, 2.5, 4.0),
    sigma_1 = rep(1, 4),
    mu_2 = c(4.0, 5.0, 7.0, 10.0),
    sigma_2 = rep(1, 4),
    pi_1 = c(0.1, 0.3, 0.55, 0.75)
  )
)

set.seed(714)
fit_data <- data.frame(
  y = c(
    rnorm(40, mean = -2, sd = 0.5),
    rnorm(60, mean = 3, sd = 0.8)
  )
)
fit <- gamlssMX(
  y ~ 1,
  data = fit_data,
  family = NO,
  K = 2,
  prob = c(0.4, 0.6),
  control = MX.control(
    cc = 1e-10,
    n.cyc = 200,
    trace = FALSE,
    seed = 991,
    plot = FALSE
  ),
  g.control = gamlss.control(
    trace = FALSE,
    n.cyc = 200,
    c.crit = 1e-10
  )
)
component_order <- order(vapply(
  fit$models,
  function(model) fitted(model, "mu")[1],
  numeric(1)
))
ordered_models <- fit$models[component_order]
ordered_probabilities <- fit$prob[component_order]
fit_reference <- data.frame(
  global_deviance = fit$G.deviance,
  component_1_mu = fitted(ordered_models[[1]], "mu")[1],
  component_1_sigma = fitted(ordered_models[[1]], "sigma")[1],
  component_2_mu = fitted(ordered_models[[2]], "mu")[1],
  component_2_sigma = fitted(ordered_models[[2]], "sigma")[1],
  pi_1 = ordered_probabilities[1],
  pi_2 = ordered_probabilities[2],
  df_fit = fit$df.fit,
  aic = fit$aic,
  sbc = fit$sbc,
  gamlss_mx_version = as.character(packageVersion("gamlss.mx"))
)
fit_posterior <- data.frame(
  posterior_1 = fit$post.prob[, component_order[1]],
  posterior_2 = fit$post.prob[, component_order[2]]
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

check_reference <- function(actual_data, path, label) {
  temporary_path <- tempfile(fileext = ".csv")
  write_references(actual_data, temporary_path)
  actual <- read.csv(
    temporary_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  expected <- read.csv(
    path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!identical(names(actual), names(expected)) ||
      nrow(actual) != nrow(expected)) {
    stop(label, " reference dimensions or columns differ")
  }
  for (column in names(actual)) {
    if (!identical(is.na(actual[[column]]), is.na(expected[[column]]))) {
      stop(label, " reference NA pattern differs for ", column)
    }
    if (is.numeric(actual[[column]]) && is.numeric(expected[[column]])) {
      present <- !is.na(actual[[column]])
      difference <- abs(actual[[column]][present] - expected[[column]][present])
      allowed <- 1e-10 * (1 + abs(expected[[column]][present]))
      if (any(difference > allowed)) {
        stop(label, " numeric parity differs for ", column)
      }
    } else if (!identical(actual[[column]], expected[[column]])) {
      stop(label, " reference values differ for ", column)
    }
  }
}

if (check_only) {
  check_reference(references, reference_path, "Mixture-family")
  check_reference(fit_data, fit_data_path, "Mixture-fit data")
  check_reference(fit_reference, fit_reference_path, "Mixture-fit")
  check_reference(
    fit_posterior,
    fit_posterior_path,
    "Mixture-fit posterior"
  )
  message(
    "Finite-mixture R parity checks passed with gamlss.mx ",
    packageVersion("gamlss.mx")
  )
} else {
  write_references(references, reference_path)
  write_references(fit_data, fit_data_path)
  write_references(fit_reference, fit_reference_path)
  write_references(fit_posterior, fit_posterior_path)
  message(
    "Wrote finite-mixture references from gamlss.mx ",
    packageVersion("gamlss.mx")
  )
}
