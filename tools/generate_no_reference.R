args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(gamlss))

reference_dir <- file.path("tests", "reference")
cases_path <- file.path(reference_dir, "no_cases.csv")
fit_data_path <- file.path(reference_dir, "no_fit_data.csv")
reference_path <- file.path(reference_dir, "no_reference.csv")
fit_reference_path <- file.path(reference_dir, "no_fit_reference.csv")
rs_fit_data_path <- file.path(reference_dir, "no_rs_fit_data.csv")
rs_reference_path <- file.path(reference_dir, "no_rs_reference.csv")

family <- NO()
cases <- read.csv(cases_path)

reference <- data.frame(
  y = cases$y,
  mu = cases$mu,
  sigma = cases$sigma,
  eta_mu = family$mu.linkfun(cases$mu),
  eta_sigma = family$sigma.linkfun(cases$sigma),
  log_density = dNO(cases$y, cases$mu, cases$sigma, log = TRUE),
  dldmu = family$dldm(cases$y, cases$mu, cases$sigma),
  dldsigma = family$dldd(cases$y, cases$mu, cases$sigma),
  d2ldmu2 = family$d2ldm2(cases$sigma),
  d2ldsigma2 = family$d2ldd2(cases$sigma),
  d2ldmudsigma = family$d2ldmdd(cases$y),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

fit_data <- read.csv(fit_data_path)
fit <- gamlss(
  y ~ x,
  sigma.formula = ~ 1,
  family = NO(),
  data = fit_data,
  control = gamlss.control(trace = FALSE, n.cyc = 200)
)

fit_reference <- data.frame(
  mu_intercept = unname(coef(fit, what = "mu")[[1]]),
  mu_x = unname(coef(fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(fit, what = "sigma")[[1]]),
  negative_log_likelihood = -as.numeric(logLik(fit)),
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

rs_fit_data <- read.csv(rs_fit_data_path)
rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = NO(),
  method = RS(),
  data = rs_fit_data,
  control = gamlss.control(c.crit = 1e-10, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-10, cyc = 200)
)

rs_reference <- data.frame(
  mu_intercept = unname(coef(rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(rs_fit, what = "sigma")[[2]]),
  global_deviance = unname(deviance(rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(rs_fit)),
  outer_iterations = rs_fit$iter,
  converged = rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

assert_close <- function(actual, expected_path, tolerance) {
  expected <- read.csv(expected_path, check.names = FALSE)
  if (!identical(names(actual), names(expected))) {
    stop("Reference columns differ in ", expected_path)
  }

  numeric_columns <- vapply(actual, is.numeric, logical(1))
  character_columns <- !numeric_columns
  difference <- abs(as.matrix(actual[numeric_columns]) -
                    as.matrix(expected[numeric_columns]))
  scale <- 1 + abs(as.matrix(expected[numeric_columns]))

  if (any(difference > tolerance * scale)) {
    stop("Numeric parity check failed for ", expected_path)
  }
  if (!identical(actual[character_columns], expected[character_columns])) {
    stop("Reference package versions differ in ", expected_path)
  }
}

if (check_only) {
  assert_close(reference, reference_path, tolerance = 1e-12)
  assert_close(fit_reference, fit_reference_path, tolerance = 1e-7)
  assert_close(rs_reference, rs_reference_path, tolerance = 1e-7)
  message("R reference parity checks passed")
} else {
  options(digits = 17, scipen = 999)
  write.csv(reference, reference_path, row.names = FALSE, quote = FALSE)
  write.csv(fit_reference, fit_reference_path, row.names = FALSE, quote = FALSE)
  write.csv(rs_reference, rs_reference_path, row.names = FALSE, quote = FALSE)
  message("Wrote R reference fixtures to ", reference_dir)
}
