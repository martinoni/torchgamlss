args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(gamlss))

reference_dir <- file.path("tests", "reference")
cases_path <- file.path(reference_dir, "no_cases.csv")
fit_data_path <- file.path(reference_dir, "no_fit_data.csv")
reference_path <- file.path(reference_dir, "no_reference.csv")
fit_reference_path <- file.path(reference_dir, "no_fit_reference.csv")
ga_cases_path <- file.path(reference_dir, "ga_cases.csv")
ga_reference_path <- file.path(reference_dir, "ga_reference.csv")
ga_fit_data_path <- file.path(reference_dir, "ga_fit_data.csv")
ga_rs_reference_path <- file.path(reference_dir, "ga_rs_reference.csv")
ga_pb_reference_path <- file.path(reference_dir, "ga_pb_reference.csv")
ga_pb_fitted_reference_path <- file.path(
  reference_dir, "ga_pb_fitted_reference.csv"
)
ga_pb_coefficient_reference_path <- file.path(
  reference_dir, "ga_pb_coefficient_reference.csv"
)
rs_fit_data_path <- file.path(reference_dir, "no_rs_fit_data.csv")
rs_reference_path <- file.path(reference_dir, "no_rs_reference.csv")
pb_fit_data_path <- file.path(reference_dir, "no_pb_fit_data.csv")
pb_reference_path <- file.path(reference_dir, "no_pb_reference.csv")
pb_fitted_reference_path <- file.path(reference_dir, "no_pb_fitted_reference.csv")
pb_coefficient_reference_path <- file.path(
  reference_dir, "no_pb_coefficient_reference.csv"
)
pb_ml_reference_path <- file.path(reference_dir, "no_pb_ml_reference.csv")
pb_ml_fitted_reference_path <- file.path(
  reference_dir, "no_pb_ml_fitted_reference.csv"
)
pb_ml_coefficient_reference_path <- file.path(
  reference_dir, "no_pb_ml_coefficient_reference.csv"
)
pb_df_reference_path <- file.path(reference_dir, "no_pb_df_reference.csv")
pb_df_fitted_reference_path <- file.path(
  reference_dir, "no_pb_df_fitted_reference.csv"
)
pb_df_coefficient_reference_path <- file.path(
  reference_dir, "no_pb_df_coefficient_reference.csv"
)
pb_gaic_reference_path <- file.path(reference_dir, "no_pb_gaic_reference.csv")
pb_gaic_fitted_reference_path <- file.path(
  reference_dir, "no_pb_gaic_fitted_reference.csv"
)
pb_gaic_coefficient_reference_path <- file.path(
  reference_dir, "no_pb_gaic_coefficient_reference.csv"
)
pb_gcv_reference_path <- file.path(reference_dir, "no_pb_gcv_reference.csv")
pb_gcv_fitted_reference_path <- file.path(
  reference_dir, "no_pb_gcv_fitted_reference.csv"
)
pb_gcv_coefficient_reference_path <- file.path(
  reference_dir, "no_pb_gcv_coefficient_reference.csv"
)

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

ga_family <- GA()
ga_cases <- read.csv(ga_cases_path)
ga_reference <- data.frame(
  y = ga_cases$y,
  mu = ga_cases$mu,
  sigma = ga_cases$sigma,
  eta_mu = ga_family$mu.linkfun(ga_cases$mu),
  eta_sigma = ga_family$sigma.linkfun(ga_cases$sigma),
  log_density = dGA(ga_cases$y, ga_cases$mu, ga_cases$sigma, log = TRUE),
  dldmu = ga_family$dldm(ga_cases$y, ga_cases$mu, ga_cases$sigma),
  dldsigma = ga_family$dldd(ga_cases$y, ga_cases$mu, ga_cases$sigma),
  d2ldmu2 = ga_family$d2ldm2(ga_cases$mu, ga_cases$sigma),
  d2ldsigma2 = ga_family$d2ldd2(ga_cases$sigma),
  d2ldmudsigma = ga_family$d2ldmdd(ga_cases$y),
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

ga_n <- 60
ga_x <- seq(-1, 1, length.out = ga_n)
ga_z <- cos(seq(0, 2 * pi, length.out = ga_n))
ga_mu_offset <- 0.05 * sin(seq(0, 3 * pi, length.out = ga_n))
ga_sigma_offset <- 0.04 * cos(seq(0, 4 * pi, length.out = ga_n))
ga_weight <- rep(c(1, 1.5, 2, 0.75), length.out = ga_n)
ga_mu <- exp(
  0.35 + 0.45 * ga_x + 0.35 * sin(pi * ga_x) + ga_mu_offset
)
ga_sigma <- exp(-0.75 + 0.20 * ga_z + ga_sigma_offset)
ga_probability <- (((seq_len(ga_n) * 37) %% ga_n) + 0.5) / ga_n
ga_fit_data <- data.frame(
  x = ga_x,
  z = ga_z,
  y = qGA(ga_probability, mu = ga_mu, sigma = ga_sigma),
  weight = ga_weight,
  mu_offset = ga_mu_offset,
  sigma_offset = ga_sigma_offset
)
ga_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = GA(),
  method = RS(),
  data = ga_fit_data,
  control = gamlss.control(c.crit = 1e-10, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-10, cyc = 200)
)
ga_rs_reference <- data.frame(
  mu_intercept = unname(coef(ga_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(ga_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(ga_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(ga_rs_fit, what = "sigma")[[2]]),
  global_deviance = unname(deviance(ga_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(ga_rs_fit)),
  outer_iterations = ga_rs_fit$iter,
  converged = ga_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

ga_pb_fit <- gamlss(
  y ~ pb(x, lambda = 12) + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = GA(),
  method = RS(),
  data = ga_fit_data,
  control = gamlss.control(c.crit = 1e-9, n.cyc = 200, trace = FALSE),
  i.control = glim.control(
    cc = 1e-10, cyc = 200, bf.tol = 1e-10, bf.cyc = 200
  )
)
ga_pb_smooth <- getSmo(ga_pb_fit, parameter = "mu", which = 1)
ga_pb_reference <- data.frame(
  mu_intercept = unname(coef(ga_pb_fit, what = "mu")[[1]]),
  mu_x = unname(coef(ga_pb_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(ga_pb_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(ga_pb_fit, what = "sigma")[[2]]),
  smoothing_parameter = unname(ga_pb_smooth$lambda),
  smooth_edf = unname(ga_pb_smooth$edf),
  global_deviance = unname(deviance(ga_pb_fit)),
  negative_log_likelihood = -as.numeric(logLik(ga_pb_fit)),
  outer_iterations = ga_pb_fit$iter,
  converged = ga_pb_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
ga_pb_fitted_reference <- data.frame(
  mu = fitted(ga_pb_fit, what = "mu"),
  sigma = fitted(ga_pb_fit, what = "sigma"),
  mu_linear_predictor = ga_pb_fit$mu.lp,
  mu_smooth = drop(ga_pb_fit$mu.s[, 1])
)
ga_pb_coefficient_reference <- data.frame(
  coefficient = drop(ga_pb_smooth$coef)
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

pb_x <- seq(-1, 1, length.out = 40)
pb_fit_data <- data.frame(
  x = pb_x,
  y = 0.7 + 0.8 * pb_x + 1.15 * sin(pi * pb_x) +
    0.16 * sin(31 * pb_x) + 0.07 * cos(23 * pb_x)
)
pb_fit <- gamlss(
  y ~ pb(x, lambda = 12),
  sigma.formula = ~ 1,
  family = NO(),
  method = RS(),
  data = pb_fit_data,
  control = gamlss.control(c.crit = 1e-10, n.cyc = 200, trace = FALSE),
  i.control = glim.control(
    cc = 1e-10, cyc = 200, bf.tol = 1e-10, bf.cyc = 200
  )
)
pb_smooth <- getSmo(pb_fit, parameter = "mu", which = 1)
pb_reference <- data.frame(
  mu_intercept = unname(coef(pb_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pb_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pb_fit, what = "sigma")[[1]]),
  smoothing_parameter = unname(pb_smooth$lambda),
  smooth_edf = unname(pb_smooth$edf),
  global_deviance = unname(deviance(pb_fit)),
  negative_log_likelihood = -as.numeric(logLik(pb_fit)),
  outer_iterations = pb_fit$iter,
  converged = pb_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
pb_fitted_reference <- data.frame(
  mu = fitted(pb_fit, what = "mu"),
  sigma = fitted(pb_fit, what = "sigma"),
  mu_linear_predictor = pb_fit$mu.lp,
  mu_smooth = drop(pb_fit$mu.s[, 1])
)
pb_coefficient_reference <- data.frame(
  coefficient = drop(pb_smooth$coef)
)

pb_ml_fit <- gamlss(
  y ~ pb(x),
  sigma.formula = ~ 1,
  family = NO(),
  method = RS(),
  data = pb_fit_data,
  control = gamlss.control(c.crit = 1e-10, n.cyc = 200, trace = FALSE),
  i.control = glim.control(
    cc = 1e-10, cyc = 200, bf.tol = 1e-10, bf.cyc = 200
  )
)
pb_ml_smooth <- getSmo(pb_ml_fit, parameter = "mu", which = 1)
pb_ml_reference <- data.frame(
  mu_intercept = unname(coef(pb_ml_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pb_ml_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pb_ml_fit, what = "sigma")[[1]]),
  smoothing_parameter = unname(pb_ml_smooth$lambda),
  smooth_edf = unname(pb_ml_smooth$edf),
  global_deviance = unname(deviance(pb_ml_fit)),
  negative_log_likelihood = -as.numeric(logLik(pb_ml_fit)),
  outer_iterations = pb_ml_fit$iter,
  converged = pb_ml_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
pb_ml_fitted_reference <- data.frame(
  mu = fitted(pb_ml_fit, what = "mu"),
  sigma = fitted(pb_ml_fit, what = "sigma"),
  mu_linear_predictor = pb_ml_fit$mu.lp,
  mu_smooth = drop(pb_ml_fit$mu.s[, 1])
)
pb_ml_coefficient_reference <- data.frame(
  coefficient = drop(pb_ml_smooth$coef)
)

pb_df_fit <- gamlss(
  y ~ pb(x, df = 3),
  sigma.formula = ~ 1,
  family = NO(),
  method = RS(),
  data = pb_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 200, trace = FALSE),
  i.control = glim.control(
    cc = 1e-10, cyc = 200, bf.tol = 1e-10, bf.cyc = 200
  )
)
pb_df_smooth <- getSmo(pb_df_fit, parameter = "mu", which = 1)
pb_df_reference <- data.frame(
  mu_intercept = unname(coef(pb_df_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pb_df_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pb_df_fit, what = "sigma")[[1]]),
  requested_df = 3,
  target_edf = 5,
  smoothing_parameter = unname(pb_df_smooth$lambda),
  smooth_edf = unname(pb_df_smooth$edf),
  global_deviance = unname(deviance(pb_df_fit)),
  negative_log_likelihood = -as.numeric(logLik(pb_df_fit)),
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
pb_df_fitted_reference <- data.frame(
  mu = fitted(pb_df_fit, what = "mu"),
  sigma = fitted(pb_df_fit, what = "sigma"),
  mu_linear_predictor = pb_df_fit$mu.lp,
  mu_smooth = drop(pb_df_fit$mu.s[, 1])
)
pb_df_coefficient_reference <- data.frame(
  coefficient = drop(pb_df_smooth$coef)
)

pb_gaic_fit <- gamlss(
  y ~ pb(x, control = pb.control(method = "GAIC", k = 2)),
  sigma.formula = ~ 1,
  family = NO(),
  method = RS(),
  data = pb_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 200, trace = FALSE),
  i.control = glim.control(
    cc = 1e-10, cyc = 200, bf.tol = 1e-10, bf.cyc = 200
  )
)
pb_gaic_smooth <- getSmo(pb_gaic_fit, parameter = "mu", which = 1)
pb_gaic_reference <- data.frame(
  mu_intercept = unname(coef(pb_gaic_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pb_gaic_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pb_gaic_fit, what = "sigma")[[1]]),
  criterion_penalty = 2,
  smoothing_parameter = unname(pb_gaic_smooth$lambda),
  smooth_edf = unname(pb_gaic_smooth$edf),
  global_deviance = unname(deviance(pb_gaic_fit)),
  negative_log_likelihood = -as.numeric(logLik(pb_gaic_fit)),
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
pb_gaic_fitted_reference <- data.frame(
  mu = fitted(pb_gaic_fit, what = "mu"),
  sigma = fitted(pb_gaic_fit, what = "sigma"),
  mu_linear_predictor = pb_gaic_fit$mu.lp,
  mu_smooth = drop(pb_gaic_fit$mu.s[, 1])
)
pb_gaic_coefficient_reference <- data.frame(
  coefficient = drop(pb_gaic_smooth$coef)
)

pb_gcv_fit <- gamlss(
  y ~ pb(x, control = pb.control(method = "GCV", k = 2)),
  sigma.formula = ~ 1,
  family = NO(),
  method = RS(),
  data = pb_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 200, trace = FALSE),
  i.control = glim.control(
    cc = 1e-10, cyc = 200, bf.tol = 1e-10, bf.cyc = 200
  )
)
pb_gcv_smooth <- getSmo(pb_gcv_fit, parameter = "mu", which = 1)
pb_gcv_reference <- data.frame(
  mu_intercept = unname(coef(pb_gcv_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pb_gcv_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pb_gcv_fit, what = "sigma")[[1]]),
  criterion_penalty = 2,
  smoothing_parameter = unname(pb_gcv_smooth$lambda),
  smooth_edf = unname(pb_gcv_smooth$edf),
  global_deviance = unname(deviance(pb_gcv_fit)),
  negative_log_likelihood = -as.numeric(logLik(pb_gcv_fit)),
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
pb_gcv_fitted_reference <- data.frame(
  mu = fitted(pb_gcv_fit, what = "mu"),
  sigma = fitted(pb_gcv_fit, what = "sigma"),
  mu_linear_predictor = pb_gcv_fit$mu.lp,
  mu_smooth = drop(pb_gcv_fit$mu.s[, 1])
)
pb_gcv_coefficient_reference <- data.frame(
  coefficient = drop(pb_gcv_smooth$coef)
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
  if (any(character_columns) &&
      !identical(actual[character_columns], expected[character_columns])) {
    stop("Reference package versions differ in ", expected_path)
  }
}

write_csv_lf <- function(data, path) {
  connection <- file(path, open = "wb")
  on.exit(close(connection))
  write.table(
    data,
    connection,
    row.names = FALSE,
    col.names = TRUE,
    sep = ",",
    quote = FALSE,
    eol = "\n",
    na = "NA",
    dec = "."
  )
}

if (check_only) {
  assert_close(reference, reference_path, tolerance = 1e-12)
  assert_close(fit_reference, fit_reference_path, tolerance = 1e-7)
  assert_close(ga_reference, ga_reference_path, tolerance = 1e-12)
  assert_close(ga_fit_data, ga_fit_data_path, tolerance = 1e-12)
  assert_close(ga_rs_reference, ga_rs_reference_path, tolerance = 1e-7)
  assert_close(ga_pb_reference, ga_pb_reference_path, tolerance = 1e-7)
  assert_close(
    ga_pb_fitted_reference, ga_pb_fitted_reference_path, tolerance = 1e-7
  )
  assert_close(
    ga_pb_coefficient_reference,
    ga_pb_coefficient_reference_path,
    tolerance = 1e-7
  )
  assert_close(rs_reference, rs_reference_path, tolerance = 1e-7)
  assert_close(pb_fit_data, pb_fit_data_path, tolerance = 1e-12)
  assert_close(pb_reference, pb_reference_path, tolerance = 1e-7)
  assert_close(
    pb_fitted_reference, pb_fitted_reference_path, tolerance = 1e-7
  )
  assert_close(
    pb_coefficient_reference, pb_coefficient_reference_path, tolerance = 1e-7
  )
  assert_close(pb_ml_reference, pb_ml_reference_path, tolerance = 1e-7)
  assert_close(
    pb_ml_fitted_reference, pb_ml_fitted_reference_path, tolerance = 1e-7
  )
  assert_close(
    pb_ml_coefficient_reference,
    pb_ml_coefficient_reference_path,
    tolerance = 1e-7
  )
  assert_close(pb_df_reference, pb_df_reference_path, tolerance = 1e-7)
  assert_close(
    pb_df_fitted_reference, pb_df_fitted_reference_path, tolerance = 1e-7
  )
  assert_close(
    pb_df_coefficient_reference,
    pb_df_coefficient_reference_path,
    tolerance = 1e-7
  )
  assert_close(pb_gaic_reference, pb_gaic_reference_path, tolerance = 1e-6)
  assert_close(
    pb_gaic_fitted_reference,
    pb_gaic_fitted_reference_path,
    tolerance = 1e-6
  )
  assert_close(
    pb_gaic_coefficient_reference,
    pb_gaic_coefficient_reference_path,
    tolerance = 1e-6
  )
  assert_close(pb_gcv_reference, pb_gcv_reference_path, tolerance = 1e-6)
  assert_close(
    pb_gcv_fitted_reference, pb_gcv_fitted_reference_path, tolerance = 1e-6
  )
  assert_close(
    pb_gcv_coefficient_reference,
    pb_gcv_coefficient_reference_path,
    tolerance = 1e-6
  )
  message("R reference parity checks passed")
} else {
  options(digits = 17, scipen = 999)
  write.csv(reference, reference_path, row.names = FALSE, quote = FALSE)
  write.csv(fit_reference, fit_reference_path, row.names = FALSE, quote = FALSE)
  write_csv_lf(ga_reference, ga_reference_path)
  write_csv_lf(ga_fit_data, ga_fit_data_path)
  write_csv_lf(ga_rs_reference, ga_rs_reference_path)
  write_csv_lf(ga_pb_reference, ga_pb_reference_path)
  write_csv_lf(ga_pb_fitted_reference, ga_pb_fitted_reference_path)
  write_csv_lf(
    ga_pb_coefficient_reference,
    ga_pb_coefficient_reference_path
  )
  write.csv(rs_reference, rs_reference_path, row.names = FALSE, quote = FALSE)
  write.csv(pb_fit_data, pb_fit_data_path, row.names = FALSE, quote = FALSE)
  write.csv(pb_reference, pb_reference_path, row.names = FALSE, quote = FALSE)
  write.csv(
    pb_fitted_reference, pb_fitted_reference_path, row.names = FALSE, quote = FALSE
  )
  write.csv(
    pb_coefficient_reference,
    pb_coefficient_reference_path,
    row.names = FALSE,
    quote = FALSE
  )
  write.csv(
    pb_ml_reference, pb_ml_reference_path, row.names = FALSE, quote = FALSE
  )
  write.csv(
    pb_ml_fitted_reference,
    pb_ml_fitted_reference_path,
    row.names = FALSE,
    quote = FALSE
  )
  write.csv(
    pb_ml_coefficient_reference,
    pb_ml_coefficient_reference_path,
    row.names = FALSE,
    quote = FALSE
  )
  write_csv_lf(pb_df_reference, pb_df_reference_path)
  write.csv(
    pb_df_fitted_reference,
    pb_df_fitted_reference_path,
    row.names = FALSE,
    quote = FALSE
  )
  write.csv(
    pb_df_coefficient_reference,
    pb_df_coefficient_reference_path,
    row.names = FALSE,
    quote = FALSE
  )
  write_csv_lf(pb_gaic_reference, pb_gaic_reference_path)
  write_csv_lf(
    pb_gaic_fitted_reference,
    pb_gaic_fitted_reference_path
  )
  write_csv_lf(
    pb_gaic_coefficient_reference,
    pb_gaic_coefficient_reference_path
  )
  write_csv_lf(pb_gcv_reference, pb_gcv_reference_path)
  write_csv_lf(pb_gcv_fitted_reference, pb_gcv_fitted_reference_path)
  write_csv_lf(
    pb_gcv_coefficient_reference,
    pb_gcv_coefficient_reference_path
  )
  message("Wrote R reference fixtures to ", reference_dir)
}
