args <- commandArgs(trailingOnly = TRUE)

option_value <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop("Missing required option ", flag)
  }
  args[[index + 1]]
}

data_path <- option_value("--data")
output_dir <- option_value("--output-dir")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

local_library <- file.path(getwd(), ".r-library")
.libPaths(c(local_library, .libPaths()))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(gamlss.cens))
suppressPackageStartupMessages(library(survival))

options(digits = 17, scipen = 999)
data <- read.csv(data_path)
GGcens <- cens(family = "GG", type = "right", local = FALSE)
fit <- gamlss(
  Surv(time, event) ~ x,
  sigma.formula = ~ 1,
  nu.formula = ~ 1,
  family = GGcens(),
  method = RS(),
  data = data,
  control = gamlss.control(
    c.crit = 1e-8,
    n.cyc = 300,
    trace = FALSE
  ),
  i.control = glim.control(
    cc = 1e-8,
    cyc = 300,
    glm.trace = FALSE
  )
)

mu <- fitted(fit, what = "mu")
sigma <- fitted(fit, what = "sigma")
nu <- fitted(fit, what = "nu")
fit_result <- data.frame(
  converged = fit$converged,
  global_deviance = unname(deviance(fit)),
  negative_log_likelihood = -as.numeric(logLik(fit)),
  effective_degrees_of_freedom = fit$df.fit,
  observation_count = nrow(data),
  event_count = sum(data$event),
  censored_count = sum(data$event == 0)
)

coefficient_rows <- function(parameter) {
  estimates <- coef(fit, what = parameter)
  terms <- names(estimates)
  terms[terms == "(Intercept)"] <- "Intercept"
  data.frame(
    parameter = parameter,
    term = terms,
    estimate = unname(estimates)
  )
}
coefficients <- rbind(
  coefficient_rows("mu"),
  coefficient_rows("sigma"),
  coefficient_rows("nu")
)
fitted_result <- data.frame(
  observation = seq_len(nrow(data)) - 1,
  mu = mu,
  sigma = sigma,
  nu = nu
)

profile_x <- c(-0.75, 0, 0.75)
times <- c(0.25, 0.5, 0.75, 1, 1.5, 2, 3)
mu_coefficient <- coef(fit, what = "mu")
sigma_coefficient <- coef(fit, what = "sigma")
nu_coefficient <- coef(fit, what = "nu")
profile_mu <- exp(mu_coefficient[[1]] + mu_coefficient[[2]] * profile_x)
profile_sigma <- rep(exp(sigma_coefficient[[1]]), length(profile_x))
profile_nu <- rep(nu_coefficient[[1]], length(profile_x))

survival_rows <- lapply(seq_along(profile_x), function(profile) {
  survival_probability <- pGG(
    times,
    mu = profile_mu[[profile]],
    sigma = profile_sigma[[profile]],
    nu = profile_nu[[profile]],
    lower.tail = FALSE
  )
  log_density <- dGG(
    times,
    mu = profile_mu[[profile]],
    sigma = profile_sigma[[profile]],
    nu = profile_nu[[profile]],
    log = TRUE
  )
  data.frame(
    profile = profile - 1,
    x = profile_x[[profile]],
    time = times,
    survival = survival_probability,
    hazard = exp(log_density) / survival_probability,
    cumulative_hazard = -log(survival_probability)
  )
})
survival_result <- do.call(rbind, survival_rows)

probabilities <- c(0.1, 0.5, 0.9)
quantile_rows <- lapply(seq_along(profile_x), function(profile) {
  data.frame(
    profile = profile - 1,
    x = profile_x[[profile]],
    probability = probabilities,
    centile = 100 * probabilities,
    quantile = qGG(
      probabilities,
      mu = profile_mu[[profile]],
      sigma = profile_sigma[[profile]],
      nu = profile_nu[[profile]]
    )
  )
})
quantiles <- do.call(rbind, quantile_rows)
metadata <- data.frame(
  case = "generalized_gamma_right_censored_mle",
  implementation = "R gamlss",
  family = "GGcens",
  algorithm = "RS",
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
  gamlss_cens_version = as.character(packageVersion("gamlss.cens"))
)

write_csv <- function(frame, file_name) {
  connection <- file(file.path(output_dir, file_name), open = "wb")
  on.exit(close(connection))
  write.table(
    frame,
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

write_csv(fit_result, "fit.csv")
write_csv(coefficients, "coefficients.csv")
write_csv(fitted_result, "fitted.csv")
write_csv(survival_result, "survival.csv")
write_csv(quantiles, "quantiles.csv")
write_csv(metadata, "metadata.csv")
