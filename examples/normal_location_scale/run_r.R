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

suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))

options(digits = 17, scipen = 999)
data <- read.csv(data_path)
fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = NO(),
  method = RS(),
  data = data,
  control = gamlss.control(
    c.crit = 1e-10,
    n.cyc = 200,
    trace = FALSE
  ),
  i.control = glim.control(
    cc = 1e-10,
    cyc = 200,
    glm.trace = FALSE
  )
)

mu <- fitted(fit, what = "mu")
sigma <- fitted(fit, what = "sigma")
probabilities <- c(0.03, 0.50, 0.97)

fit_result <- data.frame(
  converged = fit$converged,
  outer_iterations = fit$iter,
  global_deviance = unname(deviance(fit)),
  negative_log_likelihood = -as.numeric(logLik(fit)),
  effective_degrees_of_freedom = fit$df.fit,
  effective_observation_count = sum(data$weight),
  aic = unname(deviance(fit)) + 2 * fit$df.fit
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
  coefficient_rows("sigma")
)

fitted_result <- data.frame(
  observation = seq_len(nrow(data)) - 1,
  mu = mu,
  sigma = sigma
)

quantiles <- do.call(
  rbind,
  lapply(seq_len(nrow(data)), function(index) {
    data.frame(
      observation = index - 1,
      probability = probabilities,
      centile = 100 * probabilities,
      quantile = qNO(
        probabilities,
        mu = mu[[index]],
        sigma = sigma[[index]]
      )
    )
  })
)
rownames(quantiles) <- NULL

residuals <- data.frame(
  observation = seq_len(nrow(data)) - 1,
  quantile_residual = qnorm(pNO(data$y, mu = mu, sigma = sigma))
)

metadata <- data.frame(
  case = "normal_location_scale_rs",
  implementation = "R gamlss",
  family = "NO",
  algorithm = "RS",
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
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
write_csv(quantiles, "quantiles.csv")
write_csv(residuals, "residuals.csv")
write_csv(metadata, "metadata.csv")
