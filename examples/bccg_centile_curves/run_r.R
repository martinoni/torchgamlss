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
abdom_data <- read.csv(data_path)
centile_values <- c(0.4, 2, 9, 25, 50, 75, 91, 98, 99.6)
parameters <- c("mu", "sigma", "nu")

fit <- gamlss(
  y ~ pb(x, lambda = 10),
  sigma.formula = ~ pb(x, lambda = 10),
  nu.formula = ~ pb(x, lambda = 10),
  family = BCCG(),
  method = RS(),
  data = abdom_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 300, trace = FALSE),
  i.control = glim.control(
    cc = 1e-8,
    cyc = 300,
    bf.tol = 1e-8,
    bf.cyc = 300,
    glm.trace = FALSE
  )
)

fit_result <- data.frame(
  converged = fit$converged,
  outer_iterations = fit$iter,
  global_deviance = unname(deviance(fit)),
  negative_log_likelihood = -as.numeric(logLik(fit)),
  effective_degrees_of_freedom = fit$df.fit,
  residual_degrees_of_freedom = fit$df.residual,
  observation_count = fit$N,
  effective_observation_count = fit$noObs,
  aic = unname(GAIC(fit, k = 2))
)

coefficient_rows <- function(parameter) {
  estimates <- coef(fit, what = parameter)
  if (length(estimates) != 2) {
    stop("Unexpected ", parameter, " coefficient count: ", length(estimates))
  }
  data.frame(
    parameter = parameter,
    term = c("Intercept", "x_linear"),
    estimate = unname(estimates)
  )
}
coefficients <- do.call(rbind, lapply(parameters, coefficient_rows))
rownames(coefficients) <- NULL

smoothing_rows <- function(parameter) {
  smooth <- getSmo(fit, parameter = parameter, which = 1)
  data.frame(
    parameter = parameter,
    smoothing_parameter = smooth$lambda,
    effective_degrees_of_freedom = smooth$edf
  )
}
smoothing <- do.call(rbind, lapply(parameters, smoothing_rows))
rownames(smoothing) <- NULL

fitted_parameters <- lapply(
  parameters,
  function(parameter) fitted(fit, what = parameter)
)
names(fitted_parameters) <- parameters
fitted_result <- data.frame(
  observation = seq_len(nrow(abdom_data)) - 1,
  age = abdom_data$x,
  mu = fitted_parameters$mu,
  sigma = fitted_parameters$sigma,
  nu = fitted_parameters$nu
)

grid <- data.frame(
  x = seq(min(abdom_data$x), max(abdom_data$x), length.out = 121)
)
grid_parameters <- list(
  mu = predict(
    fit,
    newdata = grid,
    what = "mu",
    type = "response"
  ),
  sigma = predict(
    fit,
    newdata = grid,
    what = "sigma",
    type = "response"
  ),
  nu = predict(
    fit,
    newdata = grid,
    what = "nu",
    type = "response"
  )
)

centiles <- do.call(
  rbind,
  lapply(seq_len(nrow(grid)), function(index) {
    probability <- centile_values / 100
    data.frame(
      grid_index = index - 1,
      age = grid$x[[index]],
      probability = probability,
      centile = centile_values,
      quantile = qBCCG(
        probability,
        mu = grid_parameters$mu[[index]],
        sigma = grid_parameters$sigma[[index]],
        nu = grid_parameters$nu[[index]]
      )
    )
  })
)
rownames(centiles) <- NULL

residuals <- data.frame(
  observation = seq_len(nrow(abdom_data)) - 1,
  age = abdom_data$x,
  quantile_residual = qnorm(
    pBCCG(
      abdom_data$y,
      mu = fitted_parameters$mu,
      sigma = fitted_parameters$sigma,
      nu = fitted_parameters$nu
    )
  )
)

metadata <- data.frame(
  case = "bccg_centile_curves",
  implementation = "R gamlss",
  family = "BCCG",
  algorithm = "RS",
  grid_size = nrow(grid),
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
  gamlss_data_version = as.character(packageVersion("gamlss.data"))
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
write_csv(smoothing, "smoothing.csv")
write_csv(fitted_result, "fitted.csv")
write_csv(centiles, "centiles.csv")
write_csv(residuals, "residuals.csv")
write_csv(metadata, "metadata.csv")
