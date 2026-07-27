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
data$uniform <- (
  ((seq_len(nrow(data)) * 29) %% nrow(data)) + 0.5
) / nrow(data)

po_fit <- gamlss(
  y ~ x + offset(log_exposure),
  weights = weight,
  family = PO(),
  method = RS(),
  data = data,
  control = gamlss.control(c.crit = 1e-10, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-10, cyc = 200, glm.trace = FALSE)
)
nbi_fit <- gamlss(
  y ~ x + offset(log_exposure),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = NBI(),
  method = RS(),
  data = data,
  control = gamlss.control(c.crit = 1e-9, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-9, cyc = 200, glm.trace = FALSE)
)
fits <- list(PO = po_fit, NBI = nbi_fit)

fit_row <- function(model_name, fit) {
  mu <- fitted(fit, what = "mu")
  sigma <- if (model_name == "NBI") fitted(fit, what = "sigma") else 0
  variance <- mu + sigma * mu^2
  pearson <- sum(data$weight * (data$y - mu)^2 / variance) / fit$df.residual
  data.frame(
    model = model_name,
    converged = fit$converged,
    outer_iterations = fit$iter,
    global_deviance = unname(deviance(fit)),
    negative_log_likelihood = -as.numeric(logLik(fit)),
    effective_degrees_of_freedom = fit$df.fit,
    residual_degrees_of_freedom = fit$df.residual,
    observation_count = fit$N,
    effective_observation_count = fit$noObs,
    aic = unname(GAIC(fit, k = 2)),
    aicc = unname(GAIC(fit, k = 2, c = TRUE)),
    bic = unname(GAIC(fit, k = log(fit$noObs))),
    pearson_dispersion = pearson
  )
}
fit_result <- do.call(
  rbind,
  Map(fit_row, names(fits), fits)
)
rownames(fit_result) <- NULL

comparison <- fit_result[
  order(fit_result$aic, fit_result$model),
  c(
    "model",
    "effective_degrees_of_freedom",
    "global_deviance",
    "aic"
  )
]
names(comparison) <- c(
  "model",
  "degrees_of_freedom",
  "global_deviance",
  "criterion"
)
comparison$rank <- seq_len(nrow(comparison))
comparison$delta <- comparison$criterion - min(comparison$criterion)
relative_likelihood <- exp(-0.5 * comparison$delta)
comparison$weight <- relative_likelihood / sum(relative_likelihood)
comparison <- comparison[
  ,
  c(
    "model",
    "rank",
    "degrees_of_freedom",
    "global_deviance",
    "criterion",
    "delta",
    "weight"
  )
]
rownames(comparison) <- NULL

coefficient_rows <- function(model_name, fit, parameter) {
  estimates <- coef(fit, what = parameter)
  terms <- names(estimates)
  terms[terms == "(Intercept)"] <- "Intercept"
  data.frame(
    model = model_name,
    parameter = parameter,
    term = terms,
    estimate = unname(estimates)
  )
}
coefficients <- rbind(
  coefficient_rows("PO", po_fit, "mu"),
  coefficient_rows("NBI", nbi_fit, "mu"),
  coefficient_rows("NBI", nbi_fit, "sigma")
)
rownames(coefficients) <- NULL

fitted_rows <- function(model_name, fit) {
  mu <- fitted(fit, what = "mu")
  sigma <- if (model_name == "NBI") fitted(fit, what = "sigma") else {
    rep(0, nrow(data))
  }
  data.frame(
    model = model_name,
    observation = seq_len(nrow(data)) - 1,
    mu = mu,
    sigma = sigma,
    variance = mu + sigma * mu^2
  )
}
fitted_result <- rbind(
  fitted_rows("PO", po_fit),
  fitted_rows("NBI", nbi_fit)
)
rownames(fitted_result) <- NULL

probabilities <- c(0.05, 0.50, 0.95)
quantile_rows <- function(model_name, fit) {
  mu <- fitted(fit, what = "mu")
  sigma <- if (model_name == "NBI") fitted(fit, what = "sigma") else NULL
  do.call(
    rbind,
    lapply(seq_len(nrow(data)), function(index) {
      quantile <- if (model_name == "NBI") {
        qNBI(probabilities, mu = mu[[index]], sigma = sigma[[index]])
      } else {
        qPO(probabilities, mu = mu[[index]])
      }
      data.frame(
        model = model_name,
        observation = index - 1,
        probability = probabilities,
        centile = 100 * probabilities,
        quantile = quantile
      )
    })
  )
}
quantiles <- rbind(
  quantile_rows("PO", po_fit),
  quantile_rows("NBI", nbi_fit)
)
quantiles <- quantiles[quantiles$observation %% 7 == 0, ]
rownames(quantiles) <- NULL

residual_rows <- function(model_name, fit) {
  mu <- fitted(fit, what = "mu")
  if (model_name == "NBI") {
    sigma <- fitted(fit, what = "sigma")
    lower <- pNBI(data$y - 1, mu = mu, sigma = sigma)
    upper <- pNBI(data$y, mu = mu, sigma = sigma)
  } else {
    lower <- pPO(data$y - 1, mu = mu)
    upper <- pPO(data$y, mu = mu)
  }
  probability <- lower + data$uniform * (upper - lower)
  data.frame(
    model = model_name,
    observation = seq_len(nrow(data)) - 1,
    uniform = data$uniform,
    quantile_residual = qnorm(probability)
  )
}
residuals <- rbind(
  residual_rows("PO", po_fit),
  residual_rows("NBI", nbi_fit)
)
rownames(residuals) <- NULL

metadata <- data.frame(
  case = rep("count_model_comparison", length(fits)),
  model = names(fits),
  implementation = rep("R gamlss", length(fits)),
  algorithm = rep("RS", length(fits)),
  gamlss_version = rep(
    as.character(packageVersion("gamlss")),
    length(fits)
  ),
  gamlss_dist_version = rep(
    as.character(packageVersion("gamlss.dist")),
    length(fits)
  )
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
write_csv(comparison, "model_comparison.csv")
write_csv(coefficients, "coefficients.csv")
write_csv(fitted_result, "fitted.csv")
write_csv(quantiles, "quantiles.csv")
write_csv(residuals, "residuals.csv")
write_csv(metadata, "metadata.csv")
