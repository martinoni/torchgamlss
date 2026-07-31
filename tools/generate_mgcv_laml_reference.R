args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args

suppressPackageStartupMessages(library(mgcv))
options(digits = 17)

reference_dir <- file.path("tests", "reference")
dir.create(reference_dir, recursive = TRUE, showWarnings = FALSE)
design_path <- file.path(reference_dir, "mgcv_laml_design.csv")
penalty_path <- file.path(reference_dir, "mgcv_laml_penalties.csv")
summary_path <- file.path(reference_dir, "mgcv_laml_reference.csv")
coefficient_path <- file.path(
  reference_dir,
  "mgcv_laml_coefficient_reference.csv"
)
fitted_path <- file.path(reference_dir, "mgcv_laml_fitted_reference.csv")

observation_count <- 160
index <- seq_len(observation_count)
x <- seq(-1, 1, length.out = observation_count)
z <- cos(seq(0, 2 * pi, length.out = observation_count))
sigma_floor <- 0.01
true_sigma <- sigma_floor + exp(-1.45 + 0.4 * z + 0.8 * z^2)
set.seed(20260730)
standardized_error <- rnorm(observation_count)
y <- sin(pi * x) + 0.2 * x + true_sigma * standardized_error
weight <- rep(1, observation_count)
data <- data.frame(y = y, x = x, z = z, weight = weight)

setup <- gam(
  list(
    y ~ s(x, bs = "ps", k = 8),
    ~s(z, bs = "ps", k = 7)
  ),
  data = data,
  weights = weight,
  family = gaulss(b = sigma_floor),
  method = "REML",
  fit = FALSE
)
fit <- gam(G = setup, method = "REML")

predictor_indices <- attr(setup$X, "lpi")
mu_design <- setup$X[, predictor_indices[[1]], drop = FALSE]
sigma_design <- setup$X[, predictor_indices[[2]], drop = FALSE]
colnames(mu_design) <- paste0("mu_", seq_len(ncol(mu_design)))
colnames(sigma_design) <- paste0(
  "sigma_",
  seq_len(ncol(sigma_design))
)
design_reference <- data.frame(
  response = setup$y,
  weight = setup$w,
  mu_design,
  sigma_design,
  check.names = FALSE
)

coefficient_count <- ncol(setup$X)
penalty_references <- lapply(
  seq_along(setup$S),
  function(penalty_index) {
    full_penalty <- matrix(
      0,
      nrow = coefficient_count,
      ncol = coefficient_count
    )
    start <- setup$off[penalty_index]
    term_indices <- start:(
      start + nrow(setup$S[[penalty_index]]) - 1
    )
    full_penalty[term_indices, term_indices] <- setup$S[[penalty_index]]
    data.frame(
      penalty = penalty_index,
      row = rep(seq_len(coefficient_count), each = coefficient_count),
      column = rep(seq_len(coefficient_count), coefficient_count),
      value = as.vector(t(full_penalty))
    )
  }
)
penalty_reference <- do.call(rbind, penalty_references)

outer_gradient <- as.numeric(fit$outer.info$grad)
outer_hessian <- fit$outer.info$hess
summary_reference <- data.frame(
  mgcv_version = as.character(packageVersion("mgcv")),
  sigma_floor = sigma_floor,
  objective = as.numeric(fit$gcv.ubre),
  log_likelihood = as.numeric(fit$l),
  lambda_mu = as.numeric(fit$sp[1]),
  lambda_sigma = as.numeric(fit$sp[2]),
  effective_degrees_of_freedom = sum(fit$edf),
  outer_iterations = fit$outer.info$iter,
  outer_convergence = fit$outer.info$conv,
  gradient_mu = outer_gradient[1],
  gradient_sigma = outer_gradient[2],
  hessian_mu_mu = outer_hessian[1, 1],
  hessian_mu_sigma = outer_hessian[1, 2],
  hessian_sigma_sigma = outer_hessian[2, 2],
  coefficient_count = length(coef(fit))
)
coefficient_reference <- data.frame(
  index = seq_along(coef(fit)),
  coefficient = as.numeric(coef(fit)),
  effective_degrees_of_freedom = as.numeric(fit$edf)
)
link_prediction <- predict(fit, type = "link")
response_prediction <- fitted(fit)
fitted_reference <- data.frame(
  eta_mu = link_prediction[, 1],
  eta_sigma = link_prediction[, 2],
  mu = response_prediction[, 1],
  sigma = 1 / response_prediction[, 2]
)

tensor_design_path <- file.path(
  reference_dir,
  "mgcv_tensor_laml_design.csv"
)
tensor_penalty_path <- file.path(
  reference_dir,
  "mgcv_tensor_laml_penalties.csv"
)
tensor_summary_path <- file.path(
  reference_dir,
  "mgcv_tensor_laml_reference.csv"
)
tensor_coefficient_path <- file.path(
  reference_dir,
  "mgcv_tensor_laml_coefficient_reference.csv"
)
tensor_fitted_path <- file.path(
  reference_dir,
  "mgcv_tensor_laml_fitted_reference.csv"
)

tensor_observation_count <- 180
tensor_index <- seq_len(tensor_observation_count)
tensor_x <- seq(-1, 1, length.out = tensor_observation_count)
tensor_z <- sin(tensor_index * sqrt(2))
tensor_sigma_floor <- 0.01
tensor_true_sigma <- tensor_sigma_floor + exp(-1.55)
set.seed(20260731)
tensor_y <- (
  sin(pi * tensor_x) +
  0.55 * cos(2.2 * tensor_z) +
  0.8 * tensor_x * tensor_z +
  tensor_true_sigma * rnorm(tensor_observation_count)
)
tensor_data <- data.frame(
  y = tensor_y,
  x = tensor_x,
  z = tensor_z
)
tensor_setup <- gam(
  list(
    y ~ te(x, z, bs = c("ps", "ps"), k = c(5, 5)),
    ~1
  ),
  data = tensor_data,
  family = gaulss(b = tensor_sigma_floor),
  method = "REML",
  fit = FALSE
)
tensor_fit <- gam(G = tensor_setup, method = "REML")
tensor_predictor_indices <- attr(tensor_setup$X, "lpi")
tensor_mu_design <- tensor_setup$X[
  ,
  tensor_predictor_indices[[1]],
  drop = FALSE
]
tensor_sigma_design <- tensor_setup$X[
  ,
  tensor_predictor_indices[[2]],
  drop = FALSE
]
colnames(tensor_mu_design) <- paste0(
  "mu_",
  seq_len(ncol(tensor_mu_design))
)
colnames(tensor_sigma_design) <- paste0(
  "sigma_",
  seq_len(ncol(tensor_sigma_design))
)
tensor_design_reference <- data.frame(
  response = tensor_setup$y,
  weight = tensor_setup$w,
  tensor_mu_design,
  tensor_sigma_design,
  check.names = FALSE
)
tensor_coefficient_count <- ncol(tensor_setup$X)
tensor_penalty_references <- lapply(
  seq_along(tensor_setup$S),
  function(penalty_index) {
    full_penalty <- matrix(
      0,
      nrow = tensor_coefficient_count,
      ncol = tensor_coefficient_count
    )
    start <- tensor_setup$off[penalty_index]
    term_indices <- start:(
      start + nrow(tensor_setup$S[[penalty_index]]) - 1
    )
    full_penalty[term_indices, term_indices] <- (
      tensor_setup$S[[penalty_index]]
    )
    data.frame(
      penalty = penalty_index,
      row = rep(
        seq_len(tensor_coefficient_count),
        each = tensor_coefficient_count
      ),
      column = rep(
        seq_len(tensor_coefficient_count),
        tensor_coefficient_count
      ),
      value = as.vector(t(full_penalty))
    )
  }
)
tensor_penalty_reference <- do.call(
  rbind,
  tensor_penalty_references
)
tensor_outer_gradient <- as.numeric(tensor_fit$outer.info$grad)
tensor_outer_hessian <- tensor_fit$outer.info$hess
tensor_summary_reference <- data.frame(
  mgcv_version = as.character(packageVersion("mgcv")),
  sigma_floor = tensor_sigma_floor,
  objective = as.numeric(tensor_fit$gcv.ubre),
  log_likelihood = as.numeric(tensor_fit$l),
  lambda_x = as.numeric(tensor_fit$sp[1]),
  lambda_z = as.numeric(tensor_fit$sp[2]),
  effective_degrees_of_freedom = sum(tensor_fit$edf),
  outer_iterations = tensor_fit$outer.info$iter,
  outer_convergence = tensor_fit$outer.info$conv,
  gradient_x = tensor_outer_gradient[1],
  gradient_z = tensor_outer_gradient[2],
  hessian_x_x = tensor_outer_hessian[1, 1],
  hessian_x_z = tensor_outer_hessian[1, 2],
  hessian_z_z = tensor_outer_hessian[2, 2],
  coefficient_count = length(coef(tensor_fit))
)
tensor_coefficient_reference <- data.frame(
  index = seq_along(coef(tensor_fit)),
  coefficient = as.numeric(coef(tensor_fit)),
  effective_degrees_of_freedom = as.numeric(tensor_fit$edf)
)
tensor_link_prediction <- predict(tensor_fit, type = "link")
tensor_response_prediction <- fitted(tensor_fit)
tensor_fitted_reference <- data.frame(
  eta_mu = tensor_link_prediction[, 1],
  eta_sigma = tensor_link_prediction[, 2],
  mu = tensor_response_prediction[, 1],
  sigma = 1 / tensor_response_prediction[, 2]
)

poisson_design_path <- file.path(
  reference_dir,
  "mgcv_poisson_laml_design.csv"
)
poisson_penalty_path <- file.path(
  reference_dir,
  "mgcv_poisson_laml_penalties.csv"
)
poisson_summary_path <- file.path(
  reference_dir,
  "mgcv_poisson_laml_reference.csv"
)
poisson_coefficient_path <- file.path(
  reference_dir,
  "mgcv_poisson_laml_coefficient_reference.csv"
)
poisson_fitted_path <- file.path(
  reference_dir,
  "mgcv_poisson_laml_fitted_reference.csv"
)

poisson_observation_count <- 160
poisson_x <- seq(
  -1,
  1,
  length.out = poisson_observation_count
)
set.seed(20260801)
poisson_y <- rpois(
  poisson_observation_count,
  exp(0.3 + sin(pi * poisson_x) + 0.25 * poisson_x)
)
poisson_data <- data.frame(y = poisson_y, x = poisson_x)
poisson_setup <- gam(
  y ~ s(x, bs = "ps", k = 8),
  data = poisson_data,
  family = poisson(),
  method = "REML",
  fit = FALSE
)
poisson_fit <- gam(G = poisson_setup, method = "REML")
poisson_design <- poisson_setup$X
colnames(poisson_design) <- paste0(
  "mu_",
  seq_len(ncol(poisson_design))
)
poisson_design_reference <- data.frame(
  response = poisson_setup$y,
  weight = poisson_setup$w,
  poisson_design,
  check.names = FALSE
)
poisson_coefficient_count <- ncol(poisson_design)
poisson_penalty_references <- lapply(
  seq_along(poisson_setup$S),
  function(penalty_index) {
    full_penalty <- matrix(
      0,
      nrow = poisson_coefficient_count,
      ncol = poisson_coefficient_count
    )
    start <- poisson_setup$off[penalty_index]
    term_indices <- start:(
      start + nrow(poisson_setup$S[[penalty_index]]) - 1
    )
    full_penalty[term_indices, term_indices] <- (
      poisson_setup$S[[penalty_index]]
    )
    data.frame(
      penalty = penalty_index,
      row = rep(
        seq_len(poisson_coefficient_count),
        each = poisson_coefficient_count
      ),
      column = rep(
        seq_len(poisson_coefficient_count),
        poisson_coefficient_count
      ),
      value = as.vector(t(full_penalty))
    )
  }
)
poisson_penalty_reference <- do.call(
  rbind,
  poisson_penalty_references
)
poisson_outer_gradient <- as.numeric(
  poisson_fit$outer.info$grad
)
poisson_outer_hessian <- poisson_fit$outer.info$hess
poisson_summary_reference <- data.frame(
  mgcv_version = as.character(packageVersion("mgcv")),
  objective = as.numeric(poisson_fit$gcv.ubre),
  log_likelihood = as.numeric(logLik(poisson_fit)),
  lambda_mu = as.numeric(poisson_fit$sp[1]),
  effective_degrees_of_freedom = sum(poisson_fit$edf),
  outer_iterations = poisson_fit$outer.info$iter,
  outer_convergence = poisson_fit$outer.info$conv,
  gradient_mu = poisson_outer_gradient[1],
  hessian_mu_mu = poisson_outer_hessian[1, 1],
  coefficient_count = length(coef(poisson_fit))
)
poisson_coefficient_reference <- data.frame(
  index = seq_along(coef(poisson_fit)),
  coefficient = as.numeric(coef(poisson_fit)),
  effective_degrees_of_freedom = as.numeric(poisson_fit$edf)
)
poisson_fitted_reference <- data.frame(
  eta_mu = as.numeric(predict(poisson_fit, type = "link")),
  mu = as.numeric(fitted(poisson_fit))
)

gamma_design_path <- file.path(
  reference_dir,
  "mgcv_gamma_laml_design.csv"
)
gamma_penalty_path <- file.path(
  reference_dir,
  "mgcv_gamma_laml_penalties.csv"
)
gamma_summary_path <- file.path(
  reference_dir,
  "mgcv_gamma_laml_reference.csv"
)
gamma_coefficient_path <- file.path(
  reference_dir,
  "mgcv_gamma_laml_coefficient_reference.csv"
)
gamma_fitted_path <- file.path(
  reference_dir,
  "mgcv_gamma_laml_fitted_reference.csv"
)

gamma_observation_count <- 240
gamma_index <- seq_len(gamma_observation_count)
gamma_x <- seq(
  -1,
  1,
  length.out = gamma_observation_count
)
gamma_z <- sin(gamma_index * sqrt(2))
gamma_mu <- exp(0.4 + 0.65 * sin(pi * gamma_x))
gamma_sigma <- exp(-0.65 + 0.75 * gamma_z)
set.seed(20260802)
gamma_y <- rgamma(
  gamma_observation_count,
  shape = 1 / gamma_sigma^2,
  scale = gamma_mu * gamma_sigma^2
)
gamma_data <- data.frame(
  y = gamma_y,
  x = gamma_x,
  z = gamma_z
)
gamma_setup <- gam(
  list(
    y ~ s(x, bs = "ps", k = 8),
    ~s(z, bs = "ps", k = 7)
  ),
  data = gamma_data,
  family = gammals(link = list("identity", "identity")),
  method = "REML",
  fit = FALSE
)
gamma_fit <- gam(G = gamma_setup, method = "REML")
gamma_predictor_indices <- attr(gamma_setup$X, "lpi")
gamma_mu_design <- gamma_setup$X[
  ,
  gamma_predictor_indices[[1]],
  drop = FALSE
]
gamma_phi_design <- gamma_setup$X[
  ,
  gamma_predictor_indices[[2]],
  drop = FALSE
]
colnames(gamma_mu_design) <- paste0(
  "mu_",
  seq_len(ncol(gamma_mu_design))
)
colnames(gamma_phi_design) <- paste0(
  "phi_",
  seq_len(ncol(gamma_phi_design))
)
gamma_design_reference <- data.frame(
  response = gamma_setup$y,
  weight = gamma_setup$w,
  gamma_mu_design,
  gamma_phi_design,
  check.names = FALSE
)
gamma_coefficient_count <- ncol(gamma_setup$X)
gamma_penalty_references <- lapply(
  seq_along(gamma_setup$S),
  function(penalty_index) {
    full_penalty <- matrix(
      0,
      nrow = gamma_coefficient_count,
      ncol = gamma_coefficient_count
    )
    start <- gamma_setup$off[penalty_index]
    term_indices <- start:(
      start + nrow(gamma_setup$S[[penalty_index]]) - 1
    )
    full_penalty[term_indices, term_indices] <- (
      gamma_setup$S[[penalty_index]]
    )
    data.frame(
      penalty = penalty_index,
      row = rep(
        seq_len(gamma_coefficient_count),
        each = gamma_coefficient_count
      ),
      column = rep(
        seq_len(gamma_coefficient_count),
        gamma_coefficient_count
      ),
      value = as.vector(t(full_penalty))
    )
  }
)
gamma_penalty_reference <- do.call(
  rbind,
  gamma_penalty_references
)
gamma_outer_gradient <- as.numeric(
  gamma_fit$outer.info$grad
)
gamma_outer_hessian <- gamma_fit$outer.info$hess
gamma_summary_reference <- data.frame(
  mgcv_version = as.character(packageVersion("mgcv")),
  objective = as.numeric(gamma_fit$gcv.ubre),
  log_likelihood = as.numeric(logLik(gamma_fit)),
  lambda_mu = as.numeric(gamma_fit$sp[1]),
  lambda_phi = as.numeric(gamma_fit$sp[2]),
  effective_degrees_of_freedom = sum(gamma_fit$edf),
  outer_iterations = gamma_fit$outer.info$iter,
  outer_convergence = gamma_fit$outer.info$conv,
  gradient_mu = gamma_outer_gradient[1],
  gradient_phi = gamma_outer_gradient[2],
  hessian_mu_mu = gamma_outer_hessian[1, 1],
  hessian_mu_phi = gamma_outer_hessian[1, 2],
  hessian_phi_phi = gamma_outer_hessian[2, 2],
  coefficient_count = length(coef(gamma_fit))
)
gamma_coefficient_reference <- data.frame(
  index = seq_along(coef(gamma_fit)),
  coefficient = as.numeric(coef(gamma_fit)),
  effective_degrees_of_freedom = as.numeric(gamma_fit$edf)
)
gamma_link_prediction <- predict(gamma_fit, type = "link")
gamma_response_prediction <- fitted(gamma_fit)
gamma_fitted_reference <- data.frame(
  eta_mu = gamma_link_prediction[, 1],
  eta_phi = gamma_link_prediction[, 2],
  mu = gamma_response_prediction[, 1],
  sigma = exp(0.5 * gamma_response_prediction[, 2])
)

beta_design_path <- file.path(
  reference_dir,
  "mgcv_beta_laml_design.csv"
)
beta_penalty_path <- file.path(
  reference_dir,
  "mgcv_beta_laml_penalties.csv"
)
beta_summary_path <- file.path(
  reference_dir,
  "mgcv_beta_laml_reference.csv"
)
beta_coefficient_path <- file.path(
  reference_dir,
  "mgcv_beta_laml_coefficient_reference.csv"
)
beta_fitted_path <- file.path(
  reference_dir,
  "mgcv_beta_laml_fitted_reference.csv"
)

beta_observation_count <- 200
beta_x <- seq(
  -1,
  1,
  length.out = beta_observation_count
)
beta_mu <- plogis(
  -0.2 + 1.1 * sin(pi * beta_x) + 0.3 * beta_x
)
beta_phi <- 12
set.seed(20260803)
beta_y <- rbeta(
  beta_observation_count,
  beta_mu * beta_phi,
  (1 - beta_mu) * beta_phi
)
beta_data <- data.frame(y = beta_y, x = beta_x)
beta_setup <- gam(
  y ~ s(x, bs = "ps", k = 8),
  data = beta_data,
  family = betar(theta = beta_phi, link = "logit"),
  method = "REML",
  fit = FALSE
)
beta_fit <- gam(G = beta_setup, method = "REML")
beta_design <- beta_setup$X
colnames(beta_design) <- paste0(
  "mu_",
  seq_len(ncol(beta_design))
)
beta_design_reference <- data.frame(
  response = beta_setup$y,
  weight = beta_setup$w,
  beta_design,
  check.names = FALSE
)
beta_coefficient_count <- ncol(beta_design)
beta_penalty_references <- lapply(
  seq_along(beta_setup$S),
  function(penalty_index) {
    full_penalty <- matrix(
      0,
      nrow = beta_coefficient_count,
      ncol = beta_coefficient_count
    )
    start <- beta_setup$off[penalty_index]
    term_indices <- start:(
      start + nrow(beta_setup$S[[penalty_index]]) - 1
    )
    full_penalty[term_indices, term_indices] <- (
      beta_setup$S[[penalty_index]]
    )
    data.frame(
      penalty = penalty_index,
      row = rep(
        seq_len(beta_coefficient_count),
        each = beta_coefficient_count
      ),
      column = rep(
        seq_len(beta_coefficient_count),
        beta_coefficient_count
      ),
      value = as.vector(t(full_penalty))
    )
  }
)
beta_penalty_reference <- do.call(
  rbind,
  beta_penalty_references
)
beta_outer_gradient <- as.numeric(
  beta_fit$outer.info$grad
)
beta_outer_hessian <- beta_fit$outer.info$hess
beta_fitted_phi <- as.numeric(beta_fit$family$getTheta(TRUE))
beta_fitted_sigma <- sqrt(1 / (1 + beta_fitted_phi))
beta_summary_reference <- data.frame(
  mgcv_version = as.character(packageVersion("mgcv")),
  phi = beta_fitted_phi,
  sigma = beta_fitted_sigma,
  eta_sigma = qlogis(beta_fitted_sigma),
  objective = as.numeric(beta_fit$gcv.ubre),
  log_likelihood = as.numeric(logLik(beta_fit)),
  lambda_mu = as.numeric(beta_fit$sp[1]),
  effective_degrees_of_freedom = sum(beta_fit$edf),
  outer_iterations = beta_fit$outer.info$iter,
  outer_convergence = beta_fit$outer.info$conv,
  gradient_mu = beta_outer_gradient[1],
  hessian_mu_mu = beta_outer_hessian[1, 1],
  coefficient_count = length(coef(beta_fit))
)
beta_coefficient_reference <- data.frame(
  index = seq_along(coef(beta_fit)),
  coefficient = as.numeric(coef(beta_fit)),
  effective_degrees_of_freedom = as.numeric(beta_fit$edf)
)
beta_fitted_reference <- data.frame(
  eta_mu = as.numeric(predict(beta_fit, type = "link")),
  mu = as.numeric(fitted(beta_fit))
)

write_reference <- function(value, path) {
  connection <- file(path, open = "wb")
  on.exit(close(connection))
  write.csv(
    value,
    connection,
    row.names = FALSE,
    na = ""
  )
}

check_reference <- function(actual, path, label) {
  temporary_path <- tempfile(fileext = ".csv")
  write_reference(actual, temporary_path)
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
    # Optimizer iteration counts and nearly zero terminal gradients vary with
    # the mgcv build and linked BLAS. Treat the latter as a convergence bound;
    # all fitted statistical quantities remain strict reference comparisons.
    if (column == "outer_iterations") {
      next
    }
    if (grepl("^gradient_", column)) {
      if (any(abs(generated[[column]]) > 1e-4)) {
        stop(label, " convergence gradient is too large for ", column)
      }
      next
    }
    if (is.numeric(generated[[column]]) &&
        is.numeric(expected[[column]])) {
      difference <- abs(
        generated[[column]] - expected[[column]]
      )
      relative_tolerance <- 2e-6
      if (identical(label, "gamma_summary") &&
          grepl("^lambda_", column)) {
        # mgcv's outer Newton optimizer can settle at slightly different
        # smoothing parameters across BLAS implementations even when the
        # fitted model and convergence diagnostics agree.
        relative_tolerance <- 1e-5
      }
      allowed <- relative_tolerance * (
        1 + abs(expected[[column]])
      )
      failed <- which(difference > allowed)
      if (length(failed) > 0L) {
        index <- failed[[which.max(
          difference[failed] / allowed[failed]
        )]]
        stop(
          label,
          " numeric parity differs for ",
          column,
          " at row ",
          index,
          ": actual=",
          format(generated[[column]][[index]], digits = 16),
          ", expected=",
          format(expected[[column]][[index]], digits = 16),
          ", difference=",
          format(difference[[index]], digits = 16),
          ", allowed=",
          format(allowed[[index]], digits = 16)
        )
      }
    } else if (!identical(
      generated[[column]],
      expected[[column]]
    )) {
      stop(label, " reference values differ for ", column)
    }
  }
}

references <- list(
  design = list(value = design_reference, path = design_path),
  penalty = list(value = penalty_reference, path = penalty_path),
  summary = list(value = summary_reference, path = summary_path),
  coefficient = list(
    value = coefficient_reference,
    path = coefficient_path
  ),
  fitted = list(value = fitted_reference, path = fitted_path),
  tensor_design = list(
    value = tensor_design_reference,
    path = tensor_design_path
  ),
  tensor_penalty = list(
    value = tensor_penalty_reference,
    path = tensor_penalty_path
  ),
  tensor_summary = list(
    value = tensor_summary_reference,
    path = tensor_summary_path
  ),
  tensor_coefficient = list(
    value = tensor_coefficient_reference,
    path = tensor_coefficient_path
  ),
  tensor_fitted = list(
    value = tensor_fitted_reference,
    path = tensor_fitted_path
  ),
  poisson_design = list(
    value = poisson_design_reference,
    path = poisson_design_path
  ),
  poisson_penalty = list(
    value = poisson_penalty_reference,
    path = poisson_penalty_path
  ),
  poisson_summary = list(
    value = poisson_summary_reference,
    path = poisson_summary_path
  ),
  poisson_coefficient = list(
    value = poisson_coefficient_reference,
    path = poisson_coefficient_path
  ),
  poisson_fitted = list(
    value = poisson_fitted_reference,
    path = poisson_fitted_path
  ),
  gamma_design = list(
    value = gamma_design_reference,
    path = gamma_design_path
  ),
  gamma_penalty = list(
    value = gamma_penalty_reference,
    path = gamma_penalty_path
  ),
  gamma_summary = list(
    value = gamma_summary_reference,
    path = gamma_summary_path
  ),
  gamma_coefficient = list(
    value = gamma_coefficient_reference,
    path = gamma_coefficient_path
  ),
  gamma_fitted = list(
    value = gamma_fitted_reference,
    path = gamma_fitted_path
  ),
  beta_design = list(
    value = beta_design_reference,
    path = beta_design_path
  ),
  beta_penalty = list(
    value = beta_penalty_reference,
    path = beta_penalty_path
  ),
  beta_summary = list(
    value = beta_summary_reference,
    path = beta_summary_path
  ),
  beta_coefficient = list(
    value = beta_coefficient_reference,
    path = beta_coefficient_path
  ),
  beta_fitted = list(
    value = beta_fitted_reference,
    path = beta_fitted_path
  )
)

if (check_only) {
  for (label in names(references)) {
    check_reference(
      references[[label]]$value,
      references[[label]]$path,
      label
    )
  }
  message(
    "mgcv LAML reference checks passed with mgcv ",
    packageVersion("mgcv")
  )
} else {
  for (reference in references) {
    write_reference(reference$value, reference$path)
  }
  message(
    "Wrote mgcv LAML references with mgcv ",
    packageVersion("mgcv")
  )
}
