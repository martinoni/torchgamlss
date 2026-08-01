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
cg_smooth_reference_path <- file.path(
  reference_dir, "cg_smooth_reference.csv"
)
cg_smooth_linear_reference_path <- file.path(
  reference_dir, "cg_smooth_linear_reference.csv"
)
cg_smooth_fitted_reference_path <- file.path(
  reference_dir, "cg_smooth_fitted_reference.csv"
)
cg_smooth_coefficient_reference_path <- file.path(
  reference_dir, "cg_smooth_coefficient_reference.csv"
)
cg_multi_smooth_reference_path <- file.path(
  reference_dir, "cg_multi_smooth_reference.csv"
)
cg_multi_smooth_fit_data_path <- file.path(
  reference_dir, "cg_multi_smooth_fit_data.csv"
)
cg_multi_smooth_linear_reference_path <- file.path(
  reference_dir, "cg_multi_smooth_linear_reference.csv"
)
cg_multi_smooth_fitted_reference_path <- file.path(
  reference_dir, "cg_multi_smooth_fitted_reference.csv"
)
cg_multi_smooth_term_reference_path <- file.path(
  reference_dir, "cg_multi_smooth_term_reference.csv"
)
cg_multi_smooth_coefficient_reference_path <- file.path(
  reference_dir, "cg_multi_smooth_coefficient_reference.csv"
)
cg_multi_smooth_contribution_reference_path <- file.path(
  reference_dir, "cg_multi_smooth_contribution_reference.csv"
)
po_reference_path <- file.path(reference_dir, "po_reference.csv")
po_fit_data_path <- file.path(reference_dir, "po_fit_data.csv")
po_rs_reference_path <- file.path(reference_dir, "po_rs_reference.csv")
nbi_reference_path <- file.path(reference_dir, "nbi_reference.csv")
nbi_fit_data_path <- file.path(reference_dir, "nbi_fit_data.csv")
nbi_rs_reference_path <- file.path(reference_dir, "nbi_rs_reference.csv")
be_reference_path <- file.path(reference_dir, "be_reference.csv")
be_fit_data_path <- file.path(reference_dir, "be_fit_data.csv")
be_rs_reference_path <- file.path(reference_dir, "be_rs_reference.csv")
be_cg_reference_path <- file.path(reference_dir, "be_cg_reference.csv")
bccg_reference_path <- file.path(reference_dir, "bccg_reference.csv")
bccg_fit_data_path <- file.path(reference_dir, "bccg_fit_data.csv")
bccg_rs_reference_path <- file.path(reference_dir, "bccg_rs_reference.csv")
bct_reference_path <- file.path(reference_dir, "bct_reference.csv")
bct_fit_data_path <- file.path(reference_dir, "bct_fit_data.csv")
bct_rs_reference_path <- file.path(reference_dir, "bct_rs_reference.csv")
bct_laml_fixed_reference_path <- file.path(
  reference_dir, "bct_laml_fixed_reference.csv"
)
bct_laml_fixed_fitted_reference_path <- file.path(
  reference_dir, "bct_laml_fixed_fitted_reference.csv"
)
bct_cg_reference_path <- file.path(reference_dir, "bct_cg_reference.csv")
tf_reference_path <- file.path(reference_dir, "tf_reference.csv")
tf_fit_data_path <- file.path(reference_dir, "tf_fit_data.csv")
tf_rs_reference_path <- file.path(reference_dir, "tf_rs_reference.csv")
tf_cg_reference_path <- file.path(reference_dir, "tf_cg_reference.csv")
pe_reference_path <- file.path(reference_dir, "pe_reference.csv")
pe_fit_data_path <- file.path(reference_dir, "pe_fit_data.csv")
pe_rs_reference_path <- file.path(reference_dir, "pe_rs_reference.csv")
pe_cg_reference_path <- file.path(reference_dir, "pe_cg_reference.csv")
bcpe_reference_path <- file.path(reference_dir, "bcpe_reference.csv")
bcpe_fit_data_path <- file.path(reference_dir, "bcpe_fit_data.csv")
bcpe_rs_reference_path <- file.path(reference_dir, "bcpe_rs_reference.csv")
inference_table_path <- file.path(reference_dir, "inference_table_reference.csv")
inference_covariance_path <- file.path(
  reference_dir, "inference_covariance_reference.csv"
)
conditional_inference_table_path <- file.path(
  reference_dir, "conditional_inference_table_reference.csv"
)
conditional_inference_covariance_path <- file.path(
  reference_dir, "conditional_inference_covariance_reference.csv"
)
smooth_inference_path <- file.path(
  reference_dir, "smooth_inference_reference.csv"
)
joint_smooth_coefficient_covariance_path <- file.path(
  reference_dir, "joint_smooth_coefficient_covariance_reference.csv"
)
joint_smooth_curve_covariance_path <- file.path(
  reference_dir, "joint_smooth_curve_covariance_reference.csv"
)
model_diagnostics_path <- file.path(
  reference_dir, "model_diagnostics_reference.csv"
)
residual_plot_summary_path <- file.path(
  reference_dir, "residual_plot_summary_reference.csv"
)
worm_plot_path <- file.path(
  reference_dir, "worm_plot_reference.csv"
)
bucket_plot_path <- file.path(
  reference_dir, "bucket_plot_reference.csv"
)
quantile_residual_path <- file.path(
  reference_dir, "quantile_residual_reference.csv"
)
quantile_prediction_path <- file.path(
  reference_dir, "quantile_prediction_reference.csv"
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

po_family <- PO()
po_cases <- data.frame(
  y = c(0, 1, 2, 5, 9),
  mu = c(0.2, 0.8, 2.5, 5.5, 9)
)
po_reference <- data.frame(
  y = po_cases$y,
  mu = po_cases$mu,
  eta_mu = po_family$mu.linkfun(po_cases$mu),
  log_density = dPO(po_cases$y, po_cases$mu, log = TRUE),
  dldmu = po_family$dldm(po_cases$y, po_cases$mu),
  d2ldmu2 = po_family$d2ldm2(po_cases$mu),
  initial_mu = (po_cases$y + mean(po_cases$y)) / 2,
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

nbi_family <- NBI()
nbi_cases <- data.frame(
  y = c(0, 1, 3, 7, 12),
  mu = c(0.4, 1.2, 2.8, 6.5, 10),
  sigma = c(0.15, 0.25, 0.4, 0.7, 1.1)
)
nbi_initial_sigma <- max(
  (var(nbi_cases$y) - mean(nbi_cases$y)) / mean(nbi_cases$y)^2,
  0.1
)
nbi_reference <- data.frame(
  y = nbi_cases$y,
  mu = nbi_cases$mu,
  sigma = nbi_cases$sigma,
  eta_mu = nbi_family$mu.linkfun(nbi_cases$mu),
  eta_sigma = nbi_family$sigma.linkfun(nbi_cases$sigma),
  log_density = dNBI(
    nbi_cases$y, nbi_cases$mu, nbi_cases$sigma, log = TRUE
  ),
  dldmu = nbi_family$dldm(
    nbi_cases$y, nbi_cases$mu, nbi_cases$sigma
  ),
  dldsigma = nbi_family$dldd(
    nbi_cases$y, nbi_cases$mu, nbi_cases$sigma
  ),
  d2ldmu2 = nbi_family$d2ldm2(nbi_cases$mu, nbi_cases$sigma),
  d2ldsigma2 = nbi_family$d2ldd2(
    nbi_cases$y, nbi_cases$mu, nbi_cases$sigma
  ),
  d2ldmudsigma = nbi_family$d2ldmdd(nbi_cases$y),
  initial_mu = (nbi_cases$y + mean(nbi_cases$y)) / 2,
  initial_sigma = rep(nbi_initial_sigma, nrow(nbi_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

be_family <- BE()
be_cases <- data.frame(
  y = c(0.03, 0.15, 0.4, 0.72, 0.94),
  mu = c(0.08, 0.22, 0.5, 0.7, 0.9),
  sigma = c(0.15, 0.25, 0.4, 0.55, 0.7)
)
be_reference <- data.frame(
  y = be_cases$y,
  mu = be_cases$mu,
  sigma = be_cases$sigma,
  eta_mu = be_family$mu.linkfun(be_cases$mu),
  eta_sigma = be_family$sigma.linkfun(be_cases$sigma),
  log_density = dBE(be_cases$y, be_cases$mu, be_cases$sigma, log = TRUE),
  dldmu = be_family$dldm(be_cases$y, be_cases$mu, be_cases$sigma),
  dldsigma = be_family$dldd(be_cases$y, be_cases$mu, be_cases$sigma),
  d2ldmu2 = be_family$d2ldm2(be_cases$mu, be_cases$sigma),
  d2ldsigma2 = be_family$d2ldd2(be_cases$mu, be_cases$sigma),
  d2ldmudsigma = be_family$d2ldmdd(be_cases$mu, be_cases$sigma),
  initial_mu = (be_cases$y + mean(be_cases$y)) / 2,
  initial_sigma = rep(0.5, nrow(be_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bccg_family <- BCCG()
bccg_cases <- data.frame(
  y = c(0.65, 1.2, 2.8, 5.5, 11),
  mu = c(0.8, 1.5, 3, 5, 9),
  sigma = c(0.12, 0.2, 0.3, 0.45, 0.6),
  nu = c(-1.2, -0.35, 0.08, 0.4, 1.1)
)
bccg_reference <- data.frame(
  y = bccg_cases$y,
  mu = bccg_cases$mu,
  sigma = bccg_cases$sigma,
  nu = bccg_cases$nu,
  eta_mu = bccg_family$mu.linkfun(bccg_cases$mu),
  eta_sigma = bccg_family$sigma.linkfun(bccg_cases$sigma),
  eta_nu = bccg_family$nu.linkfun(bccg_cases$nu),
  log_density = dBCCG(
    bccg_cases$y,
    bccg_cases$mu,
    bccg_cases$sigma,
    bccg_cases$nu,
    log = TRUE
  ),
  cdf = pBCCG(
    bccg_cases$y,
    bccg_cases$mu,
    bccg_cases$sigma,
    bccg_cases$nu
  ),
  dldmu = bccg_family$dldm(
    bccg_cases$y, bccg_cases$mu, bccg_cases$sigma, bccg_cases$nu
  ),
  dldsigma = bccg_family$dldd(
    bccg_cases$y, bccg_cases$mu, bccg_cases$sigma, bccg_cases$nu
  ),
  dldnu = bccg_family$dldv(
    bccg_cases$y, bccg_cases$mu, bccg_cases$sigma, bccg_cases$nu
  ),
  d2ldmu2 = bccg_family$d2ldm2(
    bccg_cases$y, bccg_cases$mu, bccg_cases$sigma, bccg_cases$nu
  ),
  d2ldsigma2 = bccg_family$d2ldd2(bccg_cases$sigma),
  d2ldnu2 = bccg_family$d2ldv2(bccg_cases$sigma),
  d2ldmudsigma = bccg_family$d2ldmdd(
    bccg_cases$mu, bccg_cases$sigma, bccg_cases$nu
  ),
  d2ldmudnu = bccg_family$d2ldmdv(bccg_cases$mu),
  d2ldsigmadnu = bccg_family$d2ldddv(
    bccg_cases$sigma, bccg_cases$nu
  ),
  initial_mu = (bccg_cases$y + mean(bccg_cases$y)) / 2,
  initial_sigma = rep(0.1, nrow(bccg_cases)),
  initial_nu = rep(0.5, nrow(bccg_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bct_family <- BCT()
bct_cases <- data.frame(
  y = c(0.65, 1.2, 2.8, 5.5, 11),
  mu = c(0.8, 1.5, 3, 5, 9),
  sigma = c(0.12, 0.2, 0.3, 0.45, 0.6),
  nu = c(-1.2, -0.35, 0.08, 0.4, 1.1),
  tau = c(2.5, 4, 7, 12, 30)
)
bct_reference <- data.frame(
  y = bct_cases$y,
  mu = bct_cases$mu,
  sigma = bct_cases$sigma,
  nu = bct_cases$nu,
  tau = bct_cases$tau,
  eta_mu = bct_family$mu.linkfun(bct_cases$mu),
  eta_sigma = bct_family$sigma.linkfun(bct_cases$sigma),
  eta_nu = bct_family$nu.linkfun(bct_cases$nu),
  eta_tau = bct_family$tau.linkfun(bct_cases$tau),
  log_density = dBCT(
    bct_cases$y,
    bct_cases$mu,
    bct_cases$sigma,
    bct_cases$nu,
    bct_cases$tau,
    log = TRUE
  ),
  cdf = pBCT(
    bct_cases$y,
    bct_cases$mu,
    bct_cases$sigma,
    bct_cases$nu,
    bct_cases$tau
  ),
  dldmu = bct_family$dldm(
    bct_cases$y,
    bct_cases$mu,
    bct_cases$sigma,
    bct_cases$nu,
    bct_cases$tau
  ),
  dldsigma = bct_family$dldd(
    bct_cases$y,
    bct_cases$mu,
    bct_cases$sigma,
    bct_cases$nu,
    bct_cases$tau
  ),
  dldnu = bct_family$dldv(
    bct_cases$y,
    bct_cases$mu,
    bct_cases$sigma,
    bct_cases$nu,
    bct_cases$tau
  ),
  dldtau = bct_family$dldt(
    bct_cases$y,
    bct_cases$mu,
    bct_cases$sigma,
    bct_cases$nu,
    bct_cases$tau
  ),
  d2ldmu2 = bct_family$d2ldm2(
    bct_cases$mu, bct_cases$sigma, bct_cases$nu, bct_cases$tau
  ),
  d2ldsigma2 = bct_family$d2ldd2(bct_cases$sigma, bct_cases$tau),
  d2ldnu2 = bct_family$d2ldv2(bct_cases$sigma),
  d2ldtau2 = bct_family$d2ldt2(bct_cases$tau),
  d2ldmudsigma = bct_family$d2ldmdd(
    bct_cases$mu, bct_cases$sigma, bct_cases$nu, bct_cases$tau
  ),
  d2ldmudnu = bct_family$d2ldmdv(bct_cases$mu, bct_cases$tau),
  d2ldmudtau = bct_family$d2ldmdt(
    bct_cases$mu, bct_cases$nu, bct_cases$tau
  ),
  d2ldsigmadnu = bct_family$d2ldddv(
    bct_cases$sigma, bct_cases$nu, bct_cases$tau
  ),
  d2ldsigmadtau = bct_family$d2ldddt(
    bct_cases$sigma, bct_cases$tau
  ),
  d2ldnudtau = bct_family$d2ldvdt(
    bct_cases$sigma, bct_cases$nu, bct_cases$tau
  ),
  initial_mu = (bct_cases$y + mean(bct_cases$y)) / 2,
  initial_sigma = rep(0.1, nrow(bct_cases)),
  initial_nu = rep(0.5, nrow(bct_cases)),
  initial_tau = rep(10, nrow(bct_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

tf_family <- TF()
tf_cases <- data.frame(
  y = c(-3.5, -0.8, 0, 2.4, 8),
  mu = c(-2, -0.5, 0.3, 2, 6),
  sigma = c(0.4, 0.8, 1.2, 2, 3),
  nu = c(0.7, 1.5, 3, 8, 30)
)
tf_reference <- data.frame(
  y = tf_cases$y,
  mu = tf_cases$mu,
  sigma = tf_cases$sigma,
  nu = tf_cases$nu,
  eta_mu = tf_family$mu.linkfun(tf_cases$mu),
  eta_sigma = tf_family$sigma.linkfun(tf_cases$sigma),
  eta_nu = tf_family$nu.linkfun(tf_cases$nu),
  log_density = dTF(
    tf_cases$y,
    tf_cases$mu,
    tf_cases$sigma,
    tf_cases$nu,
    log = TRUE
  ),
  cdf = pTF(
    tf_cases$y,
    tf_cases$mu,
    tf_cases$sigma,
    tf_cases$nu
  ),
  dldmu = tf_family$dldm(
    tf_cases$y,
    tf_cases$mu,
    tf_cases$sigma,
    tf_cases$nu
  ),
  dldsigma = tf_family$dldd(
    tf_cases$y,
    tf_cases$mu,
    tf_cases$sigma,
    tf_cases$nu
  ),
  dldnu = tf_family$dldv(
    tf_cases$y,
    tf_cases$mu,
    tf_cases$sigma,
    tf_cases$nu
  ),
  d2ldmu2 = tf_family$d2ldm2(tf_cases$sigma, tf_cases$nu),
  d2ldsigma2 = tf_family$d2ldd2(tf_cases$sigma, tf_cases$nu),
  d2ldnu2 = tf_family$d2ldv2(
    tf_cases$y,
    tf_cases$mu,
    tf_cases$sigma,
    tf_cases$nu
  ),
  d2ldmudsigma = tf_family$d2ldmdd(tf_cases$y),
  d2ldmudnu = tf_family$d2ldmdv(tf_cases$y),
  d2ldsigmadnu = tf_family$d2ldddv(tf_cases$sigma, tf_cases$nu),
  initial_mu = (tf_cases$y + mean(tf_cases$y)) / 2,
  initial_sigma = rep(sd(tf_cases$y), nrow(tf_cases)),
  initial_nu = rep(10, nrow(tf_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

pe_family <- PE()
pe_cases <- data.frame(
  y = c(-3.5, -0.8, 0, 2.4, 8),
  mu = c(-2, -0.5, 0.3, 2, 6),
  sigma = c(0.4, 0.8, 1.2, 2, 3),
  nu = c(0.7, 1, 2, 4, 8)
)
pe_reference <- data.frame(
  y = pe_cases$y,
  mu = pe_cases$mu,
  sigma = pe_cases$sigma,
  nu = pe_cases$nu,
  eta_mu = pe_family$mu.linkfun(pe_cases$mu),
  eta_sigma = pe_family$sigma.linkfun(pe_cases$sigma),
  eta_nu = pe_family$nu.linkfun(pe_cases$nu),
  log_density = dPE(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu,
    log = TRUE
  ),
  cdf = pPE(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  dldmu = pe_family$dldm(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  dldsigma = pe_family$dldd(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  dldnu = pe_family$dldv(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  d2ldmu2 = pe_family$d2ldm2(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  d2ldsigma2 = pe_family$d2ldd2(pe_cases$sigma, pe_cases$nu),
  d2ldnu2 = pe_family$d2ldv2(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  d2ldmudsigma = pe_family$d2ldmdd(pe_cases$y),
  d2ldmudnu = pe_family$d2ldmdv(pe_cases$y),
  d2ldsigmadnu = pe_family$d2ldddv(
    pe_cases$y,
    pe_cases$mu,
    pe_cases$sigma,
    pe_cases$nu
  ),
  initial_mu = (pe_cases$y + mean(pe_cases$y)) / 2,
  initial_sigma = (
    abs(pe_cases$y - mean(pe_cases$y)) + sd(pe_cases$y)
  ) / 2,
  initial_nu = rep(1.8, nrow(pe_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bcpe_family <- BCPE()
bcpe_cases <- data.frame(
  y = c(0.65, 1.2, 2.8, 5.5, 11),
  mu = c(0.8, 1.5, 3, 5, 9),
  sigma = c(0.12, 0.2, 0.3, 0.45, 0.6),
  nu = c(-1.2, -0.35, 0.08, 0.4, 1.1),
  tau = c(1.2, 1.5, 2, 3, 5)
)
bcpe_reference <- data.frame(
  y = bcpe_cases$y,
  mu = bcpe_cases$mu,
  sigma = bcpe_cases$sigma,
  nu = bcpe_cases$nu,
  tau = bcpe_cases$tau,
  eta_mu = bcpe_family$mu.linkfun(bcpe_cases$mu),
  eta_sigma = bcpe_family$sigma.linkfun(bcpe_cases$sigma),
  eta_nu = bcpe_family$nu.linkfun(bcpe_cases$nu),
  eta_tau = bcpe_family$tau.linkfun(bcpe_cases$tau),
  log_density = dBCPE(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau,
    log = TRUE
  ),
  cdf = pBCPE(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  dldmu = bcpe_family$dldm(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  dldsigma = bcpe_family$dldd(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  dldnu = bcpe_family$dldv(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  dldtau = bcpe_family$dldt(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  d2ldmu2 = bcpe_family$d2ldm2(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  d2ldsigma2 = bcpe_family$d2ldd2(
    bcpe_cases$sigma, bcpe_cases$tau
  ),
  d2ldnu2 = bcpe_family$d2ldv2(
    bcpe_cases$sigma, bcpe_cases$tau
  ),
  d2ldtau2 = bcpe_family$d2ldt2(
    bcpe_cases$y,
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  d2ldmudsigma = bcpe_family$d2ldmdd(
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  d2ldmudnu = bcpe_family$d2ldmdv(
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  d2ldmudtau = bcpe_family$d2ldmdt(
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  d2ldsigmadnu = bcpe_family$d2ldddv(
    bcpe_cases$sigma, bcpe_cases$nu, bcpe_cases$tau
  ),
  d2ldsigmadtau = bcpe_family$d2ldddt(
    bcpe_cases$sigma, bcpe_cases$tau
  ),
  d2ldnudtau = bcpe_family$d2ldvdt(
    bcpe_cases$mu,
    bcpe_cases$sigma,
    bcpe_cases$nu,
    bcpe_cases$tau
  ),
  initial_mu = (bcpe_cases$y + mean(bcpe_cases$y)) / 2,
  initial_sigma = rep(0.1, nrow(bcpe_cases)),
  initial_nu = rep(1, nrow(bcpe_cases)),
  initial_tau = rep(2, nrow(bcpe_cases)),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

continuous_quantile_reference <- function(
  family_code, cases, probabilities
) {
  data.frame(
    family = family_code,
    case_index = seq_len(nrow(cases)) - 1,
    uniform = 0.5,
    probability = probabilities,
    residual = qnorm(probabilities),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
}

discrete_quantile_reference <- function(
  family_code, cases, lower, upper
) {
  uniforms <- seq(0.1, 0.9, length.out = nrow(cases))
  probabilities <- lower + uniforms * (upper - lower)
  data.frame(
    family = family_code,
    case_index = seq_len(nrow(cases)) - 1,
    uniform = uniforms,
    probability = probabilities,
    residual = qnorm(probabilities),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
}

quantile_residual_reference <- do.call(rbind, list(
  continuous_quantile_reference(
    "NO", cases, pNO(cases$y, cases$mu, cases$sigma)
  ),
  continuous_quantile_reference(
    "GA", ga_cases, pGA(ga_cases$y, ga_cases$mu, ga_cases$sigma)
  ),
  discrete_quantile_reference(
    "PO",
    po_cases,
    pPO(po_cases$y - 1, po_cases$mu),
    pPO(po_cases$y, po_cases$mu)
  ),
  discrete_quantile_reference(
    "NBI",
    nbi_cases,
    pNBI(nbi_cases$y - 1, nbi_cases$mu, nbi_cases$sigma),
    pNBI(nbi_cases$y, nbi_cases$mu, nbi_cases$sigma)
  ),
  continuous_quantile_reference(
    "BE", be_cases, pBE(be_cases$y, be_cases$mu, be_cases$sigma)
  ),
  continuous_quantile_reference(
    "BCCG",
    bccg_cases,
    pBCCG(
      bccg_cases$y,
      bccg_cases$mu,
      bccg_cases$sigma,
      bccg_cases$nu
    )
  ),
  continuous_quantile_reference(
    "BCT",
    bct_cases,
    pBCT(
      bct_cases$y,
      bct_cases$mu,
      bct_cases$sigma,
      bct_cases$nu,
      bct_cases$tau
    )
  ),
  continuous_quantile_reference(
    "TF",
    tf_cases,
    pTF(
      tf_cases$y,
      tf_cases$mu,
      tf_cases$sigma,
      tf_cases$nu
    )
  ),
  continuous_quantile_reference(
    "PE",
    pe_cases,
    pPE(
      pe_cases$y,
      pe_cases$mu,
      pe_cases$sigma,
      pe_cases$nu
    )
  ),
  continuous_quantile_reference(
    "BCPE",
    bcpe_cases,
    pBCPE(
      bcpe_cases$y,
      bcpe_cases$mu,
      bcpe_cases$sigma,
      bcpe_cases$nu,
      bcpe_cases$tau
    )
  )
))

quantile_probabilities <- c(0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)

quantile_prediction_rows <- function(family_code, cases, quantiles) {
  probability_count <- length(quantile_probabilities)
  data.frame(
    family = family_code,
    case_index = rep(seq_len(nrow(cases)) - 1, each = probability_count),
    probability = rep(quantile_probabilities, times = nrow(cases)),
    quantile = quantiles,
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
}

expanded_probabilities <- function(cases) {
  rep(quantile_probabilities, times = nrow(cases))
}

expanded_parameter <- function(values) {
  rep(values, each = length(quantile_probabilities))
}

quantile_prediction_reference <- do.call(rbind, list(
  quantile_prediction_rows(
    "NO",
    cases,
    qNO(
      expanded_probabilities(cases),
      expanded_parameter(cases$mu),
      expanded_parameter(cases$sigma)
    )
  ),
  quantile_prediction_rows(
    "GA",
    ga_cases,
    qGA(
      expanded_probabilities(ga_cases),
      expanded_parameter(ga_cases$mu),
      expanded_parameter(ga_cases$sigma)
    )
  ),
  quantile_prediction_rows(
    "PO",
    po_cases,
    qPO(
      expanded_probabilities(po_cases),
      expanded_parameter(po_cases$mu)
    )
  ),
  quantile_prediction_rows(
    "NBI",
    nbi_cases,
    qNBI(
      expanded_probabilities(nbi_cases),
      expanded_parameter(nbi_cases$mu),
      expanded_parameter(nbi_cases$sigma)
    )
  ),
  quantile_prediction_rows(
    "BE",
    be_cases,
    qBE(
      expanded_probabilities(be_cases),
      expanded_parameter(be_cases$mu),
      expanded_parameter(be_cases$sigma)
    )
  ),
  quantile_prediction_rows(
    "BCCG",
    bccg_cases,
    qBCCG(
      expanded_probabilities(bccg_cases),
      expanded_parameter(bccg_cases$mu),
      expanded_parameter(bccg_cases$sigma),
      expanded_parameter(bccg_cases$nu)
    )
  ),
  quantile_prediction_rows(
    "BCT",
    bct_cases,
    qBCT(
      expanded_probabilities(bct_cases),
      expanded_parameter(bct_cases$mu),
      expanded_parameter(bct_cases$sigma),
      expanded_parameter(bct_cases$nu),
      expanded_parameter(bct_cases$tau)
    )
  ),
  quantile_prediction_rows(
    "TF",
    tf_cases,
    qTF(
      expanded_probabilities(tf_cases),
      expanded_parameter(tf_cases$mu),
      expanded_parameter(tf_cases$sigma),
      expanded_parameter(tf_cases$nu)
    )
  ),
  quantile_prediction_rows(
    "PE",
    pe_cases,
    qPE(
      expanded_probabilities(pe_cases),
      expanded_parameter(pe_cases$mu),
      expanded_parameter(pe_cases$sigma),
      expanded_parameter(pe_cases$nu)
    )
  ),
  quantile_prediction_rows(
    "BCPE",
    bcpe_cases,
    qBCPE(
      expanded_probabilities(bcpe_cases),
      expanded_parameter(bcpe_cases$mu),
      expanded_parameter(bcpe_cases$sigma),
      expanded_parameter(bcpe_cases$nu),
      expanded_parameter(bcpe_cases$tau)
    )
  )
))

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

fit_residuals <- resid(fit)
fit_qq <- qqnorm(fit_residuals, plot.it = FALSE)
fit_residual_variance <- var(fit_residuals)
fit_centered_residuals <- fit_residuals - mean(fit_residuals)
residual_plot_summary_reference <- data.frame(
  observation_count = length(fit_residuals),
  mean = mean(fit_residuals),
  variance = fit_residual_variance,
  skewness = mean(fit_centered_residuals^3) /
    fit_residual_variance^(3 / 2),
  kurtosis = mean(fit_centered_residuals^4) /
    fit_residual_variance^2,
  filliben_correlation = cor(fit_qq$y, fit_qq$x),
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

worm_count <- 40
worm_quantiles <- qnorm((seq_len(worm_count) - 0.5) / worm_count)
worm_residuals <- (
  worm_quantiles
  + 0.12
  + 0.08 * worm_quantiles^2
  - 0.03 * worm_quantiles^3
)
worm_x <- seq(-2, 2, length.out = worm_count)
pdf(file = NULL)
worm_global <- wp(resid = worm_residuals, line = TRUE)
invisible(capture.output(
  worm_conditioned <- wp(
    resid = worm_residuals,
    xvar = worm_x,
    n.inter = 4,
    overlap = 0,
    line = TRUE
  )
))
dev.off()
worm_plot_reference <- rbind(
  data.frame(
    scope = "global",
    panel = 1,
    lower = NA_real_,
    upper = NA_real_,
    intercept = unname(worm_global[[1]]),
    linear = unname(worm_global[[2]]),
    quadratic = unname(worm_global[[3]]),
    cubic = unname(worm_global[[4]])
  ),
  data.frame(
    scope = "conditioned",
    panel = seq_len(nrow(worm_conditioned$classes)),
    lower = worm_conditioned$classes[, 1],
    upper = worm_conditioned$classes[, 2],
    intercept = worm_conditioned$coef[, 1],
    linear = worm_conditioned$coef[, 2],
    quadratic = worm_conditioned$coef[, 3],
    cubic = worm_conditioned$coef[, 4]
  )
)
worm_plot_reference$gamlss_version <- as.character(packageVersion("gamlss"))
worm_plot_reference$gamlss_dist_version <- as.character(
  packageVersion("gamlss.dist")
)

bucket_count <- 80
bucket_probabilities <- (seq_len(bucket_count) - 0.5) / bucket_count
bucket_quantiles <- qnorm(bucket_probabilities)
bucket_residuals <- (
  bucket_quantiles
  + 0.14 * bucket_quantiles^2
  - 0.025 * bucket_quantiles^3
  + 0.04 * sin(seq(0, 4 * pi, length.out = bucket_count))
)
bucket_weights <- rep(c(1, 2, 0.5, 1.5), length.out = bucket_count)
bucket_x <- seq(-2, 2, length.out = bucket_count)

bucket_row <- function(
  kind,
  panel,
  lower,
  upper,
  residuals,
  weights
) {
  if (grepl("^moment", kind)) {
    statistics <- momentSK(residuals, weights = weights)
    skewness <- statistics$mom.skew
    transformed_skewness <- statistics$trans.mom.skew
    kurtosis <- statistics$mom.kurt
    excess_kurtosis <- statistics$excess.mom.kurt
    transformed_kurtosis <- statistics$trans.mom.kurt
    jarque_bera <- statistics$jarque.bera.test
  } else {
    statistics <- centileSK(residuals, weights = weights)
    skewness <- if (kind == "centile.central") {
      statistics$S0.25
    } else {
      statistics$S0.01
    }
    transformed_skewness <- skewness
    kurtosis <- statistics$K0.01
    excess_kurtosis <- statistics$exc.K0.01
    transformed_kurtosis <- statistics$trans.K0.01
    jarque_bera <- NA_real_
  }
  data.frame(
    kind = kind,
    panel = panel,
    lower = lower,
    upper = upper,
    observation_count = length(residuals),
    effective_observation_count = sum(weights),
    skewness = unname(skewness),
    transformed_skewness = unname(transformed_skewness),
    kurtosis = unname(kurtosis),
    excess_kurtosis = unname(excess_kurtosis),
    transformed_kurtosis = unname(transformed_kurtosis),
    jarque_bera = unname(jarque_bera)
  )
}

bucket_classes <- co.intervals(bucket_x, number = 4, overlap = 0)
for (index in seq_len(nrow(bucket_classes) - 1)) {
  if (
    abs(bucket_classes[index, 2] - bucket_classes[index + 1, 1]) >=
      1e-4
  ) {
    bucket_classes[index + 1, 1] <- bucket_classes[index, 2]
  }
}
bucket_plot_reference <- do.call(
  rbind,
  c(
    lapply(
      c("moment", "centile.central", "centile.tail"),
      function(kind) {
        bucket_row(
          kind,
          1,
          NA_real_,
          NA_real_,
          bucket_residuals,
          bucket_weights
        )
      }
    ),
    lapply(seq_len(nrow(bucket_classes)), function(panel) {
      in_panel <- (
        bucket_x >= bucket_classes[panel, 1]
        & bucket_x <= bucket_classes[panel, 2]
      )
      bucket_row(
        "moment.conditioned",
        panel,
        bucket_classes[panel, 1],
        bucket_classes[panel, 2],
        bucket_residuals[in_panel],
        bucket_weights[in_panel]
      )
    })
  )
)
bucket_plot_reference$gamlss_version <- as.character(
  packageVersion("gamlss")
)
bucket_plot_reference$gamlss_dist_version <- as.character(
  packageVersion("gamlss.dist")
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

po_n <- 72
po_x <- seq(-1, 1, length.out = po_n)
po_mu_offset <- 0.12 * sin(seq(0, 3 * pi, length.out = po_n))
po_weight <- rep(c(1, 1.5, 2, 0.75), length.out = po_n)
po_mu <- exp(0.55 + 0.65 * po_x + po_mu_offset)
po_probability <- (((seq_len(po_n) * 37) %% po_n) + 0.5) / po_n
po_fit_data <- data.frame(
  x = po_x,
  y = qPO(po_probability, mu = po_mu),
  weight = po_weight,
  mu_offset = po_mu_offset
)
po_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  weights = weight,
  family = PO(),
  method = RS(),
  data = po_fit_data,
  control = gamlss.control(c.crit = 1e-10, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-10, cyc = 200)
)
po_rs_reference <- data.frame(
  mu_intercept = unname(coef(po_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(po_rs_fit, what = "mu")[[2]]),
  global_deviance = unname(deviance(po_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(po_rs_fit)),
  outer_iterations = po_rs_fit$iter,
  converged = po_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

nbi_n <- 84
nbi_x <- seq(-1, 1, length.out = nbi_n)
nbi_z <- cos(seq(0, 2 * pi, length.out = nbi_n))
nbi_mu_offset <- 0.08 * sin(seq(0, 3 * pi, length.out = nbi_n))
nbi_sigma_offset <- 0.05 * cos(seq(0, 4 * pi, length.out = nbi_n))
nbi_weight <- rep(c(1, 1.25, 1.75, 0.8), length.out = nbi_n)
nbi_mu <- exp(0.75 + 0.55 * nbi_x + nbi_mu_offset)
nbi_sigma <- exp(-1.1 + 0.25 * nbi_z + nbi_sigma_offset)
nbi_probability <- (((seq_len(nbi_n) * 43) %% nbi_n) + 0.5) / nbi_n
nbi_fit_data <- data.frame(
  x = nbi_x,
  z = nbi_z,
  y = qNBI(nbi_probability, mu = nbi_mu, sigma = nbi_sigma),
  weight = nbi_weight,
  mu_offset = nbi_mu_offset,
  sigma_offset = nbi_sigma_offset
)
nbi_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = NBI(),
  method = RS(),
  data = nbi_fit_data,
  control = gamlss.control(c.crit = 1e-9, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-9, cyc = 200)
)
nbi_rs_reference <- data.frame(
  mu_intercept = unname(coef(nbi_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(nbi_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(nbi_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(nbi_rs_fit, what = "sigma")[[2]]),
  global_deviance = unname(deviance(nbi_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(nbi_rs_fit)),
  outer_iterations = nbi_rs_fit$iter,
  converged = nbi_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

be_n <- 80
be_x <- seq(-1, 1, length.out = be_n)
be_z <- cos(seq(0, 2 * pi, length.out = be_n))
be_mu_offset <- 0.07 * sin(seq(0, 3 * pi, length.out = be_n))
be_sigma_offset <- 0.04 * cos(seq(0, 4 * pi, length.out = be_n))
be_weight <- rep(c(1, 1.5, 2, 0.75), length.out = be_n)
be_mu <- plogis(-0.15 + 1.05 * be_x + be_mu_offset)
be_sigma <- plogis(-1.25 + 0.2 * be_z + be_sigma_offset)
be_probability <- (((seq_len(be_n) * 37) %% be_n) + 0.5) / be_n
be_fit_data <- data.frame(
  x = be_x,
  z = be_z,
  y = qBE(be_probability, mu = be_mu, sigma = be_sigma),
  weight = be_weight,
  mu_offset = be_mu_offset,
  sigma_offset = be_sigma_offset
)
be_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = RS(),
  data = be_fit_data,
  control = gamlss.control(c.crit = 1e-9, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-9, cyc = 200)
)
be_rs_reference <- data.frame(
  mu_intercept = unname(coef(be_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(be_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(be_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(be_rs_fit, what = "sigma")[[2]]),
  global_deviance = unname(deviance(be_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(be_rs_fit)),
  outer_iterations = be_rs_fit$iter,
  converged = be_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

be_cg_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = CG(),
  data = be_fit_data,
  control = gamlss.control(c.crit = 1e-9, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-9, cyc = 200)
)
be_cg_reference <- data.frame(
  mu_intercept = unname(coef(be_cg_fit, what = "mu")[[1]]),
  mu_x = unname(coef(be_cg_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(be_cg_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(be_cg_fit, what = "sigma")[[2]]),
  global_deviance = unname(deviance(be_cg_fit)),
  negative_log_likelihood = -as.numeric(logLik(be_cg_fit)),
  outer_iterations = be_cg_fit$iter,
  converged = be_cg_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bccg_n <- 120
bccg_x <- seq(-1, 1, length.out = bccg_n)
bccg_z <- cos(seq(0, 2 * pi, length.out = bccg_n))
bccg_w <- sin(seq(0, 2 * pi, length.out = bccg_n))
bccg_mu_offset <- 0.08 * cos(seq(0, 3 * pi, length.out = bccg_n))
bccg_sigma_offset <- 0.03 * sin(seq(0, 4 * pi, length.out = bccg_n))
bccg_nu_offset <- 0.04 * cos(seq(0, 5 * pi, length.out = bccg_n))
bccg_weight <- rep(c(1, 1.5, 2, 0.75), length.out = bccg_n)
bccg_mu <- 3 + 0.7 * bccg_x + bccg_mu_offset
bccg_sigma <- exp(-1.5 + 0.18 * bccg_z + bccg_sigma_offset)
bccg_nu <- 0.35 + 0.2 * bccg_w + bccg_nu_offset
bccg_probability <- (
  ((seq_len(bccg_n) * 47) %% bccg_n) + 0.5
) / bccg_n
bccg_fit_data <- data.frame(
  x = bccg_x,
  z = bccg_z,
  w = bccg_w,
  y = qBCCG(
    bccg_probability,
    mu = bccg_mu,
    sigma = bccg_sigma,
    nu = bccg_nu
  ),
  weight = bccg_weight,
  mu_offset = bccg_mu_offset,
  sigma_offset = bccg_sigma_offset,
  nu_offset = bccg_nu_offset
)
bccg_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  weights = weight,
  family = BCCG(),
  method = RS(),
  data = bccg_fit_data,
  control = gamlss.control(c.crit = 1e-9, n.cyc = 300, trace = FALSE),
  i.control = glim.control(cc = 1e-9, cyc = 300)
)
bccg_rs_reference <- data.frame(
  mu_intercept = unname(coef(bccg_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(bccg_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(bccg_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(bccg_rs_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(bccg_rs_fit, what = "nu")[[1]]),
  nu_w = unname(coef(bccg_rs_fit, what = "nu")[[2]]),
  global_deviance = unname(deviance(bccg_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(bccg_rs_fit)),
  outer_iterations = bccg_rs_fit$iter,
  converged = bccg_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bct_n <- 160
bct_x <- seq(-1, 1, length.out = bct_n)
bct_z <- cos(seq(0, 2 * pi, length.out = bct_n))
bct_w <- sin(seq(0, 2 * pi, length.out = bct_n))
bct_mu_offset <- 0.08 * cos(seq(0, 3 * pi, length.out = bct_n))
bct_sigma_offset <- 0.03 * sin(seq(0, 4 * pi, length.out = bct_n))
bct_nu_offset <- 0.04 * cos(seq(0, 5 * pi, length.out = bct_n))
bct_weight <- rep(c(1, 1.5, 2, 0.75), length.out = bct_n)
bct_mu <- 3 + 0.7 * bct_x + bct_mu_offset
bct_sigma <- exp(-1.5 + 0.18 * bct_z + bct_sigma_offset)
bct_nu <- 0.35 + 0.2 * bct_w + bct_nu_offset
bct_tau <- 4
bct_probability <- (((seq_len(bct_n) * 67) %% bct_n) + 0.5) / bct_n
bct_fit_data <- data.frame(
  x = bct_x,
  z = bct_z,
  w = bct_w,
  y = qBCT(
    bct_probability,
    mu = bct_mu,
    sigma = bct_sigma,
    nu = bct_nu,
    tau = bct_tau
  ),
  weight = bct_weight,
  mu_offset = bct_mu_offset,
  sigma_offset = bct_sigma_offset,
  nu_offset = bct_nu_offset
)
bct_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  tau.formula = ~ 1,
  weights = weight,
  family = BCT(),
  method = RS(),
  data = bct_fit_data,
  control = gamlss.control(c.crit = 1e-7, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-7, cyc = 200)
)
bct_rs_reference <- data.frame(
  mu_intercept = unname(coef(bct_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(bct_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(bct_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(bct_rs_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(bct_rs_fit, what = "nu")[[1]]),
  nu_w = unname(coef(bct_rs_fit, what = "nu")[[2]]),
  tau_intercept = unname(coef(bct_rs_fit, what = "tau")[[1]]),
  global_deviance = unname(deviance(bct_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(bct_rs_fit)),
  outer_iterations = bct_rs_fit$iter,
  converged = bct_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bct_laml_fixed_fit <- gamlss(
  y ~ pb(x, lambda = 10),
  sigma.formula = ~ 1,
  nu.formula = ~ 1,
  tau.formula = ~ 1,
  weights = weight,
  family = BCT(),
  method = RS(),
  data = bct_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 300, trace = FALSE),
  i.control = glim.control(
    cc = 1e-8,
    cyc = 300,
    bf.tol = 1e-8,
    bf.cyc = 300,
    glm.trace = FALSE
  )
)
bct_laml_fixed_smooth <- getSmo(
  bct_laml_fixed_fit,
  parameter = "mu",
  which = 1
)
bct_laml_fixed_reference <- data.frame(
  global_deviance = unname(deviance(bct_laml_fixed_fit)),
  negative_log_likelihood = -as.numeric(logLik(bct_laml_fixed_fit)),
  smoothing_parameter = bct_laml_fixed_smooth$lambda,
  outer_iterations = bct_laml_fixed_fit$iter,
  converged = bct_laml_fixed_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)
bct_laml_fixed_fitted_reference <- data.frame(
  mu = fitted(bct_laml_fixed_fit, what = "mu"),
  sigma = fitted(bct_laml_fixed_fit, what = "sigma"),
  nu = fitted(bct_laml_fixed_fit, what = "nu"),
  tau = fitted(bct_laml_fixed_fit, what = "tau")
)

bct_cg_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  tau.formula = ~ 1,
  weights = weight,
  family = BCT(),
  method = CG(),
  data = bct_fit_data,
  mu.start = bct_mu,
  sigma.start = bct_sigma,
  nu.start = bct_nu,
  tau.start = bct_tau,
  control = gamlss.control(
    c.crit = 1e-9,
    n.cyc = 200,
    trace = FALSE,
    autostep = FALSE
  ),
  i.control = glim.control(cc = 1e-9, cyc = 200)
)
bct_cg_reference <- data.frame(
  mu_intercept = unname(coef(bct_cg_fit, what = "mu")[[1]]),
  mu_x = unname(coef(bct_cg_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(bct_cg_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(bct_cg_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(bct_cg_fit, what = "nu")[[1]]),
  nu_w = unname(coef(bct_cg_fit, what = "nu")[[2]]),
  tau_intercept = unname(coef(bct_cg_fit, what = "tau")[[1]]),
  global_deviance = unname(deviance(bct_cg_fit)),
  negative_log_likelihood = -as.numeric(logLik(bct_cg_fit)),
  outer_iterations = bct_cg_fit$iter,
  converged = bct_cg_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

tf_n <- 160
tf_x <- seq(-1, 1, length.out = tf_n)
tf_z <- cos(seq(0, 2 * pi, length.out = tf_n))
tf_w <- sin(seq(0, 2 * pi, length.out = tf_n))
tf_mu_offset <- 0.08 * cos(seq(0, 3 * pi, length.out = tf_n))
tf_sigma_offset <- 0.03 * sin(seq(0, 4 * pi, length.out = tf_n))
tf_nu_offset <- 0.04 * cos(seq(0, 5 * pi, length.out = tf_n))
tf_weight <- rep(c(1, 1.5, 2, 0.75), length.out = tf_n)
tf_mu <- 1.5 + 0.9 * tf_x + tf_mu_offset
tf_sigma <- exp(-0.55 + 0.18 * tf_z + tf_sigma_offset)
tf_nu <- exp(log(5) + 0.25 * tf_w + tf_nu_offset)
tf_probability <- (((seq_len(tf_n) * 67) %% tf_n) + 0.5) / tf_n
tf_fit_data <- data.frame(
  x = tf_x,
  z = tf_z,
  w = tf_w,
  y = qTF(
    tf_probability,
    mu = tf_mu,
    sigma = tf_sigma,
    nu = tf_nu
  ),
  weight = tf_weight,
  mu_offset = tf_mu_offset,
  sigma_offset = tf_sigma_offset,
  nu_offset = tf_nu_offset
)
tf_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  weights = weight,
  family = TF(),
  method = RS(),
  data = tf_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 300, trace = FALSE),
  i.control = glim.control(cc = 1e-8, cyc = 300)
)
tf_rs_reference <- data.frame(
  mu_intercept = unname(coef(tf_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(tf_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(tf_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(tf_rs_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(tf_rs_fit, what = "nu")[[1]]),
  nu_w = unname(coef(tf_rs_fit, what = "nu")[[2]]),
  global_deviance = unname(deviance(tf_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(tf_rs_fit)),
  outer_iterations = tf_rs_fit$iter,
  converged = tf_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

tf_cg_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  weights = weight,
  family = TF(),
  method = CG(),
  data = tf_fit_data,
  mu.start = tf_mu,
  sigma.start = tf_sigma,
  nu.start = tf_nu,
  control = gamlss.control(
    c.crit = 1e-9,
    n.cyc = 200,
    trace = FALSE,
    autostep = FALSE
  ),
  i.control = glim.control(cc = 1e-9, cyc = 200)
)
tf_cg_reference <- data.frame(
  mu_intercept = unname(coef(tf_cg_fit, what = "mu")[[1]]),
  mu_x = unname(coef(tf_cg_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(tf_cg_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(tf_cg_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(tf_cg_fit, what = "nu")[[1]]),
  nu_w = unname(coef(tf_cg_fit, what = "nu")[[2]]),
  global_deviance = unname(deviance(tf_cg_fit)),
  negative_log_likelihood = -as.numeric(logLik(tf_cg_fit)),
  outer_iterations = tf_cg_fit$iter,
  converged = tf_cg_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

pe_n <- 160
pe_x <- seq(-1, 1, length.out = pe_n)
pe_z <- cos(seq(0, 2 * pi, length.out = pe_n))
pe_w <- sin(seq(0, 2 * pi, length.out = pe_n))
pe_mu_offset <- 0.08 * cos(seq(0, 3 * pi, length.out = pe_n))
pe_sigma_offset <- 0.03 * sin(seq(0, 4 * pi, length.out = pe_n))
pe_nu_offset <- 0.04 * cos(seq(0, 5 * pi, length.out = pe_n))
pe_weight <- rep(c(1, 1.5, 2, 0.75), length.out = pe_n)
pe_mu <- 1.5 + 0.9 * pe_x + pe_mu_offset
pe_sigma <- exp(-0.55 + 0.18 * pe_z + pe_sigma_offset)
pe_nu <- exp(log(1.6) + 0.22 * pe_w + pe_nu_offset)
pe_probability <- (((seq_len(pe_n) * 67) %% pe_n) + 0.5) / pe_n
pe_fit_data <- data.frame(
  x = pe_x,
  z = pe_z,
  w = pe_w,
  y = qPE(
    pe_probability,
    mu = pe_mu,
    sigma = pe_sigma,
    nu = pe_nu
  ),
  weight = pe_weight,
  mu_offset = pe_mu_offset,
  sigma_offset = pe_sigma_offset,
  nu_offset = pe_nu_offset
)
pe_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  weights = weight,
  family = PE(),
  method = RS(),
  data = pe_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 300, trace = FALSE),
  i.control = glim.control(cc = 1e-8, cyc = 300)
)
pe_rs_reference <- data.frame(
  mu_intercept = unname(coef(pe_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pe_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pe_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(pe_rs_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(pe_rs_fit, what = "nu")[[1]]),
  nu_w = unname(coef(pe_rs_fit, what = "nu")[[2]]),
  global_deviance = unname(deviance(pe_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(pe_rs_fit)),
  outer_iterations = pe_rs_fit$iter,
  converged = pe_rs_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

pe_cg_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  weights = weight,
  family = PE(),
  method = CG(),
  data = pe_fit_data,
  mu.start = pe_mu,
  sigma.start = pe_sigma,
  nu.start = pe_nu,
  control = gamlss.control(
    c.crit = 1e-9,
    n.cyc = 200,
    trace = FALSE,
    autostep = FALSE
  ),
  i.control = glim.control(cc = 1e-9, cyc = 200)
)
pe_cg_reference <- data.frame(
  mu_intercept = unname(coef(pe_cg_fit, what = "mu")[[1]]),
  mu_x = unname(coef(pe_cg_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(pe_cg_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(pe_cg_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(pe_cg_fit, what = "nu")[[1]]),
  nu_w = unname(coef(pe_cg_fit, what = "nu")[[2]]),
  global_deviance = unname(deviance(pe_cg_fit)),
  negative_log_likelihood = -as.numeric(logLik(pe_cg_fit)),
  outer_iterations = pe_cg_fit$iter,
  converged = pe_cg_fit$converged,
  gamlss_version = as.character(packageVersion("gamlss")),
  gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
)

bcpe_n <- 160
bcpe_x <- seq(-1, 1, length.out = bcpe_n)
bcpe_z <- cos(seq(0, 2 * pi, length.out = bcpe_n))
bcpe_w <- sin(seq(0, 2 * pi, length.out = bcpe_n))
bcpe_mu_offset <- 0.08 * cos(seq(0, 3 * pi, length.out = bcpe_n))
bcpe_sigma_offset <- 0.03 * sin(seq(0, 4 * pi, length.out = bcpe_n))
bcpe_nu_offset <- 0.04 * cos(seq(0, 5 * pi, length.out = bcpe_n))
bcpe_weight <- rep(c(1, 1.5, 2, 0.75), length.out = bcpe_n)
bcpe_mu <- 3 + 0.7 * bcpe_x + bcpe_mu_offset
bcpe_sigma <- exp(-1.5 + 0.18 * bcpe_z + bcpe_sigma_offset)
bcpe_nu <- 0.35 + 0.2 * bcpe_w + bcpe_nu_offset
bcpe_tau <- 1.5
bcpe_probability <- (
  ((seq_len(bcpe_n) * 67) %% bcpe_n) + 0.5
) / bcpe_n
bcpe_fit_data <- data.frame(
  x = bcpe_x,
  z = bcpe_z,
  w = bcpe_w,
  y = qBCPE(
    bcpe_probability,
    mu = bcpe_mu,
    sigma = bcpe_sigma,
    nu = bcpe_nu,
    tau = bcpe_tau
  ),
  weight = bcpe_weight,
  mu_offset = bcpe_mu_offset,
  sigma_offset = bcpe_sigma_offset,
  nu_offset = bcpe_nu_offset
)
bcpe_rs_fit <- gamlss(
  y ~ x + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  nu.formula = ~ w + offset(nu_offset),
  tau.formula = ~ 1,
  weights = weight,
  family = BCPE(),
  method = RS(),
  data = bcpe_fit_data,
  control = gamlss.control(c.crit = 1e-7, n.cyc = 200, trace = FALSE),
  i.control = glim.control(cc = 1e-7, cyc = 200)
)
bcpe_rs_reference <- data.frame(
  mu_intercept = unname(coef(bcpe_rs_fit, what = "mu")[[1]]),
  mu_x = unname(coef(bcpe_rs_fit, what = "mu")[[2]]),
  sigma_intercept = unname(coef(bcpe_rs_fit, what = "sigma")[[1]]),
  sigma_z = unname(coef(bcpe_rs_fit, what = "sigma")[[2]]),
  nu_intercept = unname(coef(bcpe_rs_fit, what = "nu")[[1]]),
  nu_w = unname(coef(bcpe_rs_fit, what = "nu")[[2]]),
  tau_intercept = unname(coef(bcpe_rs_fit, what = "tau")[[1]]),
  global_deviance = unname(deviance(bcpe_rs_fit)),
  negative_log_likelihood = -as.numeric(logLik(bcpe_rs_fit)),
  outer_iterations = bcpe_rs_fit$iter,
  converged = bcpe_rs_fit$converged,
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

inference_reference <- function(
  fit, family_code, parameter_names, term_names
) {
  inference <- vcov(fit, type = "all", hessian.fun = "R")
  coefficient_names <- paste(parameter_names, term_names, sep = ".")
  estimates <- as.numeric(inference$coef)
  standard_errors <- as.numeric(inference$se)
  degrees_of_freedom <- fit$df.residual
  statistics <- estimates / standard_errors
  p_values <- 2 * pt(-abs(statistics), degrees_of_freedom)
  critical_value <- qt(0.975, degrees_of_freedom)
  table <- data.frame(
    family = family_code,
    coefficient_index = seq_along(estimates) - 1,
    coefficient = coefficient_names,
    estimate = estimates,
    standard_error = standard_errors,
    statistic = statistics,
    p_value = p_values,
    ci_lower = estimates - critical_value * standard_errors,
    ci_upper = estimates + critical_value * standard_errors,
    degrees_of_freedom = degrees_of_freedom,
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
  covariance_grid <- expand.grid(
    row_index = seq_along(estimates) - 1,
    column_index = seq_along(estimates) - 1
  )
  covariance <- data.frame(
    family = family_code,
    covariance_grid,
    covariance = as.vector(inference$vcov),
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
  list(table = table, covariance = covariance)
}

inference_results <- list(
  inference_reference(
    rs_fit,
    "NO",
    c("mu", "mu", "sigma", "sigma"),
    c("Intercept", "x", "Intercept", "z")
  ),
  inference_reference(
    po_rs_fit,
    "PO",
    c("mu", "mu"),
    c("Intercept", "x")
  ),
  inference_reference(
    nbi_rs_fit,
    "NBI",
    c("mu", "mu", "sigma", "sigma"),
    c("Intercept", "x", "Intercept", "z")
  ),
  inference_reference(
    be_rs_fit,
    "BE",
    c("mu", "mu", "sigma", "sigma"),
    c("Intercept", "x", "Intercept", "z")
  ),
  inference_reference(
    bccg_rs_fit,
    "BCCG",
    c("mu", "mu", "sigma", "sigma", "nu", "nu"),
    c("Intercept", "x", "Intercept", "z", "Intercept", "w")
  ),
  inference_reference(
    bct_rs_fit,
    "BCT",
    c("mu", "mu", "sigma", "sigma", "nu", "nu", "tau"),
    c("Intercept", "x", "Intercept", "z", "Intercept", "w", "Intercept")
  ),
  inference_reference(
    tf_rs_fit,
    "TF",
    c("mu", "mu", "sigma", "sigma", "nu", "nu"),
    c("Intercept", "x", "Intercept", "z", "Intercept", "w")
  ),
  inference_reference(
    pe_rs_fit,
    "PE",
    c("mu", "mu", "sigma", "sigma", "nu", "nu"),
    c("Intercept", "x", "Intercept", "z", "Intercept", "w")
  ),
  inference_reference(
    bcpe_rs_fit,
    "BCPE",
    c("mu", "mu", "sigma", "sigma", "nu", "nu", "tau"),
    c("Intercept", "x", "Intercept", "z", "Intercept", "w", "Intercept")
  )
)
inference_table_reference <- do.call(
  rbind, lapply(inference_results, function(result) result$table)
)
inference_covariance_reference <- do.call(
  rbind, lapply(inference_results, function(result) result$covariance)
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

diagnostic_reference <- function(fit, family_code) {
  data.frame(
    family = family_code,
    observation_count = fit$N,
    effective_observation_count = fit$noObs,
    effective_df = fit$df.fit,
    residual_df = fit$df.residual,
    log_likelihood = as.numeric(logLik(fit)),
    global_deviance = unname(deviance(fit)),
    aic = unname(GAIC(fit, k = 2)),
    aicc = unname(GAIC(fit, k = 2, c = TRUE)),
    gaic3 = unname(GAIC(fit, k = 3)),
    sbc = unname(GAIC(fit, k = log(fit$noObs))),
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
}

model_diagnostics_reference <- do.call(rbind, list(
  diagnostic_reference(rs_fit, "NO"),
  diagnostic_reference(ga_rs_fit, "GA"),
  diagnostic_reference(po_rs_fit, "PO"),
  diagnostic_reference(nbi_rs_fit, "NBI"),
  diagnostic_reference(be_rs_fit, "BE"),
  diagnostic_reference(bccg_rs_fit, "BCCG"),
  diagnostic_reference(bct_rs_fit, "BCT"),
  diagnostic_reference(tf_rs_fit, "TF"),
  diagnostic_reference(pe_rs_fit, "PE"),
  diagnostic_reference(bcpe_rs_fit, "BCPE"),
  diagnostic_reference(pb_fit, "NO_PB")
))

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

cg_smooth_control <- gamlss.control(
  c.crit = 1e-8,
  n.cyc = 300,
  trace = FALSE
)
cg_smooth_inner_control <- glim.control(
  cc = 1e-8,
  cyc = 300,
  bf.tol = 1e-8,
  bf.cyc = 300
)
be_cg_pb_fit <- gamlss(
  y ~ pb(x, lambda = 12) + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = CG(),
  data = be_fit_data,
  control = gamlss.control(c.crit = 1e-7, n.cyc = 300, trace = FALSE),
  i.control = glim.control(
    cc = 1e-7,
    cyc = 300,
    bf.tol = 1e-7,
    bf.cyc = 300
  )
)
no_cg_fixed_fit <- gamlss(
  y ~ pb(x, lambda = 12),
  sigma.formula = ~ 1,
  family = NO(),
  method = CG(),
  data = pb_fit_data,
  control = cg_smooth_control,
  i.control = cg_smooth_inner_control
)
no_cg_ml_fit <- gamlss(
  y ~ pb(x),
  sigma.formula = ~ 1,
  family = NO(),
  method = CG(),
  data = pb_fit_data,
  control = cg_smooth_control,
  i.control = cg_smooth_inner_control
)
no_cg_df_fit <- gamlss(
  y ~ pb(x, df = 3),
  sigma.formula = ~ 1,
  family = NO(),
  method = CG(),
  data = pb_fit_data,
  control = cg_smooth_control,
  i.control = cg_smooth_inner_control
)
no_cg_gaic_fit <- gamlss(
  y ~ pb(x, control = pb.control(method = "GAIC", k = 2)),
  sigma.formula = ~ 1,
  family = NO(),
  method = CG(),
  data = pb_fit_data,
  control = cg_smooth_control,
  i.control = cg_smooth_inner_control
)
no_cg_gcv_fit <- gamlss(
  y ~ pb(x, control = pb.control(method = "GCV", k = 2)),
  sigma.formula = ~ 1,
  family = NO(),
  method = CG(),
  data = pb_fit_data,
  control = cg_smooth_control,
  i.control = cg_smooth_inner_control
)
cg_smooth_fits <- list(
  BE_FIXED = be_cg_pb_fit,
  NO_FIXED = no_cg_fixed_fit,
  NO_ML = no_cg_ml_fit,
  NO_DF = no_cg_df_fit,
  NO_GAIC = no_cg_gaic_fit,
  NO_GCV = no_cg_gcv_fit
)
cg_smooth_family <- c("BE", rep("NO", 5))
cg_smooth_selection <- c("FIXED", "FIXED", "ML", "DF", "GAIC", "GCV")
cg_smooth_reference <- do.call(
  rbind,
  lapply(seq_along(cg_smooth_fits), function(index) {
    fit <- cg_smooth_fits[[index]]
    smooth <- getSmo(fit, parameter = "mu", which = 1)
    data.frame(
      case = names(cg_smooth_fits)[[index]],
      family = cg_smooth_family[[index]],
      selection = cg_smooth_selection[[index]],
      smoothing_parameter = unname(smooth$lambda),
      smooth_edf = unname(smooth$edf),
      mu_df = fit$mu.df,
      total_df = fit$df.fit,
      global_deviance = unname(deviance(fit)),
      negative_log_likelihood = -as.numeric(logLik(fit)),
      outer_iterations = fit$iter,
      converged = fit$converged,
      gamlss_version = as.character(packageVersion("gamlss")),
      gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
    )
  })
)
cg_smooth_linear_reference <- do.call(
  rbind,
  lapply(names(cg_smooth_fits), function(case_name) {
    fit <- cg_smooth_fits[[case_name]]
    do.call(
      rbind,
      lapply(fit$parameters, function(parameter) {
        coefficients <- unname(coef(fit, what = parameter))
        data.frame(
          case = case_name,
          parameter = parameter,
          coefficient_index = seq_along(coefficients) - 1,
          coefficient = coefficients
        )
      })
    )
  })
)
cg_smooth_fitted_reference <- do.call(
  rbind,
  lapply(names(cg_smooth_fits), function(case_name) {
    fit <- cg_smooth_fits[[case_name]]
    data.frame(
      case = case_name,
      observation_index = seq_along(fitted(fit, what = "mu")) - 1,
      mu = fitted(fit, what = "mu"),
      sigma = fitted(fit, what = "sigma"),
      mu_smooth = drop(fit$mu.s[, 1])
    )
  })
)
cg_smooth_coefficient_reference <- do.call(
  rbind,
  lapply(names(cg_smooth_fits), function(case_name) {
    coefficients <- drop(
      getSmo(
        cg_smooth_fits[[case_name]],
        parameter = "mu",
        which = 1
      )$coef
    )
    data.frame(
      case = case_name,
      coefficient_index = seq_along(coefficients) - 1,
      coefficient = coefficients
    )
  })
)
rownames(cg_smooth_reference) <- NULL
rownames(cg_smooth_linear_reference) <- NULL
rownames(cg_smooth_fitted_reference) <- NULL
rownames(cg_smooth_coefficient_reference) <- NULL

cg_multi_smooth_control <- gamlss.control(
  c.crit = 1e-9, n.cyc = 500, trace = FALSE
)
cg_multi_smooth_inner_control <- glim.control(
  cc = 1e-9, cyc = 500, bf.tol = 1e-9, bf.cyc = 500
)
cg_multi_smooth_n <- 80
cg_multi_smooth_x <- seq(-1, 1, length.out = cg_multi_smooth_n)
cg_multi_smooth_z <- cos(
  seq(0, 4 * pi, length.out = cg_multi_smooth_n)
)
cg_multi_smooth_mu_offset <- 0.03 * sin(
  seq(0, 3 * pi, length.out = cg_multi_smooth_n)
)
cg_multi_smooth_sigma_offset <- 0.025 * cos(
  seq(0, 5 * pi, length.out = cg_multi_smooth_n)
)
cg_multi_smooth_weight <- rep(
  c(1, 1.5, 2, 0.75), length.out = cg_multi_smooth_n
)
cg_multi_smooth_mu <- plogis(
  -0.2 + 0.75 * cg_multi_smooth_x +
    0.6 * sin(pi * cg_multi_smooth_x) +
    cg_multi_smooth_mu_offset
)
cg_multi_smooth_sigma <- plogis(
  -1.5 + 0.18 * cg_multi_smooth_z +
    0.55 * cos(pi * cg_multi_smooth_z) +
    cg_multi_smooth_sigma_offset
)
cg_multi_smooth_probability <- (
  ((seq_len(cg_multi_smooth_n) * 37) %% cg_multi_smooth_n) + 0.5
) / cg_multi_smooth_n
cg_multi_smooth_fit_data <- data.frame(
  x = cg_multi_smooth_x,
  z = cg_multi_smooth_z,
  y = qBE(
    cg_multi_smooth_probability,
    mu = cg_multi_smooth_mu,
    sigma = cg_multi_smooth_sigma
  ),
  weight = cg_multi_smooth_weight,
  mu_offset = cg_multi_smooth_mu_offset,
  sigma_offset = cg_multi_smooth_sigma_offset
)
be_cg_both_smooth_fit <- gamlss(
  y ~ pb(x, lambda = 12) + offset(mu_offset),
  sigma.formula = ~ pb(z, lambda = 15) + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = CG(),
  data = be_fit_data,
  control = cg_multi_smooth_control,
  i.control = cg_multi_smooth_inner_control
)
be_cg_two_mu_smooth_fit <- gamlss(
  y ~ pb(x, lambda = 12) + pb(z, lambda = 15) + offset(mu_offset),
  sigma.formula = ~ z + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = CG(),
  data = be_fit_data,
  control = cg_multi_smooth_control,
  i.control = cg_multi_smooth_inner_control
)
be_cg_both_ml_smooth_fit <- gamlss(
  y ~ pb(x) + offset(mu_offset),
  sigma.formula = ~ pb(z) + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = CG(),
  data = cg_multi_smooth_fit_data,
  control = gamlss.control(c.crit = 1e-8, n.cyc = 300, trace = FALSE),
  i.control = glim.control(
    cc = 1e-8, cyc = 300, bf.tol = 1e-8, bf.cyc = 300
  )
)
cg_multi_smooth_fits <- list(
  BOTH_PARAMETERS = be_cg_both_smooth_fit,
  TWO_MU_TERMS = be_cg_two_mu_smooth_fit,
  BOTH_ML = be_cg_both_ml_smooth_fit
)
cg_multi_smooth_specs <- list(
  BOTH_PARAMETERS = data.frame(
    parameter = c("mu", "sigma"),
    term = c("x", "z"),
    which = c(1, 1),
    selection = c("FIXED", "FIXED")
  ),
  TWO_MU_TERMS = data.frame(
    parameter = c("mu", "mu"),
    term = c("x", "z"),
    which = c(1, 2),
    selection = c("FIXED", "FIXED")
  ),
  BOTH_ML = data.frame(
    parameter = c("mu", "sigma"),
    term = c("x", "z"),
    which = c(1, 1),
    selection = c("ML", "ML")
  )
)
cg_multi_smooth_reference <- do.call(
  rbind,
  lapply(names(cg_multi_smooth_fits), function(case_name) {
    fit <- cg_multi_smooth_fits[[case_name]]
    data.frame(
      case = case_name,
      mu_df = fit$mu.df,
      sigma_df = fit$sigma.df,
      total_df = fit$df.fit,
      global_deviance = unname(deviance(fit)),
      negative_log_likelihood = -as.numeric(logLik(fit)),
      outer_iterations = fit$iter,
      converged = fit$converged,
      gamlss_version = as.character(packageVersion("gamlss")),
      gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
    )
  })
)
cg_multi_smooth_linear_reference <- do.call(
  rbind,
  lapply(names(cg_multi_smooth_fits), function(case_name) {
    fit <- cg_multi_smooth_fits[[case_name]]
    do.call(
      rbind,
      lapply(fit$parameters, function(parameter) {
        coefficients <- unname(coef(fit, what = parameter))
        data.frame(
          case = case_name,
          parameter = parameter,
          coefficient_index = seq_along(coefficients) - 1,
          coefficient = coefficients
        )
      })
    )
  })
)
cg_multi_smooth_fitted_reference <- do.call(
  rbind,
  lapply(names(cg_multi_smooth_fits), function(case_name) {
    fit <- cg_multi_smooth_fits[[case_name]]
    data.frame(
      case = case_name,
      observation_index = seq_along(fitted(fit, what = "mu")) - 1,
      mu = fitted(fit, what = "mu"),
      sigma = fitted(fit, what = "sigma")
    )
  })
)
cg_multi_smooth_term_reference <- do.call(
  rbind,
  lapply(names(cg_multi_smooth_fits), function(case_name) {
    fit <- cg_multi_smooth_fits[[case_name]]
    specs <- cg_multi_smooth_specs[[case_name]]
    do.call(
      rbind,
      lapply(seq_len(nrow(specs)), function(index) {
        parameter <- specs$parameter[[index]]
        smooth <- getSmo(
          fit,
          parameter = parameter,
          which = specs$which[[index]]
        )
        data.frame(
          case = case_name,
          parameter = parameter,
          term = specs$term[[index]],
          selection = specs$selection[[index]],
          smoothing_parameter = unname(smooth$lambda),
          smooth_edf = unname(smooth$edf)
        )
      })
    )
  })
)
cg_multi_smooth_coefficient_reference <- do.call(
  rbind,
  lapply(names(cg_multi_smooth_fits), function(case_name) {
    fit <- cg_multi_smooth_fits[[case_name]]
    specs <- cg_multi_smooth_specs[[case_name]]
    do.call(
      rbind,
      lapply(seq_len(nrow(specs)), function(index) {
        parameter <- specs$parameter[[index]]
        coefficients <- drop(
          getSmo(
            fit,
            parameter = parameter,
            which = specs$which[[index]]
          )$coef
        )
        data.frame(
          case = case_name,
          parameter = parameter,
          term = specs$term[[index]],
          coefficient_index = seq_along(coefficients) - 1,
          coefficient = coefficients
        )
      })
    )
  })
)
cg_multi_smooth_contribution_reference <- do.call(
  rbind,
  lapply(names(cg_multi_smooth_fits), function(case_name) {
    fit <- cg_multi_smooth_fits[[case_name]]
    specs <- cg_multi_smooth_specs[[case_name]]
    do.call(
      rbind,
      lapply(seq_len(nrow(specs)), function(index) {
        parameter <- specs$parameter[[index]]
        contribution <- drop(
          fit[[paste0(parameter, ".s")]][, specs$which[[index]]]
        )
        data.frame(
          case = case_name,
          parameter = parameter,
          term = specs$term[[index]],
          observation_index = seq_along(contribution) - 1,
          contribution = contribution
        )
      })
    )
  })
)
rownames(cg_multi_smooth_reference) <- NULL
rownames(cg_multi_smooth_linear_reference) <- NULL
rownames(cg_multi_smooth_fitted_reference) <- NULL
rownames(cg_multi_smooth_term_reference) <- NULL
rownames(cg_multi_smooth_coefficient_reference) <- NULL
rownames(cg_multi_smooth_contribution_reference) <- NULL

conditional_inference_results <- list(
  suppressWarnings(inference_reference(
    pb_fit,
    "NO_FIXED_RS",
    c("mu", "mu", "sigma"),
    c("Intercept", "pb(x lambda=12)", "Intercept")
  )),
  suppressWarnings(inference_reference(
    pb_ml_fit,
    "NO_ML_RS",
    c("mu", "mu", "sigma"),
    c("Intercept", "pb(x)", "Intercept")
  )),
  suppressWarnings(inference_reference(
    be_cg_pb_fit,
    "BE_FIXED_CG",
    c("mu", "mu", "sigma", "sigma"),
    c("Intercept", "pb(x lambda=12)", "Intercept", "z")
  ))
)
conditional_inference_table_reference <- do.call(
  rbind,
  lapply(conditional_inference_results, function(result) result$table)
)
conditional_inference_covariance_reference <- do.call(
  rbind,
  lapply(conditional_inference_results, function(result) result$covariance)
)
names(conditional_inference_table_reference)[
  names(conditional_inference_table_reference) == "family"
] <- "case"
names(conditional_inference_covariance_reference)[
  names(conditional_inference_covariance_reference) == "family"
] <- "case"
rownames(conditional_inference_table_reference) <- NULL
rownames(conditional_inference_covariance_reference) <- NULL

smooth_inference_reference <- function(
  fit, case_name, parameter, term_name, which
) {
  contribution <- drop(
    fit[[paste0(parameter, ".s")]][, which]
  )
  variance <- drop(
    fit[[paste0(parameter, ".var")]][, which]
  )
  standard_error <- sqrt(variance)
  critical_value <- qnorm(0.975)
  smooth <- getSmo(fit, parameter = parameter, which = which)
  data.frame(
    case = case_name,
    parameter = parameter,
    term = term_name,
    observation_index = seq_along(contribution) - 1,
    estimate = contribution,
    variance = variance,
    standard_error = standard_error,
    ci_lower = contribution - critical_value * standard_error,
    ci_upper = contribution + critical_value * standard_error,
    smoothing_parameter = unname(smooth$lambda),
    smooth_edf = unname(smooth$edf),
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
  )
}
smooth_inference_results <- list(
  smooth_inference_reference(pb_fit, "NO_FIXED_RS", "mu", "x", 1),
  smooth_inference_reference(pb_ml_fit, "NO_ML_RS", "mu", "x", 1)
)
for (case_name in names(cg_multi_smooth_fits)) {
  fit <- cg_multi_smooth_fits[[case_name]]
  specs <- cg_multi_smooth_specs[[case_name]]
  for (index in seq_len(nrow(specs))) {
    smooth_inference_results <- c(
      smooth_inference_results,
      list(smooth_inference_reference(
        fit,
        case_name,
        specs$parameter[[index]],
        specs$term[[index]],
        specs$which[[index]]
      ))
    )
  }
}
smooth_inference_reference_data <- do.call(
  rbind,
  smooth_inference_results
)
rownames(smooth_inference_reference_data) <- NULL

right_null_space <- function(matrix, expected_nullity, context) {
  decomposition <- svd(matrix, nu = 0, nv = ncol(matrix))
  tolerance <- max(dim(matrix)) * .Machine$double.eps * max(decomposition$d)
  rank <- sum(decomposition$d > tolerance)
  nullity <- ncol(matrix) - rank
  if (nullity != expected_nullity) {
    stop(
      context, " has nullity ", nullity,
      ", expected ", expected_nullity
    )
  }
  decomposition$v[, seq.int(rank + 1, ncol(matrix)), drop = FALSE]
}

joint_pb_normal_covariance_reference <- function(fit, case_name) {
  smooth_variable <- model.frame(fit)[[2]]
  basis <- as.matrix(attr(smooth_variable, "X"))
  penalty <- as.matrix(attr(smooth_variable, "D"))
  coefficient_count <- ncol(basis)
  penalty_nullity <- coefficient_count - nrow(penalty)
  penalty_null_space <- right_null_space(
    penalty,
    penalty_nullity,
    "R pb penalty"
  )
  case_weights <- fit$weights
  sigma <- fitted(fit, what = "sigma")
  mu_information <- case_weights / sigma^2
  sigma_information <- 2 * case_weights
  null_functions <- basis %*% penalty_null_space
  constraints <- t(null_functions) %*% (mu_information * basis)
  transform <- right_null_space(
    constraints,
    coefficient_count - penalty_nullity,
    "R pb identifiability constraints"
  )

  mu_linear_design <- as.matrix(fit$mu.x)
  sigma_linear_design <- as.matrix(fit$sigma.x)
  mu_reduced_design <- cbind(mu_linear_design, basis %*% transform)
  mu_reduced_count <- ncol(mu_reduced_design)
  sigma_reduced_count <- ncol(sigma_linear_design)
  reduced_count <- mu_reduced_count + sigma_reduced_count
  information <- matrix(0, nrow = reduced_count, ncol = reduced_count)
  information[
    seq_len(mu_reduced_count),
    seq_len(mu_reduced_count)
  ] <- t(mu_reduced_design) %*% (mu_information * mu_reduced_design)
  sigma_indices <- seq.int(mu_reduced_count + 1, reduced_count)
  information[sigma_indices, sigma_indices] <-
    t(sigma_linear_design) %*% (
      sigma_information * sigma_linear_design
    )
  smooth_reduced_indices <- seq.int(
    ncol(mu_linear_design) + 1,
    mu_reduced_count
  )
  information[
    smooth_reduced_indices,
    smooth_reduced_indices
  ] <- information[
    smooth_reduced_indices,
    smooth_reduced_indices
  ] + unname(getSmo(fit, parameter = "mu", which = 1)$lambda) *
    t(transform) %*% t(penalty) %*% penalty %*% transform
  reduced_covariance <- solve(information)

  full_count <- ncol(mu_linear_design) +
    coefficient_count +
    ncol(sigma_linear_design)
  coefficient_transform <- matrix(
    0,
    nrow = full_count,
    ncol = reduced_count
  )
  mu_linear_indices <- seq_len(ncol(mu_linear_design))
  coefficient_transform[mu_linear_indices, mu_linear_indices] <-
    diag(ncol(mu_linear_design))
  full_smooth_indices <- seq.int(
    ncol(mu_linear_design) + 1,
    ncol(mu_linear_design) + coefficient_count
  )
  coefficient_transform[
    full_smooth_indices,
    smooth_reduced_indices
  ] <- transform
  full_sigma_indices <- seq.int(
    ncol(mu_linear_design) + coefficient_count + 1,
    full_count
  )
  coefficient_transform[full_sigma_indices, sigma_indices] <-
    diag(ncol(sigma_linear_design))
  coefficient_covariance <- coefficient_transform %*%
    reduced_covariance %*%
    t(coefficient_transform)
  curve_covariance <- basis %*%
    coefficient_covariance[full_smooth_indices, full_smooth_indices] %*%
    t(basis)

  coefficient_grid <- expand.grid(
    row_index = seq_len(full_count) - 1,
    column_index = seq_len(full_count) - 1
  )
  curve_grid <- expand.grid(
    row_index = seq_len(nrow(basis)) - 1,
    column_index = seq_len(nrow(basis)) - 1
  )
  list(
    coefficient = data.frame(
      case = case_name,
      coefficient_grid,
      covariance = as.vector(coefficient_covariance),
      gamlss_version = as.character(packageVersion("gamlss")),
      gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
    ),
    curve = data.frame(
      case = case_name,
      parameter = "mu",
      term = "x",
      curve_grid,
      covariance = as.vector(curve_covariance),
      gamlss_version = as.character(packageVersion("gamlss")),
      gamlss_dist_version = as.character(packageVersion("gamlss.dist"))
    )
  )
}

joint_smooth_reference <- joint_pb_normal_covariance_reference(
  pb_fit,
  "NO_FIXED_RS"
)
joint_smooth_coefficient_covariance_reference <-
  joint_smooth_reference$coefficient
joint_smooth_curve_covariance_reference <- joint_smooth_reference$curve

assert_close <- function(
  actual,
  expected_path,
  tolerance,
  ignored_numeric_columns = character()
) {
  expected <- read.csv(expected_path, check.names = FALSE)
  if (!identical(names(actual), names(expected))) {
    stop("Reference columns differ in ", expected_path)
  }

  unknown_ignored_columns <- setdiff(ignored_numeric_columns, names(actual))
  if (length(unknown_ignored_columns) > 0) {
    stop(
      "Unknown ignored numeric columns for ",
      expected_path,
      ": ",
      paste(unknown_ignored_columns, collapse = ", ")
    )
  }

  numeric_columns <- vapply(actual, is.numeric, logical(1))
  invalid_ignored_columns <- ignored_numeric_columns[
    !numeric_columns[ignored_numeric_columns]
  ]
  if (length(invalid_ignored_columns) > 0) {
    stop(
      "Ignored columns must be numeric for ",
      expected_path,
      ": ",
      paste(invalid_ignored_columns, collapse = ", ")
    )
  }

  compared_numeric_columns <- numeric_columns &
    !(names(actual) %in% ignored_numeric_columns)
  character_columns <- !numeric_columns
  if (any(compared_numeric_columns)) {
    numeric_names <- names(actual)[compared_numeric_columns]
    actual_numeric <- as.matrix(actual[compared_numeric_columns])
    expected_numeric <- as.matrix(expected[compared_numeric_columns])
    difference <- abs(actual_numeric - expected_numeric)
    allowed_difference <- tolerance * (1 + abs(expected_numeric))
    failures <- which(difference > allowed_difference, arr.ind = TRUE)

    if (nrow(failures) > 0) {
      row_index <- failures[1, "row"]
      column_index <- failures[1, "col"]
      column_name <- numeric_names[[column_index]]
      row_label <- if ("case" %in% names(actual)) {
        paste0(actual$case[[row_index]], " (row ", row_index, ")")
      } else {
        as.character(row_index)
      }
      stop(
        "Numeric parity check failed for ",
        expected_path,
        ": case/row ",
        row_label,
        ", column ",
        column_name,
        ", actual=",
        format(actual_numeric[row_index, column_index], digits = 17),
        ", expected=",
        format(expected_numeric[row_index, column_index], digits = 17),
        ", absolute difference=",
        format(difference[row_index, column_index], digits = 17),
        ", allowed=",
        format(allowed_difference[row_index, column_index], digits = 17)
      )
    }
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
  assert_close(
    cg_smooth_reference,
    cg_smooth_reference_path,
    tolerance = 1e-6,
    ignored_numeric_columns = "outer_iterations"
  )
  assert_close(
    cg_smooth_linear_reference,
    cg_smooth_linear_reference_path,
    tolerance = 1e-6
  )
  assert_close(
    cg_smooth_fitted_reference,
    cg_smooth_fitted_reference_path,
    tolerance = 1e-5
  )
  assert_close(
    cg_smooth_coefficient_reference,
    cg_smooth_coefficient_reference_path,
    tolerance = 1e-5
  )
  assert_close(
    cg_multi_smooth_reference,
    cg_multi_smooth_reference_path,
    tolerance = 1e-7,
    ignored_numeric_columns = "outer_iterations"
  )
  assert_close(
    cg_multi_smooth_fit_data,
    cg_multi_smooth_fit_data_path,
    tolerance = 1e-12
  )
  assert_close(
    cg_multi_smooth_linear_reference,
    cg_multi_smooth_linear_reference_path,
    tolerance = 1e-7
  )
  assert_close(
    cg_multi_smooth_fitted_reference,
    cg_multi_smooth_fitted_reference_path,
    tolerance = 1e-7
  )
  assert_close(
    cg_multi_smooth_term_reference,
    cg_multi_smooth_term_reference_path,
    tolerance = 1e-7
  )
  assert_close(
    cg_multi_smooth_coefficient_reference,
    cg_multi_smooth_coefficient_reference_path,
    tolerance = 1e-7
  )
  assert_close(
    cg_multi_smooth_contribution_reference,
    cg_multi_smooth_contribution_reference_path,
    tolerance = 1e-7
  )
  assert_close(po_reference, po_reference_path, tolerance = 1e-12)
  assert_close(po_fit_data, po_fit_data_path, tolerance = 1e-12)
  assert_close(po_rs_reference, po_rs_reference_path, tolerance = 1e-7)
  assert_close(nbi_reference, nbi_reference_path, tolerance = 1e-12)
  assert_close(nbi_fit_data, nbi_fit_data_path, tolerance = 1e-12)
  assert_close(nbi_rs_reference, nbi_rs_reference_path, tolerance = 1e-6)
  assert_close(be_reference, be_reference_path, tolerance = 1e-12)
  assert_close(be_fit_data, be_fit_data_path, tolerance = 1e-12)
  assert_close(be_rs_reference, be_rs_reference_path, tolerance = 1e-6)
  assert_close(be_cg_reference, be_cg_reference_path, tolerance = 1e-9)
  assert_close(bccg_reference, bccg_reference_path, tolerance = 1e-10)
  assert_close(bccg_fit_data, bccg_fit_data_path, tolerance = 1e-12)
  assert_close(bccg_rs_reference, bccg_rs_reference_path, tolerance = 1e-6)
  assert_close(bct_reference, bct_reference_path, tolerance = 1e-9)
  assert_close(bct_fit_data, bct_fit_data_path, tolerance = 1e-12)
  assert_close(bct_rs_reference, bct_rs_reference_path, tolerance = 1e-6)
  assert_close(
    bct_laml_fixed_reference,
    bct_laml_fixed_reference_path,
    tolerance = 3e-6
  )
  assert_close(
    bct_laml_fixed_fitted_reference,
    bct_laml_fixed_fitted_reference_path,
    tolerance = 3e-6
  )
  assert_close(bct_cg_reference, bct_cg_reference_path, tolerance = 1e-8)
  assert_close(tf_reference, tf_reference_path, tolerance = 1e-10)
  assert_close(tf_fit_data, tf_fit_data_path, tolerance = 1e-12)
  assert_close(tf_rs_reference, tf_rs_reference_path, tolerance = 1e-6)
  assert_close(tf_cg_reference, tf_cg_reference_path, tolerance = 1e-8)
  assert_close(pe_reference, pe_reference_path, tolerance = 1e-9)
  assert_close(pe_fit_data, pe_fit_data_path, tolerance = 1e-12)
  assert_close(pe_rs_reference, pe_rs_reference_path, tolerance = 1e-6)
  assert_close(pe_cg_reference, pe_cg_reference_path, tolerance = 1e-8)
  assert_close(bcpe_reference, bcpe_reference_path, tolerance = 1e-9)
  assert_close(bcpe_fit_data, bcpe_fit_data_path, tolerance = 1e-12)
  assert_close(bcpe_rs_reference, bcpe_rs_reference_path, tolerance = 1e-6)
  assert_close(
    inference_table_reference, inference_table_path, tolerance = 1e-6
  )
  assert_close(
    inference_covariance_reference,
    inference_covariance_path,
    tolerance = 1e-6
  )
  assert_close(
    conditional_inference_table_reference,
    conditional_inference_table_path,
    tolerance = 1e-6
  )
  assert_close(
    conditional_inference_covariance_reference,
    conditional_inference_covariance_path,
    tolerance = 1e-6
  )
  assert_close(
    smooth_inference_reference_data,
    smooth_inference_path,
    tolerance = 1e-6
  )
  assert_close(
    joint_smooth_coefficient_covariance_reference,
    joint_smooth_coefficient_covariance_path,
    tolerance = 1e-10
  )
  assert_close(
    joint_smooth_curve_covariance_reference,
    joint_smooth_curve_covariance_path,
    tolerance = 1e-10
  )
  assert_close(
    model_diagnostics_reference,
    model_diagnostics_path,
    tolerance = 1e-6
  )
  assert_close(
    residual_plot_summary_reference,
    residual_plot_summary_path,
    tolerance = 1e-12
  )
  assert_close(worm_plot_reference, worm_plot_path, tolerance = 1e-12)
  assert_close(bucket_plot_reference, bucket_plot_path, tolerance = 1e-12)
  assert_close(
    quantile_residual_reference,
    quantile_residual_path,
    tolerance = 1e-12
  )
  assert_close(
    quantile_prediction_reference,
    quantile_prediction_path,
    tolerance = 1e-10
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
  write_csv_lf(cg_smooth_reference, cg_smooth_reference_path)
  write_csv_lf(
    cg_smooth_linear_reference,
    cg_smooth_linear_reference_path
  )
  write_csv_lf(
    cg_smooth_fitted_reference,
    cg_smooth_fitted_reference_path
  )
  write_csv_lf(
    cg_smooth_coefficient_reference,
    cg_smooth_coefficient_reference_path
  )
  write_csv_lf(
    cg_multi_smooth_reference,
    cg_multi_smooth_reference_path
  )
  write_csv_lf(
    cg_multi_smooth_fit_data,
    cg_multi_smooth_fit_data_path
  )
  write_csv_lf(
    cg_multi_smooth_linear_reference,
    cg_multi_smooth_linear_reference_path
  )
  write_csv_lf(
    cg_multi_smooth_fitted_reference,
    cg_multi_smooth_fitted_reference_path
  )
  write_csv_lf(
    cg_multi_smooth_term_reference,
    cg_multi_smooth_term_reference_path
  )
  write_csv_lf(
    cg_multi_smooth_coefficient_reference,
    cg_multi_smooth_coefficient_reference_path
  )
  write_csv_lf(
    cg_multi_smooth_contribution_reference,
    cg_multi_smooth_contribution_reference_path
  )
  write_csv_lf(po_reference, po_reference_path)
  write_csv_lf(po_fit_data, po_fit_data_path)
  write_csv_lf(po_rs_reference, po_rs_reference_path)
  write_csv_lf(nbi_reference, nbi_reference_path)
  write_csv_lf(nbi_fit_data, nbi_fit_data_path)
  write_csv_lf(nbi_rs_reference, nbi_rs_reference_path)
  write_csv_lf(be_reference, be_reference_path)
  write_csv_lf(be_fit_data, be_fit_data_path)
  write_csv_lf(be_rs_reference, be_rs_reference_path)
  write_csv_lf(be_cg_reference, be_cg_reference_path)
  write_csv_lf(bccg_reference, bccg_reference_path)
  write_csv_lf(bccg_fit_data, bccg_fit_data_path)
  write_csv_lf(bccg_rs_reference, bccg_rs_reference_path)
  write_csv_lf(bct_reference, bct_reference_path)
  write_csv_lf(bct_fit_data, bct_fit_data_path)
  write_csv_lf(bct_rs_reference, bct_rs_reference_path)
  write_csv_lf(bct_laml_fixed_reference, bct_laml_fixed_reference_path)
  write_csv_lf(
    bct_laml_fixed_fitted_reference,
    bct_laml_fixed_fitted_reference_path
  )
  write_csv_lf(bct_cg_reference, bct_cg_reference_path)
  write_csv_lf(tf_reference, tf_reference_path)
  write_csv_lf(tf_fit_data, tf_fit_data_path)
  write_csv_lf(tf_rs_reference, tf_rs_reference_path)
  write_csv_lf(tf_cg_reference, tf_cg_reference_path)
  write_csv_lf(pe_reference, pe_reference_path)
  write_csv_lf(pe_fit_data, pe_fit_data_path)
  write_csv_lf(pe_rs_reference, pe_rs_reference_path)
  write_csv_lf(pe_cg_reference, pe_cg_reference_path)
  write_csv_lf(bcpe_reference, bcpe_reference_path)
  write_csv_lf(bcpe_fit_data, bcpe_fit_data_path)
  write_csv_lf(bcpe_rs_reference, bcpe_rs_reference_path)
  write_csv_lf(inference_table_reference, inference_table_path)
  write_csv_lf(inference_covariance_reference, inference_covariance_path)
  write_csv_lf(
    conditional_inference_table_reference,
    conditional_inference_table_path
  )
  write_csv_lf(
    conditional_inference_covariance_reference,
    conditional_inference_covariance_path
  )
  write_csv_lf(smooth_inference_reference_data, smooth_inference_path)
  write_csv_lf(
    joint_smooth_coefficient_covariance_reference,
    joint_smooth_coefficient_covariance_path
  )
  write_csv_lf(
    joint_smooth_curve_covariance_reference,
    joint_smooth_curve_covariance_path
  )
  write_csv_lf(model_diagnostics_reference, model_diagnostics_path)
  write_csv_lf(
    residual_plot_summary_reference,
    residual_plot_summary_path
  )
  write_csv_lf(worm_plot_reference, worm_plot_path)
  write_csv_lf(bucket_plot_reference, bucket_plot_path)
  write_csv_lf(quantile_residual_reference, quantile_residual_path)
  write_csv_lf(quantile_prediction_reference, quantile_prediction_path)
  message("Wrote R reference fixtures to ", reference_dir)
}
