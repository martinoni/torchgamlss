# Changelog

All notable changes to TorchGAMLSS are documented in this file.

The project follows semantic versioning and uses the Python packaging
pre-release notation described by PEP 440.

## [0.1.0a3.dev0] - 2026-07-28

Development cycle toward the third alpha release.

### Added

- A generic dense penalized weighted least-squares solver supporting multiple
  fixed coefficient-space penalties, linear constraints through null-space
  reparameterization, effective degrees of freedom, rank diagnostics, and
  CPU/CUDA execution while preserving the classical `pb()` fitting path.
- Full tensor-product smooths and highest-order ANOVA tensor interactions with
  row-wise Kronecker designs, one fixed penalty per marginal direction,
  explicit centering transforms, prediction-state round trips, CPU/CUDA
  execution, and exact algebraic checks against `mgcv`.
- Fixed-lambda `te()` and `ti()` formula terms with per-margin basis options,
  exact absorbed identifiability constraints, stored prediction transforms,
  RS/CG backfitting through the generic constrained solver, full-batch
  L-BFGS, mini-batch Adam, and CUDA execution.
- Conditional and joint analytic tensor inference with combined
  multiple-penalty information, constrained coefficient covariance, new-grid
  prediction, multivariate tables, and simultaneous Gaussian bands.
- RS/CG parametric smooth bootstrap for multiply penalized tensor terms, with
  backward-compatible scalar results, one replicate column per tensor
  penalty, penalty-level joint labels and slices, and multivariate tables.
- Dense whole-model LAML for additive Normal location-scale models, with
  formula and tensor-level `fit_laml` APIs, jointly selected scalar and tensor
  log lambdas, structural null-space constraints, model-state updates,
  penalty-level result labels, CPU/CUDA execution, and direct
  `mgcv::gaulss(method="REML")` tensor-product parity.
- Weibull (`WEI`) and log-normal (`LOGNO`) event-time families with
  `gamlss.dist` parity for density, CDF, quantiles, survival functions,
  hazards, moments, scores, working derivatives, and starting values.
- Inverse-Gaussian (`IG`) and generalized-gamma (`GG`) event-time families,
  including stable `GG` log-normal-limit series, Torch-native differentiable
  tails, sampling, moments, RS/L-BFGS fitting, and CUDA coverage.
- `CensoredResponse` and `CensoredFamily` for exact, right-, left-, and
  interval-censored continuous responses, with `survival::Surv` status-code
  compatibility and `gamlss.cens` likelihood/score parity.
- Model prediction of survival, hazard, and cumulative-hazard curves through
  `predict_survival()` and `predict_survival_data()`, including CUDA coverage.
- A complete right-censored `GG` parity example comparing fitted parameters,
  latent quantiles, survival curves, hazards, and cumulative hazards with R.
- Observation-specific continuous and discrete truncation bounds through fixed
  one-dimensional tensors, with `gamlss.tr` `varying=TRUE` parity plus formula
  fitting, sampling, quantile, autograd, and CUDA coverage.
- Scalar and observation-specific truncation parity across the complete
  translated family catalog, including differentiable Torch CDFs for Gamma,
  Beta, and NBI normalizers and finite BCPE shape gradients at `nu=0`.

### Changed

- Full-batch Torch likelihood evaluation now calls the family likelihood
  interface, allowing composed likelihoods such as censoring to participate
  correctly in L-BFGS fitting.

## [0.1.0a2] - 2026-07-28

Second installable alpha release.

### Added

- A manually dispatched, annotated-tag-verified TestPyPI trusted-publishing
  workflow using short-lived OpenID Connect credentials and an approval
  environment instead of a repository API token.
- A reusable manifest-driven R-to-Python parity runner with standardized CSV
  artifacts, column-specific tolerances, tolerant numeric keys, and
  machine-readable failure reports.
- A complete weighted Normal location-scale RS example covering offsets,
  coefficients, fitted parameters, response quantiles, quantile residuals,
  likelihood criteria, and a diagnostic visualization.
- A weighted Poisson-versus-NBI count-regression example covering modeled
  overdispersion, information-criterion weights, fitted moments, discrete
  quantiles, and reproducible randomized Dunn-Smyth residuals.
- A smooth BCCG fetal-growth example covering location, scale, and shape
  P-splines, nine response-centile curves, and continuous quantile residuals.
- The Student-t location-scale family (`TF`) with Torch-native differentiable
  density and CDF, R-compatible scores and Fisher-scoring derivatives,
  sampling, quantiles, weighted RS fitting, inference, diagnostics, and CUDA
  stress coverage.
- The power-exponential location-scale-shape family (`PE`) with
  standard-deviation scale, tail shape, differentiable density and CDF,
  R-compatible RS and CG derivatives, inference, diagnostics, sampling,
  quantiles, and CUDA FP16/BF16 stress coverage.
- Fixed-bound continuous and discrete truncation through `TruncatedFamily`,
  with differentiable Normal and Poisson normalization, left/right/two-sided
  `gamlss.tr` parity, sampling, RS/L-BFGS fitting, and CUDA coverage.
- Committed R results for running the example parity tests without an R
  installation, while continuous integration also executes both languages.
- Continuous-integration retention of the end-to-end parity report, tables,
  metadata, and plot.
- Public-project contribution, security, conduct, citation, issue, and pull
  request guidance.

### Changed

- Published `0.1.0a1` to TestPyPI through the approved trusted-publishing
  workflow and verified the indexed wheel in an isolated installation.

## [0.1.0a1] - 2026-07-26

First installable alpha release.

### Added

- Normal, Gamma, Poisson, negative-binomial type I, Beta, BCCG, BCT, and BCPE
  response families with R-generated numerical parity fixtures.
- Formula and tensor interfaces with independent predictors, links, weights,
  offsets, P-splines, neural predictors, and shared representations.
- Rigby-Stasinopoulos and Cole-Green fitting, plus a Torch-native L-BFGS
  baseline and streaming Adam optimization.
- Fixed, ML, target-EDF, GAIC, and GCV smoothing-parameter selection.
- CUDA mini-batch fitting, FP16/BF16 automatic mixed precision, checkpointing,
  resumption, and validation early stopping.
- Parametric and smooth inference, simultaneous bands, joint bootstrap
  covariance, derived smooth functionals, and response-scale quantile bands.
- Model-selection diagnostics, randomized quantile residuals, four-panel
  residual plots, worm plots, and bucket plots.
- Python 3.10-3.13 test matrix on Linux and Windows.
- Wheel and source-distribution validation with a clean-environment smoke
  test.

### Known limitations

- This is an alpha release and is not intended for production or high-stakes
  statistical analysis without independent validation.
- Only one-dimensional, equally spaced P-spline bases are implemented.
- The bucket-plot family-locus overlays stored in R-specific serialized assets
  are not bundled.
- Production PyPI publication has not been enabled; the alpha is available
  from TestPyPI and the GitHub pre-release.
