# Changelog

All notable changes to TorchGAMLSS are documented in this file.

The project follows semantic versioning and uses the Python packaging
pre-release notation described by PEP 440.

## [0.1.0a3.dev0] - 2026-07-28

Development cycle toward the third alpha release.

### Added

- Observation-specific continuous and discrete truncation bounds through fixed
  one-dimensional tensors, with `gamlss.tr` `varying=TRUE` parity plus formula
  fitting, sampling, quantile, autograd, and CUDA coverage.

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
