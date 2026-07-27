# Roadmap

TorchGAMLSS will grow in verified vertical slices. Each slice must include
automated comparisons with the reference R implementation before it is treated
as supported.

## Phase 0 — foundation

- [x] Package skeleton and GPL-3.0-only licensing
- [x] Link protocol and identity, inverse, log, and logit links
- [x] Normal family with independent `mu` and `sigma` predictors
- [x] Differentiable full-batch negative log-likelihood
- [x] R reference-data generator
- [x] Numerical tolerances and parity-test conventions

## Phase 1 — parametric core

- [x] Weighted observations and offsets
- [x] Formula/design-matrix interface
- [x] Starting-value protocol
- [x] Normal, Gamma, Poisson, beta, and negative-binomial families
- [x] Initial joint L-BFGS optimizer and convergence diagnostics
- [x] Covariance matrix, standard errors, and basic Wald inference

## Phase 2 — classical GAMLSS fitting

- [x] Linear unpenalized RS core
- [x] RS backfitting with smooth and penalized terms
- [x] CG algorithm with cross-parameter working weights
- [x] CG backfitting and smoothing-parameter updates
- [x] Penalized weighted least squares
- [x] P-splines with fixed smoothing parameters
- [x] Automatic ML smoothing-parameter selection
- [x] Target-EDF smoothing-parameter selection
- [x] GAIC and GCV smoothing-parameter selection
- [x] Effective degrees of freedom for fixed-lambda P-splines

## Phase 3 — compatibility and diagnostics

- [x] Predictions on link, response, and term scales
- [x] Randomized quantile residuals
- [x] Deviance, AIC, AICc, GAIC, SBC/BIC, and model comparison
- [x] Three-parameter BCCG family
- [x] First four-parameter family: BCT
- [x] BCPE four-parameter family
- [x] R-to-Python API and numerical compatibility guide
- [x] Within-curve covariance and conditional simultaneous smooth bands
- [x] Parametric smooth bootstrap and bands with repeated lambda selection
- [x] Joint bootstrap covariance and simultaneous multi-smooth bands
- [x] Bootstrap smooth contrasts, derivatives, extrema, and crossings
- [x] Response-scale quantile/centile prediction and bootstrap bands
- [x] Analytic joint covariance for penalized smooth terms

## Phase 4 — Torch-native extensions

- [x] Mini-batch optimization for large data
- [x] Neural predictors for selected distribution parameters
- [x] Shared representations across distribution parameters
- [x] Holdout validation and out-of-sample early stopping
- [x] CPU/GPU benchmarks and reproducibility guidance
- [x] Streaming `Dataset`/`DataLoader` inputs with exact weighted objectives
- [x] Atomic checkpoints and exact epoch-boundary resumption
- [x] CUDA automatic mixed precision for neural mini-batch fitting
- [x] FP16/BF16 tail-stability matrix for non-normal response families
- [x] Family-aware starting values for tensor, formula, and streaming
  mini-batch fits

## Phase 5 — graphical diagnostics

- [x] `plot()` model diagnostic panels compatible with the R workflow
- [x] `wp()` global and numeric covariate-conditioned worm plots
- [x] `bp()` weighted moment and centile bucket plots with bootstrap
- [ ] Optional R family-locus overlays for `bp()` backgrounds

## Phase 6 — alpha release engineering

- [x] Single-source PEP 440 version and Alpha package metadata
- [x] Python 3.10-3.13 CI on Linux and Windows
- [x] Automated wheel and source-distribution build
- [x] Strict package metadata and README validation
- [x] Clean-environment wheel installation and API smoke test
- [x] Changelog and documented private pre-release procedure
- [x] Tag and create the private `v0.1.0a1` GitHub pre-release
- [ ] Decide whether and when to publish on TestPyPI or PyPI

## Phase 7 — reproducible parity examples

- [x] Reusable declarative R-to-Python parity harness
- [x] Complete weighted Normal location-scale example with coefficients,
  quantiles, residuals, and graphical output
- [x] Run the complete example in R and Python in CI and retain its report
- [ ] Count-regression example comparing Poisson and negative binomial models
- [ ] Response-centile example using a Box-Cox family
- [ ] Expand the translated family catalog beyond the initial eight families
