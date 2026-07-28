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
- [x] Symmetric power-exponential location-scale-shape family (`PE`)
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
- [x] Changelog and documented GitHub pre-release procedure
- [x] Tag and create the `v0.1.0a1` GitHub pre-release
- [x] Public-project community, security, citation, and issue metadata
- [x] Publish the repository and protect the default branch
- [x] Adopt TestPyPI-first trusted publishing with production PyPI gated
- [ ] Complete the first TestPyPI publication and clean-install smoke test

## Phase 7 — reproducible parity examples

- [x] Reusable declarative R-to-Python parity harness
- [x] Complete weighted Normal location-scale example with coefficients,
  quantiles, residuals, and graphical output
- [x] Run the complete example in R and Python in CI and retain its report
- [x] Count-regression example comparing Poisson and negative binomial models
- [x] Run the count-model comparison in R and Python in CI and retain its
  report
- [x] Response-centile example using a Box-Cox family
- [x] Expand the translated family catalog beyond the initial eight families
  with the Student-t location-scale family (`TF`)
- [x] Expand the translated family catalog to ten families with the symmetric
  power-exponential family (`PE`)

## Phase 8 — family composition and truncation

- [x] Generic derived-family protocol with differentiable normalization
- [x] Fixed-bound `TruncatedFamily` for continuous responses
- [x] Fixed-bound `TruncatedFamily` for discrete responses
- [x] R parity for truncated Normal and Poisson families
- [ ] Observation-specific truncation bounds compatible with `gamlss.tr`
- [ ] Extend truncation parity across the translated family catalog

## Phase 9 — survival and censoring

- [ ] Survival-oriented Weibull, log-normal, inverse-Gaussian, and generalized
  gamma families
- [ ] Right-, left-, and interval-censored response representation
- [ ] Censored likelihood composition compatible with `gamlss.cens`
- [ ] Survival, hazard, and cumulative-hazard prediction
- [ ] Reproducible censored-survival parity example

## Phase 10 — inflated and adjusted distributions

- [ ] Generic point-mass composition at zero and one
- [ ] Zero-inflated Poisson and negative-binomial families
- [ ] Zero/one-inflated and adjusted beta families
- [ ] R parity against `gamlss.inf` and `gamlss.dist`
- [ ] Diagnostics for fitted boundary masses

## Phase 11 — finite mixtures

- [ ] Mixture-family protocol with stable log-sum-exp likelihoods
- [ ] Parameter sharing and component-specific predictors
- [ ] Initialization and label-ordering conventions
- [ ] R parity against `gamlss.mx`
- [ ] Posterior component probabilities and mixture diagnostics

## Phase 12 — additive-model ecosystem

- [ ] Tensor-product and multidimensional smooths
- [ ] Free-knot and expanded penalty families
- [ ] Spatial smooths and GMRF-style penalties
- [ ] Cross-validation and rolling-origin model validation
- [ ] Expanded diagnostic and plotting compatibility with the GAMLSS ecosystem
