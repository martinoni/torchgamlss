# Roadmap

TorchGAMLSS will grow in verified vertical slices. Each slice must include
automated comparisons with the reference R implementation before it is treated
as supported.

## Phase 0 — foundation

- [x] Package skeleton and GPL-3.0-only licensing
- [x] Link protocol and identity, log, and logit links
- [x] Normal family with independent `mu` and `sigma` predictors
- [x] Differentiable full-batch negative log-likelihood
- [x] R reference-data generator
- [x] Numerical tolerances and parity-test conventions

## Phase 1 — parametric core

- [x] Weighted observations and offsets
- [ ] Formula/design-matrix interface
- [ ] Starting-value protocol
- [ ] Normal, Gamma, Poisson, beta, and negative-binomial families
- [x] Initial joint L-BFGS optimizer and convergence diagnostics
- [ ] Covariance matrix and standard errors

## Phase 2 — classical GAMLSS fitting

- [x] Linear unpenalized RS core
- [ ] RS backfitting with smooth and penalized terms
- [ ] CG algorithm
- [ ] Penalized weighted least squares
- [ ] P-splines with fixed smoothing parameters
- [ ] Automatic smoothing-parameter selection
- [ ] Effective degrees of freedom

## Phase 3 — compatibility and diagnostics

- [ ] Predictions on link, response, and term scales
- [ ] Randomized quantile residuals
- [ ] Deviance, AIC, GAIC, and model comparison
- [ ] Four-parameter families including BCT and BCPE
- [ ] R-to-Python API and numerical compatibility guide

## Phase 4 — Torch-native extensions

- [ ] Mini-batch optimization for large data
- [ ] Neural predictors for selected distribution parameters
- [ ] Shared representations across distribution parameters
- [ ] CPU/GPU benchmarks and reproducibility guidance
