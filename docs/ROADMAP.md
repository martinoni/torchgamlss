# Roadmap

TorchGAMLSS will grow in verified vertical slices. Each slice must include
automated comparisons with the reference R implementation before it is treated
as supported.

The current implementation state, accepted architectural decisions, and
next-session handoff are recorded in [`PROJECT_STATE.md`](PROJECT_STATE.md).

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
- [x] Complete the first TestPyPI publication and clean-install smoke test
- [x] Tag, publish, and clean-install validate `v0.1.0a2`

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
- [x] Observation-specific truncation bounds compatible with `gamlss.tr`
- [x] Extend truncation parity across the translated family catalog

## Phase 9 — survival and censoring

- [x] Weibull (`WEI`) and log-normal (`LOGNO`) survival families
- [x] Inverse-Gaussian and generalized gamma survival families
- [x] Right-, left-, and interval-censored response representation
- [x] Censored likelihood composition compatible with `gamlss.cens`
- [x] Survival, hazard, and cumulative-hazard prediction
- [x] Reproducible censored-survival parity example

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

- [x] Generic `design` + penalties + constraints + prediction-design contract
  layered over the GAMLSS-compatible `PSpline`
- [x] Coefficient-space penalty representation with multiple fixed lambdas
- [x] Null-space reparameterization for sum-to-zero and point constraints
- [x] First whole-model LAML prototype for Normal location-scale models and
  current one-dimensional P-splines
- [x] Generic family-driven LAML likelihood core and non-Normal vertical
  slices for Poisson log-mean, NBI mean/dispersion, Gamma mean/CV, and Beta
  mean/dispersion models, plus Student-t location/scale/shape, including direct
  `mgcv` REML/LAML parity and bootstrap refits
- [x] BCCG identity/log/identity LAML with fixed-lambda `gamlss` parity,
  implicit-versus-finite-difference derivative audits, bootstrap, and CUDA
- [x] BCT identity/log/identity/log LAML with fixed-lambda `gamlss` parity,
  RS warm starts, derivative audits, bootstrap, and CUDA
- [x] BCPE identity/log/identity/log LAML with fixed-lambda `gamlss` parity,
  RS warm starts, derivative audits, bootstrap, and CUDA
- [x] PE identity/log/log LAML with fixed-lambda `gamlss` parity, RS warm
  starts, derivative audits, bootstrap, and CUDA
- [x] Uncensored GG log/log/identity LAML with fixed-lambda `gamlss` parity,
  RS warm starts, derivative audits away from the piecewise log-normal
  transition, bootstrap, and CUDA
- [x] Implicit-function outer LAML gradients with a finite-difference audit
  fallback and reduced profile-evaluation diagnostics
- [x] Fully analytic outer LAML Hessian from second-order implicit coefficient
  sensitivities and exact autograd partials, with no displaced profile fits
- [ ] Extend whole-model LAML to the remaining multi-parameter families,
  beginning with log-normal (`LOGNO`), followed by Weibull and
  inverse-Gaussian, while retaining an explicit reference strategy for
  families without a directly overlapping `mgcv` location-scale family
- [x] Fixed-lambda tensor-product full smooths and ANOVA-style tensor
  interactions through the generic dense solver
- [x] Fixed-lambda `te()`/`ti()` formula construction, prediction, L-BFGS,
  mini-batch, and CUDA execution
- [x] RS/CG integration for fixed-lambda multiply penalized tensor terms
- [x] Conditional and joint analytic fixed-lambda inference for tensor terms
- [x] RS/CG smooth bootstrap with penalty-level vector-lambda storage for
  tensor terms
- [x] Whole-model LAML selection for tensor smoothing parameters, including
  formula integration and direct `mgcv::te()` REML reference checks
- [x] Fixed-design LAML bootstrap refits with joint scalar/tensor lambda
  reselection for smooth and response-quantile inference
- [x] Random intercepts and slopes represented as ridge-penalized terms
- [ ] Thin-plate, cyclic, shrinkage, adaptive, spatial, and GMRF smooths
- [ ] Discretized marginal bases and structured crossproducts for large data
- [ ] Unconditional covariance including log-smoothing-parameter uncertainty
- [ ] Basis-dimension, concurvity, rank, and conditioning diagnostics
- [ ] Cross-validation and rolling-origin model validation
- [ ] Expanded diagnostic and plotting compatibility with the GAMLSS ecosystem

See [`SMOOTH_ARCHITECTURE.md`](SMOOTH_ARCHITECTURE.md) for the staged design,
compatibility rules, and validation gates.
