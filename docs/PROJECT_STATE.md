# Project state and handoff

Last updated: 2026-07-30, America/Sao_Paulo.

This file is the persistent project memory for decisions that should survive a
conversation or development-session boundary. Detailed feature tracking remains
in [`ROADMAP.md`](ROADMAP.md), while the smooth-system design is specified in
[`SMOOTH_ARCHITECTURE.md`](SMOOTH_ARCHITECTURE.md).

## Project direction

TorchGAMLSS aims to combine:

1. the distributional scope and compatibility conventions of GAMLSS;
2. the basis--penalty architecture and whole-model smoothness selection of
   `mgcv`;
3. PyTorch autograd, neural predictors, streaming, and measured CPU/CUDA
   execution.

The goal is not a complete clone of `mgcv`. The differentiating target is a
general distributional additive model that can eventually use structured
large-data smooth fitting.

## Current pull requests

All five pull requests are intentionally draft and must not be merged without
an explicit decision.

| PR | Scope | Branch | Validation |
|---|---|---|---|
| [#9](https://github.com/martinoni/torchgamlss/pull/9) | Inflated and adjusted distributions | `agent/inflated-adjusted-distributions` | 11/11 CI checks passed |
| [#10](https://github.com/martinoni/torchgamlss/pull/10) | Finite mixtures | `agent/finite-mixtures` | 11/11 CI checks passed |
| [#11](https://github.com/martinoni/torchgamlss/pull/11) | Generic smooth architecture | `agent/generic-smooth-architecture` | 11/11 CI checks passed |
| [#12](https://github.com/martinoni/torchgamlss/pull/12) | Generic penalized solver | `agent/generic-penalized-solver` | 11/11 CI checks passed |
| [#13](https://github.com/martinoni/torchgamlss/pull/13) | Tensor smooths and automatic LAML selection | `agent/tensor-product-smooths` | 624 local tests passed; 11/11 CI checks passed |

PRs #9, #10, and #11 are based on `main`; PR #12 is stacked on #11, and PR
#13 currently contains the LAML/tensor slice stacked on #12. The local
random-effect branch is also stacked on #12. Their roadmap edits may need a
small conflict resolution when merged. Do not combine implementation scopes
merely to avoid that documentation conflict.

## Accepted `mgcv` and `bam` decisions

- `SmoothTerm` remains the central abstraction; it is generalized rather than
  replaced.
- The canonical penalty representation is in coefficient space:
  `sum_j lambda_j S_j`.
- The existing `PSpline.penalty_matrix()` remains a square-root difference
  penalty for exact `gamlss::pb()` compatibility.
- A smooth term must expose training design, prediction design, one or more
  penalty matrices, smoothing parameters, and coefficient constraints.
- Linear constraints are imposed by null-space reparameterization, not by an
  arbitrarily large numerical penalty.
- Local ML/EDF/GAIC/GCV selection remains available for GAMLSS compatibility.
- Whole-model Laplace approximate marginal likelihood is named `LAML` in the
  generic implementation. `REML` is an alias only where its fixed-effect and
  scale interpretation is well defined.
- Autograd supplies likelihood derivatives but does not justify blindly
  differentiating through an arbitrary number of RS/CG iterations. The outer
  smoothing algorithm must have explicit convergence and stability checks.
- Tensor products use one penalty per marginal direction and explicit
  identifiability constraints. Tensor interactions must separate lower-order
  effects.
- Random intercepts and slopes are represented as ridge-penalized terms. Sparse
  execution remains a separate backend concern.
- The future `bam`-style route is deterministic and structural: marginal basis
  discretization, observation index vectors, aggregation of changing working
  weights/responses, and structured crossproducts.
- Mini-batch Adam is not treated as an equivalent to `bam`.
- CUDA is supported where useful, but performance claims require benchmarks
  against dense and structured CPU execution.
- Smoothing-parameter uncertainty, basis-dimension checks, concurvity, rank,
  and conditioning are planned features, not current capabilities.

## Delivery state

### Phase 12A — complete in PR #11

- generic `design()` and `predict_design()` contract;
- coefficient-space `penalty_matrices()` compatibility view;
- vector-shaped `smoothing_parameters` contract;
- explicit `constraints()` contract;
- unchanged P-spline numerical path;
- 555 local Python tests passed;
- GAMLSS, truncation, survival, and censoring R reference checks passed;
- 11/11 GitHub Actions jobs passed.

### Phase 12B — complete on `agent/generic-penalized-solver`

- public dense `solve_penalized_least_squares()` API;
- multiple fixed positive-semidefinite `S_j` matrices and non-negative lambdas;
- shape, symmetry, finiteness, dtype, device, numerical PSD, and rank
  validation;
- `S_lambda = sum_j lambda_j S_j`;
- SVD null-space reparameterization for `C beta = 0`, including redundant
  constraints;
- coefficient estimates, fitted values, total EDF, combined penalty,
  component ranks, constraint rank/null space, and condition diagnostics;
- numerically thresholded PSD roots for rank-deficient penalties at extreme
  lambda;
- unchanged classical square-root solver for scalar `pb()` parity;
- CPU float64 and local CUDA coverage;
- 570 Python tests, all R reference checks, package build, Twine validation,
  and clean wheel smoke test passed.

### Phase 12C — complete locally on `agent/laml-prototype`

The first whole-model LAML experiment is implemented in a sibling worktree
based on `agent/generic-penalized-solver`; commit, push, and draft-PR creation
are pending the permitted publication window.

- Normal location-scale fitting with `sigma = sigma_floor + exp(eta_sigma)`;
- joint free or fixed `rho_j = log(lambda_j)` values across `mu` and `sigma`;
- safeguarded inner Newton fitting with the joint observed autograd Hessian;
- generalized log determinants and explicit null-space constraints;
- bounded outer BFGS with central finite differences of the converged profile;
- objective, gradient, Hessian conditioning, boundary, EDF, and penalty-DF
  diagnostics;
- exact fixture agreement with an overlapping `mgcv` LAML model and
  fixed-lambda agreement with the current GAMLSS RS path;
- 578 Python tests, all R checks, CUDA execution, package build, strict Twine
  validation, and installed-wheel smoke test passed in that worktree.

The validated LAML module and fixtures are now also present in
`agent/tensor-product-smooths`, where the high-level model adapter and tensor
selection were implemented. Preserve `agent/laml-prototype` as the
independently reviewable scalar foundation; the tensor branch should
eventually stack on it rather than duplicate it in review.

### Phase 12D — tensor and LAML integration complete locally

The tensor-product slice is implemented on
`agent/tensor-product-smooths`. Its fixed-lambda core is based on
`agent/generic-penalized-solver`; automatic selection uses the Phase 12C LAML
module.

- public row-wise Kronecker design and marginal-penalty embedding helpers;
- `TensorProductSmooth` with one penalty per marginal direction and an
  optional global sum-to-zero constraint;
- `TensorInteractionSmooth` with marginal centering transforms that remove
  lower-order main-effect directions;
- fixed or automatic `te()` and `ti()` formula syntax over two or more simple
  numeric columns, with `initial_lambda_` controlling LAML starts;
- exact absorption of the formula `te()` sum-to-zero constraint into its
  coefficient parametrization, retained for prediction;
- fixed-lambda RS/CG backfitting through the generic constrained solver,
  without changing the scalar `pb()` numerical path;
- formula L-BFGS and mini-batch fitting on CPU/CUDA;
- `fit_laml()`/`fit_laml_data()` whole-model assembly for Normal
  location-scale models, including automatic scalar/tensor penalties,
  structural null-space constraints, penalty labels/slices, and model-state
  updates;
- conditional and joint analytic fixed-lambda covariance for `te()` and
  `ti()`, including new grids, simultaneous Gaussian bands, multivariate
  tables, and exact zero covariance in constrained coefficient directions;
- reproducible RS/CG parametric bootstrap for fixed-lambda `te()` and `ti()`,
  plus whole-model LAML bootstrap for automatic tensor terms, with one stored
  lambda column per marginal penalty, penalty-level joint labels, and scalar
  `pb()` result compatibility;
- parameter-free copies of marginal basis state, without duplicated trainable
  coefficients;
- prediction/state round trips, EDF, penalty nullity, quadratic penalties,
  autograd, rescaling invariance, and CPU/CUDA generic-solver coverage;
- exact algebraic reference checks against
  `mgcv::tensor.prod.model.matrix()` and
  `mgcv::tensor.prod.penalties()`;
- direct `mgcv::gaulss(method="REML")` tensor reference checks for the LAML
  objective, both directional lambdas, EDF, coefficients, fitted location and
  scale, and outer Hessian;
- 624 Python tests passed without skips, including formula construction,
  constrained low-level RS and covariance, `te()`/`ti()` RS--CG agreement,
  conditional/joint inference, vector-lambda bootstrap, automatic
  `te()`/`ti()` LAML, L-BFGS, mini-batch, prediction, and local CUDA
  RS/inference/LAML;
- all GAMLSS, truncation, survival/censoring, and `mgcv` R checks passed;
- Ruff, dependency, bytecode, package build, strict Twine, and isolated
  installed-wheel smoke checks passed.

Automatic tensor-lambda selection and fixed-design LAML bootstrap refits are
now part of this slice. Smooth, joint-smooth, quantile, and centile bootstrap
APIs accept `algorithm="laml"` for additive Normal models and repeat joint
scalar/tensor selection in every successful replicate.

### Phase 12E — random-effect portion complete locally

The random-intercept and random-slope slice is implemented in the sibling
`agent/random-effect-terms` worktree, also based on
`agent/generic-penalized-solver`.

- ridge-penalized random intercepts and slopes;
- fixed lambda, local ML, and target-EDF modes;
- formula `random(group)` and `random(group, x)` construction;
- persisted categorical level encodings and zero contribution for unseen
  prediction groups;
- exact GAMLSS `random()` fixed/ML/DF references and `mgcv` design/identity
  penalty algebra checks;
- RS, CG, L-BFGS, mini-batch, autograd, state, and CUDA coverage;
- 600 Python tests and the complete R/package validation stack passed.

### Phase 12F — first non-Normal LAML vertical implemented

The family-driven LAML likelihood core and the first non-Normal vertical are
implemented on `agent/tensor-product-smooths`.

- `fit_gamlss_laml()` obtains parameter order, links, starts, and
  differentiable observation-wise likelihoods through the public `Family`
  contract;
- the safeguarded inner Newton solver, observed autograd Hessian, generalized
  determinants, bounded outer BFGS, constraints, and diagnostics are shared
  with the Normal path;
- `GAMLSSLAMLResult` exposes parameter-keyed coefficients, predictors, fitted
  parameters, and the common LAML diagnostics;
- whole-model `fit_laml()`/`fit_laml_data()` now accept standard Poisson
  log-mean models in addition to standard Normal location-scale models;
- fixed-design smooth bootstrap with `algorithm="laml"` reselects Poisson
  smoothing parameters without mutating the fitted model;
- a direct `mgcv::gam(..., family=poisson(), method="REML")` fixture checks
  the LAML objective, selected lambda, EDF, coefficients, link predictor,
  fitted mean, and outer Hessian;
- CPU and local CUDA coverage exercise the generic Poisson path;
- the specialized Normal API and result remain available for compatibility,
  and existing Normal/tensor LAML parity tests remain unchanged;
- 627 Python tests passed without skips; all R reference gates, Ruff,
  dependency and bytecode checks, package build, strict Twine validation, and
  isolated installed-wheel smoke checks passed.

Gamma is the next family candidate. Before claiming parity, its GAMLSS
coefficient-of-variation `sigma` must be mapped explicitly to the scale
parameterization of the chosen `mgcv` reference family.

### Later slices

1. LAML generalization beyond Poisson, beginning with Gamma;
2. cyclic, shrinkage, adaptive, thin-plate, spatial, and GMRF terms;
3. discretized marginal bases and structured crossproducts;
4. unconditional inference and smoothing-uncertainty-aware information
   criteria;
5. basis-dimension, concurvity, rank, and conditioning diagnostics.

## Open design decisions

- Final formula syntax for generic `s()` and future smooth classes; `te()`,
  `ti()`, and random-effect syntax now have implemented first slices.
- How smoothing parameters can be shared across terms or distribution
  parameters.
- Dense versus sparse storage thresholds for random effects and GMRF
  penalties.
- Sharing or tying selected smoothing parameters across terms.
- Naming of the structured large-data execution mode; avoid promising complete
  `bam` compatibility prematurely.

These choices must not block the first constrained P-spline LAML prototype,
whose internal representation is already available.

## Working rules

- Deliver verified vertical slices with tests and reference comparisons.
- Preserve GPL-3.0-only licensing while translating GPL-compatible R code.
- Keep unrelated phases in separate worktrees and pull requests.
- Keep new pull requests as drafts unless explicitly asked otherwise.
- Do not merge a pull request without explicit authorization.
- Commit and push only outside 09:00--18:00 in America/Sao_Paulo.
- Existing user changes in a worktree are never discarded or silently staged.

## Resume point

Commit and push the validated Poisson LAML slice to draft PR #13 during the
permitted publication window, then begin the explicit Gamma parameterization
mapping. Rebase onto `main` only after dependencies merge. Do not merge any
existing draft PR without explicit user authorization.
