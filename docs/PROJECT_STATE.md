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

All three pull requests are intentionally draft and must not be merged without
an explicit decision.

| PR | Scope | Branch | Validation |
|---|---|---|---|
| [#9](https://github.com/martinoni/torchgamlss/pull/9) | Inflated and adjusted distributions | `agent/inflated-adjusted-distributions` | 11/11 CI checks passed |
| [#10](https://github.com/martinoni/torchgamlss/pull/10) | Finite mixtures | `agent/finite-mixtures` | 11/11 CI checks passed |
| [#11](https://github.com/martinoni/torchgamlss/pull/11) | Generic smooth architecture | `agent/generic-smooth-architecture` | 11/11 CI checks passed |
| [#12](https://github.com/martinoni/torchgamlss/pull/12) | Generic penalized solver | `agent/generic-penalized-solver` | Local validation passed; CI tracked in PR |

The branches are independent and based on `main`. Their roadmap edits may need
a small conflict resolution when merged. Do not combine their implementation
scopes merely to avoid that documentation conflict.

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

### Phase 12C — next implementation: first LAML experiment

Use current one-dimensional P-splines in an overlapping Gaussian
location-scale model before adding tensor products.

Acceptance criteria:

- jointly optimize free `rho_j = log(lambda_j)`;
- converge the inner penalized coefficient fit for every accepted outer step;
- use the joint observed information across distribution parameters;
- use generalized log determinants for rank-deficient penalties;
- expose objective, gradient, Hessian/conditioning, boundary, and convergence
  diagnostics;
- compare fitted values, lambdas, criterion, and effective degrees of freedom
  with an overlapping `mgcv` model;
- confirm that fixed-lambda results remain compatible with the current GAMLSS
  path.

### Later slices

1. tensor-product full smooths and tensor interactions;
2. random intercepts and slopes;
3. cyclic, shrinkage, adaptive, thin-plate, spatial, and GMRF terms;
4. discretized marginal bases and structured crossproducts;
5. unconditional inference and smoothing-uncertainty-aware information
   criteria;
6. basis-dimension, concurvity, rank, and conditioning diagnostics.

## Open design decisions

- Final formula syntax for generic `s()`, `te()`, `ti()`, and random effects.
- Whether the first LAML optimizer uses safeguarded Newton, BFGS, or an
  extended Fellner--Schall prototype.
- How smoothing parameters can be shared across terms or distribution
  parameters.
- Dense versus sparse storage thresholds for random effects and GMRF
  penalties.
- Fit-result shape for estimated lambdas, outer-iteration diagnostics, and
  penalty-level effective degrees of freedom.
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

After PR #12 is accepted or while it remains under review, start Phase 12C in a
separate branch based on
`agent/generic-penalized-solver`. Rebase that branch onto `main` after its
dependencies merge. Do not add LAML to the solver PR; keep the coefficient
system and smoothing-selection algorithm reviewable as separate vertical
slices.
