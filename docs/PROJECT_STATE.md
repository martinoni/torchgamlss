# Project state and handoff

Last updated: 2026-08-01, America/Sao_Paulo.

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
| [#13](https://github.com/martinoni/torchgamlss/pull/13) | Tensor smooths and family-driven LAML | `agent/tensor-product-smooths` | PE validation passes 657 local tests, CUDA, all R gates, and the installed-wheel smoke test; remote CI is pending |

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

### Phase 12F — family-driven non-Normal LAML implemented

The family-driven LAML likelihood core and the Poisson/Gamma verticals are
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
  log-mean and Gamma mean/CV models in addition to standard Normal
  location-scale models;
- fixed-design smooth bootstrap with `algorithm="laml"` reselects Poisson and
  Gamma smoothing parameters without mutating the fitted model;
- direct `mgcv::gam(..., family=poisson(), method="REML")` and
  `mgcv::gammals()` fixtures check objectives, selected lambdas, EDF,
  coefficients, predictors, fitted parameters, and outer Hessians;
- the Gamma reference explicitly maps `mgcv` variance scale
  `phi = sigma^2` to the GAMLSS coefficient of variation;
- CPU and local CUDA coverage exercise the one- and two-parameter generic
  paths;
- the specialized Normal API and result remain available for compatibility,
  and existing Normal/tensor LAML parity tests remain unchanged;
- 630 Python tests passed without skips; all R reference gates, Ruff,
  dependency and bytecode checks, package build, strict Twine validation, and
  isolated installed-wheel smoke checks passed.

The next family requires an explicit validation strategy because `mgcv` does
not provide directly overlapping additive predictors for every GAMLSS
location-scale-shape parameterization.

### Phase 12G — implicit outer LAML gradient implemented

The outer gradient now differentiates the converged penalized score equation
with the implicit function theorem rather than differencing profile
objectives.

- coefficient sensitivities use
  `d beta / d rho_j = -H_p^-1 lambda_j S_j beta`;
- Torch autograd supplies the third-order likelihood contraction needed by
  the penalized-information determinant;
- the generalized penalty-determinant derivative uses the inverse restricted
  to the penalized subspace;
- the outer Hessian differences the implicit gradient, removing the noisier
  second difference of the profile objective;
- `outer_derivative_method="finite_difference"` preserves the original
  implementation as an audit fallback;
- results record the derivative method and number of unique profile
  evaluations;
- on the two-lambda Gamma reference, the default route preserves objective,
  gradient, lambdas, and Hessian while reducing unique profile evaluations
  from 48 to 12;
- Normal, Poisson, NBI, Gamma, Beta, tensor, bootstrap, and local CUDA LAML tests pass
  through the implicit default.

### Phase 12H — fully analytic outer LAML Hessian implemented

The implicit default now differentiates the converged coefficient solution a
second time instead of differencing gradients at displaced smoothing
parameters.

- second coefficient sensitivities solve the differentiated penalized score
  equation with the already available penalized information matrix;
- Torch autograd supplies the exact LAML partial Hessian, including the
  fourth-order likelihood derivatives induced by the information determinant;
- the chain rule combines exact partials with the first and second implicit
  coefficient sensitivities without differentiating through Newton iterations;
- the finite-difference derivative route remains available as an explicit
  audit fallback;
- the one-iteration Gamma audit preserves gradients, Hessian, objective, and
  the accepted lambda update while reducing profile evaluations from 18 to 2;
- at full Gamma convergence the reduction is from 48 to 8 profile evaluations;
- direct `mgcv` Hessian parity and CPU/CUDA LAML coverage pass through the
  analytic default.

The local derivative calculation now reaches likelihood derivatives through
fourth order. It removes repeated inner fits and finite-difference noise, but
can still be compute- or memory-intensive for large dense coefficient spaces.

### Phase 12I — Beta LAML vertical implemented locally

The first post-Hessian family extension uses `gamlss.dist::BE` and a
conditional `mgcv::betar` reference.

- whole-model `fit_laml()`/`fit_laml_data()` accepts standard Beta
  logit-mean/logit-dispersion models;
- LAML smooth, joint-smooth, response-quantile, and centile bootstrap refits
  accept Beta and reselect automatic lambdas in every successful sample;
- the low-level family-driven API accepts an `n x 0` design plus a link-scale
  offset to condition on a fixed family parameter;
- `mgcv::betar(theta=12, link="logit")` maps through
  `phi = 1/sigma^2 - 1`, fixing Torch `sigma` while both implementations
  optimize the same mean smooth;
- negative LAML, likelihood, lambda, coefficients, predictor, fitted mean, and
  analytic outer Hessian match the direct `mgcv` fixture;
- the roughly `0.004` EDF difference is documented as an extended-family EDF
  convention difference rather than silently tightened away;
- formula fitting and LAML bootstrap pass on CPU, and the conditional fixture
  passes on local CUDA.
- all 635 Python tests pass without skips; all five R gates, Ruff, dependency
  and bytecode checks, package build, strict Twine validation, and the isolated
  installed-wheel smoke test pass.

### Phase 12J — NBI LAML vertical implemented locally

Negative-binomial type I now uses the same family-driven nested LAML core.

- whole-model `fit_laml()`/`fit_laml_data()` accepts standard NBI
  log-mean/log-dispersion models;
- LAML bootstrap refits accept NBI and reselect automatic scalar or tensor
  lambdas in every successful sample;
- `mgcv::nb(theta=4, link="log")` maps exactly through `sigma = 1/theta`;
- a zero-column `sigma` design plus a fixed log-scale offset makes the direct
  comparison conditional on the same dispersion;
- negative LAML, likelihood, lambda, coefficients, predictor, fitted mean,
  and analytic outer Hessian agree with the committed `mgcv` fixture;
- the roughly `0.0045` EDF difference is documented as an extended-family
  convention difference;
- formula fitting and LAML bootstrap pass on CPU, and the conditional fixture
  passes on local CUDA 12.8.
- all 638 Python tests pass without skips; all five R gates, Ruff, build,
  strict Twine validation, and the installed-wheel smoke test pass.

### Phase 12K — Student-t LAML vertical implemented locally

The three-parameter `gamlss.dist::TF` family now uses whole-model LAML with a
direct conditional `mgcv::scat` reference.

- whole-model `fit_laml()`/`fit_laml_data()` accepts standard Student-t
  identity-location/log-scale/log-degrees-of-freedom models;
- LAML bootstrap refits estimate all three TF predictors and reselect
  automatic scalar or tensor lambdas;
- `mgcv::scat(theta=c(5, 0.8), link="identity")` uses the same
  `(Y - mu)/sigma ~ t_nu` parameterization;
- zero-column `sigma` and `nu` designs plus fixed log-link offsets isolate the
  identical conditional location-smooth problem;
- negative LAML, likelihood, lambda, coefficients, predictor, fitted location,
  and analytic outer Hessian agree numerically;
- the approximately `0.056` EDF difference is reported as the scaled-t
  extended-family convention instead of being presented as equality;
- formula fitting and ten-replicate LAML bootstrap pass on CPU, while the
  conditional fixture passes on local CUDA 12.8;
- all 641 Python tests pass without skips; all five R gates, Ruff, dependency
  and bytecode checks, package build, strict Twine validation, and the
  installed-wheel smoke test pass.

### Phase 12L — BCCG LAML vertical implemented locally

The first family without a directly overlapping `mgcv` marginal-likelihood
implementation now follows the validation protocol recorded after Phase 12K.

- whole-model `fit_laml()`/`fit_laml_data()` accepts standard BCCG
  identity-location/log-scale/identity-shape models;
- the inner Newton line search now rejects and shortens domain-invalid trials,
  including temporary `mu <= 0` proposals and non-finite BCCG likelihoods;
- with all three `pb()` lambdas fixed at 10, the joint LAML inner fit,
  initialized from the matching validated RS state, matches the committed
  `gamlss::gamlss()` reference negative log likelihood and all 1,830 fitted
  `mu`, `sigma`, and `nu` values;
- this is explicitly presented as fixed-lambda penalized-fit parity, not as an
  R implementation of the LAML criterion;
- a separate two-lambda audit matches the implicit outer gradient and analytic
  Hessian to finite differences of independently converged profiles;
- formula selection estimates all three predictors, ten-replicate LAML
  bootstrap reselects lambda, and the complete path runs on local CUDA 12.8;
- the isolated installed-wheel smoke test includes and passes a BCCG LAML fit;
- all 645 Python tests pass without skips; all five R gates, Ruff, dependency
  and bytecode checks, package build, strict Twine validation, and the
  installed-wheel smoke test pass;
- GitHub Actions passed all 11/11 jobs on implementation head `fe43741`,
  including Windows/Python 3.10 after the fixed-lambda parity warm start.

### Phase 12M — BCT LAML vertical implemented locally

The four-parameter `gamlss.dist::BCT` family now extends the same
fixed-lambda-reference plus independent-derivative-audit protocol.

- whole-model `fit_laml()`/`fit_laml_data()` accepts standard BCT
  identity-location/log-scale/identity-skewness/log-tail-shape models;
- BCT LAML bootstrap refits reselect automatic scalar or tensor lambdas;
- a weighted fixed-lambda `mu` P-spline fit initialized from its compatible RS
  state matches the committed `gamlss::gamlss()` negative log likelihood and
  all 160 fitted `mu`, `sigma`, `nu`, and `tau` values within documented
  tolerances;
- the slightly wider `tau` tolerance is deliberate: R's BCT RS working score
  uses a forward difference of `0.01`, while Torch LAML differentiates the
  actual Student-t CDF likelihood;
- a separate weighted audit matches the implicit outer gradient and analytic
  Hessian to the finite-difference profile implementation;
- automatic formula selection estimates all four predictors, ten-replicate
  bootstrap reselects lambda, and the complete path runs on local CUDA 12.8;
- a compatible RS warm start is the documented high-level workflow because
  `tau` can be weakly identified and the joint observed information can be
  indefinite near a generic family start;
- the R generator and installed-wheel smoke test now include BCT LAML gates;
- all 649 Python tests pass without skips; all five R gates, Ruff, dependency
  and bytecode checks, package build, strict Twine validation, and the
  installed-wheel smoke test pass locally;
- GitHub Actions passed all 11/11 jobs on implementation head `31075e1`
  ([run 30695264274](https://github.com/martinoni/torchgamlss/actions/runs/30695264274)),
  including Python 3.10--3.13 on Linux and Windows.

### Phase 12N — BCPE LAML vertical implemented locally

The four-parameter `gamlss.dist::BCPE` family now follows the same staged
validation protocol as BCT.

- whole-model `fit_laml()`/`fit_laml_data()` accepts standard BCPE
  identity-location/log-scale/identity-skewness/log-kurtosis models;
- BCPE LAML bootstrap refits reselect automatic scalar or tensor lambdas;
- a weighted fixed-lambda `mu` P-spline fit initialized from its compatible RS
  state matches the committed `gamlss::gamlss()` negative log likelihood and
  all 160 fitted `mu`, `sigma`, `nu`, and `tau` values within documented
  tolerances;
- R's BCPE RS working score uses a forward difference of `0.001` for part of
  the `tau` derivative, while Torch LAML differentiates the actual
  regularized-gamma CDF likelihood;
- a separate weighted audit matches the implicit outer gradient and analytic
  Hessian to the finite-difference profile implementation;
- automatic formula selection estimates all four predictors, ten-replicate
  bootstrap reselects lambda, and the complete path runs on local CUDA 12.8;
- a compatible RS warm start is the documented high-level workflow;
- the R generator and installed-wheel smoke test include BCPE LAML gates;
- all 653 Python tests pass without skips; all five R gates, Ruff, dependency
  and bytecode checks, package build, strict Twine validation, and the
  installed-wheel smoke test pass locally;
- the first remote run exposed a 1.09% platform difference in the BCPE
  finite-difference Hessian audit on Windows/Python 3.11; its focused relative
  tolerance is now 1.2%, without changing estimator code;
- GitHub Actions passed all 11/11 jobs on stabilization head `0f710ad`
  ([run 30700920488](https://github.com/martinoni/torchgamlss/actions/runs/30700920488)),
  including Python 3.10--3.13 on Linux and Windows.

### Phase 12O — PE LAML vertical implemented locally

The three-parameter `gamlss.dist::PE` family now uses the family-driven nested
LAML core with its standard identity/log/log links.

- whole-model `fit_laml()`/`fit_laml_data()` accepts PE
  location/scale/tail-shape models on the whole real line;
- PE LAML bootstrap refits reselect automatic scalar or tensor lambdas;
- a weighted fixed-lambda `mu` P-spline fit initialized from its compatible RS
  state matches the committed `gamlss::gamlss()` negative log likelihood and
  all 160 fitted `mu`, `sigma`, and `nu` values within `3e-6` relative and
  absolute tolerance;
- because no directly overlapping `mgcv` location-scale-shape family exists,
  the R fit validates the joint penalized estimator while a separate weighted
  audit matches the implicit outer gradient and analytic Hessian to the
  finite-difference profile implementation;
- automatic formula selection estimates all three predictors, ten-replicate
  bootstrap reselects lambda, and the complete path runs on local CUDA 12.8;
- a compatible RS warm start is documented for strict R parity and data where
  tail shape is weakly identified;
- the R generator and installed-wheel smoke test include PE LAML gates;
- all 657 Python tests pass without skips; all five R gates, Ruff, dependency
  and bytecode checks, package build, strict Twine validation, and the
  installed-wheel smoke test pass locally;
- remote GitHub Actions validation remains pending for the PE implementation
  commit.

### Later slices

1. generalized Gamma (`GG`) log/log/identity LAML validation using
   fixed-lambda R fits plus derivative audits, because no directly overlapping
   `mgcv` location-scale-shape family exists;
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
- On weekdays, commit and push only outside 09:00--18:00 in
  America/Sao_Paulo. Saturdays and Sundays have no time restriction.
- Existing user changes in a worktree are never discarded or silently staged.

## Resume point

The next family extension should reuse the same fixed-lambda R parity plus
derivative-audit protocol for generalized Gamma (`GG`) with its standard
log/log/identity links. Rebase onto `main` only after stacked dependencies
merge. Do not merge any existing draft PR without explicit user authorization.
