# Generic smooth architecture

TorchGAMLSS should combine the distributional scope of GAMLSS with the
basis--penalty architecture and whole-model smoothing selection developed for
general smooth models. It should not attempt to clone every `mgcv` feature.

The first implementation rule is backward compatibility: the existing
`PSpline`, local ML/EDF/GAIC/GCV selection, RS/CG fits, R parity fixtures, and
fixed-lambda inference remain supported throughout the migration.

## Current limitation

`SmoothTerm` already separates a smooth from the model and fitter. Its original
contract nevertheless encodes the assumptions of `gamlss::pb()`:

- one covariate tensor;
- one square-root difference penalty `D`;
- one smoothing parameter;
- no explicit coefficient constraints.

The current objective is

```text
lambda * ||D beta||^2
```

which is equivalent to

```text
beta.T @ (lambda * S) @ beta,  where S = D.T @ D.
```

The second form is the canonical representation for the generic architecture.
It naturally extends to

```text
beta.T @ (sum_j lambda_j * S_j) @ beta.
```

Tensor products, adaptive smooths, shrinkage penalties, structured random
effects, and global Laplace approximate marginal likelihood (LAML) selection
all require that extension.

## Term contract

The public term-level concepts are:

```python
design = term.design(covariates)
penalties = term.penalty_matrices()
constraints = term.constraints(covariates)
prediction_design = term.predict_design(new_covariates)
lambdas = term.smoothing_parameters
```

Each `S_j` returned by `penalty_matrices()` is an unscaled, symmetric
positive-semidefinite matrix in coefficient space. `constraints()` returns a
matrix `C` defining `C @ beta = 0`. Constraints will be imposed by a null-space
reparameterization rather than by adding a numerically large penalty.

During the compatibility transition:

- `basis()` remains the implementation hook for `PSpline`;
- `penalty_matrix()` continues to return its difference-penalty root `D`;
- scalar `smoothing_parameter` remains available;
- the generic methods delegate to those legacy methods.

## Architectural layers

The implementation is separated into five layers.

1. **Term construction** creates design, penalty, constraint, and prediction
   mappings without fitting a response.
2. **Penalized coefficient solving** consumes arbitrary design blocks,
   coefficient-space penalties, and constraints.
3. **Smoothness selection** treats `rho_j = log(lambda_j)` as a whole-model
   vector rather than updating one working smoother in isolation.
4. **Formula construction** maps `pb()`, future `s()`, `te()`, `ti()`, and
   random-effect syntax to term objects.
5. **Execution backends** choose dense, chunked, discretized, CPU, or CUDA
   operations without changing the statistical model.

This separation prevents tensor products, LAML, and large-data execution from
becoming special cases inside the RS or CG loops.

## Staged delivery

### 12A — representation without behavior changes

- expose the generic contract on `SmoothTerm`;
- verify `PSpline.penalty_matrices()[0] == D.T @ D`;
- retain all existing R parity and public scalar-lambda results.

### 12B — generic penalized solver

- accept multiple fixed penalties per term;
- validate symmetry, positive semidefiniteness, rank, dtype, and device;
- impose linear constraints through null-space reparameterization;
- report term and penalty-level effective degrees of freedom;
- retain the current square-root augmented least-squares path for `pb()`
  parity.

### 12C — whole-model LAML prototype

The first prototype will use existing one-dimensional P-splines and a
comparable Gaussian location-scale model. For a trial vector `rho`, it will:

1. converge the penalized coefficient fit;
2. form the joint observed information across all distribution parameters;
3. evaluate the Laplace approximate marginal likelihood using generalized
   determinants for rank-deficient penalties;
4. update all free `rho` values jointly;
5. expose gradients, Hessian conditioning, boundary status, and convergence.

Autograd can supply likelihood derivatives, but it is not a substitute for a
stable outer algorithm. Differentiating through an arbitrary number of RS/CG
iterations is not the primary implementation route. The prototype must compare
against `mgcv` for an overlapping model and against the current GAMLSS path for
fixed lambdas.

`LAML` is the generic name in the implementation. A `REML` alias should only be
advertised where its fixed-effect and scale interpretation is well defined.

### 12D — richer terms

- tensor-product smooth with one penalty per marginal direction;
- tensor interaction with explicit main-effect separation;
- random intercept and slope terms with full-rank ridge penalties;
- sum-to-zero constraints and shrinkage/double penalties.

### 12E — structured large-data backend

The large-data route is not synonymous with stochastic Adam. It will:

- evaluate marginal bases at unique or discretized values;
- retain observation-to-basis index vectors;
- aggregate changing working weights and responses by index;
- calculate structured crossproducts without materializing dense full-data
  design matrices;
- benchmark dense CPU, dense CUDA, and discretized CPU/CUDA paths.

GPU execution is selected by measured workload. For modest basis dimensions,
dense factorizations can remain latency- or memory-bandwidth-bound and may be
faster on CPU.

### 12F — inference and diagnostics

- covariance of estimated log smoothing parameters from the LAML Hessian;
- unconditional coefficient and smooth covariance;
- smoothing-uncertainty-aware information criteria;
- basis-dimension checks;
- concurvity, rank, and conditioning diagnostics.

## Validation gates

A vertical slice is supported only when it includes:

- deterministic CPU float64 tests;
- CUDA coverage where the operation has a CUDA backend;
- gradient checks for differentiable criteria;
- invariance to covariate rescaling for tensor products;
- fitted-value and criterion comparisons with `mgcv` for overlapping models;
- unchanged `gamlss` parity for `pb()` compatibility;
- numerical tests near rank deficiency and extreme smoothing parameters;
- benchmark evidence before claiming a large-data or GPU advantage.

## Primary references

- Wood, Pya, and Säfken (2016), *Smoothing parameter and model selection for
  general smooth models*, <https://doi.org/10.1080/01621459.2016.1180986>.
- `mgcv::gam()` documentation:
  <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.html>.
- `mgcv::bam()` documentation:
  <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/bam.html>.
- `mgcv` smooth constructor documentation:
  <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.construct.html>.
- `mgcv` tensor-product documentation:
  <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/te.html>.
