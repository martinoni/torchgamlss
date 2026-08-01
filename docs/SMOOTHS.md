# Smooth terms

TorchGAMLSS currently provides `PSpline`, a P-spline compatible with the core
construction of `gamlss::pb()`. It uses an equally spaced cubic B-spline basis
by default and penalizes second differences of adjacent spline coefficients.

For basis matrix `B`, coefficient vector `gamma`, difference matrix `D`,
working response `z`, and working weights `W`, each smoother update solves

```text
minimize  (z - B gamma)' W (z - B gamma) + lambda ||D gamma||^2.
```

The implementation solves the equivalent augmented least-squares problem and
reports effective degrees of freedom as

```text
trace((B' W B + lambda D' D)^-1 B' W B).
```

When `lambda` is not supplied, the default ML update treats the coefficient
differences as random effects. After each penalized fit it computes

```text
sigma^2 = sum(w_i (z_i - fitted_i)^2) / (N_positive_weight - EDF)
tau^2   = ||D gamma||^2 / (EDF - penalty_order)
lambda  = sigma^2 / max(tau^2, 1e-7).
```

`lambda` is clipped to `[1e-7, 1e7]` and iterated from the default starting
value 10. These equations and safeguards follow `gamlss.pb()`.

Alternatively, `degrees_of_freedom` selects `lambda` by inverting the
hat-matrix trace on the same log-lambda interval used by R. Its semantics match
the `df` argument of `pb()`: the requested value is the nonlinear degrees of
freedom, and two unpenalized dimensions are added to obtain the target total
EDF.

Local GAIC and GCV selection are also available. For penalty multiplier `k`,
weighted residual sum of squares `RSS`, sample size `n`, and effective degrees
of freedom `EDF`, they minimize

```text
GAIC(lambda) = RSS(lambda) + k EDF(lambda)
GCV(lambda)  = n RSS(lambda) / (n - k EDF(lambda))^2.
```

The default `k=2`, bounds `[1e-7, 1e7]`, and local working-response criteria
match `pb.control()`. TorchGAMLSS uses bounded Brent minimization on
`log(lambda)` instead of R's `nlminb()` on `lambda`; both target the same
criterion minimum.

## Example

```python
import torch
from torchgamlss import GAMLSS, Normal, PSpline, RSControl

x = torch.linspace(-1, 1, 100, dtype=torch.float64)
y = 0.7 + 0.8 * x + torch.sin(torch.pi * x)

term = PSpline.from_data(x)
model = GAMLSS(
    Normal(),
    {"mu": 2, "sigma": 1},
    smooth_terms={"mu": {"x": term}},
)
design = {
    "mu": torch.column_stack((torch.ones_like(x), x)),
    "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
}
smooth_covariates = {"mu": {"x": x}}

result = model.fit_rs(
    y,
    design,
    smooth_covariates=smooth_covariates,
    control=RSControl(),
)
mu = model.predict(
    design,
    smooth_covariates=smooth_covariates,
    type="response",
)["mu"]
edf = result.smooth_effective_degrees_of_freedom["mu"]["x"]
estimated_lambda = result.smoothing_parameters["mu"]["x"]
```

After fitting, conditional uncertainty for each smooth contribution is
available from the same data:

```python
curve = model.smooth_inference(
    y,
    design,
    smooth_covariates=smooth_covariates,
)["mu"]["x"]

curve.estimates
curve.covariance_matrix
curve.correlation_matrix
curve.standard_errors
curve.confidence_intervals
```

This follows the `gamlss.pb()` variance calculation and conditions on the
fitted smoothing parameter. The intervals are on the additive predictor scale
and are pointwise. A simulation-based simultaneous band over the supplied
covariate values uses the full within-curve covariance:

```python
band = curve.simultaneous_confidence_band(
    simulations=10_000,
    generator=torch.Generator().manual_seed(2026),
)
band.critical_value
band.confidence_intervals
```

The generator makes the Monte Carlo result reproducible. Neither kind of
interval accounts for uncertainty from estimating the smoothing parameter.

For covariance across several smooth terms or distribution parameters, use
the joint fixed-lambda information matrix:

```python
joint_analytic = model.smooth_joint_inference(
    y,
    design,
    smooth_covariates=smooth_covariates,
)

cross_covariance = joint_analytic.covariance_block(
    ("mu", "x"),
    ("sigma", "z"),
)
coefficient_covariance = joint_analytic.coefficient_covariance_matrix
joint_analytic_bands = joint_analytic.simultaneous_confidence_bands(
    generator=torch.Generator().manual_seed(2026),
)
```

This calculation includes linear and constrained spline coefficients in one
expected penalized-information system. It retains cross-term and
cross-parameter covariance while conditioning on the fitted lambdas.

To propagate that uncertainty, simulate from the fitted distribution and
repeat the complete fit, including lambda selection:

```python
bootstrap = model.smooth_bootstrap(
    y,
    design,
    smooth_covariates=smooth_covariates,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)["mu"]["x"]

bootstrap.standard_errors
bootstrap.confidence_intervals
bootstrap.bootstrap_smoothing_parameters
bootstrap.smoothing_parameter_confidence_interval
bootstrap_band = bootstrap.simultaneous_confidence_band()
```

These are pointwise percentile intervals from a fixed-design parametric
bootstrap. Failed refits are replaced by new attempts up to `max_attempts`;
inspect `failure_rate` rather than ignoring convergence problems. The
bootstrap band uses the maximum standardized error across the same successful
refits and is simultaneous over the evaluation points for that smooth.

When the model contains several smooths, preserve their replicate alignment
with the joint API:

```python
joint = model.smooth_joint_bootstrap(
    y,
    design,
    smooth_covariates=smooth_covariates,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)

cross_covariance = joint.covariance_block(("mu", "x"), ("sigma", "z"))
lambda_covariance = joint.smoothing_parameter_covariance_matrix
lambda_labels = joint.smoothing_parameter_labels
joint_bands = joint.simultaneous_confidence_bands()
```

The full `covariance_matrix` follows `term_order` and `term_slices`.
Joint bands use one max-|t| critical value over every selected curve point.
Smoothing-parameter arrays use one column per penalty:
`smoothing_parameter_labels` identifies each
`(parameter, term, penalty_index)`, and `smoothing_parameter_slices` maps
terms to their penalty columns. Scalar-only models retain one column per term.

Aligned refits can also be transformed without rerunning the bootstrap:

```python
difference = joint.difference(("mu", "x"), ("sigma", "x"))
derivative = joint.derivative(("mu", "x"))
peak = difference.extremum(kind="maximum")
root = difference.crossing(level=0.0, which="first")
```

Contrasts require a common ordered evaluation grid. Derivatives and crossings
additionally require that grid to be strictly increasing. Extrema are located
on the supplied grid; crossings use linear interpolation between its points.
See [`INFERENCE.md`](INFERENCE.md) for interval interpretation and
missing-crossing diagnostics.

The same smooth terms and selection modes work with `fit_cg()` and
`CGControl`; CG performs one `additive.fit()`-style pass per parameter update
inside its joint Gauss-Seidel cycle.

The explicit linear `x` column mirrors the parametric component that R keeps
for `pb(x)`. Smooth terms may be configured independently for every family
parameter, and multiple named terms can be attached to the same parameter.
The same names must be supplied in `smooth_covariates` during fitting and
prediction. CG parity fixtures cover both configurations: smooths in `mu` and
`sigma` simultaneously, and two smooths attached to `mu`.

`PSpline.from_data()` reproduces the range expansion and small-sample interval
rules used by `pb()`. The knots, penalty, and coefficients are registered in
the Torch module state and move with `.to(device)` or `.to(dtype)`. The most
recently estimated `lambda` is available as `term.smoothing_parameter` and is
also stored in the module state.

To keep `lambda` fixed, pass it explicitly:

```python
term = PSpline.from_data(x, smoothing_parameter=12.0)
```

To request the equivalent of `pb(x, df=3)`:

```python
term = PSpline.from_data(x, degrees_of_freedom=3.0)
# The target total EDF is 5.
```

To select `lambda` with local GAIC or GCV:

```python
gaic_term = PSpline.from_data(
    x, smoothing_method="GAIC", criterion_penalty=2.0
)
gcv_term = PSpline.from_data(
    x, smoothing_method="GCV", criterion_penalty=2.0
)
```

`smoothing_parameter` and `degrees_of_freedom` are mutually exclusive.

## Generic multiple-penalty systems

The low-level dense solver accepts one or more fixed coefficient-space
penalties and optional linear equality constraints. It minimizes

```text
sum_i w_i (y_i - X_i beta)^2 + beta' (sum_j lambda_j S_j) beta
```

subject to

```text
C beta = 0.
```

For example:

```python
import torch
from torchgamlss import solve_penalized_least_squares

design = torch.tensor(
    [[1.0, -1.0, 0.5], [1.0, 0.0, -0.2], [1.0, 1.0, 0.8]],
    dtype=torch.float64,
)
response = torch.tensor([0.0, 0.5, 1.4], dtype=torch.float64)
weights = torch.ones_like(response)

first_penalty = torch.diag(
    torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
)
second_penalty = torch.diag(
    torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
)
sum_to_zero = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

fit = solve_penalized_least_squares(
    design,
    response,
    weights,
    (first_penalty, second_penalty),
    (2.0, 5.0),
    constraints=sum_to_zero,
)
coefficients = fit.coefficients
edf = fit.effective_degrees_of_freedom
```

Each `S_j` is validated for shape, finiteness, symmetry, dtype/device
compatibility, and numerical positive semidefiniteness. Smoothing parameters
must be finite and non-negative. Constraints are applied through an SVD
null-space reparameterization; redundant constraint rows are allowed.

The result also exposes fitted values, the combined penalty matrix, individual
penalty ranks, constraint rank and null space, and the reduced system condition
number. Rank-deficient penalties are supported when the complete penalized
system remains identifiable.

This API currently uses fixed lambdas. It does not yet perform whole-model
LAML selection. Classical `PSpline` fitting through RS and CG continues to use
the existing square-root augmented solver for exact `gamlss::pb()` numerical
compatibility. A `PSpline` can nevertheless be passed explicitly through the
generic contract with `term.design(x)`, `term.penalty_matrices()`,
`term.smoothing_parameters`, and `term.constraints(x)`.

## Tensor-product terms

`TensorProductSmooth` constructs a multidimensional smooth from two or more
single-penalty marginal `SmoothTerm` bases. Its design is the row-wise
Kronecker product of the marginal designs and it embeds one coefficient-space
penalty per marginal direction. `TensorInteractionSmooth` first centers each
marginal basis, excluding its main-effect direction before constructing the
highest-order interaction.

Both terms store only their product coefficients as trainable parameters.
Marginal basis state, interaction transforms, smoothing parameters, and
prediction mappings participate in normal Torch state serialization and
device/dtype movement. Their design, penalty, constraint, EDF, and quadratic
penalty paths work with autograd and the generic dense solver on CPU or CUDA.
Formula `te()` absorbs the full tensor's global sum-to-zero constraint into
its coefficient mapping; formula `ti()` stores its marginal interaction
transforms. Both therefore retain identifiability when fitted with fixed
lambdas through RS, CG, L-BFGS, or mini-batch Adam and when predicting new
data. RS and CG delegate each multiply penalized partial-residual update to
the generic constrained solver while retaining the original scalar
square-root path for exact `pb()` compatibility.

Formula `te()`/`ti()` without `lambda_=` instead marks every marginal
parameter for joint selection. `fit_laml_data()` assembles all scalar and
tensor penalties in one Normal location-scale objective, selects the free log
lambdas together, and writes the result back to the model. Use
`initial_lambda_=` for LAML starting values. The result exposes flat
`(parameter, term, penalty_index)` labels and term slices.

See [`TENSOR_SMOOTHS.md`](TENSOR_SMOOTHS.md) for equations and examples. The
low-level row-product and penalty embedding are checked against
`mgcv::tensor.prod.model.matrix()` and
`mgcv::tensor.prod.penalties()`.

## Current limitations

- Only equally spaced P-splines are available as tensor marginal bases.
- Generic multiple-penalty systems currently use the dense low-level solver
  directly, through fixed-lambda formula RS/CG/L-BFGS/mini-batch fitting, or
  through dense whole-model LAML for Normal location-scale, Poisson log-mean,
  NBI mean/dispersion, Gamma mean/CV, Beta mean/dispersion, Student-t
  location/scale/shape, and BCCG location/scale/shape models.
- Linear-coefficient inference, within-curve covariance, pointwise smooth
  intervals, and simultaneous smooth bands are available conditionally for
  additive models, including fixed-lambda tensor terms. Analytic inference
  provides joint covariance across linear coefficients, smooth coefficients,
  smooth terms, and distribution parameters. Parametric smooth-bootstrap
  summaries support scalar- and vector-lambda terms through RS, CG, or
  supported whole-model LAML fits, with one stored bootstrap column per
  penalty.
  LAML refits repeat joint scalar/tensor selection in each successful
  replicate.
- Automatic smoothing selection is available through `fit_rs()` and
  `fit_cg()`; joint L-BFGS fitting requires fixed smoothing parameters.
- Prediction uses the stored B-spline basis. Out-of-range extrapolation parity
  with R's natural-spline prediction helper has not been verified.
