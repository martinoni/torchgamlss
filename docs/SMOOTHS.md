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

## Current limitations

- Only one-dimensional, equally spaced P-spline bases are available.
- Linear-coefficient inference, within-curve covariance, pointwise smooth
  intervals, and simultaneous smooth bands are available conditionally for
  additive models. Full joint uncertainty across smooth terms and
  smoothing-parameter estimation is not yet available.
- Automatic smoothing selection is available through `fit_rs()` and
  `fit_cg()`; joint L-BFGS fitting requires fixed smoothing parameters.
- Prediction uses the stored B-spline basis. Out-of-range extrapolation parity
  with R's natural-spline prediction helper has not been verified.
