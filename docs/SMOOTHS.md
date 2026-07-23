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
mu = model.linear_predictors(
    design, smooth_covariates=smooth_covariates
)["mu"]
edf = result.smooth_effective_degrees_of_freedom["mu"]["x"]
estimated_lambda = result.smoothing_parameters["mu"]["x"]
```

The explicit linear `x` column mirrors the parametric component that R keeps
for `pb(x)`. Smooth terms may be configured independently for every family
parameter, and multiple named terms can be attached to the same parameter.
The same names must be supplied in `smooth_covariates` during fitting and
prediction.

`PSpline.from_data()` reproduces the range expansion and small-sample interval
rules used by `pb()`. The knots, penalty, and coefficients are registered in
the Torch module state and move with `.to(device)` or `.to(dtype)`. The most
recently estimated `lambda` is available as `term.smoothing_parameter` and is
also stored in the module state.

To keep `lambda` fixed, pass it explicitly:

```python
term = PSpline.from_data(x, smoothing_parameter=12.0)
```

## Current limitations

- GAIC, GCV, and target-EDF selection are not implemented; automatic selection
  currently supports ML only.
- Only one-dimensional, equally spaced P-spline bases are available.
- Standard errors and covariance matrices are not available.
- Automatic smoothing selection is available through `fit_rs()`; joint
  L-BFGS fitting requires fixed smoothing parameters.
- Prediction uses the stored B-spline basis. Out-of-range extrapolation parity
  with R's natural-spline prediction helper has not been verified.
