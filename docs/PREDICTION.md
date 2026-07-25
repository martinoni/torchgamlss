# Prediction

`GAMLSS.predict()` exposes predictions on response, link, and additive-term
scales. It accepts the same parameter-specific design matrices, offsets, and
smooth covariates used during fitting and can be called with a different
number of observations.

## Response scale

The default returns one tensor for every distribution parameter:

```python
parameters = model.predict(
    new_design,
    new_offsets,
    smooth_covariates=new_smooth_covariates,
)
mu = parameters["mu"]
sigma = parameters["sigma"]
```

The inverse link for each parameter is applied. For example, `Gamma()` returns
its positive mean `mu` and coefficient of variation `sigma`.

## Conditional quantiles and centiles

Quantile prediction combines every fitted distribution parameter and returns
values on the response scale:

```python
quantiles = model.predict_quantiles_data(
    new_data,
    probabilities=[0.03, 0.1, 0.5, 0.9, 0.97],
)

quantiles.probabilities
quantiles.centiles
quantiles.quantiles
median = quantiles.at(0.5)
quantile_table = quantiles.to_dataframe()
```

The equivalent percentage-oriented API is:

```python
centiles = model.predict_centiles_data(
    new_data,
    centiles=[3, 10, 50, 90, 97],
)
```

Both return `QuantilePrediction`. Its `quantiles` tensor has one row per
observation and one column per requested probability. For count families,
the values are integer-valued discrete quantiles.

Formula-free workflows use `predict_quantiles()` and `predict_centiles()`
with the same design matrices, offsets, and smooth covariates as `predict()`.
Quantile inversion is not differentiable for every family because SciPy is
used where Torch has no reliable inverse CDF.

For response-scale uncertainty with repeated model and lambda estimation:

```python
bootstrap = model.centile_bootstrap_data(
    training_data,
    centiles=[3, 10, 50, 90, 97],
    weights="weight",
    new_data=new_data,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)

bootstrap.standard_errors
bootstrap.confidence_intervals
all_centiles_band = bootstrap.simultaneous_confidence_bands()
per_centile_bands = bootstrap.simultaneous_confidence_bands(joint=False)
```

The default band uses one max-|t| critical value over every prediction row and
every requested centile. `joint=False` calibrates a separate simultaneous
band over the prediction rows for each centile.

## Link scale

Pass `type="link"` to obtain each complete additive predictor:

```python
eta = model.predict(
    new_design,
    new_offsets,
    smooth_covariates=new_smooth_covariates,
    type="link",
)
eta_mu = eta["mu"]
```

## Term scale

`type="terms"` returns a `TermContributions` object for each parameter:

```python
terms = model.predict(
    new_design,
    new_offsets,
    smooth_covariates=new_smooth_covariates,
    type="terms",
)
mu_terms = terms["mu"]
```

Each object contains:

- `linear`: an `n × p` tensor with one contribution per design-matrix column;
- `smooth`: a mapping from configured smooth-term names to `n`-vectors;
- `neural`: the optional neural predictor contribution, or a zero vector;
- `shared`: the optional shared-head contribution, or a zero vector;
- `offset`: the broadcast offset, or a zero vector when no offset was given;
- `total`: the reconstructed link-scale predictor.

Consequently,

```python
mu_terms.total == model.predict(..., type="link")["mu"]
```

up to floating-point rounding. Linear column names are not stored yet because
the current low-level API receives tensors rather than formulas; columns
remain in the same order as the supplied design matrix.

## New data and validation

All parameter design matrices must:

- contain the same number of observations;
- preserve the column counts used to construct the model;
- match the model's dtype and device;
- contain only finite values.

Every configured smooth term must receive its named covariate. P-splines reuse
the fitted knots, coefficients, and smoothing parameter, so no basis is
re-estimated for new observations. Parameter, link, and term prediction remain
differentiable and can participate in Torch autograd computations.

Every configured neural predictor must likewise receive its parameter's
tensor through `neural_inputs`. See [`NEURAL.md`](NEURAL.md) for arbitrary
Torch modules, tabular column mappings, and train/evaluation mode.
Shared backbones use one `shared_input` tensor and are documented in
[`SHARED.md`](SHARED.md).

The stored P-spline basis can evaluate covariates outside the training range,
but exact parity with the natural-spline extrapolation helper used by R has
not yet been established. For strict R compatibility, keep prediction
covariates inside the fitted range for now.

Models constructed with `GAMLSS.from_formula()` can use `predict_data()` to
materialize these inputs directly from a pandas DataFrame or compatible
mapping. See [`FORMULAS.md`](FORMULAS.md).

## Conditional smooth intervals and bands

For fitted formula models, `smooth_inference_data()` adds within-curve
covariance, pointwise standard errors, and confidence intervals to each smooth
contribution:

```python
import torch

curves = model.smooth_inference_data(
    training_data,
    weights="weight",
    new_data=new_data,
)
mu_x = curves["mu"]["x"]
```

The fitted curve is `mu_x.estimates`; its pointwise uncertainty is in
`standard_errors` and `confidence_intervals`. The calculation conditions on
the fitted smoothing parameter. A simultaneous band over all supplied
evaluation points is available from:

```python
band = mu_x.simultaneous_confidence_band(
    simulations=10_000,
    generator=torch.Generator().manual_seed(2026),
)
```

See [`INFERENCE.md`](INFERENCE.md) for the statistical interpretation.

`smooth_bootstrap_data(..., new_data=new_data)` evaluates each parametric
bootstrap refit on the same new-data grid. Unlike the conditional intervals
above, it reruns smoothing-parameter selection and returns pointwise
percentile intervals from the refitted curves.

`smooth_joint_bootstrap_data(..., new_data=new_data)` additionally retains
replicate alignment across all fitted smooths. Its covariance blocks and
joint max-|t| bands therefore describe their dependence on the requested
prediction grid.

Use a shared, sorted, sufficiently dense `new_data` grid before computing
`joint.difference()`, `joint.derivative()`, extrema, or crossings. Derivative
and crossing results describe this numerical grid; they do not extrapolate or
optimize the underlying spline outside it.
