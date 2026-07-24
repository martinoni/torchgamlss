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
re-estimated for new observations. Prediction remains differentiable and can
participate in Torch autograd computations.

The stored P-spline basis can evaluate covariates outside the training range,
but exact parity with the natural-spline extrapolation helper used by R has
not yet been established. For strict R compatibility, keep prediction
covariates inside the fitted range for now.

Models constructed with `GAMLSS.from_formula()` can use `predict_data()` to
materialize these inputs directly from a pandas DataFrame or compatible
mapping. See [`FORMULAS.md`](FORMULAS.md).

## Conditional smooth intervals

For fitted formula models, `smooth_inference_data()` adds pointwise standard
errors and confidence intervals to each smooth contribution:

```python
curves = model.smooth_inference_data(
    training_data,
    weights="weight",
    new_data=new_data,
)
mu_x = curves["mu"]["x"]
```

The fitted curve is `mu_x.estimates`; its pointwise uncertainty is in
`standard_errors` and `confidence_intervals`. The calculation conditions on
the fitted smoothing parameter and does not produce a simultaneous band. See
[`INFERENCE.md`](INFERENCE.md).
