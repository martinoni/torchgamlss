# Inference

TorchGAMLSS provides joint Wald inference for fitted parametric models. The
covariance matrix is the inverse of the observed Hessian of the weighted
negative log likelihood with respect to all linear coefficients.

Because the Hessian is joint, the result retains covariance between
coefficients belonging to different distribution parameters.

## Formula API

```python
model.fit_rs_data(data, weights="weight")
inference = model.inference_data(data, weights="weight")

inference.covariance_matrix
inference.standard_errors
inference.statistics
inference.p_values
inference.confidence_intervals
inference.correlation_matrix

table = inference.to_dataframe()
```

Formula coefficient names are parameter-qualified, for example
`mu.Intercept`, `mu.x`, `sigma.Intercept`, and `sigma.z`. Low-level models use
stable positional names such as `mu[0]`. The same joint calculation supports
three-parameter BCCG models and retains covariance involving `nu`
coefficients. Four-parameter BCT and BCPE models additionally retain covariance
involving `tau`.

The corresponding tensor API is:

```python
inference = model.inference(
    response,
    design_matrices,
    weights=weights,
    offsets=offsets,
    confidence_level=0.95,
)
```

`by_parameter()` splits any coefficient-aligned tensor, such as estimates or
standard errors, into the family parameter blocks.

## Conditional inference with smooth terms

For additive models, TorchGAMLSS can reproduce the conditional calculation in
`gamlss::vcov.gamlss()`. The fitted smooth contributions are held fixed while
the observed Hessian is evaluated with respect to the linear coefficients:

```python
fit = model.fit_rs_data(data)
inference = model.inference_data(
    data,
    conditional_on_smooths=True,
    degrees_of_freedom=len(data) - fit.effective_degrees_of_freedom,
)
```

`conditional_on_smooths=True` is required explicitly. The returned
`InferenceResult.conditional_on_smooths` flag records that interpretation.
The formula API supplies the training smooth covariates automatically. The
low-level tensor API additionally requires `smooth_covariates=`.

This covariance includes cross-parameter uncertainty among the linear
coefficients, conditional on the fitted smooth functions. It does not contain
spline coefficients and does not account for spline-coefficient or
smoothing-parameter uncertainty. This is the same limitation for which R
`gamlss` warns that standard errors of linear terms may not be appropriate
when additive terms are present.

## Smooth-curve uncertainty

`smooth_inference_data()` returns pointwise standard errors and confidence
intervals for each fitted smooth contribution:

```python
curves = model.smooth_inference_data(data, weights="weight")
mu_x = curves["mu"]["x"]

mu_x.estimates
mu_x.standard_errors
mu_x.confidence_intervals
table = mu_x.to_dataframe()
```

The estimates and intervals are on the additive predictor scale. For a
P-spline basis `B`, working weights `W`, penalty `D`, and fitted `lambda`, the
raw pointwise covariance uses

```text
(B' W B + lambda D' D)^-1.
```

The variance of the unpenalized polynomial null space is then removed because
that component is already represented by the linear predictor. This
reproduces the `var` component created by `gamlss.pb()` and subsequently used
by `predict.gamlss(..., se.fit=TRUE)`.

Intervals use pointwise normal critical values. They are conditional on the
fitted smoothing parameter and final working weights: they do not incorporate
selection uncertainty in `lambda`, and they are not simultaneous confidence
bands. Smooths in several distribution parameters and multiple smooths in one
parameter are returned independently.

The formula API can evaluate a stored basis on new covariate values:

```python
new_curves = model.smooth_inference_data(
    data,
    weights="weight",
    new_data=new_data,
)
```

R does not currently provide `se.fit=TRUE` for new data through
`predict.gamlss`; TorchGAMLSS extends the same fixed-`lambda` covariance to
the stored basis. Keep new covariates inside the fitted range when strict
extrapolation compatibility matters.

## Wald tests and degrees of freedom

Statistics, two-sided p-values, and confidence intervals use a Student t
reference distribution. By default, residual degrees of freedom follow the R
fixtures:

- integer-valued weights are treated as frequency weights, so the effective
  observation count is their sum;
- non-integer weights are treated as case weights, so the effective count is
  the number of observations with positive weight;
- the total number of linear coefficients is subtracted.

Callers can set `degrees_of_freedom=` explicitly when their weighting design
requires a different convention. It is mandatory for conditional smooth
inference because the residual degrees of freedom must use the fitted model's
effective, rather than raw coefficient, dimension. For integer frequency
weights, subtract `fit.effective_degrees_of_freedom` from the sum of weights;
for case weights, subtract it from the number of positive-weight observations.

## Scope and limitations

Inference is evaluated at the model's current coefficients, so fit the model
before calling it. The Hessian must be finite and positive definite; singular
or non-identifiable designs are rejected rather than pseudo-inverted.

Parametric inference supports models fitted by RS, CG, or Torch L-BFGS.
Conditional linear-coefficient inference supports additive RS and CG fits.
Conditional pointwise smooth inference supports additive RS and CG fits.
Full joint inference across spline coefficients, smooth terms, and smoothing
parameters remains separate future work.

These are local Wald approximations. They do not replace profile likelihood,
bootstrap inference, robust sandwich covariance, or corrections for
smoothing-parameter estimation.
