# Coefficient inference

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
requires a different convention.

## Scope and limitations

Inference is evaluated at the model's current coefficients, so fit the model
before calling it. The Hessian must be finite and positive definite; singular
or non-identifiable designs are rejected rather than pseudo-inverted.

This first implementation supports linear parametric models, fitted by either
RS or Torch L-BFGS. Models containing smooth terms are rejected because
conditioning on fitted smooth contributions would omit smoothing and spline
coefficient uncertainty. Joint inference for penalized terms is separate
future work.

These are local Wald approximations. They do not replace profile likelihood,
bootstrap inference, robust sandwich covariance, or corrections for
smoothing-parameter estimation.
