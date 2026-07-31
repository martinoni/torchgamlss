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
coefficients. Three-parameter TF and PE models do the same with a log-linked
`nu`. Four-parameter BCT and BCPE models additionally retain covariance
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

`smooth_inference_data()` returns full within-curve covariance, pointwise
standard errors, and confidence intervals for each fitted smooth contribution:

```python
import torch

curves = model.smooth_inference_data(data, weights="weight")
mu_x = curves["mu"]["x"]

mu_x.estimates
mu_x.covariance_matrix
mu_x.correlation_matrix
mu_x.standard_errors
mu_x.confidence_intervals
table = mu_x.to_dataframe()
```

The estimates and intervals are on the additive predictor scale. For a
P-spline basis `B`, evaluation basis `B_*`, working weights `W`, penalty `D`,
and fitted `lambda`, the raw covariance uses

```text
B_* (B' W B + lambda D' D)^-1 B_*'.
```

The covariance of the unpenalized polynomial null space is then removed
because that component is already represented by the linear predictor. The
diagonal reproduces the `var` component created by `gamlss.pb()` and
subsequently used by `predict.gamlss(..., se.fit=TRUE)`.

For a multiply penalized tensor term, the corresponding system is

```text
B' W B + sum_j lambda_j S_j.
```

An explicit tensor constraint is imposed through the same null-space
reparameterization used during fitting. Formula `te()` has already absorbed
its sum-to-zero constraint; `ti()` has already absorbed its marginal
interaction transforms. Their remaining penalty null spaces are not
subtracted because, unlike `gamlss::pb()`, those directions are not duplicated
in the linear predictor. `smoothing_parameter` is a tuple for these terms.
For multivariate covariates, `to_dataframe()` names the inputs
`covariate_0`, `covariate_1`, and so on.

Intervals use pointwise normal critical values. They are conditional on the
fitted smoothing parameter and final working weights: they do not incorporate
selection uncertainty in `lambda`.

### Simultaneous bands

The full covariance supports a conditional simultaneous band over all
covariate values in the result:

```python
band = mu_x.simultaneous_confidence_band(
    simulations=10_000,
    generator=torch.Generator().manual_seed(2026),
)

band.critical_value
band.confidence_intervals
band.to_dataframe()
```

Each simulation draws a Gaussian curve from the conditional covariance and
records its maximum absolute standardized deviation. The empirical
`confidence_level` quantile is used in place of the pointwise normal critical
value. A seeded generator gives reproducible Monte Carlo limits; increasing
`simulations` reduces simulation error.

The band is simultaneous only across the evaluation points for one returned
smooth. Smooths in several distribution parameters and multiple smooths in
one parameter are still returned independently. The band remains conditional
on the fitted `lambda` and final working weights, so it does not incorporate
smoothing-parameter selection uncertainty.

`covariance_matrix` has one row and column per evaluation point and is
materialized only when accessed. The band calculation uses an internal
low-rank covariance factor directly, so it does not need to allocate that
square matrix.

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

## Joint analytic penalized covariance

`smooth_joint_inference_data()` constructs one fixed-`lambda` covariance for
every linear and spline coefficient in the model:

```python
joint = model.smooth_joint_inference_data(
    data,
    weights="weight",
    new_data=new_data,
)

joint.term_order
joint.coefficient_names
joint.coefficient_covariance_matrix
joint.covariance_matrix
mu_sigma_covariance = joint.covariance_block(
    ("mu", "x"),
    ("sigma", "z"),
)
joint_table = joint.to_dataframe()
```

The calculation stacks the reduced design matrices for all distribution
parameters. For parameter pair `a, b`, the corresponding information block is

```text
Z_a' W_ab Z_b,
```

where `W_ab` is the expected link-scale information, including case weights.
The diagonal smooth blocks additionally contain either `lambda D'D` or
`sum_j lambda_j S_j`. A scalar `pb()` spline is restricted to the complement
of its unpenalized null-function space under its parameter's working weights.
This is the joint form of the null-space subtraction used by `gamlss.pb()`.
A tensor instead uses its explicit coefficient constraint, if any, without
discarding its distinct lower-order null directions. Both routes retain
cross-smooth and cross-parameter covariance.

`coefficient_covariance_matrix` follows `coefficient_names`;
`linear_coefficient_slices` and `smooth_coefficient_slices` locate its blocks.
The full-coordinate spline covariance is positive semidefinite rather than
positive definite because the identifiability constraints have zero variance.
`covariance_matrix` follows `term_order`, `term_slices`, and `point_labels`.

Joint Gaussian max-|t| bands use the same covariance factor:

```python
joint_band = joint.simultaneous_confidence_bands(
    simulations=10_000,
    generator=torch.Generator().manual_seed(2026),
)
mu_x_band = joint_band["mu"]["x"]
```

One critical value is calibrated over every selected curve point. Pass
`terms=[("mu", "x")]` to restrict the family. The calculation is analytic
conditional on the fitted smoothing parameters; only the Gaussian critical
value is simulated. R fixtures independently reconstruct the complete
penalized `pb()` coefficient and curve covariance for the one-smooth Normal
case, in which the result reduces to the standard `gamlss.pb()` variance.

Use the parametric bootstrap below when uncertainty from estimating or
selecting `lambda` must also be propagated.

## Parametric bootstrap with lambda reselection

`smooth_bootstrap_data()` propagates response, coefficient, and
smoothing-selection variability by simulating from the fitted GAMLSS
distribution and refitting the complete model:

```python
bootstrap = model.smooth_bootstrap_data(
    data,
    weights="weight",
    new_data=new_data,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)
mu_x_bootstrap = bootstrap["mu"]["x"]

mu_x_bootstrap.standard_errors
mu_x_bootstrap.confidence_intervals
mu_x_bootstrap.bootstrap_estimates
mu_x_bootstrap.bootstrap_smoothing_parameters
mu_x_bootstrap.smoothing_parameter_standard_error
mu_x_bootstrap.smoothing_parameter_confidence_interval
bootstrap_band = mu_x_bootstrap.simultaneous_confidence_band()
bootstrap_table = mu_x_bootstrap.to_dataframe()
```

The same result supports both scalar- and multiple-penalty terms. For `pb()`,
`smoothing_parameter` remains a float and
`bootstrap_smoothing_parameters` retains its historical `(replicates,)`
shape. For a `te()` or `ti()` term with `J` marginal penalties,
`smoothing_parameter` is a `J`-tuple and the bootstrap tensor has shape
`(replicates, J)`. The smoothing-parameter mean, standard error, and bias are
floats for a scalar term and length-`J` tensors for a multiple-penalty term;
its percentile intervals have shape `(J, 2)`.

For each successful replicate, TorchGAMLSS:

1. draws one response at every original design row from the fitted
   distribution;
2. clones the fitted model;
3. reruns RS or CG, including ML, GAIC, GCV, or target-EDF lambda selection,
   or reruns whole-model LAML with joint scalar/tensor lambda selection;
4. evaluates every smooth on the requested covariate values.

Use the same `algorithm=` and `control=` settings as the original fit so the
bootstrap distribution represents the estimator that produced the reported
model. For a supported Normal or Poisson model fitted with
`fit_laml_data()`, use:

```python
from torchgamlss import LAMLControl

bootstrap = model.smooth_joint_bootstrap_data(
    data,
    new_data=new_data,
    replicates=999,
    algorithm="laml",
    control=LAMLControl(),
    generator=torch.Generator().manual_seed(2026),
)
```

This repeats the complete nested LAML optimization in every successful
replicate. Automatic `pb()`, `te()`, and `ti()` lambdas are selected jointly;
fixed formula lambdas remain fixed. LAML bootstrap currently supports additive
Normal models with identity-`mu`/log-`sigma` links and Poisson models with a
log-`mu` link. It is materially more expensive than RS or CG bootstrap, so
use `max_attempts` and inspect `failure_rate`.

The reported pointwise intervals are percentile bootstrap intervals.
`standard_errors` and `covariance_matrix` are empirical across the successful
refitted curves. The original model is never mutated. All ten public
families provide the response sampler required by this workflow.

`simultaneous_confidence_band()` uses the replicate distribution of the
maximum absolute standardized curve error. It therefore propagates lambda
selection into a max-|t| band over the evaluation points of that one smooth,
without another simulation pass. Inspect its `method` field to distinguish it
from the conditional Gaussian band.

For covariance and family-wise bands across several smooths, request the
aligned joint result directly:

```python
joint = model.smooth_joint_bootstrap_data(
    data,
    weights="weight",
    new_data=new_data,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)

joint.term_order
joint.bootstrap_estimates
joint.covariance_matrix
joint.correlation_matrix
mu_sigma_covariance = joint.covariance_block(
    ("mu", "x"),
    ("sigma", "z"),
)

joint.bootstrap_smoothing_parameters
joint.smoothing_parameter_labels
joint.smoothing_parameter_slices
joint.smoothing_parameter_covariance_matrix
joint.smoothing_parameter_correlation_matrix

joint_bands = joint.simultaneous_confidence_bands()
mu_x_joint_band = joint_bands["mu"]["x"]
joint_table = joint.to_dataframe()
```

Every row of `joint.bootstrap_estimates` and
`joint.bootstrap_smoothing_parameters` comes from the same simulated response
and complete model refit. `term_order`, `term_slices`, and `point_labels`
identify the stacked curve coordinates. `covariance_block()` can therefore
measure dependence between different smooths, including smooths attached to
different distribution parameters.

Joint smoothing-parameter arrays are flattened at the penalty level rather
than the term level. `smoothing_parameter_labels` records
`(parameter, term, penalty_index)` for every column, while
`smoothing_parameter_slices` locates all penalties belonging to each term.
This is unchanged for scalar-only models: every term still contributes one
column.

`simultaneous_confidence_bands()` calibrates one max-|t| critical value over
all points of all selected smooths. Pass a sequence such as
`terms=(("mu", "x"), ("sigma", "z"))` to restrict the family. Its method is
`parametric_bootstrap_joint_max_t`. The individual
`simultaneous_confidence_band()` method remains available when simultaneous
coverage is required only within one curve.

If a smoothing parameter is fixed in the formula, its bootstrap variance is
zero and its correlations are undefined (`nan`). Its covariance with every
other smoothing parameter is zero up to numerical precision.

## Response quantile and centile bootstrap

`quantile_bootstrap_data()` and `centile_bootstrap_data()` propagate the
complete fitted distribution into response-scale quantile curves:

```python
centile_bootstrap = model.centile_bootstrap_data(
    data,
    centiles=[3, 10, 50, 90, 97],
    weights="weight",
    new_data=new_data,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)

centile_bootstrap.estimates
centile_bootstrap.bootstrap_estimates
centile_bootstrap.standard_errors
centile_bootstrap.confidence_intervals
centile_bootstrap.covariance_matrix
centile_table = centile_bootstrap.to_dataframe()
```

Each successful replicate simulates one training response, refits every
distribution parameter, repeats configured smoothing selection, predicts all
parameters on `new_data`, and then evaluates the requested family quantiles.
Thus a BCT or BCPE centile curve, for example, propagates the joint variation
of `mu`, `sigma`, `nu`, and `tau`. RS and CG are available for every
sampleable supported family. `algorithm="laml"` is additionally available
for additive Normal models and repeats joint scalar/tensor selection before
evaluating each quantile curve.

Pointwise limits are percentile bootstrap intervals. The flattened
`covariance_matrix` follows prediction row first and probability second.
`at(0.5)` selects the stored median intervals.

```python
joint_band = centile_bootstrap.simultaneous_confidence_bands()
per_centile = centile_bootstrap.simultaneous_confidence_bands(joint=False)
```

The default band controls max-|t| over all prediction rows and all requested
centiles with one critical value. With `joint=False`, each centile receives
its own critical value and simultaneous band over prediction rows. Neither
choice changes the already computed bootstrap replicates.

For Poisson and NBI, fitted quantiles and bootstrap replicates are
integer-valued. Percentile interpolation and max-|t| construction can produce
non-integer interval limits. Some fitted coordinates can have zero bootstrap
variance; a simultaneous band is unavailable if an entire requested band has
no positive variance.

## Derived smooth functionals

The joint result can transform all aligned replicates without another model
fit. This supports direct inference for curve differences, derivatives,
extrema, and crossings:

```python
difference = joint.difference(
    ("mu", "x"),
    ("sigma", "x"),
)
contrast = joint.linear_contrast(
    {
        ("mu", "x"): 2.0,
        ("sigma", "x"): -0.5,
    },
    name="2 mu(x) - 0.5 sigma(x)",
)

first_derivative = joint.derivative(("mu", "x"))
second_derivative = joint.derivative(("mu", "x"), order=2)

difference.standard_errors
difference.confidence_intervals
difference_band = difference.simultaneous_confidence_band()

peak = difference.extremum(kind="maximum")
peak.estimate
peak.location
peak.confidence_interval
peak.location_confidence_interval

root = difference.crossing(
    level=0.0,
    direction="decreasing",
    which="first",
)
root.estimate
root.confidence_interval
root.missing_rate
```

`difference()` and `linear_contrast()` require the source curves to be
evaluated on the same covariate grid in the same order. Their pointwise
intervals are percentile bootstrap intervals, while
`simultaneous_confidence_band()` recalibrates max-|t| over the derived curve.
Contrasts across distribution parameters operate on their additive predictor
or link scales; only combine them when that contrast has a meaningful
scientific interpretation.

Derivatives use first-order finite differences at the boundaries and centered
differences in the interior through `torch.gradient`. The evaluation
covariate must contain at least three finite, strictly increasing points.
First and second derivatives are supported. Derivative uncertainty includes
the coefficient and smoothing-selection variability represented by the
original bootstrap but not uncertainty from choosing the evaluation grid.

`extremum()` finds the maximum or minimum over the supplied discrete grid.
Its location distribution is therefore grid-valued; use a sufficiently dense
`new_data` grid when location precision matters. Ties use the first grid
location.

`crossing()` linearly interpolates between adjacent grid points. Use
`direction=` and `which=` when a curve has several crossings. Some bootstrap
curves may not cross the requested level; these are reported through
`missing_replicates` and `missing_rate`. The crossing interval is conditional
on the valid bootstrap curves, so a large missing rate is a substantive
warning rather than a convergence failure.

Bootstrap refits can occasionally fail or reach their iteration limit.
`replicates` counts successful fits; by default the method allows up to the
larger of `replicates + 10` and `1.2 * replicates` attempts. Set
`max_attempts=` explicitly when needed, and inspect `attempts`,
`failed_replicates`, and `failure_rate`. Exhausting the attempt budget raises
an error rather than returning an undersized bootstrap sample.

This is a fixed-design parametric bootstrap: covariates, offsets, and weights
remain fixed while responses are simulated. Weights are reused as case or
prior weights. If integer weights represent literal replicated observations,
expand those observations before bootstrapping so each replicate receives an
independent simulated response.

The bootstrap is substantially more expensive than conditional inference;
`999` is a practical starting point, while final tail inference may require
more replicates. Pointwise percentile intervals remain marginal. Use the joint
max-|t| result when simultaneous coverage across several terms is required.

Methodologically, the local ML lambda update follows
[Rigby and Stasinopoulos (2014)](https://doi.org/10.1177/0962280212473302).
The simulate-and-refit design follows the general parametric-bootstrap
strategy for GAMLSS described by
[Hohberg, Pütz, and Kneib (2020)](https://doi.org/10.1371/journal.pone.0226514).
For broader context on interval estimation and smoothing-parameter variability
in penalized GAMs, see
[Wood (2006)](https://doi.org/10.1111/j.1467-842X.2006.00450.x).

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
Conditional smooth inference and within-curve simultaneous bands support
additive RS and CG fits, including fixed-lambda `te()` and `ti()` terms.
Fixed-lambda analytic inference provides joint covariance across linear
coefficients, smooth coefficients, smooth terms, and distribution parameters.
Parametric-bootstrap smooth inference supports scalar- and multiple-penalty
terms with RS, CG, and supported whole-model LAML refits. It repeats available
smoothing-parameter selection, stores one lambda column per penalty, and
provides empirical joint covariance, simultaneous bands, and derived-curve
functionals across fitted smooths. Tensor lambdas must be fixed for RS/CG
refits, while `algorithm="laml"` reselects automatic tensor lambdas jointly
in every successful bootstrap sample.

The Hessian and conditional smooth calculations are local Wald
approximations. They do not replace profile likelihood or robust sandwich
covariance. The parametric bootstrap adds a simulation-based alternative but
depends on the fitted family and convergence of the refits.
