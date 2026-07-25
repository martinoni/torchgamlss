# Migrating from R `gamlss` to TorchGAMLSS

TorchGAMLSS follows the parameterizations and classical fitting equations of
R `gamlss`, but it is not a drop-in syntactic replacement. The main lifecycle
difference is that R constructs and fits one `gamlss` object in a single call,
whereas Python first constructs a `GAMLSS` model and then fits that model.

The current numerical reference is:

- `gamlss` 5.5-0;
- `gamlss.dist` 6.1-1;
- float64 Torch calculations.

See [`PARITY.md`](PARITY.md) for the fixture protocol and verified numerical
tolerances.

## End-to-end translation

This R model:

```r
fit_r <- gamlss(
  y ~ pb(x) + offset(mu_offset),
  sigma.formula = ~ pb(z) + offset(sigma_offset),
  weights = weight,
  family = BE(),
  method = CG(),
  data = data,
  control = gamlss.control(
    c.crit = 1e-8,
    n.cyc = 300,
    trace = FALSE
  ),
  i.control = glim.control(cc = 1e-8, cyc = 300)
)
```

translates to:

```python
from torchgamlss import Beta, CGControl, GAMLSS

model = GAMLSS.from_formula(
    Beta(),
    {
        "mu": "y ~ pb(x) + offset(mu_offset)",
        "sigma": "~ pb(z) + offset(sigma_offset)",
    },
    data,
)

fit = model.fit_cg_data(
    data,
    weights="weight",
    control=CGControl(
        outer_tolerance=1e-8,
        max_outer_iterations=300,
        inner_tolerance=1e-8,
        max_inner_iterations=300,
    ),
)
```

`model` owns the fitted coefficients, smooth terms, links, and formula
encodings. `fit` is an immutable convergence and effective-degrees-of-freedom
summary. Fitting mutates `model`, so predictions are made from `model`, not
from `fit`.

## Family mapping

| R `gamlss.dist` | TorchGAMLSS | Parameters | Default links |
| --- | --- | --- | --- |
| `NO()` | `Normal()` | `mu`, `sigma` | identity, log |
| `GA()` | `Gamma()` | `mu`, `sigma` | log, log |
| `PO()` | `Poisson()` | `mu` | log |
| `NBI()` | `NegativeBinomial()` | `mu`, `sigma` | log, log |
| `BE()` | `Beta()` | `mu`, `sigma` | logit, logit |
| `BCCG()` | `BCCG()` or `BoxCoxColeGreen()` | `mu`, `sigma`, `nu` | identity, log, identity |
| `BCT()` | `BCT()` or `BoxCoxT()` | `mu`, `sigma`, `nu`, `tau` | identity, log, identity, log |
| `BCPE()` | `BCPE()` or `BoxCoxPowerExponential()` | `mu`, `sigma`, `nu`, `tau` | identity, log, identity, log |

The distribution conventions are the R conventions, not necessarily those
of a similarly named `torch.distributions` class. In particular:

- `Gamma.sigma` is the coefficient of variation;
- `NegativeBinomial.sigma` satisfies
  `Var(Y) = mu + sigma * mu**2`;
- `Beta.sigma` satisfies
  `Var(Y) = sigma**2 * mu * (1 - mu)`.

Links are supplied as objects rather than strings:

```r
family_r <- NO(mu.link = "log")
```

```python
from torchgamlss import LogLink, Normal

family = Normal(mu_link=LogLink())
```

The public link classes are `IdentityLink`, `LogLink`, `LogitLink`, and
`InverseLink`.

## Quantile and centile prediction

R family quantile functions map to one common model API:

| R | TorchGAMLSS |
| --- | --- |
| `qNO()`, `qGA()`, `qBE()` | `model.predict_quantiles_data(...)` |
| `qPO()`, `qNBI()` | `model.predict_quantiles_data(...)` |
| `qBCCG()`, `qBCT()`, `qBCPE()` | `model.predict_quantiles_data(...)` |
| centile percentages | `model.predict_centiles_data(...)` |

```python
centiles = model.predict_centiles_data(
    new_data,
    centiles=[3, 10, 50, 90, 97],
)
```

Unlike calling one R `q*` function with manually assembled parameters, the
model method predicts every parameter from its own formula before evaluating
the response quantile. `centile_bootstrap_data()` repeats the complete RS or
CG fit and returns pointwise intervals plus max-|t| bands.

## Parameter formulas

R gives the first parameter the main formula and uses named formula arguments
for the others. Python uses one mapping keyed by the family parameter names:

```r
fit_r <- gamlss(
  y ~ x + C(group),
  sigma.formula = ~ z,
  nu.formula = ~ w,
  tau.formula = ~ 1,
  family = BCT(),
  data = data
)
```

```python
from torchgamlss import BCT, GAMLSS

model = GAMLSS.from_formula(
    BCT(),
    {
        "mu": "y ~ x + C(group)",
        "sigma": "~ z",
        "nu": "~ w",
        "tau": "~ 1",
    },
    data,
)
```

Only the first parameter formula contains the response. Every family
parameter must have a formula, even when its predictor is intercept-only.

Formulaic supplies the Wilkinson formula implementation. Common numerical
terms, `x:z`, `x * z`, intercept control, and `C(group)` are available.
Formula syntax is deliberately not assumed to be identical for every advanced
R expression.

### Weights and offsets

For tabular methods, `weights=` can be a column name or array-like object:

```python
fit = model.fit_rs_data(data, weights="weight")
```

Offsets remain standalone formula terms:

```python
{
    "mu": "y ~ x + offset(mu_offset)",
    "sigma": "~ z + offset(sigma_offset)",
}
```

The low-level tensor methods instead accept a weight tensor and a mapping of
parameter-specific offset tensors.

### Missing values and categorical levels

TorchGAMLSS does not silently drop rows containing missing or non-finite
values. Clean or impute them before model construction. New categorical
levels are rejected during prediction rather than being silently mapped to a
reference level.

## Translating `pb()`

`pb()` is represented directly inside a Python formula. Python reserves the
word `lambda`, so use `smoothing_parameter` or the `lambda_` alias.

| R | Python formula |
| --- | --- |
| `pb(x)` | `pb(x)` |
| `pb(x, lambda=12)` | `pb(x, smoothing_parameter=12)` |
| `pb(x, lambda=12)` | `pb(x, lambda_=12)` |
| `pb(x, df=3)` | `pb(x, df=3)` |
| `pb(x, control=pb.control(method="GAIC", k=2))` | `pb(x, method='GAIC', k=2)` |
| `pb(x, control=pb.control(method="GCV", k=2))` | `pb(x, method='GCV', k=2)` |
| `pb(x, control=pb.control(inter=10, start=8))` | `pb(x, inter=10, initial_smoothing_parameter=8)` |

Canonical Python option names are:

- `smoothing_parameter`;
- `degrees_of_freedom`;
- `initial_smoothing_parameter`;
- `smoothing_method`;
- `criterion_penalty`;
- `intervals`, `degree`, and `penalty_order`;
- `name`.

The aliases `lambda_`, `df`, `method`, `k`, and `inter` are accepted in
formulas. Use `penalty_order` for R's `order`.

The current smoother scope is one-dimensional P-splines. A `pb()` call must be
a standalone additive term over a simple numeric column. Multiple `pb()` terms
can be attached to one predictor, and different distribution parameters can
contain smooths simultaneously.

The fitted state corresponding to `getSmo()` is available through both the
model and result:

```python
term = model.smooth_terms["mu"]["x"]

term.coefficients
term.smoothing_parameter
fit.smoothing_parameters["mu"]["x"]
fit.smooth_effective_degrees_of_freedom["mu"]["x"]
fit.smoothing_iterations["mu"]["x"]
```

See [`SMOOTHS.md`](SMOOTHS.md) for the tensor-level `PSpline` API.

## Fitting algorithm mapping

| R | TorchGAMLSS |
| --- | --- |
| `method=RS()` | `model.fit_rs_data(...)` |
| `method=CG()` | `model.fit_cg_data(...)` |
| no direct equivalent | `model.fit_data(...)` using Torch L-BFGS |

The methods without the `_data` suffix are the low-level tensor equivalents:
`fit_rs()`, `fit_cg()`, and `fit()`.

RS and CG support fixed, ML, target-EDF, GAIC, and GCV smoothing-parameter
selection. The L-BFGS path requires fixed smoothing parameters.

### Control mapping

| R control | `RSControl` | `CGControl` | Notes |
| --- | --- | --- | --- |
| `gamlss.control(c.crit=...)` | `outer_tolerance` | `outer_tolerance` | Outer deviance tolerance |
| `gamlss.control(n.cyc=...)` | `max_outer_iterations` | `max_outer_iterations` | Outer cycle limit |
| `glim.control(cc=...)` | `inner_tolerance` | `inner_tolerance` | Inner deviance tolerance |
| `glim.control(cyc=...)` | `max_inner_iterations` | `max_inner_iterations` | Inner cycle limit |
| `glim.control(bf.tol=...)` | `backfitting_tolerance` | `backfitting_tolerance` | CG performs one additive pass per joint update |
| `glim.control(bf.cyc=...)` | `max_backfitting_iterations` | no direct option | R CG also requests one additive pass |
| all four `*.step` equal | `step` | parameter-specific fields | RS currently exposes one shared step |
| `mu.step` | shared `step` | `mu_step` | |
| `sigma.step` | shared `step` | `sigma_step` | |
| `nu.step` | shared `step` | `nu_step` | |
| `tau.step` | shared `step` | `tau_step` | |
| `autostep` | `autostep` | `autostep` | |
| `gd.tol` | `deviance_tolerance` | `deviance_tolerance` | Allowed deviance increase |
| `trace`, `glm.trace`, `bf.trace` | no equivalent | no equivalent | No iterative console tracing yet |

TorchGAMLSS additionally exposes numerical tolerances and limits for ML
smoothing updates, target-EDF root finding, and GAIC/GCV optimization:

```python
control = RSControl(
    smoothing_tolerance=1e-7,
    max_smoothing_iterations=50,
    edf_tolerance=1.220703125e-4,
    max_edf_iterations=1000,
    criterion_tolerance=1e-8,
    max_criterion_iterations=100,
)
```

The same fields are available on `CGControl`.

## Starting values

R-compatible family defaults are used automatically by `fit_rs_data()` and
`fit_cg_data()`. Override selected parameters on the response-parameter
scale:

```python
fit = model.fit_rs_data(
    data,
    initial_parameters={
        "mu": "mu_start",
        "sigma": 0.5,
    },
)
```

A value may be a scalar, an observation-length array, or, for formula methods,
a column name. These are parameter values such as positive `sigma`, not link
predictors or coefficient starts.

## Accessing fitted results

| R fitted object | TorchGAMLSS |
| --- | --- |
| `fit$converged` | `fit.converged` |
| `fit$iter` | `fit.outer_iterations` |
| `deviance(fit)` or `fit$G.deviance` | `fit.global_deviance` |
| `-as.numeric(logLik(fit))` | `fit.negative_log_likelihood` |
| `fit$df.fit` | `fit.effective_degrees_of_freedom` |
| `fit$mu.df` | `fit.parameter_effective_degrees_of_freedom["mu"]` |
| `coef(fit, what="mu")` | `model.coefficients["mu"]` |
| `fitted(fit, what="mu")` | `model.predict_data(data)["mu"]` |
| `getSmo(fit, parameter="mu", which=1)` | `model.smooth_terms["mu"][term_name]` and the smooth fields on `fit` |

Formula column names can be paired with the linear coefficient tensor:

```python
names = model.formula_column_names["mu"]
values = model.coefficients["mu"].detach().cpu().tolist()
mu_coefficients = dict(zip(names, values))
```

RS and CG result objects also expose:

```python
fit.deviance_history
fit.inner_iterations
fit.backfitting_iterations
fit.smoothing_parameters
fit.smooth_effective_degrees_of_freedom
fit.parameter_effective_degrees_of_freedom
```

## Prediction and term decomposition

Response-scale fitted parameters:

```python
parameters = model.predict_data(new_data)
mu = parameters["mu"]
sigma = parameters["sigma"]
```

Link-scale predictors:

```python
eta = model.predict_data(new_data, type="link")
```

Individual linear, smooth, and offset contributions:

```python
terms = model.predict_data(new_data, type="terms")
mu_terms = terms["mu"]

mu_terms.linear
mu_terms.smooth
mu_terms.offset
mu_terms.total
```

`mu_terms.total` reconstructs `eta["mu"]` up to floating-point rounding.
Stored P-spline knots and smoothing parameters are reused for new data.
Exact R extrapolation behavior outside the fitted covariate range has not yet
been established.

See [`PREDICTION.md`](PREDICTION.md) for the complete interface.

## Diagnostics, residuals, and inference

For a linear model:

```python
diagnostics = model.diagnostics_data(data, weights="weight")

diagnostics.log_likelihood
diagnostics.global_deviance
diagnostics.aic
diagnostics.aicc
diagnostics.gaic(3)
diagnostics.sbc
```

For a model with smooth terms, pass the fitted EDF:

```python
diagnostics = model.diagnostics_data(
    data,
    weights="weight",
    degrees_of_freedom=fit.effective_degrees_of_freedom,
)
```

Quantile residuals correspond to the R quantile-residual workflow:

```python
residuals = model.quantile_residuals_data(data)
```

For discrete families, pass a seeded `torch.Generator` or explicit uniforms
when reproducibility across languages matters.

Joint observed-Hessian Wald inference is available for models without smooth
terms:

```python
inference = model.inference_data(data, weights="weight")
coefficient_table = inference.to_dataframe()
covariance = inference.covariance_matrix
```

For a fitted additive model, the R-compatible conditional calculation holds
the smooth contributions fixed and infers only the linear coefficients:

```python
inference = model.inference_data(
    data,
    weights="weight",
    conditional_on_smooths=True,
    degrees_of_freedom=effective_observations
    - fit.effective_degrees_of_freedom,
)
```

This corresponds to `vcov.gamlss(type="all")` and
`summary.gamlss(type="vcov")`. It excludes uncertainty in the spline
coefficients and smoothing parameters; full joint penalized-term inference is
not yet implemented.

Pointwise uncertainty for the smooth contributions corresponds to the
`se.fit=TRUE` term calculation in R:

```python
import torch

smooth_curves = model.smooth_inference_data(data, weights="weight")
mu_x = smooth_curves["mu"]["x"]
mu_x_table = mu_x.to_dataframe()
```

The intervals are conditional on the fitted `lambda` and use normal critical
values. TorchGAMLSS additionally exposes the implied full within-curve
covariance and can construct a conditional simultaneous band:

```python
covariance = mu_x.covariance_matrix
band = mu_x.simultaneous_confidence_band(
    generator=torch.Generator().manual_seed(2026),
)
```

The covariance and band extend the same fixed-`lambda` calculation; R parity
fixtures directly compare its diagonal (the pointwise variances). The band
does not account for smoothing-parameter selection uncertainty.

TorchGAMLSS provides an explicit parametric-bootstrap path when lambda
selection uncertainty matters:

```python
bootstrap = model.smooth_bootstrap_data(
    data,
    weights="weight",
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)
mu_x_bootstrap = bootstrap["mu"]["x"]
```

Every replicate simulates from the fitted family and reruns the complete RS or
CG fit, so automatic `pb()` lambda selection is repeated. This is a
TorchGAMLSS extension rather than a claim of direct `gamlss` output parity;
the underlying fit and conditional variance remain covered by the R fixtures.

Use `smooth_joint_bootstrap_data()` when several fitted smooths must remain
aligned within each replicate:

```python
joint = model.smooth_joint_bootstrap_data(
    data,
    weights="weight",
    replicates=999,
    generator=torch.Generator().manual_seed(2026),
)
mu_sigma_covariance = joint.covariance_block(
    ("mu", "x"),
    ("sigma", "z"),
)
joint_bands = joint.simultaneous_confidence_bands()
```

This exposes empirical covariance across curve points and selected `lambda`
values and calibrates a single max-|t| band over several smooth terms.

The same aligned result supports derived quantities:

```python
difference = joint.difference(("mu", "x"), ("sigma", "x"))
slope = joint.derivative(("mu", "x"))
peak = difference.extremum()
crossing = difference.crossing(level=0.0)
```

These are TorchGAMLSS extensions built from the simulate-and-refit
distribution, not direct translations of an R `gamlss` return object.

See [`DIAGNOSTICS.md`](DIAGNOSTICS.md) and
[`INFERENCE.md`](INFERENCE.md).

## Low-level tensor API

Advanced Torch workflows can bypass formulas:

```python
import torch
from torchgamlss import GAMLSS, Normal

response = torch.as_tensor(y, dtype=torch.float64)
design = {
    "mu": torch.column_stack((torch.ones_like(x), x)),
    "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
}
model = GAMLSS(Normal(), {"mu": 2, "sigma": 1})
fit = model.fit_rs(response, design)
```

Every parameter receives its own design matrix. Offsets and smooth covariates
are mappings keyed by parameter, matching the structure produced by
`prepare_formula_data()`.

## Numerical compatibility is not iteration identity

The translated RS and CG equations match the pinned R versions, but some
numerical paths intentionally use different primitives:

- Torch linear algebra replaces `lm.wfit`;
- target EDF uses log-scale root finding rather than R `uniroot()` on
  `lambda`;
- GAIC and GCV use bounded Brent minimization on `log(lambda)` rather than
  `nlminb()` on `lambda`;
- TorchGAMLSS keeps coefficients synchronized during CG step halving, while
  an R edge path can leave stored coefficients out of sync with fitted
  predictors.

Consequently, convergence-cycle and optimizer iteration counts can differ
even when the selected smoothing parameter, fitted predictors, deviance, and
effective degrees of freedom agree. The committed parity tests compare the
appropriate final quantities with explicit tolerances.

## Current non-equivalences

TorchGAMLSS is pre-alpha and currently covers a focused subset of R `gamlss`.
Important exclusions include:

- families other than the eight listed above;
- smoothers other than the current `pb()` implementation;
- transformed or interaction smooth terms;
- automatic missing-value row removal;
- profile-likelihood and robust covariance workflows;
- nonparametric and cluster bootstrap intervals;
- analytic joint covariance across different penalized smooth terms;
- the complete R plotting and diagnostic ecosystem;
- exact P-spline extrapolation parity outside the training range.

When translating an existing analysis, first reproduce the family,
parameter formulas, weights, offsets, starting values, algorithm, and control
tolerances. Then compare fitted parameter vectors and global deviance before
porting downstream inference or diagnostics.
