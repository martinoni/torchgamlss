# Diagnostics and model comparison

TorchGAMLSS evaluates likelihood diagnostics at the model's current fitted
coefficients. The formulas follow `gamlss`:

```text
global deviance = -2 log L
GAIC(k)          = global deviance + k * df
AIC              = GAIC(2)
SBC / BIC        = global deviance + log(n_eff) * df
```

The small-sample correction is:

```text
AICc = AIC + 2 * df * (df + 1) / (N - df - 1)
```

Here, `N` is the number of response rows. To match R GAMLSS, `n_eff` and
residual degrees of freedom use the sum for integer frequency weights and the
number of positive-weight rows for non-integer case weights.

## Model diagnostics

For a parametric formula model:

```python
fit = model.fit_rs_data(data, weights="weight")
diagnostics = model.diagnostics_data(data, weights="weight")

diagnostics.log_likelihood
diagnostics.global_deviance
diagnostics.effective_degrees_of_freedom
diagnostics.residual_degrees_of_freedom
diagnostics.aic
diagnostics.aicc
diagnostics.bic
diagnostics.sbc
diagnostics.gaic(penalty=3)
```

The low-level equivalent is `model.diagnostics(response, design_matrices,
...)`.

An additive RS fit reports both parameter-specific and total effective
degrees of freedom:

```python
fit.parameter_effective_degrees_of_freedom
fit.effective_degrees_of_freedom
```

Pass the fitted total to diagnostics for a model with smooth terms:

```python
diagnostics = model.diagnostics_data(
    data,
    degrees_of_freedom=fit.effective_degrees_of_freedom,
)
```

An explicit value is required because the effective degrees of freedom depend
on the fitted smoothing state, not only on the model structure.

## Comparing models

`compare_models()` ranks named diagnostics and calculates the usual normalized
relative-likelihood weights:

```python
from torchgamlss import compare_models

comparison = compare_models(
    {
        "linear": linear_model.diagnostics_data(data),
        "quadratic": quadratic_model.diagnostics_data(data),
    },
    criterion="aic",
)
```

Supported criteria are `"aic"`, `"aicc"`, `"bic"`, and `"gaic"`. Supply
`penalty=` for GAIC. All models must refer to the same number of original and
effective observations.

## Finite-mixture diagnostics

For a `FiniteMixture`, posterior separation diagnostics complement the
likelihood criteria:

```python
components = model.component_diagnostics(
    response,
    design_matrices,
    weights=weights,
)

components.posterior_probabilities
components.classification
components.effective_counts
components.effective_proportions
components.entropy
components.mean_entropy
components.mean_max_posterior
```

Classifications are zero-based component indices. Effective counts and
summary averages honor case weights. The generalized EM result also retains
the final posterior probabilities and effective component sizes. See
[`MIXTURES.md`](MIXTURES.md).

## Quantile residuals

Continuous families use:

```text
r = Phi^-1(F(y))
```

For discrete families, TorchGAMLSS implements the randomized Dunn-Smyth
residual:

```text
u ~ Uniform(F(y - 1), F(y))
r = Phi^-1(u)
```

Use a seeded Torch generator for reproducible randomization:

```python
generator = torch.Generator().manual_seed(2026)
residuals = model.quantile_residuals_data(data, generator=generator)
```

Alternatively, provide one uniform value in `[0, 1]` per observation. This is
useful for exact reproducibility across languages:

```python
residuals = model.quantile_residuals_data(data, uniforms="uniform_column")
```

`uniforms` applies only to discrete families. For continuous responses the CDF
determines the residual uniquely.

The Normal, Gamma, Poisson, NBI, and Beta CDF implementations use SciPy as a
non-differentiable numerical backend and return a tensor with the model's
original dtype and device, so GPU inputs incur a CPU round trip. BCCG uses a
Torch-native normal CDF, BCT and TF use a differentiable Torch Student-t CDF,
and BCPE and PE use a differentiable Torch power-exponential CDF. Quantile
residuals are a post-fit diagnostic API; callers should not rely on
differentiation through them.

## Four-panel residual plot

`plot_data()` provides the four plots used by the R `plot.gamlss()` workflow:

- quantile residuals against fitted values for `mu` or another selected
  distribution parameter;
- quantile residuals against observation order or a numeric explanatory
  variable;
- a kernel density estimate with the standard-normal density as reference;
- a normal Q-Q plot.

```python
plot = model.plot_data(
    data,
    weights="weight",
    x_variable="age",
)
plot.figure.show()
```

The result keeps the Matplotlib figure and its four axes, plus CPU tensors for
the residuals, fitted values, and plotted x-variable. It also contains the
summary statistics printed by R GAMLSS:

```python
plot.summary.mean
plot.summary.variance
plot.summary.skewness
plot.summary.kurtosis
plot.summary.filliben_correlation
```

The low-level equivalent is:

```python
plot = model.plot(
    response,
    design_matrices,
    weights=weights,
    offsets=offsets,
    smooth_covariates=smooth_covariates,
    x_variable=age,
    x_label="Age",
)
```

Pass four existing Matplotlib axes through `axes=` to control layout or style
each returned axis after plotting. No implicit `show()` call is made.

Integer frequency weights reproduce observations and zero-weight rows are
omitted, following the R residual workflow. For non-integer case weights, each
positive-weight row appears once. Discrete families accept the same
`generator=` or `uniforms=` controls as `quantile_residuals()`.

For ordered observations, replace the first two panels by ACF and PACF:

```python
plot = model.plot_data(
    data,
    time_series=True,
    max_lag=24,
)
```

## Worm plots

A worm plot is a detrended normal Q-Q plot: its vertical coordinate is the
ordered quantile residual minus the corresponding unit-normal quantile.
`wp_data()` calculates the residuals and produces a global plot:

```python
worm = model.wp_data(data)
worm.figure.show()
```

Condition on one numeric explanatory variable to look for local
misspecification:

```python
worm = model.wp_data(
    data,
    x_variable="age",
    n_intervals=4,
    overlap=0,
)
```

The default intervals reproduce `graphics::co.intervals()` and the
non-overlapping boundary adjustment used by R `gamlss::wp()`. Explicit cut
points and overlapping intervals are also supported:

```python
worm = model.wp_data(
    data,
    x_variable="age",
    cut_points=[5, 10, 15],
)

overlapping = model.wp_data(
    data,
    x_variable="age",
    n_intervals=4,
    overlap=0.5,
)
```

The returned `WormPlotResult` exposes the figure, active axes, residuals,
conditioning values, interval matrix, and one `WormPlotPanel` per interval.
Each panel contains the theoretical quantiles, deviations, 95% pointwise
reference bands, and cubic coefficients. For a conditioned plot:

```python
worm.intervals
worm.coefficients  # rows are intervals; columns are 1, z, z**2, z**3
```

The coefficients quantify the worm's displacement and shape, following R's
`lm(deviation ~ z + z**2 + z**3)` calculation. Set
`show_polynomial=False` to omit the fitted curve and coefficient calculation.
Pass existing Matplotlib axes through `axes=` for full layout control.

The low-level model method is `model.wp(response, design_matrices, ...)`.
Residuals from any other model can be inspected without a TorchGAMLSS object:

```python
from torchgamlss import worm_plot

worm = worm_plot(standardized_residuals, x_variable=age)
```

Integer frequency weights, randomized residual controls for discrete
families, model device handling, and training-mode restoration follow
`plot()`.

## Bucket plots

Bucket plots display transformed skewness against transformed kurtosis. They
can reveal residual shape departures and, for the moment version, provide a
graphical Jarque-Bera check:

```python
bucket = model.bp_data(
    data,
    weights="weight",
    bootstrap_replicates=99,
    bootstrap_generator=torch.Generator().manual_seed(2026),
)
bucket.figure.show()
```

Three R-compatible statistic variants are available:

```python
moment = model.bp_data(data, kind="moment")
central = model.bp_data(data, kind="centile.central")
tail = model.bp_data(data, kind="centile.tail")
```

The moment coordinates are

```text
x = skewness / (1 + abs(skewness))
y = excess kurtosis / (1 + abs(excess kurtosis))
```

and the returned statistics also include the untransformed skewness,
kurtosis, excess kurtosis, effective weighted observation count, and
Jarque-Bera statistic. The centile variants reproduce `centileSK()` using the
1st, 25th, 50th, 75th, and 99th weighted percentiles.

Conditioning, explicit cut points, and overlapping intervals use the same API
as worm plots:

```python
bucket = model.bp_data(
    data,
    x_variable="age",
    n_intervals=4,
    overlap=0,
    kind="moment",
)
```

`BucketPlotResult.panels` contains one `BucketPlotPanel` per interval. Each
panel exposes its `BucketStatistics` and the complete bootstrap cloud:

```python
statistics = bucket.panels[0].statistics
statistics.point
statistics.jarque_bera
bootstrap_coordinates = bucket.panels[0].bootstrap_points
```

The standalone equivalent of R's `bp(obj=residuals, ...)` is:

```python
from torchgamlss import bucket_plot

bucket = bucket_plot(
    standardized_residuals,
    weights=weights,
    kind="centile.tail",
)
```

The Matplotlib moment background includes the Pearson admissible-moment
boundary and 95% Jarque-Bera region. The additional named distribution-family
curves in R are loaded from package-specific serialized `.RData` assets and
are not bundled in TorchGAMLSS. Their absence does not change the observed
statistics, bootstrap coordinates, or Jarque-Bera calculation.
