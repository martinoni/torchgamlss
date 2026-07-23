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
Torch-native normal CDF, BCT uses a differentiable Torch Student t CDF, and
BCPE uses a differentiable Torch power-exponential CDF. Quantile residuals are
a post-fit diagnostic API; callers should not rely on differentiation through
them.
