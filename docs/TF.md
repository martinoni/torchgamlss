# Student-t location-scale family

`StudentT`, also exported as `TF`, follows
`gamlss.dist::TF(mu, sigma, nu)`. It is a heavy-tailed alternative to the
Normal family and is useful when a location-scale regression must be robust
to observations that are unlikely under Gaussian tails.

If `T_nu` is a standard Student-t variable with `nu` degrees of freedom, the
response is:

```text
Y = mu + sigma T_nu.
```

Here, `sigma` is a scale parameter rather than the response standard
deviation. The moments are:

```text
E(Y)   = mu,                         nu > 1
Var(Y) = sigma^2 nu / (nu - 2),      nu > 2.
```

The mean is undefined for `nu <= 1`, and the variance is infinite for
`nu <= 2`.

The parameter constraints and R defaults are:

| Parameter | Constraint | Default link | Initial value |
| --- | --- | --- | --- |
| `mu` | real | identity | `(y + mean(y)) / 2` |
| `sigma` | `sigma > 0` | log | `sd(y)` |
| `nu` | `nu > 0` | log | `10` |

## Formula fit

All three parameters can have independent predictors:

```python
from torchgamlss import GAMLSS, TF

model = GAMLSS.from_formula(
    TF(),
    {
        "mu": "y ~ x + pb(age)",
        "sigma": "~ x",
        "nu": "~ group",
    },
    data,
)
fit = model.fit_rs_data(data, weights="weight")
```

`StudentT()` and `TF()` construct the same family. If both TorchGAMLSS and
`torch.distributions` are imported in one module, the short `TF` alias avoids
the shared `StudentT` class name.

## Numerical implementation

Density, CDF, scores, and Hessians are evaluated with Torch operations. The
CDF reuses the differentiable regularized incomplete-beta implementation used
by BCT, so gradients with respect to `mu`, `sigma`, and `nu` remain available
for joint optimization and full-Hessian inference.

Conditional quantiles use SciPy's Student-t inverse CDF because PyTorch does
not provide `StudentT.icdf()`. The result is restored to the parameter dtype
and device. As in `gamlss.dist`, `nu > 1e6` uses the Normal limit.

## Verified scope

Committed fixtures generated with `gamlss.dist` 6.1-1 and `gamlss` 5.5-0
cover:

- density, CDF, default links, three scores, and all expected second
  derivatives;
- default starting values and unrestricted finite response support;
- autograd through all three predictors;
- mean, variance, response sampling, and the large-`nu` Normal limit;
- conditional quantiles matched to `qTF`;
- a converged weighted RS fit with parameter-specific formulas and offsets;
- a Cole–Green fit exercising all six expected second derivatives;
- prediction, full-Hessian inference, likelihood criteria, and continuous
  quantile residuals;
- CUDA FP16/BF16 mini-batch stress fitting over central and tail regimes.

The definition and parameterization follow the TF documentation and
pp. 382–383 of Rigby, Stasinopoulos, Heller, and De Bastiani (2019),
*Distributions for Modeling Location, Scale, and Shape: Using GAMLSS in R*.
