# Box-Cox power-exponential family

`BoxCoxPowerExponential`, also exported as `BCPE`, follows
`gamlss.dist::BCPE(mu, sigma, nu, tau)`. It combines the Box-Cox transformation
used by BCCG and BCT with a standardized power-exponential transformed
variable.

For a strictly positive response:

```text
        ((y / mu)^nu - 1) / (nu sigma),  nu != 0
z(y) =
        log(y / mu) / sigma,             nu  = 0
```

The standardized power-exponential density for the transformed variable is:

```text
f(t) = tau exp(-0.5 |t / c|^tau)
       / (c 2^(1 + 1/tau) Gamma(1/tau)),

c^2 = 2^(-2/tau) Gamma(1/tau) / Gamma(3/tau).
```

The response density is truncated at the boundary implied by `y > 0` and
normalized by:

```text
F_PE(1 / (sigma |nu|)).
```

The parameter constraints and R defaults are:

| Parameter | Constraint | Default link | Initial value |
| --- | --- | --- | --- |
| `mu` | `mu > 0` | identity | `(y + mean(y)) / 2` |
| `sigma` | `sigma > 0` | log | `0.1` |
| `nu` | real | identity | `1` |
| `tau` | `tau > 0` | log | `2` |

## Formula fit

All four parameters can have independent predictors:

```python
from torchgamlss import BCPE, GAMLSS

model = GAMLSS.from_formula(
    BCPE(),
    {
        "mu": "y ~ age + pb(height)",
        "sigma": "~ age",
        "nu": "~ age",
        "tau": "~ 1",
    },
    data,
)
fit = model.fit_rs_data(data, weights="weight")
```

The default identity link for `mu` matches R but does not itself enforce the
positive parameter constraint. Use an explicit log link if the fitted linear
predictor could cross zero:

```python
from torchgamlss import BCPE, LogLink

family = BCPE(mu_link=LogLink())
```

## Numerical implementation

The power-exponential CDF uses a Torch-native regularized incomplete gamma
implementation. It remains differentiable with respect to both its argument
and `tau`, enabling Torch L-BFGS fitting and joint full-Hessian inference.

The RS `tau` working score follows `gamlss.dist::BCPE` and uses its forward
difference of `0.001` for the derivative of the truncation normalizer. The
likelihood retains the differentiable CDF, so its autograd derivative can
differ slightly from that finite-difference working score.

Two useful limits are implemented explicitly and tested:

- at `nu = 0`, `log(Y) = log(mu) + sigma T`, where `T` is standardized
  power-exponential;
- at `tau = 2`, BCPE is exactly the BCCG family.

## Verified scope

Committed R fixtures cover:

- density, CDF, links, four scores, and all expected second derivatives;
- default starting values and strictly positive response support;
- the incomplete-gamma CDF and gradients with respect to shape and argument;
- the `nu = 0` log-power-exponential and `tau = 2` BCCG limits;
- a converged weighted RS fit with parameter-specific formulas and offsets;
- prediction, full-Hessian inference, likelihood criteria, and quantile
  residuals.

Sampling and inverse-CDF methods are not yet exposed by the custom Torch
distribution.
