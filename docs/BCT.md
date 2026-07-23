# Box-Cox t family

`BoxCoxT`, also exported as `BCT`, follows
`gamlss.dist::BCT(mu, sigma, nu, tau)`. It is TorchGAMLSS's first
four-parameter family.

BCT uses the same Box-Cox transformation as BCCG:

```text
        ((y / mu)^nu - 1) / (nu sigma),  nu != 0
z(y) =
        log(y / mu) / sigma,             nu  = 0
```

Instead of a normal transformed variable, BCT uses a Student t variable with
`tau` degrees of freedom, truncated at the boundary implied by `y > 0`. Its
density is normalized by:

```text
T_tau(1 / (sigma |nu|)).
```

The parameter constraints and R defaults are:

| Parameter | Constraint | Default link | Initial value |
| --- | --- | --- | --- |
| `mu` | `mu > 0` | identity | `(y + mean(y)) / 2` |
| `sigma` | `sigma > 0` | log | `0.1` |
| `nu` | real | identity | `0.5` |
| `tau` | `tau > 0` | log | `10` |

## Formula fit

All four distribution parameters have independent predictors:

```python
from torchgamlss import BCT, GAMLSS

model = GAMLSS.from_formula(
    BCT(),
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

As with BCCG, the identity link for `mu` matches R but does not itself enforce
positivity. A log link can be supplied when the predictor could cross zero:

```python
from torchgamlss import BCT, LogLink

family = BCT(mu_link=LogLink())
```

## Numerical implementation

The Student t CDF is evaluated in differentiable Torch operations through the
regularized incomplete beta function. It supports gradients and Hessians with
respect to `tau`, which are required by Torch L-BFGS and joint coefficient
inference.

The RS `tau` score follows `gamlss.dist::BCT` exactly, including its forward
difference of `0.01` for the derivative of the truncation normalizer. The
likelihood itself retains the differentiable CDF, so its autograd derivative
can differ slightly from that finite-difference working score.

At `nu = 0`, the model has a log-Student-t response:

```text
log(Y) = log(mu) + sigma T_tau.
```

The implementation uses the analytic Box-Cox limit and keeps gradients in
`nu` finite. For `tau > 1e6`, the log density follows the R convention and
uses the BCCG normal limit.

## Verified scope

Committed R fixtures cover:

- density, CDF, links, four scores, and every expected second derivative;
- default starting values and positive response support;
- autograd through the Student t CDF and all four predictors;
- the `nu = 0` log-Student-t limit and large-`tau` BCCG limit;
- a converged weighted RS fit with parameter-specific formulas and offsets;
- prediction, full-Hessian inference, likelihood criteria, and quantile
  residuals.

Sampling and inverse-CDF methods are not yet exposed by the custom Torch
distribution.
