# Box-Cox Cole-Green family

`BoxCoxColeGreen`, also exported as `BCCG`, follows
`gamlss.dist::BCCG(mu, sigma, nu)`. It is the first TorchGAMLSS family with a
separate shape predictor.

For a strictly positive response, define:

```text
        ((y / mu)^nu - 1) / (nu sigma),  nu != 0
z(y) =
        log(y / mu) / sigma,             nu  = 0
```

The transformed value follows a standard normal distribution truncated at the
boundary implied by `y > 0`. The density includes the normalizing factor:

```text
Phi(1 / (sigma |nu|)).
```

The parameter constraints and R defaults are:

| Parameter | Constraint | Default link | Initial value |
| --- | --- | --- | --- |
| `mu` | `mu > 0` | identity | `(y + mean(y)) / 2` |
| `sigma` | `sigma > 0` | log | `0.1` |
| `nu` | real | identity | `0.5` |

## Formula fit

Every parameter can have its own formula, smooth terms, and offset:

```python
from torchgamlss import BCCG, GAMLSS

model = GAMLSS.from_formula(
    BCCG(),
    {
        "mu": "y ~ age + pb(height)",
        "sigma": "~ age",
        "nu": "~ age",
    },
    data,
)
fit = model.fit_rs_data(data, weights="weight")
```

The default identity link for `mu` matches R. Because the parameter itself
must remain positive, applications whose linear predictor could cross zero
can choose a log link explicitly:

```python
from torchgamlss import BCCG, LogLink

family = BCCG(mu_link=LogLink())
```

## The `nu = 0` limit

At `nu = 0`, BCCG becomes a lognormal distribution with:

```text
log(Y) ~ Normal(log(mu), sigma).
```

TorchGAMLSS evaluates the Box-Cox quotient with a local series expansion.
This avoids the literal `0 / 0`, keeps the density and CDF continuous, and
provides a finite autograd derivative with respect to `nu` at zero.

## Verified scope

Committed R fixtures cover:

- density, CDF, links, all three scores, and expected second derivatives;
- default starting values;
- autograd agreement after the link chain rule;
- the exact lognormal limit at `nu = 0`;
- a weighted RS fit with parameter-specific predictors and offsets;
- formula prediction, full-Hessian Wald inference, likelihood criteria, and
  quantile residuals.

Sampling and an inverse-CDF method are not yet exposed by the custom Torch
distribution. The implemented surface covers fitting, prediction, inference,
and post-fit diagnostics.

