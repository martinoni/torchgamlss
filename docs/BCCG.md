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

With the standard identity/log/identity links, the same model can select all
automatic scalar or tensor smoothing parameters jointly by LAML:

```python
from torchgamlss import LAMLControl

fit = model.fit_laml_data(data, control=LAMLControl())
```

The safeguarded Newton search shortens any trial step that would temporarily
make `mu <= 0` or produce a non-finite BCCG likelihood. Fixed-design
parametric bootstrap accepts `algorithm="laml"` and reselects automatic
lambdas in every successful replicate.

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
  quantile residuals;
- response sampling and conditional quantiles matched to `qBCCG`.
- fixed-lambda LAML inner-fit parity for three simultaneous `pb()` terms
  against the end-to-end R reference, initialized from the independently
  validated fixed-lambda RS state;
- implicit LAML gradient and Hessian agreement with profile finite
  differences, plus formula bootstrap and CUDA execution.

The complete
[`bccg_centile_curves`](../examples/bccg_centile_curves/README.md) case
additionally verifies a smooth location-scale-shape fit and nine fetal-growth
response-centile curves end to end against R.

The custom Torch distribution exposes both `sample()` and `icdf()`. The
inverse maps a truncated-normal probability through the Box-Cox response
transformation.
