# Power-exponential family (`PE`)

`PowerExponential`, also exported as `PE`, follows
`gamlss.dist::PE(mu, sigma, nu)`. It is a symmetric location-scale-shape
family on the whole real line. The shape parameter controls both heavier and
lighter tails than the Normal distribution:

- `nu < 2` gives heavier tails;
- `nu = 2` is exactly Normal;
- `nu > 2` gives lighter tails.

The default links are identity for `mu` and log for `sigma` and `nu`. In this
parameterization:

```text
E(Y)   = mu
Var(Y) = sigma^2
```

Thus `sigma` remains the standard deviation as `nu` changes. If

```text
c = sqrt(2^(-2/nu) Gamma(1/nu) / Gamma(3/nu)),
z = (y - mu) / sigma,
```

then the density is:

```text
f(y) = nu exp(-0.5 |z / c|^nu)
       / (2^(1 + 1/nu) c sigma Gamma(1/nu)).
```

## Example

```python
from torchgamlss import GAMLSS, PE

model = GAMLSS.from_formula(
    PE(),
    {
        "mu": "y ~ x",
        "sigma": "~ z",
        "nu": "~ w",
    },
    data,
)
fit = model.fit_rs_data(data, weights="weight")
```

`PowerExponential()` and `PE()` construct the same family. The implementation
provides a differentiable Torch density and CDF, seeded response sampling,
conditional quantiles, and the parameter-scale scores and expected
derivatives used by the R RS and CG algorithms.

## Verified parity

Committed fixtures generated with `gamlss.dist` 6.1-1 and `gamlss` 5.5-0
cover:

- `dPE(..., log=TRUE)`, `pPE()`, and `qPE()`;
- default links, starting values, scores, and all six expected
  second-derivative entries;
- weighted RS and CG fits with separate formulas and offsets for `mu`,
  `sigma`, and `nu`;
- full-Hessian inference, information criteria, quantile residuals, response
  quantiles, and seeded sampling;
- the exact `nu=2` Normal identity;
- CUDA FP32, FP16, and BF16 stress fitting across concentrated and
  heavy-tailed regimes.

The implementation shares the independently implemented differentiable
regularized incomplete-gamma core used by BCPE, without the Box-Cox
transformation or positive-response truncation.

The family definition follows Nelson (1991), *Conditional
Heteroskedasticity in Asset Returns: A New Approach*, Econometrica 59(2),
347–370, and the parameterization summarized by Rigby, Stasinopoulos, Heller,
and De Bastiani (2019), *Distributions for Modeling Location, Scale, and
Shape*, p. 374.
