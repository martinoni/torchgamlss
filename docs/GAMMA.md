# Gamma family (`GA`)

`Gamma` follows the `gamlss.dist::GA(mu, sigma)` parameterization. Both
parameters are strictly positive and use log links by default:

```text
E(Y)   = mu
Var(Y) = sigma^2 mu^2
shape  = 1 / sigma^2
rate   = 1 / (mu sigma^2).
```

Thus `sigma` is the coefficient of variation, not the scale parameter used by
some other Gamma APIs. Torch's `Gamma` distribution is constructed from the
equivalent shape and rate above.

## Example

```python
import torch
from torchgamlss import GAMLSS, Gamma

x = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
y = torch.exp(0.4 + 0.6 * x)
design = {
    "mu": torch.column_stack((torch.ones_like(x), x)),
    "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
}
model = GAMLSS(Gamma(), {"mu": 2, "sigma": 1})
result = model.fit_rs(y, design)
```

Custom `Link` instances can be supplied as `mu_link` and `sigma_link`. The
public `InverseLink` supports the reciprocal link accepted by the R family;
identity and log links are also available.

## Verified behavior

The committed R fixtures cover:

- log density and default links;
- parameter-scale scores and Fisher-scoring second derivatives;
- mean and variance;
- R-compatible starting values;
- a weighted, heteroscedastic RS fit with offsets for both parameters;
- the same likelihood fitted jointly with Torch L-BFGS;
- an additive RS fit with a fixed-lambda P-spline for `mu`.

The response must be finite and strictly positive. The implementation is
tested against `gamlss.dist` 6.1-1 and `gamlss` 5.5-0.
