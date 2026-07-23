# Poisson, negative-binomial, and Beta families

The public classes follow named `gamlss.dist` parameterizations rather than
adopting similarly named conventions from other libraries.

## Poisson (`PO`)

`Poisson` has one positive parameter with a log link:

```text
E(Y) = Var(Y) = mu.
```

The response must contain finite non-negative integers. TorchGAMLSS constructs
`torch.distributions.Poisson(rate=mu)`.

```python
from torchgamlss import GAMLSS, Poisson

model = GAMLSS.from_formula(
    Poisson(),
    {"mu": "count ~ x + offset(exposure_log)"},
    data,
)
result = model.fit_rs_data(data, weights="weight")
```

## Negative binomial type I (`NBI`)

`NegativeBinomial` deliberately means `gamlss.dist::NBI(mu, sigma)`. Both
parameters are positive and use log links:

```text
E(Y)   = mu
Var(Y) = mu + sigma mu^2.
```

The equivalent Torch negative-binomial parameters are:

```text
total_count = 1 / sigma
logits      = log(mu sigma).
```

The response must contain finite non-negative integers. Naming the exact NBI
variant matters because `gamlss.dist` also contains other negative-binomial
parameterizations.

## Beta (`BE`)

`Beta` follows `gamlss.dist::BE(mu, sigma)`. Both parameters lie strictly
between zero and one and use logit links:

```text
E(Y)   = mu
Var(Y) = sigma^2 mu (1 - mu).
```

Torch's two concentrations are:

```text
precision = 1 / sigma^2 - 1
alpha     = mu precision
beta      = (1 - mu) precision.
```

The response must be finite and strictly between zero and one. Boundary values
zero and one require a different response family and are not silently moved
into the interior.

## Verified behavior

Committed fixtures generated with `gamlss.dist` and `gamlss` cover:

- log density or mass and default links;
- parameter-scale scores and expected second derivatives;
- family-specific default starting values;
- distribution means and variances;
- autograd after the link chain rule;
- response-support validation;
- weighted RS fits with parameter-specific offsets and formulas.

See [`PARITY.md`](PARITY.md) for package versions and regeneration commands.
