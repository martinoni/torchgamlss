# Response families

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

## Response simulation

Every public family implements `family.sample(parameters, generator=...)`.
A seeded `torch.Generator` makes a draw reproducible without changing Torch's
global random-number-generator state. Normal, Gamma, Beta, Poisson, and NBI
delegate to built-in Torch distributions. BCCG, BCT, and BCPE sample their
truncated latent normal, Student-t, and power-exponential representations,
respectively. TF and PE sample custom untruncated location-scale
representations.

This interface is used by the fixed-design parametric smooth bootstrap
described in [`INFERENCE.md`](INFERENCE.md).

## Conditional quantiles

Every public family implements
`family.quantile(probabilities, parameters)`. Parameter tensors are
broadcast, and the returned final dimension follows the probability vector.
Quantiles are numerically matched to `qNO`, `qGA`, `qPO`, `qNBI`, `qBE`,
`qBCCG`, `qBCT`, `qBCPE`, `qTF`, and `qPE`.

For Poisson and NBI, the result uses the usual left-continuous discrete
definition: the smallest non-negative integer whose CDF is at least the
requested probability. Continuous-family quantiles invert the corresponding
CDF on the response support.

The public model methods `predict_quantiles*()` and `predict_centiles*()` use
this interface after combining every fitted distribution parameter. SciPy
inverse distribution functions are used where Torch lacks a reliable
`icdf`; returned tensors preserve the model dtype and device, but quantile
inversion is not currently an autograd path for all families.

## Verified behavior

Committed fixtures generated with `gamlss.dist` and `gamlss` cover:

- log density or mass and default links;
- parameter-scale scores and expected second derivatives;
- family-specific default starting values;
- distribution means and variances;
- autograd after the link chain rule;
- response-support validation;
- seeded response simulation, including probability-integral-transform checks
  for BCCG, BCT, BCPE, TF, and PE;
- response quantiles at seven probabilities for all ten families;
- weighted RS fits with parameter-specific offsets and formulas;
- CUDA FP16/BF16 mini-batch stress fits for GA, NBI, BCCG, BCT, BCPE, TF,
  and PE across central and extreme response quantiles.

See [`PARITY.md`](PARITY.md) for package versions and regeneration commands.
The three-parameter Box-Cox Cole-Green family is documented separately in
[`BCCG.md`](BCCG.md), and the four-parameter Box-Cox t family in
[`BCT.md`](BCT.md). The four-parameter Box-Cox power-exponential family is in
[`BCPE.md`](BCPE.md), and the Student-t location-scale family is in
[`TF.md`](TF.md). The symmetric power-exponential family is in
[`PE.md`](PE.md).
