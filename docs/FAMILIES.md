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

## Event-time families and censoring

`Weibull` (`WEI`) follows `gamlss.dist::WEI`: `mu` is its positive scale and
`sigma` its positive shape, both with log links. `LogNormal` (`LOGNO`) follows
`gamlss.dist::LOGNO`: `mu` is the unrestricted mean of `log(Y)` with an
identity link, and `sigma` is its positive standard deviation with a log link.
`InverseGaussian` (`IG`) uses positive mean `mu` and dispersion `sigma`, both
with log links, so `Var(Y) = sigma² mu³`. `GeneralizedGamma` (`GG`) uses
positive `mu` and `sigma` with log links plus unrestricted shape `nu` with an
identity link. Its `nu=0` limit is log-normal with median `mu`. All four
families require strictly positive responses.

Continuous event-time families expose `survival()`, `hazard()`, and
`cumulative_hazard()` alongside density, CDF, quantile, sampling, mean, and
variance. `CensoredFamily` composes one with a fixed `CensoredResponse`
containing exact, right-, left-, or interval-censored rows compatible with
`survival::Surv` status codes. See [`CENSORING.md`](CENSORING.md) for the
representation, fitting, prediction, CUDA behavior, and current full-batch
restriction.

## Scalar- and observation-bound truncation (`gamlss.tr`)

`TruncatedFamily` composes an existing family with one or two fixed bounds.
Each bound may be a scalar shared by every observation or a one-dimensional
tensor containing one value per observation. The family preserves the base
parameter names, links, and predictors:

```python
import torch

from torchgamlss import Normal, Poisson, TruncatedFamily

positive_normal = TruncatedFamily(Normal(), lower=0.0)
bounded_counts = TruncatedFamily(Poisson(), lower=0, upper=8)

lower = torch.tensor([-1.0, -0.5, 0.0])
upper = torch.tensor([1.0, 1.5, 2.0])
varying_normal = TruncatedFamily(Normal(), lower=lower, upper=upper)
```

Continuous bounds are closed. Discrete bounds follow `gamlss.tr`'s open-bound
convention, so the second example supports integer counts `1, ..., 7`.
Use `None` for an absent side:

| R `gamlss.tr` | TorchGAMLSS |
| --- | --- |
| `trun(par=0, family="NO", type="left")` | `TruncatedFamily(Normal(), lower=0)` |
| `trun(par=2, family="NO", type="right")` | `TruncatedFamily(Normal(), upper=2)` |
| `trun(par=c(-1, 2), family="NO", type="both")` | `TruncatedFamily(Normal(), lower=-1, upper=2)` |

The log likelihood, CDF, quantiles, sampling, and parameter scores include the
truncation normalizer. Classical RS/CG working second derivatives are inherited
from the base family, matching `gamlss.tr`. Truncation is verified against R
for all ten pre-survival families and their scalar and observation-specific
definitions. Normal and Poisson cover left, right, and two-sided cases; Gamma,
NBI, Beta, BCCG, TF, PE, BCT, and BCPE additionally verify scalar and varying
two-sided cases. Their differentiable likelihoods run fully on CUDA.

Observation-specific tensors are fixed data, never trainable parameters. They
must be finite, have the same length as the response or prediction rows, and,
for discrete families, contain integer-valued endpoints. TorchGAMLSS moves
them to the parameter dtype and device when evaluating the family. Full-batch
RS, CG, and L-BFGS fitting, formula-data methods, quantiles, and sampling are
supported. The current mini-batch and streaming loaders require scalar bounds
because they do not transport bound rows with each batch. For a new prediction
dataset, construct a family with bounds aligned to those prediction rows.

## Finite mixtures (`gamlss.mx`)

`FiniteMixture` (alias `MX`) composes two or more scalar-event families with
reference-category mixing predictors. Component parameters are exposed as
`component_1_mu`, `component_2_mu`, and so on; selected parameters can instead
share one predictor across every component. The family implements stable
log-sum-exp likelihoods, CDFs, continuous quantiles, sampling, moments,
posterior probabilities, deterministic ordered starts, and separation
diagnostics.

`model.fit_mixture*()` fits linear or fixed-smooth mixture predictors by
generalized EM with Torch L-BFGS M-steps. Normal, Gamma, and Poisson
distribution operations plus an intercept-only Normal fit have committed
`gamlss.mx` parity fixtures. See [`MIXTURES.md`](MIXTURES.md) for parameter
names, sharing, fitting, label ordering, diagnostics, CUDA behavior, and
current restrictions.

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

Every public base and truncated family implements
`family.quantile(probabilities, parameters)`. Parameter tensors are
broadcast, and the returned final dimension follows the probability vector.
Quantiles are numerically matched to `qNO`, `qGA`, `qPO`, `qNBI`, `qBE`,
`qBCCG`, `qBCT`, `qBCPE`, `qTF`, `qPE`, `qWEI`, `qLOGNO`, `qIG`, and `qGG`.
Truncated quantiles map
probabilities into each observation's retained base-family probability
interval before inversion. Continuous finite mixtures numerically invert the
weighted CDF; discrete-mixture quantiles are not yet implemented.

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
- response quantiles at seven probabilities for the first ten families, plus
  event-time quantiles for WEI, LOGNO, IG, and GG;
- WEI, LOGNO, IG, and GG survival functions, hazards, cumulative hazards, moments,
  linked autograd, and `gamlss.dist` scores/working derivatives;
- exact, right-, left-, and interval-censored WEI, LOGNO, IG, and GG likelihoods and
  scores against `gamlss.cens`;
- RS/L-BFGS fitting for IG and GG, CUDA gradients and sampling, and a complete
  right-censored GG regression parity example;
- scalar- and observation-bound truncation across all ten pre-survival families, including
  density/mass, CDF, quantile, score, sampling, autograd, and CUDA behavior;
- truncated Normal RS/L-BFGS and formula-data fitting plus truncated Poisson
  RS/L-BFGS behavior;
- Normal, Gamma, and Poisson finite-mixture density/CDF/posterior/moment
  parity, plus a complete two-Normal `gamlssMX()` fit;
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
