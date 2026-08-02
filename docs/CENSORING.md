# Survival families and censored responses

TorchGAMLSS provides four event-time families:

| R family | TorchGAMLSS | Parameters | Default links |
| --- | --- | --- | --- |
| `WEI()` | `Weibull()` / `WEI()` | scale `mu`, shape `sigma` | log, log |
| `LOGNO()` | `LogNormal()` / `LOGNO()` | mean-log `mu`, sd-log `sigma` | identity, log |
| `IG()` | `InverseGaussian()` / `IG()` | mean `mu`, dispersion `sigma` | log, log |
| `GG()` | `GeneralizedGamma()` / `GG()` | median-like `mu`, scale `sigma`, shape `nu` | log, log, identity |

The parameterizations, likelihood scores, working second derivatives, moments,
quantiles, and starting values follow `gamlss.dist`. `LogNormal.mu` may be any
finite real number, as required by the log-normal mean-log parameterization.
For `IG`, `E(Y) = mu` and `Var(Y) = sigma² mu³`. The `GG` implementation uses
the Lopatatzidis-Green parameterization and approaches a log-normal
distribution with median `mu` as `nu` approaches zero. Stable series avoid
the cancellation present in a literal evaluation around that limit.

Whole-model LAML is available for uncensored GG and LOGNO models with their
standard links. The censoring wrappers currently remain on their existing RS,
CG, L-BFGS, and mini-batch routes; LAML should not yet be selected for
`CensoredFamily(GG(), ...)` or `CensoredFamily(LOGNO(), ...)`.

## Response representation

`CensoredResponse` stores fixed censoring metadata alongside the numeric
response used by formulas and fitting:

```python
import torch

from torchgamlss import CensoredResponse

time = torch.tensor([0.8, 1.5, 3.0, 5.0], dtype=torch.float64)
event = torch.tensor([1, 0, 1, 0])
response = CensoredResponse.right(time, event)
```

The status codes match the interval representation used by
`survival::Surv` and `gamlss.cens`:

| Code | Meaning | Likelihood contribution |
| --- | --- | --- |
| `0` | right censored at `t` | `log S(t)` |
| `1` | event observed exactly at `t` | `log f(t)` |
| `2` | left censored at `t` | `log F(t)` |
| `3` | event in `(lower, upper]` | `log(F(upper) - F(lower))` |

For ordinary left censoring, use `CensoredResponse.left(time, event)`. For a
dataset containing only interval-censored rows:

```python
lower = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
upper = torch.tensor([0.8, 1.6, 3.5], dtype=torch.float64)
response = CensoredResponse.interval(lower, upper)
```

Mixed data can be represented directly. `upper` is inspected only for rows
whose status is `3`, but it must be a finite vector aligned with all rows:

```python
from torchgamlss import CensoredResponse, Censoring

observed = torch.tensor([0.7, 1.2, 1.0, 2.5], dtype=torch.float64)
upper = torch.tensor([0.7, 1.2, 1.0, 3.8], dtype=torch.float64)
status = torch.tensor(
    [
        Censoring.EXACT,
        Censoring.RIGHT,
        Censoring.LEFT,
        Censoring.INTERVAL,
    ]
)
response = CensoredResponse(observed, status, upper)
```

The stored tensors are detached copies and are moved to the likelihood device
when necessary. Interval upper endpoints must be finite and strictly greater
than their lower endpoints.

## Fitting

Compose the response metadata with a continuous event-time family:

```python
import pandas as pd

from torchgamlss import CensoredFamily, GAMLSS, Weibull

family = CensoredFamily(Weibull(), response)
data = pd.DataFrame(
    {
        "time": response.observed.cpu().numpy(),
        "x": [0.2, 0.5, 0.8, 1.1],
    }
)
model = GAMLSS.from_formula(
    family,
    {
        "mu": "time ~ x",
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_rs_data(data)
```

RS and CG use censored likelihood scores while retaining the base family's
working expected second derivatives, matching `gamlss.cens`. Torch L-BFGS
differentiates the complete censored likelihood directly. The response passed
to fitting must exactly equal `response.observed`, which prevents accidental
misalignment of status and time rows.

Censoring metadata are currently fixed at the full-dataset level. Full-batch
RS, CG, and L-BFGS fitting are supported; mini-batch and streaming fitting are
not, because their row batches do not yet carry corresponding status and
upper-endpoint tensors. Continuous base families are supported; discrete
censoring conventions remain outside the current API.

## Survival prediction

The family-level methods evaluate the event-time distribution rather than the
observation mechanism:

```python
parameters = {"mu": mu, "sigma": sigma, "nu": nu}  # omit nu for 2-parameter families
survival = family.survival(times, parameters)
hazard = family.hazard(times, parameters)
cumulative_hazard = family.cumulative_hazard(times, parameters)
```

After fitting a formula model, one call predicts all three functions on a
shared time grid:

```python
curves = model.predict_survival_data(
    new_data,
    times=[0.5, 1.0, 2.0, 4.0],
)

curves.times
curves.survival
curves.hazard
curves.cumulative_hazard
```

Each function tensor has shape `(observations, times)`. Formula-free models
use `predict_survival()` with design matrices and the same optional offsets,
smooth covariates, neural inputs, and shared input accepted by `predict()`.
The computations preserve the model dtype and device and run on CUDA.

Sampling and quantiles from a `CensoredFamily` likewise refer to latent event
times. Censoring changes how observed rows contribute to fitting; it does not
define a new event-time distribution.

## R parity

Committed references generated by
`tools/generate_censored_references.R` compare:

- `WEI`, `LOGNO`, `IG`, and `GG` density, CDF, survival, hazard, cumulative hazard,
  quantiles, moments, scores, working derivatives, and starting values;
- exact, right-, left-, and mixed interval likelihood contributions;
- numerical censored scores returned by `gamlss.cens` 5.0-7;
- linked-parameter autograd, RS/L-BFGS fitting, prediction, and CUDA behavior
  on the Python side.

Run `Rscript tools/generate_censored_references.R --check` to recompute and
validate the committed artifacts.

The manifest-driven
[`generalized_gamma_survival`](../examples/generalized_gamma_survival/README.md)
example fits a three-parameter right-censored regression in both languages and
compares coefficients, fitted parameters, latent quantiles, and survival,
hazard, and cumulative-hazard curves.
