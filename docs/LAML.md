# Whole-model LAML

TorchGAMLSS has a dense whole-model Laplace approximate marginal likelihood
implementation for additive Normal location-scale, Poisson log-mean, Gamma
mean/coefficient-of-variation, NBI mean/dispersion, and Beta mean/dispersion
models. A Normal
formula model uses:

```python
from torchgamlss import GAMLSS, LAMLControl, Normal

model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ te(x, z, intervals=(6, 4), name='surface')",
        "sigma": "~ pb(w, intervals=6)",
    },
    data,
)
fit = model.fit_laml_data(data, control=LAMLControl())
```

A Gamma model can smooth both its mean and coefficient of variation:

```python
from torchgamlss import GAMLSS, Gamma, LAMLControl

model = GAMLSS.from_formula(
    Gamma(),
    {
        "mu": "y ~ pb(x, intervals=8)",
        "sigma": "~ pb(z, intervals=6)",
    },
    data,
)
fit = model.fit_laml_data(data, control=LAMLControl())
```

Omitting the tensor lambdas requests LAML selection. `te()` and `ti()` each
contribute one free log lambda per marginal direction. A scalar `pb()` keeps
its existing fixed/automatic formula semantics. `fit_laml()` is the
corresponding tensor-level model method, while `fit_normal_laml()` remains the
specialized low-level Normal API. `fit_gamlss_laml()` is the family-driven
low-level API used by every non-Normal path.

A Poisson model uses the same formula and fitting surface:

```python
from torchgamlss import GAMLSS, LAMLControl, Poisson

model = GAMLSS.from_formula(
    Poisson(),
    {"mu": "count ~ pb(x, intervals=8)"},
    data,
)
fit = model.fit_laml_data(data, control=LAMLControl())
```

A negative-binomial type I model jointly estimates the GAMLSS dispersion
predictor as well as its smooth mean:

```python
from torchgamlss import GAMLSS, LAMLControl, NegativeBinomial

model = GAMLSS.from_formula(
    NegativeBinomial(),
    {
        "mu": "count ~ pb(x, intervals=8)",
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data, control=LAMLControl())
```

A Beta model uses the `gamlss.dist::BE` logit links for both its mean and
dispersion parameter:

```python
from torchgamlss import Beta, GAMLSS, LAMLControl

model = GAMLSS.from_formula(
    Beta(),
    {
        "mu": "proportion ~ pb(x, intervals=8)",
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data, control=LAMLControl())
```

The same estimator can be repeated inside fixed-design parametric bootstrap
samples:

```python
bootstrap = model.smooth_joint_bootstrap_data(
    data,
    new_data=new_data,
    replicates=999,
    algorithm="laml",
    control=LAMLControl(),
    generator=torch.Generator().manual_seed(2026),
)
```

Every successful replicate jointly reselects all automatic scalar and tensor
lambdas and stores one smoothing-parameter column per penalty. The same
`algorithm="laml"` option is accepted by `quantile_bootstrap_data()` and
`centile_bootstrap_data()`.

`LAMLControl.inner_relaxed_gradient_multiplier` controls a guarded
near-stationary fallback used only when the inner Newton line search stalls or
reaches its iteration limit. The primary gradient tolerance remains
`inner_gradient_tolerance`; the relaxed multiplier defaults to 50 to absorb
small BLAS/platform differences in otherwise converged dense profiles.

`LAMLControl.outer_derivative_method="implicit"` is the default. It uses the
implicit function theorem after the inner fit has converged for both the
outer gradient and Hessian.
`outer_derivative_method="finite_difference"` retains the original
objective-difference implementation as a diagnostic fallback.

## Model and criterion

The Normal model has separate predictors

```text
mu_i = X_mu[i] beta_mu + offset_mu[i]
sigma_i = b + exp(X_sigma[i] beta_sigma + offset_sigma[i])
```

where `b >= 0` is `sigma_floor`. `b=0` matches the current `Normal` log link;
`b=0.01` matches the default response-scale relationship in `mgcv::gaulss`.
The complete coefficient vector is ordered as
`beta = [beta_mu, beta_sigma]`.

For penalty components `S_j` and `rho_j = log(lambda_j)`,

```text
S_lambda = sum_j exp(rho_j) S_j
```

and the inner fit minimizes

```text
L_p(beta) = -log f(y | beta) + beta' S_lambda beta / 2.
```

At the converged inner estimate, the negative LAML criterion is

```text
V(rho) =
    L_p(beta_hat)
    - log|S_lambda|_+ / 2
    + log|H_p| / 2
    - M_p log(2 pi) / 2,
```

where `H_p` is the joint observed Hessian of `L_p`, `|S_lambda|_+` is
the generalized determinant over positive eigenvalues, and `M_p` is the
dimension of the unpenalized coefficient subspace. This is the criterion in
Wood, Pya, and Säfken (2016), expressed as a minimization objective.

For Poisson, `eta_mu = X_mu beta_mu + offset_mu`,
`mu = exp(eta_mu)`, and the inner likelihood is evaluated through
`Poisson.log_prob()`. The generic path obtains links, starting parameters, and
the differentiable observation-wise likelihood from the public `Family`
interface; its smoothing optimizer is shared with the exact Normal path.

For Gamma, TorchGAMLSS follows `gamlss.dist::GA`: `sigma` is the coefficient
of variation and `Var(Y) = sigma^2 mu^2`. The overlapping `mgcv::gammals`
scale is `phi = sigma^2`, so its predictor obeys
`eta_sigma = eta_phi / 2`. The committed reference uses identity links on
`mgcv`'s internal log-mean and log-scale parameters, making this
reparameterization explicit rather than relying on `gammals`' default lower
bound transform.

For NBI, TorchGAMLSS follows `gamlss.dist::NBI`, with
`Var(Y) = mu + sigma mu^2`. The overlapping `mgcv::nb` parameter is
`theta = 1/sigma`. The direct reference fixes `theta` in `mgcv` and fixes the
equivalent Torch log-`sigma` predictor by offset. A regular TorchGAMLSS formula
fit estimates the `sigma` predictor jointly, which is broader than that
conditional `mgcv` comparison.

For Beta, TorchGAMLSS follows `gamlss.dist::BE`, with
`Var(Y) = sigma^2 mu(1-mu)`. The overlapping `mgcv::betar` precision is
`phi = 1/sigma^2 - 1`. The direct reference fixes `phi` in `mgcv` and fixes
the corresponding Torch `sigma` predictor by offset, so both systems optimize
the same mean-smooth LAML problem. A regular formula Beta fit estimates the
`sigma` predictor jointly and is therefore a broader GAMLSS model.

## Formula and model integration

The high-level adapter:

1. concatenates every linear and smooth design within each family parameter;
2. embeds every coefficient-space penalty in the complete model;
3. detects exact unidentifiable null-space directions and constrains them;
4. optimizes all requested log lambdas together;
5. writes coefficients and selected lambdas back to the `GAMLSS` object.

`NormalLAMLResult.smoothing_parameter_labels` and
`GAMLSSLAMLResult.smoothing_parameter_labels` identify flat lambda coordinates
as `(parameter, term, penalty_index)`.
`smoothing_parameter_slices`, `linear_coefficient_slices`, and
`smooth_coefficient_slices` map the complete result back to model terms.
Prediction therefore uses the selected state immediately:

```python
fit.smoothing_parameter_labels
fit.smoothing_parameters
fit.outer_gradient
fit.outer_hessian
fitted_parameters = model.predict_data(data)
```

Use `initial_smoothing_parameters=(4, 7)` or its
`initial_lambda_=(4, 7)` alias to change the tensor starting point while
retaining automatic selection. Supplying `lambda_=(4, 7)` instead fixes both
lambdas. RS, CG, L-BFGS, and mini-batch fitting continue to require fixed
tensor lambdas; automatic tensor terms explicitly direct callers to LAML.

## Low-level matrix API

Every penalty supplied to `fit_normal_laml()` acts on the complete coefficient
vector. Current `PSpline` objects expose the required coefficient-space
matrices:

```python
import torch

from torchgamlss import PSpline, fit_normal_laml

x = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
z = torch.cos(torch.linspace(0.0, 6.28, 200, dtype=torch.float64))
generator = torch.Generator().manual_seed(2026)
true_sigma = torch.exp(-1.4 + 0.3 * z + 0.6 * z.square())
y = (
    torch.sin(3.0 * x)
    + true_sigma
    * torch.randn(x.numel(), dtype=x.dtype, generator=generator)
)

mu_term = PSpline.from_data(
    x,
    initial_smoothing_parameter=10.0,
    intervals=6,
)
sigma_term = PSpline.from_data(
    z,
    initial_smoothing_parameter=10.0,
    intervals=6,
)
X_mu = mu_term.design(x)
X_sigma = sigma_term.design(z)

p_mu = X_mu.shape[1]
p_sigma = X_sigma.shape[1]
S_mu = torch.zeros((p_mu + p_sigma, p_mu + p_sigma), dtype=y.dtype)
S_sigma = torch.zeros_like(S_mu)
S_mu[:p_mu, :p_mu] = mu_term.penalty_matrices()[0]
S_sigma[p_mu:, p_mu:] = sigma_term.penalty_matrices()[0]

fit = fit_normal_laml(
    y,
    X_mu,
    X_sigma,
    (S_mu, S_sigma),
    (10.0, 10.0),
)

print(fit.smoothing_parameters)
print(fit.effective_degrees_of_freedom)
print(fit.outer_converged)
```

Parametric columns and smooth bases can instead be concatenated in each design.
If this creates redundant unpenalized columns, pass `constraints=C` for
`C beta = 0`. Constraints are imposed by SVD null-space reparameterization.
Inputs already centered and constrained by another system, such as the `mgcv`
reference design, need no additional constraint.

For a supported generic family, supply one design per family parameter:

```python
from torchgamlss import Poisson, fit_gamlss_laml

fit = fit_gamlss_laml(
    Poisson(),
    counts,
    {"mu": X_mu},
    (S_mu,),
    (10.0,),
)

fit.parameter_coefficients["mu"]
fit.linear_predictors["mu"]
fit.fitted_parameters["mu"]
```

To hold one family parameter fixed in the low-level API, give it an `n x 0`
design and its fixed link-scale value as an offset. At least one other family
parameter must retain coefficient columns. For example, the Beta and NBI
references hold `sigma` fixed while smoothing `mu`:

```python
from torchgamlss import Beta

sigma_design = response.new_empty((response.numel(), 0))
sigma_offset = response.new_full(response.shape, fixed_sigma.logit().item())

fit = fit_gamlss_laml(
    Beta(),
    response,
    {"mu": X_mu, "sigma": sigma_design},
    (S_mu,),
    (10.0,),
    offsets={"sigma": sigma_offset},
)
```

Set `estimate_smoothing=False` to evaluate a fixed-lambda fit, or pass one
boolean per penalty to mix fixed and estimated components:

```python
fit = fit_normal_laml(
    y,
    X_mu,
    X_sigma,
    (S_mu, S_sigma),
    (10.0, 4.0),
    estimate_smoothing=(True, False),
)
```

## Numerical algorithm and diagnostics

For every accepted outer iterate:

1. a safeguarded Newton iteration converges the complete penalized coefficient
   vector;
2. Torch autograd forms the exact joint observed likelihood Hessian, including
   cross-information between `mu` and `sigma`;
3. the criterion uses eigendecomposition-based generalized log determinants;
4. implicit differentiation supplies the outer LAML gradient and Hessian;
5. a bounded BFGS iteration updates all free log smoothing parameters jointly.

For `rho_j = log(lambda_j)`, differentiating the converged penalized score
equation gives the coefficient sensitivity

```text
d beta_hat / d rho_j =
    -H_p^-1 (lambda_j S_j beta_hat).
```

This avoids differentiating through an arbitrary number of inner Newton
iterations. Torch autograd supplies the third-order likelihood contraction
needed by the `log|H_p|` derivative, while the penalty determinant derivative
uses the generalized inverse on the penalized subspace.

The Hessian differentiates the same converged score equation a second time.
For two log smoothing parameters `rho_j` and `rho_k`, it solves for
`d^2 beta_hat / (d rho_j d rho_k)` using `H_p` and the required third-order
likelihood contraction. Torch autograd then supplies the exact partial
Hessian of the LAML criterion, including the fourth-order derivatives induced
by `log|H_p|`, and the implementation combines those partials with the first
and second coefficient sensitivities by the chain rule. No displaced inner
fits or finite-difference step size enter the default Hessian.

The legacy fallback can be selected explicitly:

```python
control = LAMLControl(
    outer_derivative_method="finite_difference",
)
```

On the two-lambda Gamma reference fit, the implicit route reaches the same
objective and lambdas with 8 unique profile evaluations instead of 48 at full
convergence. In the one-iteration derivative audit it uses 2 instead of 18.

`NormalLAMLResult` and `GAMLSSLAMLResult` expose the common optimization,
penalty, and smoothing diagnostics. The generic result additionally exposes
parameter-keyed coefficient, predictor, and fitted-parameter mappings:

- coefficients, predictors, fitted `mu`, and fitted `sigma`;
- lambdas and log lambdas;
- objective, log likelihood, and penalized negative log likelihood;
- outer gradient, Hessian, Hessian condition number, convergence, and boundary
  status;
- `outer_derivative_method` and the number of unique `profile_evaluations`;
- accepted-iterate history with inner convergence diagnostics;
- observed and penalized information in constrained coordinates;
- individual and combined penalty ranks and generalized log determinants;
- total EDF and the degrees of freedom removed by each penalty.
- model coefficient/lambda labels and slices when called through `GAMLSS`.

## `mgcv` reference gate

`tools/generate_mgcv_laml_reference.R` builds a two-smooth Gaussian
location-scale model and a two-penalty tensor-product model with
`mgcv::gaulss(method="REML")`, a Poisson log-mean model, and a Gamma
location-scale model. The latter two use
`mgcv::gam(..., method="REML")` with `poisson()` and `gammals()`,
respectively. It also fits Beta and NBI mean smooths with fixed family
parameters through `mgcv::betar(theta=...)` and `mgcv::nb(theta=...)`. The
generator exports the exact model matrices and coefficient-space penalties
and checks the committed reference files.

For the current fixture:

| Quantity | TorchGAMLSS | `mgcv` |
|---|---:|---:|
| negative LAML | 71.4098279372 | 71.4098279374 |
| lambda for `mu` | 5.42576296 | 5.42576630 |
| lambda for `sigma` | 59.52636558 | 59.52635840 |
| total EDF | 9.66096783 | 9.66096760 |

For the tensor fixture:

| Quantity | TorchGAMLSS | `mgcv` |
|---|---:|---:|
| negative LAML | 33.78215649 | 33.78215647 |
| lambda in the `x` direction | 0.69084 | 0.69083 |
| lambda in the `z` direction | 54.0102 | 54.0103 |
| total EDF | 19.127866 | 19.127862 |

For the Poisson fixture:

| Quantity | TorchGAMLSS | `mgcv` |
|---|---:|---:|
| negative LAML | 232.2074779663 | 232.2074779638 |
| lambda for `mu` | 9.09069376 | 9.09059115 |
| total EDF | 4.90660697 | 4.90661362 |

For the Gamma fixture:

| Quantity | TorchGAMLSS | `mgcv` |
|---|---:|---:|
| negative LAML | 246.1128227180 | 246.1128227179 |
| lambda for `mu` | 13.84549454 | 13.84547277 |
| lambda for `phi = sigma^2` | 437.1846991 | 437.1872877 |
| total EDF | 8.20690616 | 8.20690518 |

For the conditional Beta fixture:

| Quantity | TorchGAMLSS | `mgcv` |
|---|---:|---:|
| negative LAML | -140.4336887830 | -140.4336887832 |
| lambda for `mu` | 6.30659455 | 6.30659732 |
| fixed `phi = 1/sigma^2 - 1` | 12 | 12 |
| outer Hessian | 1.79952854 | 1.79952920 |

The fitted Beta coefficients agree within `1.1e-7`. `mgcv` reports an EDF
about `0.004` lower because its extended-family EDF convention differs
slightly even with fixed `phi`; the objective, likelihood, lambda, predictor,
fitted mean, and outer Hessian are the strict comparison targets.

For the conditional NBI fixture:

| Quantity | TorchGAMLSS | `mgcv` |
|---|---:|---:|
| negative LAML | 345.5059784435 | 345.5059784431 |
| lambda for `mu` | 11.64729016 | 11.64732199 |
| fixed `theta = 1/sigma` | 4 | 4 |
| outer Hessian | 1.17736244 | 1.17736368 |

The NBI coefficients, link predictor, and fitted mean agree numerically. Its
roughly `0.0045` EDF difference is likewise retained as an `mgcv`
extended-family convention difference; the criterion, likelihood, lambda,
fitted quantities, and analytic outer Hessian are the strict targets.

The tests also compare every coefficient and every fitted location and scale,
the Poisson link and fitted mean, the Gamma mean/CV predictors and parameters,
the outer Hessian, penalty ranks, the joint cross-information block, tensor
penalty directions, and fixed-lambda results from the existing classical
`GAMLSS.fit_rs()` path.

Regenerate or verify the fixture with:

```powershell
Rscript tools/generate_mgcv_laml_reference.R
Rscript tools/generate_mgcv_laml_reference.R --check
```

## Current limits

- whole-model integration currently accepts standard Normal identity/log,
  Poisson log-link, NBI and Gamma log/log, and Beta logit/logit models; the
  low-level family-driven core is deliberately exposed, but other families
  are not yet claimed as validated;
- the dense exact Hessian uses likelihood derivatives through fourth order;
  it avoids repeated inner fits and finite-difference noise, but its local
  autograd work can be expensive for models with many coefficients or free
  smoothing parameters;
- no sparse or discretized large-data backend is connected;
- smoothing-parameter uncertainty is reported through the outer Hessian but
  is not yet propagated into unconditional coefficient covariance;
- parametric LAML bootstrap is simulation-based and repeats the complete
  nested optimization, so it can be substantially slower and experience more
  failed refits than RS or CG;
- `REML` is not exposed as a general alias. `LAML` remains the correct generic
  name outside models with a well-defined restricted-likelihood
  interpretation.

## Sources

- Wood, S. N., Pya, N., and Säfken, B. (2016),
  [Smoothing Parameter and Model Selection for General Smooth
  Models](https://doi.org/10.1080/01621459.2016.1180986).
- [`mgcv::gaulss`
  documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gaulss.html).
- [`mgcv::gam`
  documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.html).
- [`mgcv::gammals`
  documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gammals.html).
- [`mgcv::gam.fit3`
  documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.fit3.html).
