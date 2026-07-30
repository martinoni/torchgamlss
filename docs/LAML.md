# Whole-model LAML

TorchGAMLSS has a dense whole-model Laplace approximate marginal likelihood
implementation for additive Normal location-scale models. Formula models use:

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

Omitting the tensor lambdas requests LAML selection. `te()` and `ti()` each
contribute one free log lambda per marginal direction. A scalar `pb()` keeps
its existing fixed/automatic formula semantics. `fit_laml()` is the
corresponding tensor-level model method, while `fit_normal_laml()` remains the
low-level assembled-matrix API.

## Model and criterion

The model has separate predictors

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

## Formula and model integration

The high-level adapter:

1. concatenates every linear and smooth design within `mu` and `sigma`;
2. embeds every coefficient-space penalty in the complete model;
3. detects exact unidentifiable null-space directions and constrains them;
4. optimizes all requested log lambdas together;
5. writes coefficients and selected lambdas back to the `GAMLSS` object.

`NormalLAMLResult.smoothing_parameter_labels` identifies flat lambda
coordinates as `(parameter, term, penalty_index)`.
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
4. a bounded BFGS iteration updates all free log smoothing parameters jointly.

The outer gradient and Hessian currently use central differences of the fully
converged profile criterion. The implementation deliberately does not
differentiate through an arbitrary number of inner iterations.

`NormalLAMLResult` exposes:

- coefficients, predictors, fitted `mu`, and fitted `sigma`;
- lambdas and log lambdas;
- objective, log likelihood, and penalized negative log likelihood;
- outer gradient, Hessian, Hessian condition number, convergence, and boundary
  status;
- accepted-iterate history with inner convergence diagnostics;
- observed and penalized information in constrained coordinates;
- individual and combined penalty ranks and generalized log determinants;
- total EDF and the degrees of freedom removed by each penalty.
- model coefficient/lambda labels and slices when called through `GAMLSS`.

## `mgcv` reference gate

`tools/generate_mgcv_laml_reference.R` builds both a two-smooth Gaussian
location-scale model and a two-penalty tensor-product model with
`mgcv::gaulss(method="REML")`. It exports the exact model matrices and
coefficient-space penalties and checks the committed reference files.

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

The tests also compare every coefficient and every fitted location and scale,
the outer Hessian, penalty ranks, the joint cross-information block, tensor
penalty directions, and fixed-lambda results from the existing classical
`GAMLSS.fit_rs()` path.

Regenerate or verify the fixture with:

```powershell
Rscript tools/generate_mgcv_laml_reference.R
Rscript tools/generate_mgcv_laml_reference.R --check
```

## Current limits

- only the Normal location-scale likelihood is connected;
- outer derivatives are numerical profile derivatives;
- no sparse or discretized large-data backend is connected;
- smoothing-parameter uncertainty is reported through the outer Hessian but
  is not yet propagated into unconditional coefficient covariance;
- classical smooth bootstrap currently refits RS or CG, not LAML, so automatic
  tensor-lambda selection is not yet repeated inside bootstrap samples;
- `REML` is not exposed as a general alias. `LAML` remains the correct generic
  name outside models with a well-defined restricted-likelihood
  interpretation.

## Sources

- Wood, S. N., Pya, N., and Säfken, B. (2016),
  [Smoothing Parameter and Model Selection for General Smooth
  Models](https://doi.org/10.1080/01621459.2016.1180986).
- [`mgcv::gaulss`
  documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gaulss.html).
