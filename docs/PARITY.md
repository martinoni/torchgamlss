# R parity protocol

TorchGAMLSS treats the R implementations as executable references. A family is
not considered supported until its parameterization, links, log likelihood,
derivatives, and at least one fitted model have numerical parity tests.

## Normal family (`NO`)

The first reference slice targets `gamlss.dist` 6.1-1 and `gamlss` 5.5-0. In
the R `NO` parameterization, `mu` is the mean and `sigma` is the standard
deviation. Their default links are identity and log, respectively.

The committed fixtures cover:

- `dNO(..., log = TRUE)`;
- the default link functions;
- `dldm` and `dldd`, the parameter-scale score functions;
- `d2ldm2`, `d2ldd2`, and `d2ldmdd`, the expected second derivatives supplied
  to the GAMLSS fitting algorithm;
- a joint fit of `mu ~ x` and `sigma ~ 1` using `gamlss(..., family = NO())`.
- a weighted RS fit with parameter-specific offsets, `mu ~ x` and
  `sigma ~ z`, including coefficients, iteration count, log likelihood, and
  global deviance.
- an additive RS fit of `mu ~ pb(x, lambda=12)` and `sigma ~ 1`, including the
  `pb()` basis and penalty, parametric and smooth coefficients, fitted values,
  fixed smoothing parameter, effective degrees of freedom, global deviance,
  and outer iteration count.
- the same additive fit using `mu ~ pb(x)`, including the automatically
  ML-selected smoothing parameter and its resulting effective degrees of
  freedom.
- a target-EDF fit using `mu ~ pb(x, df=3)`. The R convention adds the two
  unpenalized dimensions, producing total EDF 5; the fixture covers the
  selected smoothing parameter, coefficients, fitted values, and deviance.
- local GAIC and GCV fits using `pb.control(method=..., k=2)`, including
  selected smoothing parameters, effective degrees of freedom, coefficients,
  fitted values, and deviances.

The expected sigma-sigma derivative supplied by `NO()` is Fisher-scoring
information, not the observation-wise second derivative of the normal log
density. The Python API names this distinction explicitly.

## Gamma family (`GA`)

The Gamma slice uses the `GA(mu, sigma)` parameterization in which `mu` is the
mean and `sigma` is the coefficient of variation. Its fixtures cover:

- `dGA(..., log = TRUE)` and both default log links;
- the parameter-scale scores and expected second derivatives supplied by
  `GA()`;
- mean `mu` and variance `sigma^2 mu^2`;
- a weighted RS fit of `mu ~ x + offset(mu_offset)` and
  `sigma ~ z + offset(sigma_offset)`;
- the corresponding joint Torch L-BFGS fit;
- an additive RS fit using `mu ~ pb(x, lambda=12) + offset(mu_offset)`.

The fitted-model fixtures compare coefficients, likelihood or deviance,
effective degrees of freedom, fitted distribution parameters, and P-spline
coefficients where applicable. Public response-scale prediction is exercised
against the fitted Gamma P-spline fixture.

The formula layer is tested against the same committed R fits: weighted
Normal RS with parameter offsets, Normal L-BFGS, and Gamma RS with a fixed
P-spline. This verifies that formula materialization reaches the same
coefficients and likelihood as the corresponding tensor inputs.

## Poisson, negative-binomial, and Beta families

The additional family slices target `PO(mu)`, `NBI(mu, sigma)`, and
`BE(mu, sigma)` respectively. Their fixtures cover:

- log density or mass, default links, and family starting expressions;
- parameter-scale scores and all expected second derivatives exposed by the R
  family objects;
- distribution means and variances;
- weighted RS fits with parameter-specific offsets;
- formula materialization and response-scale prediction.

For NBI, `sigma` is the quadratic overdispersion in
`Var(Y) = mu + sigma * mu^2`. For BE, `sigma` is the square root of the
variance ratio, so `Var(Y) = sigma^2 * mu * (1 - mu)`.

## Coefficient inference

Inference fixtures compare TorchGAMLSS with `vcov.gamlss(type="all")` and the
default `summary.gamlss(type="vcov")` calculations for weighted, offset models
from the `NO`, `PO`, `NBI`, and `BE` families. They cover:

- the full observed-Hessian covariance matrix, including cross-parameter
  entries;
- standard errors and coefficient statistics;
- residual degrees of freedom under frequency and case weights;
- two-sided Student t p-values;
- 95% Wald confidence intervals.

R obtains the Hessian numerically with `optimHess`; TorchGAMLSS obtains it
through Torch automatic differentiation. Parity is therefore defined to the
documented numerical tolerance rather than bitwise equality.

## Reproducing the fixtures

From the repository root:

```powershell
Rscript tools/install_r_dependencies.R
Rscript tools/generate_r_references.R
Rscript tools/generate_r_references.R --check
python -m pytest
```

The generator reads the small input datasets in `tests/reference/` and writes
the R results back to that directory. `--check` recomputes the results without
changing files and compares them with explicit tolerances.

## Sources

- `gamlss.dist` 6.1-1, `R/NO.r`, distributed by CRAN under GPL-2 or GPL-3:
  <https://cran.r-project.org/package=gamlss.dist>
- `gamlss` 5.5-0, distributed by CRAN under GPL-2 or GPL-3:
  <https://cran.r-project.org/package=gamlss>
- Rigby and Stasinopoulos (2005),
  <https://doi.org/10.1111/j.1467-9876.2005.00510.x>

The RS implementation follows the working-response and Fisher-weight updates
in `gamlss` 5.5-0, `R/gamlss-5.R`. Additive backfitting and P-splines
follow `R/add.r` and `R/pb.R`; the latter is also the reference for ML,
target-EDF, GAIC, and GCV smoothing selection. See [`RS.md`](RS.md) and
[`SMOOTHS.md`](SMOOTHS.md) for the equations and current scope.

The target-EDF fixture compares final numerical results rather than requiring
the same outer iteration count. R's finite-tolerance `uniroot()` updates can
produce a longer RS convergence path even when the final fit agrees.

The GAIC and GCV fixtures follow the same final-result convention.
TorchGAMLSS uses bounded Brent minimization on `log(lambda)`, while R uses
`nlminb()` on `lambda`; optimizer and outer-cycle counts may differ even when
the selected smoothing parameter and final fit agree.
