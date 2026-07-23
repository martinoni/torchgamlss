# TorchGAMLSS

TorchGAMLSS is an early-stage PyTorch implementation of Generalized Additive
Models for Location, Scale and Shape (GAMLSS).

The project aims to combine numerical compatibility with the R packages
[`gamlss`](https://github.com/gamlss-dev/gamlss) and `gamlss.dist` with
PyTorch automatic differentiation, composable predictors, and optional GPU
execution.

> [!WARNING]
> TorchGAMLSS is pre-alpha software. It is not yet suitable for statistical
> analysis or production use.

## Initial design

Every parameter of a response distribution has its own predictor and link:

```text
eta_k = X_k beta_k
theta_k = inverse_link_k(eta_k)
Y | X ~ D(theta_1, ..., theta_K)
```

The current implementation contains:

- identity, inverse, log, and logit links;
- Normal (`NO`), Gamma (`GA`), Poisson (`PO`), negative-binomial type I
  (`NBI`), Beta (`BE`), three-parameter Box-Cox Cole-Green (`BCCG`), and
  four-parameter Box-Cox t (`BCT`) and Box-Cox power-exponential (`BCPE`)
  families;
- a differentiable negative log-likelihood;
- full-batch joint fitting with Torch L-BFGS;
- likelihood weights and offsets for every distribution parameter;
- R-compatible family defaults and user-supplied parameter-scale starting
  values for RS and CG fitting;
- Rigby-Stasinopoulos fitting cycles for linear and additive predictors;
- Cole-Green joint fitting with cross-parameter working weights for
  parametric predictors;
- fixed-lambda P-splines, penalized weighted least squares, additive
  backfitting, and effective degrees of freedom;
- automatic P-spline smoothing-parameter selection with the `pb()` ML update;
- target-EDF P-splines compatible with `pb(x, df=...)`;
- local GAIC and GCV P-spline smoothing-parameter selection compatible with
  `pb.control(method=...)`;
- prediction on response, link, and additive-term scales, including new data;
- joint full-Hessian covariance, standard errors, Wald tests, and confidence
  intervals for parametric models;
- global deviance, AIC, AICc, GAIC, SBC/BIC, and model-comparison weights;
- continuous and randomized discrete normal quantile residuals;
- Wilkinson formulas for tabular fitting and prediction, including categorical
  variables, `offset()`, and `pb()`;
- independent design matrices for each distribution parameter;
- R-generated parity fixtures for all supported families' densities or masses,
  links, derivatives, starting values, and fitted models;
- tests for parameter recovery, gradients, link round trips, and R parity.

CG backfitting for smooth terms and broader graphical diagnostics are planned
work. The L-BFGS path remains a Torch-native numerical baseline and requires
fixed smoothing parameters. Inference for penalized smooth terms is not yet
supported.

## Development

```bash
python -m venv .venv
python -m pip install -e ".[test]"
python -m pytest
```

To regenerate and validate the reference values with R:

```bash
Rscript tools/install_r_dependencies.R
Rscript tools/generate_r_references.R
Rscript tools/generate_r_references.R --check
```

See [`docs/PARITY.md`](docs/PARITY.md) for the numerical compatibility
conventions and provenance. See [`docs/SMOOTHS.md`](docs/SMOOTHS.md) for the
P-spline API and [`docs/GAMMA.md`](docs/GAMMA.md) for the Gamma
parameterization. Poisson, NBI, and Beta are described in
[`docs/FAMILIES.md`](docs/FAMILIES.md), BCCG in
[`docs/BCCG.md`](docs/BCCG.md), BCT in [`docs/BCT.md`](docs/BCT.md), and BCPE
in [`docs/BCPE.md`](docs/BCPE.md). Classical starting values are documented in
[`docs/INITIALIZATION.md`](docs/INITIALIZATION.md). Classical fitting is
described in [`docs/RS.md`](docs/RS.md) and [`docs/CG.md`](docs/CG.md). The
prediction interface is documented in
[`docs/PREDICTION.md`](docs/PREDICTION.md), and the tabular formula API in
[`docs/FORMULAS.md`](docs/FORMULAS.md). See
[`docs/INFERENCE.md`](docs/INFERENCE.md) for covariance and Wald inference.
Likelihood criteria, model comparison, and quantile residuals are documented
in [`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md).

## Attribution and license

GAMLSS was introduced by Rigby and Stasinopoulos (2005). This project is a
translation-oriented implementation based on the GPL-licensed R ecosystem and
is distributed under the GNU General Public License v3.0 only.

TorchGAMLSS is an independent project and is not affiliated with or endorsed
by the original GAMLSS authors.

## Reference

Rigby, R. A. and Stasinopoulos, D. M. (2005). Generalized additive models for
location, scale and shape. *Applied Statistics*, 54(3), 507–554.
https://doi.org/10.1111/j.1467-9876.2005.00510.x
