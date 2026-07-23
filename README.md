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
- normal (`NO`) and Gamma (`GA`) GAMLSS families with predictors for `mu`
  and `sigma`;
- a differentiable negative log-likelihood;
- full-batch joint fitting with Torch L-BFGS;
- likelihood weights and offsets for every distribution parameter;
- Rigby-Stasinopoulos fitting cycles for linear and additive predictors;
- fixed-lambda P-splines, penalized weighted least squares, additive
  backfitting, and effective degrees of freedom;
- automatic P-spline smoothing-parameter selection with the `pb()` ML update;
- target-EDF P-splines compatible with `pb(x, df=...)`;
- local GAIC and GCV P-spline smoothing-parameter selection compatible with
  `pb.control(method=...)`;
- prediction on response, link, and additive-term scales, including new data;
- independent design matrices for each distribution parameter;
- R-generated parity fixtures for the `NO` density, links, derivatives, and a
  fitted model;
- tests for parameter recovery, gradients, link round trips, and R parity.

Formula parsing, the CG algorithm, inference, and diagnostics are planned
work. The L-BFGS path remains a Torch-native numerical baseline and requires
fixed smoothing parameters.

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
parameterization. The supported prediction interface is documented in
[`docs/PREDICTION.md`](docs/PREDICTION.md).

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
