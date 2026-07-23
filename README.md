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

The first vertical slice contains:

- identity, log, and logit links;
- a normal GAMLSS family with predictors for `mu` and `sigma`;
- a differentiable negative log-likelihood;
- independent design matrices for each distribution parameter;
- tests for parameter recovery, gradients, and link round trips.

Formula parsing, RS/CG fitting, penalized splines, smoothing-parameter
selection, inference, diagnostics, and R parity tests are planned work.

## Development

```bash
python -m venv .venv
python -m pip install -e ".[test]"
python -m pytest
```

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

