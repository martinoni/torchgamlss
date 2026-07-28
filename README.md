# TorchGAMLSS

TorchGAMLSS is an early-stage PyTorch implementation of Generalized Additive
Models for Location, Scale and Shape (GAMLSS).

The project aims to combine numerical compatibility with the R packages
[`gamlss`](https://github.com/gamlss-dev/gamlss), `gamlss.dist`, and
`gamlss.tr` with PyTorch automatic differentiation, composable predictors,
and optional GPU execution.

> [!WARNING]
> TorchGAMLSS is alpha software. The latest private pre-release is 0.1.0a1;
> the main development line is 0.1.0a2.dev0. It is not yet suitable for
> production or high-stakes statistical use without independent validation.

## Initial design

Every parameter of a response distribution has its own predictor and link:

```text
eta_k = X_k beta_k + sum_j f_jk(x_j) + g_k(z_k; phi_k) + offset_k
theta_k = inverse_link_k(eta_k)
Y | X ~ D(theta_1, ..., theta_K)
```

The smooth and neural terms are optional. A neural module can be attached to
only the parameters where it is scientifically useful.

The current implementation contains:

- identity, inverse, log, and logit links;
- Normal (`NO`), Gamma (`GA`), Poisson (`PO`), negative-binomial type I
  (`NBI`), Beta (`BE`), three-parameter Box-Cox Cole-Green (`BCCG`), and
  Student-t location-scale (`TF`), power-exponential location-scale-shape
  (`PE`), four-parameter Box-Cox t (`BCT`), and Box-Cox power-exponential
  (`BCPE`) families;
- fixed-bound continuous and discrete family truncation through
  `TruncatedFamily`, with R parity and on-device gradients verified for
  Normal and Poisson bases;
- a differentiable negative log-likelihood;
- full-batch joint fitting with Torch L-BFGS;
- bounded-intermediate mini-batch fitting with Adam, deterministic chunked
  objective evaluation, holdout early stopping with best-state restoration,
  CUDA FP16/BF16 automatic mixed precision, and CPU/CUDA benchmarking;
- re-iterable `Dataset`/`DataLoader` fitting that streams CPU or on-disk
  batches to the model device without resident full-data tensors, with atomic
  epoch checkpoints and exact optimizer/RNG resumption;
- optional Torch modules and a standard MLP predictor for selected
  distribution parameters, composable with linear terms, P-splines, and
  offsets;
- shared neural backbones with parameter-specific heads for learning common
  representations across location, scale, and shape predictors;
- likelihood weights and offsets for every distribution parameter;
- R-compatible family defaults and user-supplied parameter-scale starting
  values for RS, CG, and tensor, formula, or streaming mini-batch fitting;
- Rigby-Stasinopoulos fitting cycles for linear and additive predictors;
- Cole-Green joint fitting with cross-parameter working weights for
  linear and additive predictors;
- fixed-lambda P-splines, penalized weighted least squares, additive
  backfitting, and effective degrees of freedom;
- automatic P-spline smoothing-parameter selection with the `pb()` ML update;
- target-EDF P-splines compatible with `pb(x, df=...)`;
- local GAIC and GCV P-spline smoothing-parameter selection compatible with
  `pb.control(method=...)`;
- prediction on response, link, additive-term, quantile, and centile scales,
  including new data;
- joint full-Hessian covariance, standard errors, Wald tests, and confidence
  intervals for parametric models and conditional linear-coefficient inference
  with fitted smooth contributions held fixed;
- R-compatible pointwise standard errors, full within-curve covariance, and
  simulation-based simultaneous confidence bands for fitted P-spline
  contributions, conditional on their smoothing parameters;
- analytic fixed-lambda joint covariance across linear coefficients, spline
  coefficients, smooth terms, and distribution parameters;
- fixed-design parametric bootstrap intervals, cross-smooth covariance, and
  joint bands that refit RS or CG and repeat smoothing-parameter selection;
- aligned bootstrap inference for smooth contrasts, differences, derivatives,
  extrema, and linearly interpolated crossings;
- fixed-design bootstrap intervals and joint bands for response-scale centile
  curves, including repeated smoothing selection;
- global deviance, AIC, AICc, GAIC, SBC/BIC, and model-comparison weights;
- continuous and randomized discrete normal quantile residuals;
- four-panel quantile-residual diagnostics through `plot()` and `plot_data()`,
  including the time-series ACF/PACF variant;
- global and covariate-conditioned worm plots through `wp()`, `wp_data()`,
  and `worm_plot()`;
- moment and centile bucket plots through `bp()`, `bp_data()`, and
  `bucket_plot()`, with weighted statistics and nonparametric bootstrap;
- Wilkinson formulas for tabular fitting and prediction, including categorical
  variables, `offset()`, and `pb()`;
- independent design matrices for each distribution parameter;
- R-generated parity fixtures for all supported families' densities or masses,
  links, derivatives, starting values, and fitted models;
- a declarative R-to-Python parity harness with a complete weighted Normal
  location-scale example, standardized result tables, and an error report;
- tests for parameter recovery, gradients, link round trips, and R parity.

The L-BFGS path remains a Torch-native numerical baseline and requires fixed
smoothing parameters.
Analytic fixed-lambda inference now provides joint covariance across linear
coefficients, penalized spline coefficients, smooth terms, and distribution
parameters. Aligned parametric-bootstrap refits additionally propagate
smoothing-parameter selection and retain the dependence among reselected
parameters.

## Installation

The current alpha is distributed through the private
[`v0.1.0a1` GitHub pre-release](https://github.com/martinoni/torchgamlss/releases/tag/v0.1.0a1).
Development wheels are also retained as private GitHub workflow artifacts.
After downloading the release wheel:

```bash
python -m pip install torchgamlss-0.1.0a1-py3-none-any.whl
```

TorchGAMLSS supports Python 3.10 through 3.13 on Linux and Windows. PyTorch
selects CPU or CUDA support through the installed Torch build; TorchGAMLSS
itself is a pure-Python wheel.

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
Rscript tools/generate_truncated_references.R
```

The complete weighted Normal location-scale example can be run in both
languages with:

```bash
python tools/run_parity.py examples/normal_location_scale/parity.json \
  --output-dir work/parity/normal-location-scale
```

It produces aligned fit, coefficient, fitted-parameter, quantile, and residual
tables, a machine-readable numerical error report, and a compact diagnostic
figure. See
[`examples/normal_location_scale/README.md`](examples/normal_location_scale/README.md)
for the formulas and the no-R validation mode.

The Poisson-versus-NBI count-regression comparison is the second complete
case:

```bash
python tools/run_parity.py examples/count_model_comparison/parity.json \
  --output-dir work/parity/count-model-comparison
```

It compares common mean predictors under equidispersion and modeled
overdispersion, including AIC weights, conditional variances, discrete
quantiles, and reproducible randomized residuals. See
[`examples/count_model_comparison/README.md`](examples/count_model_comparison/README.md)
for the statistical interpretation.

The third complete case fits smooth BCCG fetal-growth centile curves:

```bash
python tools/run_parity.py examples/bccg_centile_curves/parity.json \
  --output-dir work/parity/bccg-centile-curves
```

It compares smooth location, scale, and shape predictors, their EDFs, all
fitted distribution parameters, quantile residuals, and nine response
centiles on a common age grid. See
[`examples/bccg_centile_curves/README.md`](examples/bccg_centile_curves/README.md)
for the model, visualization, data provenance, and no-R validation mode.

See [`docs/PARITY.md`](docs/PARITY.md) for the numerical compatibility
conventions and provenance. See [`docs/SMOOTHS.md`](docs/SMOOTHS.md) for the
P-spline API and [`docs/GAMMA.md`](docs/GAMMA.md) for the Gamma
parameterization. Poisson, NBI, and Beta are described in
[`docs/FAMILIES.md`](docs/FAMILIES.md), BCCG in
[`docs/BCCG.md`](docs/BCCG.md), BCT in [`docs/BCT.md`](docs/BCT.md), and BCPE
in [`docs/BCPE.md`](docs/BCPE.md). The Student-t location-scale family is
documented in [`docs/TF.md`](docs/TF.md), and the power-exponential
location-scale-shape family in [`docs/PE.md`](docs/PE.md). Classical starting
values are documented in
[`docs/INITIALIZATION.md`](docs/INITIALIZATION.md). Classical fitting is
described in [`docs/RS.md`](docs/RS.md) and [`docs/CG.md`](docs/CG.md). The
R-to-Python workflow mapping is in
[`docs/R_TO_PYTHON.md`](docs/R_TO_PYTHON.md). The prediction interface is
documented in
[`docs/PREDICTION.md`](docs/PREDICTION.md), and the tabular formula API in
[`docs/FORMULAS.md`](docs/FORMULAS.md). See
[`docs/INFERENCE.md`](docs/INFERENCE.md) for covariance and Wald inference.
Likelihood criteria, model comparison, and quantile residuals are documented
in [`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md). Mini-batch optimization,
CPU/CUDA benchmarks, and reproducibility guidance are in
[`docs/MINIBATCH.md`](docs/MINIBATCH.md). Hybrid neural distributional
predictors are described in [`docs/NEURAL.md`](docs/NEURAL.md), and shared
backbones in [`docs/SHARED.md`](docs/SHARED.md). Release verification is
documented in [`docs/RELEASING.md`](docs/RELEASING.md), and user-visible
changes in [`CHANGELOG.md`](CHANGELOG.md).

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
