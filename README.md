# TorchGAMLSS

[![tests](https://github.com/martinoni/torchgamlss/actions/workflows/tests.yml/badge.svg)](https://github.com/martinoni/torchgamlss/actions/workflows/tests.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue)](https://www.python.org/)
[![GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue)](LICENSE)

TorchGAMLSS is an early-stage PyTorch implementation of Generalized Additive
Models for Location, Scale and Shape (GAMLSS).

The project aims to combine numerical compatibility with the R packages
[`gamlss`](https://github.com/gamlss-dev/gamlss), `gamlss.dist`,
`gamlss.tr`, and `gamlss.cens` with PyTorch automatic differentiation,
composable predictors, and optional GPU execution.

> [!WARNING]
> TorchGAMLSS is alpha software. The latest GitHub pre-release is 0.1.0a2;
> the main development line is 0.1.0a3.dev0. It is not yet suitable for
> production or high-stakes statistical use without independent validation.

## Project scope

TorchGAMLSS is an independent project and is not affiliated with or endorsed
by the original GAMLSS authors. The R packages remain the canonical reference
for the established GAMLSS methodology.

Another independent project,
[`gamlss-python`](https://github.com/fzhao70/gamlss-python), provides a
NumPy/SciPy-oriented Python port that prioritizes close reproduction of the R
API and algorithms. TorchGAMLSS instead focuses on PyTorch autograd, CPU/CUDA
execution, streaming and mini-batch optimization, neural predictors, and
differentiable composition with classical GAMLSS fitting. Users should choose
the implementation whose scope best matches their analysis and validate
important results against the R reference.

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
  (`BCPE`), Weibull (`WEI`), log-normal (`LOGNO`), inverse-Gaussian (`IG`),
  and generalized gamma (`GG`) families;
- scalar- or observation-bound continuous and discrete family truncation
  through `TruncatedFamily`, with `gamlss.tr` parity and on-device gradients
  verified across all ten pre-survival base families;
- exact, right-, left-, and interval-censored continuous likelihoods through
  `CensoredResponse` and `CensoredFamily`, with `survival::Surv` status
  compatibility and `gamlss.cens` parity for Weibull, log-normal,
  inverse-Gaussian, and generalized-gamma models;
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
- a dense low-level solver for multiple fixed positive-semidefinite smooth
  penalties and null-space-reparameterized linear constraints on CPU or CUDA;
- full tensor-product smooths and highest-order ANOVA tensor interactions,
  with one penalty per marginal direction, explicit identifiability
  constraints, and `mgcv` algebraic parity;
- dense whole-model LAML for additive Normal location-scale, Poisson log-mean,
  NBI mean/dispersion, Gamma mean/CV, Beta mean/dispersion, and Student-t
  location/scale/shape models, including formula
  `fit_laml_data()`, automatic
  scalar and tensor lambdas, null-space constraints, outer diagnostics,
  implicit outer gradients and Hessians, CPU/CUDA execution, and direct
  `mgcv` REML parity;
- automatic P-spline smoothing-parameter selection with the `pb()` ML update;
- target-EDF P-splines compatible with `pb(x, df=...)`;
- local GAIC and GCV P-spline smoothing-parameter selection compatible with
  `pb.control(method=...)`;
- prediction on response, link, additive-term, quantile, centile, survival,
  hazard, and cumulative-hazard scales, including new data;
- joint full-Hessian covariance, standard errors, Wald tests, and confidence
  intervals for parametric models and conditional linear-coefficient inference
  with fitted smooth contributions held fixed;
- R-compatible pointwise standard errors, full within-curve covariance, and
  simulation-based simultaneous confidence bands for fitted P-spline
  contributions, conditional on their smoothing parameters;
- analytic fixed-lambda joint covariance across linear coefficients, spline
  coefficients, smooth terms, and distribution parameters, including
  multiply penalized tensor surfaces;
- fixed-design parametric bootstrap intervals, cross-smooth covariance, and
  joint bands that refit RS, CG, or supported whole-model LAML models, repeat
  available smoothing-parameter selection, and retain one lambda column per
  tensor penalty;
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
  variables, `offset()`, `pb()`, and fixed or LAML-selected `te()`/`ti()`
  tensor products;
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

The current alpha is published on
[TestPyPI](https://test.pypi.org/project/torchgamlss/0.1.0a2/) and attached to
the [`v0.1.0a2` GitHub pre-release](https://github.com/martinoni/torchgamlss/releases/tag/v0.1.0a2).
Download only TorchGAMLSS from TestPyPI, then install the local wheel so its
runtime dependencies are resolved from the production PyPI index:

```bash
python -m pip download --no-deps \
  --index-url https://test.pypi.org/simple/ \
  torchgamlss==0.1.0a2
python -m pip install torchgamlss-0.1.0a2-py3-none-any.whl
```

TorchGAMLSS supports Python 3.10 through 3.13 on Linux and Windows. PyTorch
selects CPU or CUDA support through the installed Torch build; TorchGAMLSS
itself is a pure-Python wheel. Install the matching CUDA-enabled Torch build
first when GPU execution is required; see
[`docs/NEURAL.md`](docs/NEURAL.md#cuda-and-benchmark).

Development wheels are also retained as GitHub Actions workflow artifacts.

To install the current development branch directly from GitHub:

```bash
python -m pip install \
  "torchgamlss @ git+https://github.com/martinoni/torchgamlss.git"
```

## Quick start

```python
import pandas as pd

from torchgamlss import GAMLSS, Normal

data = pd.DataFrame(
    {
        "x": [-2.0, -1.4, -0.8, -0.2, 0.3, 0.9, 1.5, 2.1],
        "y": [-2.2, -1.3, -0.7, -0.1, 0.5, 1.0, 1.8, 2.4],
    }
)

model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ x",
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_rs_data(data)

print(fit.converged)
print(model.predict_data(data)["mu"])
```

Every distribution parameter can have its own formula, smooth terms, offsets,
or neural contribution. See
[`docs/R_TO_PYTHON.md`](docs/R_TO_PYTHON.md) for a complete mapping from an R
workflow. Formula `te()` and `ti()` terms can be fitted with fixed lambdas
through RS, CG, L-BFGS, or mini-batch Adam, or with jointly selected marginal
lambdas through `fit_laml_data()` for Normal location-scale, Poisson log-mean,
NBI mean/dispersion, Gamma mean/CV, Beta mean/dispersion, and Student-t
location/scale/shape models. Their parametric bootstrap
stores one value per
marginal lambda while preserving the scalar `pb()` API; `algorithm="laml"`
repeats automatic joint tensor selection in every successful bootstrap
replicate. The generic tensor-product API, examples, and current limitations
are documented in
[`docs/TENSOR_SMOOTHS.md`](docs/TENSOR_SMOOTHS.md).

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
Rscript tools/generate_truncated_references.R --check
Rscript tools/generate_censored_references.R
Rscript tools/generate_censored_references.R --check
Rscript tools/generate_mgcv_tensor_reference.R
Rscript tools/generate_mgcv_tensor_reference.R --check
Rscript tools/generate_mgcv_laml_reference.R
Rscript tools/generate_mgcv_laml_reference.R --check
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

The fourth complete case fits a right-censored generalized-gamma regression:

```bash
python tools/run_parity.py examples/generalized_gamma_survival/parity.json \
  --output-dir work/parity/generalized-gamma-survival
```

It compares the three fitted distribution parameters, latent event-time
quantiles, and survival, hazard, and cumulative-hazard curves. See
[`examples/generalized_gamma_survival/README.md`](examples/generalized_gamma_survival/README.md)
for the censoring design and no-R validation mode.

See [`docs/PARITY.md`](docs/PARITY.md) for the numerical compatibility
conventions and provenance. See [`docs/SMOOTHS.md`](docs/SMOOTHS.md) for the
P-spline API, [`docs/LAML.md`](docs/LAML.md) for whole-model smoothing
selection, and [`docs/GAMMA.md`](docs/GAMMA.md) for the Gamma
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
[`docs/CENSORING.md`](docs/CENSORING.md) for event-time families, censored
responses, and survival prediction. See
[`docs/INFERENCE.md`](docs/INFERENCE.md) for covariance and Wald inference.
Likelihood criteria, model comparison, and quantile residuals are documented
in [`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md). Mini-batch optimization,
CPU/CUDA benchmarks, and reproducibility guidance are in
[`docs/MINIBATCH.md`](docs/MINIBATCH.md). Hybrid neural distributional
predictors are described in [`docs/NEURAL.md`](docs/NEURAL.md), and shared
backbones in [`docs/SHARED.md`](docs/SHARED.md). Release verification is
documented in [`docs/RELEASING.md`](docs/RELEASING.md), and user-visible
changes in [`CHANGELOG.md`](CHANGELOG.md).

## Contributing, security, and citation

Contributions and reproducible parity reports are welcome. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request and use the
private process in [`SECURITY.md`](SECURITY.md) for security-sensitive
reports. Community participation follows the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

Citation metadata are available in [`CITATION.cff`](CITATION.cff). Scientific
work using TorchGAMLSS should also cite the original GAMLSS methodology
described below.

## Attribution and license

GAMLSS was introduced by Rigby and Stasinopoulos (2005). This project is a
translation-oriented implementation based on the GPL-licensed R ecosystem and
is distributed under the GNU General Public License v3.0 only.

## Reference

Rigby, R. A. and Stasinopoulos, D. M. (2005). Generalized additive models for
location, scale and shape. *Applied Statistics*, 54(3), 507–554.
https://doi.org/10.1111/j.1467-9876.2005.00510.x
