# R parity protocol

TorchGAMLSS treats the R implementations as executable references. A family is
not considered supported until its parameterization, links, log likelihood,
derivatives, and at least one fitted model have numerical parity tests.

The protocol has two complementary layers:

- focused fixtures in `tests/reference/` isolate densities, derivatives,
  fitting components, diagnostics, and API behavior;
- manifest-driven cases in `examples/` execute a complete analysis in R and
  Python, align standardized tables by semantic keys, and report numerical
  errors for every declared result.

## Declarative end-to-end parity

`tools/run_parity.py` reads a JSON manifest that declares the R and Python
commands, result files, row keys, exact columns, numeric columns, and their
absolute and relative tolerances. Commands are passed directly to subprocesses
without invoking a shell. Numeric keys, such as probability grids, can also
have tolerances so harmless cross-language decimal serialization does not
prevent row alignment.

The first complete case is
[`examples/normal_location_scale`](../examples/normal_location_scale/README.md).
It fits the same weighted Normal RS model with parameter-specific offsets and
predictors for both `mu` and `sigma`. Run both implementations from the
repository root:

```powershell
python tools/run_parity.py `
  examples/normal_location_scale/parity.json `
  --output-dir work/parity/normal-location-scale
```

The output includes each implementation's tables and metadata, the Python
diagnostic figure, and `report.json`. Each numeric column reports its maximum
absolute and relative error; a failure identifies the first divergent
semantic key and value.

Python-only environments use the committed R result:

```powershell
python tools/run_parity.py `
  examples/normal_location_scale/parity.json `
  --r-reference examples/normal_location_scale/reference/r `
  --output-dir work/parity/normal-location-scale-reference
```

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

## Box-Cox Cole-Green family (`BCCG`)

The BCCG slice is the first three-parameter parity target. Its fixtures cover:

- `dBCCG(..., log=TRUE)`, `pBCCG()`, and the identity/log/identity links;
- the `mu`, `sigma`, and `nu` scores and all expected second derivatives
  exposed by `BCCG()`;
- R-compatible starts and strictly positive response support;
- a weighted RS fit with independent formulas and offsets for all three
  parameters;
- formula prediction, full-Hessian inference, information criteria, and
  continuous quantile residuals.

The implementation evaluates the Box-Cox transform through its analytic
series at `nu=0`. This is algebraically the same lognormal limit while avoiding
the indeterminate divisions in a literal evaluation of the R expressions.
Tests compare nonzero values directly with R and independently verify density,
CDF, and score continuity at zero.

## Box-Cox t family (`BCT`)

BCT is the first four-parameter parity target. Its fixtures cover:

- `dBCT(..., log=TRUE)`, `pBCT()`, and the identity/log/identity/log links;
- all four parameter scores and all ten expected second-derivative entries
  supplied by `BCT()`;
- a converged weighted RS fit with separate predictors for `mu`, `sigma`, and
  `nu`, plus an estimated `tau` intercept;
- formula prediction, seven-coefficient full-Hessian inference, information
  criteria, and continuous quantile residuals.

TorchGAMLSS evaluates the Student t CDF through a differentiable regularized
incomplete beta implementation. Direct CDF values agree with R and SciPy at
float64 precision. For RS, the `tau` working score retains R's finite
difference of `0.01`; autograd of the likelihood uses the differentiable CDF
itself.

## Box-Cox power-exponential family (`BCPE`)

The second four-parameter slice targets `BCPE(mu, sigma, nu, tau)`. Its
fixtures cover:

- `dBCPE(..., log=TRUE)`, `pBCPE()`, and the
  identity/log/identity/log links;
- all four parameter scores and all ten expected second-derivative entries
  supplied by `BCPE()`;
- a converged weighted RS fit with separate predictors for `mu`, `sigma`, and
  `nu`, plus an estimated `tau` intercept;
- formula prediction, seven-coefficient full-Hessian inference, information
  criteria, and continuous quantile residuals.

The power-exponential CDF is evaluated through a differentiable regularized
incomplete gamma implementation. Its values agree with both R and SciPy at
float64 precision, including gradients with respect to the gamma shape. BCPE
uses R's finite difference of `0.001` for the RS `tau` working score, while
autograd differentiates the likelihood CDF directly. Tests also verify the
`nu=0` log-power-exponential limit and the exact `tau=2` BCCG limit.

## Coefficient inference

Inference fixtures compare TorchGAMLSS with `vcov.gamlss(type="all")` and the
default `summary.gamlss(type="vcov")` calculations for weighted, offset models
from the `NO`, `PO`, `NBI`, `BE`, `BCCG`, `BCT`, and `BCPE` families. They
cover:

- the full observed-Hessian covariance matrix, including cross-parameter
  entries;
- standard errors and coefficient statistics;
- residual degrees of freedom under frequency and case weights;
- two-sided Student t p-values;
- 95% Wald confidence intervals.

R obtains the Hessian numerically with `optimHess`; TorchGAMLSS obtains it
through Torch automatic differentiation. Parity is therefore defined to the
documented numerical tolerance rather than bitwise equality.

Conditional-inference fixtures additionally cover fixed- and ML-lambda Normal
RS fits and a weighted, offset, fixed-lambda Beta CG fit. As in
`gen.likelihood()` and `vcov.gamlss()`, the fitted smooth contributions are
held fixed and the Hessian contains only the linear coefficients. The
fixtures compare their covariance, standard errors, tests, confidence
intervals, and fitted residual degrees of freedom.

Smooth-curve inference fixtures compare the fitted contribution, conditional
variance, pointwise standard error, and 95% interval produced from
`fit$mu.var` and `fit$sigma.var`. They cover fixed and ML smoothing in RS,
simultaneous `mu` and `sigma` smooths in CG, and two smooth terms attached to
one parameter. TorchGAMLSS exposes the corresponding full within-curve
covariance and simulation-based simultaneous band as extensions; the R
fixtures directly validate the covariance diagonal.

Quantile fixtures compare seven probabilities for all eight families directly
against `qNO`, `qGA`, `qPO`, `qNBI`, `qBE`, `qBCCG`, `qBCT`, and `qBCPE`.
Discrete tests additionally verify the smallest-count CDF definition.
Response-scale centile bootstrap bands are a simulate-and-refit extension over
these parity-tested quantiles.

Parametric smooth bootstrap is also an extension: it composes the
parity-tested family distributions, RS/CG fitting, and smoothing-selection
updates rather than reproducing a single `gamlss` return value. Seeded
samplers for BCCG, BCT, and BCPE are checked through their fitted CDFs using
the probability integral transform. Joint bootstrap covariance and
multi-smooth max-|t| bands reuse the same aligned refits; they do not claim a
direct R return-value counterpart. Smooth contrasts, numerical derivatives,
grid extrema, and interpolated crossings are likewise derived TorchGAMLSS
extensions over those aligned bootstrap curves.

## Cole-Green fitting

The CG fixtures compare the joint cross-derivative cycles and additive
backfitting in TorchGAMLSS with `gamlss(..., method=CG())`. They cover:

- a weighted Beta model with offsets in both predictors and default starting
  values;
- a weighted BCT model with seven coefficients, explicit parameter-scale
  starts, and all six cross-derivative blocks;
- a weighted Beta model combining cross derivatives, offsets, and a
  fixed-lambda `pb()` term;
- Normal `pb()` fits with fixed and ML-selected smoothing parameters;
- target-EDF, local GAIC, and local GCV smoothing-parameter selection;
- simultaneous Beta `mu` and `sigma` smooths, including independent interior
  ML smoothing-parameter estimates;
- two fixed-lambda smooths in one `mu` predictor;
- smooth coefficients and fitted contributions, fitted distribution
  parameters, smoothing parameters, and effective degrees of freedom;
- coefficients, deviance, likelihood, convergence, and outer iteration
  counts.

The Beta results agree at float64 precision. The BCT fixture disables
automatic step halving in both implementations because the R final-iteration
halving path can leave its saved coefficients out of sync with its fitted
predictors. With that path excluded, the four-parameter coefficients agree to
approximately `1e-11` and the deviance to float64 precision.

The fixed-lambda CG fits and the ML case reproduce R's outer iteration counts.
For target-EDF, GAIC, and GCV, parity is based on the selected smoothing
parameter and final fit. Root-finder and optimizer paths can produce different
cycle counts; the same convention is used for the corresponding RS fixtures.
The simultaneous-smooth fixtures additionally compare every separate smooth
contribution and coefficient vector, not only their summed predictors.

## Diagnostics and quantile residuals

Diagnostics fixtures cover all eight supported families and a fixed-lambda
Normal P-spline fit. They compare:

- log likelihood and global deviance;
- model and residual effective degrees of freedom;
- original and effective observation counts under integer frequency weights
  and non-integer case weights;
- AIC, AICc, GAIC with `k=3`, and SBC/BIC.

The effective degrees of freedom for an additive parameter equal its linear
design dimension plus each smooth EDF minus the nullity of that smooth's
penalty. This reproduces `fit$df.fit` without counting the unpenalized
polynomial subspace twice.

Quantile-residual fixtures compare the `pNO`, `pGA`, `pPO`, `pNBI`, `pBE`,
`pBCCG`, `pBCT`, and `pBCPE` CDFs and normal-score transformations. The
discrete PO and NBI fixtures use committed uniform values inside
`[F(y-1), F(y)]`, making the randomized Dunn-Smyth calculation exactly
reproducible.

The four-panel `plot()` fixture additionally compares the residual mean,
sample variance, skewness, kurtosis, and Filliben correlation produced by R
`plot.gamlss()` for the Normal reference fit. Plotting positions follow R
`qqnorm()`.

The worm-plot fixture compares a global plot and four non-overlapping
covariate-conditioned panels against R `wp()`. It checks every interval
boundary and all four coefficients from the cubic detrended-Q-Q regressions.
The theoretical quantiles, 95% pointwise bands, default limits, and interval
construction follow `R/wp.R` and `graphics::co.intervals()`.

Bucket-plot fixtures compare weighted moment, central-centile, and
tail-centile statistics against `momentSK()` and `centileSK()`. A four-panel
conditioned fixture additionally checks interval boundaries, transformed
coordinates, effective weighted counts, and Jarque-Bera values. Bootstrap
resampling follows R's paired resampling of residuals and prior weights;
cross-language parity is asserted for the statistics rather than random
sample indices.

## Reproducing the fixtures

From the repository root:

```powershell
Rscript tools/install_r_dependencies.R
Rscript tools/generate_r_references.R
Rscript tools/generate_r_references.R --check
python tools/run_parity.py examples/normal_location_scale/parity.json `
  --output-dir work/parity/normal-location-scale
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

The RS and CG implementations follow the working-response, diagonal, and
cross-parameter Fisher-weight updates in `gamlss` 5.5-0, `R/gamlss-5.R`.
Additive backfitting and P-splines follow `R/add.r` and `R/pb.R`; the latter
is also the reference for ML, target-EDF, GAIC, and GCV smoothing selection.
See [`RS.md`](RS.md), [`CG.md`](CG.md), and [`SMOOTHS.md`](SMOOTHS.md) for the
equations and current scope.

The target-EDF fixture compares final numerical results rather than requiring
the same outer iteration count. R's finite-tolerance `uniroot()` updates can
produce a longer classical-fitting convergence path even when the final fit
agrees.

The GAIC and GCV fixtures follow the same final-result convention.
TorchGAMLSS uses bounded Brent minimization on `log(lambda)`, while R uses
`nlminb()` on `lambda`; optimizer and outer-cycle counts may differ even when
the selected smoothing parameter and final fit agree.

Likelihood criteria follow `GAIC.gamlss()` and `logLik.gamlss()`. Quantile
residuals follow the family `rqres` definitions and the shared `rqres()`
helper in `gamlss.dist`. The four-panel layout and summary statistics follow
`plot.gamlss()`. Worm plots follow `wp()` in `R/wp.R`. Bucket statistics and
conditioning follow `bp()` in `R/bp.R`, `momentSK()`, and `centileSK()`.
