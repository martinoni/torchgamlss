# Smooth terms

TorchGAMLSS provides `PSpline`, `CyclicPSpline`, `ShrinkagePSpline`,
`AdaptivePSpline`, `AdaptiveTensorPSpline`, `MarkovRandomField`, and
`ThinPlateSpline`. The GAMLSS-compatible terms follow the core constructions
of `gamlss::pb()`, `gamlss::pbc()`, and `gamlss::pbz()`; the thin-plate,
adaptive, and graph terms follow the corresponding constructions in `mgcv`.
The ordinary P-spline uses an equally spaced cubic B-spline basis by default
and penalizes second differences of adjacent spline coefficients.

For basis matrix `B`, coefficient vector `gamma`, difference matrix `D`,
working response `z`, and working weights `W`, each smoother update solves

```text
minimize  (z - B gamma)' W (z - B gamma) + lambda ||D gamma||^2.
```

The implementation solves the equivalent augmented least-squares problem and
reports effective degrees of freedom as

```text
trace((B' W B + lambda D' D)^-1 B' W B).
```

When `lambda` is not supplied, the default ML update treats the coefficient
differences as random effects. After each penalized fit it computes

```text
sigma^2 = sum(w_i (z_i - fitted_i)^2) / (N_positive_weight - EDF)
tau^2   = ||D gamma||^2 / (EDF - penalty_order)
lambda  = sigma^2 / max(tau^2, 1e-7).
```

`lambda` is clipped to `[1e-7, 1e7]` and iterated from the default starting
value 10. These equations and safeguards follow `gamlss.pb()`.

Alternatively, `degrees_of_freedom` selects `lambda` by inverting the
hat-matrix trace on the same log-lambda interval used by R. Its semantics match
the `df` argument of `pb()`: the requested value is the nonlinear degrees of
freedom, and two unpenalized dimensions are added to obtain the target total
EDF.

Local GAIC and GCV selection are also available. For penalty multiplier `k`,
weighted residual sum of squares `RSS`, sample size `n`, and effective degrees
of freedom `EDF`, they minimize

```text
GAIC(lambda) = RSS(lambda) + k EDF(lambda)
GCV(lambda)  = n RSS(lambda) / (n - k EDF(lambda))^2.
```

The default `k=2`, bounds `[1e-7, 1e7]`, and local working-response criteria
match `pb.control()`. TorchGAMLSS uses bounded Brent minimization on
`log(lambda)` instead of R's `nlminb()` on `lambda`; both target the same
criterion minimum.

## Example

```python
import torch
from torchgamlss import GAMLSS, Normal, PSpline, RSControl

x = torch.linspace(-1, 1, 100, dtype=torch.float64)
y = 0.7 + 0.8 * x + torch.sin(torch.pi * x)

term = PSpline.from_data(x)
model = GAMLSS(
    Normal(),
    {"mu": 2, "sigma": 1},
    smooth_terms={"mu": {"x": term}},
)
design = {
    "mu": torch.column_stack((torch.ones_like(x), x)),
    "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
}
smooth_covariates = {"mu": {"x": x}}

result = model.fit_rs(
    y,
    design,
    smooth_covariates=smooth_covariates,
    control=RSControl(),
)
mu = model.predict(
    design,
    smooth_covariates=smooth_covariates,
    type="response",
)["mu"]
edf = result.smooth_effective_degrees_of_freedom["mu"]["x"]
estimated_lambda = result.smoothing_parameters["mu"]["x"]
```

After fitting, conditional uncertainty for each smooth contribution is
available from the same data:

```python
curve = model.smooth_inference(
    y,
    design,
    smooth_covariates=smooth_covariates,
)["mu"]["x"]

curve.estimates
curve.covariance_matrix
curve.correlation_matrix
curve.standard_errors
curve.confidence_intervals
```

This follows the `gamlss.pb()` variance calculation and conditions on the
fitted smoothing parameter. The intervals are on the additive predictor scale
and are pointwise. A simulation-based simultaneous band over the supplied
covariate values uses the full within-curve covariance:

```python
band = curve.simultaneous_confidence_band(
    simulations=10_000,
    generator=torch.Generator().manual_seed(2026),
)
band.critical_value
band.confidence_intervals
```

The generator makes the Monte Carlo result reproducible. Neither kind of
interval accounts for uncertainty from estimating the smoothing parameter.

For covariance across several smooth terms or distribution parameters, use
the joint fixed-lambda information matrix:

```python
joint_analytic = model.smooth_joint_inference(
    y,
    design,
    smooth_covariates=smooth_covariates,
)

cross_covariance = joint_analytic.covariance_block(
    ("mu", "x"),
    ("sigma", "z"),
)
coefficient_covariance = joint_analytic.coefficient_covariance_matrix
joint_analytic_bands = joint_analytic.simultaneous_confidence_bands(
    generator=torch.Generator().manual_seed(2026),
)
```

This calculation includes linear and constrained spline coefficients in one
expected penalized-information system. It retains cross-term and
cross-parameter covariance while conditioning on the fitted lambdas.

To propagate that uncertainty, simulate from the fitted distribution and
repeat the complete fit, including lambda selection:

```python
bootstrap = model.smooth_bootstrap(
    y,
    design,
    smooth_covariates=smooth_covariates,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)["mu"]["x"]

bootstrap.standard_errors
bootstrap.confidence_intervals
bootstrap.bootstrap_smoothing_parameters
bootstrap.smoothing_parameter_confidence_interval
bootstrap_band = bootstrap.simultaneous_confidence_band()
```

These are pointwise percentile intervals from a fixed-design parametric
bootstrap. Failed refits are replaced by new attempts up to `max_attempts`;
inspect `failure_rate` rather than ignoring convergence problems. The
bootstrap band uses the maximum standardized error across the same successful
refits and is simultaneous over the evaluation points for that smooth.

When the model contains several smooths, preserve their replicate alignment
with the joint API:

```python
joint = model.smooth_joint_bootstrap(
    y,
    design,
    smooth_covariates=smooth_covariates,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)

cross_covariance = joint.covariance_block(("mu", "x"), ("sigma", "z"))
lambda_covariance = joint.smoothing_parameter_covariance_matrix
lambda_labels = joint.smoothing_parameter_labels
joint_bands = joint.simultaneous_confidence_bands()
```

The full `covariance_matrix` follows `term_order` and `term_slices`.
Joint bands use one max-|t| critical value over every selected curve point.
Smoothing-parameter arrays use one column per penalty:
`smoothing_parameter_labels` identifies each
`(parameter, term, penalty_index)`, and `smoothing_parameter_slices` maps
terms to their penalty columns. Scalar-only models retain one column per term.

Aligned refits can also be transformed without rerunning the bootstrap:

```python
difference = joint.difference(("mu", "x"), ("sigma", "x"))
derivative = joint.derivative(("mu", "x"))
peak = difference.extremum(kind="maximum")
root = difference.crossing(level=0.0, which="first")
```

Contrasts require a common ordered evaluation grid. Derivatives and crossings
additionally require that grid to be strictly increasing. Extrema are located
on the supplied grid; crossings use linear interpolation between its points.
See [`INFERENCE.md`](INFERENCE.md) for interval interpretation and
missing-crossing diagnostics.

The same smooth terms and selection modes work with `fit_cg()` and
`CGControl`; CG performs one `additive.fit()`-style pass per parameter update
inside its joint Gauss-Seidel cycle.

The explicit linear `x` column mirrors the parametric component that R keeps
for `pb(x)`. Smooth terms may be configured independently for every family
parameter, and multiple named terms can be attached to the same parameter.
The same names must be supplied in `smooth_covariates` during fitting and
prediction. CG parity fixtures cover both configurations: smooths in `mu` and
`sigma` simultaneously, and two smooths attached to `mu`.

`PSpline.from_data()` reproduces the range expansion and small-sample interval
rules used by `pb()`. The knots, penalty, and coefficients are registered in
the Torch module state and move with `.to(device)` or `.to(dtype)`. The most
recently estimated `lambda` is available as `term.smoothing_parameter` and is
also stored in the module state.

To keep `lambda` fixed, pass it explicitly:

```python
term = PSpline.from_data(x, smoothing_parameter=12.0)
```

To request the equivalent of `pb(x, df=3)`:

```python
term = PSpline.from_data(x, degrees_of_freedom=3.0)
# The target total EDF is 5.
```

To select `lambda` with local GAIC or GCV:

```python
gaic_term = PSpline.from_data(
    x, smoothing_method="GAIC", criterion_penalty=2.0
)
gcv_term = PSpline.from_data(
    x, smoothing_method="GCV", criterion_penalty=2.0
)
```

`smoothing_parameter` and `degrees_of_freedom` are mutually exclusive.

## Cyclic and shrinkage P-splines

`CyclicPSpline` closes the final B-spline columns into the first columns and
uses the circular difference matrix from `pbc()`. Its default second-order
stencil is `(-1, 2*cos(2*pi/K), -1)`, giving the sine/cosine null space used by
R. Prediction wraps covariates by the fitted period, so values separated by
any integer multiple of the training range have identical designs and fitted
values.

Formula syntax follows R while using Python-safe keyword names:

```python
cyclic = GAMLSS.from_formula(
    Normal(),
    {"mu": "y ~ pbc(angle, lambda_=12, intervals=12)", "sigma": "~ 1"},
    data,
)
```

Omit `lambda_` for local ML selection under RS/CG or whole-model selection
under LAML. `df`, `method`, `k`, `inter`, and `sin` aliases follow the existing
`pb()` conventions. The committed fixture compares the complete basis,
circular difference root, coefficients, EDF, likelihood, and fitted values
with `gamlss::pbc()`.

`ShrinkagePSpline` implements the activated double-penalty branch of
`gamlss::pbz()`: the ordinary difference penalty is accompanied by a
first-difference penalty. A sum-to-zero constraint separates the smooth from
the model intercept. Fixed RS/CG fits receive both lambdas explicitly:

```python
fixed = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ pbz(x, lambda_=(100000, 50), intervals=20)",
        "sigma": "~ 1",
    },
    data,
)
```

For automatic whole-model LAML selection, use two positive starting values:

```python
automatic = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ pbz(x, initial_lambda_=(10, 10), intervals=10)",
        "sigma": "~ 1",
    },
    data,
)
fit = automatic.fit_laml_data(data)
```

Both selected lambdas are reported with distinct penalty indices and are
reselected in every successful LAML bootstrap replicate. The R `pbz()` code
activates its first-difference penalty only after the preliminary EDF falls
below `lim`; Torch fixed-fit parity targets that activated branch, while LAML
keeps both declared penalties active and selects them jointly. Formula
L-BFGS/mini-batch fitting is not yet claimed for `pbz()` because those paths do
not currently impose explicit coefficient constraints.

## Thin-plate regression splines

`ThinPlateSpline` implements a low-rank isotropic thin-plate regression spline
without choosing knots by hand. It follows the `mgcv` construction:

1. center the training coordinates;
2. form the full thin-plate radial kernel at the unique locations;
3. retain the eigenvectors with largest-magnitude eigenvalues;
4. impose the polynomial side conditions and retain the penalty null space;
5. scale the design columns and absorb the empirical sum-to-zero constraint.

The absorbed constraint separates the surface from the formula intercept, so
fixed-lambda `tp()` terms work through RS, CG, L-BFGS, and mini-batch fitting.
Omitting `lambda_` marks the one penalty for whole-model LAML selection:

```python
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ tp(x, z, k=15, m=2, initial_lambda_=1, name='surface')",
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data)
prediction = model.predict_data(new_data)["mu"]
```

The Python-safe formula aliases are `k` for `basis_size`, `m` for
`penalty_order`, `lambda_` for a fixed smoothing parameter, and
`initial_lambda_` for the LAML starting value. One or more covariates are
accepted. Because the penalty is isotropic, coordinates should use comparable
physical units; otherwise rescale them before fitting.

The default basis dimensions match `mgcv`: null-space dimension plus 8, 27,
or 100 columns for one, two, or at least three coordinates, respectively.
After the sum-to-zero constraint is absorbed, the fitted coefficient count is
one smaller. TorchGAMLSS currently caps the exact kernel eigendecomposition at
500 unique locations by default and selects a deterministic subset above that
threshold. This differs from `mgcv`'s larger randomized-knot/Lanczos path and
is a deliberate dense-prototype limitation pending the structured backend.

The committed two-dimensional reference compares the complete fixed-lambda
smoothing operator and out-of-sample predictions with
`mgcv::smoothCon(..., bs="tp", absorb.cons=TRUE, scale.penalty=FALSE)`.
Agreement is at floating-point precision even though eigenvector signs and
rotations need not match column by column. Automatic LAML selection,
ten-replicate bootstrap reselection, and CUDA execution are tested separately.

## Adaptive P-splines

`AdaptivePSpline` implements the one-dimensional `mgcv` adaptive smooth. A
cubic P-spline basis models the function, while a smaller non-negative
B-spline basis models how the second-difference penalty changes along the
coefficient sequence. If `D` is the second-difference root and `v_j` is one
column of the penalty basis, each component is

```text
S_j = D' diag(v_j) D,
```

and the total penalty is `sum_j lambda_j S_j`. Different selected lambdas
therefore allow roughness to vary across the covariate range rather than
forcing one global smoothness level.

Formula `ad()` mirrors the important `mgcv::s(..., bs="ad")` controls:

```python
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": (
            "y ~ ad(x, k=20, m=5, initial_lambda_=(1, 1, 1, 1, 1), "
            "name='trend')"
        ),
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data)
```

Here `k` is the main spline basis size and `m` is the penalty-basis size, hence
the number of lambdas. `lambda_=(...)` fixes every component for RS, CG,
L-BFGS, or mini-batch fitting; `initial_lambda_=(...)` supplies positive LAML
starting values. Omitting both uses `m=5` with all starts equal to one. The
empirical sum-to-zero constraint is absorbed into the stored design and
prediction mapping.

The main basis, penalty basis, local penalty matrices, and linear boundary
extrapolation reproduce the one-dimensional `mgcv` construction. The parity
fixture compares the invariant fixed smoothing operator on 100 training
values and eight predictions, including values outside the training range.
Individual centered basis columns are not compared because constraint
null-space bases are only identifiable up to rotation.

For two covariates, formula `ad()` constructs `AdaptiveTensorPSpline`:

```python
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": (
            "y ~ ad(x, z, k=(7, 6), m=(3, 3), "
            "initial_lambda_=(1, 1, 1, 1, 1, 1, 1, 1, 1), "
            "name='surface')"
        ),
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data)
```

The raw function basis is the row tensor product of the two cubic P-spline
bases. On the coefficient grid, `Drr` and `Dcc` take second differences in the
two directions and `Dcr` is the scaled mixed derivative. If `v_j` denotes a
column of the tensor penalty basis evaluated at the appropriate operator
locations, then

```text
S_j = Drr' diag(vrr_j) Drr
    + Dcc' diag(vcc_j) Dcc
    + Dcr' diag(vcr_j) Dcr.
```

Here `k=(k1, k2)` gives the two main basis sizes and `m=(m1, m2)` gives the
penalty-basis dimensions, so the term has `m1*m2` lambdas. Both penalty-basis
dimensions must currently be at least three. The two-dimensional fixed
reference deliberately uses unequal main dimensions to audit coefficient-grid
ordering; Torch and `mgcv` agree within `6e-14` for 120 fits and seven
predictions, including extrapolation. Automatic nine-lambda LAML selection,
ten-replicate bootstrap reselection, and CUDA execution are tested separately.

## Areal Markov random fields

`MarkovRandomField` represents a discrete areal effect with one raw indicator
coefficient per region. For each undirected edge `(a, b)`, the graph Laplacian
adds one to diagonal entries `a` and `b` and minus one to the two off-diagonal
entries. The resulting quadratic penalty is

```text
lambda * sum_(a,b in edges) (beta_a - beta_b)^2.
```

The formula interface accepts zero-based integer region codes:

```python
edges = ((0, 1), (1, 2), (0, 3), (1, 4), (2, 5))
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": (
            f"y ~ mrf(region, edges={edges}, area_count=6, "
            "initial_lambda_=1, name='area')"
        ),
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data)
```

Categorical regions can instead use a mapping whose keys establish the fitted
level order and whose values name neighboring regions:

```python
neighbors = {
    "north": ("center",),
    "center": ("north", "south"),
    "south": ("center",),
}
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": (
            f"y ~ mrf(region, adjacency={neighbors!r}, k=3, "
            "initial_lambda_=1, name='area')"
        ),
        "sigma": "~ 1",
    },
    data,
)
```

`nb=` is an `mgcv`-style alias for `adjacency=`. For direct tensor workflows,
`MarkovRandomField.from_labels(regions, neighbors, ...)` performs the same
encoding, while `encode_regions()` and `predict_design_labels()` reuse it for
new observations. Labelled edge pairs may use `levels=(...)` to declare an
order explicitly.

Polygon coordinates can generate that mapping directly:

```python
polygons = {
    "north": ((0, 0), (1, 0), (0.5, 0.8)),
    "center": ((1, 0), (2, 0), (1.5, 0.8)),
    "south": ((2, 0), (3, 0), (2.5, 0.8)),
}
term = MarkovRandomField.from_polygons(regions, polygons, basis_size=3)

# The same literal mapping is accepted by mrf(..., polygons=polygons).
```

`polygon_adjacency(polygons)` is also public when only the neighbor mapping is
needed. It reproduces `mgcv::pol2nb`: any exactly shared vertex creates an
edge, so corner contact counts. A region value may contain one `(n, 2)` ring
or several components. Coordinates must be finite, with at least three unique
vertices per component. There is intentionally no hidden snapping tolerance;
clean or snap noisy GIS boundaries before passing them to the model.

When the spatial penalty is already available, it can replace graph
construction entirely:

```python
term = MarkovRandomField.from_labels(
    regions,
    graph_penalty=penalty_tensor,
    levels=("north", "center", "south"),
    basis_size=3,
)

# Formula options are literals, hence a nested numeric tuple/list:
formula = (
    "y ~ mrf(region, penalty=((1, -1), (-1, 1)), "
    "levels=('north', 'south'), initial_lambda_=1)"
)
```

Exactly one of edges, adjacency, polygons, or a supplied penalty is accepted.
Supplied matrices must be square, finite, symmetric, positive semidefinite,
and follow the integer-code or explicit `levels` order. All sources
participate in the same centering and full-/low-rank natural parameterization.

`area_count` may include regions absent from the training observations, while
every observed code and edge endpoint must lie between zero and
`area_count-1`. The empirical intercept constraint is absorbed into the
stored design. Connected graphs therefore have no remaining penalty null
space after centering. Fixed `lambda_` works with RS, CG, L-BFGS, and
mini-batch fitting; `initial_lambda_` selects the graph penalty through LAML.

By default `k=area_count`, giving the full graph basis. A smaller `k` applies
the `mgcv::nat.param(type=0)` construction and retains the `k`
least-penalized graph coordinates before absorbing centering. Unobserved
regions receive the same temporary dummy rows used by `mgcv` during this
construction, so their predictions remain connected to the fitted graph.
Because truncating inside an exactly repeated eigenvalue is
implementation-dependent, TorchGAMLSS rejects such a `k` and asks for a size
that retains the complete eigenspace.

The committed 3-by-3 grid fixture matches the full-rank
`mgcv::s(region, bs="mrf")` smoothing operator for all 135 observations and
nine region predictions within `6e-15`. A second fixture uses `k=6` and agrees
for the low-rank fits and predictions within `5e-15`. Full- and low-rank LAML
selection, ten-replicate bootstrap reselection, state serialization, missing
region prediction, named-factor encoding, and CUDA execution are tested
separately. Relabeling the reference graph leaves its invariant smoother and
`mgcv` parity unchanged. A separately weighted supplied-Laplacian fixture
checks full- and low-rank `xt=list(penalty=S)` parity within `4e-15`.
A six-region polygon fixture checks `mgcv::pol2nb` adjacency and full-/low-rank
fits and predictions within `5e-15`.

## Generic multiple-penalty systems

The low-level dense solver accepts one or more fixed coefficient-space
penalties and optional linear equality constraints. It minimizes

```text
sum_i w_i (y_i - X_i beta)^2 + beta' (sum_j lambda_j S_j) beta
```

subject to

```text
C beta = 0.
```

For example:

```python
import torch
from torchgamlss import solve_penalized_least_squares

design = torch.tensor(
    [[1.0, -1.0, 0.5], [1.0, 0.0, -0.2], [1.0, 1.0, 0.8]],
    dtype=torch.float64,
)
response = torch.tensor([0.0, 0.5, 1.4], dtype=torch.float64)
weights = torch.ones_like(response)

first_penalty = torch.diag(
    torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
)
second_penalty = torch.diag(
    torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
)
sum_to_zero = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

fit = solve_penalized_least_squares(
    design,
    response,
    weights,
    (first_penalty, second_penalty),
    (2.0, 5.0),
    constraints=sum_to_zero,
)
coefficients = fit.coefficients
edf = fit.effective_degrees_of_freedom
```

Each `S_j` is validated for shape, finiteness, symmetry, dtype/device
compatibility, and numerical positive semidefiniteness. Smoothing parameters
must be finite and non-negative. Constraints are applied through an SVD
null-space reparameterization; redundant constraint rows are allowed.

The result also exposes fitted values, the combined penalty matrix, individual
penalty ranks, constraint rank and null space, and the reduced system condition
number. Rank-deficient penalties are supported when the complete penalized
system remains identifiable.

The low-level API uses fixed lambdas; whole-model selection is supplied by the
model LAML adapter. Classical `PSpline` fitting through RS and CG continues to use
the existing square-root augmented solver for exact `gamlss::pb()` numerical
compatibility. A `PSpline` can nevertheless be passed explicitly through the
generic contract with `term.design(x)`, `term.penalty_matrices()`,
`term.smoothing_parameters`, and `term.constraints(x)`.

## Tensor-product terms

`TensorProductSmooth` constructs a multidimensional smooth from two or more
single-penalty marginal `SmoothTerm` bases. Its design is the row-wise
Kronecker product of the marginal designs and it embeds one coefficient-space
penalty per marginal direction. `TensorInteractionSmooth` first centers each
marginal basis, excluding its main-effect direction before constructing the
highest-order interaction.

Both terms store only their product coefficients as trainable parameters.
Marginal basis state, interaction transforms, smoothing parameters, and
prediction mappings participate in normal Torch state serialization and
device/dtype movement. Their design, penalty, constraint, EDF, and quadratic
penalty paths work with autograd and the generic dense solver on CPU or CUDA.
Formula `te()` absorbs the full tensor's global sum-to-zero constraint into
its coefficient mapping; formula `ti()` stores its marginal interaction
transforms. Both therefore retain identifiability when fitted with fixed
lambdas through RS, CG, L-BFGS, or mini-batch Adam and when predicting new
data. RS and CG delegate each multiply penalized partial-residual update to
the generic constrained solver while retaining the original scalar
square-root path for exact `pb()` compatibility.

Formula `te()`/`ti()` without `lambda_=` instead marks every marginal
parameter for joint selection. `fit_laml_data()` assembles all scalar and
tensor penalties in one Normal location-scale objective, selects the free log
lambdas together, and writes the result back to the model. Use
`initial_lambda_=` for LAML starting values. The result exposes flat
`(parameter, term, penalty_index)` labels and term slices.

See [`TENSOR_SMOOTHS.md`](TENSOR_SMOOTHS.md) for equations and examples. The
low-level row-product and penalty embedding are checked against
`mgcv::tensor.prod.model.matrix()` and
`mgcv::tensor.prod.penalties()`.

## Current limitations

- Only equally spaced ordinary P-splines are available as tensor-product
  marginals; cyclic and shrinkage terms are one-dimensional, while thin-plate
  terms are isotropic multivariate smooths rather than tensor marginals. The
  current adaptive term is one-dimensional; the `mgcv` two-dimensional
  adaptive construction is not yet implemented.
- Generic multiple-penalty systems currently use the dense low-level solver
  directly, through fixed-lambda formula RS/CG/L-BFGS/mini-batch fitting, or
  through dense whole-model LAML for Normal location-scale, Poisson log-mean,
  NBI mean/dispersion, Gamma mean/CV, Beta mean/dispersion, Student-t
  location/scale/shape, BCCG location/scale/shape, BCT
  location/scale/skewness/tail-shape, and BCPE
  location/scale/skewness/kurtosis, PE location/scale/shape, and uncensored GG
  positive-location/scale/shape, uncensored IG mean/dispersion, uncensored
  LOGNO mean-log/scale, and uncensored WEI scale/shape models.
- Linear-coefficient inference, within-curve covariance, pointwise smooth
  intervals, and simultaneous smooth bands are available conditionally for
  additive models, including fixed-lambda tensor terms. Analytic inference
  provides joint covariance across linear coefficients, smooth coefficients,
  smooth terms, and distribution parameters. Parametric smooth-bootstrap
  summaries support scalar- and vector-lambda terms through RS, CG, or
  supported whole-model LAML fits, with one stored bootstrap column per
  penalty.
  LAML refits repeat joint scalar/tensor selection in each successful
  replicate.
- P-spline automatic smoothing selection is available through `fit_rs()` and
  `fit_cg()`. Automatic tensor, shrinkage, thin-plate, and adaptive terms use
  whole-model LAML; joint L-BFGS fitting requires fixed smoothing parameters.
- Prediction uses the stored B-spline basis. Out-of-range extrapolation parity
  with R's natural-spline prediction helper has not been verified.
