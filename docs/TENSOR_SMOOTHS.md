# Tensor-product smooths

TorchGAMLSS provides the first fixed-smoothing-parameter slice of
`mgcv::te()`- and `mgcv::ti()`-style tensor products. The implementation is
Torch-native and uses the generic design, coefficient-space penalty, and
constraint contract.

For marginal model matrices `B_1, ..., B_d`, the tensor design is their
row-wise Kronecker product:

```text
B[i, :] = B_1[i, :] kron ... kron B_d[i, :].
```

If marginal `j` has penalty `S_j`, its penalty in the product coefficient
space is

```text
I kron ... kron S_j kron ... kron I.
```

Consequently, a `d`-dimensional tensor has `d` smoothing parameters and
penalty

```text
sum_j lambda_j * beta' S_j beta.
```

This separate penalty per marginal direction is what lets the fitted surface
have different smoothness along variables measured in different units.

## Full tensor smooth

`TensorProductSmooth` retains every marginal basis direction and, by default,
exposes one global sum-to-zero constraint to separate the smooth from the
model intercept:

```python
import torch

from torchgamlss import (
    PSpline,
    TensorProductSmooth,
    solve_penalized_least_squares,
)

x = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64)
z = torch.linspace(0.0, 2.0, 10, dtype=torch.float64)
covariates = torch.cartesian_prod(x, z)
response = (
    torch.sin(2.0 * covariates[:, 0])
    + 0.4 * covariates[:, 0] * covariates[:, 1]
)

term = TensorProductSmooth(
    (
        PSpline(-1.0, 1.0, 2.0, intervals=4, dtype=torch.float64),
        PSpline(0.0, 2.0, 8.0, intervals=4, dtype=torch.float64),
    )
)
fit = solve_penalized_least_squares(
    term.design(covariates),
    response,
    torch.ones_like(response),
    term.penalty_matrices(),
    term.smoothing_parameters,
    constraints=term.constraints(covariates),
)

with torch.no_grad():
    term.coefficients.copy_(fit.coefficients)
fitted_surface = term(covariates)
```

The marginal objects are copied as parameter-free basis definitions. Only the
tensor coefficient vector is registered as a trainable parameter, so their
coefficients are not accidentally duplicated in an optimizer.

Set `center=False` only when identifiability is handled by another model-level
constraint.

## Pure tensor interaction

`TensorInteractionSmooth` applies a sum-to-zero constraint to each marginal
basis before taking the tensor product. This removes lower-order marginal
main-effect directions and leaves the highest-order interaction, analogous to
the ANOVA construction of `mgcv::ti()`:

```python
from torchgamlss import TensorInteractionSmooth

interaction = TensorInteractionSmooth(
    (
        PSpline(-1.0, 1.0, 2.0, intervals=4, dtype=torch.float64),
        PSpline(0.0, 2.0, 8.0, intervals=4, dtype=torch.float64),
    ),
    covariates,
)
```

The training covariates define the marginal centering transforms and those
transforms are stored in the module state for prediction. A model that uses
this interaction should include the desired lower-order main effects as
separate terms.

## Formula API

Fixed-lambda tensor products can be constructed directly in a parameter
formula:

```python
from torchgamlss import GAMLSS, Normal

surface_model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": (
            "y ~ te(x, z, smoothing_parameters=(2, 8), "
            "intervals=(6, 4), name='surface')"
        ),
        "sigma": "~ 1",
    },
    data,
)
surface_fit = surface_model.fit_rs_data(data)
```

`te()` represents the complete tensor surface. Its training design is
reparameterized into the null space of the global sum-to-zero constraint, so
the constraint remains exact in L-BFGS and mini-batch fitting and the stored
mapping is reused for new data.

Use `ti()` for only the highest-order interaction and include the lower-order
effects explicitly:

```python
interaction_model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": (
            "y ~ pb(x, smoothing_parameter=3) "
            "+ pb(z, smoothing_parameter=5) "
            "+ ti(x, z, lambda_=(2, 8), intervals=(6, 4))"
        ),
        "sigma": "~ 1",
    },
    data,
)
interaction_fit = interaction_model.fit_minibatch_data(data)
```

`lambda_` fixes one smoothing parameter per margin. Omitting it makes every
margin automatic with an initial value of 10; use
`initial_lambda_=(4, 7)` to choose other LAML starting values.
`intervals`, `degree`, and `penalty_order` may be scalars or sequences with
one integer per margin.
Formula tensor marginals are equally spaced P-splines and use ten intervals
per margin when `intervals` is omitted. `name` controls the term key.
`te()` additionally accepts `center=False` when identifiability is imposed
elsewhere.

The tensor call must be a standalone additive factor over simple numeric
columns. With fixed lambdas, `fit_rs_data()` and `fit_cg_data()` use the
generic constrained multiple-penalty solver inside their backfitting cycles;
`fit_data()` and `fit_minibatch_data()` optimize the same penalty through
Torch. `fit_laml_data()` selects automatic tensor and scalar smoothness
parameters jointly for Normal location-scale, Poisson log-mean, Gamma mean/CV,
NBI mean/dispersion, Beta mean/dispersion, Student-t location/scale/shape, and
BCCG location/scale/shape, BCT location/scale/skewness/tail-shape, and BCPE
location/scale/skewness/kurtosis, and PE location/scale/shape models. All
paths run on CPU or CUDA.

## Whole-model LAML selection

Each tensor penalty becomes an independent outer coordinate
`rho_j = log(lambda_j)`:

```python
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ te(x, z, intervals=(6, 4), name='surface')",
        "sigma": "~ 1",
    },
    data,
)
fit = model.fit_laml_data(data)

fit.smoothing_parameter_labels
# (("mu", "surface", 0), ("mu", "surface", 1))
fit.smoothing_parameters
```

The adapter assembles every model coefficient and penalty, removes exact
unidentifiable directions by null-space reparameterization, minimizes the
joint Normal negative LAML, and stores selected lambdas and coefficients back
in the model. A direct `mgcv::te(..., method="REML")` fixture checks the
objective, both directional lambdas, EDF, coefficients, fitted location and
scale, and outer Hessian. See [`LAML.md`](LAML.md).

## Fixed-lambda inference

`smooth_inference_data()` and `smooth_joint_inference_data()` consume
`sum_j lambda_j S_j` directly. They provide pointwise intervals, full
within-surface covariance, single- or multi-term simultaneous Gaussian bands,
and joint coefficient covariance on training or new covariate grids:

```python
surface = surface_model.smooth_inference_data(
    data,
    new_data=prediction_grid,
)["mu"]["surface"]
joint = surface_model.smooth_joint_inference_data(
    data,
    new_data=prediction_grid,
)

surface.smoothing_parameter  # (2.0, 8.0)
surface.covariance_matrix
surface.simultaneous_confidence_band()
```

Explicit low-level tensor constraints are applied to the covariance through a
null-space transform; constrained coefficient directions consequently have
zero variance. Formula `te()` and `ti()` reuse their stored transforms.
Multivariate inference tables contain `covariate_0`, `covariate_1`, and so on.

`smooth_bootstrap_data()` and `smooth_joint_bootstrap_data()` refit tensor
models and store one lambda column per marginal penalty.
For a two-margin surface,
`bootstrap_smoothing_parameters.shape == (replicates, 2)` and the
penalty-wise mean, standard error, bias, and percentile intervals are
available on each curve result. The joint result flattens all penalty columns
and labels them by `(parameter, term, penalty_index)`. For fixed tensor
lambdas their bootstrap variance is zero, while the fitted surface still
varies across simulated responses. RS and CG accept fixed tensor lambdas.
For automatic tensor terms in an additive Normal model,
`algorithm="laml"` repeats joint marginal-lambda selection within every
successful bootstrap sample. Unconditional analytic LAML coefficient
covariance remains future work.

## Low-level algebra

The public helpers can also operate on arbitrary marginal matrices:

```python
from torchgamlss import row_tensor_product, tensor_product_penalties

tensor_design = row_tensor_product((first_design, second_design))
embedded_penalties = tensor_product_penalties(
    (first_penalty, second_penalty)
)
```

Their ordering and values are checked exactly against
`mgcv::tensor.prod.model.matrix()` and `mgcv::tensor.prod.penalties()`.

## Current scope

- Marginals must currently be `SmoothTerm` instances with exactly one
  coefficient-space penalty each.
- Joint automatic selection is available through dense whole-model LAML for
  Normal location-scale, Poisson log-mean, NBI mean/dispersion, Gamma mean/CV,
  Beta mean/dispersion, Student-t location/scale/shape, BCCG
  location/scale/shape, BCT location/scale/skewness/tail-shape, and BCPE
  location/scale/skewness/kurtosis, and PE location/scale/shape models,
  including parametric bootstrap refits.
- Tensor terms are consumed through LAML, the generic dense penalized solver,
  the classical fixed-lambda RS/CG paths, or formula fixed-lambda
  L-BFGS/mini-batch fitting.
- Dense row-wise Kronecker matrices are materialized. Discretized bases and
  structured crossproducts are planned for the large-data backend.
- The current interaction is the highest-order ANOVA interaction. A complete
  automatic formula expansion into every requested lower-order component
  remains future work; specify those components explicitly with `ti()`.

The reference algebra follows Wood (2006), *Low-rank scale-invariant tensor
product smooths for generalized additive mixed models*,
<https://doi.org/10.1111/j.1541-0420.2006.00574.x>, and the
[`mgcv` tensor-product documentation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/te.html).
