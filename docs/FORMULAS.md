# Formulas and tabular data

TorchGAMLSS uses
[`formulaic`](https://matthewwardrop.github.io/formulaic/) to compile
Wilkinson formulas into design matrices. The fitted encodings are retained so
categorical contrasts and column order remain identical when predicting new
data. See [`R_TO_PYTHON.md`](R_TO_PYTHON.md) for a side-by-side translation
from R `gamlss` calls.

## Constructing and fitting a model

Supply one formula for every family parameter. The first parameter formula
contains the response; the remaining formulas are right-hand-side formulas:

```python
import pandas as pd
from torchgamlss import GAMLSS, Gamma

data = pd.read_csv("observations.csv")
model = GAMLSS.from_formula(
    Gamma(),
    {
        "mu": "y ~ x + C(group) + pb(age) + offset(mu_offset)",
        "sigma": "~ z + offset(sigma_offset)",
    },
    data,
)
result = model.fit_rs_data(data, weights="weight")
```

RS starting values may be scalars, vectors, or names of columns in `data`:

```python
result = model.fit_rs_data(
    data,
    weights="weight",
    initial_parameters={"mu": "mu_start", "sigma": 0.5},
)
```

See [`INITIALIZATION.md`](INITIALIZATION.md) for validation and default
behavior.

Formula models can instead use the joint Cole-Green cycles:

```python
from torchgamlss import CGControl

cg_model = GAMLSS.from_formula(
    Gamma(),
    {
        "mu": "y ~ x + C(group) + pb(age) + offset(mu_offset)",
        "sigma": "~ z + offset(sigma_offset)",
    },
    data,
)
result = cg_model.fit_cg_data(data, control=CGControl())
```

CG supports the same `pb()` smoothing-parameter modes as RS. See
[`CG.md`](CG.md).

`fit_data()` provides the corresponding Torch L-BFGS path:

```python
result = model.fit_data(data)
```

As in the tensor API, L-BFGS requires fixed smoothing parameters. Use
`fit_rs_data()` or `fit_cg_data()` for automatic ML, target-EDF, GAIC, or GCV
selection.

For larger fixed-lambda models, the formula API can use bounded-intermediate
Adam updates:

```python
import torch
from torchgamlss import MiniBatchControl

result = model.fit_minibatch_data(
    data,
    validation_data=validation_data,
    weights="weight",
    control=MiniBatchControl(
        batch_size=2048,
        epochs=100,
        learning_rate_decay=0.99,
    ),
    generator=torch.Generator().manual_seed(2026),
)
```

See [`MINIBATCH.md`](MINIBATCH.md) for objective scaling, convergence,
holdout early stopping, CPU/CUDA benchmarking, and reproducibility. The same
column selectors for weights and neural inputs are applied to the training
and validation frames.

## Prediction

The stored formula encodings are applied to new tabular data:

```python
parameters = model.predict_data(new_data)
eta = model.predict_data(new_data, type="link")
terms = model.predict_data(new_data, type="terms")
```

For fitted parametric models, `inference_data()` applies the same encodings
and returns joint coefficient covariance and Wald inference:

```python
inference = model.inference_data(data, weights="weight")
table = inference.to_dataframe()
```

Models with smooth terms support explicit conditional inference for their
linear coefficients. Pass `conditional_on_smooths=True`, the fitted residual
degrees of freedom, and see [`INFERENCE.md`](INFERENCE.md) for its scope.

Conditional inference for the fitted smooth contributions uses the same
stored formula encodings:

```python
import torch

smooth_curves = model.smooth_inference_data(data, weights="weight")
mu_x = smooth_curves["mu"]["x"]
mu_x_table = mu_x.to_dataframe()
band = mu_x.simultaneous_confidence_band(
    generator=torch.Generator().manual_seed(2026),
)
```

`mu_x.covariance_matrix` contains the full within-curve covariance. The
default intervals in `mu_x_table` are pointwise; `band.confidence_intervals`
contains a simulation-based simultaneous band at the same confidence level.

Several stored formula smooths can share one analytic fixed-lambda covariance:

```python
joint_analytic = model.smooth_joint_inference_data(
    data,
    weights="weight",
    new_data=new_data,
)
mu_sigma_covariance = joint_analytic.covariance_block(
    ("mu", "x"),
    ("sigma", "z"),
)
joint_analytic_bands = joint_analytic.simultaneous_confidence_bands(
    generator=torch.Generator().manual_seed(2026),
)
```

This includes cross-term and cross-parameter covariance. It conditions on the
fitted smoothing parameters.

To include smoothing-selection variability, use a seeded parametric bootstrap.
Each replicate reuses the stored formulas and refits the complete model:

```python
bootstrap = model.smooth_bootstrap_data(
    data,
    weights="weight",
    new_data=new_data,
    replicates=999,
    algorithm="rs",
    generator=torch.Generator().manual_seed(2026),
)
mu_x_bootstrap = bootstrap["mu"]["x"]
```

For empirical covariance that also propagates smoothing selection, call
`smooth_joint_bootstrap_data()` to preserve alignment across refits and
calibrate max-|t| bands over all selected curves.

Response-scale centiles reuse every stored parameter formula:

```python
centiles = model.predict_centiles_data(
    new_data,
    centiles=[3, 10, 50, 90, 97],
)
centile_bootstrap = model.centile_bootstrap_data(
    data,
    centiles=[3, 10, 50, 90, 97],
    new_data=new_data,
    replicates=999,
    generator=torch.Generator().manual_seed(2026),
)
```

The original fitted model is not changed. See
[`INFERENCE.md`](INFERENCE.md) for intervals, failed-refit accounting, and the
fixed-design interpretation.

Diagnostics and quantile residuals use the same stored encodings:

```python
diagnostics = model.diagnostics_data(data, weights="weight")
residuals = model.quantile_residuals_data(data)
```

Discrete residual randomization accepts a seeded `torch.Generator` or a
column name through `uniforms=`. See [`DIAGNOSTICS.md`](DIAGNOSTICS.md).

The new data need not contain the response, but must contain every predictor,
smooth covariate, and offset referenced by the formulas. Categorical levels
that were absent during construction are rejected because silently treating
them as the reference category would change the model.

The generated column names and response name are available through:

```python
model.formula_column_names
model.formula_response_name
```

`prepare_formula_data()` exposes the generated tensors as `FormulaData` for
advanced workflows that need to mix formula and low-level APIs.

Formula models may also attach neural predictors. Neural features stay
explicit instead of being silently mixed into the conventional design:

```python
model = GAMLSS.from_formula(
    Normal(),
    {"mu": "y ~ age", "sigma": "~ 1"},
    data,
    neural_predictors={"mu": MLPPredictor(2, (32, 32))},
)
result = model.fit_minibatch_data(
    data,
    neural_inputs={"mu": ["sensor_1", "sensor_2"]},
)
```

The same mapping is supplied to `predict_data()` for new data. See
[`NEURAL.md`](NEURAL.md).

A backbone shared by multiple distribution parameters instead receives one
common matrix:

```python
model = GAMLSS.from_formula(
    Normal(),
    {"mu": "y ~ age", "sigma": "~ age"},
    data,
    shared_predictor=SharedMLPPredictor(
        3,
        ("mu", "sigma"),
        (32, 32),
    ),
)
result = model.fit_minibatch_data(
    data,
    shared_input=["sensor_1", "sensor_2", "sensor_3"],
)
```

See [`SHARED.md`](SHARED.md) for custom shared modules and identifiability.

## P-splines

`pb()` contributes its covariate to the linear design and attaches a named
P-spline to the same parameter, matching the decomposition used by R
`gamlss::pb()`.

```python
"y ~ pb(x)"
"y ~ pb(x, smoothing_parameter=12)"
"y ~ pb(x, df=3, name='trend')"
"y ~ pb(x, method='GAIC', k=2)"
```

The canonical keyword options are:

- `smoothing_parameter`;
- `degrees_of_freedom`;
- `initial_smoothing_parameter`;
- `smoothing_method`;
- `criterion_penalty`;
- `intervals`, `degree`, and `penalty_order`;
- `name`, which identifies the smooth contribution.

For familiarity, `lambda_`, `df`, `method`, `k`, and `inter` are accepted as
aliases. Python reserves the word `lambda`, hence the trailing underscore in
`lambda_`.

`pb()` and `offset()` must currently be standalone additive terms. Smooth
interactions, transformed smooth covariates, multiple offsets for one
parameter, and missing-value row dropping are deliberately rejected.

## Formula semantics

Standard numerical terms, interactions, intercept control, transformations,
and categorical encoding such as `C(group)` follow Formulaic's
[formula grammar](https://matthewwardrop.github.io/formulaic/latest/guides/grammar/).
TorchGAMLSS requires every generated value to be finite and numeric.
