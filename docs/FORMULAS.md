# Formulas and tabular data

TorchGAMLSS uses
[`formulaic`](https://matthewwardrop.github.io/formulaic/) to compile
Wilkinson formulas into design matrices. The fitted encodings are retained so
categorical contrasts and column order remain identical when predicting new
data.

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

See [`INFERENCE.md`](INFERENCE.md).

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
