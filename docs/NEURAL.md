# Neural predictors

TorchGAMLSS can attach a Torch module to selected distribution parameters. A
hybrid predictor has the form

```text
eta_k =
  X_k beta_k
  + sum_j f_jk(x_j)
  + g_k(z_k; phi_k)
  + offset_k,
```

where `g_k` is optional and returns one unconstrained contribution on the link
scale. The family link still maps `eta_k` to the valid parameter space. For
example, a neural contribution to Normal `sigma` is added on the log scale,
so the resulting scale remains positive.

This makes it possible to use a network for `mu`, conventional GAMLSS terms
for `sigma`, and no neural term for `nu` or `tau`. Linear coefficients,
P-splines, neural contributions, and offsets can coexist for the same
parameter.

## Standard MLP

`MLPPredictor` is a convenience network for tabular features:

```python
import torch
from torchgamlss import GAMLSS, MLPPredictor, Normal

model = GAMLSS(
    Normal(),
    {"mu": 1, "sigma": 1},
    neural_predictors={
        "mu": MLPPredictor(
            input_size=8,
            hidden_sizes=(64, 64),
            activation="silu",
        ),
    },
    dtype=torch.float32,
    device="cuda",
)

result = model.fit_minibatch(
    response,
    {
        "mu": torch.ones((len(response), 1), device="cuda"),
        "sigma": torch.ones((len(response), 1), device="cuda"),
    },
    neural_inputs={"mu": features},
    control=control,
    generator=torch.Generator(device="cpu").manual_seed(2026),
)
```

The intercept-only `mu` design above lets the network learn the nonlinear
part while retaining a conventional intercept. A richer design can retain
selected linear effects alongside the network.

`hidden_sizes=()` creates a single linear neural layer. Supported convenience
activations are `silu`, `relu`, and `tanh`; dropout is optional.

## Arbitrary Torch modules

Any `torch.nn.Module` can be supplied. It must accept the corresponding tensor
from `neural_inputs` and return either shape `(observations,)` or
`(observations, 1)`. Inputs and outputs must be finite and match the model
dtype and device.

```python
network = torch.nn.Sequential(
    torch.nn.Linear(8, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 1),
)
model = GAMLSS(
    family,
    design_sizes,
    neural_predictors={"mu": network},
    device="cuda",
    dtype=torch.float32,
)
```

Attached modules are registered in `model.neural_predictors` and converted to
the model dtype and device. Their parameters therefore participate in
`state_dict()`, `.to()`, automatic differentiation, L-BFGS, and Adam
mini-batch fitting.

## Formula data

Formula models retain the same explicit separation between conventional
terms and neural features:

```python
model = GAMLSS.from_formula(
    Normal(),
    {"mu": "y ~ age + pb(time, smoothing_parameter=10)", "sigma": "~ 1"},
    training_data,
    neural_predictors={"mu": MLPPredictor(3, (32, 32))},
    dtype=torch.float32,
    device="cuda",
)

result = model.fit_minibatch_data(
    training_data,
    neural_inputs={"mu": ["sensor_1", "sensor_2", "sensor_3"]},
    control=control,
)

prediction = model.predict_data(
    new_data,
    neural_inputs={"mu": ["sensor_1", "sensor_2", "sensor_3"]},
)
```

A single column name is converted to a two-dimensional one-feature tensor.
Column lists preserve the declared order. The mapping must contain exactly
the parameters that have configured neural predictors, which prevents a
network from silently receiving the wrong covariates.

## Prediction and diagnostics

`predict()`, `predict_quantiles()`, `predict_centiles()`,
`negative_log_likelihood()`, `distribution()`, `diagnostics()`, and quantile
residuals accept `neural_inputs`. On `type="terms"`,
`TermContributions.neural` contains the link-scale network contribution.

For networks with dropout or other train/evaluation-dependent layers, use
normal Torch conventions:

```python
model.train()
result = model.fit_minibatch(...)

model.eval()
prediction = model.predict(...)
```

Effective degrees of freedom for a neural network is not generally equal to a
classical GAMLSS spline EDF. Consequently, model diagnostics require an
explicit `degrees_of_freedom` for neural models. Analytic Wald/smooth
inference, RS, CG, and classical parametric bootstraps currently reject neural
predictors. Torch L-BFGS and mini-batch Adam are the supported fitting paths.

## CUDA and benchmark

The CUDA build of Torch must be installed; an NVIDIA driver alone does not
turn a CPU-only wheel into a CUDA wheel. On the development Windows machine,
the matching installation was:

```powershell
python -m pip install --force-reinstall `
  torch==2.11.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128
```

Choose a wheel supported by the installed driver and follow the current
PyTorch installation selector for other systems. Verify the environment
itself rather than inferring hardware support from one project:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Run the hybrid benchmark with:

```bash
python tools/benchmark_neural.py \
  --rows 1000000 \
  --features 8 \
  --hidden-size 64 \
  --hidden-layers 2 \
  --batch-size 8192 \
  --epochs 20 \
  --device cuda \
  --dtype float32 \
  --deterministic
```

One local measurement on 2026-07-25 used an RTX 4090, Torch 2.11.0+cu128,
one million rows, 4,803 trainable parameters, and 2,460 Adam updates. It took
19.02 seconds, processed approximately 1.05 million training rows per second,
used 587 MB of peak allocated CUDA memory, recovered `sigma=0.3503` from a
true value of `0.35`, and achieved location MSE `0.00060`. This is a
reproducibility record, not a performance guarantee.

## Current boundary

Exclusive networks use an input mapping owned by one distribution parameter.
For a single backbone with multiple parameter-specific heads, use
`SharedMLPPredictor` as described in [`SHARED.md`](SHARED.md). The tensor API
keeps inputs resident in memory; `fit_minibatch_loader()` can instead stream
neural features from CPU or an on-disk dataset. Fixed holdout validation and
out-of-sample early stopping are available through the mini-batch API in
[`MINIBATCH.md`](MINIBATCH.md).
