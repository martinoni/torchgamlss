# Shared neural representations

A shared predictor evaluates one backbone once per batch and routes its latent
representation through separate distribution-parameter heads:

```text
h_i = B(z_i; phi)
g_ik = H_k(h_i; psi_k)

eta_ik =
  x_ik' beta_k
  + sum_j f_jk(x_ij)
  + q_k(v_ik)
  + g_ik
  + offset_ik.
```

`q_k` denotes an optional neural predictor exclusive to parameter `k`;
`g_ik` is the contribution from the shared backbone. Both are unconstrained
link-scale contributions. The family inverse links still enforce valid
response parameters.

This architecture is useful when the same covariates contain information
about multiple aspects of a conditional distribution. For a Normal response,
for example, one representation can feed separate `mu` and `sigma` heads.

## Standard shared MLP

`SharedMLPPredictor` provides a tabular backbone and one linear head per named
parameter:

```python
import torch
from torchgamlss import (
    GAMLSS,
    MiniBatchControl,
    Normal,
    SharedMLPPredictor,
)

model = GAMLSS(
    Normal(),
    {"mu": 1, "sigma": 1},
    shared_predictor=SharedMLPPredictor(
        input_size=8,
        parameter_names=("mu", "sigma"),
        hidden_sizes=(64, 64),
        activation="silu",
    ),
    dtype=torch.float32,
    device="cuda",
)

result = model.fit_minibatch(
    response,
    design_matrices,
    shared_input=features,
    control=MiniBatchControl(batch_size=8192, epochs=100),
    generator=torch.Generator(device="cpu").manual_seed(2026),
)
```

The parameter names are inferred from `SharedMLPPredictor.parameter_names`.
The module is registered as `model.shared_predictor`; its backbone and heads
participate in `state_dict()`, `.to()`, automatic differentiation, L-BFGS,
and mini-batch Adam.

`hidden_sizes=()` uses the input itself as the representation and fits one
linear head per parameter. `silu`, `relu`, and `tanh` activations and optional
dropout are available.

During mini-batch fitting, the shared module is in training mode for Adam
updates and evaluation mode for complete-objective and final-gradient
diagnostics. This gives Dropout and BatchNorm their usual Torch semantics.

## Custom shared modules

An arbitrary module must return a mapping with exactly one item per declared
parameter:

```python
class MySharedPredictor(torch.nn.Module):
    def forward(self, inputs):
        representation = self.backbone(inputs)
        return {
            "mu": self.mu_head(representation).squeeze(-1),
            "sigma": self.sigma_head(representation).squeeze(-1),
        }


model = GAMLSS(
    Normal(),
    {"mu": 1, "sigma": 1},
    shared_predictor=MySharedPredictor(),
    shared_parameters=("mu", "sigma"),
)
```

Each output may have shape `(observations,)` or `(observations, 1)`. Inputs
and outputs must be finite and match the model dtype and device. Exact output
keys are checked so a head cannot silently update the wrong distribution
parameter.

## Formula data and prediction

A formula model receives one common feature tensor through `shared_input`:

```python
model = GAMLSS.from_formula(
    Normal(),
    {
        "mu": "y ~ age + pb(time, smoothing_parameter=10)",
        "sigma": "~ age",
    },
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
    control=control,
)

prediction = model.predict_data(
    new_data,
    shared_input=["sensor_1", "sensor_2", "sensor_3"],
)
```

A single column name is promoted to a one-feature matrix. A list preserves
column order. Low-level prediction, quantiles, centiles, likelihood,
distribution construction, diagnostics, and quantile residuals all accept
`shared_input`.

On `type="terms"`, `TermContributions.shared` contains the contribution from
the corresponding head. Parameters without a shared head receive a zero
vector. The shared module is evaluated once per prediction or optimization
batch, not once per head.

## Composition and identifiability

Linear terms, P-splines, an exclusive `neural_predictors` module, a shared
head, and an offset may all contribute to the same parameter. This flexibility
also creates a statistical identifiability issue: if two components can
represent the same function, their sum may be well determined while the
individual decomposition is not.

This is particularly relevant when:

- the same feature is present in a linear/smooth term and the network;
- the conventional design contains an intercept and a neural head has a bias;
- exclusive and shared networks receive overlapping features.

Prediction and likelihood optimization remain valid, but component-level
interpretation requires constraints such as centering, orthogonalization,
removing redundant biases, or a scientifically fixed architecture. Automatic
orthogonalization is not yet implemented. `TermContributions` reports the
literal fitted decomposition and must not be interpreted as a unique
attribution without such constraints.

RS, CG, classical parametric bootstraps, and analytic Wald/smooth inference
reject shared predictors. Effective degrees of freedom for a shared neural
model is not automatically defined, so information-criterion diagnostics
require an explicit `degrees_of_freedom`.

## CUDA benchmark

Run:

```bash
python tools/benchmark_shared.py \
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

One local measurement on 2026-07-25 used an RTX 4090 and Torch 2.11.0+cu128.
The `mu`/`sigma` model contained 4,868 trainable parameters versus 9,604 for
two equivalent independent MLPs, a 49.3% reduction. Twenty epochs over one
million rows and 2,460 Adam updates took 30.40 seconds, corresponding to about
658 thousand training rows per second, with 595 MB peak allocated CUDA
memory. Location MSE was `0.000237` and log-scale MSE was `0.000126`. This is
a reproducibility record, not a performance guarantee.

## Current boundary

The input tensor still resides in memory, although intermediate activations
are batch-bounded. Streaming `Dataset`/`DataLoader` support, automatic
orthogonalization of structured and neural terms, validation splits, and
early stopping on out-of-sample likelihood remain future work.
