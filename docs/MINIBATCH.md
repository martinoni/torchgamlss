# Mini-batch optimization and benchmarks

`fit_minibatch()` is the first Torch-native large-data fitting path. It uses
Adam updates while keeping intermediate predictors and spline bases bounded by
the selected batch size:

```python
import torch
from torchgamlss import MiniBatchControl

result = model.fit_minibatch(
    response,
    design_matrices,
    weights=weights,
    offsets=offsets,
    smooth_covariates=smooth_covariates,
    initial_parameters={"mu": 2.0, "sigma": 0.5},
    control=MiniBatchControl(
        batch_size=2048,
        epochs=100,
        learning_rate=0.01,
        learning_rate_decay=0.99,
        evaluation_frequency=5,
    ),
    generator=torch.Generator().manual_seed(2026),
)
```

Formula models expose the same optimizer:

```python
result = model.fit_minibatch_data(
    data,
    weights="weight",
    initial_parameters={"mu": "mu_start", "sigma": 0.5},
    control=MiniBatchControl(batch_size=2048, epochs=100),
    generator=torch.Generator().manual_seed(2026),
)
```

The returned `MiniBatchFitResult` contains the weighted negative
log-likelihood, penalized mean objective, number of epochs and updates, exact
final full-gradient maximum, evaluated objective history, initial and final
learning rates, and the stopping reason. When a validation holdout is
supplied, it also reports the holdout history, best epoch, best validation
loss, final holdout negative log-likelihood, and whether the best parameters
were restored. `updates` counts attempted batches and `skipped_updates`
reports the subset skipped by FP16 dynamic loss scaling.

## Starting values

`fit_minibatch()` accepts `initial_parameters` on the distribution-parameter
scale, consistently with RS and CG. Values may be scalars or vectors with one
value per training observation, and omitted parameters use the family
defaults. The formula method additionally accepts training-data column names.

The starts are transformed through the family links and projected onto the
first coefficient of each predictor. That first design column must therefore
be an explicit intercept containing only ones. The projection centers the
complete initial predictor, after accounting for existing non-intercept
coefficients, offsets, smooths, parameter-specific neural predictors, and
shared predictors. A varying start is matched in weighted mean on the link
scale; it is not necessarily reproduced row by row by an intercept-only
predictor.

This is especially important for positive-support families whose zero
coefficient model is invalid. For example:

```python
result = bccg_model.fit_minibatch(
    response,
    design_matrices,
    initial_parameters={"mu": 20.0, "sigma": 0.1, "nu": 0.5},
    control=control,
)
```

The streaming method cannot calculate response-wide family defaults or accept
one start per observation without retaining the population. Its
`initial_parameters` mapping must therefore provide one scalar for every
family parameter:

```python
result = bccg_model.fit_minibatch_loader(
    training_loader,
    initial_parameters={"mu": 20.0, "sigma": 0.1, "nu": 0.5},
    control=control,
)
```

Initialization makes one bounded-memory training-loader pass and preserves
the loader's exposed RNG state. Do not pass `initial_parameters` together with
`resume_from`: the checkpoint already contains the initialized fitted state
and is loaded before any likelihood evaluation.

## Objective scaling

For negative log-likelihood contributions `l_i`, case weights `w_i`, and
fixed smooth penalties, the optimized objective is

```text
J(theta) =
  [sum_i w_i l_i(theta) + 0.5 sum_j lambda_j ||D_j gamma_j||^2]
  / sum_i w_i.
```

For a uniformly sampled batch of `b` rows out of `n`, the likelihood gradient
uses

```text
n / (b sum_i w_i) * sum_{i in batch} w_i l_i(theta).
```

It is therefore an unbiased estimator of the complete weighted-mean
likelihood gradient. The fixed smooth penalty is included once in every
update, divided by the total case weight. Frequency weights, fractional case
weights, offsets, and batches containing only zero-weight observations are
supported.

Automatic ML, target-EDF, GAIC, or GCV smoothing selection is intentionally
not performed inside stochastic updates. Select lambdas with RS or CG first,
or construct fixed-lambda smooth terms.

## Convergence

The optimizer evaluates the complete objective in deterministic sequential
chunks at epoch zero and every `evaluation_frequency` epochs. This does not
materialize a full spline basis. `tolerance_change`, `patience`, and
`minimum_epochs` control loss-based early stopping.

After the last epoch, TorchGAMLSS makes one chunked pass to calculate the exact
gradient of the complete penalized objective. `gradient_max` is therefore not
a last-batch diagnostic. A run is marked converged when loss-change stopping
occurs or the final gradient satisfies `tolerance_gradient`.

Neural modules use training mode during Adam updates and evaluation mode
during complete-objective and final-gradient passes. This disables dropout
and prevents full-data diagnostics from updating BatchNorm running
statistics. The model's original train/evaluation mode is restored after a
successful fit.

Adam with a constant learning rate can keep moving around a likelihood
optimum. Set `learning_rate_decay` below one when final numerical precision is
important. For example, `0.99` multiplies the learning rate by `0.99` after
each epoch. Compare `learning_rate` with `final_learning_rate` in the result.

## Holdout validation and early stopping

Pass a fixed holdout through `MiniBatchValidationData`:

```python
from torchgamlss import (
    MiniBatchControl,
    MiniBatchValidationData,
)

validation = MiniBatchValidationData(
    response=validation_response,
    design_matrices=validation_design_matrices,
    weights=validation_weights,
    offsets=validation_offsets,
    smooth_covariates=validation_smooth_covariates,
    neural_inputs=validation_neural_inputs,
    shared_input=validation_shared_input,
)

result = model.fit_minibatch(
    response,
    design_matrices,
    weights=weights,
    neural_inputs=neural_inputs,
    shared_input=shared_input,
    validation=validation,
    control=MiniBatchControl(
        batch_size=2048,
        epochs=100,
        minimum_epochs=10,
        evaluation_frequency=2,
        validation_patience=5,
        validation_minimum_delta=1e-4,
        restore_best_parameters=True,
    ),
)
```

The holdout is evaluated in deterministic sequential chunks at epoch zero
and every `evaluation_frequency` epochs. `validation_history` stores its
weighted mean negative log-likelihood; `validation_negative_log_likelihood`
stores the final weighted sum to match the training NLL convention. The
holdout criterion intentionally excludes the smooth penalty.

An improvement must exceed `validation_minimum_delta`.
`validation_patience` counts consecutive holdout evaluations without such an
improvement, rather than raw epochs. With a holdout present, this criterion
replaces training-loss-change stopping; `minimum_epochs` still prevents an
early stop before the requested epoch.

By default, the complete `state_dict()` from the best validation epoch is
restored, including neural BatchNorm buffers. Epoch zero is eligible, so a
run that immediately worsens can recover the exact initial state.
`objective_history` and `validation_history` retain the optimization
trajectory, while the final objectives describe the restored model. Set
`restore_best_parameters=False` to retain the last trained state.

Formula models prepare a validation frame with the encodings learned from the
training frame:

```python
result = model.fit_minibatch_data(
    training_data,
    validation_data=validation_data,
    weights="weight",
    neural_inputs={"mu": ["sensor_1", "sensor_2"]},
    shared_input=["common_1", "common_2"],
    control=control,
)
```

Column selectors are applied to both frames. When training and validation
features already exist as separate tensors, use the low-level
`MiniBatchValidationData` interface instead.

## Streaming datasets and data loaders

`fit_minibatch_loader()` accepts a re-iterable Torch `DataLoader`. A
map-style `Dataset` can return one observation at a time and use the standard
Torch collation:

```python
import torch
from torch.utils.data import DataLoader, Dataset


class DistributionalDataset(Dataset):
    def __len__(self):
        return number_of_rows

    def __getitem__(self, index):
        return {
            "response": response_for(index),
            "design_matrices": {
                "mu": mu_design_row(index),
                "sigma": sigma_design_row(index),
            },
            "weights": weight_for(index),
            "offsets": {"mu": mu_offset_for(index)},
            "neural_inputs": {"mu": neural_features_for(index)},
            "shared_input": shared_features_for(index),
        }


training_loader = DataLoader(
    DistributionalDataset(),
    batch_size=8192,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator().manual_seed(2026),
)

result = model.fit_minibatch_loader(
    training_loader,
    validation_loader=validation_loader,
    control=MiniBatchControl(
        epochs=100,
        learning_rate=0.01,
        validation_patience=5,
    ),
    non_blocking=True,
)
```

Only `response` and `design_matrices` are required. The complete batch schema
also accepts `weights`, `offsets`, `smooth_covariates`, `neural_inputs`, and
`shared_input`, with the same nested mappings as `fit_minibatch()`. A custom
collate function or an `IterableDataset` may instead yield complete
pre-batched mappings.

The loader owns its batch size, sampling, collation, workers, and
shuffle generator. Consequently, `MiniBatchControl.batch_size`,
`MiniBatchControl.shuffle`, and the tensor API's `generator` do not configure
this path. `MiniBatchFitResult.batch_size` reports the largest observed
batch.

The implementation makes an initial complete objective pass to infer the exact
observation count and total case weight. Explicit initialization adds one
preceding centering pass; checkpoint resume instead uses a likelihood-free
metadata pass so positive-support models can load their saved state first.
Every later training, objective, and gradient pass must emit the same count
and total weight; changes raise an error. `drop_last=True` is rejected. A
batch may contain only zero-weight observations, but the complete loader must
have positive total weight.

This preserves the weighted objective without requiring duplicated metadata:

```text
n / (b sum_i w_i) * sum_{i in batch} w_i l_i(theta).
```

The usual unbiased-gradient argument requires batches sampled uniformly from
the loader population. Replacement, importance, distributed, or
data-dependent samplers can change that argument and are the user's
responsibility. Torch RNG states exposed by the loader and sampler are
preserved around diagnostic passes, so changing `evaluation_frequency` does
not silently change subsequent shuffled index order. Random transformations
inside persistent worker processes cannot be rewound; datasets used for exact
diagnostics should emit a stable population on every pass.

CPU batches are transferred to the model device one at a time. Their dtype
must already match the model. Use `pin_memory=True` in the loader together
with `non_blocking=True` for asynchronous-capable CPU-to-CUDA transfers.
Multi-worker `IterableDataset` implementations must shard their stream so
workers do not duplicate observations.

A validation loader uses the same schema and remains streaming. The training
and validation loaders must both be finite and re-iterable because objective
history, early stopping, best-state restoration, and the exact final gradient
require repeated passes. This is bounded-memory optimization, not a one-pass
online estimator.

## Checkpoint and resume

Long loader fits can write an atomic checkpoint after complete epochs:

```python
result = model.fit_minibatch_loader(
    training_loader,
    validation_loader=validation_loader,
    control=MiniBatchControl(epochs=100),
    checkpoint_path="checkpoints/model.pt",
    checkpoint_frequency=5,
)
```

If the process stops after epoch 37, for example, the checkpoint from epoch 35
remains valid. Resume with the same model architecture and loaders:

```python
result = model.fit_minibatch_loader(
    training_loader,
    validation_loader=validation_loader,
    control=MiniBatchControl(epochs=100),
    resume_from="checkpoints/model.pt",
)
```

`control.epochs` is the total target, not a number of additional epochs. It
may be increased when resuming. Optimizer and stopping controls must otherwise
match; `batch_size` and `shuffle` remain loader-owned and are ignored for
compatibility checks.

The checkpoint contains:

- the complete model, Adam, learning-rate scheduler, and AMP scaler states;
- completed epochs, attempted and skipped update counts, and objective and
  validation histories;
- early-stopping counters, the best epoch, and the best model state;
- exact training and validation population metadata;
- CPU, model-device CUDA, and exposed loader/sampler RNG states.

Model and optimizer tensors are stored on CPU. The format is a versioned
tensor-and-primitive dictionary loaded with `weights_only=True`, and each save
uses a temporary file followed by an atomic replacement. A failed write
therefore does not replace the preceding valid checkpoint.

Resume requires the same family parameters, architecture, dtype, CPU/CUDA
device type, validation-loader presence, loader population, and generator
configuration. The generator's saved state replaces its construction seed.
Random state held privately inside persistent worker processes cannot be
restored, so exact continuation still requires stable dataset output.

Checkpoint format version 2 adds AMP state. Version 1 checkpoints written by
earlier TorchGAMLSS revisions remain resumable when `amp_dtype=None`; an AMP
run cannot be resumed from a version 1 checkpoint because no loss-scaler state
was stored.

Checkpoints are written after the scheduler and any scheduled validation
evaluation, but before final best-state restoration. An interrupted run
therefore continues its optimization trajectory rather than restarting from
the restored inference model. An interruption within an epoch replays that
epoch from the last completed checkpoint.

## CPU, CUDA, and reproducibility

The model, response, designs, offsets, weights, and smooth covariates must be
on the same device. A formula model can be created directly on CUDA:

```python
model = GAMLSS.from_formula(
    family,
    formulas,
    data,
    dtype=torch.float32,
    device="cuda",
)
```

Batch permutations are generated on the CPU. A seeded CPU
`torch.Generator` therefore gives the same row order independently of the
training device:

```python
generator = torch.Generator(device="cpu").manual_seed(2026)
```

For stricter repeatability, seed Torch before model construction and enable
deterministic algorithms:

```python
torch.manual_seed(2026)
torch.use_deterministic_algorithms(True)
```

Float64 remains the recommended dtype for tight R parity. Float32 is generally
the relevant throughput dtype on GPUs. Exact cross-device equality is not
promised because kernels, hardware, and Torch versions can differ.

## CUDA automatic mixed precision

Both `fit_minibatch()` and `fit_minibatch_loader()` can use automatic mixed
precision for the stochastic updates of a CUDA float32 model:

```python
control = MiniBatchControl(
    batch_size=8192,
    epochs=20,
    amp_dtype="float16",
)
result = model.fit_minibatch(
    response,
    design_matrices,
    neural_inputs=neural_inputs,
    shared_input=shared_input,
    control=control,
)
```

`amp_dtype` accepts `"float16"`, `"bfloat16"`, or `None`. The model and its
inputs remain float32; Torch selects eligible CUDA operations for the lower
throughput dtype. Float16 uses dynamic `GradScaler` scaling. Gradient clipping,
when requested, occurs after gradients are unscaled. Bfloat16 uses autocast
without loss scaling and requires a CUDA device with bfloat16 support.

The FP16 scaler starts at `256`, rather than Torch's general-purpose default
of `65536`. Heavy-tailed GAMLSS likelihoods can have finite gradients in the
hundreds even after weighted-mean normalization; the lower starting scale
avoids spending several initial batches reducing an unnecessarily large
scale. Scaling remains dynamic and a genuine overflow still skips the update.
An epoch containing no applied updates does not advance the learning-rate
scheduler.

Only Adam update batches run under autocast. Complete training-objective
evaluations, holdout validation, smooth penalties, and the exact final
full-gradient diagnostic remain float32. This keeps stopping and reported
objectives on the existing diagnostic path. Mixed-precision optimization is
not expected to reproduce an FP32 trajectory bit for bit, so statistical
agreement should be assessed with a problem-appropriate tolerance.

`result.updates - result.skipped_updates` is the number of applied optimizer
updates. A nonzero skipped count means the float16 scaler detected an overflow,
skipped that Adam step, and reduced its scale. Loader checkpoints preserve the
scaler and skipped count, allowing exact epoch-boundary continuation on the
same CUDA setup.

### Family tail-stability matrix

`tests/test_amp_stability.py` exercises Gamma, negative-binomial type I,
BCCG, BCT, BCPE, TF, and PE with two parameter regimes per family and response
quantiles from `0.0001` through `0.9999`. The resulting responses extend from
approximately `-3.44e3` to `1.22e4`. Each mode starts from the same float32
two-layer shared MLP and completes eight epochs over 32 attempted batches.

The deterministic RTX 4090 run on 2026-07-25 produced:

| Family | Response maximum | FP16 relative NLL difference | BF16 relative NLL difference | FP16 skips |
| --- | ---: | ---: | ---: | ---: |
| GA | 1,193 | `1.64e-2` | `2.97e-5` | 1 |
| NBI | 1,214 | `7.01e-7` | `1.04e-5` | 0 |
| BCCG | 4,226 | `2.36e-6` | `1.41e-5` | 0 |
| BCT | 12,172 | `4.10e-7` | `1.99e-5` | 0 |
| BCPE | 8,701 | `4.11e-5` | `4.50e-5` | 0 |
| TF | 3,484 | `3.99e-6` | `9.21e-5` | 0 |
| PE | 1,553 | `4.07e-7` | `1.85e-5` | 0 |

The first five rows were recorded on 2026-07-25; TF was added and measured on
the same RTX 4090 on 2026-07-27, and PE on 2026-07-28. The relative difference
is measured against the final FP32 negative log-likelihood from the same
initialization and batch order. All objectives, full gradients, and predicted
parameters remained finite. The Gamma FP16 run correctly skipped one
overflowing update and stayed within the committed 3% stress tolerance. This
is a stability regression matrix, not a claim that such a short fit has
converged or that every possible parameter extreme is safe.

Run the matrix on a CUDA development machine with:

```bash
python -m pytest tests/test_amp_stability.py
```

## Benchmark command

The benchmark produces machine-readable JSON and excludes synthetic-data
generation from the timed region:

```bash
python tools/benchmark_minibatch.py \
  --rows 100000 \
  --features 8 \
  --batch-size 2048 \
  --epochs 20 \
  --device cpu \
  --dtype float32 \
  --deterministic
```

Use `--device cuda` for a GPU run. `--compare-full-batch` additionally runs
the existing L-BFGS baseline on the same generated data:

```bash
python tools/benchmark_minibatch.py \
  --rows 20000 \
  --features 8 \
  --batch-size 2048 \
  --epochs 20 \
  --device cuda \
  --compare-full-batch
```

The shared-representation benchmark also accepts a generated holdout:

```bash
python tools/benchmark_shared.py \
  --rows 500000 \
  --validation-rows 100000 \
  --features 8 \
  --hidden-size 64 \
  --hidden-layers 2 \
  --batch-size 8192 \
  --epochs 30 \
  --minimum-epochs 5 \
  --evaluation-frequency 1 \
  --validation-patience 4 \
  --validation-minimum-delta 0.00005 \
  --device cuda \
  --dtype float32 \
  --deterministic
```

The streaming benchmark generates deterministic pre-batched observations on
demand through an `IterableDataset`:

```bash
python tools/benchmark_dataloader.py \
  --rows 500000 \
  --validation-rows 100000 \
  --features 8 \
  --hidden-size 64 \
  --hidden-layers 2 \
  --batch-size 8192 \
  --epochs 20 \
  --minimum-epochs 5 \
  --evaluation-frequency 1 \
  --validation-patience 4 \
  --pin-memory \
  --checkpoint-path checkpoint.pt \
  --checkpoint-frequency 5 \
  --device cuda \
  --dtype float32 \
  --amp-dtype float16 \
  --deterministic
```

Omit `--amp-dtype` for the FP32 baseline, or use
`--amp-dtype bfloat16` on supported CUDA devices. The JSON report records the
requested AMP dtype and the number of scaler-skipped updates.

One local CPU measurement on 2026-07-25 used Torch 2.13.0, float32,
deterministic algorithms, 100,000 rows, eight covariates, batch size 2,048,
and 20 epochs. It completed in 1.247 seconds, corresponding to approximately
1.60 million training rows per second over 980 Adam updates. This is a
reproducibility record for that machine, not a performance guarantee.

The first environment contained a CPU-only Torch wheel even though the machine
has an RTX 4090. After installing Torch 2.11.0+cu128, the same benchmark path
was verified on CUDA with one million rows, eight covariates, batch size 8,192,
and 20 epochs. It completed in 7.594 seconds, corresponding to approximately
2.63 million training rows per second over 2,460 Adam updates, with 107 MB of
peak allocated CUDA memory. The benchmark reports its Torch build, device name,
and peak allocated memory so a CPU-only installation cannot be mistaken for
missing hardware.

The holdout-enabled shared benchmark was also run locally on 2026-07-25 with
the RTX 4090 and Torch 2.11.0+cu128. It used 500,000 training rows, 100,000
validation rows, two 64-unit hidden layers, batch size 8,192, and deterministic
float32 algorithms. Validation early stopping selected epoch 4, stopped at
epoch 8, and restored the best state. The run completed in 4.063 seconds,
processed approximately 985 thousand training rows per second, and used
306 MB of peak allocated CUDA memory. The final weighted mean validation NLL
was `0.536691`.

The streaming command above was run on the same RTX 4090 and Torch build. It
completed 20 epochs without checkpoints in 13.807 seconds, or approximately
724 thousand training rows per second, while generating all training and
validation observations on demand. Peak allocated CUDA memory was 31.2 MB.
The timing includes the initial scan, synthetic generation, holdout
evaluation, complete training objective at every epoch, final objective, and
exact final gradient; it is not an update-only throughput figure.

Writing a checkpoint every five epochs on the same deterministic problem took
13.943 seconds and produced a 101,299-byte final checkpoint. The fitted state
and all reported statistical values were identical to the run without
checkpointing.

AMP was measured separately on 2026-07-25 with the same RTX 4090 and Torch
build, 500,000 training rows, 100,000 validation rows, two 64-unit hidden
layers, batch size 8,192, and 20 deterministic epochs. FP32 took 15.121
seconds (661 thousand training rows per second), FP16 took 14.817 seconds
(675 thousand rows per second), and BF16 took 16.444 seconds (608 thousand
rows per second). All 1,240 FP16 updates were applied without a scaler skip.
The final training NLL per row was `0.423574`, `0.423597`, and `0.423402`,
respectively. Peak allocated memory was approximately 31.2 MB in all three
runs. This small streaming MLP is substantially affected by data generation
and full-precision diagnostic passes, so its modest FP16 gain and slower BF16
run must not be generalized to larger neural architectures.

## Scope

The tensor and formula paths keep their input tensors resident in memory.
`fit_minibatch_loader()` additionally bounds input residency to the active
and prefetched loader batches. Formula parsing itself is not performed
lazily; an on-disk dataset must produce encoded design rows using a stable
training schema. RS and CG remain the reference paths for strict translation
parity; L-BFGS and mini-batch Adam support neural predictors. Inference
routines may also construct full-data information or design objects and
should be assessed separately for very large samples.
