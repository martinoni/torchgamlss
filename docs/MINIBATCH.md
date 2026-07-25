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
    control=MiniBatchControl(batch_size=2048, epochs=100),
    generator=torch.Generator().manual_seed(2026),
)
```

The returned `MiniBatchFitResult` contains the weighted negative
log-likelihood, penalized mean objective, number of epochs and updates, exact
final full-gradient maximum, evaluated objective history, initial and final
learning rates, and the stopping reason.

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

## Scope

The current implementation bounds model intermediates, including neural
activations, but the input tensors still reside in memory. Streaming
`Dataset`/`DataLoader` input is separate future work. RS and CG remain the
reference paths for strict translation parity; L-BFGS and mini-batch Adam
support neural predictors. Inference routines may also construct full-data
information or design objects and should be assessed separately for very
large samples.
