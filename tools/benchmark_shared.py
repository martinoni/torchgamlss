"""Benchmark a shared-backbone TorchGAMLSS model on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from typing import Any

import torch

from torchgamlss import (
    GAMLSS,
    MiniBatchControl,
    Normal,
    SharedMLPPredictor,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--learning-rate-decay", type=float, default=0.98)
    parser.add_argument("--evaluation-frequency", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _validated_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _synthetic_problem(
    *,
    rows: int,
    features: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if rows < 2:
        raise ValueError("rows must be at least 2")
    if features < 1:
        raise ValueError("features must be at least 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(
        (rows, features),
        dtype=dtype,
        generator=generator,
    )
    weights = torch.linspace(-0.8, 0.9, features, dtype=dtype)
    latent = inputs @ weights / math.sqrt(features)
    location = 0.3 + torch.sin(latent) + 0.1 * inputs[:, 0].square()
    log_scale = -1.1 + 0.25 * torch.cos(latent)
    scale = log_scale.exp()
    response = location + scale * torch.randn(
        rows,
        dtype=dtype,
        generator=generator,
    )
    designs = {
        "mu": torch.ones((rows, 1), dtype=dtype, device=device),
        "sigma": torch.ones((rows, 1), dtype=dtype, device=device),
    }
    return (
        response.to(device),
        designs,
        inputs.to(device),
        location.to(device),
        log_scale.to(device),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _independent_parameter_count(
    input_size: int,
    hidden_size: int,
    hidden_layers: int,
    outputs: int,
) -> int:
    layer_sizes = [
        input_size,
        *([hidden_size] * hidden_layers),
        1,
    ]
    one_network = sum(
        (input_width + 1) * output_width
        for input_width, output_width in zip(
            layer_sizes,
            layer_sizes[1:],
        )
    )
    return outputs * one_network + outputs


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    device = _validated_device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.float64
    if arguments.hidden_size < 1:
        raise ValueError("hidden_size must be at least 1")
    if arguments.hidden_layers < 0:
        raise ValueError("hidden_layers must be non-negative")
    if arguments.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)
        torch.cuda.reset_peak_memory_stats(device)

    response, designs, shared_input, true_location, true_log_scale = (
        _synthetic_problem(
            rows=arguments.rows,
            features=arguments.features,
            dtype=dtype,
            device=device,
            seed=arguments.seed,
        )
    )
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            arguments.features,
            ("mu", "sigma"),
            (arguments.hidden_size,) * arguments.hidden_layers,
        ),
        dtype=dtype,
        device=device,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    independent_parameters = _independent_parameter_count(
        arguments.features,
        arguments.hidden_size,
        arguments.hidden_layers,
        outputs=2,
    )
    control = MiniBatchControl(
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        learning_rate_decay=arguments.learning_rate_decay,
        minimum_epochs=arguments.epochs,
        patience=arguments.epochs,
        evaluation_frequency=arguments.evaluation_frequency,
    )

    _synchronize(device)
    started = time.perf_counter()
    result = model.fit_minibatch(
        response,
        designs,
        shared_input=shared_input,
        control=control,
        generator=torch.Generator(device="cpu").manual_seed(arguments.seed),
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        fitted = model.predict(
            designs,
            shared_input=shared_input,
            type="response",
        )
        location_mse = float(
            (fitted["mu"] - true_location).square().mean()
        )
        log_scale_mse = float(
            (fitted["sigma"].log() - true_log_scale).square().mean()
        )

    return {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "CPU"
        ),
        "dtype": arguments.dtype,
        "rows": arguments.rows,
        "features": arguments.features,
        "hidden_size": arguments.hidden_size,
        "hidden_layers": arguments.hidden_layers,
        "shared_parameters": ["mu", "sigma"],
        "trainable_parameters": trainable_parameters,
        "independent_equivalent_parameters": independent_parameters,
        "parameter_reduction_fraction": (
            1.0 - trainable_parameters / independent_parameters
        ),
        "batch_size": result.batch_size,
        "epochs": result.epochs,
        "updates": result.updates,
        "elapsed_seconds": elapsed,
        "training_rows_per_second": arguments.rows * result.epochs / elapsed,
        "negative_log_likelihood_per_row": (
            result.negative_log_likelihood / arguments.rows
        ),
        "location_mean_squared_error": location_mse,
        "log_scale_mean_squared_error": log_scale_mse,
        "gradient_max": result.gradient_max,
        "stop_reason": result.stop_reason,
        "initial_learning_rate": result.learning_rate,
        "final_learning_rate": result.final_learning_rate,
        "deterministic_algorithms": arguments.deterministic,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    print(json.dumps(run_benchmark(_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
