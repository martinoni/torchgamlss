"""Benchmark TorchGAMLSS mini-batch fitting on CPU or CUDA."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from typing import Any

import torch

from torchgamlss import GAMLSS, MiniBatchControl, Normal


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--learning-rate-decay", type=float, default=0.98)
    parser.add_argument("--evaluation-frequency", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compare-full-batch", action="store_true")
    parser.add_argument("--full-max-iter", type=int, default=50)
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if rows < 2:
        raise ValueError("rows must be at least 2")
    if features < 1:
        raise ValueError("features must be at least 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    covariates = torch.randn(
        (rows, features),
        dtype=dtype,
        generator=generator,
    )
    coefficients = torch.linspace(
        -0.6,
        0.8,
        features,
        dtype=dtype,
    )
    location = 0.4 + covariates @ coefficients
    scale = 0.55
    response = location + scale * torch.randn(
        rows,
        dtype=dtype,
        generator=generator,
    )
    design_matrices = {
        "mu": torch.column_stack(
            (torch.ones(rows, dtype=dtype), covariates)
        ).to(device),
        "sigma": torch.ones((rows, 1), dtype=dtype, device=device),
    }
    return response.to(device), design_matrices


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    device = _validated_device(arguments.device)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.float64
    if arguments.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)
        torch.cuda.reset_peak_memory_stats(device)

    response, design_matrices = _synthetic_problem(
        rows=arguments.rows,
        features=arguments.features,
        dtype=dtype,
        device=device,
        seed=arguments.seed,
    )
    model = GAMLSS(
        Normal(),
        {"mu": arguments.features + 1, "sigma": 1},
        dtype=dtype,
        device=device,
    )
    full_batch_model = copy.deepcopy(model) if arguments.compare_full_batch else None
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
        design_matrices,
        control=control,
        generator=torch.Generator(device="cpu").manual_seed(arguments.seed),
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
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
        "batch_size": result.batch_size,
        "epochs": result.epochs,
        "updates": result.updates,
        "elapsed_seconds": elapsed,
        "training_rows_per_second": arguments.rows * result.epochs / elapsed,
        "negative_log_likelihood_per_row": (
            result.negative_log_likelihood / arguments.rows
        ),
        "penalized_objective": result.penalized_objective,
        "gradient_max": result.gradient_max,
        "stop_reason": result.stop_reason,
        "initial_learning_rate": result.learning_rate,
        "final_learning_rate": result.final_learning_rate,
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": arguments.deterministic,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }

    if full_batch_model is not None:
        _synchronize(device)
        full_started = time.perf_counter()
        full_result = full_batch_model.fit(
            response,
            design_matrices,
            max_iter=arguments.full_max_iter,
        )
        _synchronize(device)
        full_elapsed = time.perf_counter() - full_started
        report["full_batch"] = {
            "max_iter": arguments.full_max_iter,
            "elapsed_seconds": full_elapsed,
            "negative_log_likelihood_per_row": (
                full_result.negative_log_likelihood / arguments.rows
            ),
            "gradient_max": full_result.gradient_max,
            "converged": full_result.converged,
            "speed_ratio_full_over_minibatch": (
                full_elapsed / elapsed if elapsed > 0 else math.nan
            ),
        }
    return report


def main() -> None:
    report = run_benchmark(_arguments())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
