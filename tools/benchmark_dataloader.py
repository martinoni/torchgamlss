"""Benchmark bounded-memory DataLoader fitting on synthetic streaming data."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from torchgamlss import (
    GAMLSS,
    MiniBatchControl,
    Normal,
    SharedMLPPredictor,
)


class SyntheticBatchStream(IterableDataset):
    """Deterministic synthetic batches generated without resident full data."""

    def __init__(
        self,
        *,
        rows: int,
        features: int,
        batch_size: int,
        dtype: torch.dtype,
        seed: int,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.features = features
        self.batch_size = batch_size
        self.dtype = dtype
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        batch_count = math.ceil(self.rows / self.batch_size)
        for batch_index in range(worker_id, batch_count, worker_count):
            start = batch_index * self.batch_size
            rows = min(self.batch_size, self.rows - start)
            generator = torch.Generator().manual_seed(
                self.seed + 104_729 * batch_index
            )
            features = torch.randn(
                (rows, self.features),
                dtype=self.dtype,
                generator=generator,
            )
            location = 0.35 + 1.1 * features[:, 0]
            if self.features > 1:
                location = location + 0.45 * torch.sin(features[:, 1])
                log_scale = -1.0 + 0.2 * torch.tanh(features[:, 1])
            else:
                log_scale = torch.full_like(location, -1.0)
            response = (
                location
                + torch.exp(log_scale)
                * torch.randn(
                    rows,
                    dtype=self.dtype,
                    generator=generator,
                )
            )
            yield {
                "response": response,
                "design_matrices": {
                    "mu": torch.ones((rows, 1), dtype=self.dtype),
                    "sigma": torch.ones((rows, 1), dtype=self.dtype),
                },
                "shared_input": features,
            }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--validation-rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--learning-rate-decay", type=float, default=0.98)
    parser.add_argument("--evaluation-frequency", type=int, default=1)
    parser.add_argument("--validation-patience", type=int, default=4)
    parser.add_argument("--validation-minimum-delta", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-frequency", type=int, default=1)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Torch installation cannot use it"
        )
    return torch.device(requested)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loader(
    stream: SyntheticBatchStream,
    *,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        stream,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.rows < 1 or arguments.validation_rows < 0:
        raise ValueError("row counts must be non-negative with training rows positive")
    if arguments.features < 1 or arguments.batch_size < 1:
        raise ValueError("features and batch_size must be positive")
    if arguments.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    device = _device(arguments.device)
    dtype = getattr(torch, arguments.dtype)
    torch.manual_seed(arguments.seed)
    torch.use_deterministic_algorithms(arguments.deterministic)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    training_loader = _loader(
        SyntheticBatchStream(
            rows=arguments.rows,
            features=arguments.features,
            batch_size=arguments.batch_size,
            dtype=dtype,
            seed=arguments.seed,
        ),
        num_workers=arguments.num_workers,
        pin_memory=arguments.pin_memory,
    )
    validation_loader = None
    if arguments.validation_rows > 0:
        validation_loader = _loader(
            SyntheticBatchStream(
                rows=arguments.validation_rows,
                features=arguments.features,
                batch_size=arguments.batch_size,
                dtype=dtype,
                seed=arguments.seed + 1,
            ),
            num_workers=arguments.num_workers,
            pin_memory=arguments.pin_memory,
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
    control = MiniBatchControl(
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        learning_rate_decay=arguments.learning_rate_decay,
        minimum_epochs=(
            arguments.epochs
            if arguments.minimum_epochs is None
            else arguments.minimum_epochs
        ),
        patience=1_000_000,
        evaluation_frequency=arguments.evaluation_frequency,
        validation_patience=arguments.validation_patience,
        validation_minimum_delta=arguments.validation_minimum_delta,
    )

    _synchronize(device)
    started = time.perf_counter()
    result = model.fit_minibatch_loader(
        training_loader,
        validation_loader=validation_loader,
        control=control,
        non_blocking=arguments.pin_memory and device.type == "cuda",
        checkpoint_path=arguments.checkpoint_path,
        checkpoint_frequency=arguments.checkpoint_frequency,
        resume_from=arguments.resume_from,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    checkpoint_file = (
        arguments.checkpoint_path or arguments.resume_from
    )

    return {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": device.type,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or "CPU"
        ),
        "dtype": arguments.dtype,
        "rows": arguments.rows,
        "validation_rows": arguments.validation_rows,
        "features": arguments.features,
        "batch_size": result.batch_size,
        "epochs": result.epochs,
        "updates": result.updates,
        "num_workers": arguments.num_workers,
        "pin_memory": arguments.pin_memory,
        "checkpoint_enabled": checkpoint_file is not None,
        "checkpoint_bytes": (
            checkpoint_file.stat().st_size
            if checkpoint_file is not None
            and checkpoint_file.is_file()
            else None
        ),
        "resumed": arguments.resume_from is not None,
        "elapsed_seconds": elapsed,
        "training_rows_per_second": (
            arguments.rows * result.epochs / elapsed
        ),
        "negative_log_likelihood_per_row": (
            result.negative_log_likelihood / arguments.rows
        ),
        "validation_negative_log_likelihood": (
            result.validation_negative_log_likelihood
        ),
        "best_validation_loss": result.best_validation_loss,
        "best_epoch": result.best_epoch,
        "restored_best_parameters": result.restored_best_parameters,
        "gradient_max": result.gradient_max,
        "stop_reason": result.stop_reason,
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
