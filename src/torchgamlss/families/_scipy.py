"""CPU CDF bridge for non-differentiable distribution diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor


def scipy_call(
    reference: Tensor,
    function: Callable[..., Any],
    *arguments: Tensor,
) -> Tensor:
    """Evaluate a SciPy function and restore the reference dtype and device."""
    numpy_arguments = [
        argument.detach().to(device="cpu", dtype=torch.float64).numpy()
        for argument in arguments
    ]
    values = np.asarray(function(*numpy_arguments))
    return torch.as_tensor(
        values.copy(),
        dtype=reference.dtype,
        device=reference.device,
    )


def scipy_cdf(
    reference: Tensor,
    function: Callable[..., Any],
    *arguments: Tensor,
) -> Tensor:
    """Evaluate a SciPy CDF and restore the reference dtype and device."""
    return scipy_call(reference, function, *arguments)
