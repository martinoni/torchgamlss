"""Reusable Torch-native predictor modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from torch import Tensor, nn

ActivationName = Literal["relu", "silu", "tanh"]


class MLPPredictor(nn.Module):
    """A compact multilayer perceptron with one link-scale output per row.

    Arbitrary ``nn.Module`` instances can be attached to :class:`GAMLSS`.
    This class is the standard convenience implementation for tabular inputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (32, 32),
        *,
        activation: ActivationName = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(input_size, bool)
            or not isinstance(input_size, int)
            or input_size < 1
        ):
            raise ValueError("input_size must be an integer of at least 1")
        normalized_hidden_sizes = tuple(hidden_sizes)
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in normalized_hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain only positive integers")
        activation_types: dict[ActivationName, type[nn.Module]] = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in activation_types:
            raise ValueError("activation must be one of: 'relu', 'silu', 'tanh'")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in normalized_hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation_types[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))

        self.input_size = input_size
        self.hidden_sizes = normalized_hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one unconstrained link-scale contribution per row."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(
                "MLPPredictor inputs must have shape "
                f"(observations, {self.input_size})"
            )
        return self.network(inputs).squeeze(-1)
