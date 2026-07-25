"""Reusable Torch-native predictor modules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from torch import Tensor, nn

ActivationName = Literal["relu", "silu", "tanh"]

_ACTIVATION_TYPES: dict[ActivationName, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


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
        normalized_hidden_sizes = _validated_architecture(
            hidden_sizes,
            activation,
            dropout,
        )

        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in normalized_hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(_ACTIVATION_TYPES[activation]())
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


class SharedMLPPredictor(nn.Module):
    """One MLP backbone with a separate link-scale head per parameter."""

    def __init__(
        self,
        input_size: int,
        parameter_names: Sequence[str],
        hidden_sizes: Sequence[int] = (64, 64),
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
        try:
            normalized_parameter_names = tuple(parameter_names)
        except TypeError as error:
            raise ValueError(
                "parameter_names must be a sequence of names"
            ) from error
        if (
            not normalized_parameter_names
            or any(
                not isinstance(name, str) or not name or "." in name
                for name in normalized_parameter_names
            )
            or len(set(normalized_parameter_names))
            != len(normalized_parameter_names)
        ):
            raise ValueError(
                "parameter_names must contain distinct non-empty names "
                "without dots"
            )
        normalized_hidden_sizes = _validated_architecture(
            hidden_sizes,
            activation,
            dropout,
        )

        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in normalized_hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(_ACTIVATION_TYPES[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size

        self.input_size = input_size
        self.parameter_names = normalized_parameter_names
        self.hidden_sizes = normalized_hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.heads = nn.ModuleDict(
            {
                parameter: nn.Linear(previous_size, 1)
                for parameter in normalized_parameter_names
            }
        )

    def forward(self, inputs: Tensor) -> Mapping[str, Tensor]:
        """Return one unconstrained link-scale contribution for every head."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(
                "SharedMLPPredictor inputs must have shape "
                f"(observations, {self.input_size})"
            )
        representation = self.backbone(inputs)
        return {
            parameter: head(representation).squeeze(-1)
            for parameter, head in self.heads.items()
        }


def _validated_architecture(
    hidden_sizes: Sequence[int],
    activation: ActivationName,
    dropout: float,
) -> tuple[int, ...]:
    try:
        normalized_hidden_sizes = tuple(hidden_sizes)
    except TypeError as error:
        raise ValueError(
            "hidden_sizes must be a sequence of positive integers"
        ) from error
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in normalized_hidden_sizes
    ):
        raise ValueError("hidden_sizes must contain only positive integers")
    if activation not in _ACTIVATION_TYPES:
        raise ValueError("activation must be one of: 'relu', 'silu', 'tanh'")
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return normalized_hidden_sizes
