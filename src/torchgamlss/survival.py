"""Conditional survival-function predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchgamlss.families import Family


@dataclass(frozen=True)
class SurvivalPrediction:
    """Survival, hazard, and cumulative-hazard curves."""

    family: str
    times: Tensor
    survival: Tensor
    hazard: Tensor
    cumulative_hazard: Tensor


def survival_prediction(
    family: Family,
    times: Any,
    parameters: Mapping[str, Tensor],
) -> SurvivalPrediction:
    """Evaluate event-time functions over one shared time grid."""
    missing = set(family.parameter_names).difference(parameters)
    extra = set(parameters).difference(family.parameter_names)
    if missing or extra:
        raise ValueError(
            "Survival parameters do not match family parameters: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if family.is_discrete:
        raise NotImplementedError(
            f"survival prediction is not implemented for the discrete "
            f"{family.name} family"
        )

    reference = parameters[family.parameter_names[0]]
    if not isinstance(reference, Tensor) or not reference.is_floating_point():
        raise ValueError("survival parameters must be floating-point tensors")
    parameter_values = []
    for parameter in family.parameter_names:
        value = parameters[parameter]
        if (
            not isinstance(value, Tensor)
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "survival parameters must be finite tensors with one common "
                "dtype and device"
            )
        parameter_values.append(value)
    try:
        broadcast_values = torch.broadcast_tensors(*parameter_values)
    except RuntimeError as error:
        raise ValueError("survival parameters cannot be broadcast together") from error

    try:
        time_tensor = torch.as_tensor(
            times,
            dtype=reference.dtype,
            device=reference.device,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "times must be convertible to a floating-point tensor"
        ) from error
    if (
        time_tensor.ndim != 1
        or time_tensor.numel() < 1
        or not torch.isfinite(time_tensor).all()
    ):
        raise ValueError("times must be a non-empty finite vector")

    batch_shape = broadcast_values[0].shape
    output_shape = batch_shape + (time_tensor.numel(),)
    evaluation_times = time_tensor.reshape(
        (1,) * len(batch_shape) + (time_tensor.numel(),)
    ).expand(output_shape)
    expanded_parameters = {
        parameter: value.unsqueeze(-1).expand(output_shape).reshape(-1)
        for parameter, value in zip(
            family.parameter_names,
            broadcast_values,
            strict=True,
        )
    }
    flat_times = evaluation_times.reshape(-1)
    survival = family.survival(flat_times, expanded_parameters).reshape(output_shape)
    hazard = family.hazard(flat_times, expanded_parameters).reshape(output_shape)
    cumulative_hazard = family.cumulative_hazard(
        flat_times,
        expanded_parameters,
    ).reshape(output_shape)
    for label, value in (
        ("survival", survival),
        ("hazard", hazard),
        ("cumulative hazard", cumulative_hazard),
    ):
        if value.shape != output_shape or not torch.isfinite(value).all():
            raise RuntimeError(
                f"{family.name} {label} evaluation returned invalid values"
            )

    return SurvivalPrediction(
        family=family.name,
        times=time_tensor.detach().clone(),
        survival=survival,
        hazard=hazard,
        cumulative_hazard=cumulative_hazard,
    )
