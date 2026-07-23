"""Wald inference for parametric GAMLSS coefficients."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class InferenceResult:
    """Joint coefficient covariance and Wald inference."""

    coefficient_names: tuple[str, ...]
    parameter_slices: Mapping[str, slice]
    estimates: Tensor
    covariance_matrix: Tensor
    standard_errors: Tensor
    statistics: Tensor
    p_values: Tensor
    confidence_intervals: Tensor
    degrees_of_freedom: float
    confidence_level: float

    @property
    def correlation_matrix(self) -> Tensor:
        """Return the coefficient correlation matrix."""
        scale = self.standard_errors.unsqueeze(1) * self.standard_errors.unsqueeze(0)
        return (self.covariance_matrix / scale).clamp(-1.0, 1.0)

    def by_parameter(self, values: Tensor) -> dict[str, Tensor]:
        """Split a coefficient-aligned tensor using the family parameters."""
        if values.ndim == 0 or values.shape[0] != self.estimates.numel():
            raise ValueError("values must have one row per inferred coefficient")
        return {
            parameter: values[parameter_slice]
            for parameter, parameter_slice in self.parameter_slices.items()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return the coefficient table as a pandas DataFrame."""
        values = torch.column_stack(
            (
                self.estimates,
                self.standard_errors,
                self.statistics,
                self.p_values,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            index=pd.Index(self.coefficient_names, name="coefficient"),
            columns=[
                "estimate",
                "standard_error",
                "statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            ],
        )


def coefficient_inference(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    confidence_level: float = 0.95,
    degrees_of_freedom: float | None = None,
) -> InferenceResult:
    """Compute full-Hessian covariance and t-based Wald inference."""
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    if any(model.smooth_terms[parameter] for parameter in model.family.parameter_names):
        raise ValueError(
            "coefficient inference currently supports parametric models without "
            "smooth terms"
        )

    model_parameter = next(model.parameters())
    if (
        response.ndim != 1
        or response.numel() < 1
        or response.dtype != model_parameter.dtype
        or response.device != model_parameter.device
        or not torch.isfinite(response).all()
    ):
        raise ValueError(
            "inference response must be a non-empty finite vector matching "
            "the model dtype and device"
        )

    contributions = model.term_contributions(design_matrices, offsets)
    observation_count = response.numel()
    if any(
        contribution.offset.numel() != observation_count
        for contribution in contributions.values()
    ):
        raise ValueError("design matrices must have one row per response")
    case_weights = model._validated_weights(response, weights)

    estimates = torch.cat(
        [
            model.coefficients[parameter].detach()
            for parameter in model.family.parameter_names
        ]
    )
    parameter_slices = _parameter_slices(model)
    coefficient_count = estimates.numel()
    if degrees_of_freedom is None:
        if torch.equal(case_weights, case_weights.round()):
            effective_observations = float(case_weights.sum())
        else:
            effective_observations = float((case_weights > 0).sum())
        degrees_of_freedom = effective_observations - coefficient_count
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError(
            "inference requires positive residual degrees of freedom "
            "(sum of weights minus coefficient count)"
        )

    def objective(flat_coefficients: Tensor) -> Tensor:
        predictors = {
            parameter: design_matrices[parameter]
            @ flat_coefficients[parameter_slices[parameter]]
            + contributions[parameter].offset
            for parameter in model.family.parameter_names
        }
        parameters = model.family.parameters_from_predictors(predictors)
        return -(model.family.log_prob(response, parameters) * case_weights).sum()

    hessian = torch.autograd.functional.hessian(objective, estimates)
    hessian = (hessian + hessian.mT) / 2.0
    if not torch.isfinite(hessian).all():
        raise RuntimeError("coefficient Hessian is not finite")
    factor, info = torch.linalg.cholesky_ex(hessian)
    if int(info) != 0:
        raise RuntimeError(
            "coefficient Hessian is not positive definite; covariance is unavailable"
        )
    covariance = torch.cholesky_inverse(factor)
    covariance = (covariance + covariance.mT) / 2.0
    standard_errors = torch.diagonal(covariance).sqrt()
    statistics = estimates / standard_errors

    statistic_values = statistics.detach().cpu().numpy()
    p_values_numpy = 2.0 * student_t.sf(
        np.abs(statistic_values),
        degrees_of_freedom,
    )
    critical_value = float(
        student_t.ppf(
            0.5 + confidence_level / 2.0,
            degrees_of_freedom,
        )
    )
    p_values = torch.as_tensor(
        p_values_numpy,
        dtype=estimates.dtype,
        device=estimates.device,
    )
    confidence_intervals = torch.column_stack(
        (
            estimates - critical_value * standard_errors,
            estimates + critical_value * standard_errors,
        )
    )
    return InferenceResult(
        coefficient_names=_coefficient_names(model),
        parameter_slices=parameter_slices,
        estimates=estimates.clone(),
        covariance_matrix=covariance.detach(),
        standard_errors=standard_errors.detach(),
        statistics=statistics.detach(),
        p_values=p_values,
        confidence_intervals=confidence_intervals.detach(),
        degrees_of_freedom=degrees_of_freedom,
        confidence_level=confidence_level,
    )


def _parameter_slices(model: GAMLSS) -> dict[str, slice]:
    result = {}
    start = 0
    for parameter in model.family.parameter_names:
        stop = start + model.coefficients[parameter].numel()
        result[parameter] = slice(start, stop)
        start = stop
    return result


def _coefficient_names(model: GAMLSS) -> tuple[str, ...]:
    if model._formula_encoder is None:
        return tuple(
            f"{parameter}[{index}]"
            for parameter in model.family.parameter_names
            for index in range(model.coefficients[parameter].numel())
        )
    return tuple(
        f"{parameter}.{column}"
        for parameter in model.family.parameter_names
        for column in model.formula_column_names[parameter]
    )
