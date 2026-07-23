"""Classical fitting algorithms translated from the R GAMLSS implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class RSControl:
    """Controls for the linear Rigby-Stasinopoulos fitting algorithm."""

    outer_tolerance: float = 1e-3
    max_outer_iterations: int = 20
    inner_tolerance: float = 1e-3
    max_inner_iterations: int = 50
    step: float = 1.0
    autostep: bool = True
    deviance_tolerance: float = float("inf")

    def __post_init__(self) -> None:
        if self.outer_tolerance <= 0 or self.inner_tolerance <= 0:
            raise ValueError("RS tolerances must be positive")
        if self.max_outer_iterations < 1 or self.max_inner_iterations < 1:
            raise ValueError("RS iteration limits must be at least 1")
        if not 0 < self.step <= 1:
            raise ValueError("RS step must be in the interval (0, 1]")
        if self.deviance_tolerance < 0:
            raise ValueError("deviance_tolerance must be non-negative")


@dataclass(frozen=True)
class RSFitResult:
    """Summary of a parametric RS fit."""

    global_deviance: float
    outer_iterations: int
    inner_iterations: Mapping[str, int]
    converged: bool
    deviance_history: tuple[float, ...]

    @property
    def negative_log_likelihood(self) -> float:
        return self.global_deviance / 2.0


def fit_rs(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    control: RSControl | None = None,
) -> RSFitResult:
    """Fit unpenalized linear predictors using the R GAMLSS RS equations."""
    control = control or RSControl()
    _validate_rs_inputs(model, response, design_matrices)
    case_weights = model._validated_weights(response, weights)
    parameter_offsets = _parameter_offsets(model, response, offsets)
    parameters = {
        name: value.detach().clone()
        for name, value in model.family.initial_parameters(response).items()
    }
    coefficients = {
        name: coefficient.detach().clone()
        for name, coefficient in model.coefficients.items()
    }
    inner_iterations = {name: 0 for name in model.family.parameter_names}

    global_deviance = _global_deviance(model, response, parameters, case_weights)
    history = [float(global_deviance)]
    converged = False
    outer_iterations = 0

    with torch.no_grad():
        for outer_iteration in range(1, control.max_outer_iterations + 1):
            old_global_deviance = global_deviance
            for parameter in model.family.parameter_names:
                coefficient, fitted_parameter, iterations = _fit_parameter(
                    model,
                    parameter,
                    response,
                    design_matrices[parameter],
                    parameter_offsets[parameter],
                    case_weights,
                    parameters,
                    control,
                )
                coefficients[parameter] = coefficient
                parameters[parameter] = fitted_parameter
                inner_iterations[parameter] += iterations

            global_deviance = _global_deviance(
                model, response, parameters, case_weights
            )
            history.append(float(global_deviance))
            outer_iterations = outer_iteration

            if (
                outer_iteration > 1
                and global_deviance > old_global_deviance + control.deviance_tolerance
            ):
                raise RuntimeError("global deviance increased during RS fitting")
            if abs(float(old_global_deviance - global_deviance)) < (
                control.outer_tolerance
            ):
                converged = True
                break

        for parameter, coefficient in coefficients.items():
            model.coefficients[parameter].copy_(coefficient)

    return RSFitResult(
        global_deviance=float(global_deviance),
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        converged=converged,
        deviance_history=tuple(history),
    )


def _fit_parameter(
    model: GAMLSS,
    parameter: str,
    response: Tensor,
    design_matrix: Tensor,
    offset: Tensor,
    case_weights: Tensor,
    parameters: dict[str, Tensor],
    control: RSControl,
) -> tuple[Tensor, Tensor, int]:
    link = model.family.links[parameter]
    eta = link(parameters[parameter])
    linear_predictor = eta - offset
    deviance = _global_deviance(model, response, parameters, case_weights)
    coefficient: Tensor | None = None

    for inner_iteration in range(1, control.max_inner_iterations + 1):
        old_deviance = deviance
        old_linear_predictor = linear_predictor
        old_coefficient = coefficient
        working_response, working_weights = _working_values(
            model, parameter, response, parameters, eta
        )
        working_response = working_response - offset
        raw_coefficient = _weighted_least_squares(
            design_matrix,
            working_response,
            working_weights * case_weights,
        )

        if inner_iteration == 1:
            coefficient = raw_coefficient
        else:
            assert old_coefficient is not None
            coefficient = (
                control.step * raw_coefficient + (1.0 - control.step) * old_coefficient
            )
        linear_predictor = design_matrix @ coefficient
        eta = linear_predictor + offset
        fitted_parameter = link.inverse(eta)
        parameters[parameter] = fitted_parameter
        deviance = _global_deviance(model, response, parameters, case_weights)

        if control.autostep and deviance > old_deviance and inner_iteration >= 2:
            for _ in range(5):
                coefficient = (coefficient + old_coefficient) / 2.0
                linear_predictor = design_matrix @ coefficient
                eta = linear_predictor + offset
                fitted_parameter = link.inverse(eta)
                parameters[parameter] = fitted_parameter
                deviance = _global_deviance(model, response, parameters, case_weights)
                if old_deviance - deviance > control.inner_tolerance:
                    break

        if abs(float(old_deviance - deviance)) <= control.inner_tolerance:
            return coefficient, fitted_parameter, inner_iteration

        if not torch.isfinite(deviance):
            raise FloatingPointError("global deviance is not finite during RS fitting")

        if inner_iteration > 1 and torch.equal(linear_predictor, old_linear_predictor):
            return coefficient, fitted_parameter, inner_iteration

    return coefficient, fitted_parameter, control.max_inner_iterations


def _working_values(
    model: GAMLSS,
    parameter: str,
    response: Tensor,
    parameters: Mapping[str, Tensor],
    eta: Tensor,
) -> tuple[Tensor, Tensor]:
    score = model.family.score(response, parameters)[parameter]
    second = model.family.expected_second_derivatives(response, parameters)[
        (parameter, parameter)
    ]
    second = second.clamp(max=-1e-15)
    inverse_link_derivative = model.family.links[parameter].inverse_derivative(eta)
    derivative = inverse_link_derivative.reciprocal()
    working_weights = -(second / derivative.square())
    working_weights = working_weights.clamp(min=1e-10, max=1e10)
    working_response = eta + score / (derivative * working_weights)
    return working_response, working_weights


def _weighted_least_squares(
    design_matrix: Tensor, response: Tensor, weights: Tensor
) -> Tensor:
    square_root_weights = weights.sqrt()
    weighted_design = design_matrix * square_root_weights.unsqueeze(-1)
    weighted_response = response * square_root_weights
    result = torch.linalg.lstsq(weighted_design, weighted_response)
    if result.rank.numel() and int(result.rank) < design_matrix.shape[1]:
        raise ValueError("RS design matrix is rank deficient")
    return result.solution


def _global_deviance(
    model: GAMLSS,
    response: Tensor,
    parameters: Mapping[str, Tensor],
    weights: Tensor,
) -> Tensor:
    return -2.0 * (model.family.log_prob(response, parameters) * weights).sum()


def _parameter_offsets(
    model: GAMLSS,
    response: Tensor,
    offsets: Mapping[str, Tensor] | None,
) -> dict[str, Tensor]:
    offsets = offsets or {}
    extra = set(offsets).difference(model.family.parameter_names)
    if extra:
        raise ValueError(f"Offsets contain unknown parameters: {sorted(extra)}")
    result = {}
    for parameter in model.family.parameter_names:
        if parameter not in offsets:
            result[parameter] = torch.zeros_like(response)
            continue
        if (
            offsets[parameter].dtype != response.dtype
            or offsets[parameter].device != response.device
        ):
            raise ValueError(
                f"offset for {parameter!r} must match response dtype and device"
            )
        try:
            result[parameter] = torch.broadcast_to(offsets[parameter], response.shape)
        except RuntimeError as error:
            raise ValueError(
                f"offset for {parameter!r} is not broadcastable to the response"
            ) from error
        if not torch.isfinite(result[parameter]).all():
            raise ValueError(f"offset for {parameter!r} must be finite")
    return result


def _validate_rs_inputs(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
) -> None:
    if response.ndim != 1:
        raise ValueError("RS fitting currently requires a one-dimensional response")
    if not torch.isfinite(response).all():
        raise ValueError("RS response must be finite")
    model_parameter = next(model.parameters())
    if response.dtype != model_parameter.dtype or response.device != (
        model_parameter.device
    ):
        raise ValueError("RS response dtype and device must match the model")
    expected = set(model.family.parameter_names)
    received = set(design_matrices)
    if expected != received:
        raise ValueError(
            "Design matrices do not match family parameters: "
            f"missing={sorted(expected - received)}, "
            f"extra={sorted(received - expected)}"
        )
    for parameter, design_matrix in design_matrices.items():
        if design_matrix.ndim != 2 or design_matrix.shape[0] != response.numel():
            raise ValueError(
                f"design matrix for {parameter!r} must have one row per observation"
            )
        if not torch.isfinite(design_matrix).all():
            raise ValueError(f"design matrix for {parameter!r} must be finite")
        if design_matrix.dtype != response.dtype or design_matrix.device != (
            response.device
        ):
            raise ValueError(
                f"design matrix for {parameter!r} must match response dtype and device"
            )
    model.linear_predictors(design_matrices)
