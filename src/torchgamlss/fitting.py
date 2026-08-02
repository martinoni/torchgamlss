"""Classical fitting algorithms translated from the R GAMLSS implementation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from torchgamlss.penalties import solve_penalized_least_squares
from torchgamlss.smooths import SmoothTerm

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


SmoothingParameterValue = float | tuple[float, ...]


@dataclass(frozen=True)
class CGControl:
    """Controls for the Cole-Green fitting algorithm."""

    outer_tolerance: float = 1e-3
    max_outer_iterations: int = 20
    inner_tolerance: float = 1e-3
    max_inner_iterations: int = 50
    backfitting_tolerance: float = 1e-3
    smoothing_tolerance: float = 1e-7
    max_smoothing_iterations: int = 50
    edf_tolerance: float = 1.220703125e-4
    max_edf_iterations: int = 1000
    criterion_tolerance: float = 1e-8
    max_criterion_iterations: int = 100
    mu_step: float = 1.0
    sigma_step: float = 1.0
    nu_step: float = 1.0
    tau_step: float = 1.0
    autostep: bool = True
    deviance_tolerance: float = float("inf")

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.outer_tolerance)
            or self.outer_tolerance <= 0
            or not math.isfinite(self.inner_tolerance)
            or self.inner_tolerance <= 0
            or not math.isfinite(self.backfitting_tolerance)
            or self.backfitting_tolerance <= 0
            or not math.isfinite(self.smoothing_tolerance)
            or self.smoothing_tolerance <= 0
            or not math.isfinite(self.edf_tolerance)
            or self.edf_tolerance <= 0
            or not math.isfinite(self.criterion_tolerance)
            or self.criterion_tolerance <= 0
        ):
            raise ValueError("CG tolerances must be positive")
        if (
            self.max_outer_iterations < 1
            or self.max_inner_iterations < 1
            or self.max_smoothing_iterations < 1
            or self.max_edf_iterations < 1
            or self.max_criterion_iterations < 1
        ):
            raise ValueError("CG iteration limits must be at least 1")
        if any(
            not 0 < step <= 1
            for step in (self.mu_step, self.sigma_step, self.nu_step, self.tau_step)
        ):
            raise ValueError("CG parameter steps must be in the interval (0, 1]")
        if math.isnan(self.deviance_tolerance) or self.deviance_tolerance < 0:
            raise ValueError("deviance_tolerance must be non-negative")


@dataclass(frozen=True)
class CGFitResult:
    """Summary of a linear or additive Cole-Green fit."""

    global_deviance: float
    outer_iterations: int
    inner_iterations: tuple[int, ...]
    converged: bool
    deviance_history: tuple[float, ...]
    backfitting_iterations: Mapping[str, int]
    smooth_effective_degrees_of_freedom: Mapping[str, Mapping[str, float]]
    smoothing_parameters: Mapping[
        str,
        Mapping[str, SmoothingParameterValue],
    ]
    smoothing_iterations: Mapping[str, Mapping[str, int]]
    parameter_effective_degrees_of_freedom: Mapping[str, float]
    effective_degrees_of_freedom: float

    @property
    def negative_log_likelihood(self) -> float:
        return self.global_deviance / 2.0


@dataclass(frozen=True)
class RSControl:
    """Controls for the Rigby-Stasinopoulos fitting algorithm."""

    outer_tolerance: float = 1e-3
    max_outer_iterations: int = 20
    inner_tolerance: float = 1e-3
    max_inner_iterations: int = 50
    backfitting_tolerance: float = 1e-3
    max_backfitting_iterations: int = 30
    smoothing_tolerance: float = 1e-7
    max_smoothing_iterations: int = 50
    edf_tolerance: float = 1.220703125e-4
    max_edf_iterations: int = 1000
    criterion_tolerance: float = 1e-8
    max_criterion_iterations: int = 100
    step: float = 1.0
    autostep: bool = True
    deviance_tolerance: float = float("inf")

    def __post_init__(self) -> None:
        if (
            self.outer_tolerance <= 0
            or self.inner_tolerance <= 0
            or self.backfitting_tolerance <= 0
            or self.smoothing_tolerance <= 0
            or self.edf_tolerance <= 0
            or self.criterion_tolerance <= 0
        ):
            raise ValueError("RS tolerances must be positive")
        if (
            self.max_outer_iterations < 1
            or self.max_inner_iterations < 1
            or self.max_backfitting_iterations < 1
            or self.max_smoothing_iterations < 1
            or self.max_edf_iterations < 1
            or self.max_criterion_iterations < 1
        ):
            raise ValueError("RS iteration limits must be at least 1")
        if not 0 < self.step <= 1:
            raise ValueError("RS step must be in the interval (0, 1]")
        if self.deviance_tolerance < 0:
            raise ValueError("deviance_tolerance must be non-negative")


@dataclass(frozen=True)
class RSFitResult:
    """Summary of a linear or additive RS fit."""

    global_deviance: float
    outer_iterations: int
    inner_iterations: Mapping[str, int]
    converged: bool
    deviance_history: tuple[float, ...]
    backfitting_iterations: Mapping[str, int]
    smooth_effective_degrees_of_freedom: Mapping[str, Mapping[str, float]]
    smoothing_parameters: Mapping[
        str,
        Mapping[str, SmoothingParameterValue],
    ]
    smoothing_iterations: Mapping[str, Mapping[str, int]]
    parameter_effective_degrees_of_freedom: Mapping[str, float]
    effective_degrees_of_freedom: float

    @property
    def negative_log_likelihood(self) -> float:
        return self.global_deviance / 2.0


@dataclass(frozen=True)
class _ParameterFitResult:
    coefficient: Tensor
    fitted_parameter: Tensor
    inner_iterations: int
    smooth_coefficients: dict[str, Tensor]
    backfitting_iterations: int
    smooth_edf: dict[str, float]
    smoothing_parameters: dict[str, SmoothingParameterValue]
    smoothing_iterations: dict[str, int]


@dataclass(frozen=True)
class _AdditiveFitResult:
    linear_coefficient: Tensor
    smooth_coefficients: dict[str, Tensor]
    iterations: int
    smooth_edf: dict[str, float]
    smoothing_parameters: dict[str, SmoothingParameterValue]
    smoothing_iterations: dict[str, int]


@dataclass(frozen=True)
class _SmoothFitResult:
    coefficient: Tensor
    smoothing_parameter: SmoothingParameterValue
    effective_degrees_of_freedom: float
    smoothing_iterations: int


def _term_smoothing_parameter_value(
    term: SmoothTerm,
) -> SmoothingParameterValue:
    values = term.smoothing_parameters
    return values[0] if len(values) == 1 else tuple(values)


def _set_term_smoothing_parameter_value(
    term: SmoothTerm,
    value: SmoothingParameterValue,
) -> None:
    if isinstance(value, tuple):
        if value != term.smoothing_parameters:
            raise RuntimeError(
                "multiply penalized term smoothing parameters are fixed "
                "during RS/CG fitting"
            )
        return
    term._set_fitted_smoothing_parameter(value)


def _linear_design_overlap(term: SmoothTerm) -> int:
    """Return smooth null-space dimensions duplicated in the linear block."""
    return term.penalty_nullity if len(term.penalty_matrices()) == 1 else 0


def fit_cg(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
    control: CGControl | None = None,
) -> CGFitResult:
    """Fit additive predictors using the R GAMLSS Cole-Green equations."""
    control = control or CGControl()
    _validate_classical_inputs(
        model,
        response,
        design_matrices,
        smooth_covariates,
        algorithm="CG",
    )
    case_weights = model._validated_weights(response, weights)
    parameter_offsets = _parameter_offsets(model, response, offsets)
    parameter_names = model.family.parameter_names
    parameters = {
        name: value.detach().clone()
        for name, value in model.family.initial_parameters(
            response, initial_parameters
        ).items()
    }
    coefficients = {
        name: coefficient.detach().clone()
        for name, coefficient in model.coefficients.items()
    }
    smooth_coefficients = {
        parameter: {
            name: term.coefficients.detach().clone()
            for name, term in model.smooth_terms[parameter].items()
        }
        for parameter in parameter_names
    }
    smoothing_parameters = {
        parameter: {
            name: _term_smoothing_parameter_value(term)
            for name, term in model.smooth_terms[parameter].items()
        }
        for parameter in parameter_names
    }
    backfitting_iterations = {name: 0 for name in parameter_names}
    smoothing_iterations: dict[str, dict[str, int]] = {
        parameter: {name: 0 for name in model.smooth_terms[parameter]}
        for parameter in parameter_names
    }
    smooth_edf: dict[str, dict[str, float]] = {name: {} for name in parameter_names}
    global_deviance = _global_deviance(model, response, parameters, case_weights)
    history = [float(global_deviance)]
    inner_iterations = []
    converged = False
    outer_iterations = 0

    with torch.no_grad():
        for outer_iteration in range(1, control.max_outer_iterations + 1):
            old_global_deviance = global_deviance
            old_coefficients = {
                name: value.detach().clone() for name, value in coefficients.items()
            }
            old_smooth_coefficients = {
                parameter: {
                    name: value.detach().clone()
                    for name, value in smooth_coefficients[parameter].items()
                }
                for parameter in parameter_names
            }
            old_eta = {
                name: model.family.links[name](parameters[name])
                for name in parameter_names
            }
            eta = {name: value.detach().clone() for name, value in old_eta.items()}
            scores = model.family.score(response, parameters)
            second = model.family.expected_second_derivatives(response, parameters)
            derivatives = {
                name: model.family.links[name]
                .inverse_derivative(old_eta[name])
                .reciprocal()
                for name in parameter_names
            }
            working_weights = {
                name: -second[(name, name)] / derivatives[name].square()
                for name in parameter_names
            }
            for name, values in working_weights.items():
                if not torch.isfinite(values).all() or (values <= 0).any():
                    raise FloatingPointError(
                        f"CG working weights for {name!r} must be finite and positive"
                    )
            working_responses = {
                name: old_eta[name]
                - parameter_offsets[name]
                + _cg_step(control, name)
                * scores[name]
                / (derivatives[name] * working_weights[name])
                for name in parameter_names
            }
            cross_weights = {}
            for index, left in enumerate(parameter_names):
                for right in parameter_names[index + 1 :]:
                    cross_second = _cross_derivative(second, left, right)
                    cross_weights[(left, right)] = -cross_second / (
                        derivatives[left] * derivatives[right]
                    )
            if any(not values.isfinite().all() for values in cross_weights.values()):
                raise FloatingPointError(
                    "CG cross-parameter working weights must be finite"
                )

            previous_inner_deviance = global_deviance + 1.0
            inner_deviance = global_deviance
            inner_iteration = 0
            while (
                abs(float(previous_inner_deviance - inner_deviance))
                > control.inner_tolerance
                and inner_iteration < control.max_inner_iterations
            ):
                previous_inner_deviance = inner_deviance
                for index, name in enumerate(parameter_names):
                    adjustment = torch.zeros_like(response)
                    for other_index, other in enumerate(parameter_names):
                        if other == name:
                            continue
                        pair = (name, other) if index < other_index else (other, name)
                        adjustment = adjustment + cross_weights[pair] * (
                            eta[other] - old_eta[other]
                        )
                    adjusted_response = (
                        working_responses[name] - adjustment / working_weights[name]
                    )
                    combined_weights = working_weights[name] * case_weights
                    smooth_terms = model.smooth_terms[name]
                    if smooth_terms:
                        additive_fit = _additive_fit(
                            design_matrices[name],
                            adjusted_response,
                            combined_weights,
                            smooth_terms,
                            (smooth_covariates or {}).get(name, {}),
                            smooth_coefficients[name],
                            smoothing_parameters[name],
                            control,
                            max_iterations=1,
                        )
                        coefficient = additive_fit.linear_coefficient
                        smooth_coefficients[name] = additive_fit.smooth_coefficients
                        smoothing_parameters[name] = additive_fit.smoothing_parameters
                        smooth_edf[name] = additive_fit.smooth_edf
                        backfitting_iterations[name] += additive_fit.iterations
                        for (
                            term_name,
                            iteration_count,
                        ) in additive_fit.smoothing_iterations.items():
                            smoothing_iterations[name][term_name] += iteration_count
                    else:
                        coefficient = _weighted_least_squares(
                            design_matrices[name],
                            adjusted_response,
                            combined_weights,
                        )
                    coefficients[name] = coefficient
                    eta[name] = (
                        _additive_predictor(
                            design_matrices[name],
                            coefficient,
                            smooth_terms,
                            (smooth_covariates or {}).get(name, {}),
                            smooth_coefficients[name],
                        )
                        + parameter_offsets[name]
                    )
                    parameters[name] = model.family.links[name].inverse(eta[name])

                inner_deviance = _global_deviance(
                    model,
                    response,
                    parameters,
                    case_weights,
                )
                inner_iteration += 1
                if not torch.isfinite(inner_deviance):
                    raise FloatingPointError(
                        "global deviance is not finite during CG fitting"
                    )
                if (
                    outer_iteration > 2
                    and inner_deviance
                    > previous_inner_deviance + control.deviance_tolerance
                ):
                    raise RuntimeError(
                        "global deviance increased during the inner CG loop"
                    )

            global_deviance = inner_deviance
            if (
                control.autostep
                and outer_iteration > 2
                and global_deviance > old_global_deviance
            ):
                for _ in range(5):
                    for name in parameter_names:
                        coefficients[name] = (
                            coefficients[name] + old_coefficients[name]
                        ) / 2.0
                        smooth_coefficients[name] = {
                            term_name: (
                                smooth_coefficients[name][term_name]
                                + old_smooth_coefficients[name][term_name]
                            )
                            / 2.0
                            for term_name in model.smooth_terms[name]
                        }
                        eta[name] = (
                            _additive_predictor(
                                design_matrices[name],
                                coefficients[name],
                                model.smooth_terms[name],
                                (smooth_covariates or {}).get(name, {}),
                                smooth_coefficients[name],
                            )
                            + parameter_offsets[name]
                        )
                        parameters[name] = model.family.links[name].inverse(eta[name])
                    global_deviance = _global_deviance(
                        model,
                        response,
                        parameters,
                        case_weights,
                    )
                    if global_deviance < old_global_deviance:
                        break

            history.append(float(global_deviance))
            inner_iterations.append(inner_iteration)
            outer_iterations = outer_iteration
            if not torch.isfinite(global_deviance):
                raise FloatingPointError(
                    "global deviance is not finite during CG fitting"
                )
            if (
                outer_iteration > 2
                and global_deviance > old_global_deviance + control.deviance_tolerance
            ):
                raise RuntimeError("global deviance increased during CG fitting")
            if abs(float(old_global_deviance - global_deviance)) < (
                control.outer_tolerance
            ):
                converged = True
                break

        for name, coefficient in coefficients.items():
            model.coefficients[name].copy_(coefficient)
            for term_name, term_coefficient in smooth_coefficients[name].items():
                model.smooth_terms[name][term_name].coefficients.copy_(term_coefficient)
                _set_term_smoothing_parameter_value(
                    model.smooth_terms[name][term_name],
                    smoothing_parameters[name][term_name],
                )

    parameter_edf = {
        parameter: float(model.coefficients[parameter].numel())
        + sum(
            smooth_edf[parameter][name] - _linear_design_overlap(term)
            for name, term in model.smooth_terms[parameter].items()
        )
        for parameter in parameter_names
    }
    return CGFitResult(
        global_deviance=float(global_deviance),
        outer_iterations=outer_iterations,
        inner_iterations=tuple(inner_iterations),
        converged=converged,
        deviance_history=tuple(history),
        backfitting_iterations=backfitting_iterations,
        smooth_effective_degrees_of_freedom=smooth_edf,
        smoothing_parameters=smoothing_parameters,
        smoothing_iterations=smoothing_iterations,
        parameter_effective_degrees_of_freedom=parameter_edf,
        effective_degrees_of_freedom=sum(parameter_edf.values()),
    )


def _cg_step(control: CGControl, parameter: str) -> float:
    try:
        return {
            "mu": control.mu_step,
            "sigma": control.sigma_step,
            "nu": control.nu_step,
            "tau": control.tau_step,
        }[parameter]
    except KeyError as error:
        raise NotImplementedError(
            f"CG does not define a step control for parameter {parameter!r}"
        ) from error


def _cross_derivative(
    derivatives: Mapping[tuple[str, str], Tensor],
    left: str,
    right: str,
) -> Tensor:
    if (left, right) in derivatives:
        return derivatives[(left, right)]
    if (right, left) in derivatives:
        return derivatives[(right, left)]
    raise KeyError(
        f"family does not provide the CG cross derivative for {left!r}, {right!r}"
    )


def fit_rs(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
    control: RSControl | None = None,
) -> RSFitResult:
    """Fit additive predictors using the R GAMLSS RS equations."""
    control = control or RSControl()
    _validate_classical_inputs(
        model,
        response,
        design_matrices,
        smooth_covariates,
        algorithm="RS",
    )
    case_weights = model._validated_weights(response, weights)
    parameter_offsets = _parameter_offsets(model, response, offsets)
    parameters = {
        name: value.detach().clone()
        for name, value in model.family.initial_parameters(
            response, initial_parameters
        ).items()
    }
    coefficients = {
        name: coefficient.detach().clone()
        for name, coefficient in model.coefficients.items()
    }
    smooth_coefficients = {
        parameter: {
            name: term.coefficients.detach().clone()
            for name, term in model.smooth_terms[parameter].items()
        }
        for parameter in model.family.parameter_names
    }
    smoothing_parameters = {
        parameter: {
            name: _term_smoothing_parameter_value(term)
            for name, term in model.smooth_terms[parameter].items()
        }
        for parameter in model.family.parameter_names
    }
    inner_iterations = {name: 0 for name in model.family.parameter_names}
    backfitting_iterations = {name: 0 for name in model.family.parameter_names}
    smoothing_iterations: dict[str, dict[str, int]] = {
        parameter: {name: 0 for name in model.smooth_terms[parameter]}
        for parameter in model.family.parameter_names
    }
    smooth_edf: dict[str, dict[str, float]] = {
        name: {} for name in model.family.parameter_names
    }

    global_deviance = _global_deviance(model, response, parameters, case_weights)
    history = [float(global_deviance)]
    converged = False
    outer_iterations = 0

    with torch.no_grad():
        for outer_iteration in range(1, control.max_outer_iterations + 1):
            old_global_deviance = global_deviance
            for parameter in model.family.parameter_names:
                parameter_fit = _fit_parameter(
                    model,
                    parameter,
                    response,
                    design_matrices[parameter],
                    parameter_offsets[parameter],
                    case_weights,
                    parameters,
                    model.smooth_terms[parameter],
                    (smooth_covariates or {}).get(parameter, {}),
                    smooth_coefficients[parameter],
                    smoothing_parameters[parameter],
                    control,
                )
                coefficients[parameter] = parameter_fit.coefficient
                smooth_coefficients[parameter] = parameter_fit.smooth_coefficients
                smoothing_parameters[parameter] = parameter_fit.smoothing_parameters
                parameters[parameter] = parameter_fit.fitted_parameter
                inner_iterations[parameter] += parameter_fit.inner_iterations
                backfitting_iterations[parameter] += (
                    parameter_fit.backfitting_iterations
                )
                smooth_edf[parameter] = parameter_fit.smooth_edf
                for (
                    term_name,
                    iteration_count,
                ) in parameter_fit.smoothing_iterations.items():
                    smoothing_iterations[parameter][term_name] += iteration_count

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
            for term_name, term_coefficient in smooth_coefficients[parameter].items():
                model.smooth_terms[parameter][term_name].coefficients.copy_(
                    term_coefficient
                )
                _set_term_smoothing_parameter_value(
                    model.smooth_terms[parameter][term_name],
                    smoothing_parameters[parameter][term_name]
                )

    parameter_edf = {
        parameter: float(model.coefficients[parameter].numel())
        + sum(
            smooth_edf[parameter][name] - _linear_design_overlap(term)
            for name, term in model.smooth_terms[parameter].items()
        )
        for parameter in model.family.parameter_names
    }
    return RSFitResult(
        global_deviance=float(global_deviance),
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        converged=converged,
        deviance_history=tuple(history),
        backfitting_iterations=backfitting_iterations,
        smooth_effective_degrees_of_freedom=smooth_edf,
        smoothing_parameters=smoothing_parameters,
        smoothing_iterations=smoothing_iterations,
        parameter_effective_degrees_of_freedom=parameter_edf,
        effective_degrees_of_freedom=sum(parameter_edf.values()),
    )


def _fit_parameter(
    model: GAMLSS,
    parameter: str,
    response: Tensor,
    design_matrix: Tensor,
    offset: Tensor,
    case_weights: Tensor,
    parameters: dict[str, Tensor],
    smooth_terms: Mapping[str, SmoothTerm],
    smooth_covariates: Mapping[str, Tensor],
    initial_smooth_coefficients: Mapping[str, Tensor],
    initial_smoothing_parameters: Mapping[str, SmoothingParameterValue],
    control: RSControl,
) -> _ParameterFitResult:
    link = model.family.links[parameter]
    eta = link(parameters[parameter])
    linear_predictor = eta - offset
    deviance = _global_deviance(model, response, parameters, case_weights)
    coefficient: Tensor | None = None
    smooth_coefficients = dict(initial_smooth_coefficients)
    smoothing_parameters = dict(initial_smoothing_parameters)
    backfitting_iterations = 0
    smoothing_iterations = {name: 0 for name in smooth_terms}
    smooth_edf: dict[str, float] = {}

    for inner_iteration in range(1, control.max_inner_iterations + 1):
        old_deviance = deviance
        old_linear_predictor = linear_predictor
        old_coefficient = coefficient
        old_smooth_coefficients = smooth_coefficients
        working_response, working_weights = _working_values(
            model, parameter, response, parameters, eta
        )
        working_response = working_response - offset
        combined_weights = working_weights * case_weights
        if smooth_terms:
            additive_fit = _additive_fit(
                design_matrix,
                working_response,
                combined_weights,
                smooth_terms,
                smooth_covariates,
                smooth_coefficients,
                smoothing_parameters,
                control,
            )
            raw_coefficient = additive_fit.linear_coefficient
            raw_smooth_coefficients = additive_fit.smooth_coefficients
            smooth_edf = additive_fit.smooth_edf
            smoothing_parameters = additive_fit.smoothing_parameters
            backfitting_iterations += additive_fit.iterations
            for name, iteration_count in additive_fit.smoothing_iterations.items():
                smoothing_iterations[name] += iteration_count
        else:
            raw_coefficient = _weighted_least_squares(
                design_matrix,
                working_response,
                combined_weights,
            )
            raw_smooth_coefficients = {}

        if inner_iteration == 1:
            coefficient = raw_coefficient
            smooth_coefficients = raw_smooth_coefficients
        else:
            assert old_coefficient is not None
            coefficient = (
                control.step * raw_coefficient + (1.0 - control.step) * old_coefficient
            )
            smooth_coefficients = {
                name: control.step * raw_smooth_coefficients[name]
                + (1.0 - control.step) * old_smooth_coefficients[name]
                for name in smooth_terms
            }
        linear_predictor = _additive_predictor(
            design_matrix,
            coefficient,
            smooth_terms,
            smooth_covariates,
            smooth_coefficients,
        )
        eta = linear_predictor + offset
        fitted_parameter = link.inverse(eta)
        parameters[parameter] = fitted_parameter
        deviance = _global_deviance(model, response, parameters, case_weights)

        if control.autostep and deviance > old_deviance and inner_iteration >= 2:
            for _ in range(5):
                coefficient = (coefficient + old_coefficient) / 2.0
                smooth_coefficients = {
                    name: (smooth_coefficients[name] + old_smooth_coefficients[name])
                    / 2.0
                    for name in smooth_terms
                }
                linear_predictor = _additive_predictor(
                    design_matrix,
                    coefficient,
                    smooth_terms,
                    smooth_covariates,
                    smooth_coefficients,
                )
                eta = linear_predictor + offset
                fitted_parameter = link.inverse(eta)
                parameters[parameter] = fitted_parameter
                deviance = _global_deviance(model, response, parameters, case_weights)
                if old_deviance - deviance > control.inner_tolerance:
                    break

        if abs(float(old_deviance - deviance)) <= control.inner_tolerance:
            return _ParameterFitResult(
                coefficient=coefficient,
                fitted_parameter=fitted_parameter,
                inner_iterations=inner_iteration,
                smooth_coefficients=smooth_coefficients,
                backfitting_iterations=backfitting_iterations,
                smooth_edf=smooth_edf,
                smoothing_parameters=smoothing_parameters,
                smoothing_iterations=smoothing_iterations,
            )

        if not torch.isfinite(deviance):
            raise FloatingPointError("global deviance is not finite during RS fitting")

        if inner_iteration > 1 and torch.equal(linear_predictor, old_linear_predictor):
            return _ParameterFitResult(
                coefficient=coefficient,
                fitted_parameter=fitted_parameter,
                inner_iterations=inner_iteration,
                smooth_coefficients=smooth_coefficients,
                backfitting_iterations=backfitting_iterations,
                smooth_edf=smooth_edf,
                smoothing_parameters=smoothing_parameters,
                smoothing_iterations=smoothing_iterations,
            )

    assert coefficient is not None
    return _ParameterFitResult(
        coefficient=coefficient,
        fitted_parameter=fitted_parameter,
        inner_iterations=control.max_inner_iterations,
        smooth_coefficients=smooth_coefficients,
        backfitting_iterations=backfitting_iterations,
        smooth_edf=smooth_edf,
        smoothing_parameters=smoothing_parameters,
        smoothing_iterations=smoothing_iterations,
    )


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
        raise ValueError("weighted least-squares design matrix is rank deficient")
    return result.solution


def _additive_fit(
    design_matrix: Tensor,
    response: Tensor,
    weights: Tensor,
    smooth_terms: Mapping[str, SmoothTerm],
    smooth_covariates: Mapping[str, Tensor],
    initial_coefficients: Mapping[str, Tensor],
    initial_smoothing_parameters: Mapping[str, SmoothingParameterValue],
    control: RSControl | CGControl,
    *,
    max_iterations: int | None = None,
) -> _AdditiveFitResult:
    """Alternate parametric and penalized terms as in ``additive.fit()``."""
    bases = {
        name: term.design(smooth_covariates[name])
        for name, term in smooth_terms.items()
    }
    constraints = {
        name: term.constraints(smooth_covariates[name])
        for name, term in smooth_terms.items()
    }
    coefficients = dict(initial_coefficients)
    smoothing_parameters = dict(initial_smoothing_parameters)
    smoothing_iterations = {name: 0 for name in smooth_terms}
    smooth_fits = {name: bases[name] @ coefficients[name] for name in smooth_terms}
    residuals = response - sum(smooth_fits.values(), torch.zeros_like(response))
    linear_fit = torch.zeros_like(response)
    linear_coefficient = torch.zeros(
        design_matrix.shape[1], dtype=response.dtype, device=response.device
    )
    effective_degrees_of_freedom: dict[str, float] = {}

    iteration_limit = (
        control.max_backfitting_iterations if max_iterations is None else max_iterations
    )
    for iteration in range(1, iteration_limit + 1):
        partial_response = residuals + linear_fit
        linear_coefficient = _weighted_least_squares(
            design_matrix, partial_response, weights
        )
        linear_fit = design_matrix @ linear_coefficient
        residuals = partial_response - linear_fit
        change = torch.zeros((), dtype=response.dtype, device=response.device)

        for name, term in smooth_terms.items():
            old_fit = smooth_fits[name]
            partial_response = residuals + old_fit
            smooth_fit = _fit_smooth_term(
                term,
                bases[name],
                partial_response,
                weights,
                smoothing_parameters[name],
                control,
                constraints=constraints[name],
            )
            coefficient = smooth_fit.coefficient
            fitted = bases[name] @ coefficient
            coefficients[name] = coefficient
            smoothing_parameters[name] = smooth_fit.smoothing_parameter
            smoothing_iterations[name] += smooth_fit.smoothing_iterations
            smooth_fits[name] = fitted
            residuals = partial_response - fitted
            change = change + (weights * (fitted - old_fit).square()).sum() / (
                weights.sum()
            )
            effective_degrees_of_freedom[name] = smooth_fit.effective_degrees_of_freedom

        smooth_sum = sum(smooth_fits.values(), torch.zeros_like(response))
        denominator = (weights * smooth_sum.square()).sum()
        if denominator <= 0:
            relative_change = 0.0 if change <= 0 else float("inf")
        else:
            relative_change = float(torch.sqrt(change / denominator))
        if relative_change <= control.backfitting_tolerance:
            return _AdditiveFitResult(
                linear_coefficient=linear_coefficient,
                smooth_coefficients=coefficients,
                iterations=iteration,
                smooth_edf=effective_degrees_of_freedom,
                smoothing_parameters=smoothing_parameters,
                smoothing_iterations=smoothing_iterations,
            )

    return _AdditiveFitResult(
        linear_coefficient=linear_coefficient,
        smooth_coefficients=coefficients,
        iterations=iteration_limit,
        smooth_edf=effective_degrees_of_freedom,
        smoothing_parameters=smoothing_parameters,
        smoothing_iterations=smoothing_iterations,
    )


def _fit_smooth_term(
    term: SmoothTerm,
    basis: Tensor,
    response: Tensor,
    weights: Tensor,
    starting_smoothing_parameter: SmoothingParameterValue,
    control: RSControl | CGControl,
    *,
    constraints: Tensor,
) -> _SmoothFitResult:
    penalties = term.penalty_matrices()
    if len(penalties) > 1:
        if term.estimates_smoothing_parameter:
            raise ValueError(
                "automatic smoothing selection for multiply penalized terms "
                "requires whole-model LAML"
            )
        if not isinstance(starting_smoothing_parameter, tuple):
            raise RuntimeError(
                "multiply penalized terms require one smoothing parameter "
                "per penalty"
            )
        fit = solve_penalized_least_squares(
            basis,
            response,
            weights,
            penalties,
            starting_smoothing_parameter,
            constraints=constraints,
        )
        return _SmoothFitResult(
            coefficient=fit.coefficients,
            smoothing_parameter=starting_smoothing_parameter,
            effective_degrees_of_freedom=float(
                fit.effective_degrees_of_freedom
            ),
            smoothing_iterations=0,
        )
    if isinstance(starting_smoothing_parameter, tuple):
        raise RuntimeError(
            "single-penalty terms require one scalar smoothing parameter"
        )
    if term.estimates_smoothing_parameter:
        if term.smoothing_method == "DF":
            target_edf = term.target_effective_degrees_of_freedom
            if target_edf is None:
                raise RuntimeError("EDF smoothing selection requires a target EDF")
            return _select_smoothing_parameter_for_edf(
                basis,
                response,
                weights,
                term.penalty_matrix(),
                target_edf,
                control,
            )
        if term.smoothing_method in {"GAIC", "GCV"}:
            return _select_smoothing_parameter_by_criterion(
                basis,
                response,
                weights,
                term.penalty_matrix(),
                method=term.smoothing_method,
                criterion_penalty=term.criterion_penalty,
                starting_smoothing_parameter=starting_smoothing_parameter,
                control=control,
            )
        if term.smoothing_method == "ML":
            if starting_smoothing_parameter <= 1e-7 or (
                starting_smoothing_parameter >= 1e7
            ):
                smoothing_parameter = min(max(starting_smoothing_parameter, 1e-7), 1e7)
            else:
                return _estimate_ml_smoothing_parameter(
                    basis,
                    response,
                    weights,
                    term.penalty_matrix(),
                    term.penalty_nullity,
                    starting_smoothing_parameter,
                    control,
                )
        else:
            raise NotImplementedError(
                f"Unsupported smoothing method: {term.smoothing_method!r}"
            )
    else:
        smoothing_parameter = starting_smoothing_parameter

    coefficient = _penalized_least_squares(
        basis,
        response,
        weights,
        smoothing_parameter,
        term.penalty_matrix(),
    )
    edf = _effective_degrees_of_freedom(
        basis, weights, smoothing_parameter, term.penalty_matrix()
    )
    return _SmoothFitResult(
        coefficient=coefficient,
        smoothing_parameter=smoothing_parameter,
        effective_degrees_of_freedom=float(edf),
        smoothing_iterations=0,
    )


def _select_smoothing_parameter_by_criterion(
    basis: Tensor,
    response: Tensor,
    weights: Tensor,
    penalty: Tensor,
    *,
    method: str,
    criterion_penalty: float,
    starting_smoothing_parameter: float,
    control: RSControl | CGControl,
) -> _SmoothFitResult:
    """Minimize the local GAIC or GCV criterion used by ``gamlss.pb()``."""

    def objective(log_smoothing_parameter: float) -> float:
        smoothing_parameter = math.exp(log_smoothing_parameter)
        coefficient = _penalized_least_squares(
            basis, response, weights, smoothing_parameter, penalty
        )
        fitted = basis @ coefficient
        edf = _effective_degrees_of_freedom(
            basis, weights, smoothing_parameter, penalty
        )
        residual_sum_of_squares = float((weights * (response - fitted).square()).sum())
        if method == "GAIC":
            return residual_sum_of_squares + criterion_penalty * float(edf)
        denominator = basis.shape[0] - criterion_penalty * float(edf)
        if denominator == 0:
            return float("inf")
        return basis.shape[0] * residual_sum_of_squares / denominator**2

    log_smoothing_parameter, iterations = _brent_minimize(
        objective,
        math.log(1e-7),
        math.log(1e7),
        initial=math.log(min(max(starting_smoothing_parameter, 1e-7), 1e7)),
        tolerance=control.criterion_tolerance,
        max_iterations=control.max_criterion_iterations,
    )
    smoothing_parameter = math.exp(log_smoothing_parameter)
    coefficient = _penalized_least_squares(
        basis, response, weights, smoothing_parameter, penalty
    )
    edf = _effective_degrees_of_freedom(basis, weights, smoothing_parameter, penalty)
    return _SmoothFitResult(
        coefficient=coefficient,
        smoothing_parameter=smoothing_parameter,
        effective_degrees_of_freedom=float(edf),
        smoothing_iterations=iterations,
    )


def _brent_minimize(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    initial: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, int]:
    """Bounded Brent minimization with a caller-supplied starting value."""
    golden_complement = (3.0 - math.sqrt(5.0)) / 2.0
    x = w = v = min(max(initial, lower), upper)
    x_value = w_value = v_value = function(x)
    step = previous_step = 0.0

    for iteration in range(1, max_iterations + 1):
        midpoint = (lower + upper) / 2.0
        local_tolerance = tolerance * abs(x) + 1e-12
        doubled_tolerance = 2.0 * local_tolerance
        if abs(x - midpoint) <= doubled_tolerance - (upper - lower) / 2.0:
            return x, iteration

        if abs(previous_step) > local_tolerance:
            first = (x - w) * (x_value - v_value)
            second = (x - v) * (x_value - w_value)
            numerator = (x - v) * second - (x - w) * first
            denominator = 2.0 * (second - first)
            if denominator > 0:
                numerator = -numerator
            denominator = abs(denominator)
            old_previous_step = previous_step
            previous_step = step
            if (
                abs(numerator) < abs(0.5 * denominator * old_previous_step)
                and numerator > denominator * (lower - x)
                and numerator < denominator * (upper - x)
            ):
                step = numerator / denominator
                candidate = x + step
                if candidate - lower < doubled_tolerance or (
                    upper - candidate < doubled_tolerance
                ):
                    step = math.copysign(local_tolerance, midpoint - x)
            else:
                previous_step = lower - x if x >= midpoint else upper - x
                step = golden_complement * previous_step
        else:
            previous_step = lower - x if x >= midpoint else upper - x
            step = golden_complement * previous_step

        candidate = (
            x + step
            if abs(step) >= local_tolerance
            else x + math.copysign(local_tolerance, step)
        )
        candidate_value = function(candidate)
        if candidate_value <= x_value:
            if candidate >= x:
                lower = x
            else:
                upper = x
            v, w, x = w, x, candidate
            v_value, w_value, x_value = w_value, x_value, candidate_value
        else:
            if candidate < x:
                lower = candidate
            else:
                upper = candidate
            if candidate_value <= w_value or w == x:
                v, w = w, candidate
                v_value, w_value = w_value, candidate_value
            elif candidate_value <= v_value or v == x or v == w:
                v = candidate
                v_value = candidate_value

    return x, max_iterations


def _select_smoothing_parameter_for_edf(
    basis: Tensor,
    response: Tensor,
    weights: Tensor,
    penalty: Tensor,
    target_effective_degrees_of_freedom: float,
    control: RSControl | CGControl,
) -> _SmoothFitResult:
    """Invert the P-spline hat-matrix trace as in ``pb(x, df=...)``."""

    def difference(log_smoothing_parameter: float) -> float:
        edf = _effective_degrees_of_freedom(
            basis,
            weights,
            math.exp(log_smoothing_parameter),
            penalty,
        )
        return float(edf) - target_effective_degrees_of_freedom

    lower = -30.0
    upper = 30.0
    lower_difference = difference(lower)
    upper_difference = difference(upper)
    iterations = 0

    if lower_difference * upper_difference > 0:
        log_smoothing_parameter = upper
    else:
        log_smoothing_parameter, iterations = _brent_root(
            difference,
            lower,
            upper,
            lower_difference,
            upper_difference,
            tolerance=control.edf_tolerance,
            max_iterations=control.max_edf_iterations,
        )

    smoothing_parameter = math.exp(log_smoothing_parameter)
    coefficient = _penalized_least_squares(
        basis, response, weights, smoothing_parameter, penalty
    )
    edf = _effective_degrees_of_freedom(basis, weights, smoothing_parameter, penalty)
    return _SmoothFitResult(
        coefficient=coefficient,
        smoothing_parameter=smoothing_parameter,
        effective_degrees_of_freedom=float(edf),
        smoothing_iterations=iterations,
    )


def _brent_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    lower_value: float,
    upper_value: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, int]:
    """Find a bracketed root with the Brent-Dekker method used by uniroot."""
    if lower_value == 0:
        return lower, 0
    if upper_value == 0:
        return upper, 0
    if lower_value * upper_value > 0:
        raise ValueError("Brent root requires values with opposite signs")

    a, b = lower, upper
    fa, fb = lower_value, upper_value
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    d = c
    used_bisection = True

    for iteration in range(1, max_iterations + 1):
        if fa != fc and fb != fc:
            candidate = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            candidate = b - fb * (b - a) / (fb - fa)

        boundary = (3.0 * a + b) / 4.0
        outside_bracket = not (min(boundary, b) < candidate < max(boundary, b))
        insufficient_progress = (
            used_bisection and abs(candidate - b) >= abs(b - c) / 2.0
        ) or (not used_bisection and abs(candidate - b) >= abs(c - d) / 2.0)
        bracket_is_small = (used_bisection and abs(b - c) < tolerance) or (
            not used_bisection and abs(c - d) < tolerance
        )
        if outside_bracket or insufficient_progress or bracket_is_small:
            candidate = (a + b) / 2.0
            used_bisection = True
        else:
            used_bisection = False

        candidate_value = function(candidate)
        d, c = c, b
        fc = fb
        if fa * candidate_value < 0:
            b, fb = candidate, candidate_value
        else:
            a, fa = candidate, candidate_value
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        if fb == 0 or abs(b - a) < tolerance:
            return b, iteration

    return b, max_iterations


def _estimate_ml_smoothing_parameter(
    basis: Tensor,
    response: Tensor,
    weights: Tensor,
    penalty: Tensor,
    penalty_nullity: int,
    starting_smoothing_parameter: float,
    control: RSControl | CGControl,
) -> _SmoothFitResult:
    """Apply the variance-component ML update used by ``gamlss.pb()``."""
    smoothing_parameter = starting_smoothing_parameter
    positive_weight_count = int(torch.count_nonzero(weights))

    for iteration in range(1, control.max_smoothing_iterations + 1):
        coefficient = _penalized_least_squares(
            basis, response, weights, smoothing_parameter, penalty
        )
        fitted = basis @ coefficient
        edf = _effective_degrees_of_freedom(
            basis, weights, smoothing_parameter, penalty
        )
        residual_denominator = positive_weight_count - float(edf)
        random_effect_denominator = float(edf) - penalty_nullity
        if residual_denominator <= 0 or random_effect_denominator <= 0:
            raise RuntimeError(
                "ML smoothing update requires positive residual and penalized "
                "degrees of freedom"
            )
        residual_variance = float(
            (weights * (response - fitted).square()).sum() / residual_denominator
        )
        coefficient_differences = penalty @ coefficient
        random_effect_variance = float(
            coefficient_differences.square().sum() / random_effect_denominator
        )
        random_effect_variance = max(random_effect_variance, 1e-7)
        updated_smoothing_parameter = min(
            max(residual_variance / random_effect_variance, 1e-7),
            1e7,
        )
        if (
            abs(updated_smoothing_parameter - smoothing_parameter)
            < control.smoothing_tolerance
        ):
            return _SmoothFitResult(
                coefficient=coefficient,
                smoothing_parameter=updated_smoothing_parameter,
                effective_degrees_of_freedom=float(edf),
                smoothing_iterations=iteration,
            )
        smoothing_parameter = updated_smoothing_parameter

    return _SmoothFitResult(
        coefficient=coefficient,
        smoothing_parameter=smoothing_parameter,
        effective_degrees_of_freedom=float(edf),
        smoothing_iterations=control.max_smoothing_iterations,
    )


def _penalized_least_squares(
    basis: Tensor,
    response: Tensor,
    weights: Tensor,
    smoothing_parameter: float,
    penalty: Tensor,
) -> Tensor:
    square_root_weights = weights.sqrt()
    augmented_design = torch.cat(
        (
            basis * square_root_weights.unsqueeze(-1),
            (smoothing_parameter**0.5) * penalty,
        )
    )
    augmented_response = torch.cat(
        (
            response * square_root_weights,
            torch.zeros(penalty.shape[0], dtype=response.dtype, device=response.device),
        )
    )
    return torch.linalg.lstsq(augmented_design, augmented_response).solution


def _effective_degrees_of_freedom(
    basis: Tensor,
    weights: Tensor,
    smoothing_parameter: float,
    penalty: Tensor,
) -> Tensor:
    gram = basis.mT @ (weights.unsqueeze(-1) * basis)
    system = gram + smoothing_parameter * (penalty.mT @ penalty)
    return torch.trace(torch.linalg.pinv(system) @ gram)


def _additive_predictor(
    design_matrix: Tensor,
    coefficient: Tensor,
    smooth_terms: Mapping[str, SmoothTerm],
    smooth_covariates: Mapping[str, Tensor],
    smooth_coefficients: Mapping[str, Tensor],
) -> Tensor:
    predictor = design_matrix @ coefficient
    for name, term in smooth_terms.items():
        predictor = (
            predictor
            + term.design(smooth_covariates[name]) @ (smooth_coefficients[name])
        )
    return predictor


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


def _validate_classical_inputs(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None,
    *,
    algorithm: str,
) -> None:
    if response.ndim != 1:
        raise ValueError(
            f"{algorithm} fitting currently requires a one-dimensional response"
        )
    if not torch.isfinite(response).all():
        raise ValueError(f"{algorithm} response must be finite")
    model_parameter = next(model.parameters())
    if response.dtype != model_parameter.dtype or response.device != (
        model_parameter.device
    ):
        raise ValueError(f"{algorithm} response dtype and device must match the model")
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
    model.linear_predictors(design_matrices, smooth_covariates=smooth_covariates)
