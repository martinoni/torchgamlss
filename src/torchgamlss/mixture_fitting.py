"""Expectation-maximization fitting for finite-mixture GAMLSS models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from torchgamlss.families import FiniteMixture

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class MixtureControl:
    """Control convergence and inner optimization for finite-mixture EM."""

    tolerance: float = 1e-4
    max_iterations: int = 200
    m_step_max_iter: int = 100
    m_step_tolerance_grad: float = 1e-9
    m_step_tolerance_change: float = 1e-12
    minimum_effective_count: float = 1e-8
    trace: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "max_iterations": self.max_iterations,
            "m_step_max_iter": self.m_step_max_iter,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        positive_fields = {
            "tolerance": self.tolerance,
            "m_step_tolerance_grad": self.m_step_tolerance_grad,
            "m_step_tolerance_change": self.m_step_tolerance_change,
            "minimum_effective_count": self.minimum_effective_count,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.trace, bool):
            raise ValueError("trace must be boolean")


@dataclass(frozen=True)
class MixtureFitResult:
    """Summary of a finite-mixture expectation-maximization fit."""

    global_deviance: float
    iterations: int
    converged: bool
    deviance_history: tuple[float, ...]
    m_step_function_evaluations: tuple[int, ...]
    posterior_probabilities: Tensor
    effective_counts: Tensor
    effective_proportions: Tensor

    @property
    def negative_log_likelihood(self) -> float:
        """Return half the fitted global deviance."""
        return self.global_deviance / 2.0


def fit_mixture(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
    control: MixtureControl | None = None,
) -> MixtureFitResult:
    """Fit a finite mixture with generalized EM and Torch L-BFGS M-steps."""
    control = control or MixtureControl()
    family = _require_finite_mixture(model)
    if model.neural_predictors or model.shared_predictor is not None:
        raise ValueError(
            "fit_mixture() currently supports linear and fixed-smooth "
            "predictors; use fit() or fit_minibatch() for neural mixtures"
        )
    if any(
        term.estimates_smoothing_parameter
        for terms in model.smooth_terms.values()
        for term in terms.values()
    ):
        raise ValueError(
            "fit_mixture() requires fixed smoothing parameters"
        )

    family.validate_response(response, context="mixture fitting")
    case_weights = model._validated_weights(response, weights)
    parameter_offsets = _validated_offsets(
        model,
        response,
        offsets,
    )
    parameter_smooth_covariates = smooth_covariates or {}
    model.linear_predictors(
        design_matrices,
        parameter_offsets,
        smooth_covariates=parameter_smooth_covariates,
    )
    _initialize_model(
        model,
        response,
        design_matrices,
        case_weights,
        parameter_offsets,
        initial_parameters,
    )

    parameters = model.predict(
        design_matrices,
        parameter_offsets,
        smooth_covariates=parameter_smooth_covariates,
        type="response",
    )
    assert isinstance(parameters, dict)
    global_deviance = _global_deviance(
        family,
        response,
        parameters,
        case_weights,
    )
    history = [float(global_deviance)]
    m_step_evaluations = []
    converged = False
    iterations = 0

    for iteration in range(1, control.max_iterations + 1):
        with torch.no_grad():
            responsibilities = family.posterior_probabilities(
                response,
                parameters,
            ).detach()
            effective_counts = (
                responsibilities * case_weights.unsqueeze(-1)
            ).sum(dim=0)
            if bool(
                (
                    effective_counts
                    < control.minimum_effective_count
                ).any()
            ):
                raise FloatingPointError(
                    "a mixture component collapsed below "
                    "minimum_effective_count"
                )

        previous_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        function_evaluations = _m_step(
            model,
            family,
            response,
            design_matrices,
            case_weights,
            parameter_offsets,
            parameter_smooth_covariates,
            responsibilities,
            control,
        )
        parameters = model.predict(
            design_matrices,
            parameter_offsets,
            smooth_covariates=parameter_smooth_covariates,
            type="response",
        )
        assert isinstance(parameters, dict)
        new_global_deviance = _global_deviance(
            family,
            response,
            parameters,
            case_weights,
        )
        allowed_increase = 1e-8 * (1.0 + abs(float(global_deviance)))
        if float(new_global_deviance - global_deviance) > allowed_increase:
            model.load_state_dict(previous_state)
            raise RuntimeError(
                "global deviance increased during the mixture M-step"
            )

        change = abs(float(global_deviance - new_global_deviance))
        global_deviance = new_global_deviance
        history.append(float(global_deviance))
        m_step_evaluations.append(function_evaluations)
        iterations = iteration
        if control.trace:
            print(
                f"TorchGAMLSS-MX iteration {iteration} "
                f"Global deviance = {float(global_deviance):.12g}"
            )
        if change < control.tolerance:
            converged = True
            break

    with torch.no_grad():
        posterior = family.posterior_probabilities(
            response,
            parameters,
        ).detach()
        effective_counts = (
            posterior * case_weights.unsqueeze(-1)
        ).sum(dim=0)
        effective_proportions = effective_counts / case_weights.sum()
    return MixtureFitResult(
        global_deviance=float(global_deviance),
        iterations=iterations,
        converged=converged,
        deviance_history=tuple(history),
        m_step_function_evaluations=tuple(m_step_evaluations),
        posterior_probabilities=posterior,
        effective_counts=effective_counts,
        effective_proportions=effective_proportions,
    )


def _require_finite_mixture(model: GAMLSS) -> FiniteMixture:
    if not isinstance(model.family, FiniteMixture):
        raise ValueError("fit_mixture() requires a FiniteMixture family")
    return model.family


def _validated_offsets(
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
        raw_offset = offsets.get(parameter, torch.zeros_like(response))
        if (
            not isinstance(raw_offset, Tensor)
            or raw_offset.dtype != response.dtype
            or raw_offset.device != response.device
        ):
            raise ValueError(
                f"offset for {parameter!r} must match response dtype and device"
            )
        try:
            offset = torch.broadcast_to(raw_offset, response.shape)
        except RuntimeError as error:
            raise ValueError(
                f"offset for {parameter!r} cannot be broadcast to the response"
            ) from error
        if not torch.isfinite(offset).all():
            raise ValueError(f"offset for {parameter!r} must be finite")
        result[parameter] = offset
    return result


def _initialize_model(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    case_weights: Tensor,
    offsets: Mapping[str, Tensor],
    initial_parameters: Mapping[str, Any] | None,
) -> None:
    starts = model.family.initial_parameters(response, initial_parameters)
    square_root_weights = case_weights.sqrt().unsqueeze(-1)
    with torch.no_grad():
        for parameter in model.family.parameter_names:
            for term in model.smooth_terms[parameter].values():
                term.coefficients.zero_()
            target = model.family.links[parameter](starts[parameter])
            target = target - offsets[parameter]
            design = design_matrices[parameter]
            weighted_design = design * square_root_weights
            weighted_target = target * square_root_weights.squeeze(-1)
            coefficient = torch.linalg.lstsq(
                weighted_design,
                weighted_target,
            ).solution
            if not torch.isfinite(coefficient).all():
                raise FloatingPointError(
                    f"mixture initialization for {parameter!r} is not finite"
                )
            model.coefficients[parameter].copy_(coefficient)


def _m_step(
    model: GAMLSS,
    family: FiniteMixture,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    case_weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    responsibilities: Tensor,
    control: MixtureControl,
) -> int:
    trainable_parameters = list(model.parameters())
    optimizer = torch.optim.LBFGS(
        trainable_parameters,
        max_iter=control.m_step_max_iter,
        tolerance_grad=control.m_step_tolerance_grad,
        tolerance_change=control.m_step_tolerance_change,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        parameters = model.predict(
            design_matrices,
            offsets,
            smooth_covariates=smooth_covariates,
            type="response",
        )
        assert isinstance(parameters, dict)
        component_log_probabilities = family.component_log_probabilities(
            response,
            parameters,
        )
        component_log_weights = family.component_log_weights(parameters)
        complete_loss = -(
            case_weights.unsqueeze(-1)
            * responsibilities
            * (component_log_weights + component_log_probabilities)
        ).sum()
        objective = complete_loss + 0.5 * model.smooth_penalty()
        if not torch.isfinite(objective):
            raise FloatingPointError(
                "mixture complete-data objective is not finite"
            )
        objective.backward()
        return objective

    optimizer.step(closure)
    final_objective = closure()
    if not torch.isfinite(final_objective):
        raise FloatingPointError("mixture M-step objective is not finite")
    state = optimizer.state[trainable_parameters[0]]
    return int(state.get("func_evals", 0))


def _global_deviance(
    family: FiniteMixture,
    response: Tensor,
    parameters: Mapping[str, Tensor],
    case_weights: Tensor,
) -> Tensor:
    deviance = -2.0 * (
        case_weights * family.log_prob(response, parameters)
    ).sum()
    if not torch.isfinite(deviance):
        raise FloatingPointError("mixture global deviance is not finite")
    return deviance.detach()
