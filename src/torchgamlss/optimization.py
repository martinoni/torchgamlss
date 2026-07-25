"""Torch-native stochastic optimization for large fixed-design models."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class MiniBatchControl:
    """Controls for Adam mini-batch fitting."""

    batch_size: int = 1_024
    epochs: int = 100
    learning_rate: float = 1e-2
    learning_rate_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    shuffle: bool = True
    minimum_epochs: int = 5
    patience: int = 5
    tolerance_change: float = 1e-7
    tolerance_gradient: float = 1e-5
    evaluation_frequency: int = 1
    clip_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "minimum_epochs": self.minimum_epochs,
            "patience": self.patience,
            "evaluation_frequency": self.evaluation_frequency,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs must not exceed epochs")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if (
            not math.isfinite(self.learning_rate_decay)
            or not 0.0 < self.learning_rate_decay <= 1.0
        ):
            raise ValueError(
                "learning_rate_decay must be finite and in (0, 1]"
            )
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(
                not math.isfinite(value) or not 0.0 <= value < 1.0
                for value in self.betas
            )
        ):
            raise ValueError("betas must contain two finite values in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if (
            not math.isfinite(self.tolerance_change)
            or self.tolerance_change < 0
        ):
            raise ValueError("tolerance_change must be finite and non-negative")
        if (
            not math.isfinite(self.tolerance_gradient)
            or self.tolerance_gradient < 0
        ):
            raise ValueError(
                "tolerance_gradient must be finite and non-negative"
            )
        if self.clip_gradient_norm is not None and (
            not math.isfinite(self.clip_gradient_norm)
            or self.clip_gradient_norm <= 0
        ):
            raise ValueError(
                "clip_gradient_norm must be finite and positive when supplied"
            )


@dataclass(frozen=True)
class MiniBatchFitResult:
    """Summary of a Torch-native mini-batch optimization run."""

    negative_log_likelihood: float
    penalized_objective: float
    epochs: int
    updates: int
    gradient_max: float
    converged: bool
    stop_reason: Literal["loss_change", "gradient", "max_epochs"]
    objective_history: tuple[float, ...]
    evaluation_epochs: tuple[int, ...]
    batch_size: int
    learning_rate: float
    final_learning_rate: float


def fit_minibatch(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    control: MiniBatchControl | None = None,
    generator: torch.Generator | None = None,
) -> MiniBatchFitResult:
    """Fit a fixed-lambda model with bounded-memory stochastic updates.

    Uniform row sampling gives an unbiased gradient for the full weighted-mean
    penalized objective. End-of-epoch objectives and the final gradient are
    evaluated in deterministic chunks, so they do not allocate full predictor
    or spline-basis tensors.
    """
    control = control or MiniBatchControl()
    if not isinstance(control, MiniBatchControl):
        raise ValueError("control must be a MiniBatchControl")
    if generator is not None and torch.device(generator.device).type != "cpu":
        raise ValueError("mini-batch permutation generator must be on the CPU")
    if any(
        term.estimates_smoothing_parameter
        for terms in model.smooth_terms.values()
        for term in terms.values()
    ):
        raise ValueError(
            "automatic smoothing-parameter estimation requires fit_rs() or fit_cg()"
        )

    (
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
    ) = _validate_inputs(
        model,
        response,
        design_matrices,
        weights,
        offsets,
        smooth_covariates,
    )
    observation_count = response.numel()
    batch_size = min(control.batch_size, observation_count)
    total_weight = case_weights.sum().detach()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=control.learning_rate,
        betas=control.betas,
        eps=control.epsilon,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=control.learning_rate_decay,
    )

    _, initial_objective = _objective_values(
        model,
        response,
        design_matrices,
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
        batch_size,
        total_weight,
    )
    objective_history = [initial_objective]
    evaluation_epochs = [0]
    stable_evaluations = 0
    updates = 0
    completed_epochs = 0
    stop_reason: Literal["loss_change", "gradient", "max_epochs"] = "max_epochs"

    for epoch in range(1, control.epochs + 1):
        for batch_indices in _batch_indices(
            observation_count,
            batch_size,
            shuffle=control.shuffle,
            generator=generator,
            device=response.device,
        ):
            batch = _slice_inputs(
                response,
                design_matrices,
                case_weights,
                normalized_offsets,
                normalized_smooth_covariates,
                batch_indices,
            )
            (
                batch_response,
                batch_designs,
                batch_weights,
                batch_offsets,
                batch_smooth,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            batch_nll = _weighted_nll_sum(
                model,
                batch_response,
                batch_designs,
                batch_weights,
                batch_offsets,
                batch_smooth,
            )
            likelihood_scale = (
                observation_count / batch_response.numel()
            ) / total_weight
            loss = likelihood_scale * batch_nll
            penalty = model.smooth_penalty()
            if penalty.requires_grad:
                loss = loss + 0.5 * penalty / total_weight
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "mini-batch penalized objective is not finite"
                )
            loss.backward()
            _validate_gradients(parameters)
            if control.clip_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    control.clip_gradient_norm,
                )
            optimizer.step()
            updates += 1

        scheduler.step()
        completed_epochs = epoch
        should_evaluate = (
            epoch % control.evaluation_frequency == 0
            or epoch == control.epochs
        )
        if not should_evaluate:
            continue
        _, objective = _objective_values(
            model,
            response,
            design_matrices,
            case_weights,
            normalized_offsets,
            normalized_smooth_covariates,
            batch_size,
            total_weight,
        )
        previous = objective_history[-1]
        relative_change = abs(objective - previous) / max(1.0, abs(previous))
        objective_history.append(objective)
        evaluation_epochs.append(epoch)
        if relative_change <= control.tolerance_change:
            stable_evaluations += 1
        else:
            stable_evaluations = 0
        if (
            epoch >= control.minimum_epochs
            and stable_evaluations >= control.patience
        ):
            stop_reason = "loss_change"
            break

    final_nll, final_objective = _objective_values(
        model,
        response,
        design_matrices,
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
        batch_size,
        total_weight,
    )
    if evaluation_epochs[-1] != completed_epochs:
        objective_history.append(final_objective)
        evaluation_epochs.append(completed_epochs)
    gradient_max = _full_gradient_max(
        model,
        response,
        design_matrices,
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
        batch_size,
        total_weight,
    )
    if stop_reason == "max_epochs" and gradient_max <= control.tolerance_gradient:
        stop_reason = "gradient"

    return MiniBatchFitResult(
        negative_log_likelihood=final_nll,
        penalized_objective=final_objective,
        epochs=completed_epochs,
        updates=updates,
        gradient_max=gradient_max,
        converged=stop_reason != "max_epochs",
        stop_reason=stop_reason,
        objective_history=tuple(objective_history),
        evaluation_epochs=tuple(evaluation_epochs),
        batch_size=batch_size,
        learning_rate=control.learning_rate,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
    )


def _validate_inputs(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor | None,
    offsets: Mapping[str, Tensor] | None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    dict[str, dict[str, Tensor]],
]:
    model_parameter = next(model.parameters())
    if (
        response.ndim != 1
        or response.numel() < 1
        or response.dtype != model_parameter.dtype
        or response.device != model_parameter.device
        or not torch.isfinite(response).all()
    ):
        raise ValueError(
            "mini-batch response must be a non-empty finite vector matching "
            "the model dtype and device"
        )
    model.family.validate_response(response, context="mini-batch fit")
    expected_parameters = set(model.family.parameter_names)
    if set(design_matrices) != expected_parameters:
        raise ValueError(
            "Design matrices do not match family parameters: "
            f"missing={sorted(expected_parameters - set(design_matrices))}, "
            f"extra={sorted(set(design_matrices) - expected_parameters)}"
        )
    for parameter, design in design_matrices.items():
        if (
            design.ndim != 2
            or design.shape
            != (response.numel(), model.coefficients[parameter].numel())
        ):
            raise ValueError(
                f"design matrix for {parameter!r} has an invalid shape"
            )
        if (
            design.dtype != response.dtype
            or design.device != response.device
            or not torch.isfinite(design).all()
        ):
            raise ValueError(
                f"design matrix for {parameter!r} must be finite and match "
                "the response dtype and device"
            )

    case_weights = model._validated_weights(response, weights)
    offsets = offsets or {}
    extra_offsets = set(offsets).difference(expected_parameters)
    if extra_offsets:
        raise ValueError(
            f"Offsets contain unknown parameters: {sorted(extra_offsets)}"
        )
    normalized_offsets = {}
    for parameter, offset in offsets.items():
        if offset.dtype != response.dtype or offset.device != response.device:
            raise ValueError(
                f"offset for {parameter!r} must match response dtype and device"
            )
        try:
            normalized = torch.broadcast_to(offset, response.shape)
        except RuntimeError as error:
            raise ValueError(
                f"offset for {parameter!r} cannot be broadcast to the response"
            ) from error
        if not torch.isfinite(normalized).all():
            raise ValueError(f"offset for {parameter!r} must be finite")
        normalized_offsets[parameter] = normalized

    smooth_covariates = smooth_covariates or {}
    extra_smooth_parameters = set(smooth_covariates).difference(
        expected_parameters
    )
    if extra_smooth_parameters:
        raise ValueError(
            "Smooth covariates contain unknown parameters: "
            f"{sorted(extra_smooth_parameters)}"
        )
    normalized_smooth_covariates: dict[str, dict[str, Tensor]] = {}
    for parameter in model.family.parameter_names:
        expected_terms = set(model.smooth_terms[parameter])
        supplied = smooth_covariates.get(parameter, {})
        supplied_terms = set(supplied)
        if expected_terms != supplied_terms:
            raise ValueError(
                f"Smooth covariates for {parameter!r} do not match configured "
                f"terms: missing={sorted(expected_terms - supplied_terms)}, "
                f"extra={sorted(supplied_terms - expected_terms)}"
            )
        normalized_smooth_covariates[parameter] = {}
        for term_name, covariate in supplied.items():
            if (
                covariate.shape != response.shape
                or covariate.dtype != response.dtype
                or covariate.device != response.device
                or not torch.isfinite(covariate).all()
            ):
                raise ValueError(
                    f"smooth covariate {term_name!r} for {parameter!r} "
                    "must be finite with one value per response and matching "
                    "dtype and device"
                )
            normalized_smooth_covariates[parameter][term_name] = covariate
    return case_weights, normalized_offsets, normalized_smooth_covariates


def _batch_indices(
    observation_count: int,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> Iterator[Tensor]:
    if shuffle:
        indices = torch.randperm(
            observation_count,
            generator=generator,
            device="cpu",
        )
    else:
        indices = torch.arange(observation_count, device="cpu")
    for start in range(0, observation_count, batch_size):
        yield indices[start : start + batch_size].to(device=device)


def _slice_inputs(
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    indices: Tensor,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    Tensor,
    dict[str, Tensor],
    dict[str, dict[str, Tensor]],
]:
    return (
        response[indices],
        {
            parameter: design[indices]
            for parameter, design in design_matrices.items()
        },
        weights[indices],
        {
            parameter: offset[indices]
            for parameter, offset in offsets.items()
        },
        {
            parameter: {
                term: covariate[indices]
                for term, covariate in parameter_covariates.items()
            }
            for parameter, parameter_covariates in smooth_covariates.items()
        },
    )


def _weighted_nll_sum(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
) -> Tensor:
    losses = model.negative_log_likelihood(
        response,
        design_matrices,
        offsets=offsets,
        smooth_covariates=smooth_covariates,
        reduction="none",
    )
    return (losses * weights).sum()


def _sequential_indices(
    observation_count: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[Tensor]:
    for start in range(0, observation_count, batch_size):
        stop = min(start + batch_size, observation_count)
        yield torch.arange(start, stop, device=device)


def _objective_values(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    batch_size: int,
    total_weight: Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        negative_log_likelihood = torch.zeros(
            (),
            dtype=response.dtype,
            device=response.device,
        )
        for indices in _sequential_indices(
            response.numel(),
            batch_size,
            response.device,
        ):
            batch = _slice_inputs(
                response,
                design_matrices,
                weights,
                offsets,
                smooth_covariates,
                indices,
            )
            negative_log_likelihood += _weighted_nll_sum(model, *batch)
        penalized = (
            negative_log_likelihood + 0.5 * model.smooth_penalty()
        ) / total_weight
    if not torch.isfinite(negative_log_likelihood) or not torch.isfinite(penalized):
        raise FloatingPointError("mini-batch full objective is not finite")
    return float(negative_log_likelihood), float(penalized)


def _full_gradient_max(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    batch_size: int,
    total_weight: Tensor,
) -> float:
    parameters = list(model.parameters())
    model.zero_grad(set_to_none=True)
    for indices in _sequential_indices(
        response.numel(),
        batch_size,
        response.device,
    ):
        batch = _slice_inputs(
            response,
            design_matrices,
            weights,
            offsets,
            smooth_covariates,
            indices,
        )
        (_weighted_nll_sum(model, *batch) / total_weight).backward()
    penalty = model.smooth_penalty()
    if penalty.requires_grad:
        (0.5 * penalty / total_weight).backward()
    _validate_gradients(parameters)
    gradient_max = max(
        float(parameter.grad.detach().abs().max())
        for parameter in parameters
        if parameter.grad is not None
    )
    model.zero_grad(set_to_none=True)
    return gradient_max


def _validate_gradients(parameters: list[Tensor]) -> None:
    if any(
        parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
        for parameter in parameters
    ):
        raise FloatingPointError("mini-batch gradient is not finite")
