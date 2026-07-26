"""Torch-native stochastic optimization for large fixed-design models."""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader

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
    validation_patience: int = 10
    validation_minimum_delta: float = 0.0
    restore_best_parameters: bool = True
    amp_dtype: Literal["float16", "bfloat16"] | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "minimum_epochs": self.minimum_epochs,
            "patience": self.patience,
            "evaluation_frequency": self.evaluation_frequency,
            "validation_patience": self.validation_patience,
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
        if (
            not math.isfinite(self.validation_minimum_delta)
            or self.validation_minimum_delta < 0
        ):
            raise ValueError(
                "validation_minimum_delta must be finite and non-negative"
            )
        if not isinstance(self.restore_best_parameters, bool):
            raise ValueError("restore_best_parameters must be boolean")
        if self.amp_dtype not in {None, "float16", "bfloat16"}:
            raise ValueError(
                "amp_dtype must be None, 'float16', or 'bfloat16'"
            )


@dataclass(frozen=True)
class MiniBatchValidationData:
    """Fixed holdout tensors evaluated during mini-batch fitting."""

    response: Tensor
    design_matrices: Mapping[str, Tensor]
    weights: Tensor | None = None
    offsets: Mapping[str, Tensor] | None = None
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None
    neural_inputs: Mapping[str, Tensor] | None = None
    shared_input: Tensor | None = None


@dataclass(frozen=True)
class MiniBatchFitResult:
    """Summary of a Torch-native mini-batch optimization run."""

    negative_log_likelihood: float
    penalized_objective: float
    epochs: int
    updates: int
    gradient_max: float
    converged: bool
    stop_reason: Literal["loss_change", "validation", "gradient", "max_epochs"]
    objective_history: tuple[float, ...]
    evaluation_epochs: tuple[int, ...]
    batch_size: int
    learning_rate: float
    final_learning_rate: float
    validation_negative_log_likelihood: float | None = None
    validation_history: tuple[float, ...] = ()
    validation_epochs: tuple[int, ...] = ()
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    restored_best_parameters: bool = False
    skipped_updates: int = 0


@dataclass(frozen=True)
class _LoaderMetadata:
    observation_count: int
    total_weight: Tensor
    maximum_batch_size: int
    batches: int


_LOADER_CHECKPOINT_FORMAT = "torchgamlss.minibatch_loader"
_LOADER_CHECKPOINT_VERSION = 2
_AMP_INITIAL_SCALE = 256.0


def _validated_amp_dtype(
    model: GAMLSS,
    control: MiniBatchControl,
) -> torch.dtype | None:
    if control.amp_dtype is None:
        return None
    model_parameter = next(model.parameters())
    if model_parameter.device.type != "cuda":
        raise ValueError("automatic mixed precision requires a CUDA model")
    if model_parameter.dtype != torch.float32:
        raise ValueError(
            "automatic mixed precision requires a float32 model"
        )
    if (
        control.amp_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("this CUDA device does not support bfloat16")
    return (
        torch.float16
        if control.amp_dtype == "float16"
        else torch.bfloat16
    )


def _optimizer_step(
    loss: Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: list[Tensor],
    clip_gradient_norm: float | None,
) -> bool:
    """Backpropagate and return whether GradScaler skipped the update."""
    if scaler.is_enabled():
        previous_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                clip_gradient_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        return scaler.get_scale() < previous_scale
    loss.backward()
    _validate_gradients(parameters)
    if clip_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            parameters,
            clip_gradient_norm,
        )
    optimizer.step()
    return False


def fit_minibatch(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    neural_inputs: Mapping[str, Tensor] | None = None,
    shared_input: Tensor | None = None,
    validation: MiniBatchValidationData | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
    control: MiniBatchControl | None = None,
    generator: torch.Generator | None = None,
) -> MiniBatchFitResult:
    """Fit a fixed-lambda model with bounded-memory stochastic updates.

    Uniform row sampling gives an unbiased gradient for the full weighted-mean
    penalized objective. End-of-epoch objectives and the final gradient are
    evaluated in deterministic chunks, so they do not allocate full predictor
    or spline-basis tensors. When a fixed holdout is supplied, its weighted
    mean negative log-likelihood controls early stopping and can restore the
    best complete model state.
    """
    control = control or MiniBatchControl()
    if not isinstance(control, MiniBatchControl):
        raise ValueError("control must be a MiniBatchControl")
    amp_dtype = _validated_amp_dtype(model, control)
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
        normalized_neural_inputs,
        normalized_shared_input,
    ) = _validate_inputs(
        model,
        response,
        design_matrices,
        weights,
        offsets,
        smooth_covariates,
        neural_inputs,
        shared_input,
    )
    if validation is not None and not isinstance(
        validation,
        MiniBatchValidationData,
    ):
        raise ValueError("validation must be a MiniBatchValidationData")
    if validation is None:
        validation_values = None
        validation_batch_size = None
        validation_total_weight = None
    else:
        try:
            validation_values = _validate_inputs(
                model,
                validation.response,
                validation.design_matrices,
                validation.weights,
                validation.offsets,
                validation.smooth_covariates,
                validation.neural_inputs,
                validation.shared_input,
            )
        except ValueError as error:
            raise ValueError(f"invalid validation data: {error}") from error
        validation_batch_size = min(
            control.batch_size,
            validation.response.numel(),
        )
        validation_total_weight = validation_values[0].sum().detach()
    observation_count = response.numel()
    batch_size = min(control.batch_size, observation_count)
    total_weight = case_weights.sum().detach()
    if initial_parameters is not None:
        _initialize_tensor_intercepts(
            model,
            response,
            design_matrices,
            case_weights,
            normalized_offsets,
            normalized_smooth_covariates,
            normalized_neural_inputs,
            normalized_shared_input,
            initial_parameters,
            batch_size=batch_size,
            total_weight=total_weight,
        )
    original_training_mode = model.training
    model.train()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=control.learning_rate,
        betas=control.betas,
        eps=control.epsilon,
    )
    scaler = torch.amp.GradScaler(
        next(model.parameters()).device.type,
        enabled=amp_dtype == torch.float16,
        init_scale=_AMP_INITIAL_SCALE,
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
        normalized_neural_inputs,
        normalized_shared_input,
        batch_size,
        total_weight,
    )
    objective_history = [initial_objective]
    evaluation_epochs = [0]
    stable_evaluations = 0
    validation_history: list[float] = []
    validation_epochs: list[int] = []
    best_validation_loss: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, Tensor] | None = None
    evaluations_without_improvement = 0
    if validation is not None:
        assert validation_values is not None
        assert validation_batch_size is not None
        assert validation_total_weight is not None
        _, initial_validation_loss = _validation_values(
            model,
            validation,
            validation_values,
            validation_batch_size,
            validation_total_weight,
        )
        validation_history.append(initial_validation_loss)
        validation_epochs.append(0)
        best_validation_loss = initial_validation_loss
        best_epoch = 0
        if control.restore_best_parameters:
            best_state = _state_dict_copy(model)
    updates = 0
    skipped_updates = 0
    completed_epochs = 0
    stop_reason: Literal[
        "loss_change",
        "validation",
        "gradient",
        "max_epochs",
    ] = "max_epochs"

    for epoch in range(1, control.epochs + 1):
        applied_updates_in_epoch = 0
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
                normalized_neural_inputs,
                normalized_shared_input,
                batch_indices,
            )
            (
                batch_response,
                batch_designs,
                batch_weights,
                batch_offsets,
                batch_smooth,
                batch_neural,
                batch_shared,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=response.device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                batch_nll = _weighted_nll_sum(
                    model,
                    batch_response,
                    batch_designs,
                    batch_weights,
                    batch_offsets,
                    batch_smooth,
                    batch_neural,
                    batch_shared,
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
            skipped = _optimizer_step(
                loss,
                optimizer=optimizer,
                scaler=scaler,
                parameters=parameters,
                clip_gradient_norm=control.clip_gradient_norm,
            )
            updates += 1
            skipped_updates += int(skipped)
            applied_updates_in_epoch += int(not skipped)

        if applied_updates_in_epoch > 0:
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
            normalized_neural_inputs,
            normalized_shared_input,
            batch_size,
            total_weight,
        )
        previous = objective_history[-1]
        relative_change = abs(objective - previous) / max(1.0, abs(previous))
        objective_history.append(objective)
        evaluation_epochs.append(epoch)
        if validation is None:
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
        else:
            assert validation_values is not None
            assert validation_batch_size is not None
            assert validation_total_weight is not None
            _, validation_loss = _validation_values(
                model,
                validation,
                validation_values,
                validation_batch_size,
                validation_total_weight,
            )
            validation_history.append(validation_loss)
            validation_epochs.append(epoch)
            assert best_validation_loss is not None
            if (
                validation_loss
                < best_validation_loss - control.validation_minimum_delta
            ):
                best_validation_loss = validation_loss
                best_epoch = epoch
                evaluations_without_improvement = 0
                if control.restore_best_parameters:
                    best_state = _state_dict_copy(model)
            else:
                evaluations_without_improvement += 1
            if (
                epoch >= control.minimum_epochs
                and evaluations_without_improvement
                >= control.validation_patience
            ):
                stop_reason = "validation"
                break

    restored_best_parameters = False
    if (
        validation is not None
        and control.restore_best_parameters
        and best_state is not None
        and best_epoch != completed_epochs
    ):
        model.load_state_dict(best_state)
        restored_best_parameters = True

    final_nll, final_objective = _objective_values(
        model,
        response,
        design_matrices,
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
        normalized_neural_inputs,
        normalized_shared_input,
        batch_size,
        total_weight,
    )
    if evaluation_epochs[-1] != completed_epochs:
        objective_history.append(final_objective)
        evaluation_epochs.append(completed_epochs)
    if validation is None:
        final_validation_nll = None
    else:
        assert validation_values is not None
        assert validation_batch_size is not None
        assert validation_total_weight is not None
        final_validation_nll, _ = _validation_values(
            model,
            validation,
            validation_values,
            validation_batch_size,
            validation_total_weight,
        )
    gradient_max = _full_gradient_max(
        model,
        response,
        design_matrices,
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
        normalized_neural_inputs,
        normalized_shared_input,
        batch_size,
        total_weight,
    )
    if stop_reason == "max_epochs" and gradient_max <= control.tolerance_gradient:
        stop_reason = "gradient"

    result = MiniBatchFitResult(
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
        validation_negative_log_likelihood=final_validation_nll,
        validation_history=tuple(validation_history),
        validation_epochs=tuple(validation_epochs),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        restored_best_parameters=restored_best_parameters,
        skipped_updates=skipped_updates,
    )
    model.train(original_training_mode)
    return result


def fit_minibatch_loader(
    model: GAMLSS,
    loader: DataLoader[Any],
    *,
    validation_loader: DataLoader[Any] | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
    control: MiniBatchControl | None = None,
    non_blocking: bool = False,
    checkpoint_path: str | os.PathLike[str] | None = None,
    checkpoint_frequency: int = 1,
    resume_from: str | os.PathLike[str] | None = None,
) -> MiniBatchFitResult:
    """Fit from re-iterable DataLoaders without resident full-data tensors.

    Each loader batch must be a mapping with ``response`` and
    ``design_matrices`` plus any optional tensor inputs accepted by
    :meth:`GAMLSS.fit_minibatch`. The loader owns batching, sampling, workers,
    and collation. A deterministic full pass before optimization infers the
    observation count and total case weight used by the exact weighted
    objective.
    """
    control = control or MiniBatchControl()
    if not isinstance(control, MiniBatchControl):
        raise ValueError("control must be a MiniBatchControl")
    amp_dtype = _validated_amp_dtype(model, control)
    if not isinstance(non_blocking, bool):
        raise ValueError("non_blocking must be boolean")
    if (
        isinstance(checkpoint_frequency, bool)
        or not isinstance(checkpoint_frequency, int)
        or checkpoint_frequency < 1
    ):
        raise ValueError("checkpoint_frequency must be an integer of at least 1")
    if resume_from is not None and checkpoint_path is None:
        checkpoint_path = resume_from
    if resume_from is not None and initial_parameters is not None:
        raise ValueError(
            "initial_parameters cannot be supplied when resuming a checkpoint"
        )
    _validate_loader_configuration(loader, context="training loader")
    if validation_loader is not None:
        _validate_loader_configuration(
            validation_loader,
            context="validation loader",
        )
    if any(
        term.estimates_smoothing_parameter
        for terms in model.smooth_terms.values()
        for term in terms.values()
    ):
        raise ValueError(
            "automatic smoothing-parameter estimation requires fit_rs() or fit_cg()"
        )

    initial_objective: float | None = None
    initial_validation_loss: float | None = None
    if resume_from is None:
        if initial_parameters is not None:
            _initialize_loader_intercepts(
                model,
                loader,
                initial_parameters,
                non_blocking=non_blocking,
            )
        _, initial_objective, training_metadata = (
            _loader_objective_values(
                model,
                loader,
                non_blocking=non_blocking,
            )
        )
        validation_metadata: _LoaderMetadata | None = None
        if validation_loader is not None:
            initial_validation_nll, validation_metadata = (
                _loader_likelihood_values(
                    model,
                    validation_loader,
                    non_blocking=non_blocking,
                    context="validation loader",
                )
            )
            initial_validation_loss = (
                initial_validation_nll
                / float(validation_metadata.total_weight)
            )
    else:
        training_metadata = _loader_metadata_values(
            model,
            loader,
            non_blocking=non_blocking,
            context="training loader",
        )
        validation_metadata = (
            _loader_metadata_values(
                model,
                validation_loader,
                non_blocking=non_blocking,
                context="validation loader",
            )
            if validation_loader is not None
            else None
        )

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=control.learning_rate,
        betas=control.betas,
        eps=control.epsilon,
    )
    scaler = torch.amp.GradScaler(
        next(model.parameters()).device.type,
        enabled=amp_dtype == torch.float16,
        init_scale=_AMP_INITIAL_SCALE,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=control.learning_rate_decay,
    )
    original_training_mode = model.training
    model.train()
    if resume_from is None:
        assert initial_objective is not None
        objective_history = [initial_objective]
        evaluation_epochs = [0]
        validation_history = (
            [initial_validation_loss]
            if initial_validation_loss is not None
            else []
        )
        validation_epochs = [0] if validation_loader is not None else []
        best_validation_loss = initial_validation_loss
        best_epoch = 0 if validation_loader is not None else None
        stable_evaluations = 0
        evaluations_without_improvement = 0
        best_state = (
            _state_dict_copy(model)
            if validation_loader is not None
            and control.restore_best_parameters
            else None
        )
        updates = 0
        skipped_updates = 0
        completed_epochs = 0
    else:
        checkpoint = _load_loader_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            control=control,
            training_metadata=training_metadata,
            validation_metadata=validation_metadata,
            loader=loader,
            validation_loader=validation_loader,
        )
        objective_history = list(checkpoint["objective_history"])
        evaluation_epochs = list(checkpoint["evaluation_epochs"])
        validation_history = list(checkpoint["validation_history"])
        validation_epochs = list(checkpoint["validation_epochs"])
        best_validation_loss = checkpoint["best_validation_loss"]
        best_epoch = checkpoint["best_epoch"]
        stable_evaluations = checkpoint["stable_evaluations"]
        evaluations_without_improvement = checkpoint[
            "evaluations_without_improvement"
        ]
        best_state = checkpoint["best_state"]
        updates = checkpoint["updates"]
        skipped_updates = checkpoint["skipped_updates"]
        completed_epochs = checkpoint["completed_epochs"]
    stop_reason: Literal[
        "loss_change",
        "validation",
        "gradient",
        "max_epochs",
    ] = "max_epochs"

    try:
        for epoch in range(completed_epochs + 1, control.epochs + 1):
            epoch_observations = 0
            epoch_total_weight = torch.zeros_like(
                training_metadata.total_weight
            )
            epoch_maximum_batch_size = 0
            epoch_batches = 0
            applied_updates_in_epoch = 0
            for batch in _loader_batches(
                model,
                loader,
                non_blocking=non_blocking,
                context="training loader",
            ):
                (
                    batch_response,
                    batch_designs,
                    batch_weights,
                    batch_offsets,
                    batch_smooth,
                    batch_neural,
                    batch_shared,
                ) = batch
                batch_observations = batch_response.numel()
                epoch_observations += batch_observations
                epoch_total_weight += batch_weights.sum().detach()
                epoch_maximum_batch_size = max(
                    epoch_maximum_batch_size,
                    batch_observations,
                )
                epoch_batches += 1

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=batch_response.device.type,
                    dtype=amp_dtype,
                    enabled=amp_dtype is not None,
                ):
                    batch_nll = _weighted_nll_sum(
                        model,
                        batch_response,
                        batch_designs,
                        batch_weights,
                        batch_offsets,
                        batch_smooth,
                        batch_neural,
                        batch_shared,
                    )
                    likelihood_scale = (
                        training_metadata.observation_count
                        / batch_observations
                    ) / training_metadata.total_weight
                    loss = likelihood_scale * batch_nll
                penalty = model.smooth_penalty()
                if penalty.requires_grad:
                    loss = (
                        loss
                        + 0.5
                        * penalty
                        / training_metadata.total_weight
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "DataLoader mini-batch penalized objective is not finite"
                    )
                skipped = _optimizer_step(
                    loss,
                    optimizer=optimizer,
                    scaler=scaler,
                    parameters=parameters,
                    clip_gradient_norm=control.clip_gradient_norm,
                )
                updates += 1
                skipped_updates += int(skipped)
                applied_updates_in_epoch += int(not skipped)

            epoch_metadata = _LoaderMetadata(
                observation_count=epoch_observations,
                total_weight=epoch_total_weight,
                maximum_batch_size=epoch_maximum_batch_size,
                batches=epoch_batches,
            )
            _validate_loader_metadata(
                epoch_metadata,
                training_metadata,
                context=f"training loader epoch {epoch}",
            )
            if applied_updates_in_epoch > 0:
                scheduler.step()
            completed_epochs = epoch
            should_evaluate = (
                epoch % control.evaluation_frequency == 0
                or epoch == control.epochs
            )
            stop_after_checkpoint = False
            if should_evaluate:
                _, objective, _ = _loader_objective_values(
                    model,
                    loader,
                    non_blocking=non_blocking,
                    expected_metadata=training_metadata,
                )
                previous = objective_history[-1]
                relative_change = abs(objective - previous) / max(
                    1.0,
                    abs(previous),
                )
                objective_history.append(objective)
                evaluation_epochs.append(epoch)
                if validation_loader is None:
                    if relative_change <= control.tolerance_change:
                        stable_evaluations += 1
                    else:
                        stable_evaluations = 0
                    if (
                        epoch >= control.minimum_epochs
                        and stable_evaluations >= control.patience
                    ):
                        stop_reason = "loss_change"
                        stop_after_checkpoint = True
                else:
                    assert validation_metadata is not None
                    validation_nll, _ = _loader_likelihood_values(
                        model,
                        validation_loader,
                        non_blocking=non_blocking,
                        context="validation loader",
                        expected_metadata=validation_metadata,
                    )
                    validation_loss = (
                        validation_nll
                        / float(validation_metadata.total_weight)
                    )
                    validation_history.append(validation_loss)
                    validation_epochs.append(epoch)
                    assert best_validation_loss is not None
                    if (
                        validation_loss
                        < best_validation_loss
                        - control.validation_minimum_delta
                    ):
                        best_validation_loss = validation_loss
                        best_epoch = epoch
                        evaluations_without_improvement = 0
                        if control.restore_best_parameters:
                            best_state = _state_dict_copy(model)
                    else:
                        evaluations_without_improvement += 1
                    if (
                        epoch >= control.minimum_epochs
                        and evaluations_without_improvement
                        >= control.validation_patience
                    ):
                        stop_reason = "validation"
                        stop_after_checkpoint = True

            should_checkpoint = (
                checkpoint_path is not None
                and (
                    epoch % checkpoint_frequency == 0
                    or epoch == control.epochs
                    or stop_after_checkpoint
                )
            )
            if should_checkpoint:
                assert checkpoint_path is not None
                _save_loader_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    control=control,
                    completed_epochs=completed_epochs,
                    updates=updates,
                    skipped_updates=skipped_updates,
                    objective_history=objective_history,
                    evaluation_epochs=evaluation_epochs,
                    validation_history=validation_history,
                    validation_epochs=validation_epochs,
                    stable_evaluations=stable_evaluations,
                    evaluations_without_improvement=(
                        evaluations_without_improvement
                    ),
                    best_validation_loss=best_validation_loss,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    training_metadata=training_metadata,
                    validation_metadata=validation_metadata,
                    loader=loader,
                    validation_loader=validation_loader,
                )
            if stop_after_checkpoint:
                break

        restored_best_parameters = False
        if (
            validation_loader is not None
            and control.restore_best_parameters
            and best_state is not None
            and best_epoch != completed_epochs
        ):
            model.load_state_dict(best_state)
            restored_best_parameters = True

        final_nll, final_objective, _ = _loader_objective_values(
            model,
            loader,
            non_blocking=non_blocking,
            expected_metadata=training_metadata,
        )
        if evaluation_epochs[-1] != completed_epochs:
            objective_history.append(final_objective)
            evaluation_epochs.append(completed_epochs)
        if validation_loader is None:
            final_validation_nll = None
        else:
            assert validation_metadata is not None
            final_validation_nll, _ = _loader_likelihood_values(
                model,
                validation_loader,
                non_blocking=non_blocking,
                context="validation loader",
                expected_metadata=validation_metadata,
            )
        gradient_max = _loader_full_gradient_max(
            model,
            loader,
            training_metadata,
            non_blocking=non_blocking,
        )
        if (
            stop_reason == "max_epochs"
            and gradient_max <= control.tolerance_gradient
        ):
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
            batch_size=training_metadata.maximum_batch_size,
            learning_rate=control.learning_rate,
            final_learning_rate=float(optimizer.param_groups[0]["lr"]),
            validation_negative_log_likelihood=final_validation_nll,
            validation_history=tuple(validation_history),
            validation_epochs=tuple(validation_epochs),
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            restored_best_parameters=restored_best_parameters,
            skipped_updates=skipped_updates,
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_training_mode)


def _validate_loader_configuration(
    loader: DataLoader[Any],
    *,
    context: str,
) -> None:
    if not isinstance(loader, DataLoader):
        raise ValueError(f"{context} must be a torch DataLoader")
    if loader.drop_last:
        raise ValueError(
            f"{context} must use drop_last=False for an exact objective"
        )


def _loader_batches(
    model: GAMLSS,
    loader: DataLoader[Any],
    *,
    non_blocking: bool,
    context: str,
) -> Iterator[
    tuple[
        Tensor,
        dict[str, Tensor],
        Tensor,
        dict[str, Tensor],
        dict[str, dict[str, Tensor]],
        dict[str, Tensor],
        Tensor | None,
    ]
]:
    model_parameter = next(model.parameters())
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch_context = f"{context} batch {batch_index}"
        if not isinstance(raw_batch, Mapping):
            raise ValueError(f"{batch_context} must be a mapping")
        required = {"response", "design_matrices"}
        optional = {
            "weights",
            "offsets",
            "smooth_covariates",
            "neural_inputs",
            "shared_input",
        }
        missing = required.difference(raw_batch)
        extra = set(raw_batch).difference(required | optional)
        if missing or extra:
            raise ValueError(
                f"{batch_context} fields do not match the loader schema: "
                f"missing={sorted(missing)}, extra={sorted(extra, key=str)}"
            )
        moved = {
            key: _move_loader_structure(
                value,
                device=model_parameter.device,
                non_blocking=non_blocking,
                context=f"{batch_context} field {key!r}",
            )
            for key, value in raw_batch.items()
        }
        response = moved["response"]
        design_matrices = moved["design_matrices"]
        weights = moved.get("weights")
        offsets = moved.get("offsets")
        smooth_covariates = moved.get("smooth_covariates")
        neural_inputs = moved.get("neural_inputs")
        shared_input = moved.get("shared_input")
        if not isinstance(response, Tensor):
            raise ValueError(f"{batch_context} response must be a tensor")
        mapping_fields = {
            "design_matrices": design_matrices,
            "offsets": offsets,
            "smooth_covariates": smooth_covariates,
            "neural_inputs": neural_inputs,
        }
        for name, value in mapping_fields.items():
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(
                    f"{batch_context} {name} must be a mapping"
                )
        if weights is not None and not isinstance(weights, Tensor):
            raise ValueError(f"{batch_context} weights must be a tensor")
        if shared_input is not None and not isinstance(shared_input, Tensor):
            raise ValueError(
                f"{batch_context} shared_input must be a tensor"
            )
        try:
            (
                normalized_weights,
                normalized_offsets,
                normalized_smooth_covariates,
                normalized_neural_inputs,
                normalized_shared_input,
            ) = _validate_inputs(
                model,
                response,
                design_matrices,
                weights,
                offsets,
                smooth_covariates,
                neural_inputs,
                shared_input,
                require_positive_weight=False,
            )
        except ValueError as error:
            raise ValueError(f"invalid {batch_context}: {error}") from error
        yield (
            response,
            dict(design_matrices),
            normalized_weights,
            normalized_offsets,
            normalized_smooth_covariates,
            normalized_neural_inputs,
            normalized_shared_input,
        )


def _move_loader_structure(
    value: Any,
    *,
    device: torch.device,
    non_blocking: bool,
    context: str,
) -> Any:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: _move_loader_structure(
                nested,
                device=device,
                non_blocking=non_blocking,
                context=f"{context}.{key}",
            )
            for key, nested in value.items()
        }
    raise ValueError(f"{context} must contain only tensors and mappings")


@contextmanager
def _preserve_loader_rng(loader: DataLoader[Any]) -> Iterator[None]:
    global_state = torch.random.get_rng_state()
    generators: list[tuple[torch.Generator, Tensor]] = []
    seen: set[int] = set()
    owners = [
        loader,
        getattr(loader, "sampler", None),
        getattr(loader, "batch_sampler", None),
        getattr(getattr(loader, "batch_sampler", None), "sampler", None),
    ]
    for owner in owners:
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator) and id(generator) not in seen:
            seen.add(id(generator))
            generators.append((generator, generator.get_state()))
    try:
        yield
    finally:
        torch.random.set_rng_state(global_state)
        for generator, state in generators:
            generator.set_state(state)


def _initialize_loader_intercepts(
    model: GAMLSS,
    loader: DataLoader[Any],
    initial_parameters: Mapping[str, Any],
    *,
    non_blocking: bool,
) -> None:
    starts = _validated_loader_initial_parameters(
        model,
        initial_parameters,
    )
    model_parameter = next(model.parameters())
    weighted_residual_sums = {
        parameter: model_parameter.new_zeros(())
        for parameter in model.family.parameter_names
    }
    target_predictors: dict[str, Tensor] | None = None
    observation_count = 0
    total_weight = model_parameter.new_zeros(())
    maximum_batch_size = 0
    batches = 0
    training_mode = model.training
    model.eval()
    try:
        with _preserve_loader_rng(loader), torch.no_grad():
            for batch in _loader_batches(
                model,
                loader,
                non_blocking=non_blocking,
                context="training loader initialization",
            ):
                (
                    batch_response,
                    batch_designs,
                    batch_weights,
                    batch_offsets,
                    batch_smooth,
                    batch_neural,
                    batch_shared,
                ) = batch
                if target_predictors is None:
                    expanded = model.family.initial_parameters(
                        batch_response,
                        starts,
                    )
                    target_predictors = {
                        parameter: model.family.links[parameter](
                            expanded[parameter]
                        )[0]
                        for parameter in model.family.parameter_names
                    }
                contributions = model.term_contributions(
                    batch_designs,
                    batch_offsets,
                    smooth_covariates=batch_smooth,
                    neural_inputs=batch_neural,
                    shared_input=batch_shared,
                )
                for parameter in model.family.parameter_names:
                    design = batch_designs[parameter]
                    _validate_initialization_intercept(design, parameter)
                    current_intercept = (
                        model.coefficients[parameter][0].detach()
                    )
                    non_intercept_predictor = (
                        contributions[parameter].total
                        - design[:, 0] * current_intercept
                    )
                    weighted_residual_sums[parameter] += (
                        batch_weights
                        * (
                            target_predictors[parameter]
                            - non_intercept_predictor
                        )
                    ).sum()
                batch_observations = batch_response.numel()
                observation_count += batch_observations
                total_weight += batch_weights.sum()
                maximum_batch_size = max(
                    maximum_batch_size,
                    batch_observations,
                )
                batches += 1
    finally:
        model.train(training_mode)
    metadata = _LoaderMetadata(
        observation_count=observation_count,
        total_weight=total_weight.detach(),
        maximum_batch_size=maximum_batch_size,
        batches=batches,
    )
    _validate_loader_metadata(
        metadata,
        None,
        context="training loader initialization",
    )
    with torch.no_grad():
        for parameter in model.family.parameter_names:
            model.coefficients[parameter][0].copy_(
                weighted_residual_sums[parameter] / metadata.total_weight
            )


def _validated_loader_initial_parameters(
    model: GAMLSS,
    values: Mapping[str, Any],
) -> dict[str, Tensor]:
    if not isinstance(values, Mapping):
        raise ValueError("initial parameters must be supplied as a mapping")
    expected = set(model.family.parameter_names)
    received = set(values)
    if received != expected:
        raise ValueError(
            "DataLoader initial parameters must provide one scalar for every "
            "family parameter: "
            f"missing={sorted(expected - received)}, "
            f"extra={sorted(received - expected)}"
        )
    model_parameter = next(model.parameters())
    starts = {}
    for parameter in model.family.parameter_names:
        try:
            value = torch.as_tensor(
                values[parameter],
                dtype=model_parameter.dtype,
                device=model_parameter.device,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"initial parameter {parameter!r} cannot be converted to a tensor"
            ) from error
        if value.ndim != 0:
            raise ValueError(
                "DataLoader initial parameters must be scalars; "
                f"{parameter!r} is not scalar"
            )
        starts[parameter] = value
    return starts


def _loader_metadata_values(
    model: GAMLSS,
    loader: DataLoader[Any],
    *,
    non_blocking: bool,
    context: str,
) -> _LoaderMetadata:
    model_parameter = next(model.parameters())
    observation_count = 0
    total_weight = model_parameter.new_zeros(())
    maximum_batch_size = 0
    batches = 0
    with _preserve_loader_rng(loader), torch.no_grad():
        for batch in _loader_batches(
            model,
            loader,
            non_blocking=non_blocking,
            context=context,
        ):
            batch_response = batch[0]
            batch_weights = batch[2]
            batch_observations = batch_response.numel()
            observation_count += batch_observations
            total_weight += batch_weights.sum()
            maximum_batch_size = max(
                maximum_batch_size,
                batch_observations,
            )
            batches += 1
    metadata = _LoaderMetadata(
        observation_count=observation_count,
        total_weight=total_weight.detach(),
        maximum_batch_size=maximum_batch_size,
        batches=batches,
    )
    _validate_loader_metadata(metadata, None, context=context)
    return metadata


def _save_loader_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: GAMLSS,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    control: MiniBatchControl,
    completed_epochs: int,
    updates: int,
    skipped_updates: int,
    objective_history: list[float],
    evaluation_epochs: list[int],
    validation_history: list[float],
    validation_epochs: list[int],
    stable_evaluations: int,
    evaluations_without_improvement: int,
    best_validation_loss: float | None,
    best_epoch: int | None,
    best_state: dict[str, Tensor] | None,
    training_metadata: _LoaderMetadata,
    validation_metadata: _LoaderMetadata | None,
    loader: DataLoader[Any],
    validation_loader: DataLoader[Any] | None,
) -> None:
    model_parameter = next(model.parameters())
    checkpoint = {
        "format": _LOADER_CHECKPOINT_FORMAT,
        "version": _LOADER_CHECKPOINT_VERSION,
        "model_dtype": str(model_parameter.dtype),
        "model_device_type": model_parameter.device.type,
        "family_parameters": tuple(model.family.parameter_names),
        "family_signature": _family_checkpoint_signature(model),
        "control": asdict(control),
        "completed_epochs": completed_epochs,
        "updates": updates,
        "skipped_updates": skipped_updates,
        "objective_history": tuple(objective_history),
        "evaluation_epochs": tuple(evaluation_epochs),
        "validation_history": tuple(validation_history),
        "validation_epochs": tuple(validation_epochs),
        "stable_evaluations": stable_evaluations,
        "evaluations_without_improvement": (
            evaluations_without_improvement
        ),
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "training_metadata": _checkpoint_metadata(training_metadata),
        "validation_metadata": (
            _checkpoint_metadata(validation_metadata)
            if validation_metadata is not None
            else None
        ),
        "model_state_dict": _state_dict_copy(model),
        "optimizer_state_dict": _checkpoint_cpu_copy(
            optimizer.state_dict()
        ),
        "scheduler_state_dict": _checkpoint_cpu_copy(
            scheduler.state_dict()
        ),
        "scaler_state_dict": _checkpoint_cpu_copy(
            scaler.state_dict()
        ),
        "torch_rng_state": torch.random.get_rng_state().clone(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(model_parameter.device).cpu()
            if model_parameter.device.type == "cuda"
            else None
        ),
        "training_loader_generators": _loader_generator_states(loader),
        "validation_loader_generators": (
            _loader_generator_states(validation_loader)
            if validation_loader is not None
            else None
        ),
    }
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_loader_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: GAMLSS,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    control: MiniBatchControl,
    training_metadata: _LoaderMetadata,
    validation_metadata: _LoaderMetadata | None,
    loader: DataLoader[Any],
    validation_loader: DataLoader[Any] | None,
) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        checkpoint = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError(
            f"could not load mini-batch checkpoint {str(source)!r}"
        ) from error
    if not isinstance(checkpoint, dict):
        raise ValueError("mini-batch checkpoint must contain a dictionary")
    required_v2 = {
        "format",
        "version",
        "model_dtype",
        "model_device_type",
        "family_parameters",
        "family_signature",
        "control",
        "completed_epochs",
        "updates",
        "skipped_updates",
        "objective_history",
        "evaluation_epochs",
        "validation_history",
        "validation_epochs",
        "stable_evaluations",
        "evaluations_without_improvement",
        "best_validation_loss",
        "best_epoch",
        "best_state",
        "training_metadata",
        "validation_metadata",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "torch_rng_state",
        "cuda_rng_state",
        "training_loader_generators",
        "validation_loader_generators",
    }
    required_v1 = required_v2 - {
        "skipped_updates",
        "scaler_state_dict",
    }
    checkpoint_version = checkpoint.get("version")
    if checkpoint.get("format") != _LOADER_CHECKPOINT_FORMAT:
        raise ValueError("mini-batch checkpoint format is not supported")
    if checkpoint_version == 1:
        if set(checkpoint) != required_v1:
            raise ValueError("mini-batch checkpoint has an invalid schema")
        checkpoint["skipped_updates"] = 0
        checkpoint["scaler_state_dict"] = {}
        saved_control = checkpoint.get("control")
        if isinstance(saved_control, dict):
            saved_control = dict(saved_control)
            saved_control["amp_dtype"] = None
            checkpoint["control"] = saved_control
    elif checkpoint_version == _LOADER_CHECKPOINT_VERSION:
        if set(checkpoint) != required_v2:
            raise ValueError("mini-batch checkpoint has an invalid schema")
    else:
        raise ValueError("mini-batch checkpoint format is not supported")
    if set(checkpoint) != required_v2:
        raise ValueError("mini-batch checkpoint has an invalid schema")
    model_parameter = next(model.parameters())
    if checkpoint["model_dtype"] != str(model_parameter.dtype):
        raise ValueError("mini-batch checkpoint model dtype does not match")
    if checkpoint["model_device_type"] != model_parameter.device.type:
        raise ValueError(
            "mini-batch checkpoint model device type does not match"
        )
    if tuple(checkpoint["family_parameters"]) != tuple(
        model.family.parameter_names
    ):
        raise ValueError("mini-batch checkpoint family parameters do not match")
    if checkpoint["family_signature"] != _family_checkpoint_signature(model):
        raise ValueError("mini-batch checkpoint family does not match")
    _validate_checkpoint_control(checkpoint["control"], control)
    completed_epochs = checkpoint["completed_epochs"]
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs < 1
        or completed_epochs > control.epochs
    ):
        raise ValueError(
            "mini-batch checkpoint epoch is incompatible with control.epochs"
        )
    _validate_checkpoint_progress(
        checkpoint,
        completed_epochs=completed_epochs,
        has_validation=validation_metadata is not None,
    )
    saved_training_metadata = _metadata_from_checkpoint(
        checkpoint["training_metadata"],
        reference=training_metadata,
        context="training",
    )
    _validate_loader_metadata(
        training_metadata,
        saved_training_metadata,
        context="resumed training loader",
    )
    saved_validation = checkpoint["validation_metadata"]
    if (saved_validation is None) != (validation_metadata is None):
        raise ValueError(
            "mini-batch checkpoint validation-loader presence does not match"
        )
    if validation_metadata is not None:
        saved_validation_metadata = _metadata_from_checkpoint(
            saved_validation,
            reference=validation_metadata,
            context="validation",
        )
        _validate_loader_metadata(
            validation_metadata,
            saved_validation_metadata,
            context="resumed validation loader",
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint_version >= 2:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "mini-batch checkpoint state does not match the model"
        ) from error
    _restore_checkpoint_rng(
        checkpoint,
        model_device=model_parameter.device,
        loader=loader,
        validation_loader=validation_loader,
    )
    return checkpoint


def _validate_checkpoint_control(
    saved_control: Any,
    control: MiniBatchControl,
) -> None:
    if not isinstance(saved_control, dict):
        raise ValueError("mini-batch checkpoint control is invalid")
    current_control = asdict(control)
    ignored = {"epochs", "batch_size", "shuffle"}
    for name, current_value in current_control.items():
        if name in ignored:
            continue
        if name not in saved_control or saved_control[name] != current_value:
            raise ValueError(
                f"mini-batch checkpoint control {name!r} does not match"
            )


def _family_checkpoint_signature(model: GAMLSS) -> dict[str, Any]:
    family_type = type(model.family)
    return {
        "type": f"{family_type.__module__}.{family_type.__qualname__}",
        "name": model.family.name,
        "links": {
            parameter: (
                f"{type(link).__module__}.{type(link).__qualname__}"
            )
            for parameter, link in model.family.links.items()
        },
    }


def _validate_checkpoint_progress(
    checkpoint: Mapping[str, Any],
    *,
    completed_epochs: int,
    has_validation: bool,
) -> None:
    integer_fields = {
        "updates": checkpoint["updates"],
        "skipped_updates": checkpoint["skipped_updates"],
        "stable_evaluations": checkpoint["stable_evaluations"],
        "evaluations_without_improvement": checkpoint[
            "evaluations_without_improvement"
        ],
    }
    for name, value in integer_fields.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"mini-batch checkpoint field {name!r} is invalid"
            )
    if checkpoint["updates"] < completed_epochs:
        raise ValueError("mini-batch checkpoint update count is invalid")
    if checkpoint["skipped_updates"] > checkpoint["updates"]:
        raise ValueError(
            "mini-batch checkpoint skipped-update count is invalid"
        )
    objective_history = _checkpoint_float_sequence(
        checkpoint["objective_history"],
        context="objective history",
        nonempty=True,
    )
    evaluation_epochs = _checkpoint_epoch_sequence(
        checkpoint["evaluation_epochs"],
        context="evaluation epochs",
        completed_epochs=completed_epochs,
        nonempty=True,
    )
    if len(objective_history) != len(evaluation_epochs):
        raise ValueError(
            "mini-batch checkpoint objective history is misaligned"
        )
    validation_history = _checkpoint_float_sequence(
        checkpoint["validation_history"],
        context="validation history",
        nonempty=has_validation,
    )
    validation_epochs = _checkpoint_epoch_sequence(
        checkpoint["validation_epochs"],
        context="validation epochs",
        completed_epochs=completed_epochs,
        nonempty=has_validation,
    )
    if len(validation_history) != len(validation_epochs):
        raise ValueError(
            "mini-batch checkpoint validation history is misaligned"
        )
    best_validation_loss = checkpoint["best_validation_loss"]
    best_epoch = checkpoint["best_epoch"]
    best_state = checkpoint["best_state"]
    if has_validation:
        if validation_epochs != evaluation_epochs:
            raise ValueError(
                "mini-batch checkpoint validation epochs are misaligned"
            )
        if (
            isinstance(best_validation_loss, bool)
            or not isinstance(best_validation_loss, (int, float))
            or not math.isfinite(best_validation_loss)
        ):
            raise ValueError(
                "mini-batch checkpoint best validation loss is invalid"
            )
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch not in validation_epochs
        ):
            raise ValueError(
                "mini-batch checkpoint best epoch is invalid"
            )
        if not any(
            loss == float(best_validation_loss)
            for loss in validation_history
        ):
            raise ValueError(
                "mini-batch checkpoint best validation loss is inconsistent"
            )
        restore_best = checkpoint["control"].get(
            "restore_best_parameters"
        )
        if (
            restore_best is True
            and not isinstance(best_state, dict)
        ) or (
            restore_best is False
            and best_state is not None
        ):
            raise ValueError(
                "mini-batch checkpoint best model state is invalid"
            )
    elif (
        validation_history
        or validation_epochs
        or best_validation_loss is not None
        or best_epoch is not None
        or best_state is not None
    ):
        raise ValueError(
            "mini-batch checkpoint unexpectedly contains validation state"
        )
    mapping_fields = {
        "model_state_dict": checkpoint["model_state_dict"],
        "optimizer_state_dict": checkpoint["optimizer_state_dict"],
        "scheduler_state_dict": checkpoint["scheduler_state_dict"],
        "scaler_state_dict": checkpoint["scaler_state_dict"],
    }
    for name, value in mapping_fields.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"mini-batch checkpoint field {name!r} is invalid"
            )


def _checkpoint_float_sequence(
    value: Any,
    *,
    context: str,
    nonempty: bool,
) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"mini-batch checkpoint {context} is invalid")
    result = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
        ):
            raise ValueError(
                f"mini-batch checkpoint {context} is invalid"
            )
        result.append(float(item))
    if nonempty and not result:
        raise ValueError(f"mini-batch checkpoint {context} is empty")
    if not nonempty and result:
        raise ValueError(
            f"mini-batch checkpoint {context} must be empty"
        )
    return result


def _checkpoint_epoch_sequence(
    value: Any,
    *,
    context: str,
    completed_epochs: int,
    nonempty: bool,
) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"mini-batch checkpoint {context} is invalid")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        or item > completed_epochs
        for item in value
    ):
        raise ValueError(f"mini-batch checkpoint {context} is invalid")
    result = list(value)
    if nonempty:
        if (
            not result
            or result[0] != 0
            or any(
                current <= previous
                for previous, current in zip(result, result[1:])
            )
        ):
            raise ValueError(
                f"mini-batch checkpoint {context} is invalid"
            )
    elif result:
        raise ValueError(
            f"mini-batch checkpoint {context} must be empty"
        )
    return result


def _checkpoint_metadata(metadata: _LoaderMetadata) -> dict[str, Any]:
    return {
        "observation_count": metadata.observation_count,
        "total_weight": float(metadata.total_weight),
        "maximum_batch_size": metadata.maximum_batch_size,
        "batches": metadata.batches,
    }


def _metadata_from_checkpoint(
    value: Any,
    *,
    reference: _LoaderMetadata,
    context: str,
) -> _LoaderMetadata:
    if not isinstance(value, dict) or set(value) != {
        "observation_count",
        "total_weight",
        "maximum_batch_size",
        "batches",
    }:
        raise ValueError(
            f"mini-batch checkpoint {context} metadata is invalid"
        )
    saved_total_weight = value["total_weight"]
    if (
        isinstance(saved_total_weight, bool)
        or not isinstance(saved_total_weight, (int, float))
        or not math.isfinite(saved_total_weight)
        or saved_total_weight <= 0
    ):
        raise ValueError(
            f"mini-batch checkpoint {context} metadata is invalid"
        )
    try:
        total_weight = torch.as_tensor(
            saved_total_weight,
            dtype=reference.total_weight.dtype,
            device=reference.total_weight.device,
        )
        return _LoaderMetadata(
            observation_count=_checkpoint_integer(
                value["observation_count"],
                context=f"{context} observation count",
                minimum=1,
            ),
            total_weight=total_weight,
            maximum_batch_size=_checkpoint_integer(
                value["maximum_batch_size"],
                context=f"{context} maximum batch size",
                minimum=1,
            ),
            batches=_checkpoint_integer(
                value["batches"],
                context=f"{context} batch count",
                minimum=1,
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"mini-batch checkpoint {context} metadata is invalid"
        ) from error


def _checkpoint_integer(
    value: Any,
    *,
    context: str,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"mini-batch checkpoint {context} is invalid")
    return value


def _checkpoint_cpu_copy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {
            key: _checkpoint_cpu_copy(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_checkpoint_cpu_copy(nested) for nested in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_cpu_copy(nested) for nested in value)
    return value


def _loader_generator_states(loader: DataLoader[Any]) -> dict[str, Tensor]:
    states: dict[str, Tensor] = {}
    owners = {
        "loader": loader,
        "sampler": getattr(loader, "sampler", None),
        "batch_sampler": getattr(loader, "batch_sampler", None),
        "batch_sampler.sampler": getattr(
            getattr(loader, "batch_sampler", None),
            "sampler",
            None,
        ),
    }
    for name, owner in owners.items():
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator):
            states[name] = generator.get_state().clone()
    return states


def _restore_loader_generator_states(
    loader: DataLoader[Any],
    saved_states: Any,
    *,
    context: str,
) -> None:
    if not isinstance(saved_states, dict):
        raise ValueError(
            f"mini-batch checkpoint {context} generator states are invalid"
        )
    current_states = _loader_generator_states(loader)
    if set(current_states) != set(saved_states):
        raise ValueError(
            f"mini-batch checkpoint {context} generator configuration "
            "does not match"
        )
    owners = {
        "loader": loader,
        "sampler": getattr(loader, "sampler", None),
        "batch_sampler": getattr(loader, "batch_sampler", None),
        "batch_sampler.sampler": getattr(
            getattr(loader, "batch_sampler", None),
            "sampler",
            None,
        ),
    }
    for name, state in saved_states.items():
        if not isinstance(state, Tensor):
            raise ValueError(
                f"mini-batch checkpoint {context} generator state is invalid"
            )
        generator = getattr(owners[name], "generator")
        generator.set_state(state.cpu())


def _restore_checkpoint_rng(
    checkpoint: Mapping[str, Any],
    *,
    model_device: torch.device,
    loader: DataLoader[Any],
    validation_loader: DataLoader[Any] | None,
) -> None:
    torch_rng_state = checkpoint["torch_rng_state"]
    if not isinstance(torch_rng_state, Tensor):
        raise ValueError("mini-batch checkpoint Torch RNG state is invalid")
    torch.random.set_rng_state(torch_rng_state.cpu())
    cuda_rng_state = checkpoint["cuda_rng_state"]
    if model_device.type == "cuda":
        if not isinstance(cuda_rng_state, Tensor):
            raise ValueError("mini-batch checkpoint CUDA RNG state is missing")
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=model_device)
    elif cuda_rng_state is not None:
        raise ValueError(
            "CUDA mini-batch checkpoint cannot resume on a CPU model"
        )
    _restore_loader_generator_states(
        loader,
        checkpoint["training_loader_generators"],
        context="training loader",
    )
    saved_validation_states = checkpoint["validation_loader_generators"]
    if validation_loader is None:
        if saved_validation_states is not None:
            raise ValueError(
                "mini-batch checkpoint validation generator state does not match"
            )
    else:
        _restore_loader_generator_states(
            validation_loader,
            saved_validation_states,
            context="validation loader",
        )


def _loader_likelihood_values(
    model: GAMLSS,
    loader: DataLoader[Any],
    *,
    non_blocking: bool,
    context: str,
    expected_metadata: _LoaderMetadata | None = None,
) -> tuple[float, _LoaderMetadata]:
    model_parameter = next(model.parameters())
    training_mode = model.training
    model.eval()
    negative_log_likelihood = torch.zeros(
        (),
        dtype=model_parameter.dtype,
        device=model_parameter.device,
    )
    observation_count = 0
    total_weight = torch.zeros_like(negative_log_likelihood)
    maximum_batch_size = 0
    batches = 0
    try:
        with _preserve_loader_rng(loader), torch.no_grad():
            for batch in _loader_batches(
                model,
                loader,
                non_blocking=non_blocking,
                context=context,
            ):
                batch_response = batch[0]
                batch_weights = batch[2]
                batch_observations = batch_response.numel()
                observation_count += batch_observations
                total_weight += batch_weights.sum()
                maximum_batch_size = max(
                    maximum_batch_size,
                    batch_observations,
                )
                batches += 1
                negative_log_likelihood += _weighted_nll_sum(model, *batch)
    finally:
        model.train(training_mode)
    metadata = _LoaderMetadata(
        observation_count=observation_count,
        total_weight=total_weight.detach(),
        maximum_batch_size=maximum_batch_size,
        batches=batches,
    )
    _validate_loader_metadata(
        metadata,
        expected_metadata,
        context=context,
    )
    if not torch.isfinite(negative_log_likelihood):
        raise FloatingPointError(f"{context} negative log-likelihood is not finite")
    return float(negative_log_likelihood), metadata


def _loader_objective_values(
    model: GAMLSS,
    loader: DataLoader[Any],
    *,
    non_blocking: bool,
    expected_metadata: _LoaderMetadata | None = None,
) -> tuple[float, float, _LoaderMetadata]:
    negative_log_likelihood, metadata = _loader_likelihood_values(
        model,
        loader,
        non_blocking=non_blocking,
        context="training loader",
        expected_metadata=expected_metadata,
    )
    penalty = float(model.smooth_penalty().detach())
    penalized = (
        negative_log_likelihood + 0.5 * penalty
    ) / float(metadata.total_weight)
    if not math.isfinite(penalized):
        raise FloatingPointError("DataLoader full objective is not finite")
    return negative_log_likelihood, penalized, metadata


def _validate_loader_metadata(
    actual: _LoaderMetadata,
    expected: _LoaderMetadata | None,
    *,
    context: str,
) -> None:
    if (
        actual.batches < 1
        or actual.observation_count < 1
        or actual.maximum_batch_size < 1
    ):
        raise ValueError(f"{context} must yield at least one non-empty batch")
    if (
        not torch.isfinite(actual.total_weight)
        or actual.total_weight <= 0
    ):
        raise ValueError(
            f"{context} must have a finite positive total case weight"
        )
    if expected is None:
        return
    if actual.observation_count != expected.observation_count:
        raise ValueError(
            f"{context} changed its observation count from "
            f"{expected.observation_count} to {actual.observation_count}"
        )
    low_precision = expected.total_weight.dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }
    relative_tolerance = 1e-5 if low_precision else 1e-10
    absolute_tolerance = 1e-6 if low_precision else 1e-12
    if not math.isclose(
        float(actual.total_weight),
        float(expected.total_weight),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(
            f"{context} changed its total case weight from "
            f"{float(expected.total_weight)} to "
            f"{float(actual.total_weight)}"
        )


def _loader_full_gradient_max(
    model: GAMLSS,
    loader: DataLoader[Any],
    metadata: _LoaderMetadata,
    *,
    non_blocking: bool,
) -> float:
    parameters = list(model.parameters())
    training_mode = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    observation_count = 0
    total_weight = torch.zeros_like(metadata.total_weight)
    maximum_batch_size = 0
    batches = 0
    try:
        with _preserve_loader_rng(loader):
            for batch in _loader_batches(
                model,
                loader,
                non_blocking=non_blocking,
                context="training loader",
            ):
                batch_observations = batch[0].numel()
                observation_count += batch_observations
                total_weight += batch[2].sum().detach()
                maximum_batch_size = max(
                    maximum_batch_size,
                    batch_observations,
                )
                batches += 1
                (
                    _weighted_nll_sum(model, *batch)
                    / metadata.total_weight
                ).backward()
        pass_metadata = _LoaderMetadata(
            observation_count=observation_count,
            total_weight=total_weight,
            maximum_batch_size=maximum_batch_size,
            batches=batches,
        )
        _validate_loader_metadata(
            pass_metadata,
            metadata,
            context="training loader final-gradient pass",
        )
        penalty = model.smooth_penalty()
        if penalty.requires_grad:
            (0.5 * penalty / metadata.total_weight).backward()
        _validate_gradients(parameters)
        gradient_max = max(
            float(parameter.grad.detach().abs().max())
            for parameter in parameters
            if parameter.grad is not None
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(training_mode)
    return gradient_max


def _validate_inputs(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor | None,
    offsets: Mapping[str, Tensor] | None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None,
    neural_inputs: Mapping[str, Tensor] | None,
    shared_input: Tensor | None,
    *,
    require_positive_weight: bool = True,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    dict[str, dict[str, Tensor]],
    dict[str, Tensor],
    Tensor | None,
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

    case_weights = _validated_minibatch_weights(
        response,
        weights,
        require_positive_total=require_positive_weight,
    )
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

    if neural_inputs is None:
        neural_inputs = {}
    elif not isinstance(neural_inputs, Mapping):
        raise ValueError("neural inputs must be supplied as a mapping")
    expected_neural_parameters = set(model.neural_predictors)
    supplied_neural_parameters = set(neural_inputs)
    if expected_neural_parameters != supplied_neural_parameters:
        raise ValueError(
            "Neural inputs do not match configured predictors: "
            "missing="
            f"{sorted(expected_neural_parameters - supplied_neural_parameters)}, "
            f"extra={sorted(supplied_neural_parameters - expected_neural_parameters)}"
        )
    normalized_neural_inputs = {}
    for parameter, inputs in neural_inputs.items():
        if (
            not isinstance(inputs, Tensor)
            or inputs.ndim < 1
            or inputs.shape[0] != response.numel()
            or inputs.dtype != response.dtype
            or inputs.device != response.device
            or not torch.isfinite(inputs).all()
        ):
            raise ValueError(
                f"neural input for {parameter!r} must be finite with one row "
                "per response and matching dtype and device"
            )
        normalized_neural_inputs[parameter] = inputs
    if model.shared_predictor is None:
        if shared_input is not None:
            raise ValueError(
                "shared_input was supplied without a shared_predictor"
            )
        normalized_shared_input = None
    else:
        if (
            not isinstance(shared_input, Tensor)
            or shared_input.ndim < 1
            or shared_input.shape[0] != response.numel()
            or shared_input.dtype != response.dtype
            or shared_input.device != response.device
            or not torch.isfinite(shared_input).all()
        ):
            raise ValueError(
                "shared_input must be finite with one row per response and "
                "matching dtype and device"
            )
        normalized_shared_input = shared_input
    return (
        case_weights,
        normalized_offsets,
        normalized_smooth_covariates,
        normalized_neural_inputs,
        normalized_shared_input,
    )


def _validated_minibatch_weights(
    response: Tensor,
    weights: Tensor | None,
    *,
    require_positive_total: bool,
) -> Tensor:
    if weights is None:
        return torch.ones_like(response)
    if not isinstance(weights, Tensor):
        raise ValueError("weights must be supplied as a tensor")
    if weights.device != response.device:
        raise ValueError("weights must be on the same device as the response")
    try:
        observation_weights = torch.broadcast_to(weights, response.shape)
    except RuntimeError as error:
        raise ValueError("weights are not broadcastable to the response") from error
    if not torch.isfinite(observation_weights).all():
        raise ValueError("weights must be finite")
    if (observation_weights < 0).any():
        raise ValueError("weights must be non-negative")
    if require_positive_total and observation_weights.sum() <= 0:
        raise ValueError("at least one observation weight must be positive")
    return observation_weights


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
    neural_inputs: Mapping[str, Tensor],
    shared_input: Tensor | None,
    indices: Tensor,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    Tensor,
    dict[str, Tensor],
    dict[str, dict[str, Tensor]],
    dict[str, Tensor],
    Tensor | None,
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
        {
            parameter: inputs[indices]
            for parameter, inputs in neural_inputs.items()
        },
        shared_input[indices] if shared_input is not None else None,
    )


def _initialize_tensor_intercepts(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    neural_inputs: Mapping[str, Tensor],
    shared_input: Tensor | None,
    initial_parameters: Mapping[str, Any],
    *,
    batch_size: int,
    total_weight: Tensor,
) -> None:
    starts = model.family.initial_parameters(
        response,
        initial_parameters,
    )
    target_predictors = {
        parameter: model.family.links[parameter](starts[parameter])
        for parameter in model.family.parameter_names
    }
    weighted_residual_sums = {
        parameter: torch.zeros_like(total_weight)
        for parameter in model.family.parameter_names
    }
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
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
                    neural_inputs,
                    shared_input,
                    indices,
                )
                (
                    _,
                    batch_designs,
                    batch_weights,
                    batch_offsets,
                    batch_smooth,
                    batch_neural,
                    batch_shared,
                ) = batch
                contributions = model.term_contributions(
                    batch_designs,
                    batch_offsets,
                    smooth_covariates=batch_smooth,
                    neural_inputs=batch_neural,
                    shared_input=batch_shared,
                )
                for parameter in model.family.parameter_names:
                    design = batch_designs[parameter]
                    _validate_initialization_intercept(design, parameter)
                    current_intercept = (
                        model.coefficients[parameter][0].detach()
                    )
                    non_intercept_predictor = (
                        contributions[parameter].total
                        - design[:, 0] * current_intercept
                    )
                    weighted_residual_sums[parameter] += (
                        batch_weights
                        * (
                            target_predictors[parameter][indices]
                            - non_intercept_predictor
                        )
                    ).sum()
    finally:
        model.train(training_mode)
    with torch.no_grad():
        for parameter in model.family.parameter_names:
            model.coefficients[parameter][0].copy_(
                weighted_residual_sums[parameter] / total_weight
            )


def _validate_initialization_intercept(
    design: Tensor,
    parameter: str,
) -> None:
    if design.shape[1] < 1 or not torch.equal(
        design[:, 0],
        torch.ones_like(design[:, 0]),
    ):
        raise ValueError(
            "initial_parameters requires the first design column for "
            f"{parameter!r} to be an intercept containing only ones"
        )


def _weighted_nll_sum(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    neural_inputs: Mapping[str, Tensor],
    shared_input: Tensor | None,
) -> Tensor:
    losses = model.negative_log_likelihood(
        response,
        design_matrices,
        offsets=offsets,
        smooth_covariates=smooth_covariates,
        neural_inputs=neural_inputs,
        shared_input=shared_input,
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
    neural_inputs: Mapping[str, Tensor],
    shared_input: Tensor | None,
    batch_size: int,
    total_weight: Tensor,
) -> tuple[float, float]:
    training_mode = model.training
    model.eval()
    try:
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
                    neural_inputs,
                    shared_input,
                    indices,
                )
                negative_log_likelihood += _weighted_nll_sum(model, *batch)
            penalized = (
                negative_log_likelihood + 0.5 * model.smooth_penalty()
            ) / total_weight
    finally:
        model.train(training_mode)
    if not torch.isfinite(negative_log_likelihood) or not torch.isfinite(penalized):
        raise FloatingPointError("mini-batch full objective is not finite")
    return float(negative_log_likelihood), float(penalized)


def _validation_values(
    model: GAMLSS,
    validation: MiniBatchValidationData,
    validation_values: tuple[
        Tensor,
        dict[str, Tensor],
        dict[str, dict[str, Tensor]],
        dict[str, Tensor],
        Tensor | None,
    ],
    batch_size: int,
    total_weight: Tensor,
) -> tuple[float, float]:
    (
        weights,
        offsets,
        smooth_covariates,
        neural_inputs,
        shared_input,
    ) = validation_values
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            negative_log_likelihood = torch.zeros(
                (),
                dtype=validation.response.dtype,
                device=validation.response.device,
            )
            for indices in _sequential_indices(
                validation.response.numel(),
                batch_size,
                validation.response.device,
            ):
                batch = _slice_inputs(
                    validation.response,
                    validation.design_matrices,
                    weights,
                    offsets,
                    smooth_covariates,
                    neural_inputs,
                    shared_input,
                    indices,
                )
                negative_log_likelihood += _weighted_nll_sum(model, *batch)
            mean_loss = negative_log_likelihood / total_weight
    finally:
        model.train(training_mode)
    if (
        not torch.isfinite(negative_log_likelihood)
        or not torch.isfinite(mean_loss)
    ):
        raise FloatingPointError("mini-batch validation loss is not finite")
    return float(negative_log_likelihood), float(mean_loss)


def _state_dict_copy(model: GAMLSS) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _full_gradient_max(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    weights: Tensor,
    offsets: Mapping[str, Tensor],
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    neural_inputs: Mapping[str, Tensor],
    shared_input: Tensor | None,
    batch_size: int,
    total_weight: Tensor,
) -> float:
    parameters = list(model.parameters())
    training_mode = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
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
                neural_inputs,
                shared_input,
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
    finally:
        model.zero_grad(set_to_none=True)
        model.train(training_mode)
    return gradient_max


def _validate_gradients(parameters: list[Tensor]) -> None:
    if any(
        parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
        for parameter in parameters
    ):
        raise FloatingPointError("mini-batch gradient is not finite")
