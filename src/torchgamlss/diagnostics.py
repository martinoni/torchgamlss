"""Model-selection criteria and quantile residual diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from scipy.linalg import toeplitz
from scipy.stats import gaussian_kde, norm
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class ModelDiagnostics:
    """Likelihood criteria evaluated at a model's current coefficients."""

    log_likelihood: float
    global_deviance: float
    effective_degrees_of_freedom: float
    residual_degrees_of_freedom: float
    observation_count: int
    effective_observation_count: float

    def gaic(self, penalty: float = 2.0) -> float:
        """Return generalized AIC with caller-selected penalty ``k``."""
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("GAIC penalty must be finite and non-negative")
        return self.global_deviance + penalty * self.effective_degrees_of_freedom

    @property
    def aic(self) -> float:
        return self.gaic(2.0)

    @property
    def aicc(self) -> float:
        denominator = self.observation_count - self.effective_degrees_of_freedom - 1.0
        if denominator <= 0:
            return float("inf")
        correction = (
            2.0
            * self.effective_degrees_of_freedom
            * (self.effective_degrees_of_freedom + 1.0)
            / denominator
        )
        return self.aic + correction

    @property
    def bic(self) -> float:
        return (
            self.global_deviance
            + math.log(self.effective_observation_count)
            * self.effective_degrees_of_freedom
        )

    @property
    def sbc(self) -> float:
        """Alias for the Schwarz Bayesian criterion used by R GAMLSS."""
        return self.bic


@dataclass(frozen=True)
class QuantileResidualSummary:
    """R-compatible summary statistics for normal quantile residuals."""

    observation_count: int
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    filliben_correlation: float


@dataclass(frozen=True)
class ResidualDiagnosticPlot:
    """Matplotlib objects and values underlying a four-panel residual plot."""

    figure: Any
    axes: tuple[Any, Any, Any, Any]
    residuals: Tensor
    fitted_values: Tensor
    x_values: Tensor
    summary: QuantileResidualSummary


def model_diagnostics(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    neural_inputs: Mapping[str, Tensor] | None = None,
    shared_input: Tensor | None = None,
    degrees_of_freedom: float | None = None,
) -> ModelDiagnostics:
    """Evaluate likelihood criteria for the model's current fitted state."""
    losses = model.negative_log_likelihood(
        response,
        design_matrices,
        weights=weights,
        offsets=offsets,
        smooth_covariates=smooth_covariates,
        neural_inputs=neural_inputs,
        shared_input=shared_input,
        reduction="none",
    )
    if losses.ndim != 1 or losses.numel() != response.numel():
        raise ValueError("diagnostics require one likelihood value per response")
    case_weights = model._validated_weights(response, weights)
    if degrees_of_freedom is None:
        if any(
            model.smooth_terms[parameter] for parameter in model.family.parameter_names
        ) or model.neural_predictors or model.shared_predictor is not None:
            raise ValueError(
                "degrees_of_freedom is required for models with smooth or "
                "neural terms"
            )
        degrees_of_freedom = float(
            sum(
                model.coefficients[parameter].numel()
                for parameter in model.family.parameter_names
            )
        )
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be finite and positive")

    effective_observations = _effective_observation_count(case_weights)
    residual_degrees_of_freedom = effective_observations - degrees_of_freedom
    if residual_degrees_of_freedom <= 0:
        raise ValueError("diagnostics require positive residual degrees of freedom")
    negative_log_likelihood = float(losses.sum().detach())
    return ModelDiagnostics(
        log_likelihood=-negative_log_likelihood,
        global_deviance=2.0 * negative_log_likelihood,
        effective_degrees_of_freedom=degrees_of_freedom,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        observation_count=response.numel(),
        effective_observation_count=effective_observations,
    )


def quantile_residuals(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    neural_inputs: Mapping[str, Tensor] | None = None,
    shared_input: Tensor | None = None,
    uniforms: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return continuous or randomized discrete normal quantile residuals."""
    if response.ndim != 1 or response.numel() < 1 or not torch.isfinite(response).all():
        raise ValueError("quantile residuals require a non-empty finite response")
    model_parameter = next(model.parameters())
    if (
        response.dtype != model_parameter.dtype
        or response.device != model_parameter.device
    ):
        raise ValueError("response must match the model dtype and device")
    if uniforms is not None and generator is not None:
        raise ValueError("provide either uniforms or a generator, not both")
    parameters = model.predict(
        design_matrices,
        offsets,
        smooth_covariates=smooth_covariates,
        neural_inputs=neural_inputs,
        shared_input=shared_input,
        type="response",
    )
    model.family.validate_response(response, context="quantile residuals")
    if model.family.is_discrete:
        if uniforms is None:
            uniforms = torch.rand(
                response.shape,
                dtype=response.dtype,
                device=response.device,
                generator=generator,
            )
        elif (
            uniforms.shape != response.shape
            or uniforms.dtype != response.dtype
            or uniforms.device != response.device
            or not torch.isfinite(uniforms).all()
            or (uniforms < 0).any()
            or (uniforms > 1).any()
        ):
            raise ValueError(
                "uniforms must match the response and lie in the interval [0, 1]"
            )
        lower = model.family.cdf(response - 1.0, parameters)
        upper = model.family.cdf(response, parameters)
        probabilities = lower + uniforms * (upper - lower)
    else:
        if uniforms is not None:
            raise ValueError("uniforms apply only to discrete response families")
        if generator is not None:
            raise ValueError("generator applies only to discrete response families")
        probabilities = model.family.cdf(response, parameters)

    if (
        probabilities.shape != response.shape
        or probabilities.dtype != response.dtype
        or probabilities.device != response.device
        or not torch.isfinite(probabilities).all()
    ):
        raise RuntimeError("family CDF returned invalid probabilities")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("family CDF probabilities must lie in [0, 1]")
    epsilon = torch.finfo(response.dtype).eps
    probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
    return torch.special.ndtri(probabilities)


def plot_residual_diagnostics(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    neural_inputs: Mapping[str, Tensor] | None = None,
    shared_input: Tensor | None = None,
    x_variable: Tensor | None = None,
    x_label: str | None = None,
    fitted_parameter: str | None = None,
    uniforms: Tensor | None = None,
    generator: torch.Generator | None = None,
    time_series: bool = False,
    max_lag: int | None = None,
    axes: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> ResidualDiagnosticPlot:
    """Plot the four normal-quantile-residual diagnostics used by R GAMLSS."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "plot() requires matplotlib; install TorchGAMLSS with its "
            "declared plotting dependency"
        ) from error

    if not isinstance(time_series, bool):
        raise ValueError("time_series must be boolean")
    if time_series and x_variable is not None:
        raise ValueError("x_variable cannot be supplied for time-series diagnostics")
    if not time_series and max_lag is not None:
        raise ValueError("max_lag requires time_series=True")
    if x_label is not None and (
        not isinstance(x_label, str) or not x_label
    ):
        raise ValueError("x_label must be a non-empty string when supplied")
    if fitted_parameter is None:
        fitted_parameter = model.family.parameter_names[0]
    if fitted_parameter not in model.family.parameter_names:
        raise ValueError(
            f"fitted_parameter must be one of {model.family.parameter_names}"
        )
    indices = _diagnostic_observation_indices(model, response, weights)
    (
        expanded_response,
        expanded_designs,
        expanded_offsets,
        expanded_smooth,
        expanded_neural,
        expanded_shared,
    ) = _expanded_diagnostic_inputs(
        response,
        design_matrices,
        offsets,
        smooth_covariates,
        neural_inputs,
        shared_input,
        indices,
    )
    expanded_uniforms = _expanded_uniforms(
        uniforms,
        response,
        indices,
    )
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            residuals = quantile_residuals(
                model,
                expanded_response,
                expanded_designs,
                offsets=expanded_offsets,
                smooth_covariates=expanded_smooth,
                neural_inputs=expanded_neural,
                shared_input=expanded_shared,
                uniforms=expanded_uniforms,
                generator=generator,
            ).detach().cpu()
            fitted_values = model.predict(
                expanded_designs,
                expanded_offsets,
                smooth_covariates=expanded_smooth,
                neural_inputs=expanded_neural,
                shared_input=expanded_shared,
                type="response",
            )[fitted_parameter].detach().cpu()
    finally:
        model.train(training_mode)
    if residuals.numel() < 3:
        raise ValueError("plot diagnostics require at least three observations")
    if torch.unique(residuals).numel() < 2:
        raise ValueError("plot diagnostics require non-constant residuals")

    if x_variable is None:
        x_values = torch.arange(
            1,
            residuals.numel() + 1,
            dtype=response.dtype,
        )
        x_axis_label = x_label or "Observation"
    else:
        if (
            not isinstance(x_variable, Tensor)
            or x_variable.ndim != 1
            or x_variable.shape != response.shape
            or not torch.isfinite(x_variable).all()
        ):
            raise ValueError(
                "x_variable must be a finite vector with one value per response"
            )
        x_values = x_variable.to(device=indices.device)[indices].detach().cpu()
        x_axis_label = x_label or "Explanatory variable"

    residual_values = residuals.numpy()
    fitted_array = fitted_values.numpy()
    x_array = x_values.numpy()
    summary, theoretical_quantiles, ordered_residuals = _residual_summary(
        residual_values
    )

    if time_series:
        autocorrelation, partial_autocorrelation = _correlations(
            residual_values,
            max_lag=max_lag,
        )
    else:
        autocorrelation = None
        partial_autocorrelation = None
    plot_axes, figure = _diagnostic_axes(plt, axes, figsize)
    if time_series:
        assert autocorrelation is not None
        assert partial_autocorrelation is not None
        _plot_correlation_panel(
            plot_axes[0],
            autocorrelation,
            title="Residual autocorrelation",
            ylabel="ACF",
            observation_count=residuals.numel(),
        )
        _plot_correlation_panel(
            plot_axes[1],
            partial_autocorrelation,
            title="Residual partial autocorrelation",
            ylabel="PACF",
            observation_count=residuals.numel(),
        )
    else:
        _scatter_residual_panel(
            plot_axes[0],
            fitted_array,
            residual_values,
            title=f"Residuals vs fitted {fitted_parameter}",
            xlabel=f"Fitted {fitted_parameter}",
        )
        _scatter_residual_panel(
            plot_axes[1],
            x_array,
            residual_values,
            title=f"Residuals vs {x_axis_label.lower()}",
            xlabel=x_axis_label,
        )
    _density_panel(plot_axes[2], residual_values)
    _qq_panel(
        plot_axes[3],
        theoretical_quantiles,
        ordered_residuals,
    )
    figure.tight_layout()
    return ResidualDiagnosticPlot(
        figure=figure,
        axes=plot_axes,
        residuals=residuals,
        fitted_values=fitted_values,
        x_values=x_values,
        summary=summary,
    )


def compare_models(
    diagnostics: Mapping[str, ModelDiagnostics],
    *,
    criterion: Literal["aic", "aicc", "bic", "gaic"] = "aic",
    penalty: float = 2.0,
) -> pd.DataFrame:
    """Rank comparable fitted models and calculate criterion weights."""
    if not diagnostics:
        raise ValueError("at least one model diagnostic is required")
    if criterion not in {"aic", "aicc", "bic", "gaic"}:
        raise ValueError("criterion must be one of: aic, aicc, bic, gaic")
    observation_counts = {
        (
            result.observation_count,
            result.effective_observation_count,
        )
        for result in diagnostics.values()
    }
    if len(observation_counts) != 1:
        raise ValueError("model diagnostics must use comparable observations")

    rows = []
    for name, result in diagnostics.items():
        value = (
            result.gaic(penalty) if criterion == "gaic" else getattr(result, criterion)
        )
        if not math.isfinite(value):
            raise ValueError("model comparison criterion values must be finite")
        rows.append(
            {
                "model": name,
                "degrees_of_freedom": result.effective_degrees_of_freedom,
                "global_deviance": result.global_deviance,
                "criterion": value,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["criterion", "model"],
        ignore_index=True,
    )
    table["delta"] = table["criterion"] - table["criterion"].min()
    relative_likelihood = (-0.5 * table["delta"]).map(math.exp)
    table["weight"] = relative_likelihood / relative_likelihood.sum()
    return table.set_index("model")


def _effective_observation_count(weights: Tensor) -> float:
    if torch.equal(weights, weights.round()):
        return float(weights.sum())
    return float((weights > 0).sum())


def _diagnostic_axes(
    pyplot: Any,
    axes: Sequence[Any] | None,
    figsize: tuple[float, float],
) -> tuple[tuple[Any, Any, Any, Any], Any]:
    if axes is None:
        figure, raw_axes = pyplot.subplots(2, 2, figsize=figsize)
        flattened = tuple(raw_axes.reshape(-1))
        return (
            flattened[0],
            flattened[1],
            flattened[2],
            flattened[3],
        ), figure
    flattened = tuple(np.asarray(axes, dtype=object).reshape(-1))
    if len(flattened) != 4:
        raise ValueError("axes must contain exactly four Matplotlib axes")
    figures = {id(axis.figure): axis.figure for axis in flattened}
    if len(figures) != 1:
        raise ValueError("all diagnostic axes must belong to the same figure")
    return (
        flattened[0],
        flattened[1],
        flattened[2],
        flattened[3],
    ), next(iter(figures.values()))


def _diagnostic_observation_indices(
    model: GAMLSS,
    response: Tensor,
    weights: Tensor | None,
) -> Tensor:
    case_weights = model._validated_weights(response, weights)
    if torch.equal(case_weights, case_weights.round()):
        repetitions = case_weights.round().to(dtype=torch.long)
    else:
        repetitions = (case_weights > 0).to(dtype=torch.long)
    indices = torch.repeat_interleave(
        torch.arange(response.numel(), device=response.device),
        repetitions,
    )
    if indices.numel() < 1:
        raise ValueError("plot diagnostics require at least one positive weight")
    return indices


def _expanded_diagnostic_inputs(
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    offsets: Mapping[str, Tensor] | None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None,
    neural_inputs: Mapping[str, Tensor] | None,
    shared_input: Tensor | None,
    indices: Tensor,
) -> tuple[
    Tensor,
    dict[str, Tensor],
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
        {
            parameter: offset[indices]
            for parameter, offset in (offsets or {}).items()
        },
        {
            parameter: {
                term: covariate[indices]
                for term, covariate in parameter_covariates.items()
            }
            for parameter, parameter_covariates in (
                smooth_covariates or {}
            ).items()
        },
        {
            parameter: inputs[indices]
            for parameter, inputs in (neural_inputs or {}).items()
        },
        shared_input[indices] if shared_input is not None else None,
    )


def _expanded_uniforms(
    uniforms: Tensor | None,
    response: Tensor,
    indices: Tensor,
) -> Tensor | None:
    if uniforms is None:
        return None
    if uniforms.shape == response.shape:
        return uniforms[indices]
    if uniforms.shape == indices.shape:
        return uniforms
    raise ValueError(
        "uniforms must have one value per response or expanded frequency weight"
    )


def _plotting_positions(observation_count: int) -> np.ndarray:
    adjustment = 3.0 / 8.0 if observation_count <= 10 else 0.5
    ranks = np.arange(1, observation_count + 1, dtype=np.float64)
    return (ranks - adjustment) / (
        observation_count + 1.0 - 2.0 * adjustment
    )


def _residual_summary(
    residuals: np.ndarray,
) -> tuple[QuantileResidualSummary, np.ndarray, np.ndarray]:
    ordered = np.sort(residuals.astype(np.float64, copy=False))
    theoretical = norm.ppf(_plotting_positions(ordered.size))
    mean = float(np.mean(ordered))
    variance = float(np.var(ordered, ddof=1))
    centered = ordered - mean
    third_moment = float(np.mean(centered**3))
    fourth_moment = float(np.mean(centered**4))
    skewness = third_moment / variance**1.5
    kurtosis = fourth_moment / variance**2
    filliben = float(np.corrcoef(theoretical, ordered)[0, 1])
    return (
        QuantileResidualSummary(
            observation_count=ordered.size,
            mean=mean,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            filliben_correlation=filliben,
        ),
        theoretical,
        ordered,
    )


def _scatter_residual_panel(
    axis: Any,
    x_values: np.ndarray,
    residuals: np.ndarray,
    *,
    title: str,
    xlabel: str,
) -> None:
    axis.scatter(
        x_values,
        residuals,
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Quantile residuals")
    axis.grid(alpha=0.2)


def _density_panel(axis: Any, residuals: np.ndarray) -> None:
    standard_deviation = float(np.std(residuals, ddof=1))
    lower = min(-4.0, float(np.min(residuals) - standard_deviation))
    upper = max(4.0, float(np.max(residuals) + standard_deviation))
    grid = np.linspace(lower, upper, 400)
    density = gaussian_kde(residuals)
    axis.plot(grid, density(grid), label="Residual KDE", linewidth=1.8)
    axis.plot(
        grid,
        norm.pdf(grid),
        label="Standard normal",
        linestyle="--",
        linewidth=1.2,
    )
    rug_height = max(float(np.max(density(grid))) * 0.025, 1e-6)
    axis.vlines(
        residuals,
        0.0,
        rug_height,
        color="0.35",
        alpha=0.35,
        linewidth=0.7,
    )
    axis.set_title("Residual density")
    axis.set_xlabel("Quantile residuals")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)


def _qq_panel(
    axis: Any,
    theoretical_quantiles: np.ndarray,
    ordered_residuals: np.ndarray,
) -> None:
    lower = min(
        float(theoretical_quantiles[0]),
        float(ordered_residuals[0]),
    )
    upper = max(
        float(theoretical_quantiles[-1]),
        float(ordered_residuals[-1]),
    )
    axis.scatter(
        theoretical_quantiles,
        ordered_residuals,
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    axis.plot([lower, upper], [lower, upper], color="0.35", linewidth=1.0)
    axis.set_title("Normal Q-Q plot")
    axis.set_xlabel("Theoretical quantiles")
    axis.set_ylabel("Sample quantiles")
    axis.grid(alpha=0.2)


def _correlations(
    residuals: np.ndarray,
    *,
    max_lag: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    observation_count = residuals.size
    maximum_allowed = min(40, observation_count // 2)
    if max_lag is None:
        max_lag = maximum_allowed
    if (
        isinstance(max_lag, bool)
        or not isinstance(max_lag, int)
        or max_lag < 1
        or max_lag > maximum_allowed
    ):
        raise ValueError(
            f"max_lag must be an integer between 1 and {maximum_allowed}"
        )
    centered = residuals.astype(np.float64, copy=False) - np.mean(residuals)
    denominator = float(np.dot(centered, centered))
    autocorrelation = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        autocorrelation[lag - 1] = (
            np.dot(centered[:-lag], centered[lag:]) / denominator
        )
    correlations_with_zero = np.concatenate(([1.0], autocorrelation))
    partial = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        system = toeplitz(correlations_with_zero[:lag])
        partial[lag - 1] = np.linalg.lstsq(
            system,
            correlations_with_zero[1 : lag + 1],
            rcond=None,
        )[0][-1]
    return autocorrelation, partial


def _plot_correlation_panel(
    axis: Any,
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    observation_count: int,
) -> None:
    lags = np.arange(1, values.size + 1)
    confidence_limit = 1.96 / math.sqrt(observation_count)
    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.axhline(
        confidence_limit,
        color="0.5",
        linestyle="--",
        linewidth=0.9,
    )
    axis.axhline(
        -confidence_limit,
        color="0.5",
        linestyle="--",
        linewidth=0.9,
    )
    axis.vlines(lags, 0.0, values, linewidth=1.2)
    axis.scatter(lags, values, s=16)
    axis.set_title(title)
    axis.set_xlabel("Lag")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
