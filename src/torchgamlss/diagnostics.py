"""Model-selection criteria and quantile residual diagnostics."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from scipy.linalg import toeplitz
from scipy.stats import chi2, gaussian_kde, norm
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


@dataclass(frozen=True)
class WormPlotPanel:
    """Numerical values underlying one worm-plot panel."""

    axis: Any
    residuals: Tensor
    theoretical_quantiles: Tensor
    deviations: Tensor
    confidence_grid: Tensor
    confidence_lower: Tensor
    confidence_upper: Tensor
    coefficients: Tensor | None
    interval: tuple[float, float] | None


@dataclass(frozen=True)
class WormPlotResult:
    """Matplotlib objects and detrended normal Q-Q values for ``wp()``."""

    figure: Any
    axes: tuple[Any, ...]
    residuals: Tensor
    x_values: Tensor | None
    intervals: Tensor | None
    panels: tuple[WormPlotPanel, ...]
    coefficients: Tensor | None


@dataclass(frozen=True)
class BucketStatistics:
    """Skewness and kurtosis coordinates underlying a bucket plot."""

    observation_count: int
    effective_observation_count: float
    skewness: float
    transformed_skewness: float
    kurtosis: float
    excess_kurtosis: float
    transformed_kurtosis: float
    jarque_bera: float | None

    @property
    def point(self) -> tuple[float, float]:
        """Return the displayed skewness-kurtosis coordinate."""
        return self.transformed_skewness, self.transformed_kurtosis


@dataclass(frozen=True)
class BucketPlotPanel:
    """Numerical values and Matplotlib axis for one bucket-plot panel."""

    axis: Any
    statistics: BucketStatistics
    bootstrap_points: Tensor
    interval: tuple[float, float] | None


@dataclass(frozen=True)
class BucketPlotResult:
    """Matplotlib objects and statistics returned by ``bp()``."""

    figure: Any
    axes: tuple[Any, ...]
    residuals: Tensor
    weights: Tensor
    x_values: Tensor | None
    intervals: Tensor | None
    panels: tuple[BucketPlotPanel, ...]
    kind: Literal["moment", "centile.central", "centile.tail"]


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


def worm_plot_diagnostics(
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
    uniforms: Tensor | None = None,
    generator: torch.Generator | None = None,
    n_intervals: int = 4,
    cut_points: Sequence[float] | Tensor | None = None,
    overlap: float = 0.0,
    x_limit: float | None = None,
    y_limit: float | None = None,
    show_intervals: bool = True,
    show_polynomial: bool = True,
    axes: Sequence[Any] | Any | None = None,
    figsize: tuple[float, float] | None = None,
) -> WormPlotResult:
    """Calculate model quantile residuals and draw an R-style worm plot."""
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
    expanded_uniforms = _expanded_uniforms(uniforms, response, indices)
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
    finally:
        model.train(training_mode)

    expanded_x = None
    if x_variable is not None:
        if (
            not isinstance(x_variable, Tensor)
            or x_variable.ndim != 1
            or x_variable.shape != response.shape
            or not torch.isfinite(x_variable).all()
        ):
            raise ValueError(
                "x_variable must be a finite vector with one value per response"
            )
        expanded_x = x_variable.to(device=indices.device)[indices].detach().cpu()
    return worm_plot(
        residuals,
        x_variable=expanded_x,
        x_label=x_label,
        n_intervals=n_intervals,
        cut_points=cut_points,
        overlap=overlap,
        x_limit=x_limit,
        y_limit=y_limit,
        show_intervals=show_intervals,
        show_polynomial=show_polynomial,
        axes=axes,
        figsize=figsize,
    )


def worm_plot(
    residuals: Tensor | Sequence[float] | np.ndarray,
    *,
    x_variable: Tensor | Sequence[float] | np.ndarray | None = None,
    x_label: str | None = None,
    n_intervals: int = 4,
    cut_points: Sequence[float] | Tensor | None = None,
    overlap: float = 0.0,
    x_limit: float | None = None,
    y_limit: float | None = None,
    show_intervals: bool = True,
    show_polynomial: bool = True,
    axes: Sequence[Any] | Any | None = None,
    figsize: tuple[float, float] | None = None,
) -> WormPlotResult:
    """Draw a global or covariate-conditioned detrended normal Q-Q plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "wp() requires matplotlib; install TorchGAMLSS with its "
            "declared plotting dependency"
        ) from error

    residual_tensor = _finite_cpu_vector(residuals, context="residuals")
    if residual_tensor.numel() < 2:
        raise ValueError("worm plots require at least two residuals")
    if x_label is not None and (not isinstance(x_label, str) or not x_label):
        raise ValueError("x_label must be a non-empty string when supplied")
    if not isinstance(n_intervals, int) or isinstance(n_intervals, bool):
        raise ValueError("n_intervals must be an integer")
    if n_intervals < 1:
        raise ValueError("n_intervals must be positive")
    if (
        not isinstance(overlap, (int, float))
        or isinstance(overlap, bool)
        or not math.isfinite(float(overlap))
        or not 0.0 <= float(overlap) < 1.0
    ):
        raise ValueError("overlap must lie in [0, 1)")
    if not isinstance(show_intervals, bool):
        raise ValueError("show_intervals must be boolean")
    if not isinstance(show_polynomial, bool):
        raise ValueError("show_polynomial must be boolean")
    if x_limit is not None and (
        not isinstance(x_limit, (int, float))
        or isinstance(x_limit, bool)
        or not math.isfinite(float(x_limit))
        or float(x_limit) <= 0.0
    ):
        raise ValueError("x_limit must be finite and positive")
    if y_limit is not None and (
        not isinstance(y_limit, (int, float))
        or isinstance(y_limit, bool)
        or not math.isfinite(float(y_limit))
        or float(y_limit) <= 0.0
    ):
        raise ValueError("y_limit must be finite and positive")

    residual_values = residual_tensor.numpy().astype(np.float64, copy=False)
    x_tensor = None
    intervals = None
    panel_masks: list[np.ndarray]
    if x_variable is None:
        if cut_points is not None:
            raise ValueError("cut_points requires x_variable")
        panel_masks = [np.ones(residual_values.size, dtype=bool)]
    else:
        x_tensor = _finite_cpu_vector(x_variable, context="x_variable")
        if x_tensor.shape != residual_tensor.shape:
            raise ValueError(
                "x_variable must contain one value per residual"
            )
        x_values = x_tensor.numpy().astype(np.float64, copy=False)
        if cut_points is None:
            interval_values = _co_intervals(
                x_values,
                number=n_intervals,
                overlap=float(overlap),
            )
        else:
            interval_values = _cut_point_intervals(x_values, cut_points)
        panel_masks = [
            (x_values >= lower) & (x_values <= upper)
            for lower, upper in interval_values
        ]
        if any(not np.any(mask) for mask in panel_masks):
            raise ValueError("each worm-plot interval must contain observations")
        intervals = torch.from_numpy(interval_values.copy())

    panel_count = len(panel_masks)
    plot_axes, figure = _worm_axes(
        plt,
        axes,
        panel_count=panel_count,
        figsize=figsize,
    )
    selected_x_limit = float(
        x_limit if x_limit is not None else (4.0 if x_tensor is None else 3.5)
    )
    selected_y_limit = float(
        y_limit
        if y_limit is not None
        else (
            12.0 / math.sqrt(residual_values.size)
            if x_tensor is None
            else 12.0 * math.sqrt(n_intervals / residual_values.size)
        )
    )

    panels = []
    coefficient_rows = []
    for panel_index, (axis, mask) in enumerate(zip(plot_axes, panel_masks)):
        interval = (
            None
            if intervals is None
            else (
                float(intervals[panel_index, 0]),
                float(intervals[panel_index, 1]),
            )
        )
        panel = _worm_panel(
            axis,
            residual_values[mask],
            x_limit=selected_x_limit,
            y_limit=selected_y_limit,
            show_polynomial=show_polynomial,
            interval=interval,
            x_label=x_label,
            show_interval=show_intervals,
        )
        panels.append(panel)
        if panel.coefficients is not None:
            coefficient_rows.append(panel.coefficients)
    figure.tight_layout()
    coefficients = (
        torch.stack(coefficient_rows)
        if show_polynomial
        else None
    )
    if coefficients is not None and x_tensor is None:
        coefficients = coefficients[0]
    return WormPlotResult(
        figure=figure,
        axes=plot_axes,
        residuals=residual_tensor,
        x_values=x_tensor,
        intervals=intervals,
        panels=tuple(panels),
        coefficients=coefficients,
    )


def bucket_plot_diagnostics(
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
    uniforms: Tensor | None = None,
    residual_generator: torch.Generator | None = None,
    kind: Literal["moment", "centile.central", "centile.tail"] = "moment",
    bootstrap: bool = True,
    bootstrap_replicates: int = 99,
    bootstrap_generator: torch.Generator | None = None,
    n_intervals: int = 4,
    cut_points: Sequence[float] | Tensor | None = None,
    overlap: float = 0.0,
    label: str | None = None,
    show_intervals: bool = True,
    show_legend: bool = False,
    axes: Sequence[Any] | Any | None = None,
    figsize: tuple[float, float] | None = None,
) -> BucketPlotResult:
    """Calculate model quantile residuals and draw an R-style bucket plot."""
    case_weights = model._validated_weights(response, weights).detach().cpu()
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            residuals = quantile_residuals(
                model,
                response,
                design_matrices,
                offsets=offsets,
                smooth_covariates=smooth_covariates,
                neural_inputs=neural_inputs,
                shared_input=shared_input,
                uniforms=uniforms,
                generator=residual_generator,
            ).detach().cpu()
    finally:
        model.train(training_mode)

    x_values = None
    if x_variable is not None:
        if (
            not isinstance(x_variable, Tensor)
            or x_variable.ndim != 1
            or x_variable.shape != response.shape
            or not torch.isfinite(x_variable).all()
        ):
            raise ValueError(
                "x_variable must be a finite vector with one value per response"
            )
        x_values = x_variable.detach().cpu()
    return bucket_plot(
        residuals,
        weights=case_weights,
        x_variable=x_values,
        x_label=x_label,
        kind=kind,
        bootstrap=bootstrap,
        bootstrap_replicates=bootstrap_replicates,
        generator=bootstrap_generator,
        n_intervals=n_intervals,
        cut_points=cut_points,
        overlap=overlap,
        label=label,
        show_intervals=show_intervals,
        show_legend=show_legend,
        axes=axes,
        figsize=figsize,
    )


def bucket_plot(
    residuals: Tensor | Sequence[float] | np.ndarray,
    *,
    weights: Tensor | Sequence[float] | np.ndarray | None = None,
    x_variable: Tensor | Sequence[float] | np.ndarray | None = None,
    x_label: str | None = None,
    kind: Literal["moment", "centile.central", "centile.tail"] = "moment",
    bootstrap: bool = True,
    bootstrap_replicates: int = 99,
    generator: torch.Generator | None = None,
    n_intervals: int = 4,
    cut_points: Sequence[float] | Tensor | None = None,
    overlap: float = 0.0,
    label: str | None = None,
    show_intervals: bool = True,
    show_legend: bool = False,
    axes: Sequence[Any] | Any | None = None,
    figsize: tuple[float, float] | None = None,
) -> BucketPlotResult:
    """Plot transformed skewness and kurtosis for arbitrary residuals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "bp() requires matplotlib; install TorchGAMLSS with its "
            "declared plotting dependency"
        ) from error

    residual_tensor = _finite_cpu_vector(residuals, context="residuals")
    if residual_tensor.numel() < 4:
        raise ValueError("bucket plots require at least four residuals")
    if weights is None:
        weight_tensor = torch.ones(
            residual_tensor.shape,
            dtype=torch.float64,
        )
    else:
        weight_tensor = _finite_cpu_vector(weights, context="weights").to(
            dtype=torch.float64
        )
        if weight_tensor.shape != residual_tensor.shape:
            raise ValueError("weights must contain one value per residual")
        if (weight_tensor < 0).any() or not torch.any(weight_tensor > 0):
            raise ValueError("weights must be non-negative with a positive sum")
    if kind not in {"moment", "centile.central", "centile.tail"}:
        raise ValueError(
            "kind must be 'moment', 'centile.central', or 'centile.tail'"
        )
    if not isinstance(bootstrap, bool):
        raise ValueError("bootstrap must be boolean")
    if (
        not isinstance(bootstrap_replicates, int)
        or isinstance(bootstrap_replicates, bool)
        or bootstrap_replicates < 0
        or (bootstrap and bootstrap_replicates < 1)
    ):
        raise ValueError(
            "bootstrap_replicates must be positive when bootstrap is enabled"
        )
    if not isinstance(n_intervals, int) or isinstance(n_intervals, bool):
        raise ValueError("n_intervals must be an integer")
    if n_intervals < 1:
        raise ValueError("n_intervals must be positive")
    if (
        not isinstance(overlap, (int, float))
        or isinstance(overlap, bool)
        or not math.isfinite(float(overlap))
        or not 0.0 <= float(overlap) < 1.0
    ):
        raise ValueError("overlap must lie in [0, 1)")
    if x_label is not None and (not isinstance(x_label, str) or not x_label):
        raise ValueError("x_label must be a non-empty string when supplied")
    if label is not None and (not isinstance(label, str) or not label):
        raise ValueError("label must be a non-empty string when supplied")
    if not isinstance(show_intervals, bool):
        raise ValueError("show_intervals must be boolean")
    if not isinstance(show_legend, bool):
        raise ValueError("show_legend must be boolean")

    residual_values = residual_tensor.numpy().astype(np.float64, copy=False)
    weight_values = weight_tensor.numpy()
    x_tensor = None
    intervals = None
    panel_masks: list[np.ndarray]
    if x_variable is None:
        if cut_points is not None:
            raise ValueError("cut_points requires x_variable")
        panel_masks = [np.ones(residual_values.size, dtype=bool)]
    else:
        x_tensor = _finite_cpu_vector(x_variable, context="x_variable")
        if x_tensor.shape != residual_tensor.shape:
            raise ValueError(
                "x_variable must contain one value per residual"
            )
        x_values = x_tensor.numpy().astype(np.float64, copy=False)
        if cut_points is None:
            interval_values = _co_intervals(
                x_values,
                number=n_intervals,
                overlap=float(overlap),
            )
        else:
            interval_values = _cut_point_intervals(x_values, cut_points)
        panel_masks = [
            (x_values >= lower) & (x_values <= upper)
            for lower, upper in interval_values
        ]
        if any(
            not np.any(mask & (weight_values > 0.0))
            for mask in panel_masks
        ):
            raise ValueError(
                "each bucket-plot interval must contain positive-weight "
                "observations"
            )
        intervals = torch.from_numpy(interval_values.copy())

    plot_axes, figure = _bucket_axes(
        plt,
        axes,
        panel_count=len(panel_masks),
        figsize=figsize,
    )
    panels = []
    for panel_index, (axis, mask) in enumerate(zip(plot_axes, panel_masks)):
        panel_residuals = residual_values[mask]
        panel_weights = weight_values[mask]
        statistics = _bucket_statistics(
            panel_residuals,
            panel_weights,
            kind=kind,
        )
        bootstrap_points = _bucket_bootstrap(
            panel_residuals,
            panel_weights,
            kind=kind,
            replicates=bootstrap_replicates if bootstrap else 0,
            generator=generator,
        )
        interval = (
            None
            if intervals is None
            else (
                float(intervals[panel_index, 0]),
                float(intervals[panel_index, 1]),
            )
        )
        _draw_bucket_panel(
            axis,
            statistics,
            bootstrap_points,
            kind=kind,
            interval=interval,
            x_label=x_label,
            label=label,
            show_interval=show_intervals,
            show_legend=show_legend,
        )
        panels.append(
            BucketPlotPanel(
                axis=axis,
                statistics=statistics,
                bootstrap_points=torch.from_numpy(
                    bootstrap_points.copy()
                ),
                interval=interval,
            )
        )
    figure.tight_layout()
    return BucketPlotResult(
        figure=figure,
        axes=plot_axes,
        residuals=residual_tensor,
        weights=weight_tensor,
        x_values=x_tensor,
        intervals=intervals,
        panels=tuple(panels),
        kind=kind,
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


def _finite_cpu_vector(
    values: Tensor | Sequence[float] | np.ndarray,
    *,
    context: str,
) -> Tensor:
    if isinstance(values, Tensor):
        tensor = values.detach().cpu()
    else:
        try:
            tensor = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{context} must be a numeric vector") from error
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{context} must be a non-empty vector")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{context} must contain only finite values")
    return tensor


def _co_intervals(
    values: np.ndarray,
    *,
    number: int,
    overlap: float,
) -> np.ndarray:
    """Reproduce the interval calculation in R graphics::co.intervals()."""
    observation_count = values.size
    if number > observation_count:
        raise ValueError(
            "n_intervals cannot exceed the number of observations"
        )
    ordered = np.sort(values)
    panel_size = observation_count / (
        number * (1.0 - overlap) + overlap
    )
    offsets = (
        np.arange(number, dtype=np.float64)
        * (1.0 - overlap)
        * panel_size
    )
    lower_indices = np.rint(1.0 + offsets).astype(np.int64) - 1
    upper_indices = np.rint(panel_size + offsets).astype(np.int64) - 1
    lower_indices = np.clip(lower_indices, 0, observation_count - 1)
    upper_indices = np.clip(upper_indices, 0, observation_count - 1)
    lower_values = ordered[lower_indices]
    upper_values = ordered[upper_indices]
    keep = np.concatenate(
        (
            np.array([True]),
            (np.diff(lower_values) > 0.0) | (np.diff(upper_values) > 0.0),
        )
    )
    positive_jumps = np.diff(ordered)
    positive_jumps = positive_jumps[positive_jumps > 0.0]
    epsilon = (
        0.5 * float(np.min(positive_jumps))
        if positive_jumps.size
        else 0.0
    )
    intervals = np.column_stack(
        (
            lower_values[keep] - epsilon,
            upper_values[keep] + epsilon,
        )
    )
    if overlap == 0.0 and intervals.shape[0] > 1:
        for index in range(intervals.shape[0] - 1):
            if abs(intervals[index, 1] - intervals[index + 1, 0]) >= 1e-4:
                intervals[index + 1, 0] = intervals[index, 1]
    return intervals


def _cut_point_intervals(
    values: np.ndarray,
    cut_points: Sequence[float] | Tensor,
) -> np.ndarray:
    if isinstance(cut_points, Tensor):
        points = cut_points.detach().cpu().numpy()
    else:
        try:
            points = np.asarray(cut_points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("cut_points must be a numeric vector") from error
    if points.ndim != 1 or points.size < 1 or not np.isfinite(points).all():
        raise ValueError("cut_points must be a non-empty finite vector")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.any(points < minimum) or np.any(points > maximum):
        raise ValueError(
            f"cut_points must lie within the x range [{minimum}, {maximum}]"
        )
    if np.any(np.diff(points) <= 0.0):
        raise ValueError("cut_points must be strictly increasing")
    extra = (maximum - minimum) / 100000.0
    boundaries = np.concatenate(
        (
            np.array([minimum]),
            points.astype(np.float64, copy=False),
            np.array([maximum + 2.0 * extra]),
        )
    )
    intervals = np.column_stack(
        (
            boundaries[:-1],
            boundaries[1:] - extra,
        )
    )
    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ValueError("cut_points do not define increasing intervals")
    return intervals


def _worm_axes(
    pyplot: Any,
    axes: Sequence[Any] | Any | None,
    *,
    panel_count: int,
    figsize: tuple[float, float] | None,
) -> tuple[tuple[Any, ...], Any]:
    if axes is None:
        columns = 1 if panel_count == 1 else min(2, panel_count)
        rows = math.ceil(panel_count / columns)
        selected_figsize = figsize or (6.0 * columns, 4.0 * rows)
        figure, raw_axes = pyplot.subplots(
            rows,
            columns,
            figsize=selected_figsize,
            squeeze=False,
        )
        flattened = tuple(raw_axes.reshape(-1))
        for unused_axis in flattened[panel_count:]:
            unused_axis.set_visible(False)
        return flattened[:panel_count], figure
    flattened = tuple(np.asarray(axes, dtype=object).reshape(-1))
    if len(flattened) != panel_count:
        raise ValueError(
            f"axes must contain exactly {panel_count} Matplotlib axes"
        )
    figures = {id(axis.figure): axis.figure for axis in flattened}
    if len(figures) != 1:
        raise ValueError("all worm-plot axes must belong to the same figure")
    return flattened, next(iter(figures.values()))


def _worm_panel(
    axis: Any,
    residuals: np.ndarray,
    *,
    x_limit: float,
    y_limit: float,
    show_polynomial: bool,
    interval: tuple[float, float] | None,
    x_label: str | None,
    show_interval: bool,
) -> WormPlotPanel:
    ordered = np.sort(residuals.astype(np.float64, copy=False))
    theoretical = norm.ppf(_plotting_positions(ordered.size))
    deviations = ordered - theoretical
    confidence_grid = np.arange(
        -x_limit,
        x_limit + np.finfo(np.float64).eps * max(1.0, x_limit),
        0.25,
        dtype=np.float64,
    )
    probabilities = norm.cdf(confidence_grid)
    standard_errors = (
        np.sqrt(
            probabilities
            * (1.0 - probabilities)
            / ordered.size
        )
        / norm.pdf(confidence_grid)
    )
    confidence_lower = norm.ppf(0.025) * standard_errors
    confidence_upper = -confidence_lower

    if np.any(np.abs(deviations) > y_limit):
        warnings.warn(
            "some worm-plot deviations fall outside y_limit",
            stacklevel=3,
        )
    if np.any(np.abs(theoretical) > x_limit):
        warnings.warn(
            "some theoretical quantiles fall outside x_limit",
            stacklevel=3,
        )

    axis.scatter(
        theoretical,
        deviations,
        s=24,
        facecolors="wheat",
        edgecolors="0.25",
        linewidths=0.7,
        alpha=0.9,
    )
    axis.axhline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=0.9,
    )
    axis.axvline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=0.9,
    )
    axis.plot(
        confidence_grid,
        confidence_lower,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
    )
    axis.plot(
        confidence_grid,
        confidence_upper,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
    )
    coefficient_values = None
    coefficient_tensor = None
    if show_polynomial:
        coefficient_values = _worm_coefficients(theoretical, deviations)
        polynomial_limit = x_limit if interval is None else min(x_limit, 2.5)
        polynomial_grid = np.linspace(
            -polynomial_limit,
            polynomial_limit,
            300,
        )
        fitted_deviation = sum(
            coefficient_values[power] * polynomial_grid**power
            for power in range(4)
            if math.isfinite(coefficient_values[power])
        )
        axis.plot(
            polynomial_grid,
            fitted_deviation,
            color="red",
            linewidth=1.1,
        )
        coefficient_tensor = torch.from_numpy(coefficient_values.copy())

    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(-y_limit, y_limit)
    axis.set_xlabel("Unit normal quantile")
    axis.set_ylabel("Deviation")
    if interval is None:
        axis.set_title("Worm plot")
    elif show_interval:
        label = x_label or "x"
        lower = 0.0 if abs(interval[0]) < 5e-12 else interval[0]
        upper = 0.0 if abs(interval[1]) < 5e-12 else interval[1]
        axis.set_title(
            f"{label}: [{lower:.4g}, {upper:.4g}]"
        )
    axis.grid(alpha=0.2)
    return WormPlotPanel(
        axis=axis,
        residuals=torch.from_numpy(ordered.copy()),
        theoretical_quantiles=torch.from_numpy(theoretical.copy()),
        deviations=torch.from_numpy(deviations.copy()),
        confidence_grid=torch.from_numpy(confidence_grid.copy()),
        confidence_lower=torch.from_numpy(confidence_lower.copy()),
        confidence_upper=torch.from_numpy(confidence_upper.copy()),
        coefficients=coefficient_tensor,
        interval=interval,
    )


def _worm_coefficients(
    theoretical_quantiles: np.ndarray,
    deviations: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        tuple(theoretical_quantiles**power for power in range(4))
    )
    estimable_columns = min(theoretical_quantiles.size, 4)
    estimated, *_ = np.linalg.lstsq(
        design[:, :estimable_columns],
        deviations,
        rcond=None,
    )
    coefficients = np.full(4, np.nan, dtype=np.float64)
    coefficients[:estimable_columns] = estimated
    return coefficients


def _bucket_statistics(
    residuals: np.ndarray,
    weights: np.ndarray,
    *,
    kind: Literal["moment", "centile.central", "centile.tail"],
) -> BucketStatistics:
    valid = (
        np.isfinite(residuals)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    values = residuals[valid]
    positive_weights = weights[valid]
    if values.size < 2 or not positive_weights.size:
        raise ValueError(
            "bucket statistics require positive-weight observations"
        )
    normalized_weights = positive_weights / np.sum(positive_weights)
    effective_count = float(np.sum(positive_weights))
    if kind == "moment":
        mean = float(np.sum(normalized_weights * values))
        centered = values - mean
        second_moment = float(
            np.sum(normalized_weights * centered**2)
        )
        if second_moment <= 0.0:
            raise ValueError(
                "moment bucket plots require non-constant residuals"
            )
        third_moment = float(
            np.sum(normalized_weights * centered**3)
        )
        fourth_moment = float(
            np.sum(normalized_weights * centered**4)
        )
        skewness = third_moment / second_moment**1.5
        kurtosis = fourth_moment / second_moment**2
        excess_kurtosis = kurtosis - 3.0
        transformed_skewness = _signed_transform(skewness)
        transformed_kurtosis = _signed_transform(excess_kurtosis)
        jarque_bera = (
            effective_count / 6.0 * skewness**2
            + effective_count / 24.0 * excess_kurtosis**2
        )
    else:
        quantiles = _weighted_quantiles(
            values,
            positive_weights,
            np.array([0.01, 0.25, 0.5, 0.75, 0.99]),
        )
        lower_tail, lower_quartile, median, upper_quartile, upper_tail = (
            quantiles
        )
        central_scale = (upper_quartile - lower_quartile) / 2.0
        tail_scale = (upper_tail - lower_tail) / 2.0
        if central_scale <= 0.0 or tail_scale <= 0.0:
            raise ValueError(
                "centile bucket plots require distinct residual quantiles"
            )
        central_skewness = (
            (lower_quartile + upper_quartile) / 2.0 - median
        ) / central_scale
        tail_skewness = (
            (lower_tail + upper_tail) / 2.0 - median
        ) / tail_scale
        kurtosis = (
            upper_tail - lower_tail
        ) / (upper_quartile - lower_quartile)
        excess_kurtosis = kurtosis - 3.449
        skewness = (
            central_skewness
            if kind == "centile.central"
            else tail_skewness
        )
        transformed_skewness = skewness
        transformed_kurtosis = _signed_transform(excess_kurtosis)
        jarque_bera = None
    return BucketStatistics(
        observation_count=residuals.size,
        effective_observation_count=effective_count,
        skewness=float(skewness),
        transformed_skewness=float(transformed_skewness),
        kurtosis=float(kurtosis),
        excess_kurtosis=float(excess_kurtosis),
        transformed_kurtosis=float(transformed_kurtosis),
        jarque_bera=(
            None if jarque_bera is None else float(jarque_bera)
        ),
    )


def _signed_transform(value: float) -> float:
    return value / (1.0 + abs(value))


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Reproduce the weighted type-7 quantiles used by R centileSK()."""
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("quantile probabilities must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    unique_values, inverse = np.unique(
        ordered_values,
        return_inverse=True,
    )
    aggregated_weights = np.zeros(
        unique_values.size,
        dtype=np.float64,
    )
    np.add.at(aggregated_weights, inverse, ordered_weights)
    cumulative_weights = np.cumsum(aggregated_weights)
    total_weight = float(cumulative_weights[-1])
    positions = 1.0 + (total_weight - 1.0) * probabilities
    lower = np.maximum(np.floor(positions), 1.0)
    upper = np.minimum(lower + 1.0, total_weight)
    fractions = np.mod(positions, 1.0)
    targets = np.concatenate((lower, upper))
    indices = np.searchsorted(
        cumulative_weights,
        targets,
        side="left",
    )
    indices = np.clip(indices, 0, unique_values.size - 1)
    selected = unique_values[indices]
    count = probabilities.size
    return (
        (1.0 - fractions) * selected[:count]
        + fractions * selected[count:]
    )


def _bucket_bootstrap(
    residuals: np.ndarray,
    weights: np.ndarray,
    *,
    kind: Literal["moment", "centile.central", "centile.tail"],
    replicates: int,
    generator: torch.Generator | None,
) -> np.ndarray:
    points = np.empty((replicates, 2), dtype=np.float64)
    for replicate in range(replicates):
        indices = torch.randint(
            residuals.size,
            (residuals.size,),
            generator=generator,
        ).numpy()
        try:
            statistics = _bucket_statistics(
                residuals[indices],
                weights[indices],
                kind=kind,
            )
            points[replicate] = statistics.point
        except ValueError:
            points[replicate] = np.nan
    return points


def _bucket_axes(
    pyplot: Any,
    axes: Sequence[Any] | Any | None,
    *,
    panel_count: int,
    figsize: tuple[float, float] | None,
) -> tuple[tuple[Any, ...], Any]:
    if axes is None:
        columns = 1 if panel_count == 1 else min(2, panel_count)
        rows = math.ceil(panel_count / columns)
        selected_figsize = figsize or (6.0 * columns, 5.0 * rows)
        figure, raw_axes = pyplot.subplots(
            rows,
            columns,
            figsize=selected_figsize,
            squeeze=False,
        )
        flattened = tuple(raw_axes.reshape(-1))
        for unused_axis in flattened[panel_count:]:
            unused_axis.set_visible(False)
        return flattened[:panel_count], figure
    flattened = tuple(np.asarray(axes, dtype=object).reshape(-1))
    if len(flattened) != panel_count:
        raise ValueError(
            f"axes must contain exactly {panel_count} Matplotlib axes"
        )
    figures = {id(axis.figure): axis.figure for axis in flattened}
    if len(figures) != 1:
        raise ValueError("all bucket-plot axes must belong to the same figure")
    return flattened, next(iter(figures.values()))


def _draw_bucket_panel(
    axis: Any,
    statistics: BucketStatistics,
    bootstrap_points: np.ndarray,
    *,
    kind: Literal["moment", "centile.central", "centile.tail"],
    interval: tuple[float, float] | None,
    x_label: str | None,
    label: str | None,
    show_interval: bool,
    show_legend: bool,
) -> None:
    if kind == "moment":
        _draw_moment_bucket_background(
            axis,
            observation_count=statistics.observation_count,
        )
        axis.set_xlabel("Transformed moment skewness")
        axis.set_ylabel("Transformed excess kurtosis")
        title = "Moment bucket plot"
    elif kind == "centile.central":
        axis.set_xlabel("Central centile skewness")
        axis.set_ylabel("Transformed centile kurtosis")
        title = "Central centile bucket plot"
    else:
        axis.set_xlabel("Tail centile skewness")
        axis.set_ylabel("Transformed centile kurtosis")
        title = "Tail centile bucket plot"

    valid_bootstrap = np.isfinite(bootstrap_points).all(axis=1)
    if np.any(valid_bootstrap):
        axis.scatter(
            bootstrap_points[valid_bootstrap, 0],
            bootstrap_points[valid_bootstrap, 1],
            s=22,
            facecolors="lightblue",
            edgecolors="steelblue",
            linewidths=0.5,
            alpha=0.55,
            label="Bootstrap",
            zorder=3,
        )
    point_x, point_y = statistics.point
    axis.scatter(
        [point_x],
        [point_y],
        s=70,
        marker="x",
        color="black",
        linewidths=2.0,
        label="Observed",
        zorder=5,
    )
    if label is not None:
        axis.annotate(
            label,
            (point_x, point_y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )
    axis.axhline(0.0, color="0.75", linewidth=0.8)
    axis.axvline(0.0, color="0.75", linewidth=0.8)
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    axis.grid(alpha=0.2)
    axis.text(
        -0.94,
        -0.93,
        f"n={statistics.observation_count}",
        fontsize=8,
        color="0.35",
    )
    if interval is None:
        axis.set_title(title)
    elif show_interval:
        interval_label = x_label or "x"
        lower = 0.0 if abs(interval[0]) < 5e-12 else interval[0]
        upper = 0.0 if abs(interval[1]) < 5e-12 else interval[1]
        axis.set_title(
            f"{interval_label}: [{lower:.4g}, {upper:.4g}]"
        )
    if show_legend:
        axis.legend(frameon=False, loc="upper left")


def _draw_moment_bucket_background(
    axis: Any,
    *,
    observation_count: int,
) -> None:
    transformed_skewness = np.linspace(-0.99, 0.99, 801)
    skewness = transformed_skewness / (
        1.0 - np.abs(transformed_skewness)
    )
    critical_value = float(chi2.ppf(0.95, df=2))
    squared_limit = (
        24.0
        / observation_count
        * (
            critical_value
            - observation_count / 6.0 * skewness**2
        )
    )
    accepted = squared_limit >= 0.0
    excess_limit = np.sqrt(np.maximum(squared_limit, 0.0))
    transformed_limit = excess_limit / (1.0 + excess_limit)
    axis.fill_between(
        transformed_skewness[accepted],
        -transformed_limit[accepted],
        transformed_limit[accepted],
        color="0.96",
        label="Jarque-Bera 95% region",
        zorder=0,
    )
    pearson_excess = skewness**2 - 2.0
    pearson_boundary = pearson_excess / (
        1.0 + np.abs(pearson_excess)
    )
    axis.plot(
        transformed_skewness,
        pearson_boundary,
        color="0.25",
        linewidth=1.2,
        label="Moment boundary",
        zorder=2,
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
