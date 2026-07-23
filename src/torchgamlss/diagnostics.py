"""Model-selection criteria and quantile residual diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
import torch
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


def model_diagnostics(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    degrees_of_freedom: float | None = None,
) -> ModelDiagnostics:
    """Evaluate likelihood criteria for the model's current fitted state."""
    losses = model.negative_log_likelihood(
        response,
        design_matrices,
        weights=weights,
        offsets=offsets,
        smooth_covariates=smooth_covariates,
        reduction="none",
    )
    if losses.ndim != 1 or losses.numel() != response.numel():
        raise ValueError("diagnostics require one likelihood value per response")
    case_weights = model._validated_weights(response, weights)
    if degrees_of_freedom is None:
        if any(
            model.smooth_terms[parameter] for parameter in model.family.parameter_names
        ):
            raise ValueError(
                "degrees_of_freedom is required for models with smooth terms; "
                "use RSFitResult.effective_degrees_of_freedom"
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
