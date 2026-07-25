"""Conditional quantile prediction and parametric-bootstrap inference."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.fitting import CGControl, RSControl
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class QuantilePrediction:
    """Conditional response quantiles for fitted family parameters."""

    family: str
    probabilities: Tensor
    quantiles: Tensor

    @property
    def centiles(self) -> Tensor:
        """Return probabilities expressed as percentages."""
        return 100.0 * self.probabilities

    def at(self, probability: float) -> Tensor:
        """Return the conditional quantile at one stored probability."""
        index = _probability_index(self.probabilities, probability)
        return self.quantiles[..., index]

    def to_dataframe(self) -> pd.DataFrame:
        """Return observation-wise quantiles in long format."""
        return _quantile_dataframe(
            self.probabilities,
            {"quantile": self.quantiles},
        )


@dataclass(frozen=True)
class QuantileBandResult:
    """Bootstrap max-|t| bands for conditional quantile curves."""

    family: str
    probabilities: Tensor
    estimates: Tensor
    confidence_intervals: Tensor
    critical_values: Tensor
    confidence_level: float
    replicates: int
    joint: bool
    method: str

    @property
    def centiles(self) -> Tensor:
        """Return probabilities expressed as percentages."""
        return 100.0 * self.probabilities

    def at(self, probability: float) -> Tensor:
        """Return interval limits for one stored probability."""
        index = _probability_index(self.probabilities, probability)
        return self.confidence_intervals[:, index]

    def to_dataframe(self) -> pd.DataFrame:
        """Return all bands in long format."""
        return _quantile_dataframe(
            self.probabilities,
            {
                "estimate": self.estimates,
                "ci_lower": self.confidence_intervals[..., 0],
                "ci_upper": self.confidence_intervals[..., 1],
            },
        )


@dataclass(frozen=True)
class QuantileBootstrapResult:
    """Fixed-design parametric-bootstrap inference for response quantiles."""

    family: str
    probabilities: Tensor
    estimates: Tensor
    bootstrap_estimates: Tensor = field(repr=False)
    standard_errors: Tensor
    confidence_intervals: Tensor
    confidence_level: float
    replicates: int
    attempts: int
    failed_replicates: int
    algorithm: str

    @property
    def centiles(self) -> Tensor:
        """Return probabilities expressed as percentages."""
        return 100.0 * self.probabilities

    @property
    def bootstrap_mean(self) -> Tensor:
        """Return mean quantile curves across successful refits."""
        return self.bootstrap_estimates.mean(dim=0)

    @property
    def bias(self) -> Tensor:
        """Return bootstrap mean minus the fitted quantile curves."""
        return self.bootstrap_mean - self.estimates

    @property
    def covariance_matrix(self) -> Tensor:
        """Return covariance over flattened grid-by-probability coordinates."""
        flattened = self.bootstrap_estimates.flatten(start_dim=1)
        centered = flattened - flattened.mean(dim=0)
        return centered.mT @ centered / (self.replicates - 1)

    @property
    def failure_rate(self) -> float:
        """Return the fraction of attempted model refits that failed."""
        return self.failed_replicates / self.attempts

    def at(self, probability: float) -> Tensor:
        """Return pointwise intervals for one stored probability."""
        index = _probability_index(self.probabilities, probability)
        return self.confidence_intervals[:, index]

    def simultaneous_confidence_bands(
        self,
        *,
        joint: bool = True,
    ) -> QuantileBandResult:
        """Return max-|t| bands over all centiles or one band per centile."""
        standardized = torch.zeros_like(self.bootstrap_estimates)
        positive = self.standard_errors > 0
        standardized[:, positive] = (
            self.bootstrap_estimates[:, positive] - self.estimates[positive]
        ) / self.standard_errors[positive]

        if joint:
            if not positive.any():
                raise RuntimeError(
                    "bootstrap quantiles have no positive variance; "
                    "simultaneous bands are unavailable"
                )
            maximum_statistics = standardized[:, positive].abs().amax(dim=1)
            critical_values = torch.quantile(
                maximum_statistics,
                self.confidence_level,
            ).reshape(1)
            widths = critical_values[0] * self.standard_errors
            method = "parametric_bootstrap_joint_quantile_max_t"
        else:
            statistics = []
            for probability_index in range(self.probabilities.numel()):
                positive_for_probability = positive[:, probability_index]
                if not positive_for_probability.any():
                    raise RuntimeError(
                        "a bootstrap quantile curve has no positive variance; "
                        "its simultaneous band is unavailable"
                    )
                statistics.append(
                    standardized[
                        :,
                        positive_for_probability,
                        probability_index,
                    ]
                    .abs()
                    .amax(dim=1)
                )
            critical_values = torch.stack(
                [
                    torch.quantile(statistic, self.confidence_level)
                    for statistic in statistics
                ]
            )
            widths = self.standard_errors * critical_values.unsqueeze(0)
            method = "parametric_bootstrap_per_quantile_max_t"

        confidence_intervals = torch.stack(
            (self.estimates - widths, self.estimates + widths),
            dim=-1,
        )
        return QuantileBandResult(
            family=self.family,
            probabilities=self.probabilities,
            estimates=self.estimates,
            confidence_intervals=confidence_intervals,
            critical_values=critical_values,
            confidence_level=self.confidence_level,
            replicates=self.replicates,
            joint=joint,
            method=method,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return pointwise bootstrap quantile inference in long format."""
        return _quantile_dataframe(
            self.probabilities,
            {
                "estimate": self.estimates,
                "bootstrap_mean": self.bootstrap_mean,
                "bias": self.bias,
                "standard_error": self.standard_errors,
                "ci_lower": self.confidence_intervals[..., 0],
                "ci_upper": self.confidence_intervals[..., 1],
            },
        )


def centiles_to_probabilities(centiles: Any, reference: Tensor) -> Tensor:
    """Convert centile percentages to a validated probability vector."""
    try:
        values = torch.as_tensor(
            centiles,
            dtype=reference.dtype,
            device=reference.device,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "centiles must be convertible to a floating-point tensor"
        ) from error
    if (
        values.ndim != 1
        or values.numel() < 1
        or not torch.isfinite(values).all()
        or not torch.all((values > 0) & (values < 100))
    ):
        raise ValueError(
            "centiles must be a non-empty finite vector strictly between "
            "zero and 100"
        )
    return values / 100.0


def quantile_bootstrap(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    probabilities: Any,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    evaluation_design_matrices: Mapping[str, Tensor] | None = None,
    evaluation_offsets: Mapping[str, Tensor] | None = None,
    evaluation_smooth_covariates: (
        Mapping[str, Mapping[str, Tensor]] | None
    ) = None,
    replicates: int = 999,
    max_attempts: int | None = None,
    algorithm: Literal["rs", "cg"] = "rs",
    control: RSControl | CGControl | None = None,
    confidence_level: float = 0.95,
    generator: torch.Generator | None = None,
) -> QuantileBootstrapResult:
    """Refit parametric samples and summarize conditional response quantiles."""
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 10
    ):
        raise ValueError("replicates must be an integer of at least 10")
    if max_attempts is None:
        max_attempts = max(replicates + 10, math.ceil(1.2 * replicates))
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < replicates
    ):
        raise ValueError("max_attempts must be an integer not smaller than replicates")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    if algorithm not in {"rs", "cg"}:
        raise ValueError("algorithm must be 'rs' or 'cg'")

    from torchgamlss.fitting import CGControl, RSControl

    expected_control = RSControl if algorithm == "rs" else CGControl
    if control is not None and not isinstance(control, expected_control):
        raise ValueError(
            f"control must be {expected_control.__name__} when algorithm="
            f"{algorithm!r}"
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
            "quantile bootstrap response must be a non-empty finite vector "
            "matching the model dtype and device"
        )
    model.family.validate_response(response, context="quantile bootstrap")
    case_weights = model._validated_weights(response, weights)
    contributions = model.term_contributions(
        design_matrices,
        offsets,
        smooth_covariates=smooth_covariates,
    )
    if any(
        contribution.offset.numel() != response.numel()
        for contribution in contributions.values()
    ):
        raise ValueError("design matrices must have one row per response")

    fitted_parameters = {
        parameter: value.detach()
        for parameter, value in model.family.parameters_from_predictors(
            {
                parameter: contribution.total.detach()
                for parameter, contribution in contributions.items()
            }
        ).items()
    }
    evaluation_design_matrices = evaluation_design_matrices or design_matrices
    evaluation_offsets = (
        offsets if evaluation_offsets is None else evaluation_offsets
    )
    evaluation_smooth_covariates = (
        smooth_covariates
        if evaluation_smooth_covariates is None
        else evaluation_smooth_covariates
    )
    original_parameters = model.predict(
        evaluation_design_matrices,
        evaluation_offsets,
        smooth_covariates=evaluation_smooth_covariates,
        type="response",
    )
    assert isinstance(original_parameters, dict)
    estimates = model.family.quantile(
        probabilities,
        original_parameters,
    ).detach()
    probability_tensor = torch.as_tensor(
        probabilities,
        dtype=response.dtype,
        device=response.device,
    ).detach().clone()
    bootstrap_estimates = torch.empty(
        (replicates,) + estimates.shape,
        dtype=estimates.dtype,
        device=estimates.device,
    )

    successful_replicates = 0
    attempts = 0
    failure_messages: list[str] = []
    while successful_replicates < replicates and attempts < max_attempts:
        attempts += 1
        bootstrap_response = model.family.sample(
            fitted_parameters,
            generator=generator,
        )
        if (
            bootstrap_response.shape != response.shape
            or bootstrap_response.dtype != response.dtype
            or bootstrap_response.device != response.device
            or not torch.isfinite(bootstrap_response).all()
        ):
            failure_messages.append(
                "family sampling returned an invalid bootstrap response"
            )
            continue
        try:
            model.family.validate_response(
                bootstrap_response,
                context=f"quantile bootstrap attempt {attempts}",
            )
        except ValueError as error:
            failure_messages.append(str(error))
            continue

        bootstrap_model = copy.deepcopy(model)
        try:
            if algorithm == "rs":
                fit_result = bootstrap_model.fit_rs(
                    bootstrap_response,
                    design_matrices,
                    weights=case_weights,
                    offsets=offsets,
                    smooth_covariates=smooth_covariates,
                    initial_parameters=fitted_parameters,
                    control=control,
                )
            else:
                fit_result = bootstrap_model.fit_cg(
                    bootstrap_response,
                    design_matrices,
                    weights=case_weights,
                    offsets=offsets,
                    smooth_covariates=smooth_covariates,
                    initial_parameters=fitted_parameters,
                    control=control,
                )
            if not fit_result.converged:
                failure_messages.append("classical fit did not converge")
                continue
            bootstrap_parameters = bootstrap_model.predict(
                evaluation_design_matrices,
                evaluation_offsets,
                smooth_covariates=evaluation_smooth_covariates,
                type="response",
            )
            assert isinstance(bootstrap_parameters, dict)
            replicate_quantiles = model.family.quantile(
                probability_tensor,
                bootstrap_parameters,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failure_messages.append(str(error))
            continue
        if (
            replicate_quantiles.shape != estimates.shape
            or not torch.isfinite(replicate_quantiles).all()
        ):
            failure_messages.append(
                "family quantile evaluation returned invalid values"
            )
            continue
        bootstrap_estimates[successful_replicates].copy_(replicate_quantiles)
        successful_replicates += 1

    if successful_replicates < replicates:
        last_failure = failure_messages[-1] if failure_messages else "unknown failure"
        raise RuntimeError(
            f"quantile bootstrap obtained {successful_replicates} successful "
            f"fits out of {replicates} after {attempts} attempts; "
            f"last failure: {last_failure}"
        )

    tail_probability = (1.0 - confidence_level) / 2.0
    interval_probabilities = torch.tensor(
        [tail_probability, 1.0 - tail_probability],
        dtype=response.dtype,
        device=response.device,
    )
    confidence_intervals = torch.quantile(
        bootstrap_estimates,
        interval_probabilities,
        dim=0,
    ).movedim(0, -1)
    return QuantileBootstrapResult(
        family=model.family.name,
        probabilities=probability_tensor,
        estimates=estimates,
        bootstrap_estimates=bootstrap_estimates,
        standard_errors=bootstrap_estimates.std(dim=0, correction=1),
        confidence_intervals=confidence_intervals,
        confidence_level=confidence_level,
        replicates=replicates,
        attempts=attempts,
        failed_replicates=attempts - replicates,
        algorithm=algorithm,
    )


def _probability_index(probabilities: Tensor, probability: float) -> int:
    if not math.isfinite(probability):
        raise ValueError("probability must be finite")
    matches = torch.isclose(
        probabilities,
        probabilities.new_tensor(probability),
        rtol=1e-7,
        atol=10.0 * torch.finfo(probabilities.dtype).eps,
    )
    indices = torch.nonzero(matches).flatten()
    if indices.numel() != 1:
        raise KeyError(f"probability {probability!r} is not uniquely stored")
    return int(indices[0])


def _quantile_dataframe(
    probabilities: Tensor,
    values: Mapping[str, Tensor],
) -> pd.DataFrame:
    first = next(iter(values.values()))
    observation_count, probability_count = first.shape
    frame = {
        "observation": torch.arange(
            observation_count,
            device=first.device,
        )
        .repeat_interleave(probability_count)
        .cpu()
        .numpy(),
        "probability": probabilities.repeat(observation_count).cpu().numpy(),
        "centile": (100.0 * probabilities)
        .repeat(observation_count)
        .cpu()
        .numpy(),
    }
    for name, value in values.items():
        if value.shape != first.shape:
            raise ValueError("quantile table values must share one shape")
        frame[name] = value.detach().flatten().cpu().numpy()
    return pd.DataFrame(frame)
