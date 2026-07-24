"""Bootstrap inference for derived smooth-curve functionals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.inference import SmoothJointBootstrapResult

SmoothTermKey = tuple[str, str]


@dataclass(frozen=True)
class SmoothDerivedBandResult:
    """Max-|t| bootstrap band for a derived smooth curve."""

    name: str
    operation: str
    source_terms: tuple[SmoothTermKey, ...]
    covariate: Tensor
    estimates: Tensor
    confidence_intervals: Tensor
    critical_value: float
    confidence_level: float
    replicates: int
    method: str = "parametric_bootstrap_derived_max_t"

    def to_dataframe(self) -> pd.DataFrame:
        """Return the derived curve and simultaneous limits."""
        values = torch.column_stack(
            (self.covariate, self.estimates, self.confidence_intervals)
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=["covariate", "estimate", "ci_lower", "ci_upper"],
        )


@dataclass(frozen=True)
class SmoothDerivedBootstrapResult:
    """Pointwise bootstrap inference for a derived smooth curve."""

    name: str
    operation: str
    source_terms: tuple[SmoothTermKey, ...]
    covariate: Tensor
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
    def bootstrap_mean(self) -> Tensor:
        """Return the mean derived curve across aligned refits."""
        return self.bootstrap_estimates.mean(dim=0)

    @property
    def bias(self) -> Tensor:
        """Return bootstrap mean minus the original derived curve."""
        return self.bootstrap_mean - self.estimates

    @property
    def covariance_matrix(self) -> Tensor:
        """Return empirical covariance across the derived curve."""
        centered = self.bootstrap_estimates - self.bootstrap_mean
        return centered.mT @ centered / (self.replicates - 1)

    @property
    def failure_rate(self) -> float:
        """Return the failed-refit fraction inherited from the joint run."""
        return self.failed_replicates / self.attempts

    def simultaneous_confidence_band(self) -> SmoothDerivedBandResult:
        """Return a max-|t| band over the derived curve."""
        positive_standard_errors = self.standard_errors > 0
        if not positive_standard_errors.any():
            raise RuntimeError(
                "derived bootstrap curve has no positive variance; "
                "a simultaneous band is unavailable"
            )
        standardized_deviations = (
            self.bootstrap_estimates[:, positive_standard_errors]
            - self.estimates[positive_standard_errors]
        ) / self.standard_errors[positive_standard_errors]
        maximum_statistics = standardized_deviations.abs().amax(dim=1)
        critical_value = float(
            torch.quantile(maximum_statistics, self.confidence_level)
        )
        confidence_intervals = torch.column_stack(
            (
                self.estimates - critical_value * self.standard_errors,
                self.estimates + critical_value * self.standard_errors,
            )
        )
        return SmoothDerivedBandResult(
            name=self.name,
            operation=self.operation,
            source_terms=self.source_terms,
            covariate=self.covariate,
            estimates=self.estimates,
            confidence_intervals=confidence_intervals,
            critical_value=critical_value,
            confidence_level=self.confidence_level,
            replicates=self.replicates,
        )

    def extremum(
        self,
        *,
        kind: Literal["maximum", "minimum"] = "maximum",
    ) -> SmoothExtremumBootstrapResult:
        """Return bootstrap inference for a maximum or minimum."""
        return smooth_extremum(self, kind=kind)

    def crossing(
        self,
        *,
        level: float = 0.0,
        direction: Literal["any", "increasing", "decreasing"] = "any",
        which: Literal["first", "last"] = "first",
    ) -> SmoothCrossingBootstrapResult:
        """Return bootstrap inference for a linearly interpolated crossing."""
        return smooth_crossing(
            self,
            level=level,
            direction=direction,
            which=which,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return pointwise inference for the derived curve."""
        values = torch.column_stack(
            (
                self.covariate,
                self.estimates,
                self.bootstrap_mean,
                self.bias,
                self.standard_errors,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=[
                "covariate",
                "estimate",
                "bootstrap_mean",
                "bias",
                "standard_error",
                "ci_lower",
                "ci_upper",
            ],
        )


@dataclass(frozen=True)
class SmoothExtremumBootstrapResult:
    """Bootstrap inference for a smooth-curve extremum and its location."""

    name: str
    kind: Literal["maximum", "minimum"]
    source_terms: tuple[SmoothTermKey, ...]
    estimate: float
    location: float
    bootstrap_estimates: Tensor = field(repr=False)
    bootstrap_locations: Tensor = field(repr=False)
    standard_error: float
    location_standard_error: float
    confidence_interval: Tensor
    location_confidence_interval: Tensor
    confidence_level: float
    replicates: int
    attempts: int
    failed_replicates: int
    algorithm: str

    @property
    def bootstrap_mean(self) -> float:
        """Return the bootstrap mean extremum value."""
        return float(self.bootstrap_estimates.mean())

    @property
    def bias(self) -> float:
        """Return bootstrap mean minus the fitted extremum value."""
        return self.bootstrap_mean - self.estimate

    @property
    def location_bootstrap_mean(self) -> float:
        """Return the bootstrap mean extremum location."""
        return float(self.bootstrap_locations.mean())

    @property
    def location_bias(self) -> float:
        """Return bootstrap mean minus the fitted extremum location."""
        return self.location_bootstrap_mean - self.location

    @property
    def failure_rate(self) -> float:
        """Return the failed-refit fraction inherited from the joint run."""
        return self.failed_replicates / self.attempts

    def to_dataframe(self) -> pd.DataFrame:
        """Return value and location summaries in long format."""
        return pd.DataFrame(
            {
                "metric": ["value", "location"],
                "estimate": [self.estimate, self.location],
                "bootstrap_mean": [
                    self.bootstrap_mean,
                    self.location_bootstrap_mean,
                ],
                "bias": [self.bias, self.location_bias],
                "standard_error": [
                    self.standard_error,
                    self.location_standard_error,
                ],
                "ci_lower": [
                    float(self.confidence_interval[0]),
                    float(self.location_confidence_interval[0]),
                ],
                "ci_upper": [
                    float(self.confidence_interval[1]),
                    float(self.location_confidence_interval[1]),
                ],
            }
        )


@dataclass(frozen=True)
class SmoothCrossingBootstrapResult:
    """Bootstrap inference for one selected smooth-curve crossing."""

    name: str
    source_terms: tuple[SmoothTermKey, ...]
    level: float
    direction: Literal["any", "increasing", "decreasing"]
    which: Literal["first", "last"]
    estimate: float
    bootstrap_estimates: Tensor = field(repr=False)
    standard_error: float
    confidence_interval: Tensor
    confidence_level: float
    replicates: int
    attempts: int
    failed_replicates: int
    valid_replicates: int
    missing_replicates: int
    algorithm: str

    @property
    def valid_bootstrap_estimates(self) -> Tensor:
        """Return crossing locations from replicates that contain the crossing."""
        return self.bootstrap_estimates[torch.isfinite(self.bootstrap_estimates)]

    @property
    def bootstrap_mean(self) -> float:
        """Return the mean valid bootstrap crossing location."""
        return float(self.valid_bootstrap_estimates.mean())

    @property
    def bias(self) -> float:
        """Return bootstrap mean minus the fitted crossing location."""
        return self.bootstrap_mean - self.estimate

    @property
    def missing_rate(self) -> float:
        """Return the fraction of refitted curves without this crossing."""
        return self.missing_replicates / self.replicates

    @property
    def failure_rate(self) -> float:
        """Return the failed model-refit fraction before crossing selection."""
        return self.failed_replicates / self.attempts

    def to_dataframe(self) -> pd.DataFrame:
        """Return the selected crossing summary."""
        return pd.DataFrame(
            {
                "estimate": [self.estimate],
                "bootstrap_mean": [self.bootstrap_mean],
                "bias": [self.bias],
                "standard_error": [self.standard_error],
                "ci_lower": [float(self.confidence_interval[0])],
                "ci_upper": [float(self.confidence_interval[1])],
                "valid_replicates": [self.valid_replicates],
                "missing_replicates": [self.missing_replicates],
                "missing_rate": [self.missing_rate],
            }
        )


def smooth_linear_contrast(
    joint: SmoothJointBootstrapResult,
    coefficients: Mapping[SmoothTermKey, float],
    *,
    name: str | None = None,
) -> SmoothDerivedBootstrapResult:
    """Combine aligned smooth curves with fixed scalar coefficients."""
    if not coefficients:
        raise ValueError("coefficients must contain at least one smooth term")
    unknown_terms = set(coefficients).difference(joint.term_order)
    if unknown_terms:
        raise KeyError(f"unknown smooth terms: {sorted(unknown_terms)!r}")
    normalized_coefficients = {}
    for key, coefficient in coefficients.items():
        try:
            value = float(coefficient)
        except (TypeError, ValueError) as error:
            raise ValueError("contrast coefficients must be finite scalars") from error
        if not math.isfinite(value):
            raise ValueError("contrast coefficients must be finite scalars")
        if value != 0.0:
            normalized_coefficients[key] = value
    if not normalized_coefficients:
        raise ValueError("at least one contrast coefficient must be nonzero")

    source_terms = tuple(
        key for key in joint.term_order if key in normalized_coefficients
    )
    reference = joint.curves[source_terms[0][0]][source_terms[0][1]]
    _validate_point_grid(reference.covariate)
    for parameter, term in source_terms[1:]:
        curve = joint.curves[parameter][term]
        if (
            curve.covariate.shape != reference.covariate.shape
            or not torch.allclose(curve.covariate, reference.covariate)
        ):
            raise ValueError(
                "contrasted smooths must use the same ordered covariate grid"
            )

    estimates = torch.zeros_like(reference.estimates)
    bootstrap_estimates = torch.zeros_like(reference.bootstrap_estimates)
    for parameter, term in source_terms:
        coefficient = normalized_coefficients[(parameter, term)]
        curve = joint.curves[parameter][term]
        estimates = estimates + coefficient * curve.estimates
        bootstrap_estimates = (
            bootstrap_estimates + coefficient * curve.bootstrap_estimates
        )
    return _derived_result(
        joint,
        name=name or _contrast_name(normalized_coefficients, source_terms),
        operation="linear_contrast",
        source_terms=source_terms,
        covariate=reference.covariate,
        estimates=estimates,
        bootstrap_estimates=bootstrap_estimates,
    )


def smooth_derivative(
    joint: SmoothJointBootstrapResult,
    key: SmoothTermKey,
    *,
    order: Literal[1, 2] = 1,
    name: str | None = None,
) -> SmoothDerivedBootstrapResult:
    """Differentiate one aligned bootstrap curve on its evaluation grid."""
    if isinstance(order, bool) or order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    try:
        curve = joint.curves[key[0]][key[1]]
    except (KeyError, IndexError, TypeError) as error:
        raise KeyError(f"unknown smooth term {key!r}") from error
    covariate = curve.covariate
    _validate_curve_grid(covariate, minimum_points=3)
    estimates = curve.estimates
    bootstrap_estimates = curve.bootstrap_estimates
    for _ in range(order):
        estimates = torch.gradient(
            estimates,
            spacing=(covariate,),
            dim=(0,),
        )[0]
        bootstrap_estimates = torch.gradient(
            bootstrap_estimates,
            spacing=(covariate,),
            dim=(1,),
        )[0]
    return _derived_result(
        joint,
        name=name or f"derivative_{order}({key[0]}.{key[1]})",
        operation=f"derivative_{order}",
        source_terms=(key,),
        covariate=covariate,
        estimates=estimates,
        bootstrap_estimates=bootstrap_estimates,
    )


def smooth_curve_result(
    joint: SmoothJointBootstrapResult,
    key: SmoothTermKey,
) -> SmoothDerivedBootstrapResult:
    """Represent one fitted curve through the derived-result interface."""
    try:
        curve = joint.curves[key[0]][key[1]]
    except (KeyError, IndexError, TypeError) as error:
        raise KeyError(f"unknown smooth term {key!r}") from error
    return SmoothDerivedBootstrapResult(
        name=f"{key[0]}.{key[1]}",
        operation="identity",
        source_terms=(key,),
        covariate=curve.covariate,
        estimates=curve.estimates,
        bootstrap_estimates=curve.bootstrap_estimates,
        standard_errors=curve.standard_errors,
        confidence_intervals=curve.confidence_intervals,
        confidence_level=joint.confidence_level,
        replicates=joint.replicates,
        attempts=joint.attempts,
        failed_replicates=joint.failed_replicates,
        algorithm=joint.algorithm,
    )


def smooth_extremum(
    result: SmoothDerivedBootstrapResult,
    *,
    kind: Literal["maximum", "minimum"] = "maximum",
) -> SmoothExtremumBootstrapResult:
    """Summarize the discrete-grid extremum of a derived curve."""
    if kind not in {"maximum", "minimum"}:
        raise ValueError("kind must be 'maximum' or 'minimum'")
    _validate_point_grid(result.covariate)
    reducer = torch.argmax if kind == "maximum" else torch.argmin
    fitted_index = int(reducer(result.estimates))
    bootstrap_indices = reducer(result.bootstrap_estimates, dim=1)
    bootstrap_values = result.bootstrap_estimates.gather(
        1,
        bootstrap_indices.unsqueeze(1),
    ).squeeze(1)
    bootstrap_locations = result.covariate[bootstrap_indices]
    probabilities = _quantile_probabilities(
        result.confidence_level,
        result.estimates,
    )
    return SmoothExtremumBootstrapResult(
        name=f"{kind}({result.name})",
        kind=kind,
        source_terms=result.source_terms,
        estimate=float(result.estimates[fitted_index]),
        location=float(result.covariate[fitted_index]),
        bootstrap_estimates=bootstrap_values,
        bootstrap_locations=bootstrap_locations,
        standard_error=float(bootstrap_values.std(correction=1)),
        location_standard_error=float(
            bootstrap_locations.std(correction=1)
        ),
        confidence_interval=torch.quantile(bootstrap_values, probabilities),
        location_confidence_interval=torch.quantile(
            bootstrap_locations,
            probabilities,
        ),
        confidence_level=result.confidence_level,
        replicates=result.replicates,
        attempts=result.attempts,
        failed_replicates=result.failed_replicates,
        algorithm=result.algorithm,
    )


def smooth_crossing(
    result: SmoothDerivedBootstrapResult,
    *,
    level: float = 0.0,
    direction: Literal["any", "increasing", "decreasing"] = "any",
    which: Literal["first", "last"] = "first",
) -> SmoothCrossingBootstrapResult:
    """Summarize a linearly interpolated crossing of a derived curve."""
    if not math.isfinite(level):
        raise ValueError("level must be finite")
    if direction not in {"any", "increasing", "decreasing"}:
        raise ValueError("direction must be 'any', 'increasing', or 'decreasing'")
    if which not in {"first", "last"}:
        raise ValueError("which must be 'first' or 'last'")
    _validate_curve_grid(result.covariate)
    fitted_location = _crossing_location(
        result.covariate,
        result.estimates,
        level=level,
        direction=direction,
        which=which,
    )
    if fitted_location is None:
        raise ValueError("the fitted curve does not contain the requested crossing")

    bootstrap_locations = torch.full(
        (result.replicates,),
        torch.nan,
        dtype=result.estimates.dtype,
        device=result.estimates.device,
    )
    for replicate in range(result.replicates):
        location = _crossing_location(
            result.covariate,
            result.bootstrap_estimates[replicate],
            level=level,
            direction=direction,
            which=which,
        )
        if location is not None:
            bootstrap_locations[replicate] = location
    valid_locations = bootstrap_locations[torch.isfinite(bootstrap_locations)]
    valid_replicates = valid_locations.numel()
    if valid_replicates < 2:
        raise RuntimeError(
            "fewer than two bootstrap curves contain the requested crossing"
        )
    probabilities = _quantile_probabilities(
        result.confidence_level,
        result.estimates,
    )
    return SmoothCrossingBootstrapResult(
        name=f"crossing({result.name})",
        source_terms=result.source_terms,
        level=float(level),
        direction=direction,
        which=which,
        estimate=fitted_location,
        bootstrap_estimates=bootstrap_locations,
        standard_error=float(valid_locations.std(correction=1)),
        confidence_interval=torch.quantile(valid_locations, probabilities),
        confidence_level=result.confidence_level,
        replicates=result.replicates,
        attempts=result.attempts,
        failed_replicates=result.failed_replicates,
        valid_replicates=valid_replicates,
        missing_replicates=result.replicates - valid_replicates,
        algorithm=result.algorithm,
    )


def _derived_result(
    joint: SmoothJointBootstrapResult,
    *,
    name: str,
    operation: str,
    source_terms: tuple[SmoothTermKey, ...],
    covariate: Tensor,
    estimates: Tensor,
    bootstrap_estimates: Tensor,
) -> SmoothDerivedBootstrapResult:
    tail_probability = (1.0 - joint.confidence_level) / 2.0
    probabilities = torch.tensor(
        [tail_probability, 1.0 - tail_probability],
        dtype=estimates.dtype,
        device=estimates.device,
    )
    return SmoothDerivedBootstrapResult(
        name=name,
        operation=operation,
        source_terms=source_terms,
        covariate=covariate.detach().clone(),
        estimates=estimates.detach(),
        bootstrap_estimates=bootstrap_estimates.detach(),
        standard_errors=bootstrap_estimates.std(dim=0, correction=1),
        confidence_intervals=torch.quantile(
            bootstrap_estimates,
            probabilities,
            dim=0,
        ).mT,
        confidence_level=joint.confidence_level,
        replicates=joint.replicates,
        attempts=joint.attempts,
        failed_replicates=joint.failed_replicates,
        algorithm=joint.algorithm,
    )


def _validate_curve_grid(
    covariate: Tensor,
    *,
    minimum_points: int = 2,
) -> None:
    _validate_point_grid(covariate, minimum_points=minimum_points)
    if not torch.all(torch.diff(covariate) > 0):
        raise ValueError(
            "curve covariate must contain strictly increasing points"
        )


def _validate_point_grid(
    covariate: Tensor,
    *,
    minimum_points: int = 1,
) -> None:
    if (
        covariate.ndim != 1
        or covariate.numel() < minimum_points
        or not torch.isfinite(covariate).all()
    ):
        raise ValueError(
            f"curve covariate must contain at least {minimum_points} "
            "finite points"
        )


def _quantile_probabilities(confidence_level: float, reference: Tensor) -> Tensor:
    tail_probability = (1.0 - confidence_level) / 2.0
    return torch.tensor(
        [tail_probability, 1.0 - tail_probability],
        dtype=reference.dtype,
        device=reference.device,
    )


def _contrast_name(
    coefficients: Mapping[SmoothTermKey, float],
    source_terms: tuple[SmoothTermKey, ...],
) -> str:
    components = []
    for parameter, term in source_terms:
        coefficient = coefficients[(parameter, term)]
        components.append(f"{coefficient:g}*{parameter}.{term}")
    return " + ".join(components)


def _crossing_location(
    covariate: Tensor,
    values: Tensor,
    *,
    level: float,
    direction: Literal["any", "increasing", "decreasing"],
    which: Literal["first", "last"],
) -> float | None:
    x_values = covariate.detach().cpu().tolist()
    shifted_values = (values.detach().cpu() - level).tolist()
    candidates = []
    for index in range(len(x_values) - 1):
        left = shifted_values[index]
        right = shifted_values[index + 1]
        slope = right - left
        direction_matches = (
            direction == "any"
            or (direction == "increasing" and slope > 0)
            or (direction == "decreasing" and slope < 0)
        )
        if not direction_matches:
            continue
        if left == 0.0:
            candidates.append(float(x_values[index]))
        if left * right < 0.0:
            fraction = -left / slope
            candidates.append(
                float(
                    x_values[index]
                    + fraction * (x_values[index + 1] - x_values[index])
                )
            )
        if index == len(x_values) - 2 and right == 0.0:
            candidates.append(float(x_values[index + 1]))
    if not candidates:
        return None
    return candidates[0] if which == "first" else candidates[-1]
