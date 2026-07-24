"""Wald, conditional smooth-curve, and parametric-bootstrap inference."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.stats import t as student_t
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.fitting import CGControl, RSControl
    from torchgamlss.functionals import (
        SmoothCrossingBootstrapResult,
        SmoothDerivedBootstrapResult,
        SmoothExtremumBootstrapResult,
    )
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class InferenceResult:
    """Joint coefficient covariance and Wald inference."""

    coefficient_names: tuple[str, ...]
    parameter_slices: Mapping[str, slice]
    estimates: Tensor
    covariance_matrix: Tensor
    standard_errors: Tensor
    statistics: Tensor
    p_values: Tensor
    confidence_intervals: Tensor
    degrees_of_freedom: float
    confidence_level: float
    conditional_on_smooths: bool = False

    @property
    def correlation_matrix(self) -> Tensor:
        """Return the coefficient correlation matrix."""
        scale = self.standard_errors.unsqueeze(1) * self.standard_errors.unsqueeze(0)
        return (self.covariance_matrix / scale).clamp(-1.0, 1.0)

    def by_parameter(self, values: Tensor) -> dict[str, Tensor]:
        """Split a coefficient-aligned tensor using the family parameters."""
        if values.ndim == 0 or values.shape[0] != self.estimates.numel():
            raise ValueError("values must have one row per inferred coefficient")
        return {
            parameter: values[parameter_slice]
            for parameter, parameter_slice in self.parameter_slices.items()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return the coefficient table as a pandas DataFrame."""
        values = torch.column_stack(
            (
                self.estimates,
                self.standard_errors,
                self.statistics,
                self.p_values,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            index=pd.Index(self.coefficient_names, name="coefficient"),
            columns=[
                "estimate",
                "standard_error",
                "statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            ],
        )


@dataclass(frozen=True)
class SmoothSimultaneousBand:
    """Simulation-based simultaneous band for one smooth contribution."""

    parameter: str
    term: str
    covariate: Tensor
    estimates: Tensor
    confidence_intervals: Tensor
    critical_value: float
    confidence_level: float
    simulations: int
    method: str = "conditional_gaussian_max_t"

    def to_dataframe(self) -> pd.DataFrame:
        """Return covariates, estimates, and limits as a pandas DataFrame."""
        values = torch.column_stack(
            (
                self.covariate,
                self.estimates,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=["covariate", "estimate", "ci_lower", "ci_upper"],
        )


@dataclass(frozen=True)
class SmoothInferenceResult:
    """Conditional inference for one fitted smooth contribution."""

    parameter: str
    term: str
    covariate: Tensor
    estimates: Tensor
    _covariance_root: Tensor = field(repr=False)
    standard_errors: Tensor
    confidence_intervals: Tensor
    smoothing_parameter: float
    effective_degrees_of_freedom: float
    confidence_level: float

    @property
    def variances(self) -> Tensor:
        """Return pointwise variances on the additive predictor scale."""
        return self.standard_errors.square()

    @property
    def covariance_matrix(self) -> Tensor:
        """Return the conditional covariance matrix of the fitted curve."""
        return self._covariance_root @ self._covariance_root.mT

    @property
    def correlation_matrix(self) -> Tensor:
        """Return the conditional correlation matrix of the fitted curve."""
        scale = self.standard_errors.unsqueeze(1) * self.standard_errors.unsqueeze(0)
        safe_scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
        correlation = (self.covariance_matrix / safe_scale).clamp(-1.0, 1.0)
        diagonal = torch.arange(
            self.standard_errors.numel(),
            device=self.standard_errors.device,
        )
        correlation[diagonal, diagonal] = (
            self.standard_errors > 0
        ).to(correlation.dtype)
        return correlation

    def simultaneous_confidence_band(
        self,
        *,
        simulations: int = 10_000,
        generator: torch.Generator | None = None,
    ) -> SmoothSimultaneousBand:
        """Return a conditional Gaussian max-|t| confidence band.

        Monte Carlo draws use the full curve covariance. Pass a seeded
        ``torch.Generator`` for reproducible limits.
        """
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations < 100
        ):
            raise ValueError("simulations must be an integer of at least 100")

        positive_standard_errors = self.standard_errors > 0
        if self._covariance_root.shape[1] == 0 or not positive_standard_errors.any():
            raise RuntimeError(
                "smooth covariance has no positive variance; "
                "a simultaneous band is unavailable"
            )

        standardized_root = (
            self._covariance_root[positive_standard_errors]
            / self.standard_errors[positive_standard_errors].unsqueeze(1)
        )
        maximum_statistics = torch.empty(
            simulations,
            dtype=self.estimates.dtype,
            device=self.estimates.device,
        )
        batch_size = 1_024
        for start in range(0, simulations, batch_size):
            stop = min(start + batch_size, simulations)
            normal_draws = torch.randn(
                (stop - start, self._covariance_root.shape[1]),
                dtype=self.estimates.dtype,
                device=self.estimates.device,
                generator=generator,
            )
            maximum_statistics[start:stop] = (
                normal_draws @ standardized_root.mT
            ).abs().amax(dim=1)

        critical_value = float(
            torch.quantile(maximum_statistics, self.confidence_level)
        )
        confidence_intervals = torch.column_stack(
            (
                self.estimates - critical_value * self.standard_errors,
                self.estimates + critical_value * self.standard_errors,
            )
        )
        return SmoothSimultaneousBand(
            parameter=self.parameter,
            term=self.term,
            covariate=self.covariate,
            estimates=self.estimates,
            confidence_intervals=confidence_intervals.detach(),
            critical_value=critical_value,
            confidence_level=self.confidence_level,
            simulations=simulations,
            method="conditional_gaussian_max_t",
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return covariates, fitted contributions, and pointwise intervals."""
        values = torch.column_stack(
            (
                self.covariate,
                self.estimates,
                self.standard_errors,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=[
                "covariate",
                "estimate",
                "standard_error",
                "ci_lower",
                "ci_upper",
            ],
        )


@dataclass(frozen=True)
class SmoothBootstrapResult:
    """Parametric-bootstrap inference for one fitted smooth contribution."""

    parameter: str
    term: str
    covariate: Tensor
    estimates: Tensor
    bootstrap_estimates: Tensor = field(repr=False)
    standard_errors: Tensor
    confidence_intervals: Tensor
    smoothing_parameter: float
    bootstrap_smoothing_parameters: Tensor = field(repr=False)
    confidence_level: float
    replicates: int
    attempts: int
    failed_replicates: int
    algorithm: str

    @property
    def bootstrap_mean(self) -> Tensor:
        """Return the replicate-wise mean curve."""
        return self.bootstrap_estimates.mean(dim=0)

    @property
    def bias(self) -> Tensor:
        """Return bootstrap mean minus the original fitted curve."""
        return self.bootstrap_mean - self.estimates

    @property
    def covariance_matrix(self) -> Tensor:
        """Return the empirical covariance across bootstrap curves."""
        centered = self.bootstrap_estimates - self.bootstrap_mean
        return centered.mT @ centered / (self.replicates - 1)

    @property
    def smoothing_parameter_standard_error(self) -> float:
        """Return the bootstrap standard error of the smoothing parameter."""
        return float(self.bootstrap_smoothing_parameters.std(correction=1))

    @property
    def smoothing_parameter_bootstrap_mean(self) -> float:
        """Return the mean smoothing parameter across successful refits."""
        return float(self.bootstrap_smoothing_parameters.mean())

    @property
    def smoothing_parameter_bias(self) -> float:
        """Return bootstrap mean lambda minus the original fitted lambda."""
        return self.smoothing_parameter_bootstrap_mean - self.smoothing_parameter

    @property
    def smoothing_parameter_confidence_interval(self) -> Tensor:
        """Return the percentile interval for the smoothing parameter."""
        tail_probability = (1.0 - self.confidence_level) / 2.0
        probabilities = torch.tensor(
            [tail_probability, 1.0 - tail_probability],
            dtype=self.bootstrap_smoothing_parameters.dtype,
            device=self.bootstrap_smoothing_parameters.device,
        )
        return torch.quantile(self.bootstrap_smoothing_parameters, probabilities)

    @property
    def failure_rate(self) -> float:
        """Return the fraction of attempted refits that failed."""
        return self.failed_replicates / self.attempts

    def simultaneous_confidence_band(self) -> SmoothSimultaneousBand:
        """Return a max-|t| band from the successful bootstrap refits."""
        positive_standard_errors = self.standard_errors > 0
        if not positive_standard_errors.any():
            raise RuntimeError(
                "bootstrap curves have no positive variance; "
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
        return SmoothSimultaneousBand(
            parameter=self.parameter,
            term=self.term,
            covariate=self.covariate,
            estimates=self.estimates,
            confidence_intervals=confidence_intervals,
            critical_value=critical_value,
            confidence_level=self.confidence_level,
            simulations=self.replicates,
            method="parametric_bootstrap_max_t",
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return curve estimates and pointwise bootstrap inference."""
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
class SmoothJointBandResult:
    """Bootstrap max-|t| bands calibrated jointly over several smooths."""

    bands: Mapping[str, Mapping[str, SmoothSimultaneousBand]]
    term_order: tuple[tuple[str, str], ...]
    critical_value: float
    confidence_level: float
    replicates: int
    method: str = "parametric_bootstrap_joint_max_t"

    def __getitem__(self, parameter: str) -> Mapping[str, SmoothSimultaneousBand]:
        """Return all jointly calibrated bands for one parameter."""
        return self.bands[parameter]

    def to_dataframe(self) -> pd.DataFrame:
        """Return all jointly calibrated bands in long format."""
        frames = []
        for parameter, term in self.term_order:
            frame = self.bands[parameter][term].to_dataframe()
            frame.insert(0, "term", term)
            frame.insert(0, "parameter", parameter)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class SmoothJointBootstrapResult:
    """Aligned parametric-bootstrap inference for several fitted smooths."""

    curves: Mapping[str, Mapping[str, SmoothBootstrapResult]]
    term_order: tuple[tuple[str, str], ...]
    confidence_level: float
    replicates: int
    attempts: int
    failed_replicates: int
    algorithm: str

    @classmethod
    def _from_curves(
        cls,
        curves: Mapping[str, Mapping[str, SmoothBootstrapResult]],
    ) -> SmoothJointBootstrapResult:
        """Build a joint result from internally aligned bootstrap curves."""
        term_order = tuple(
            (parameter, term)
            for parameter, parameter_curves in curves.items()
            for term in parameter_curves
        )
        if not term_order:
            raise ValueError("joint smooth bootstrap requires at least one curve")
        first_parameter, first_term = term_order[0]
        first = curves[first_parameter][first_term]
        for parameter, term in term_order[1:]:
            current = curves[parameter][term]
            metadata = (
                current.confidence_level,
                current.replicates,
                current.attempts,
                current.failed_replicates,
                current.algorithm,
            )
            expected = (
                first.confidence_level,
                first.replicates,
                first.attempts,
                first.failed_replicates,
                first.algorithm,
            )
            if metadata != expected:
                raise ValueError(
                    "joint smooth bootstrap curves must come from the same run"
                )
        copied_curves = {
            parameter: dict(parameter_curves)
            for parameter, parameter_curves in curves.items()
        }
        return cls(
            curves=copied_curves,
            term_order=term_order,
            confidence_level=first.confidence_level,
            replicates=first.replicates,
            attempts=first.attempts,
            failed_replicates=first.failed_replicates,
            algorithm=first.algorithm,
        )

    def __getitem__(self, parameter: str) -> Mapping[str, SmoothBootstrapResult]:
        """Return all aligned bootstrap curves for one parameter."""
        return self.curves[parameter]

    @property
    def failure_rate(self) -> float:
        """Return the fraction of attempted joint refits that failed."""
        return self.failed_replicates / self.attempts

    @property
    def term_slices(self) -> dict[tuple[str, str], slice]:
        """Return slices locating each curve in stacked point-wise results."""
        result = {}
        start = 0
        for key in self.term_order:
            curve = self._curve(key)
            stop = start + curve.estimates.numel()
            result[key] = slice(start, stop)
            start = stop
        return result

    @property
    def point_labels(self) -> tuple[tuple[str, str, int], ...]:
        """Return parameter, term, and point-index labels for stacked results."""
        return tuple(
            (parameter, term, index)
            for parameter, term in self.term_order
            for index in range(self.curves[parameter][term].estimates.numel())
        )

    @property
    def estimates(self) -> Tensor:
        """Return original curves concatenated in ``term_order``."""
        return torch.cat([self._curve(key).estimates for key in self.term_order])

    @property
    def bootstrap_estimates(self) -> Tensor:
        """Return aligned bootstrap curves as replicates by stacked points."""
        return torch.cat(
            [self._curve(key).bootstrap_estimates for key in self.term_order],
            dim=1,
        )

    @property
    def covariance_matrix(self) -> Tensor:
        """Return the empirical covariance over every stacked curve point."""
        estimates = self.bootstrap_estimates
        centered = estimates - estimates.mean(dim=0)
        return centered.mT @ centered / (self.replicates - 1)

    @property
    def correlation_matrix(self) -> Tensor:
        """Return the empirical correlation over every stacked curve point."""
        covariance = self.covariance_matrix
        standard_errors = torch.diagonal(covariance).clamp_min(0).sqrt()
        scale = standard_errors.unsqueeze(1) * standard_errors.unsqueeze(0)
        correlation = covariance / scale
        return correlation.clamp(-1.0, 1.0)

    def covariance_block(
        self,
        first: tuple[str, str],
        second: tuple[str, str],
    ) -> Tensor:
        """Return empirical cross-covariance between two fitted curves."""
        first_curve = self._curve(first).bootstrap_estimates
        second_curve = self._curve(second).bootstrap_estimates
        first_centered = first_curve - first_curve.mean(dim=0)
        second_centered = second_curve - second_curve.mean(dim=0)
        return first_centered.mT @ second_centered / (self.replicates - 1)

    @property
    def smoothing_parameters(self) -> Tensor:
        """Return fitted smoothing parameters in ``term_order``."""
        reference = self._curve(self.term_order[0]).bootstrap_smoothing_parameters
        return torch.tensor(
            [
                self._curve(key).smoothing_parameter
                for key in self.term_order
            ],
            dtype=reference.dtype,
            device=reference.device,
        )

    @property
    def bootstrap_smoothing_parameters(self) -> Tensor:
        """Return aligned smoothing parameters as replicates by terms."""
        return torch.column_stack(
            [
                self._curve(key).bootstrap_smoothing_parameters
                for key in self.term_order
            ]
        )

    @property
    def smoothing_parameter_covariance_matrix(self) -> Tensor:
        """Return empirical covariance among reselected smoothing parameters."""
        parameters = self.bootstrap_smoothing_parameters
        centered = parameters - parameters.mean(dim=0)
        return centered.mT @ centered / (self.replicates - 1)

    @property
    def smoothing_parameter_correlation_matrix(self) -> Tensor:
        """Return empirical correlations among smoothing parameters."""
        covariance = self.smoothing_parameter_covariance_matrix
        standard_errors = torch.diagonal(covariance).clamp_min(0).sqrt()
        scale = standard_errors.unsqueeze(1) * standard_errors.unsqueeze(0)
        correlation = covariance / scale
        return correlation.clamp(-1.0, 1.0)

    def linear_contrast(
        self,
        coefficients: Mapping[tuple[str, str], float],
        *,
        name: str | None = None,
    ) -> SmoothDerivedBootstrapResult:
        """Return a fixed linear combination of aligned smooth curves."""
        from torchgamlss.functionals import smooth_linear_contrast

        return smooth_linear_contrast(self, coefficients, name=name)

    def difference(
        self,
        first: tuple[str, str],
        second: tuple[str, str],
        *,
        name: str | None = None,
    ) -> SmoothDerivedBootstrapResult:
        """Return the aligned pointwise difference between two smooths."""
        if first == second:
            raise ValueError("difference requires two distinct smooth terms")
        return self.linear_contrast(
            {first: 1.0, second: -1.0},
            name=name or f"{first[0]}.{first[1]} - {second[0]}.{second[1]}",
        )

    def derivative(
        self,
        term: tuple[str, str],
        *,
        order: Literal[1, 2] = 1,
        name: str | None = None,
    ) -> SmoothDerivedBootstrapResult:
        """Return first- or second-derivative bootstrap inference."""
        from torchgamlss.functionals import smooth_derivative

        return smooth_derivative(self, term, order=order, name=name)

    def extremum(
        self,
        term: tuple[str, str],
        *,
        kind: Literal["maximum", "minimum"] = "maximum",
    ) -> SmoothExtremumBootstrapResult:
        """Return bootstrap inference for one curve's grid extremum."""
        from torchgamlss.functionals import smooth_curve_result

        return smooth_curve_result(self, term).extremum(kind=kind)

    def crossing(
        self,
        term: tuple[str, str],
        *,
        level: float = 0.0,
        direction: Literal["any", "increasing", "decreasing"] = "any",
        which: Literal["first", "last"] = "first",
    ) -> SmoothCrossingBootstrapResult:
        """Return bootstrap inference for one curve's selected crossing."""
        from torchgamlss.functionals import smooth_curve_result

        return smooth_curve_result(self, term).crossing(
            level=level,
            direction=direction,
            which=which,
        )

    def simultaneous_confidence_bands(
        self,
        terms: Sequence[tuple[str, str]] | None = None,
    ) -> SmoothJointBandResult:
        """Return max-|t| bands jointly calibrated over selected smooths."""
        selected_terms = self._selected_terms(terms)
        standardized_deviations = []
        for key in selected_terms:
            curve = self._curve(key)
            positive_standard_errors = curve.standard_errors > 0
            if positive_standard_errors.any():
                standardized_deviations.append(
                    (
                        curve.bootstrap_estimates[:, positive_standard_errors]
                        - curve.estimates[positive_standard_errors]
                    )
                    / curve.standard_errors[positive_standard_errors]
                )
        if not standardized_deviations:
            raise RuntimeError(
                "selected bootstrap curves have no positive variance; "
                "joint simultaneous bands are unavailable"
            )
        maximum_statistics = torch.cat(
            standardized_deviations,
            dim=1,
        ).abs().amax(dim=1)
        critical_value = float(
            torch.quantile(maximum_statistics, self.confidence_level)
        )
        bands: dict[str, dict[str, SmoothSimultaneousBand]] = {}
        for parameter, term in selected_terms:
            curve = self.curves[parameter][term]
            confidence_intervals = torch.column_stack(
                (
                    curve.estimates - critical_value * curve.standard_errors,
                    curve.estimates + critical_value * curve.standard_errors,
                )
            )
            bands.setdefault(parameter, {})[term] = SmoothSimultaneousBand(
                parameter=parameter,
                term=term,
                covariate=curve.covariate,
                estimates=curve.estimates,
                confidence_intervals=confidence_intervals,
                critical_value=critical_value,
                confidence_level=self.confidence_level,
                simulations=self.replicates,
                method="parametric_bootstrap_joint_max_t",
            )
        return SmoothJointBandResult(
            bands=bands,
            term_order=selected_terms,
            critical_value=critical_value,
            confidence_level=self.confidence_level,
            replicates=self.replicates,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all pointwise curve summaries in long format."""
        frames = []
        for parameter, term in self.term_order:
            frame = self.curves[parameter][term].to_dataframe()
            frame.insert(0, "term", term)
            frame.insert(0, "parameter", parameter)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _curve(self, key: tuple[str, str]) -> SmoothBootstrapResult:
        try:
            parameter, term = key
            return self.curves[parameter][term]
        except (KeyError, TypeError, ValueError) as error:
            raise KeyError(f"unknown smooth term {key!r}") from error

    def _selected_terms(
        self,
        terms: Sequence[tuple[str, str]] | None,
    ) -> tuple[tuple[str, str], ...]:
        if terms is None:
            return self.term_order
        if not terms:
            raise ValueError("terms must contain at least one smooth term")
        if len(set(terms)) != len(terms):
            raise ValueError("terms must not contain duplicates")
        selected_terms = tuple(terms)
        for key in selected_terms:
            self._curve(key)
        return selected_terms


def coefficient_inference(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    conditional_on_smooths: bool = False,
    confidence_level: float = 0.95,
    degrees_of_freedom: float | None = None,
) -> InferenceResult:
    """Compute full-Hessian covariance and t-based Wald inference.

    When ``conditional_on_smooths`` is true, fitted smooth contributions are
    held fixed and only the linear coefficients enter the Hessian. This follows
    ``gamlss::vcov.gamlss()`` and intentionally excludes spline-coefficient and
    smoothing-parameter uncertainty.
    """
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    has_smooths = any(
        model.smooth_terms[parameter] for parameter in model.family.parameter_names
    )
    if has_smooths and not conditional_on_smooths:
        raise ValueError(
            "coefficient inference currently supports parametric models without "
            "smooth terms; set conditional_on_smooths=True for conditional "
            "linear-coefficient inference"
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
            "inference response must be a non-empty finite vector matching "
            "the model dtype and device"
        )

    contributions = model.term_contributions(
        design_matrices,
        offsets,
        smooth_covariates=smooth_covariates,
    )
    observation_count = response.numel()
    if any(
        contribution.offset.numel() != observation_count
        for contribution in contributions.values()
    ):
        raise ValueError("design matrices must have one row per response")
    case_weights = model._validated_weights(response, weights)

    estimates = torch.cat(
        [
            model.coefficients[parameter].detach()
            for parameter in model.family.parameter_names
        ]
    )
    parameter_slices = _parameter_slices(model)
    coefficient_count = estimates.numel()
    if degrees_of_freedom is None:
        if has_smooths:
            raise ValueError(
                "degrees_of_freedom is required for conditional smooth "
                "inference; use the effective observation count minus the "
                "fit result's effective_degrees_of_freedom"
            )
        if torch.equal(case_weights, case_weights.round()):
            effective_observations = float(case_weights.sum())
        else:
            effective_observations = float((case_weights > 0).sum())
        degrees_of_freedom = effective_observations - coefficient_count
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError(
            "inference requires positive residual degrees of freedom "
            "(sum of weights minus coefficient count)"
        )

    fixed_contributions = {}
    for parameter, contribution in contributions.items():
        fixed = contribution.offset
        for smooth in contribution.smooth.values():
            fixed = fixed + smooth.detach()
        fixed_contributions[parameter] = fixed

    def objective(flat_coefficients: Tensor) -> Tensor:
        predictors = {
            parameter: design_matrices[parameter]
            @ flat_coefficients[parameter_slices[parameter]]
            + fixed_contributions[parameter]
            for parameter in model.family.parameter_names
        }
        parameters = model.family.parameters_from_predictors(predictors)
        return -(model.family.log_prob(response, parameters) * case_weights).sum()

    hessian = torch.autograd.functional.hessian(objective, estimates)
    hessian = (hessian + hessian.mT) / 2.0
    if not torch.isfinite(hessian).all():
        raise RuntimeError("coefficient Hessian is not finite")
    factor, info = torch.linalg.cholesky_ex(hessian)
    if int(info) != 0:
        raise RuntimeError(
            "coefficient Hessian is not positive definite; covariance is unavailable"
        )
    covariance = torch.cholesky_inverse(factor)
    covariance = (covariance + covariance.mT) / 2.0
    standard_errors = torch.diagonal(covariance).sqrt()
    statistics = estimates / standard_errors

    statistic_values = statistics.detach().cpu().numpy()
    p_values_numpy = 2.0 * student_t.sf(
        np.abs(statistic_values),
        degrees_of_freedom,
    )
    critical_value = float(
        student_t.ppf(
            0.5 + confidence_level / 2.0,
            degrees_of_freedom,
        )
    )
    p_values = torch.as_tensor(
        p_values_numpy,
        dtype=estimates.dtype,
        device=estimates.device,
    )
    confidence_intervals = torch.column_stack(
        (
            estimates - critical_value * standard_errors,
            estimates + critical_value * standard_errors,
        )
    )
    return InferenceResult(
        coefficient_names=_coefficient_names(model),
        parameter_slices=parameter_slices,
        estimates=estimates.clone(),
        covariance_matrix=covariance.detach(),
        standard_errors=standard_errors.detach(),
        statistics=statistics.detach(),
        p_values=p_values,
        confidence_intervals=confidence_intervals.detach(),
        degrees_of_freedom=degrees_of_freedom,
        confidence_level=confidence_level,
        conditional_on_smooths=has_smooths,
    )


def smooth_term_inference(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    evaluation_smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, SmoothInferenceResult]]:
    """Infer fitted smooth curves conditional on their smoothing parameters.

    The pointwise variance follows ``gamlss.pb()``: it uses the inverse
    penalized working-weight system and removes the unpenalized polynomial
    null-space contribution already represented by the linear predictor.
    """
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    if not any(
        model.smooth_terms[parameter] for parameter in model.family.parameter_names
    ):
        raise ValueError("smooth inference requires at least one smooth term")

    model_parameter = next(model.parameters())
    if (
        response.ndim != 1
        or response.numel() < 1
        or response.dtype != model_parameter.dtype
        or response.device != model_parameter.device
        or not torch.isfinite(response).all()
    ):
        raise ValueError(
            "inference response must be a non-empty finite vector matching "
            "the model dtype and device"
        )

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
    case_weights = model._validated_weights(response, weights)
    predictors = {
        parameter: contribution.total.detach()
        for parameter, contribution in contributions.items()
    }
    parameters = model.family.parameters_from_predictors(predictors)
    second_derivatives = model.family.expected_second_derivatives(response, parameters)

    evaluation_covariates = (
        smooth_covariates
        if evaluation_smooth_covariates is None
        else evaluation_smooth_covariates
    )
    _validate_smooth_evaluation_mapping(model, evaluation_covariates)
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    results: dict[str, dict[str, SmoothInferenceResult]] = {}

    for parameter in model.family.parameter_names:
        parameter_terms = model.smooth_terms[parameter]
        results[parameter] = {}
        if not parameter_terms:
            continue

        second = second_derivatives[(parameter, parameter)].clamp(max=-1e-15)
        inverse_link_derivative = model.family.links[parameter].inverse_derivative(
            predictors[parameter]
        )
        derivative = inverse_link_derivative.reciprocal()
        working_weights = -(second / derivative.square())
        working_weights = working_weights.clamp(min=1e-10, max=1e10).detach()
        combined_weights = working_weights * case_weights
        if (
            not torch.isfinite(combined_weights).all()
            or (combined_weights < 0).any()
            or combined_weights.sum() <= 0
        ):
            raise RuntimeError(
                f"conditional smooth weights for {parameter!r} are invalid"
            )

        for term_name, term in parameter_terms.items():
            training_covariate = smooth_covariates[parameter][term_name].detach()
            evaluation_covariate = evaluation_covariates[parameter][term_name].detach()
            training_basis = term.basis(training_covariate)
            evaluation_basis = term.basis(evaluation_covariate)
            penalty = term.penalty_matrix()
            system = training_basis.mT @ (
                combined_weights.unsqueeze(-1) * training_basis
            ) + term.smoothing_parameter * (penalty.mT @ penalty)
            system_inverse = torch.linalg.pinv(system, hermitian=True)
            coefficient_covariance = system_inverse

            nullity = term.penalty_nullity
            if nullity:
                powers = torch.arange(
                    nullity,
                    dtype=training_covariate.dtype,
                    device=training_covariate.device,
                )
                training_null_basis = training_covariate.unsqueeze(-1).pow(powers)
                null_system = training_null_basis.mT @ (
                    combined_weights.unsqueeze(-1) * training_null_basis
                )
                null_inverse = torch.linalg.pinv(null_system, hermitian=True)
                null_basis_coefficients = torch.linalg.lstsq(
                    training_basis,
                    training_null_basis,
                ).solution
                coefficient_covariance = coefficient_covariance - (
                    null_basis_coefficients
                    @ null_inverse
                    @ null_basis_coefficients.mT
                )
            coefficient_covariance = (
                coefficient_covariance + coefficient_covariance.mT
            ) / 2.0
            eigenvalues, eigenvectors = torch.linalg.eigh(coefficient_covariance)
            covariance_scale = eigenvalues.abs().max().clamp_min(
                torch.finfo(eigenvalues.dtype).tiny
            )
            covariance_tolerance = (
                math.sqrt(torch.finfo(eigenvalues.dtype).eps) * covariance_scale
            )
            if (eigenvalues < -covariance_tolerance).any():
                raise RuntimeError(
                    f"conditional smooth covariance for "
                    f"{parameter!r}.{term_name} is not positive semidefinite"
                )
            retained = eigenvalues > covariance_tolerance
            coefficient_root = (
                eigenvectors[:, retained] * eigenvalues[retained].sqrt()
            )
            covariance_root = evaluation_basis @ coefficient_root
            variances = covariance_root.square().sum(dim=1)
            standard_errors = variances.sqrt()
            estimates = term(evaluation_covariate).detach()
            confidence_intervals = torch.column_stack(
                (
                    estimates - critical_value * standard_errors,
                    estimates + critical_value * standard_errors,
                )
            )
            effective_degrees_of_freedom = float(
                term.effective_degrees_of_freedom(
                    training_covariate,
                    combined_weights,
                )
            )
            results[parameter][term_name] = SmoothInferenceResult(
                parameter=parameter,
                term=term_name,
                covariate=evaluation_covariate.detach().clone(),
                estimates=estimates,
                _covariance_root=covariance_root.detach(),
                standard_errors=standard_errors.detach(),
                confidence_intervals=confidence_intervals.detach(),
                smoothing_parameter=term.smoothing_parameter,
                effective_degrees_of_freedom=effective_degrees_of_freedom,
                confidence_level=confidence_level,
            )

    return results


def smooth_term_bootstrap(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    evaluation_smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    replicates: int = 999,
    max_attempts: int | None = None,
    algorithm: Literal["rs", "cg"] = "rs",
    control: RSControl | CGControl | None = None,
    confidence_level: float = 0.95,
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, SmoothBootstrapResult]]:
    """Refit parametric bootstrap samples and summarize smooth uncertainty.

    Each replicate draws a response from the fitted distribution and reruns
    the selected classical fitting algorithm. Automatic smoothing-parameter
    selection is therefore repeated rather than held fixed.
    """
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
    if not any(
        model.smooth_terms[parameter] for parameter in model.family.parameter_names
    ):
        raise ValueError("smooth bootstrap requires at least one smooth term")

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
            "bootstrap response must be a non-empty finite vector matching "
            "the model dtype and device"
        )
    model.family.validate_response(response, context="bootstrap")
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

    evaluation_covariates = (
        smooth_covariates
        if evaluation_smooth_covariates is None
        else evaluation_smooth_covariates
    )
    _validate_smooth_evaluation_mapping(model, evaluation_covariates)
    fitted_parameters = {
        parameter: value.detach()
        for parameter, value in model.family.parameters_from_predictors(
            {
                parameter: contribution.total.detach()
                for parameter, contribution in contributions.items()
            }
        ).items()
    }

    original_estimates: dict[str, dict[str, Tensor]] = {}
    original_smoothing_parameters: dict[str, dict[str, float]] = {}
    bootstrap_estimates: dict[str, dict[str, Tensor]] = {}
    bootstrap_smoothing_parameters: dict[str, dict[str, Tensor]] = {}
    for parameter in model.family.parameter_names:
        original_estimates[parameter] = {}
        original_smoothing_parameters[parameter] = {}
        bootstrap_estimates[parameter] = {}
        bootstrap_smoothing_parameters[parameter] = {}
        for term_name, term in model.smooth_terms[parameter].items():
            evaluation_covariate = evaluation_covariates[parameter][term_name]
            estimate = term(evaluation_covariate).detach()
            original_estimates[parameter][term_name] = estimate.clone()
            original_smoothing_parameters[parameter][
                term_name
            ] = term.smoothing_parameter
            bootstrap_estimates[parameter][term_name] = torch.empty(
                (replicates, estimate.numel()),
                dtype=estimate.dtype,
                device=estimate.device,
            )
            bootstrap_smoothing_parameters[parameter][term_name] = torch.empty(
                replicates,
                dtype=estimate.dtype,
                device=estimate.device,
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
                context=f"bootstrap attempt {attempts}",
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
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failure_messages.append(str(error))
            continue
        if not fit_result.converged:
            failure_messages.append("classical fit did not converge")
            continue

        with torch.no_grad():
            for parameter in model.family.parameter_names:
                for term_name, term in bootstrap_model.smooth_terms[
                    parameter
                ].items():
                    bootstrap_estimates[parameter][term_name][
                        successful_replicates
                    ].copy_(
                        term(evaluation_covariates[parameter][term_name])
                    )
                    bootstrap_smoothing_parameters[parameter][term_name][
                        successful_replicates
                    ] = term.smoothing_parameter
        successful_replicates += 1

    if successful_replicates < replicates:
        last_failure = failure_messages[-1] if failure_messages else "unknown failure"
        raise RuntimeError(
            f"bootstrap obtained {successful_replicates} successful fits out of "
            f"{replicates} after {attempts} attempts; last failure: {last_failure}"
        )

    tail_probability = (1.0 - confidence_level) / 2.0
    quantile_probabilities = torch.tensor(
        [tail_probability, 1.0 - tail_probability],
        dtype=response.dtype,
        device=response.device,
    )
    results: dict[str, dict[str, SmoothBootstrapResult]] = {}
    for parameter in model.family.parameter_names:
        results[parameter] = {}
        for term_name in model.smooth_terms[parameter]:
            term_bootstrap_estimates = bootstrap_estimates[parameter][term_name]
            standard_errors = term_bootstrap_estimates.std(dim=0, correction=1)
            confidence_intervals = torch.quantile(
                term_bootstrap_estimates,
                quantile_probabilities,
                dim=0,
            ).mT
            results[parameter][term_name] = SmoothBootstrapResult(
                parameter=parameter,
                term=term_name,
                covariate=evaluation_covariates[parameter][term_name]
                .detach()
                .clone(),
                estimates=original_estimates[parameter][term_name],
                bootstrap_estimates=term_bootstrap_estimates,
                standard_errors=standard_errors,
                confidence_intervals=confidence_intervals,
                smoothing_parameter=original_smoothing_parameters[parameter][
                    term_name
                ],
                bootstrap_smoothing_parameters=bootstrap_smoothing_parameters[
                    parameter
                ][term_name],
                confidence_level=confidence_level,
                replicates=replicates,
                attempts=attempts,
                failed_replicates=attempts - replicates,
                algorithm=algorithm,
            )

    return results


def _parameter_slices(model: GAMLSS) -> dict[str, slice]:
    result = {}
    start = 0
    for parameter in model.family.parameter_names:
        stop = start + model.coefficients[parameter].numel()
        result[parameter] = slice(start, stop)
        start = stop
    return result


def _coefficient_names(model: GAMLSS) -> tuple[str, ...]:
    if model._formula_encoder is None:
        return tuple(
            f"{parameter}[{index}]"
            for parameter in model.family.parameter_names
            for index in range(model.coefficients[parameter].numel())
        )
    return tuple(
        f"{parameter}.{column}"
        for parameter in model.family.parameter_names
        for column in model.formula_column_names[parameter]
    )


def _validate_smooth_evaluation_mapping(
    model: GAMLSS,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
) -> None:
    expected_parameters = set(model.family.parameter_names)
    extra_parameters = set(smooth_covariates).difference(expected_parameters)
    if extra_parameters:
        raise ValueError(
            "Evaluation smooth covariates contain unknown parameters: "
            f"{sorted(extra_parameters)}"
        )
    for parameter in model.family.parameter_names:
        expected_terms = set(model.smooth_terms[parameter])
        supplied_terms = set(smooth_covariates.get(parameter, {}))
        if expected_terms != supplied_terms:
            raise ValueError(
                f"Evaluation smooth covariates for {parameter!r} do not match "
                f"configured terms: missing={sorted(expected_terms - supplied_terms)}, "
                f"extra={sorted(supplied_terms - expected_terms)}"
            )
