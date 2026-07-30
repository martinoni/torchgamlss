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
    from torchgamlss.smooths import SmoothTerm


SmoothingParameterValue = float | tuple[float, ...]


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
        covariates, covariate_columns = _covariate_frame_values(self.covariate)
        values = torch.column_stack(
            (
                covariates,
                self.estimates,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=[
                *covariate_columns,
                "estimate",
                "ci_lower",
                "ci_upper",
            ],
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
    smoothing_parameter: SmoothingParameterValue
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
        covariates, covariate_columns = _covariate_frame_values(self.covariate)
        values = torch.column_stack(
            (
                covariates,
                self.estimates,
                self.standard_errors,
                self.confidence_intervals,
            )
        )
        return pd.DataFrame(
            values.detach().cpu().numpy(),
            columns=[
                *covariate_columns,
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
    smoothing_parameter: SmoothingParameterValue
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
    def smoothing_parameter_count(self) -> int:
        """Return the number of penalties represented by this smooth term."""
        if isinstance(self.smoothing_parameter, tuple):
            return len(self.smoothing_parameter)
        return 1

    @property
    def smoothing_parameter_standard_error(self) -> float | Tensor:
        """Return bootstrap standard errors for the smoothing parameters."""
        standard_errors = self.bootstrap_smoothing_parameters.std(
            dim=0,
            correction=1,
        )
        if self.smoothing_parameter_count == 1:
            return float(standard_errors)
        return standard_errors

    @property
    def smoothing_parameter_bootstrap_mean(self) -> float | Tensor:
        """Return mean smoothing parameters across successful refits."""
        means = self.bootstrap_smoothing_parameters.mean(dim=0)
        if self.smoothing_parameter_count == 1:
            return float(means)
        return means

    @property
    def smoothing_parameter_bias(self) -> float | Tensor:
        """Return bootstrap mean lambdas minus the original fitted lambdas."""
        bootstrap_mean = self.smoothing_parameter_bootstrap_mean
        if self.smoothing_parameter_count == 1:
            original = _smoothing_parameter_tensor(
                self.smoothing_parameter,
                reference=self.bootstrap_smoothing_parameters,
            )
            return float(bootstrap_mean) - float(original[0])
        reference = _smoothing_parameter_tensor(
            self.smoothing_parameter,
            reference=self.bootstrap_smoothing_parameters,
        )
        return bootstrap_mean - reference

    @property
    def smoothing_parameter_confidence_interval(self) -> Tensor:
        """Return the percentile interval for the smoothing parameter."""
        tail_probability = (1.0 - self.confidence_level) / 2.0
        probabilities = torch.tensor(
            [tail_probability, 1.0 - tail_probability],
            dtype=self.bootstrap_smoothing_parameters.dtype,
            device=self.bootstrap_smoothing_parameters.device,
        )
        intervals = torch.quantile(
            self.bootstrap_smoothing_parameters,
            probabilities,
            dim=0,
        )
        if self.smoothing_parameter_count == 1:
            return intervals
        return intervals.mT

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
        covariates, covariate_columns = _covariate_frame_values(self.covariate)
        values = torch.column_stack(
            (
                covariates,
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
                *covariate_columns,
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
class SmoothJointInferenceBand:
    """Gaussian max-|t| bands from the joint penalized covariance."""

    bands: Mapping[str, Mapping[str, SmoothSimultaneousBand]]
    term_order: tuple[tuple[str, str], ...]
    critical_value: float
    confidence_level: float
    simulations: int
    method: str = "analytic_joint_gaussian_max_t"

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
class SmoothJointInferenceResult:
    """Fixed-lambda analytic covariance for all fitted smooth terms."""

    curves: Mapping[str, Mapping[str, SmoothInferenceResult]]
    term_order: tuple[tuple[str, str], ...]
    coefficient_names: tuple[str, ...]
    linear_coefficient_slices: Mapping[str, slice]
    smooth_coefficient_slices: Mapping[tuple[str, str], slice]
    coefficient_estimates: Tensor
    _coefficient_covariance_root: Tensor = field(repr=False)
    confidence_level: float

    def __getitem__(self, parameter: str) -> Mapping[str, SmoothInferenceResult]:
        """Return all analytic smooth results for one parameter."""
        return self.curves[parameter]

    @property
    def coefficient_covariance_matrix(self) -> Tensor:
        """Return covariance for linear and constrained spline coefficients."""
        root = self._coefficient_covariance_root
        return root @ root.mT

    @property
    def coefficient_standard_errors(self) -> Tensor:
        """Return coefficient standard errors in ``coefficient_names`` order."""
        return self._coefficient_covariance_root.square().sum(dim=1).sqrt()

    @property
    def coefficient_correlation_matrix(self) -> Tensor:
        """Return correlations among linear and spline coefficients."""
        covariance = self.coefficient_covariance_matrix
        standard_errors = self.coefficient_standard_errors
        scale = standard_errors.unsqueeze(1) * standard_errors.unsqueeze(0)
        safe_scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
        correlation = (covariance / safe_scale).clamp(-1.0, 1.0)
        diagonal = torch.arange(standard_errors.numel(), device=covariance.device)
        correlation[diagonal, diagonal] = (standard_errors > 0).to(covariance.dtype)
        return correlation

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
        """Return fitted curves concatenated in ``term_order``."""
        return torch.cat([self._curve(key).estimates for key in self.term_order])

    @property
    def _covariance_root(self) -> Tensor:
        return torch.cat(
            [self._curve(key)._covariance_root for key in self.term_order],
            dim=0,
        )

    @property
    def covariance_matrix(self) -> Tensor:
        """Return analytic covariance over every stacked curve point."""
        root = self._covariance_root
        return root @ root.mT

    @property
    def standard_errors(self) -> Tensor:
        """Return stacked curve standard errors."""
        return self._covariance_root.square().sum(dim=1).sqrt()

    @property
    def correlation_matrix(self) -> Tensor:
        """Return correlations over every stacked curve point."""
        covariance = self.covariance_matrix
        standard_errors = self.standard_errors
        scale = standard_errors.unsqueeze(1) * standard_errors.unsqueeze(0)
        safe_scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
        correlation = (covariance / safe_scale).clamp(-1.0, 1.0)
        diagonal = torch.arange(standard_errors.numel(), device=covariance.device)
        correlation[diagonal, diagonal] = (standard_errors > 0).to(covariance.dtype)
        return correlation

    def covariance_block(
        self,
        first: tuple[str, str],
        second: tuple[str, str],
    ) -> Tensor:
        """Return analytic cross-covariance between two fitted curves."""
        first_root = self._curve(first)._covariance_root
        second_root = self._curve(second)._covariance_root
        return first_root @ second_root.mT

    def simultaneous_confidence_bands(
        self,
        terms: Sequence[tuple[str, str]] | None = None,
        *,
        simulations: int = 10_000,
        generator: torch.Generator | None = None,
    ) -> SmoothJointInferenceBand:
        """Return Gaussian max-|t| bands calibrated over selected smooths."""
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations < 100
        ):
            raise ValueError("simulations must be an integer of at least 100")
        selected_terms = self._selected_terms(terms)
        selected_roots = torch.cat(
            [self._curve(key)._covariance_root for key in selected_terms],
            dim=0,
        )
        selected_standard_errors = selected_roots.square().sum(dim=1).sqrt()
        positive = selected_standard_errors > 0
        if selected_roots.shape[1] == 0 or not positive.any():
            raise RuntimeError(
                "selected analytic curves have no positive variance; "
                "joint simultaneous bands are unavailable"
            )
        standardized_root = (
            selected_roots[positive]
            / selected_standard_errors[positive].unsqueeze(1)
        )
        maximum_statistics = torch.empty(
            simulations,
            dtype=selected_roots.dtype,
            device=selected_roots.device,
        )
        batch_size = 1_024
        for start in range(0, simulations, batch_size):
            stop = min(start + batch_size, simulations)
            normal_draws = torch.randn(
                (stop - start, selected_roots.shape[1]),
                dtype=selected_roots.dtype,
                device=selected_roots.device,
                generator=generator,
            )
            maximum_statistics[start:stop] = (
                normal_draws @ standardized_root.mT
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
                simulations=simulations,
                method="analytic_joint_gaussian_max_t",
            )
        return SmoothJointInferenceBand(
            bands=bands,
            term_order=selected_terms,
            critical_value=critical_value,
            confidence_level=self.confidence_level,
            simulations=simulations,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all pointwise analytic curve summaries in long format."""
        frames = []
        for parameter, term in self.term_order:
            frame = self.curves[parameter][term].to_dataframe()
            frame.insert(0, "term", term)
            frame.insert(0, "parameter", parameter)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _curve(self, key: tuple[str, str]) -> SmoothInferenceResult:
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
    def smoothing_parameter_slices(self) -> dict[tuple[str, str], slice]:
        """Return slices locating each term in penalty-level results."""
        result = {}
        start = 0
        for key in self.term_order:
            stop = start + self._curve(key).smoothing_parameter_count
            result[key] = slice(start, stop)
            start = stop
        return result

    @property
    def smoothing_parameter_labels(self) -> tuple[tuple[str, str, int], ...]:
        """Return parameter, term, and penalty-index labels for lambdas."""
        return tuple(
            (parameter, term, penalty_index)
            for parameter, term in self.term_order
            for penalty_index in range(
                self.curves[parameter][term].smoothing_parameter_count
            )
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
        """Return fitted lambdas flattened by term and penalty index."""
        reference = self._curve(self.term_order[0]).bootstrap_smoothing_parameters
        return torch.cat(
            [
                _smoothing_parameter_tensor(
                    self._curve(key).smoothing_parameter,
                    reference=reference,
                )
                for key in self.term_order
            ]
        )

    @property
    def bootstrap_smoothing_parameters(self) -> Tensor:
        """Return aligned lambdas as replicates by flattened penalties."""
        return torch.cat(
            [
                _bootstrap_smoothing_parameter_matrix(self._curve(key))
                for key in self.term_order
            ],
            dim=1,
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
            training_basis = term.design(training_covariate)
            evaluation_basis = term.predict_design(evaluation_covariate)
            penalties = term.penalty_matrices()
            if len(penalties) == 1:
                penalty = term.penalty_matrix()
                system = training_basis.mT @ (
                    combined_weights.unsqueeze(-1) * training_basis
                ) + term.smoothing_parameter * (penalty.mT @ penalty)
                coefficient_covariance = torch.linalg.pinv(
                    system,
                    hermitian=True,
                )

                nullity = term.penalty_nullity
                if nullity:
                    powers = torch.arange(
                        nullity,
                        dtype=training_covariate.dtype,
                        device=training_covariate.device,
                    )
                    training_null_basis = training_covariate.unsqueeze(-1).pow(
                        powers
                    )
                    null_system = training_null_basis.mT @ (
                        combined_weights.unsqueeze(-1) * training_null_basis
                    )
                    null_inverse = torch.linalg.pinv(
                        null_system,
                        hermitian=True,
                    )
                    null_basis_coefficients = torch.linalg.lstsq(
                        training_basis,
                        training_null_basis,
                    ).solution
                    coefficient_covariance = coefficient_covariance - (
                        null_basis_coefficients
                        @ null_inverse
                        @ null_basis_coefficients.mT
                    )
            else:
                transform = _term_constraint_transform(
                    term,
                    training_covariate,
                    context=f"constraints for {parameter!r}.{term_name}",
                )
                reduced_basis = training_basis @ transform
                combined_penalty = _combined_term_penalty(term)
                reduced_system = reduced_basis.mT @ (
                    combined_weights.unsqueeze(-1) * reduced_basis
                ) + transform.mT @ combined_penalty @ transform
                reduced_covariance = torch.linalg.pinv(
                    reduced_system,
                    hermitian=True,
                )
                coefficient_covariance = (
                    transform @ reduced_covariance @ transform.mT
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
                smoothing_parameter=_term_smoothing_parameter_value(term),
                effective_degrees_of_freedom=effective_degrees_of_freedom,
                confidence_level=confidence_level,
            )

    return results


def smooth_joint_inference(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    evaluation_smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    confidence_level: float = 0.95,
) -> SmoothJointInferenceResult:
    """Infer all smooth curves from one fixed-lambda penalized information matrix.

    The expected information contains every distribution parameter, linear
    coefficient, and spline coefficient. Each spline is constrained to be
    orthogonal to its unpenalized null-function space under the parameter's
    working weights. This removes the same linear component subtracted by
    ``gamlss.pb()`` while retaining cross-term and cross-parameter covariance.
    """
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    if not any(
        model.smooth_terms[parameter] for parameter in model.family.parameter_names
    ):
        raise ValueError("joint smooth inference requires at least one smooth term")

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
    link_derivatives = {
        parameter: model.family.links[parameter]
        .inverse_derivative(predictors[parameter])
        .reciprocal()
        for parameter in model.family.parameter_names
    }
    diagonal_information = {
        parameter: -second_derivatives[(parameter, parameter)]
        / link_derivatives[parameter].square()
        for parameter in model.family.parameter_names
    }
    if any(
        not information.isfinite().all() or (information <= 0).any()
        for information in diagonal_information.values()
    ):
        raise RuntimeError(
            "joint smooth diagonal working information must be finite and positive"
        )

    evaluation_covariates = (
        smooth_covariates
        if evaluation_smooth_covariates is None
        else evaluation_smooth_covariates
    )
    _validate_smooth_evaluation_mapping(model, evaluation_covariates)

    full_transform_blocks: list[Tensor] = []
    parameter_designs: dict[str, Tensor] = {}
    parameter_reduced_slices: dict[str, slice] = {}
    linear_full_slices: dict[str, slice] = {}
    smooth_full_slices: dict[tuple[str, str], slice] = {}
    smooth_reduced_slices: dict[tuple[str, str], slice] = {}
    smooth_transforms: dict[tuple[str, str], Tensor] = {}
    full_estimates: list[Tensor] = []
    coefficient_names: list[str] = []
    full_start = 0
    reduced_start = 0

    for parameter in model.family.parameter_names:
        design = design_matrices[parameter]
        parameter_parts = [design]
        linear_count = design.shape[1]
        linear_full_slices[parameter] = slice(
            full_start,
            full_start + linear_count,
        )
        full_start += linear_count
        full_transform_blocks.append(
            torch.eye(
                linear_count,
                dtype=response.dtype,
                device=response.device,
            )
        )
        full_estimates.append(model.coefficients[parameter].detach())
        coefficient_names.extend(_linear_coefficient_names(model, parameter))
        parameter_reduced_start = reduced_start
        reduced_start += linear_count

        combined_weights = (
            diagonal_information[parameter] * case_weights
        ).detach()
        for term_name, term in model.smooth_terms[parameter].items():
            key = (parameter, term_name)
            training_covariate = smooth_covariates[parameter][term_name].detach()
            basis = term.design(training_covariate)
            penalties = term.penalty_matrices()
            coefficient_count = term.coefficients.numel()
            if len(penalties) == 1:
                penalty = term.penalty_matrix()
                nullity = term.penalty_nullity
                if (
                    penalty.ndim != 2
                    or penalty.shape[1] != coefficient_count
                    or nullity < 0
                ):
                    raise RuntimeError(
                        f"invalid penalty structure for "
                        f"{parameter!r}.{term_name}"
                    )
                penalty_null_space = _right_null_space(
                    penalty,
                    expected_nullity=nullity,
                    context=f"penalty for {parameter!r}.{term_name}",
                )
                if nullity:
                    null_functions = basis @ penalty_null_space
                    constraints = null_functions.mT @ (
                        combined_weights.unsqueeze(-1) * basis
                    )
                    transform = _right_null_space(
                        constraints,
                        expected_nullity=coefficient_count - nullity,
                        context=f"identifiability constraints for "
                        f"{parameter!r}.{term_name}",
                    )
                else:
                    transform = torch.eye(
                        coefficient_count,
                        dtype=response.dtype,
                        device=response.device,
                    )
            else:
                transform = _term_constraint_transform(
                    term,
                    training_covariate,
                    context=f"constraints for {parameter!r}.{term_name}",
                )

            reduced_count = transform.shape[1]
            full_transform_blocks.append(transform)
            parameter_parts.append(basis @ transform)
            smooth_transforms[key] = transform
            smooth_full_slices[key] = slice(
                full_start,
                full_start + coefficient_count,
            )
            smooth_reduced_slices[key] = slice(
                reduced_start,
                reduced_start + reduced_count,
            )
            full_start += coefficient_count
            reduced_start += reduced_count
            full_estimates.append(term.coefficients.detach())
            coefficient_names.extend(
                f"{parameter}.{term_name}[{index}]"
                for index in range(coefficient_count)
            )

        parameter_designs[parameter] = torch.cat(parameter_parts, dim=1)
        parameter_reduced_slices[parameter] = slice(
            parameter_reduced_start,
            reduced_start,
        )

    coefficient_transform = torch.block_diag(*full_transform_blocks)
    reduced_count = coefficient_transform.shape[1]
    information = torch.zeros(
        (reduced_count, reduced_count),
        dtype=response.dtype,
        device=response.device,
    )
    parameter_names = model.family.parameter_names
    for left_index, left in enumerate(parameter_names):
        left_design = parameter_designs[left]
        left_slice = parameter_reduced_slices[left]
        for right in parameter_names[left_index:]:
            right_design = parameter_designs[right]
            right_slice = parameter_reduced_slices[right]
            if left == right:
                pair_information = diagonal_information[left]
            else:
                cross_second = _expected_cross_derivative(
                    second_derivatives,
                    left,
                    right,
                )
                pair_information = -cross_second / (
                    link_derivatives[left] * link_derivatives[right]
                )
            combined_information = pair_information * case_weights
            if not combined_information.isfinite().all():
                raise RuntimeError(
                    f"joint smooth information for {left!r}, {right!r} "
                    "must be finite"
                )
            block = left_design.mT @ (
                combined_information.unsqueeze(-1) * right_design
            )
            information[left_slice, right_slice] = block
            if left != right:
                information[right_slice, left_slice] = block.mT

    for key, reduced_slice in smooth_reduced_slices.items():
        parameter, term_name = key
        term = model.smooth_terms[parameter][term_name]
        transform = smooth_transforms[key]
        combined_penalty = _combined_term_penalty(term)
        information[reduced_slice, reduced_slice] += (
            transform.mT @ combined_penalty @ transform
        )

    information = (information + information.mT) / 2.0
    if not torch.isfinite(information).all():
        raise RuntimeError("joint penalized information matrix is not finite")
    factor, info = torch.linalg.cholesky_ex(information)
    if int(info) != 0:
        raise RuntimeError(
            "joint penalized information matrix is not positive definite; "
            "covariance is unavailable"
        )
    identity = torch.eye(
        reduced_count,
        dtype=response.dtype,
        device=response.device,
    )
    reduced_covariance_root = torch.linalg.solve_triangular(
        factor.mT,
        identity,
        upper=True,
    )
    coefficient_covariance_root = (
        coefficient_transform @ reduced_covariance_root
    )

    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    curves: dict[str, dict[str, SmoothInferenceResult]] = {
        parameter: {} for parameter in parameter_names
    }
    term_order: list[tuple[str, str]] = []
    for parameter in parameter_names:
        combined_weights = (
            diagonal_information[parameter] * case_weights
        ).detach()
        for term_name, term in model.smooth_terms[parameter].items():
            key = (parameter, term_name)
            term_order.append(key)
            evaluation_covariate = evaluation_covariates[parameter][
                term_name
            ].detach()
            evaluation_basis = term.predict_design(evaluation_covariate)
            curve_root = (
                evaluation_basis
                @ coefficient_covariance_root[smooth_full_slices[key]]
            )
            standard_errors = curve_root.square().sum(dim=1).sqrt()
            estimates = term(evaluation_covariate).detach()
            confidence_intervals = torch.column_stack(
                (
                    estimates - critical_value * standard_errors,
                    estimates + critical_value * standard_errors,
                )
            )
            effective_degrees_of_freedom = float(
                term.effective_degrees_of_freedom(
                    smooth_covariates[parameter][term_name],
                    combined_weights,
                )
            )
            curves[parameter][term_name] = SmoothInferenceResult(
                parameter=parameter,
                term=term_name,
                covariate=evaluation_covariate.clone(),
                estimates=estimates,
                _covariance_root=curve_root.detach(),
                standard_errors=standard_errors.detach(),
                confidence_intervals=confidence_intervals.detach(),
                smoothing_parameter=_term_smoothing_parameter_value(term),
                effective_degrees_of_freedom=effective_degrees_of_freedom,
                confidence_level=confidence_level,
            )

    return SmoothJointInferenceResult(
        curves=curves,
        term_order=tuple(term_order),
        coefficient_names=tuple(coefficient_names),
        linear_coefficient_slices=linear_full_slices,
        smooth_coefficient_slices=smooth_full_slices,
        coefficient_estimates=torch.cat(full_estimates).clone(),
        _coefficient_covariance_root=coefficient_covariance_root.detach(),
        confidence_level=confidence_level,
    )


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
    original_smoothing_parameters: dict[
        str,
        dict[str, SmoothingParameterValue],
    ] = {}
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
            smoothing_parameter = _term_smoothing_parameter_value(term)
            original_smoothing_parameters[parameter][
                term_name
            ] = smoothing_parameter
            bootstrap_estimates[parameter][term_name] = torch.empty(
                (replicates, estimate.numel()),
                dtype=estimate.dtype,
                device=estimate.device,
            )
            smoothing_parameter_count = len(term.smoothing_parameters)
            smoothing_parameter_shape = (
                (replicates,)
                if smoothing_parameter_count == 1
                else (replicates, smoothing_parameter_count)
            )
            bootstrap_smoothing_parameters[parameter][term_name] = torch.empty(
                smoothing_parameter_shape,
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
                    smoothing_parameter = _term_smoothing_parameter_value(term)
                    target = bootstrap_smoothing_parameters[parameter][
                        term_name
                    ][successful_replicates]
                    if isinstance(smoothing_parameter, tuple):
                        target.copy_(
                            torch.tensor(
                                smoothing_parameter,
                                dtype=target.dtype,
                                device=target.device,
                            )
                        )
                    else:
                        target.copy_(target.new_tensor(smoothing_parameter))
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


def _term_smoothing_parameter_value(
    term: SmoothTerm,
) -> SmoothingParameterValue:
    values = term.smoothing_parameters
    return values[0] if len(values) == 1 else tuple(values)


def _smoothing_parameter_tensor(
    value: SmoothingParameterValue,
    *,
    reference: Tensor,
) -> Tensor:
    values = value if isinstance(value, tuple) else (value,)
    return torch.tensor(
        values,
        dtype=reference.dtype,
        device=reference.device,
    )


def _bootstrap_smoothing_parameter_matrix(
    result: SmoothBootstrapResult,
) -> Tensor:
    parameters = result.bootstrap_smoothing_parameters
    if parameters.ndim == 1:
        return parameters.unsqueeze(1)
    if (
        parameters.ndim == 2
        and parameters.shape[1] == result.smoothing_parameter_count
    ):
        return parameters
    raise RuntimeError(
        "bootstrap smoothing parameters must have shape (replicates,) "
        "or (replicates, penalties)"
    )


def _combined_term_penalty(term: SmoothTerm) -> Tensor:
    penalties = term.penalty_matrices()
    smoothing_parameters = term.smoothing_parameters
    if len(penalties) != len(smoothing_parameters):
        raise RuntimeError(
            "smooth penalties and smoothing parameters have different lengths"
        )
    return sum(
        (
            smoothing_parameter * penalty
            for smoothing_parameter, penalty in zip(
                smoothing_parameters,
                penalties,
                strict=True,
            )
        ),
        torch.zeros_like(penalties[0]),
    )


def _term_constraint_transform(
    term: SmoothTerm,
    covariate: Tensor,
    *,
    context: str,
) -> Tensor:
    coefficient_count = term.coefficients.numel()
    constraints = term.constraints(covariate)
    if (
        constraints.ndim != 2
        or constraints.shape[1] != coefficient_count
        or constraints.dtype != term.coefficients.dtype
        or constraints.device != term.coefficients.device
        or not torch.isfinite(constraints).all()
    ):
        raise RuntimeError(
            f"{context} must be a finite matrix with one column per coefficient"
        )
    if constraints.shape[0] == 0:
        return torch.eye(
            coefficient_count,
            dtype=term.coefficients.dtype,
            device=term.coefficients.device,
        )
    _, singular_values, right_vectors = torch.linalg.svd(
        constraints,
        full_matrices=True,
    )
    tolerance = (
        max(constraints.shape)
        * torch.finfo(constraints.dtype).eps
        * singular_values.max()
    )
    rank = int((singular_values > tolerance).sum())
    if rank >= coefficient_count:
        raise RuntimeError(f"{context} leave no coefficient free")
    return right_vectors.mT[:, rank:]


def _covariate_frame_values(
    covariate: Tensor,
) -> tuple[Tensor, tuple[str, ...]]:
    if covariate.ndim == 1:
        return covariate.unsqueeze(-1), ("covariate",)
    if covariate.ndim == 2 and covariate.shape[1] > 0:
        return (
            covariate,
            tuple(
                f"covariate_{index}" for index in range(covariate.shape[1])
            ),
        )
    raise RuntimeError(
        "smooth inference covariates must be a vector or non-empty matrix"
    )


def _right_null_space(
    matrix: Tensor,
    *,
    expected_nullity: int,
    context: str,
) -> Tensor:
    if matrix.ndim != 2:
        raise RuntimeError(f"{context} must be a matrix")
    _, singular_values, right_vectors = torch.linalg.svd(
        matrix,
        full_matrices=True,
    )
    if singular_values.numel():
        tolerance = (
            max(matrix.shape)
            * torch.finfo(matrix.dtype).eps
            * singular_values.max()
        )
        rank = int((singular_values > tolerance).sum())
    else:
        rank = 0
    nullity = matrix.shape[1] - rank
    if nullity != expected_nullity:
        raise RuntimeError(
            f"{context} has nullity {nullity}, expected {expected_nullity}"
        )
    return right_vectors.mT[:, rank:]


def _expected_cross_derivative(
    derivatives: Mapping[tuple[str, str], Tensor],
    left: str,
    right: str,
) -> Tensor:
    if (left, right) in derivatives:
        return derivatives[(left, right)]
    if (right, left) in derivatives:
        return derivatives[(right, left)]
    raise NotImplementedError(
        f"family does not provide expected cross information for "
        f"{left!r}, {right!r}"
    )


def _linear_coefficient_names(
    model: GAMLSS,
    parameter: str,
) -> tuple[str, ...]:
    if model._formula_encoder is None:
        return tuple(
            f"{parameter}[{index}]"
            for index in range(model.coefficients[parameter].numel())
        )
    return tuple(
        f"{parameter}.{column}"
        for column in model.formula_column_names[parameter]
    )


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
