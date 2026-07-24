"""Wald, conditional smooth-curve, and parametric-bootstrap inference."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
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
