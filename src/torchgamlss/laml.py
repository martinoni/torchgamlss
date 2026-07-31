"""Laplace approximate marginal likelihood for smooth GAMLSS models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torchgamlss.families import Family
    from torchgamlss.model import GAMLSS


@dataclass(frozen=True)
class LAMLControl:
    """Numerical controls for the experimental nested LAML optimizer."""

    inner_max_iterations: int = 100
    outer_max_iterations: int = 30
    inner_gradient_tolerance: float = 1e-7
    inner_relaxed_gradient_multiplier: float = 50.0
    inner_step_tolerance: float = 1e-11
    outer_gradient_tolerance: float = 5e-5
    outer_step_tolerance: float = 1e-7
    finite_difference_step: float = 2e-4
    hessian_difference_step: float = 2e-3
    outer_derivative_method: Literal[
        "implicit",
        "finite_difference",
    ] = "implicit"
    log_smoothing_parameter_bounds: tuple[float, float] = (-20.0, 20.0)
    max_line_search_steps: int = 25

    def __post_init__(self) -> None:
        if self.inner_max_iterations < 1:
            raise ValueError("inner_max_iterations must be at least 1")
        if self.outer_max_iterations < 1:
            raise ValueError("outer_max_iterations must be at least 1")
        for name in (
            "inner_gradient_tolerance",
            "inner_step_tolerance",
            "outer_gradient_tolerance",
            "outer_step_tolerance",
            "finite_difference_step",
            "hessian_difference_step",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.inner_relaxed_gradient_multiplier)
            or self.inner_relaxed_gradient_multiplier < 1.0
        ):
            raise ValueError(
                "inner_relaxed_gradient_multiplier must be finite and at least one"
            )
        lower, upper = self.log_smoothing_parameter_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                "log_smoothing_parameter_bounds must be finite and increasing"
            )
        if self.max_line_search_steps < 1:
            raise ValueError("max_line_search_steps must be at least 1")
        if self.outer_derivative_method not in {
            "implicit",
            "finite_difference",
        }:
            raise ValueError(
                "outer_derivative_method must be 'implicit' or 'finite_difference'"
            )


@dataclass(frozen=True)
class LAMLHistoryEntry:
    """One accepted outer smoothing-parameter iterate."""

    iteration: int
    objective: float
    log_smoothing_parameters: Tensor
    smoothing_parameters: Tensor
    gradient: Tensor
    projected_gradient_max: float
    step_norm: float
    inner_iterations: int
    inner_gradient_max: float


@dataclass(frozen=True)
class NormalLAMLResult:
    """Fitted Normal location-scale model and nested LAML diagnostics."""

    coefficients: Tensor
    mu_coefficients: Tensor
    sigma_coefficients: Tensor
    linear_predictor_mu: Tensor
    linear_predictor_sigma: Tensor
    fitted_mu: Tensor
    fitted_sigma: Tensor
    log_smoothing_parameters: Tensor
    smoothing_parameters: Tensor
    estimated_smoothing_parameters: tuple[bool, ...]
    objective: Tensor
    log_likelihood: Tensor
    penalized_negative_log_likelihood: Tensor
    outer_gradient: Tensor
    outer_hessian: Tensor
    outer_hessian_condition_number: Tensor
    outer_derivative_method: str
    profile_evaluations: int
    boundary_status: tuple[str, ...]
    outer_converged: bool
    outer_iterations: int
    inner_converged: bool
    inner_iterations: int
    inner_gradient_max: float
    history: tuple[LAMLHistoryEntry, ...]
    coefficient_transform: Tensor
    reduced_observed_information: Tensor
    reduced_penalized_information: Tensor
    combined_penalty_matrix: Tensor
    reduced_combined_penalty_matrix: Tensor
    generalized_log_determinant_penalty: Tensor
    log_determinant_penalized_information: Tensor
    penalty_ranks: tuple[int, ...]
    combined_penalty_rank: int
    unpenalized_dimension: int
    constraint_rank: int
    effective_degrees_of_freedom: Tensor
    penalty_degrees_of_freedom: Tensor
    linear_coefficient_slices: Mapping[str, slice] = field(default_factory=dict)
    smooth_coefficient_slices: Mapping[tuple[str, str], slice] = field(
        default_factory=dict
    )
    smoothing_parameter_labels: tuple[tuple[str, str, int], ...] = ()
    smoothing_parameter_slices: Mapping[tuple[str, str], slice] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class GAMLSSLAMLResult:
    """Fitted multi-parameter family model and nested LAML diagnostics."""

    family: str
    parameter_names: tuple[str, ...]
    coefficients: Tensor
    parameter_coefficients: Mapping[str, Tensor]
    coefficient_slices: Mapping[str, slice]
    linear_predictors: Mapping[str, Tensor]
    fitted_parameters: Mapping[str, Tensor]
    log_smoothing_parameters: Tensor
    smoothing_parameters: Tensor
    estimated_smoothing_parameters: tuple[bool, ...]
    objective: Tensor
    log_likelihood: Tensor
    penalized_negative_log_likelihood: Tensor
    outer_gradient: Tensor
    outer_hessian: Tensor
    outer_hessian_condition_number: Tensor
    outer_derivative_method: str
    profile_evaluations: int
    boundary_status: tuple[str, ...]
    outer_converged: bool
    outer_iterations: int
    inner_converged: bool
    inner_iterations: int
    inner_gradient_max: float
    history: tuple[LAMLHistoryEntry, ...]
    coefficient_transform: Tensor
    reduced_observed_information: Tensor
    reduced_penalized_information: Tensor
    combined_penalty_matrix: Tensor
    reduced_combined_penalty_matrix: Tensor
    generalized_log_determinant_penalty: Tensor
    log_determinant_penalized_information: Tensor
    penalty_ranks: tuple[int, ...]
    combined_penalty_rank: int
    unpenalized_dimension: int
    constraint_rank: int
    effective_degrees_of_freedom: Tensor
    penalty_degrees_of_freedom: Tensor
    linear_coefficient_slices: Mapping[str, slice] = field(default_factory=dict)
    smooth_coefficient_slices: Mapping[tuple[str, str], slice] = field(
        default_factory=dict
    )
    smoothing_parameter_labels: tuple[tuple[str, str, int], ...] = ()
    smoothing_parameter_slices: Mapping[tuple[str, str], slice] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _ProfileEvaluation:
    objective: Tensor
    reduced_coefficients: Tensor
    coefficients: Tensor
    linear_predictor_mu: Tensor
    linear_predictor_sigma: Tensor
    fitted_mu: Tensor
    fitted_sigma: Tensor
    log_likelihood: Tensor
    penalized_negative_log_likelihood: Tensor
    observed_information: Tensor
    penalized_information: Tensor
    combined_penalty: Tensor
    reduced_combined_penalty: Tensor
    log_determinant_penalty: Tensor
    log_determinant_information: Tensor
    combined_penalty_rank: int
    unpenalized_dimension: int
    effective_degrees_of_freedom: Tensor
    penalty_degrees_of_freedom: Tensor
    inner_iterations: int
    inner_gradient_max: float


@dataclass(frozen=True)
class _GAMLSSProfileEvaluation:
    objective: Tensor
    reduced_coefficients: Tensor
    coefficients: Tensor
    parameter_coefficients: Mapping[str, Tensor]
    linear_predictors: Mapping[str, Tensor]
    fitted_parameters: Mapping[str, Tensor]
    log_likelihood: Tensor
    penalized_negative_log_likelihood: Tensor
    observed_information: Tensor
    penalized_information: Tensor
    combined_penalty: Tensor
    reduced_combined_penalty: Tensor
    log_determinant_penalty: Tensor
    log_determinant_information: Tensor
    combined_penalty_rank: int
    unpenalized_dimension: int
    effective_degrees_of_freedom: Tensor
    penalty_degrees_of_freedom: Tensor
    inner_iterations: int
    inner_gradient_max: float


@dataclass(frozen=True)
class _OuterOptimization:
    profile: _ProfileEvaluation | _GAMLSSProfileEvaluation
    log_smoothing_parameters: Tensor
    smoothing_parameters: Tensor
    gradient: Tensor
    hessian: Tensor
    hessian_condition_number: Tensor
    derivative_method: str
    profile_evaluations: int
    boundary_status: tuple[str, ...]
    converged: bool
    iterations: int
    history: tuple[LAMLHistoryEntry, ...]


class _NormalProfileEvaluator:
    def __init__(
        self,
        response: Tensor,
        mu_design: Tensor,
        sigma_design: Tensor,
        penalty_matrices: tuple[Tensor, ...],
        initial_smoothing_parameters: Tensor,
        estimated: tuple[bool, ...],
        weights: Tensor,
        mu_offset: Tensor,
        sigma_offset: Tensor,
        transform: Tensor,
        initial_reduced_coefficients: Tensor,
        sigma_floor: float,
        control: LAMLControl,
    ) -> None:
        self.response = response
        self.mu_design = mu_design
        self.sigma_design = sigma_design
        self.penalty_matrices = penalty_matrices
        self.initial_smoothing_parameters = initial_smoothing_parameters
        self.estimated = estimated
        self.weights = weights
        self.mu_offset = mu_offset
        self.sigma_offset = sigma_offset
        self.transform = transform
        self.initial_reduced_coefficients = initial_reduced_coefficients
        self.sigma_floor = sigma_floor
        self.control = control
        self.mu_count = mu_design.shape[1]
        self.free_indices = tuple(index for index, free in enumerate(estimated) if free)
        self.fixed_log_parameters = initial_smoothing_parameters.log()
        self.cache: dict[tuple[float, ...], _ProfileEvaluation] = {}

    def full_log_parameters(self, free_log_parameters: Tensor) -> Tensor:
        full = self.fixed_log_parameters.clone()
        if self.free_indices:
            indices = torch.tensor(
                self.free_indices,
                dtype=torch.long,
                device=free_log_parameters.device,
            )
            full[indices] = free_log_parameters
        return full

    def evaluate(
        self,
        free_log_parameters: Tensor,
        *,
        start: Tensor | None = None,
    ) -> _ProfileEvaluation:
        key = tuple(float(value) for value in free_log_parameters.detach().cpu())
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if start is None:
            start = self._nearest_start(free_log_parameters)
        full_log_parameters = self.full_log_parameters(free_log_parameters)
        smoothing_parameters = full_log_parameters.exp()
        combined_penalty = sum(
            (
                smoothing_parameter * penalty
                for smoothing_parameter, penalty in zip(
                    smoothing_parameters,
                    self.penalty_matrices,
                    strict=True,
                )
            ),
            torch.zeros_like(self.penalty_matrices[0]),
        )
        combined_penalty = _symmetrize(combined_penalty)
        reduced_penalty = _symmetrize(
            self.transform.mT @ combined_penalty @ self.transform
        )
        reduced_components = tuple(
            _symmetrize(self.transform.mT @ penalty @ self.transform)
            for penalty in self.penalty_matrices
        )

        coefficients, iterations, gradient_max = self._fit_inner(
            reduced_penalty,
            start,
        )
        coefficient_value = coefficients.detach().requires_grad_(True)

        def likelihood_objective(value: Tensor) -> Tensor:
            return self._negative_log_likelihood(value)

        negative_log_likelihood = likelihood_objective(coefficient_value)
        observed_information = torch.autograd.functional.hessian(
            likelihood_objective,
            coefficient_value,
        )
        observed_information = _symmetrize(observed_information).detach()
        penalized_information = _symmetrize(observed_information + reduced_penalty)
        information_eigenvalues = torch.linalg.eigvalsh(penalized_information)
        information_tolerance = _matrix_tolerance(penalized_information)
        if float(information_eigenvalues.detach().min()) <= information_tolerance:
            raise RuntimeError("inner penalized information is not positive definite")
        log_determinant_information = information_eigenvalues.log().sum()
        (
            log_determinant_penalty,
            combined_rank,
        ) = _generalized_log_determinant(reduced_penalty)
        unpenalized_dimension = reduced_penalty.shape[0] - combined_rank

        penalty_value = 0.5 * (
            coefficient_value.detach() @ reduced_penalty @ coefficient_value.detach()
        )
        penalized_negative_log_likelihood = (
            negative_log_likelihood.detach() + penalty_value
        )
        objective = (
            penalized_negative_log_likelihood
            - 0.5 * log_determinant_penalty
            + 0.5 * log_determinant_information
            - 0.5 * unpenalized_dimension * math.log(2.0 * math.pi)
        )
        information_inverse = torch.linalg.inv(penalized_information)
        effective_degrees_of_freedom = torch.trace(
            information_inverse @ observed_information
        )
        penalty_degrees_of_freedom = torch.stack(
            tuple(
                torch.trace(information_inverse @ (smoothing_parameter * component))
                for smoothing_parameter, component in zip(
                    smoothing_parameters,
                    reduced_components,
                    strict=True,
                )
            )
        )

        full_coefficients = self.transform @ coefficient_value.detach()
        mu_coefficients = full_coefficients[: self.mu_count]
        sigma_coefficients = full_coefficients[self.mu_count :]
        eta_mu = self.mu_design @ mu_coefficients + self.mu_offset
        eta_sigma = self.sigma_design @ sigma_coefficients + self.sigma_offset
        fitted_sigma = self.sigma_floor + eta_sigma.exp()
        profile = _ProfileEvaluation(
            objective=objective.detach(),
            reduced_coefficients=coefficient_value.detach(),
            coefficients=full_coefficients.detach(),
            linear_predictor_mu=eta_mu.detach(),
            linear_predictor_sigma=eta_sigma.detach(),
            fitted_mu=eta_mu.detach(),
            fitted_sigma=fitted_sigma.detach(),
            log_likelihood=(-negative_log_likelihood).detach(),
            penalized_negative_log_likelihood=(
                penalized_negative_log_likelihood.detach()
            ),
            observed_information=observed_information,
            penalized_information=penalized_information.detach(),
            combined_penalty=combined_penalty.detach(),
            reduced_combined_penalty=reduced_penalty.detach(),
            log_determinant_penalty=log_determinant_penalty.detach(),
            log_determinant_information=log_determinant_information.detach(),
            combined_penalty_rank=combined_rank,
            unpenalized_dimension=unpenalized_dimension,
            effective_degrees_of_freedom=(effective_degrees_of_freedom.detach()),
            penalty_degrees_of_freedom=penalty_degrees_of_freedom.detach(),
            inner_iterations=iterations,
            inner_gradient_max=gradient_max,
        )
        self.cache[key] = profile
        return profile

    def _nearest_start(self, free_log_parameters: Tensor) -> Tensor:
        if not self.cache:
            return self.initial_reduced_coefficients
        nearest_key = min(
            self.cache,
            key=lambda key: sum(
                (float(value) - key_value) ** 2
                for value, key_value in zip(
                    free_log_parameters.detach().cpu(),
                    key,
                    strict=True,
                )
            ),
        )
        return self.cache[nearest_key].reduced_coefficients

    def _negative_log_likelihood(self, reduced_coefficients: Tensor) -> Tensor:
        full_coefficients = self.transform @ reduced_coefficients
        mu_coefficients = full_coefficients[: self.mu_count]
        sigma_coefficients = full_coefficients[self.mu_count :]
        eta_mu = self.mu_design @ mu_coefficients + self.mu_offset
        eta_sigma = self.sigma_design @ sigma_coefficients + self.sigma_offset
        sigma = self.sigma_floor + eta_sigma.exp()
        standardized = (self.response - eta_mu) / sigma
        losses = (
            sigma.log() + 0.5 * standardized.square() + 0.5 * math.log(2.0 * math.pi)
        )
        return (self.weights * losses).sum()

    def _fit_inner(
        self,
        reduced_penalty: Tensor,
        start: Tensor,
    ) -> tuple[Tensor, int, float]:
        coefficients = start.detach().clone()
        converged = False
        gradient_max = math.inf
        last_iteration = 0

        def objective(value: Tensor) -> Tensor:
            return self._negative_log_likelihood(value) + 0.5 * (
                value @ reduced_penalty @ value
            )

        for iteration in range(1, self.control.inner_max_iterations + 1):
            last_iteration = iteration
            current = coefficients.detach().requires_grad_(True)
            value = objective(current)
            if not bool(torch.isfinite(value).detach()):
                raise RuntimeError("inner penalized objective is not finite")
            gradient = torch.autograd.grad(value, current)[0].detach()
            gradient_max = float(gradient.abs().max().detach())
            threshold = self.control.inner_gradient_tolerance * (
                1.0 + float(current.detach().abs().max())
            )
            if gradient_max <= threshold:
                coefficients = current.detach()
                converged = True
                break

            hessian = torch.autograd.functional.hessian(objective, current)
            hessian = _symmetrize(hessian).detach()
            if not bool(torch.isfinite(hessian).all().detach()):
                raise RuntimeError("inner penalized Hessian is not finite")
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            curvature_floor = max(
                math.sqrt(torch.finfo(hessian.dtype).eps)
                * max(float(eigenvalues.detach().abs().max()), 1.0),
                torch.finfo(hessian.dtype).eps,
            )
            inverse_curvature = eigenvalues.clamp_min(curvature_floor).reciprocal()
            step = -(eigenvectors @ (inverse_curvature * (eigenvectors.mT @ gradient)))
            maximum_step = 5.0 * (1.0 + float(current.detach().norm()))
            step_norm = float(step.detach().norm())
            if step_norm > maximum_step:
                step = step * (maximum_step / step_norm)

            directional_derivative = float((gradient @ step).detach())
            if directional_derivative >= 0:
                step = -gradient / max(float(gradient.norm()), 1.0)
                directional_derivative = float((gradient @ step).detach())
            accepted = False
            scale = 1.0
            current_value = float(value.detach())
            candidate = current.detach()
            for _ in range(self.control.max_line_search_steps):
                trial = current.detach() + scale * step
                trial_value = objective(trial)
                sufficient_decrease = (
                    current_value + 1e-4 * scale * directional_derivative
                )
                if (
                    bool(torch.isfinite(trial_value).detach())
                    and float(trial_value.detach()) <= sufficient_decrease
                ):
                    candidate = trial.detach()
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                if (
                    gradient_max
                    <= self.control.inner_relaxed_gradient_multiplier * threshold
                ):
                    coefficients = current.detach()
                    converged = True
                    break
                raise RuntimeError(
                    "inner Newton line search failed to reduce the objective; "
                    f"gradient max={gradient_max:.6g}"
                )
            change = float((candidate - current.detach()).norm())
            coefficients = candidate
            if change <= self.control.inner_step_tolerance * (
                1.0 + float(coefficients.norm())
            ):
                final = coefficients.detach().requires_grad_(True)
                final_value = objective(final)
                final_gradient = torch.autograd.grad(final_value, final)[0]
                gradient_max = float(final_gradient.detach().abs().max())
                final_threshold = self.control.inner_gradient_tolerance * (
                    1.0 + float(final.detach().abs().max())
                )
                converged = gradient_max <= final_threshold
                break

        relaxed_threshold = (
            self.control.inner_relaxed_gradient_multiplier
            * self.control.inner_gradient_tolerance
            * (1.0 + float(coefficients.detach().abs().max()))
        )
        if not converged and gradient_max <= relaxed_threshold:
            converged = True
        if not converged:
            raise RuntimeError(
                "inner penalized coefficient fit did not converge; "
                f"gradient max={gradient_max:.6g}"
            )
        return coefficients, last_iteration, gradient_max


class _GAMLSSProfileEvaluator(_NormalProfileEvaluator):
    """Evaluate a profiled LAML criterion through the public Family contract."""

    def __init__(
        self,
        family: Family,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        coefficient_slices: Mapping[str, slice],
        penalty_matrices: tuple[Tensor, ...],
        initial_smoothing_parameters: Tensor,
        estimated: tuple[bool, ...],
        weights: Tensor,
        offsets: Mapping[str, Tensor],
        transform: Tensor,
        initial_reduced_coefficients: Tensor,
        control: LAMLControl,
    ) -> None:
        self.family = family
        self.response = response
        self.design_matrices = dict(design_matrices)
        self.coefficient_slices = dict(coefficient_slices)
        self.penalty_matrices = penalty_matrices
        self.initial_smoothing_parameters = initial_smoothing_parameters
        self.estimated = estimated
        self.weights = weights
        self.offsets = dict(offsets)
        self.transform = transform
        self.initial_reduced_coefficients = initial_reduced_coefficients
        self.control = control
        self.free_indices = tuple(index for index, free in enumerate(estimated) if free)
        self.fixed_log_parameters = initial_smoothing_parameters.log()
        self.cache: dict[
            tuple[float, ...],
            _GAMLSSProfileEvaluation,
        ] = {}

    def evaluate(
        self,
        free_log_parameters: Tensor,
        *,
        start: Tensor | None = None,
    ) -> _GAMLSSProfileEvaluation:
        key = tuple(float(value) for value in free_log_parameters.detach().cpu())
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if start is None:
            start = self._nearest_start(free_log_parameters)
        full_log_parameters = self.full_log_parameters(free_log_parameters)
        smoothing_parameters = full_log_parameters.exp()
        combined_penalty = sum(
            (
                smoothing_parameter * penalty
                for smoothing_parameter, penalty in zip(
                    smoothing_parameters,
                    self.penalty_matrices,
                    strict=True,
                )
            ),
            torch.zeros_like(self.penalty_matrices[0]),
        )
        combined_penalty = _symmetrize(combined_penalty)
        reduced_penalty = _symmetrize(
            self.transform.mT @ combined_penalty @ self.transform
        )
        reduced_components = tuple(
            _symmetrize(self.transform.mT @ penalty @ self.transform)
            for penalty in self.penalty_matrices
        )

        coefficients, iterations, gradient_max = self._fit_inner(
            reduced_penalty,
            start,
        )
        coefficient_value = coefficients.detach().requires_grad_(True)
        negative_log_likelihood = self._negative_log_likelihood(coefficient_value)
        observed_information = torch.autograd.functional.hessian(
            self._negative_log_likelihood,
            coefficient_value,
        )
        observed_information = _symmetrize(observed_information).detach()
        penalized_information = _symmetrize(observed_information + reduced_penalty)
        information_eigenvalues = torch.linalg.eigvalsh(penalized_information)
        information_tolerance = _matrix_tolerance(penalized_information)
        if float(information_eigenvalues.detach().min()) <= information_tolerance:
            raise RuntimeError("inner penalized information is not positive definite")
        log_determinant_information = information_eigenvalues.log().sum()
        (
            log_determinant_penalty,
            combined_rank,
        ) = _generalized_log_determinant(reduced_penalty)
        unpenalized_dimension = reduced_penalty.shape[0] - combined_rank

        penalty_value = 0.5 * (
            coefficient_value.detach() @ reduced_penalty @ coefficient_value.detach()
        )
        penalized_negative_log_likelihood = (
            negative_log_likelihood.detach() + penalty_value
        )
        objective = (
            penalized_negative_log_likelihood
            - 0.5 * log_determinant_penalty
            + 0.5 * log_determinant_information
            - 0.5 * unpenalized_dimension * math.log(2.0 * math.pi)
        )
        information_inverse = torch.linalg.inv(penalized_information)
        effective_degrees_of_freedom = torch.trace(
            information_inverse @ observed_information
        )
        penalty_degrees_of_freedom = torch.stack(
            tuple(
                torch.trace(information_inverse @ (smoothing_parameter * component))
                for smoothing_parameter, component in zip(
                    smoothing_parameters,
                    reduced_components,
                    strict=True,
                )
            )
        )

        full_coefficients = self.transform @ coefficient_value.detach()
        parameter_coefficients = {
            parameter: full_coefficients[self.coefficient_slices[parameter]]
            for parameter in self.family.parameter_names
        }
        linear_predictors = {
            parameter: (
                self.design_matrices[parameter] @ parameter_coefficients[parameter]
                + self.offsets[parameter]
            )
            for parameter in self.family.parameter_names
        }
        fitted_parameters = self.family.parameters_from_predictors(linear_predictors)
        profile = _GAMLSSProfileEvaluation(
            objective=objective.detach(),
            reduced_coefficients=coefficient_value.detach(),
            coefficients=full_coefficients.detach(),
            parameter_coefficients={
                name: value.detach() for name, value in parameter_coefficients.items()
            },
            linear_predictors={
                name: value.detach() for name, value in linear_predictors.items()
            },
            fitted_parameters={
                name: value.detach() for name, value in fitted_parameters.items()
            },
            log_likelihood=(-negative_log_likelihood).detach(),
            penalized_negative_log_likelihood=(
                penalized_negative_log_likelihood.detach()
            ),
            observed_information=observed_information,
            penalized_information=penalized_information.detach(),
            combined_penalty=combined_penalty.detach(),
            reduced_combined_penalty=reduced_penalty.detach(),
            log_determinant_penalty=log_determinant_penalty.detach(),
            log_determinant_information=log_determinant_information.detach(),
            combined_penalty_rank=combined_rank,
            unpenalized_dimension=unpenalized_dimension,
            effective_degrees_of_freedom=(effective_degrees_of_freedom.detach()),
            penalty_degrees_of_freedom=penalty_degrees_of_freedom.detach(),
            inner_iterations=iterations,
            inner_gradient_max=gradient_max,
        )
        self.cache[key] = profile
        return profile

    def _negative_log_likelihood(
        self,
        reduced_coefficients: Tensor,
    ) -> Tensor:
        full_coefficients = self.transform @ reduced_coefficients
        predictors = {
            parameter: (
                self.design_matrices[parameter]
                @ full_coefficients[self.coefficient_slices[parameter]]
                + self.offsets[parameter]
            )
            for parameter in self.family.parameter_names
        }
        parameters = self.family.parameters_from_predictors(predictors)
        log_probabilities = self.family.log_prob(self.response, parameters)
        if log_probabilities.shape != self.response.shape:
            raise RuntimeError(
                f"{self.family.name} log_prob must return one value per response"
            )
        losses = -log_probabilities
        if not bool(torch.isfinite(losses).all().detach()):
            raise RuntimeError(
                f"{self.family.name} negative log-likelihood is not finite"
            )
        return (self.weights * losses).sum()


def fit_normal_laml(
    response: Tensor,
    mu_design: Tensor,
    sigma_design: Tensor,
    penalty_matrices: Sequence[Tensor],
    smoothing_parameters: Sequence[float | Tensor],
    *,
    estimate_smoothing: Sequence[bool] | bool = True,
    weights: Tensor | None = None,
    mu_offset: Tensor | None = None,
    sigma_offset: Tensor | None = None,
    constraints: Tensor | None = None,
    initial_coefficients: Tensor | None = None,
    sigma_floor: float = 0.0,
    control: LAMLControl | None = None,
) -> NormalLAMLResult:
    """Fit a smooth Normal location-scale model by nested LAML optimization.

    The coefficient vector is ordered as all ``mu`` coefficients followed by
    all ``sigma`` coefficients. Every penalty matrix acts on this complete
    vector. At a trial ``rho = log(lambda)``, the inner problem minimizes the
    exact joint Normal negative log-likelihood plus
    ``0.5 * beta.T @ S_lambda @ beta``. The outer criterion is

    ``L_p - 0.5 log|S_lambda|_+ + 0.5 log|H_p| - M_p log(2*pi)/2``,

    where ``L_p`` is the penalized negative log-likelihood, ``H_p`` is its
    observed Hessian, and ``M_p`` is the unpenalized coefficient dimension.

    Likelihood gradients and Hessians use Torch autograd. The experimental
    outer BFGS algorithm differentiates the fully converged profile criterion
    by central differences; it does not differentiate through inner solver
    iterations.
    """
    control = control or LAMLControl()
    _validate_inputs(
        response,
        mu_design,
        sigma_design,
        sigma_floor=sigma_floor,
    )
    coefficient_count = mu_design.shape[1] + sigma_design.shape[1]
    penalties, initial_parameters, penalty_ranks = _validate_penalties(
        response,
        coefficient_count,
        penalty_matrices,
        smoothing_parameters,
    )
    estimated = _normalize_estimated(estimate_smoothing, len(penalties))
    if any(
        free and rank == 0 for free, rank in zip(estimated, penalty_ranks, strict=True)
    ):
        raise ValueError("an estimated smoothing parameter requires a nonzero penalty")
    observation_weights = _observation_vector(
        response,
        weights,
        default=1.0,
        name="weights",
        non_negative=True,
    )
    if float(observation_weights.sum()) <= 0:
        raise ValueError("at least one weight must be positive")
    mean_offset = _observation_vector(
        response,
        mu_offset,
        default=0.0,
        name="mu_offset",
    )
    scale_offset = _observation_vector(
        response,
        sigma_offset,
        default=0.0,
        name="sigma_offset",
    )
    constraint_rank, transform = _constraint_transform(
        response,
        coefficient_count,
        constraints,
    )
    starting_coefficients = _initial_coefficients(
        response,
        mu_design,
        sigma_design,
        mean_offset,
        scale_offset,
        transform,
        sigma_floor,
        initial_coefficients,
    )
    evaluator = _NormalProfileEvaluator(
        response,
        mu_design,
        sigma_design,
        penalties,
        initial_parameters,
        estimated,
        observation_weights,
        mean_offset,
        scale_offset,
        transform,
        starting_coefficients,
        sigma_floor,
        control,
    )

    optimization = _optimize_profile(
        evaluator,
        response,
        initial_parameters,
        estimated,
        control,
    )
    current = optimization.profile
    assert isinstance(current, _ProfileEvaluation)
    mu_count = mu_design.shape[1]
    return NormalLAMLResult(
        coefficients=current.coefficients,
        mu_coefficients=current.coefficients[:mu_count],
        sigma_coefficients=current.coefficients[mu_count:],
        linear_predictor_mu=current.linear_predictor_mu,
        linear_predictor_sigma=current.linear_predictor_sigma,
        fitted_mu=current.fitted_mu,
        fitted_sigma=current.fitted_sigma,
        log_smoothing_parameters=optimization.log_smoothing_parameters,
        smoothing_parameters=optimization.smoothing_parameters,
        estimated_smoothing_parameters=estimated,
        objective=current.objective,
        log_likelihood=current.log_likelihood,
        penalized_negative_log_likelihood=(current.penalized_negative_log_likelihood),
        outer_gradient=optimization.gradient,
        outer_hessian=optimization.hessian,
        outer_hessian_condition_number=(optimization.hessian_condition_number),
        outer_derivative_method=optimization.derivative_method,
        profile_evaluations=optimization.profile_evaluations,
        boundary_status=optimization.boundary_status,
        outer_converged=optimization.converged,
        outer_iterations=optimization.iterations,
        inner_converged=True,
        inner_iterations=current.inner_iterations,
        inner_gradient_max=current.inner_gradient_max,
        history=optimization.history,
        coefficient_transform=transform,
        reduced_observed_information=current.observed_information,
        reduced_penalized_information=current.penalized_information,
        combined_penalty_matrix=current.combined_penalty,
        reduced_combined_penalty_matrix=(current.reduced_combined_penalty),
        generalized_log_determinant_penalty=(current.log_determinant_penalty),
        log_determinant_penalized_information=(current.log_determinant_information),
        penalty_ranks=penalty_ranks,
        combined_penalty_rank=current.combined_penalty_rank,
        unpenalized_dimension=current.unpenalized_dimension,
        constraint_rank=constraint_rank,
        effective_degrees_of_freedom=(current.effective_degrees_of_freedom),
        penalty_degrees_of_freedom=current.penalty_degrees_of_freedom,
    )


def fit_gamlss_laml(
    family: Family,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    penalty_matrices: Sequence[Tensor],
    smoothing_parameters: Sequence[float | Tensor],
    *,
    estimate_smoothing: Sequence[bool] | bool = True,
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    constraints: Tensor | None = None,
    initial_coefficients: Tensor | None = None,
    control: LAMLControl | None = None,
) -> GAMLSSLAMLResult:
    """Fit a smooth additive family model by nested LAML optimization.

    Designs and coefficients follow ``family.parameter_names`` order. The
    likelihood is evaluated only through the public ``Family`` contract, so
    links and differentiable log densities remain family-specific while the
    penalized inner Newton fit and outer smoothing selection are shared.
    """
    control = control or LAMLControl()
    _validate_family_inputs(family, response, design_matrices)
    parameter_names = family.parameter_names
    coefficient_slices: dict[str, slice] = {}
    coefficient_count = 0
    for parameter in parameter_names:
        start = coefficient_count
        coefficient_count += design_matrices[parameter].shape[1]
        coefficient_slices[parameter] = slice(start, coefficient_count)

    penalties, initial_parameters, penalty_ranks = _validate_penalties(
        response,
        coefficient_count,
        penalty_matrices,
        smoothing_parameters,
    )
    estimated = _normalize_estimated(estimate_smoothing, len(penalties))
    if any(
        free and rank == 0 for free, rank in zip(estimated, penalty_ranks, strict=True)
    ):
        raise ValueError("an estimated smoothing parameter requires a nonzero penalty")
    observation_weights = _observation_vector(
        response,
        weights,
        default=1.0,
        name="weights",
        non_negative=True,
    )
    if float(observation_weights.sum()) <= 0:
        raise ValueError("at least one weight must be positive")
    supplied_offsets = offsets or {}
    extra_offsets = set(supplied_offsets).difference(parameter_names)
    if extra_offsets:
        raise ValueError(f"offsets contain unknown parameters: {sorted(extra_offsets)}")
    parameter_offsets = {
        parameter: _observation_vector(
            response,
            supplied_offsets.get(parameter),
            default=0.0,
            name=f"{parameter}_offset",
        )
        for parameter in parameter_names
    }
    constraint_rank, transform = _constraint_transform(
        response,
        coefficient_count,
        constraints,
    )
    starting_coefficients = _initial_family_coefficients(
        family,
        response,
        design_matrices,
        coefficient_slices,
        parameter_offsets,
        transform,
        initial_coefficients,
    )
    evaluator = _GAMLSSProfileEvaluator(
        family,
        response,
        design_matrices,
        coefficient_slices,
        penalties,
        initial_parameters,
        estimated,
        observation_weights,
        parameter_offsets,
        transform,
        starting_coefficients,
        control,
    )
    optimization = _optimize_profile(
        evaluator,
        response,
        initial_parameters,
        estimated,
        control,
    )
    current = optimization.profile
    assert isinstance(current, _GAMLSSProfileEvaluation)
    return GAMLSSLAMLResult(
        family=family.name,
        parameter_names=parameter_names,
        coefficients=current.coefficients,
        parameter_coefficients=current.parameter_coefficients,
        coefficient_slices=dict(coefficient_slices),
        linear_predictors=current.linear_predictors,
        fitted_parameters=current.fitted_parameters,
        log_smoothing_parameters=optimization.log_smoothing_parameters,
        smoothing_parameters=optimization.smoothing_parameters,
        estimated_smoothing_parameters=estimated,
        objective=current.objective,
        log_likelihood=current.log_likelihood,
        penalized_negative_log_likelihood=(current.penalized_negative_log_likelihood),
        outer_gradient=optimization.gradient,
        outer_hessian=optimization.hessian,
        outer_hessian_condition_number=(optimization.hessian_condition_number),
        outer_derivative_method=optimization.derivative_method,
        profile_evaluations=optimization.profile_evaluations,
        boundary_status=optimization.boundary_status,
        outer_converged=optimization.converged,
        outer_iterations=optimization.iterations,
        inner_converged=True,
        inner_iterations=current.inner_iterations,
        inner_gradient_max=current.inner_gradient_max,
        history=optimization.history,
        coefficient_transform=transform,
        reduced_observed_information=current.observed_information,
        reduced_penalized_information=current.penalized_information,
        combined_penalty_matrix=current.combined_penalty,
        reduced_combined_penalty_matrix=current.reduced_combined_penalty,
        generalized_log_determinant_penalty=(current.log_determinant_penalty),
        log_determinant_penalized_information=(current.log_determinant_information),
        penalty_ranks=penalty_ranks,
        combined_penalty_rank=current.combined_penalty_rank,
        unpenalized_dimension=current.unpenalized_dimension,
        constraint_rank=constraint_rank,
        effective_degrees_of_freedom=current.effective_degrees_of_freedom,
        penalty_degrees_of_freedom=current.penalty_degrees_of_freedom,
    )


def fit_gamlss_model_laml(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    control: LAMLControl | None = None,
    warm_start: bool = False,
) -> NormalLAMLResult | GAMLSSLAMLResult:
    """Fit a supported complete additive model and update its Torch state.

    Linear columns and every smooth design are assembled parameter by
    parameter. Each coefficient-space penalty contributes its own lambda to
    the outer LAML problem, so a tensor term contributes one free coordinate
    per marginal direction. Exact unidentifiable directions are removed by
    null-space constraints before the nested optimization.
    """
    from torchgamlss.families import Beta, Gamma, Normal, Poisson
    from torchgamlss.links import IdentityLink, LogitLink, LogLink

    is_normal = isinstance(model.family, Normal)
    is_poisson = isinstance(model.family, Poisson)
    is_gamma = isinstance(model.family, Gamma)
    is_beta = isinstance(model.family, Beta)
    if not is_normal and not is_poisson and not is_gamma and not is_beta:
        raise ValueError(
            "whole-model LAML currently supports Normal, Poisson, Gamma, "
            "and Beta families"
        )
    if is_normal and (
        not isinstance(model.family.links["mu"], IdentityLink)
        or not isinstance(model.family.links["sigma"], LogLink)
    ):
        raise ValueError(
            "whole-model Normal LAML requires identity mu and log sigma links"
        )
    if is_poisson and not isinstance(model.family.links["mu"], LogLink):
        raise ValueError("whole-model Poisson LAML requires a log mu link")
    if is_gamma and (
        not isinstance(model.family.links["mu"], LogLink)
        or not isinstance(model.family.links["sigma"], LogLink)
    ):
        raise ValueError("whole-model Gamma LAML requires log mu and log sigma links")
    if is_beta and (
        not isinstance(model.family.links["mu"], LogitLink)
        or not isinstance(model.family.links["sigma"], LogitLink)
    ):
        raise ValueError(
            "whole-model Beta LAML requires logit mu and logit sigma links"
        )
    if model.neural_predictors or model.shared_predictor is not None:
        raise ValueError(
            "whole-model LAML does not support neural or shared predictors"
        )
    if not any(terms for terms in model.smooth_terms.values()):
        raise ValueError("whole-model LAML requires at least one smooth term")

    model.family.validate_response(response, context="LAML")
    contributions = model.term_contributions(
        design_matrices,
        offsets,
        smooth_covariates=smooth_covariates,
    )
    observation_weights = model._validated_weights(response, weights)

    parameter_designs: dict[str, Tensor] = {}
    local_linear_slices: dict[str, slice] = {}
    local_smooth_slices: dict[tuple[str, str], slice] = {}
    parameter_penalties: dict[str, Tensor] = {}
    for parameter in model.family.parameter_names:
        linear_design = design_matrices[parameter]
        local_linear_slices[parameter] = slice(0, linear_design.shape[1])
        components = [linear_design]
        cursor = linear_design.shape[1]
        for term_name, term in model.smooth_terms[parameter].items():
            covariate = smooth_covariates[parameter][term_name]
            basis = term.design(covariate)
            stop = cursor + basis.shape[1]
            local_smooth_slices[(parameter, term_name)] = slice(
                cursor,
                stop,
            )
            components.append(basis)
            cursor = stop
        parameter_design = torch.cat(components, dim=1)
        parameter_designs[parameter] = parameter_design
        parameter_penalties[parameter] = response.new_zeros((cursor, cursor))

    parameter_offsets: dict[str, int] = {}
    cursor = 0
    for parameter in model.family.parameter_names:
        parameter_offsets[parameter] = cursor
        cursor += parameter_designs[parameter].shape[1]
    coefficient_count = cursor

    linear_slices: dict[str, slice] = {}
    smooth_slices: dict[tuple[str, str], slice] = {}
    for parameter in model.family.parameter_names:
        offset = parameter_offsets[parameter]
        local = local_linear_slices[parameter]
        linear_slices[parameter] = slice(
            offset + local.start,
            offset + local.stop,
        )
        for term_name in model.smooth_terms[parameter]:
            local = local_smooth_slices[(parameter, term_name)]
            smooth_slices[(parameter, term_name)] = slice(
                offset + local.start,
                offset + local.stop,
            )

    penalties: list[Tensor] = []
    smoothing_parameters: list[float] = []
    estimated: list[bool] = []
    smoothing_labels: list[tuple[str, str, int]] = []
    smoothing_slices: dict[tuple[str, str], slice] = {}
    constraint_rows: list[Tensor] = []
    penalty_cursor = 0
    for parameter in model.family.parameter_names:
        parameter_offset = parameter_offsets[parameter]
        for term_name, term in model.smooth_terms[parameter].items():
            term_penalties = term.penalty_matrices()
            term_smoothing = term.smoothing_parameters
            term_estimated = term.estimated_smoothing_parameters
            if not (len(term_penalties) == len(term_smoothing) == len(term_estimated)):
                raise RuntimeError(
                    f"smooth term {parameter!r}.{term_name} has inconsistent "
                    "penalty metadata"
                )
            local_slice = local_smooth_slices[(parameter, term_name)]
            global_slice = smooth_slices[(parameter, term_name)]
            smoothing_slices[(parameter, term_name)] = slice(
                penalty_cursor,
                penalty_cursor + len(term_penalties),
            )
            for penalty_index, (
                penalty,
                smoothing_parameter,
                estimate,
            ) in enumerate(
                zip(
                    term_penalties,
                    term_smoothing,
                    term_estimated,
                    strict=True,
                )
            ):
                full_penalty = response.new_zeros(
                    (coefficient_count, coefficient_count)
                )
                full_penalty[global_slice, global_slice] = penalty
                penalties.append(full_penalty)
                parameter_penalties[parameter][
                    local_slice,
                    local_slice,
                ] += penalty
                smoothing_parameters.append(smoothing_parameter)
                estimated.append(estimate)
                smoothing_labels.append((parameter, term_name, penalty_index))
                penalty_cursor += 1

            explicit = term.constraints(smooth_covariates[parameter][term_name])
            if explicit.shape[0]:
                embedded = response.new_zeros((explicit.shape[0], coefficient_count))
                embedded[:, global_slice] = explicit
                constraint_rows.append(embedded)

        weighted_design = parameter_designs[parameter] * (
            observation_weights.sqrt().unsqueeze(-1)
        )
        identifiability_system = _symmetrize(
            weighted_design.mT @ weighted_design + parameter_penalties[parameter]
        )
        unidentified = _null_directions(identifiability_system)
        if unidentified.shape[1]:
            embedded = response.new_zeros((unidentified.shape[1], coefficient_count))
            start = parameter_offset
            stop = start + parameter_designs[parameter].shape[1]
            embedded[:, start:stop] = unidentified.mT
            constraint_rows.append(embedded)

    constraints = torch.cat(constraint_rows, dim=0) if constraint_rows else None
    initial_coefficients = None
    if warm_start:
        initial_coefficients = response.new_empty(coefficient_count)
        for parameter in model.family.parameter_names:
            initial_coefficients[linear_slices[parameter]] = model.coefficients[
                parameter
            ].detach()
            for term_name, term in model.smooth_terms[parameter].items():
                initial_coefficients[smooth_slices[(parameter, term_name)]] = (
                    term.coefficients.detach()
                )
    if is_normal:
        result = fit_normal_laml(
            response,
            parameter_designs["mu"],
            parameter_designs["sigma"],
            penalties,
            smoothing_parameters,
            estimate_smoothing=estimated,
            weights=observation_weights,
            mu_offset=contributions["mu"].offset,
            sigma_offset=contributions["sigma"].offset,
            constraints=constraints,
            initial_coefficients=initial_coefficients,
            control=control,
        )
    else:
        result = fit_gamlss_laml(
            model.family,
            response,
            parameter_designs,
            penalties,
            smoothing_parameters,
            estimate_smoothing=estimated,
            weights=observation_weights,
            offsets={
                parameter: contributions[parameter].offset
                for parameter in model.family.parameter_names
            },
            constraints=constraints,
            initial_coefficients=initial_coefficients,
            control=control,
        )
    result = replace(
        result,
        linear_coefficient_slices=dict(linear_slices),
        smooth_coefficient_slices=dict(smooth_slices),
        smoothing_parameter_labels=tuple(smoothing_labels),
        smoothing_parameter_slices=dict(smoothing_slices),
    )

    with torch.no_grad():
        for parameter in model.family.parameter_names:
            model.coefficients[parameter].copy_(
                result.coefficients[linear_slices[parameter]]
            )
            for term_name, term in model.smooth_terms[parameter].items():
                term.coefficients.copy_(
                    result.coefficients[smooth_slices[(parameter, term_name)]]
                )
                parameter_slice = smoothing_slices[(parameter, term_name)]
                selected = result.smoothing_parameters[parameter_slice].detach()
                retained = tuple(
                    (float(value) if estimate else current)
                    for value, estimate, current in zip(
                        selected,
                        term.estimated_smoothing_parameters,
                        term.smoothing_parameters,
                        strict=True,
                    )
                )
                term._set_fitted_smoothing_parameters(retained)
    return result


def fit_normal_gamlss_laml(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]],
    weights: Tensor | None = None,
    offsets: Mapping[str, Tensor] | None = None,
    control: LAMLControl | None = None,
    warm_start: bool = False,
) -> NormalLAMLResult:
    """Backward-compatible whole-model Normal LAML entry point."""
    from torchgamlss.families import Normal

    if not isinstance(model.family, Normal):
        raise ValueError("fit_normal_gamlss_laml requires a Normal family model")
    result = fit_gamlss_model_laml(
        model,
        response,
        design_matrices,
        smooth_covariates=smooth_covariates,
        weights=weights,
        offsets=offsets,
        control=control,
        warm_start=warm_start,
    )
    assert isinstance(result, NormalLAMLResult)
    return result


def _optimize_profile(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    response: Tensor,
    initial_parameters: Tensor,
    estimated: tuple[bool, ...],
    control: LAMLControl,
) -> _OuterOptimization:
    """Optimize the profiled Laplace criterion over free log lambdas."""
    free_indices = evaluator.free_indices
    free_log_parameters = initial_parameters[
        torch.tensor(free_indices, dtype=torch.long, device=response.device)
    ].log()
    lower, upper = control.log_smoothing_parameter_bounds
    free_log_parameters = free_log_parameters.clamp(lower, upper)
    current = evaluator.evaluate(free_log_parameters)
    history: list[LAMLHistoryEntry] = []

    if free_indices:
        gradient = _outer_profile_gradient(
            evaluator,
            free_log_parameters,
            current,
            control,
        )
        projected = _projected_gradient(
            free_log_parameters,
            gradient,
            lower,
            upper,
        )
        history.append(
            _history_entry(
                evaluator,
                0,
                current,
                free_log_parameters,
                gradient,
                projected,
                0.0,
            )
        )
        inverse_hessian = torch.eye(
            len(free_indices),
            dtype=response.dtype,
            device=response.device,
        )
        outer_converged = float(projected.abs().max()) <= (
            control.outer_gradient_tolerance
        )
        accepted_iterations = 0

        while (
            not outer_converged and accepted_iterations < control.outer_max_iterations
        ):
            direction = -(inverse_hessian @ projected)
            if float(gradient @ direction) >= 0:
                direction = -projected
                inverse_hessian = torch.eye(
                    len(free_indices),
                    dtype=response.dtype,
                    device=response.device,
                )
            accepted = False
            scale = 1.0
            trial_log_parameters = free_log_parameters
            trial = current
            delta = torch.zeros_like(free_log_parameters)
            for _ in range(control.max_line_search_steps):
                candidate_log_parameters = (
                    free_log_parameters + scale * direction
                ).clamp(lower, upper)
                candidate_delta = candidate_log_parameters - free_log_parameters
                if float(candidate_delta.abs().max()) == 0.0:
                    scale *= 0.5
                    continue
                try:
                    candidate = evaluator.evaluate(
                        candidate_log_parameters,
                        start=current.reduced_coefficients,
                    )
                except RuntimeError:
                    scale *= 0.5
                    continue
                armijo = float(current.objective) + 1e-4 * float(
                    gradient @ candidate_delta
                )
                if float(candidate.objective) <= armijo:
                    trial_log_parameters = candidate_log_parameters
                    trial = candidate
                    delta = candidate_delta
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break

            trial_gradient = _outer_profile_gradient(
                evaluator,
                trial_log_parameters,
                trial,
                control,
            )
            trial_projected = _projected_gradient(
                trial_log_parameters,
                trial_gradient,
                lower,
                upper,
            )
            difference = trial_gradient - gradient
            curvature = float(difference @ delta)
            if curvature > (
                torch.finfo(response.dtype).eps
                * max(float(delta.norm() * difference.norm()), 1.0)
            ):
                identity = torch.eye(
                    len(free_indices),
                    dtype=response.dtype,
                    device=response.device,
                )
                rho = 1.0 / curvature
                left = identity - rho * torch.outer(delta, difference)
                right = identity - rho * torch.outer(difference, delta)
                inverse_hessian = left @ inverse_hessian @ right + rho * torch.outer(
                    delta, delta
                )
                inverse_hessian = _symmetrize(inverse_hessian)
            else:
                inverse_hessian = torch.eye(
                    len(free_indices),
                    dtype=response.dtype,
                    device=response.device,
                )

            free_log_parameters = trial_log_parameters
            current = trial
            gradient = trial_gradient
            projected = trial_projected
            accepted_iterations += 1
            step_norm = float(delta.norm())
            history.append(
                _history_entry(
                    evaluator,
                    accepted_iterations,
                    current,
                    free_log_parameters,
                    gradient,
                    projected,
                    step_norm,
                )
            )
            outer_converged = float(projected.abs().max()) <= (
                control.outer_gradient_tolerance
            )
            if not outer_converged and step_norm <= control.outer_step_tolerance * (
                1.0 + float(free_log_parameters.norm())
            ):
                break

        outer_gradient_free = gradient
        outer_hessian_free = _outer_profile_hessian(
            evaluator,
            free_log_parameters,
            current,
            control,
        )
    else:
        outer_converged = True
        accepted_iterations = 0
        outer_gradient_free = response.new_empty((0,))
        outer_hessian_free = response.new_empty((0, 0))
        history.append(
            _history_entry(
                evaluator,
                0,
                current,
                free_log_parameters,
                outer_gradient_free,
                outer_gradient_free,
                0.0,
            )
        )

    full_log_parameters = evaluator.full_log_parameters(free_log_parameters)
    full_smoothing_parameters = full_log_parameters.exp()
    outer_gradient = response.new_zeros(initial_parameters.numel())
    outer_hessian = response.new_zeros(
        (initial_parameters.numel(), initial_parameters.numel())
    )
    if free_indices:
        index = torch.tensor(
            free_indices,
            dtype=torch.long,
            device=response.device,
        )
        outer_gradient[index] = outer_gradient_free
        outer_hessian[index.unsqueeze(1), index.unsqueeze(0)] = outer_hessian_free
        hessian_condition = torch.linalg.cond(outer_hessian_free)
    else:
        hessian_condition = response.new_tensor(1.0)
    boundary_status = tuple(
        _boundary_status(
            float(value),
            free,
            lower,
            upper,
            control.outer_step_tolerance,
        )
        for value, free in zip(
            full_log_parameters,
            estimated,
            strict=True,
        )
    )
    return _OuterOptimization(
        profile=current,
        log_smoothing_parameters=full_log_parameters.detach(),
        smoothing_parameters=full_smoothing_parameters.detach(),
        gradient=outer_gradient,
        hessian=outer_hessian,
        hessian_condition_number=hessian_condition.detach(),
        derivative_method=control.outer_derivative_method,
        profile_evaluations=len(evaluator.cache),
        boundary_status=boundary_status,
        converged=outer_converged,
        iterations=accepted_iterations,
        history=tuple(history),
    )


def _outer_profile_gradient(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    log_parameters: Tensor,
    central: _ProfileEvaluation | _GAMLSSProfileEvaluation,
    control: LAMLControl,
) -> Tensor:
    if control.outer_derivative_method == "implicit":
        return _implicit_profile_gradient(
            evaluator,
            log_parameters,
            central,
        )
    return _profile_gradient(
        evaluator,
        log_parameters,
        central,
        control.finite_difference_step,
    )


def _outer_profile_hessian(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    log_parameters: Tensor,
    central: _ProfileEvaluation | _GAMLSSProfileEvaluation,
    control: LAMLControl,
) -> Tensor:
    if control.outer_derivative_method == "implicit":
        return _implicit_profile_hessian(
            evaluator,
            log_parameters,
            central,
        )
    return _profile_hessian(
        evaluator,
        log_parameters,
        central,
        control.hessian_difference_step,
    )


def _implicit_profile_gradient(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    log_parameters: Tensor,
    central: _ProfileEvaluation | _GAMLSSProfileEvaluation,
) -> Tensor:
    """Differentiate the converged LAML profile by the implicit function theorem."""
    full_log_parameters = evaluator.full_log_parameters(log_parameters)
    smoothing_parameters = full_log_parameters.exp()
    reduced_components = tuple(
        _symmetrize(evaluator.transform.mT @ penalty @ evaluator.transform)
        for penalty in evaluator.penalty_matrices
    )
    scaled_components = tuple(
        smoothing_parameter * component
        for smoothing_parameter, component in zip(
            smoothing_parameters,
            reduced_components,
            strict=True,
        )
    )

    coefficients = central.reduced_coefficients.detach()
    penalized_information = central.penalized_information.detach()
    information_inverse = torch.linalg.inv(penalized_information)
    penalty_inverse = _symmetric_pseudoinverse(central.reduced_combined_penalty)

    differentiable_coefficients = coefficients.requires_grad_(True)
    observed_information = torch.autograd.functional.hessian(
        evaluator._negative_log_likelihood,
        differentiable_coefficients,
        create_graph=True,
    )
    differentiable_information = _symmetrize(
        observed_information + central.reduced_combined_penalty
    )
    sign, log_determinant = torch.linalg.slogdet(differentiable_information)
    if float(sign.detach()) <= 0:
        raise RuntimeError(
            "implicit LAML gradient requires positive definite information"
        )
    if log_determinant.requires_grad:
        information_coefficient_gradient = torch.autograd.grad(
            0.5 * log_determinant,
            differentiable_coefficients,
        )[0].detach()
    else:
        information_coefficient_gradient = torch.zeros_like(coefficients)

    values: list[Tensor] = []
    for penalty_index in evaluator.free_indices:
        derivative_penalty = scaled_components[penalty_index]
        coefficient_sensitivity = -torch.linalg.solve(
            penalized_information,
            derivative_penalty @ coefficients,
        )
        penalized_objective_derivative = 0.5 * (
            coefficients @ derivative_penalty @ coefficients
        )
        penalty_determinant_derivative = -0.5 * torch.trace(
            penalty_inverse @ derivative_penalty
        )
        information_direct_derivative = 0.5 * torch.trace(
            information_inverse @ derivative_penalty
        )
        information_implicit_derivative = (
            information_coefficient_gradient @ coefficient_sensitivity
        )
        values.append(
            penalized_objective_derivative
            + penalty_determinant_derivative
            + information_direct_derivative
            + information_implicit_derivative
        )
    return torch.stack(values).detach()


def _implicit_profile_hessian(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    log_parameters: Tensor,
    central: _ProfileEvaluation | _GAMLSSProfileEvaluation,
) -> Tensor:
    """Build the exact profile Hessian from implicit coefficient sensitivities."""
    coefficients = central.reduced_coefficients.detach()
    coefficient_count = coefficients.numel()
    free_smoothing_count = log_parameters.numel()
    penalized_information = central.penalized_information.detach()

    reduced_components = tuple(
        _symmetrize(evaluator.transform.mT @ penalty @ evaluator.transform)
        for penalty in evaluator.penalty_matrices
    )
    full_log_parameters = evaluator.full_log_parameters(log_parameters)
    smoothing_parameters = full_log_parameters.exp()
    scaled_components = tuple(
        smoothing_parameter * component
        for smoothing_parameter, component in zip(
            smoothing_parameters,
            reduced_components,
            strict=True,
        )
    )
    free_components = tuple(
        scaled_components[index] for index in evaluator.free_indices
    )

    sensitivity_right_hand_side = torch.stack(
        tuple(component @ coefficients for component in free_components),
        dim=1,
    )
    coefficient_sensitivity = -torch.linalg.solve(
        penalized_information,
        sensitivity_right_hand_side,
    )

    second_sensitivity = coefficients.new_empty(
        (coefficient_count, free_smoothing_count, free_smoothing_count)
    )
    # If g(beta, rho) is the penalized score, differentiating g = 0 twice gives
    # H_p beta_jk = -(l'''[beta_j, beta_k] + D_j beta_k + D_k beta_j
    #                  + 1[j=k] D_j beta), where D_j = lambda_j S_j.
    for left in range(free_smoothing_count):
        left_sensitivity = coefficient_sensitivity[:, left]

        def information_times_left(value: Tensor) -> Tensor:
            likelihood_gradient = torch.autograd.grad(
                evaluator._negative_log_likelihood(value),
                value,
                create_graph=True,
            )[0]
            return torch.autograd.grad(
                likelihood_gradient @ left_sensitivity,
                value,
                create_graph=True,
            )[0]

        for right in range(left, free_smoothing_count):
            right_sensitivity = coefficient_sensitivity[:, right]
            _, third_derivative_contraction = torch.autograd.functional.jvp(
                information_times_left,
                coefficients,
                right_sensitivity,
            )
            right_hand_side = (
                third_derivative_contraction
                + free_components[right] @ left_sensitivity
                + free_components[left] @ right_sensitivity
            )
            if left == right:
                right_hand_side = right_hand_side + free_components[left] @ coefficients
            value = -torch.linalg.solve(
                penalized_information,
                right_hand_side,
            )
            second_sensitivity[:, left, right] = value
            second_sensitivity[:, right, left] = value

    penalty_eigenvalues, penalty_eigenvectors = torch.linalg.eigh(
        central.reduced_combined_penalty
    )
    # Positive lambdas keep the combined penalty range fixed locally. Restricting
    # to this detached basis makes the generalized log determinant differentiable.
    penalty_range = penalty_eigenvectors[
        :, penalty_eigenvalues > _matrix_tolerance(central.reduced_combined_penalty)
    ].detach()

    def partial_profile_objective(value: Tensor) -> Tensor:
        trial_coefficients = value[:coefficient_count]
        trial_log_parameters = value[coefficient_count:]
        trial_full_log_parameters = evaluator.full_log_parameters(trial_log_parameters)
        trial_smoothing_parameters = trial_full_log_parameters.exp()
        trial_penalty = _symmetrize(
            sum(
                (
                    smoothing_parameter * component
                    for smoothing_parameter, component in zip(
                        trial_smoothing_parameters,
                        reduced_components,
                        strict=True,
                    )
                ),
                torch.zeros_like(reduced_components[0]),
            )
        )
        negative_log_likelihood = evaluator._negative_log_likelihood(trial_coefficients)
        observed_information = torch.autograd.functional.hessian(
            evaluator._negative_log_likelihood,
            trial_coefficients,
            create_graph=True,
        )
        trial_information = _symmetrize(observed_information + trial_penalty)
        penalty_log_determinant = trial_coefficients.new_zeros(())
        if penalty_range.shape[1] > 0:
            penalty_log_determinant = torch.linalg.slogdet(
                penalty_range.mT @ trial_penalty @ penalty_range
            ).logabsdet
        information_log_determinant = torch.linalg.slogdet(trial_information).logabsdet
        return (
            negative_log_likelihood
            + 0.5 * trial_coefficients @ trial_penalty @ trial_coefficients
            - 0.5 * penalty_log_determinant
            + 0.5 * information_log_determinant
        )

    joint_point = torch.cat((coefficients, log_parameters.detach()))
    partial_gradient = torch.autograd.functional.jacobian(
        partial_profile_objective,
        joint_point,
    )
    partial_hessian = torch.autograd.functional.hessian(
        partial_profile_objective,
        joint_point,
    )
    joint_sensitivity = torch.cat(
        (
            coefficient_sensitivity,
            torch.eye(
                free_smoothing_count,
                dtype=log_parameters.dtype,
                device=log_parameters.device,
            ),
        ),
        dim=0,
    )
    # For z(rho) = (beta_hat(rho), rho), apply
    # d2 V / d rho2 = z' V_zz z + V_beta beta_jk.
    hessian = joint_sensitivity.mT @ partial_hessian @ joint_sensitivity
    hessian = hessian + torch.einsum(
        "a,ajk->jk",
        partial_gradient[:coefficient_count],
        second_sensitivity,
    )
    return _symmetrize(hessian.detach())


def _profile_gradient(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    log_parameters: Tensor,
    central: _ProfileEvaluation | _GAMLSSProfileEvaluation,
    step_size: float,
) -> Tensor:
    values: list[Tensor] = []
    for index in range(log_parameters.numel()):
        step = step_size * max(1.0, abs(float(log_parameters[index])))
        displacement = torch.zeros_like(log_parameters)
        displacement[index] = step
        plus = evaluator.evaluate(
            log_parameters + displacement,
            start=central.reduced_coefficients,
        )
        minus = evaluator.evaluate(
            log_parameters - displacement,
            start=central.reduced_coefficients,
        )
        values.append((plus.objective - minus.objective) / (2.0 * step))
    return torch.stack(values)


def _profile_hessian(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    log_parameters: Tensor,
    central: _ProfileEvaluation | _GAMLSSProfileEvaluation,
    step_size: float,
) -> Tensor:
    count = log_parameters.numel()
    hessian = log_parameters.new_zeros((count, count))
    steps = tuple(step_size * max(1.0, abs(float(value))) for value in log_parameters)
    for left in range(count):
        left_step = torch.zeros_like(log_parameters)
        left_step[left] = steps[left]
        plus = evaluator.evaluate(
            log_parameters + left_step,
            start=central.reduced_coefficients,
        )
        minus = evaluator.evaluate(
            log_parameters - left_step,
            start=central.reduced_coefficients,
        )
        hessian[left, left] = (
            plus.objective - 2.0 * central.objective + minus.objective
        ) / steps[left] ** 2
        for right in range(left + 1, count):
            right_step = torch.zeros_like(log_parameters)
            right_step[right] = steps[right]
            plus_plus = evaluator.evaluate(
                log_parameters + left_step + right_step,
                start=central.reduced_coefficients,
            )
            plus_minus = evaluator.evaluate(
                log_parameters + left_step - right_step,
                start=central.reduced_coefficients,
            )
            minus_plus = evaluator.evaluate(
                log_parameters - left_step + right_step,
                start=central.reduced_coefficients,
            )
            minus_minus = evaluator.evaluate(
                log_parameters - left_step - right_step,
                start=central.reduced_coefficients,
            )
            value = (
                plus_plus.objective
                - plus_minus.objective
                - minus_plus.objective
                + minus_minus.objective
            ) / (4.0 * steps[left] * steps[right])
            hessian[left, right] = value
            hessian[right, left] = value
    return _symmetrize(hessian)


def _projected_gradient(
    log_parameters: Tensor,
    gradient: Tensor,
    lower: float,
    upper: float,
) -> Tensor:
    projected = gradient.clone()
    tolerance = 10.0 * torch.finfo(log_parameters.dtype).eps
    projected[(log_parameters <= lower + tolerance) & (gradient > 0)] = 0
    projected[(log_parameters >= upper - tolerance) & (gradient < 0)] = 0
    return projected


def _history_entry(
    evaluator: _NormalProfileEvaluator | _GAMLSSProfileEvaluator,
    iteration: int,
    profile: _ProfileEvaluation | _GAMLSSProfileEvaluation,
    free_log_parameters: Tensor,
    free_gradient: Tensor,
    projected_gradient: Tensor,
    step_norm: float,
) -> LAMLHistoryEntry:
    full_log_parameters = evaluator.full_log_parameters(free_log_parameters)
    full_gradient = full_log_parameters.new_zeros(full_log_parameters.shape)
    if evaluator.free_indices:
        indices = torch.tensor(
            evaluator.free_indices,
            dtype=torch.long,
            device=full_gradient.device,
        )
        full_gradient[indices] = free_gradient
    projected_max = (
        float(projected_gradient.abs().max()) if projected_gradient.numel() else 0.0
    )
    return LAMLHistoryEntry(
        iteration=iteration,
        objective=float(profile.objective),
        log_smoothing_parameters=full_log_parameters.detach().clone(),
        smoothing_parameters=full_log_parameters.exp().detach(),
        gradient=full_gradient.detach(),
        projected_gradient_max=projected_max,
        step_norm=step_norm,
        inner_iterations=profile.inner_iterations,
        inner_gradient_max=profile.inner_gradient_max,
    )


def _validate_family_inputs(
    family: Family,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
) -> None:
    if response.ndim != 1 or response.numel() < 2:
        raise ValueError("response must be one-dimensional with at least 2 values")
    if not response.is_floating_point() or not torch.isfinite(response).all():
        raise ValueError("response must use a finite floating-point dtype")
    family.validate_response(response, context="LAML")
    missing = set(family.parameter_names).difference(design_matrices)
    extra = set(design_matrices).difference(family.parameter_names)
    if missing or extra:
        raise ValueError(
            "design_matrices do not match family parameters: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for parameter in family.parameter_names:
        design = design_matrices[parameter]
        if design.ndim != 2 or design.shape[0] != response.numel():
            raise ValueError(
                f"{parameter!r} design must be two-dimensional with one "
                "row per response"
            )
        if design.dtype != response.dtype or design.device != response.device:
            raise ValueError(
                f"{parameter!r} design must match the response dtype and device"
            )
        if not torch.isfinite(design).all():
            raise ValueError(f"{parameter!r} design must be finite")
    if not any(
        design_matrices[parameter].shape[1] > 0 for parameter in family.parameter_names
    ):
        raise ValueError("at least one family parameter design must contain a column")


def _validate_inputs(
    response: Tensor,
    mu_design: Tensor,
    sigma_design: Tensor,
    *,
    sigma_floor: float,
) -> None:
    if response.ndim != 1 or response.numel() < 2:
        raise ValueError("response must be one-dimensional with at least 2 values")
    if not response.is_floating_point() or not torch.isfinite(response).all():
        raise ValueError("response must use a finite floating-point dtype")
    for name, design in (
        ("mu_design", mu_design),
        ("sigma_design", sigma_design),
    ):
        if (
            design.ndim != 2
            or design.shape[0] != response.numel()
            or design.shape[1] < 1
        ):
            raise ValueError(
                f"{name} must be two-dimensional with one row per response"
            )
        if design.dtype != response.dtype or design.device != response.device:
            raise ValueError(f"{name} must match the response dtype and device")
        if not torch.isfinite(design).all():
            raise ValueError(f"{name} must be finite")
    if not math.isfinite(sigma_floor) or sigma_floor < 0:
        raise ValueError("sigma_floor must be finite and non-negative")


def _validate_penalties(
    response: Tensor,
    coefficient_count: int,
    penalty_matrices: Sequence[Tensor],
    smoothing_parameters: Sequence[float | Tensor],
) -> tuple[tuple[Tensor, ...], Tensor, tuple[int, ...]]:
    if isinstance(penalty_matrices, Tensor):
        raise ValueError("penalty_matrices must be a sequence of square tensors")
    if isinstance(smoothing_parameters, Tensor):
        raise ValueError("smoothing_parameters must be a sequence of scalars")
    penalties = tuple(penalty_matrices)
    parameters = tuple(smoothing_parameters)
    if not penalties:
        raise ValueError("at least one penalty matrix is required")
    if len(penalties) != len(parameters):
        raise ValueError(
            "penalty_matrices and smoothing_parameters must have equal lengths"
        )
    validated: list[Tensor] = []
    ranks: list[int] = []
    smoothing_values: list[Tensor] = []
    for index, (penalty, parameter) in enumerate(
        zip(penalties, parameters, strict=True)
    ):
        if (
            not isinstance(penalty, Tensor)
            or penalty.ndim != 2
            or penalty.shape != (coefficient_count, coefficient_count)
        ):
            raise ValueError(
                f"penalty matrix {index} must have shape "
                f"({coefficient_count}, {coefficient_count})"
            )
        if penalty.dtype != response.dtype or penalty.device != response.device:
            raise ValueError(
                f"penalty matrix {index} must match the response dtype and device"
            )
        if not torch.isfinite(penalty).all():
            raise ValueError(f"penalty matrix {index} must be finite")
        tolerance = _matrix_tolerance(penalty)
        if float((penalty - penalty.mT).abs().max()) > tolerance:
            raise ValueError(f"penalty matrix {index} must be symmetric")
        symmetric = _symmetrize(penalty)
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError(f"penalty matrix {index} must be positive semidefinite")
        validated.append(symmetric)
        ranks.append(int((eigenvalues > tolerance).sum()))

        if isinstance(parameter, Tensor):
            if (
                parameter.ndim != 0
                or parameter.dtype != response.dtype
                or parameter.device != response.device
            ):
                raise ValueError(
                    f"smoothing parameter {index} must be a matching scalar"
                )
            value = parameter.detach()
        else:
            try:
                value = response.new_tensor(float(parameter))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"smoothing parameter {index} must be scalar"
                ) from error
        if not bool(torch.isfinite(value)) or float(value) <= 0:
            raise ValueError(f"smoothing parameter {index} must be finite and positive")
        smoothing_values.append(value)
    return (
        tuple(validated),
        torch.stack(smoothing_values),
        tuple(ranks),
    )


def _normalize_estimated(
    estimate_smoothing: Sequence[bool] | bool,
    count: int,
) -> tuple[bool, ...]:
    if isinstance(estimate_smoothing, bool):
        return (estimate_smoothing,) * count
    values = tuple(estimate_smoothing)
    if len(values) != count or any(not isinstance(value, bool) for value in values):
        raise ValueError("estimate_smoothing must contain one boolean per penalty")
    return values


def _observation_vector(
    response: Tensor,
    value: Tensor | None,
    *,
    default: float,
    name: str,
    non_negative: bool = False,
) -> Tensor:
    if value is None:
        return torch.full_like(response, default)
    if (
        value.ndim != 1
        or value.shape != response.shape
        or value.dtype != response.dtype
        or value.device != response.device
    ):
        raise ValueError(
            f"{name} must have one value per response with matching dtype and device"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if non_negative and (value < 0).any():
        raise ValueError(f"{name} must be non-negative")
    return value


def _constraint_transform(
    response: Tensor,
    coefficient_count: int,
    constraints: Tensor | None,
) -> tuple[int, Tensor]:
    if constraints is None:
        return (
            0,
            torch.eye(
                coefficient_count,
                dtype=response.dtype,
                device=response.device,
            ),
        )
    if constraints.ndim != 2 or constraints.shape[1] != coefficient_count:
        raise ValueError(f"constraints must have shape (q, {coefficient_count})")
    if constraints.dtype != response.dtype or constraints.device != response.device:
        raise ValueError("constraints must match the response dtype and device")
    if not torch.isfinite(constraints).all():
        raise ValueError("constraints must be finite")
    if constraints.shape[0] == 0:
        return (
            0,
            torch.eye(
                coefficient_count,
                dtype=response.dtype,
                device=response.device,
            ),
        )
    _, singular_values, right_vectors = torch.linalg.svd(
        constraints,
        full_matrices=True,
    )
    largest = float(singular_values.max()) if singular_values.numel() else 0.0
    tolerance = max(constraints.shape) * torch.finfo(response.dtype).eps * largest
    rank = int((singular_values > tolerance).sum())
    if rank >= coefficient_count:
        raise ValueError("constraints must leave at least one coefficient free")
    return rank, right_vectors[rank:].mT


def _initial_coefficients(
    response: Tensor,
    mu_design: Tensor,
    sigma_design: Tensor,
    mu_offset: Tensor,
    sigma_offset: Tensor,
    transform: Tensor,
    sigma_floor: float,
    initial_coefficients: Tensor | None,
) -> Tensor:
    coefficient_count = mu_design.shape[1] + sigma_design.shape[1]
    if initial_coefficients is not None:
        if (
            initial_coefficients.ndim != 1
            or initial_coefficients.numel() != coefficient_count
            or initial_coefficients.dtype != response.dtype
            or initial_coefficients.device != response.device
        ):
            raise ValueError(
                "initial_coefficients must match the complete coefficient vector"
            )
        if not torch.isfinite(initial_coefficients).all():
            raise ValueError("initial_coefficients must be finite")
        full = initial_coefficients
    else:
        mu_coefficients = torch.linalg.pinv(mu_design) @ (response - mu_offset)
        residual = response - mu_offset - mu_design @ mu_coefficients
        residual_scale = max(
            float(residual.std(correction=1)),
            math.sqrt(torch.finfo(response.dtype).eps),
        )
        adjusted = (residual.abs() - sigma_floor).clamp_min(residual_scale * 1e-3)
        sigma_target = adjusted.log() - sigma_offset
        sigma_coefficients = torch.linalg.pinv(sigma_design) @ sigma_target
        full = torch.cat((mu_coefficients, sigma_coefficients))
    return transform.mT @ full


def _initial_family_coefficients(
    family: Family,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    coefficient_slices: Mapping[str, slice],
    offsets: Mapping[str, Tensor],
    transform: Tensor,
    initial_coefficients: Tensor | None,
) -> Tensor:
    coefficient_count = sum(
        design_matrices[parameter].shape[1] for parameter in family.parameter_names
    )
    if initial_coefficients is not None:
        if (
            initial_coefficients.ndim != 1
            or initial_coefficients.numel() != coefficient_count
            or initial_coefficients.dtype != response.dtype
            or initial_coefficients.device != response.device
        ):
            raise ValueError(
                "initial_coefficients must match the complete coefficient vector"
            )
        if not torch.isfinite(initial_coefficients).all():
            raise ValueError("initial_coefficients must be finite")
        full = initial_coefficients
    else:
        initial_parameters = family.initial_parameters(response)
        full = response.new_empty(coefficient_count)
        for parameter in family.parameter_names:
            coefficient_slice = coefficient_slices[parameter]
            if coefficient_slice.start == coefficient_slice.stop:
                continue
            target = (
                family.links[parameter](initial_parameters[parameter])
                - offsets[parameter]
            )
            full[coefficient_slice] = (
                torch.linalg.pinv(design_matrices[parameter]) @ target
            )
    return transform.mT @ full


def _generalized_log_determinant(matrix: Tensor) -> tuple[Tensor, int]:
    eigenvalues = torch.linalg.eigvalsh(_symmetrize(matrix))
    tolerance = _matrix_tolerance(matrix)
    if float(eigenvalues.min()) < -tolerance:
        raise RuntimeError("combined penalty is not positive semidefinite")
    retained = eigenvalues[eigenvalues > tolerance]
    if retained.numel() == 0:
        return matrix.new_zeros(()), 0
    return retained.log().sum(), retained.numel()


def _symmetric_pseudoinverse(matrix: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetrize(matrix))
    tolerance = _matrix_tolerance(matrix)
    retained = eigenvalues > tolerance
    if not bool(retained.any()):
        return torch.zeros_like(matrix)
    vectors = eigenvectors[:, retained]
    return _symmetrize((vectors * eigenvalues[retained].reciprocal()) @ vectors.mT)


def _matrix_tolerance(matrix: Tensor) -> float:
    scale = max(float(matrix.detach().abs().max()), 1.0)
    return 100.0 * torch.finfo(matrix.dtype).eps * max(matrix.shape) * scale


def _null_directions(matrix: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetrize(matrix))
    tolerance = _matrix_tolerance(matrix)
    return eigenvectors[:, eigenvalues <= tolerance]


def _boundary_status(
    value: float,
    estimated: bool,
    lower: float,
    upper: float,
    tolerance: float,
) -> str:
    if not estimated:
        return "fixed"
    if value <= lower + tolerance:
        return "lower"
    if value >= upper - tolerance:
        return "upper"
    return "interior"


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.mT)
