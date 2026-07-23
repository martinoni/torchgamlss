"""Smooth terms for additive GAMLSS predictors."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class SmoothTerm(nn.Module, ABC):
    """Protocol for a penalized additive predictor term."""

    coefficients: nn.Parameter

    @abstractmethod
    def basis(self, covariate: Tensor) -> Tensor:
        """Return the term's design matrix for a one-dimensional covariate."""

    @abstractmethod
    def penalty_matrix(self) -> Tensor:
        """Return the square-root penalty matrix acting on the coefficients."""

    @property
    @abstractmethod
    def smoothing_parameter(self) -> float:
        """Return the fixed or most recently estimated smoothing parameter."""

    @property
    def estimates_smoothing_parameter(self) -> bool:
        """Whether RS fitting should update the smoothing parameter."""
        return False

    @property
    def smoothing_method(self) -> str | None:
        """Return the smoothing-parameter selection method, when enabled."""
        return None

    @property
    def target_effective_degrees_of_freedom(self) -> float | None:
        """Return the requested total EDF, when lambda is selected by EDF."""
        return None

    @property
    def criterion_penalty(self) -> float:
        """Return the GAIC/GCV effective-degrees-of-freedom multiplier."""
        return 2.0

    @property
    def penalty_nullity(self) -> int:
        """Return the dimension of the unpenalized coefficient subspace."""
        return self.coefficients.numel() - self.penalty_matrix().shape[0]

    def _set_fitted_smoothing_parameter(self, value: float) -> None:
        if value != self.smoothing_parameter:
            raise RuntimeError("This smooth term has a fixed smoothing parameter")

    def forward(self, covariate: Tensor) -> Tensor:
        return self.basis(covariate) @ self.coefficients

    def quadratic_penalty(self) -> Tensor:
        differences = self.penalty_matrix() @ self.coefficients
        return self.smoothing_parameter * differences.square().sum()

    def effective_degrees_of_freedom(
        self, covariate: Tensor, weights: Tensor
    ) -> Tensor:
        """Return trace of the penalized weighted-least-squares hat matrix."""
        basis = self.basis(covariate)
        if weights.ndim != 1 or weights.shape[0] != basis.shape[0]:
            raise ValueError("weights must have one value per smooth covariate")
        if weights.dtype != basis.dtype or weights.device != basis.device:
            raise ValueError("weights must match the smooth covariate dtype and device")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("at least one smooth weight must be positive")
        gram = basis.mT @ (weights.unsqueeze(-1) * basis)
        penalty = self.penalty_matrix()
        system = gram + self.smoothing_parameter * (penalty.mT @ penalty)
        return torch.trace(torch.linalg.pinv(system) @ gram)


class PSpline(SmoothTerm):
    """Eilers-Marx P-spline with an equally spaced B-spline basis.

    The basis, difference penalty, and ML, target-EDF, GAIC, and GCV smoothing
    selection modes follow ``gamlss::pb()``.
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        smoothing_parameter: float | None = None,
        *,
        degrees_of_freedom: float | None = None,
        initial_smoothing_parameter: float = 10.0,
        smoothing_method: str = "ML",
        criterion_penalty: float = 2.0,
        intervals: int = 20,
        degree: int = 3,
        penalty_order: int = 2,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
            raise ValueError("P-spline bounds must be finite")
        if not upper_bound > lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound")
        if smoothing_parameter is not None:
            if not math.isfinite(smoothing_parameter):
                raise ValueError("smoothing_parameter must be finite")
            if smoothing_parameter < 0:
                raise ValueError("smoothing_parameter must be non-negative")
        if smoothing_parameter is not None and degrees_of_freedom is not None:
            raise ValueError(
                "Specify either smoothing_parameter or degrees_of_freedom, not both"
            )
        if degrees_of_freedom is not None:
            if not math.isfinite(degrees_of_freedom):
                raise ValueError("degrees_of_freedom must be finite")
            if degrees_of_freedom < 0:
                raise ValueError("degrees_of_freedom must be non-negative")
        if not math.isfinite(initial_smoothing_parameter):
            raise ValueError("initial_smoothing_parameter must be finite")
        if initial_smoothing_parameter <= 0:
            raise ValueError("initial_smoothing_parameter must be positive")
        if smoothing_method not in {"ML", "GAIC", "GCV"}:
            raise ValueError("smoothing_method must be 'ML', 'GAIC', or 'GCV'")
        if not math.isfinite(criterion_penalty) or criterion_penalty <= 0:
            raise ValueError("criterion_penalty must be finite and positive")
        if intervals < 1:
            raise ValueError("intervals must be at least 1")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        if penalty_order < 0:
            raise ValueError("penalty_order must be non-negative")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("P-splines require a floating-point dtype")

        basis_size = intervals + degree
        if penalty_order >= basis_size:
            raise ValueError("penalty_order must be smaller than the basis size")
        if degrees_of_freedom is not None and degrees_of_freedom >= basis_size - 2:
            raise ValueError(
                "degrees_of_freedom must be smaller than basis size minus 2"
            )

        data_range = upper_bound - lower_bound
        left = lower_bound - 0.01 * data_range
        right = upper_bound + 0.01 * data_range
        spacing = (right - left) / intervals
        knot_count = intervals + 2 * degree + 1
        knots = (
            torch.arange(knot_count, dtype=dtype, device=device) * spacing
            + left
            - degree * spacing
        )
        basis_differences = torch.diff(
            torch.eye(knot_count, dtype=dtype, device=device),
            n=degree + 1,
            dim=0,
        ) / (math.gamma(degree + 1) * spacing**degree)
        penalty = (
            torch.eye(basis_size, dtype=dtype, device=device)
            if penalty_order == 0
            else torch.diff(
                torch.eye(basis_size, dtype=dtype, device=device),
                n=penalty_order,
                dim=0,
            )
        )

        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.intervals = intervals
        self.degree = degree
        self.penalty_order = penalty_order
        self._estimates_smoothing_parameter = smoothing_parameter is None
        self._smoothing_method = (
            "DF" if degrees_of_freedom is not None else smoothing_method
        )
        self._target_effective_degrees_of_freedom = (
            None if degrees_of_freedom is None else float(degrees_of_freedom + 2)
        )
        self._criterion_penalty = float(criterion_penalty)
        smoothing_value = (
            initial_smoothing_parameter
            if smoothing_parameter is None
            else smoothing_parameter
        )
        self.register_buffer("knots", knots)
        self.register_buffer("_basis_differences", basis_differences)
        self.register_buffer("_penalty", penalty)
        self.register_buffer(
            "_smoothing_parameter_value",
            torch.tensor(smoothing_value, dtype=dtype, device=device),
        )
        self.coefficients = nn.Parameter(
            torch.zeros(basis_size, dtype=dtype, device=device)
        )

    @classmethod
    def from_data(
        cls,
        covariate: Tensor,
        smoothing_parameter: float | None = None,
        *,
        degrees_of_freedom: float | None = None,
        initial_smoothing_parameter: float = 10.0,
        smoothing_method: str = "ML",
        criterion_penalty: float = 2.0,
        intervals: int = 20,
        degree: int = 3,
        penalty_order: int = 2,
    ) -> PSpline:
        """Construct a term using the range and interval rules of ``pb()``."""
        if intervals < 1:
            raise ValueError("intervals must be at least 1")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        if penalty_order < 0:
            raise ValueError("penalty_order must be non-negative")
        if covariate.ndim != 1 or covariate.numel() < 2:
            raise ValueError("covariate must be one-dimensional with at least 2 values")
        if not covariate.is_floating_point():
            raise ValueError("covariate must use a floating-point dtype")
        if not torch.isfinite(covariate).all():
            raise ValueError("covariate must be finite")
        distinct_values = torch.unique(covariate).numel()
        if distinct_values < 2:
            raise ValueError("covariate must contain at least 2 distinct values")

        # gamlss::pb() lowers the default basis size for small samples and then
        # caps it at the number of distinct covariate values.
        effective_intervals = 10 if covariate.numel() < 99 else intervals
        effective_intervals = min(effective_intervals, distinct_values)
        return cls(
            float(covariate.min()),
            float(covariate.max()),
            smoothing_parameter,
            degrees_of_freedom=degrees_of_freedom,
            initial_smoothing_parameter=initial_smoothing_parameter,
            smoothing_method=smoothing_method,
            criterion_penalty=criterion_penalty,
            intervals=effective_intervals,
            degree=degree,
            penalty_order=penalty_order,
            dtype=covariate.dtype,
            device=covariate.device,
        )

    @property
    def smoothing_parameter(self) -> float:
        return float(self._smoothing_parameter_value)

    @property
    def estimates_smoothing_parameter(self) -> bool:
        return self._estimates_smoothing_parameter

    @property
    def smoothing_method(self) -> str | None:
        return self._smoothing_method if self.estimates_smoothing_parameter else None

    @property
    def target_effective_degrees_of_freedom(self) -> float | None:
        return self._target_effective_degrees_of_freedom

    @property
    def criterion_penalty(self) -> float:
        return self._criterion_penalty

    def _set_fitted_smoothing_parameter(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "fitted smoothing parameter must be finite and non-negative"
            )
        if not self.estimates_smoothing_parameter and value != self.smoothing_parameter:
            raise RuntimeError("This P-spline has a fixed smoothing parameter")
        self._smoothing_parameter_value.fill_(value)

    def basis(self, covariate: Tensor) -> Tensor:
        if covariate.ndim != 1:
            raise ValueError("smooth covariate must be one-dimensional")
        if covariate.dtype != self.knots.dtype or covariate.device != self.knots.device:
            raise ValueError(
                "smooth covariate must match the P-spline dtype and device"
            )
        if not torch.isfinite(covariate).all():
            raise ValueError("smooth covariate must be finite")
        truncated = (covariate.unsqueeze(-1) - self.knots).clamp_min(0)
        powers = truncated.pow(self.degree)
        return ((-1) ** (self.degree + 1)) * (powers @ self._basis_differences.mT)

    def penalty_matrix(self) -> Tensor:
        return self._penalty
