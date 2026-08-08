"""Smooth terms for additive GAMLSS predictors."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


def row_tensor_product(marginal_designs: Sequence[Tensor]) -> Tensor:
    """Return the row-wise Kronecker product of marginal model matrices."""
    if isinstance(marginal_designs, Tensor):
        raise ValueError("marginal_designs must be a sequence of tensors")
    designs = tuple(marginal_designs)
    if not designs:
        raise ValueError("at least one marginal design is required")
    reference = designs[0]
    if (
        not isinstance(reference, Tensor)
        or reference.ndim != 2
        or reference.shape[0] < 1
        or reference.shape[1] < 1
    ):
        raise ValueError("marginal design 0 must be a non-empty matrix")
    if not reference.is_floating_point() or not torch.isfinite(reference).all():
        raise ValueError("marginal design 0 must be finite and floating-point")
    for index, design in enumerate(designs[1:], start=1):
        if (
            not isinstance(design, Tensor)
            or design.ndim != 2
            or design.shape[0] != reference.shape[0]
            or design.shape[1] < 1
        ):
            raise ValueError(
                f"marginal design {index} must have one row per observation"
            )
        if design.dtype != reference.dtype or design.device != reference.device:
            raise ValueError(
                f"marginal design {index} must match dtype and device"
            )
        if not torch.isfinite(design).all():
            raise ValueError(f"marginal design {index} must be finite")

    product = reference
    for design in designs[1:]:
        product = torch.einsum(
            "ni,nj->nij",
            product,
            design,
        ).reshape(reference.shape[0], -1)
    return product


def tensor_product_penalties(
    marginal_penalties: Sequence[Tensor],
) -> tuple[Tensor, ...]:
    """Embed one coefficient-space penalty for each tensor margin."""
    if isinstance(marginal_penalties, Tensor):
        raise ValueError("marginal_penalties must be a sequence of tensors")
    penalties = tuple(marginal_penalties)
    if not penalties:
        raise ValueError("at least one marginal penalty is required")
    reference = penalties[0]
    sizes: list[int] = []
    for index, penalty in enumerate(penalties):
        if (
            not isinstance(penalty, Tensor)
            or penalty.ndim != 2
            or penalty.shape[0] < 1
            or penalty.shape[0] != penalty.shape[1]
        ):
            raise ValueError(f"marginal penalty {index} must be square")
        if not penalty.is_floating_point() or not torch.isfinite(penalty).all():
            raise ValueError(
                f"marginal penalty {index} must be finite and floating-point"
            )
        if penalty.dtype != reference.dtype or penalty.device != reference.device:
            raise ValueError(
                f"marginal penalty {index} must match dtype and device"
            )
        tolerance = (
            100.0
            * torch.finfo(penalty.dtype).eps
            * max(penalty.shape)
            * max(float(penalty.detach().abs().max()), 1.0)
        )
        if float((penalty - penalty.mT).detach().abs().max()) > tolerance:
            raise ValueError(f"marginal penalty {index} must be symmetric")
        sizes.append(penalty.shape[0])

    embedded: list[Tensor] = []
    for index, penalty in enumerate(penalties):
        left_size = math.prod(sizes[:index])
        right_size = math.prod(sizes[index + 1 :])
        value = penalty
        if left_size > 1:
            value = torch.kron(
                torch.eye(
                    left_size,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                value,
            )
        if right_size > 1:
            value = torch.kron(
                value,
                torch.eye(
                    right_size,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
            )
        embedded.append(0.5 * (value + value.mT))
    return tuple(embedded)


def _thin_plate_polynomial_powers(
    dimension: int,
    penalty_order: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the mgcv ordering of the thin-plate penalty null space."""
    count = math.comb(penalty_order + dimension - 1, dimension)
    index = [0] * dimension
    powers: list[tuple[int, ...]] = []
    for _ in range(count):
        powers.append(tuple(index))
        total = sum(index)
        if total < penalty_order - 1:
            index[0] += 1
            continue
        total -= index[0]
        index[0] = 0
        for coordinate in range(1, dimension):
            index[coordinate] += 1
            total += 1
            if total == penalty_order:
                total -= index[coordinate]
                index[coordinate] = 0
            else:
                break
    return tuple(powers)


def _thin_plate_polynomial_basis(
    covariates: Tensor,
    powers: Sequence[Sequence[int]],
) -> Tensor:
    columns: list[Tensor] = []
    for exponents in powers:
        column = torch.ones(
            covariates.shape[0],
            dtype=covariates.dtype,
            device=covariates.device,
        )
        for coordinate, exponent in enumerate(exponents):
            if exponent:
                column = column * covariates[:, coordinate].pow(exponent)
        columns.append(column)
    return torch.stack(columns, dim=1)


def _thin_plate_kernel(
    left: Tensor,
    right: Tensor,
    penalty_order: int,
) -> Tensor:
    """Evaluate Wood's thin-plate radial basis kernel."""
    dimension = left.shape[1]
    squared_distance = torch.cdist(left, right).square()
    half_dimension = dimension // 2
    if dimension % 2 == 0:
        constant = (-1.0) ** (penalty_order + 1 + half_dimension)
        constant /= 2.0 ** (2 * penalty_order - 1)
        constant /= math.pi ** (dimension / 2)
        constant /= math.factorial(penalty_order - 1)
        constant /= math.factorial(penalty_order - half_dimension)
        safe_distance = squared_distance.clamp_min(
            torch.finfo(squared_distance.dtype).tiny
        )
        values = (
            constant
            * 0.5
            * torch.log(safe_distance)
            * squared_distance.pow(penalty_order - half_dimension)
        )
        return torch.where(
            squared_distance > 0,
            values,
            torch.zeros_like(values),
        )

    constant = math.gamma(dimension / 2 - penalty_order)
    constant /= 2.0 ** (2 * penalty_order)
    constant /= math.pi ** (dimension / 2)
    constant /= math.factorial(penalty_order - 1)
    return constant * squared_distance.pow(penalty_order - dimension / 2)


def _mgcv_pspline_basis_state(
    covariate: Tensor,
    basis_size: int,
    spline_order: int,
) -> tuple[Tensor, Tensor, float, float, int]:
    """Construct the equally spaced B-spline state used by ``mgcv::ps``."""
    lower = float(covariate.min())
    upper = float(covariate.max())
    data_range = upper - lower
    lower_boundary = lower - 0.001 * data_range
    upper_boundary = upper + 0.001 * data_range
    interval_count = basis_size - spline_order
    spacing = (upper_boundary - lower_boundary) / (interval_count - 1)
    degree = spline_order + 1
    knot_count = interval_count + 2 * spline_order + 2
    knots = (
        torch.arange(
            knot_count,
            dtype=covariate.dtype,
            device=covariate.device,
        )
        * spacing
        + lower_boundary
        - (spline_order + 1) * spacing
    )
    basis_differences = torch.diff(
        torch.eye(
            knot_count,
            dtype=covariate.dtype,
            device=covariate.device,
        ),
        n=degree + 1,
        dim=0,
    ) / (math.gamma(degree + 1) * spacing**degree)
    return (
        knots,
        basis_differences,
        lower_boundary,
        upper_boundary,
        degree,
    )


def _mgcv_pspline_basis(
    covariate: Tensor,
    knots: Tensor,
    basis_differences: Tensor,
    lower_boundary: float,
    upper_boundary: float,
    degree: int,
) -> Tensor:
    """Evaluate an ``mgcv::ps`` basis with linear boundary extrapolation."""

    def evaluate(values: Tensor, *, derivative: bool = False) -> Tensor:
        truncated = (values.unsqueeze(-1) - knots).clamp_min(0)
        power = degree - 1 if derivative else degree
        multiplier = degree if derivative else 1.0
        return (
            ((-1) ** (degree + 1))
            * multiplier
            * (truncated.pow(power) @ basis_differences.mT)
        )

    basis = evaluate(covariate)
    below = covariate < lower_boundary
    above = covariate > upper_boundary
    if bool(below.any()):
        boundary = covariate.new_tensor([lower_boundary])
        lower_extrapolation = evaluate(boundary) + (
            (covariate - lower_boundary).unsqueeze(1)
            * evaluate(boundary, derivative=True)
        )
        basis = torch.where(below.unsqueeze(1), lower_extrapolation, basis)
    if bool(above.any()):
        boundary = covariate.new_tensor([upper_boundary])
        upper_extrapolation = evaluate(boundary) + (
            (covariate - upper_boundary).unsqueeze(1)
            * evaluate(boundary, derivative=True)
        )
        basis = torch.where(above.unsqueeze(1), upper_extrapolation, basis)
    return basis


def _adaptive_difference_operators_2d(
    first_size: int,
    second_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Translate the coefficient-grid operators used by ``mgcv::D2``."""
    coefficient_count = first_size * second_size
    row_indices = torch.arange(first_size, device=device)
    column_indices = torch.arange(second_size, device=device)

    rr_i = row_indices[1:-1].repeat(second_size)
    rr_j = column_indices.repeat_interleave(first_size - 2)
    rr_center = rr_j * first_size + rr_i
    drr = torch.zeros(
        (rr_center.numel(), coefficient_count),
        dtype=dtype,
        device=device,
    )
    rr_rows = torch.arange(rr_center.numel(), device=device)
    drr[rr_rows, rr_center - 1] = 1.0
    drr[rr_rows, rr_center] = -2.0
    drr[rr_rows, rr_center + 1] = 1.0

    cc_i = row_indices.repeat(second_size - 2)
    cc_j = column_indices[1:-1].repeat_interleave(first_size)
    cc_center = cc_j * first_size + cc_i
    dcc = torch.zeros(
        (cc_center.numel(), coefficient_count),
        dtype=dtype,
        device=device,
    )
    cc_rows = torch.arange(cc_center.numel(), device=device)
    dcc[cc_rows, cc_center - first_size] = 1.0
    dcc[cc_rows, cc_center] = -2.0
    dcc[cc_rows, cc_center + first_size] = 1.0

    cr_i = row_indices[1:-1].repeat(second_size - 2)
    cr_j = column_indices[1:-1].repeat_interleave(first_size - 2)
    cr_center = cr_j * first_size + cr_i
    dcr = torch.zeros(
        (cr_center.numel(), coefficient_count),
        dtype=dtype,
        device=device,
    )
    cr_rows = torch.arange(cr_center.numel(), device=device)
    mixed_scale = math.sqrt(0.125)
    dcr[cr_rows, cr_center - first_size - 1] = mixed_scale
    dcr[cr_rows, cr_center + first_size + 1] = mixed_scale
    dcr[cr_rows, cr_center + first_size - 1] = -mixed_scale
    dcr[cr_rows, cr_center - first_size + 1] = -mixed_scale

    rr_coordinates = torch.column_stack((rr_i + 1, rr_j + 1)).to(dtype)
    cc_coordinates = torch.column_stack((cc_i + 1, cc_j + 1)).to(dtype)
    cr_coordinates = torch.column_stack((cr_i + 1, cr_j + 1)).to(dtype)
    return drr, dcc, dcr, rr_coordinates, cc_coordinates, cr_coordinates


class SmoothTerm(nn.Module, ABC):
    """Protocol for a penalized additive predictor term.

    ``basis()`` and ``penalty_matrix()`` are the original scalar-penalty
    interface retained for GAMLSS ``pb()`` compatibility. The generic
    ``design()``, ``penalty_matrices()``, ``constraints()``, and
    ``predict_design()`` methods expose the basis--penalty representation
    needed by multidimensional and multiply penalized smooths.
    """

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
        """Whether classical fitting should update the smoothing parameter."""
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

    def design(self, covariates: Tensor) -> Tensor:
        """Return the training design for the supplied smooth covariates.

        This generic name deliberately does not imply one-dimensional input.
        Existing smooths delegate to ``basis()``; future multivariate terms can
        specialize the representation after the classical fitter consumes this
        interface directly.
        """
        return self.basis(covariates)

    def predict_design(self, new_covariates: Tensor) -> Tensor:
        """Return the design mapping used for out-of-sample prediction."""
        return self.design(new_covariates)

    def penalty_matrices(self) -> tuple[Tensor, ...]:
        """Return unscaled positive-semidefinite coefficient penalties.

        The legacy ``penalty_matrix()`` is a square-root penalty ``D``. The
        corresponding coefficient-space matrix is ``S = D.T @ D``. Returning
        a tuple establishes the representation required for terms with
        ``sum_j lambda_j S_j`` penalties while preserving scalar ``pb()``
        behavior.
        """
        penalty_root = self.penalty_matrix()
        return (penalty_root.mT @ penalty_root,)

    @property
    def smoothing_parameters(self) -> tuple[float, ...]:
        """Return one smoothing parameter for each coefficient penalty."""
        return (self.smoothing_parameter,)

    @property
    def estimated_smoothing_parameters(self) -> tuple[bool, ...]:
        """Return one LAML-selection flag for each coefficient penalty."""
        return (self.estimates_smoothing_parameter,)

    def constraints(self, covariates: Tensor) -> Tensor:
        """Return coefficient constraints ``C`` for ``C @ beta = 0``.

        Classical ``pb()`` compatibility currently uses no explicit
        constraint. Smooths requiring centering or point constraints can
        override this method and will be fitted through a null-space
        reparameterization in a later vertical slice.
        """
        design = self.design(covariates)
        return design.new_empty((0, self.coefficients.numel()))

    @property
    def penalty_nullity(self) -> int:
        """Return the dimension of the unpenalized coefficient subspace."""
        return self.coefficients.numel() - self.penalty_matrix().shape[0]

    @property
    def linear_design_overlap(self) -> int:
        """Return smooth EDF dimensions represented by the linear design."""
        return self.penalty_nullity if len(self.penalty_matrices()) == 1 else 0

    def _set_fitted_smoothing_parameter(self, value: float) -> None:
        if value != self.smoothing_parameter:
            raise RuntimeError("This smooth term has a fixed smoothing parameter")

    def _set_fitted_smoothing_parameters(
        self,
        values: Sequence[float],
    ) -> None:
        """Store smoothing parameters selected by a whole-model method."""
        normalized = tuple(values)
        if len(normalized) != 1:
            raise ValueError(
                "single-penalty smooth requires one smoothing parameter"
            )
        self._set_fitted_smoothing_parameter(float(normalized[0]))

    def forward(self, covariate: Tensor) -> Tensor:
        return self.design(covariate) @ self.coefficients

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


class CyclicPSpline(PSpline):
    """Periodic Eilers-Marx P-spline compatible with ``gamlss::pbc()``."""

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
        sine_penalty: bool = True,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        SmoothTerm.__init__(self)
        if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
            raise ValueError("cyclic P-spline bounds must be finite")
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
        if intervals <= degree:
            raise ValueError("intervals must be greater than degree")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        if penalty_order not in {1, 2}:
            raise ValueError("cyclic penalty_order must be 1 or 2")
        if not isinstance(sine_penalty, bool):
            raise ValueError("sine_penalty must be boolean")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("cyclic P-splines require a floating-point dtype")
        if degrees_of_freedom is not None and degrees_of_freedom >= intervals - 2:
            raise ValueError(
                "degrees_of_freedom must be smaller than intervals minus 2"
            )

        spacing = (upper_bound - lower_bound) / intervals
        knot_count = intervals + 2 * degree + 1
        knots = (
            torch.arange(knot_count, dtype=dtype, device=device) * spacing
            + lower_bound
            - degree * spacing
        )
        basis_differences = torch.diff(
            torch.eye(knot_count, dtype=dtype, device=device),
            n=degree + 1,
            dim=0,
        ) / (math.gamma(degree + 1) * spacing**degree)
        identity = torch.eye(intervals, dtype=dtype, device=device)
        if penalty_order == 1:
            penalty = torch.roll(identity, shifts=-1, dims=1) - identity
        else:
            diagonal = (
                2.0 * math.cos(2.0 * math.pi / intervals)
                if sine_penalty
                else 2.0
            )
            penalty = (
                diagonal * identity
                - torch.roll(identity, shifts=1, dims=1)
                - torch.roll(identity, shifts=-1, dims=1)
            )

        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.intervals = intervals
        self.degree = degree
        self.penalty_order = penalty_order
        self.sine_penalty = sine_penalty
        self._estimates_smoothing_parameter = smoothing_parameter is None
        self._smoothing_method = (
            "DF" if degrees_of_freedom is not None else smoothing_method
        )
        self._target_effective_degrees_of_freedom = (
            None if degrees_of_freedom is None else float(degrees_of_freedom + 1)
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
            torch.zeros(intervals, dtype=dtype, device=device)
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
        sine_penalty: bool = True,
    ) -> CyclicPSpline:
        """Construct a cyclic term using the range rules of ``pbc()``."""
        if covariate.ndim != 1 or covariate.numel() < 2:
            raise ValueError("covariate must be one-dimensional with at least 2 values")
        if not covariate.is_floating_point():
            raise ValueError("covariate must use a floating-point dtype")
        if not torch.isfinite(covariate).all():
            raise ValueError("covariate must be finite")
        distinct_values = torch.unique(covariate).numel()
        if distinct_values < 2:
            raise ValueError("covariate must contain at least 2 distinct values")
        effective_intervals = 10 if covariate.numel() < 100 else intervals
        effective_intervals = min(effective_intervals, distinct_values)
        if effective_intervals <= degree:
            raise ValueError(
                "cyclic covariate must provide more distinct values than degree"
            )
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
            sine_penalty=sine_penalty,
            dtype=covariate.dtype,
            device=covariate.device,
        )

    def basis(self, covariate: Tensor) -> Tensor:
        if covariate.ndim != 1:
            raise ValueError("smooth covariate must be one-dimensional")
        if covariate.dtype != self.knots.dtype or covariate.device != self.knots.device:
            raise ValueError(
                "smooth covariate must match the cyclic P-spline dtype and device"
            )
        if not torch.isfinite(covariate).all():
            raise ValueError("smooth covariate must be finite")
        period = self.upper_bound - self.lower_bound
        wrapped = torch.remainder(covariate - self.lower_bound, period)
        wrapped = wrapped + self.lower_bound
        within_bounds = (covariate >= self.lower_bound) & (
            covariate <= self.upper_bound
        )
        wrapped = torch.where(within_bounds, covariate, wrapped)
        truncated = (wrapped.unsqueeze(-1) - self.knots).clamp_min(0)
        powers = truncated.pow(self.degree)
        full_basis = ((-1) ** (self.degree + 1)) * (
            powers @ self._basis_differences.mT
        )
        basis = full_basis[:, : self.intervals].clone()
        basis[:, : self.degree] += full_basis[:, self.intervals :]
        return basis

    @property
    def penalty_nullity(self) -> int:
        if self.penalty_order == 1 or not self.sine_penalty:
            return 1
        return 2

    @property
    def linear_design_overlap(self) -> int:
        return 1


class ShrinkagePSpline(PSpline):
    """Double-penalty P-spline compatible with ``gamlss::pbz()``."""

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        smoothing_parameters: Sequence[float] = (1e-4, 1e-4),
        *,
        estimate_smoothing: bool = True,
        intervals: int = 20,
        degree: int = 3,
        penalty_order: int = 2,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        values = tuple(smoothing_parameters)
        if len(values) != 2:
            raise ValueError("pbz() requires two smoothing parameters")
        if not isinstance(estimate_smoothing, bool):
            raise ValueError("estimate_smoothing must be boolean")
        normalized: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool):
                raise ValueError(
                    f"smoothing parameter {index} must be finite and non-negative"
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"smoothing parameter {index} must be finite and non-negative"
                )
            if estimate_smoothing and numeric <= 0:
                raise ValueError(
                    "estimated smoothing parameters must be strictly positive"
                )
            normalized.append(numeric)
        if penalty_order < 2:
            raise ValueError("pbz() penalty_order must be at least 2")

        super().__init__(
            lower_bound,
            upper_bound,
            None if estimate_smoothing else normalized[0],
            initial_smoothing_parameter=normalized[0],
            intervals=intervals,
            degree=degree,
            penalty_order=penalty_order,
            dtype=dtype,
            device=device,
        )
        first_difference = torch.diff(
            torch.eye(
                self.coefficients.numel(),
                dtype=dtype,
                device=device,
            ),
            dim=0,
        )
        self.register_buffer("_shrinkage_penalty", first_difference)
        self.register_buffer(
            "_shrinkage_smoothing_parameter_value",
            torch.tensor(normalized[1], dtype=dtype, device=device),
        )

    @classmethod
    def from_data(
        cls,
        covariate: Tensor,
        smoothing_parameters: Sequence[float] = (1e-4, 1e-4),
        *,
        estimate_smoothing: bool = True,
        intervals: int = 20,
        degree: int = 3,
        penalty_order: int = 2,
    ) -> ShrinkagePSpline:
        """Construct a shrinkage term using the range rules of ``pbz()``."""
        if covariate.ndim != 1 or covariate.numel() < 2:
            raise ValueError("covariate must be one-dimensional with at least 2 values")
        if not covariate.is_floating_point():
            raise ValueError("covariate must use a floating-point dtype")
        if not torch.isfinite(covariate).all():
            raise ValueError("covariate must be finite")
        distinct_values = torch.unique(covariate).numel()
        if distinct_values < 2:
            raise ValueError("covariate must contain at least 2 distinct values")
        effective_intervals = 10 if covariate.numel() < 99 else intervals
        effective_intervals = min(effective_intervals, distinct_values)
        return cls(
            float(covariate.min()),
            float(covariate.max()),
            smoothing_parameters,
            estimate_smoothing=estimate_smoothing,
            intervals=effective_intervals,
            degree=degree,
            penalty_order=penalty_order,
            dtype=covariate.dtype,
            device=covariate.device,
        )

    def penalty_matrices(self) -> tuple[Tensor, ...]:
        return (
            self._penalty.mT @ self._penalty,
            self._shrinkage_penalty.mT @ self._shrinkage_penalty,
        )

    @property
    def smoothing_parameters(self) -> tuple[float, ...]:
        return (
            self.smoothing_parameter,
            float(self._shrinkage_smoothing_parameter_value),
        )

    @property
    def estimated_smoothing_parameters(self) -> tuple[bool, ...]:
        return (self.estimates_smoothing_parameter,) * 2

    @property
    def smoothing_method(self) -> str | None:
        return None

    def _set_fitted_smoothing_parameters(
        self,
        values: Sequence[float],
    ) -> None:
        normalized = tuple(float(value) for value in values)
        if len(normalized) != 2:
            raise ValueError("pbz() requires two fitted smoothing parameters")
        if any(not math.isfinite(value) or value < 0 for value in normalized):
            raise ValueError(
                "fitted smoothing parameters must be finite and non-negative"
            )
        if not self.estimates_smoothing_parameter and normalized != (
            self.smoothing_parameters
        ):
            raise RuntimeError("This shrinkage P-spline has fixed smoothing parameters")
        self._smoothing_parameter_value.fill_(normalized[0])
        self._shrinkage_smoothing_parameter_value.fill_(normalized[1])

    def constraints(self, covariates: Tensor) -> Tensor:
        return self.design(covariates).sum(dim=0, keepdim=True)

    @property
    def penalty_nullity(self) -> int:
        return 1

    @property
    def linear_design_overlap(self) -> int:
        return 0

    def quadratic_penalty(self) -> Tensor:
        return sum(
            smoothing_parameter
            * torch.dot(
                self.coefficients,
                penalty @ self.coefficients,
            )
            for smoothing_parameter, penalty in zip(
                self.smoothing_parameters,
                self.penalty_matrices(),
                strict=True,
            )
        )


class AdaptivePSpline(SmoothTerm):
    """One-dimensional adaptive P-spline compatible with ``mgcv::ad``."""

    def __init__(
        self,
        training_covariate: Tensor,
        smoothing_parameters: Sequence[float] | None = None,
        *,
        initial_smoothing_parameters: Sequence[float] | None = None,
        basis_size: int = 40,
        penalty_basis_size: int = 5,
    ) -> None:
        super().__init__()
        if (
            not isinstance(training_covariate, Tensor)
            or training_covariate.ndim != 1
            or training_covariate.numel() < 2
        ):
            raise ValueError(
                "adaptive training covariate must be a one-dimensional tensor "
                "with at least two values"
            )
        if (
            training_covariate.dtype not in {torch.float32, torch.float64}
            or not torch.isfinite(training_covariate).all()
        ):
            raise ValueError(
                "adaptive training covariate must be finite float32 or float64"
            )
        if torch.unique(training_covariate).numel() < 2:
            raise ValueError(
                "adaptive training covariate must contain distinct values"
            )
        if isinstance(basis_size, bool) or not isinstance(basis_size, int):
            raise ValueError("basis_size must be an integer")
        if basis_size < 5:
            raise ValueError("basis_size must be at least 5")
        if (
            isinstance(penalty_basis_size, bool)
            or not isinstance(penalty_basis_size, int)
        ):
            raise ValueError("penalty_basis_size must be an integer")
        if penalty_basis_size < 2:
            raise ValueError("penalty_basis_size must be at least 2")
        if penalty_basis_size >= basis_size - 2:
            raise ValueError(
                "penalty_basis_size must be smaller than basis_size minus 2"
            )
        if (
            smoothing_parameters is not None
            and initial_smoothing_parameters is not None
        ):
            raise ValueError(
                "specify either smoothing_parameters or "
                "initial_smoothing_parameters, not both"
            )
        estimates_smoothing = smoothing_parameters is None
        supplied = (
            initial_smoothing_parameters
            if estimates_smoothing and initial_smoothing_parameters is not None
            else smoothing_parameters
        )
        if supplied is None:
            supplied = (1.0,) * penalty_basis_size
        if isinstance(supplied, Tensor) or not isinstance(supplied, Sequence):
            raise ValueError("adaptive smoothing parameters must be a sequence")
        values = tuple(supplied)
        if len(values) != penalty_basis_size:
            raise ValueError(
                "adaptive smoothing parameters must have one value per "
                "penalty-basis column"
            )
        normalized: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool):
                raise ValueError(
                    f"adaptive smoothing parameter {index} must be numeric"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"adaptive smoothing parameter {index} must be numeric"
                ) from error
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"adaptive smoothing parameter {index} must be finite "
                    "and non-negative"
                )
            if estimates_smoothing and numeric <= 0:
                raise ValueError(
                    "automatic adaptive smoothing parameters must start positive"
                )
            normalized.append(numeric)

        (
            knots,
            basis_differences,
            lower_boundary,
            upper_boundary,
            degree,
        ) = _mgcv_pspline_basis_state(training_covariate, basis_size, 2)
        raw_training_design = _mgcv_pspline_basis(
            training_covariate,
            knots,
            basis_differences,
            lower_boundary,
            upper_boundary,
            degree,
        )
        roughness_root = torch.diff(
            torch.eye(
                basis_size,
                dtype=training_covariate.dtype,
                device=training_covariate.device,
            ),
            n=2,
            dim=0,
        )
        penalty_locations = torch.arange(
            1,
            basis_size - 1,
            dtype=training_covariate.dtype,
            device=training_covariate.device,
        ) / basis_size
        if penalty_basis_size == 2:
            penalty_basis = torch.column_stack(
                (torch.ones_like(penalty_locations), penalty_locations)
            )
        else:
            penalty_spline_order = 1 if penalty_basis_size == 3 else 2
            (
                penalty_knots,
                penalty_differences,
                penalty_lower,
                penalty_upper,
                penalty_degree,
            ) = _mgcv_pspline_basis_state(
                penalty_locations,
                penalty_basis_size,
                penalty_spline_order,
            )
            penalty_basis = _mgcv_pspline_basis(
                penalty_locations,
                penalty_knots,
                penalty_differences,
                penalty_lower,
                penalty_upper,
                penalty_degree,
            )

        raw_penalties = tuple(
            roughness_root.mT
            @ (penalty_basis[:, index].unsqueeze(1) * roughness_root)
            for index in range(penalty_basis_size)
        )
        _, _, constraint_right_vectors = torch.linalg.svd(
            raw_training_design.sum(dim=0, keepdim=True),
            full_matrices=True,
        )
        centering_transform = constraint_right_vectors.mT[:, 1:]
        penalties: list[Tensor] = []
        penalty_roots: list[Tensor] = []
        for raw_penalty in raw_penalties:
            penalty = centering_transform.mT @ raw_penalty @ centering_transform
            penalty = 0.5 * (penalty + penalty.mT)
            eigenvalues, eigenvectors = torch.linalg.eigh(penalty)
            tolerance = (
                1000.0
                * torch.finfo(training_covariate.dtype).eps
                * max(penalty.shape)
                * max(float(penalty.detach().abs().max()), 1.0)
            )
            if float(eigenvalues.detach().min()) < -tolerance:
                raise RuntimeError(
                    "adaptive construction produced a non-PSD penalty"
                )
            penalties.append(penalty)
            penalty_roots.append(
                eigenvalues.clamp_min(0).sqrt().unsqueeze(1) * eigenvectors.mT
            )

        self.basis_size = basis_size
        self.penalty_basis_size = penalty_basis_size
        self.degree = degree
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self._estimates_smoothing_parameter = estimates_smoothing
        self.register_buffer("knots", knots)
        self.register_buffer("_basis_differences", basis_differences)
        self.register_buffer("_centering_transform", centering_transform)
        self.register_buffer("_penalty_basis", penalty_basis)
        self.register_buffer(
            "_penalties",
            torch.stack(penalties),
        )
        self.register_buffer(
            "_penalty_roots",
            torch.stack(penalty_roots),
        )
        self.register_buffer(
            "_smoothing_parameter_values",
            torch.tensor(
                normalized,
                dtype=training_covariate.dtype,
                device=training_covariate.device,
            ),
        )
        self.coefficients = nn.Parameter(
            torch.zeros(
                basis_size - 1,
                dtype=training_covariate.dtype,
                device=training_covariate.device,
            )
        )

    @classmethod
    def from_data(
        cls,
        covariate: Tensor,
        smoothing_parameters: Sequence[float] | None = None,
        *,
        initial_smoothing_parameters: Sequence[float] | None = None,
        basis_size: int = 40,
        penalty_basis_size: int = 5,
    ) -> AdaptivePSpline:
        """Construct an adaptive P-spline from its training covariate."""
        return cls(
            covariate,
            smoothing_parameters,
            initial_smoothing_parameters=initial_smoothing_parameters,
            basis_size=basis_size,
            penalty_basis_size=penalty_basis_size,
        )

    @property
    def smoothing_parameter(self) -> float:
        return float(self._smoothing_parameter_values[0])

    @property
    def smoothing_parameters(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._smoothing_parameter_values)

    @property
    def estimates_smoothing_parameter(self) -> bool:
        return self._estimates_smoothing_parameter

    @property
    def estimated_smoothing_parameters(self) -> tuple[bool, ...]:
        return (self.estimates_smoothing_parameter,) * self.penalty_basis_size

    @property
    def smoothing_method(self) -> str | None:
        return None

    def _set_fitted_smoothing_parameters(
        self,
        values: Sequence[float],
    ) -> None:
        normalized = tuple(float(value) for value in values)
        if len(normalized) != self.penalty_basis_size:
            raise ValueError(
                "adaptive smooth requires one fitted lambda per penalty"
            )
        if any(not math.isfinite(value) or value < 0 for value in normalized):
            raise ValueError(
                "fitted smoothing parameters must be finite and non-negative"
            )
        if (
            not self.estimates_smoothing_parameter
            and normalized != self.smoothing_parameters
        ):
            raise RuntimeError("This adaptive P-spline has fixed lambdas")
        self._smoothing_parameter_values.copy_(
            self._smoothing_parameter_values.new_tensor(normalized)
        )

    def basis(self, covariate: Tensor) -> Tensor:
        if not isinstance(covariate, Tensor) or covariate.ndim != 1:
            raise ValueError("adaptive smooth covariate must be one-dimensional")
        if (
            covariate.dtype != self.knots.dtype
            or covariate.device != self.knots.device
            or not torch.isfinite(covariate).all()
        ):
            raise ValueError(
                "adaptive smooth covariate must be finite and match the term "
                "dtype and device"
            )
        raw_basis = _mgcv_pspline_basis(
            covariate,
            self.knots,
            self._basis_differences,
            self.lower_boundary,
            self.upper_boundary,
            self.degree,
        )
        return raw_basis @ self._centering_transform

    def penalty_matrix(self) -> Tensor:
        return self._penalty_roots[0]

    def penalty_matrices(self) -> tuple[Tensor, ...]:
        return tuple(self._penalties.unbind(dim=0))

    def constraints(self, covariates: Tensor) -> Tensor:
        design = self.design(covariates)
        return design.new_empty((0, design.shape[1]))

    @property
    def penalty_nullity(self) -> int:
        return 1

    @property
    def linear_design_overlap(self) -> int:
        return 0

    def effective_degrees_of_freedom(
        self,
        covariate: Tensor,
        weights: Tensor,
    ) -> Tensor:
        basis = self.design(covariate)
        if weights.ndim != 1 or weights.shape[0] != basis.shape[0]:
            raise ValueError("weights must have one value per smooth covariate")
        if weights.dtype != basis.dtype or weights.device != basis.device:
            raise ValueError(
                "weights must match the smooth covariate dtype and device"
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("at least one smooth weight must be positive")
        gram = basis.mT @ (weights.unsqueeze(1) * basis)
        combined_penalty = sum(
            smoothing_parameter * penalty
            for smoothing_parameter, penalty in zip(
                self.smoothing_parameters,
                self.penalty_matrices(),
                strict=True,
            )
        )
        return torch.trace(torch.linalg.pinv(gram + combined_penalty) @ gram)

    def quadratic_penalty(self) -> Tensor:
        return sum(
            smoothing_parameter
            * torch.dot(
                self.coefficients,
                penalty @ self.coefficients,
            )
            for smoothing_parameter, penalty in zip(
                self.smoothing_parameters,
                self.penalty_matrices(),
                strict=True,
            )
        )


class AdaptiveTensorPSpline(AdaptivePSpline):
    """Two-dimensional adaptive tensor P-spline compatible with ``mgcv``."""

    def __init__(
        self,
        training_covariates: Tensor,
        smoothing_parameters: Sequence[float] | None = None,
        *,
        initial_smoothing_parameters: Sequence[float] | None = None,
        basis_size: int | Sequence[int] = 15,
        penalty_basis_size: int | Sequence[int] = 3,
    ) -> None:
        SmoothTerm.__init__(self)
        if (
            not isinstance(training_covariates, Tensor)
            or training_covariates.ndim != 2
            or training_covariates.shape[0] < 2
            or training_covariates.shape[1] != 2
        ):
            raise ValueError(
                "adaptive tensor training covariates must have shape (n, 2)"
            )
        if (
            training_covariates.dtype not in {torch.float32, torch.float64}
            or not torch.isfinite(training_covariates).all()
        ):
            raise ValueError(
                "adaptive tensor training covariates must be finite float32 "
                "or float64"
            )
        if any(
            torch.unique(training_covariates[:, index]).numel() < 2
            for index in range(2)
        ):
            raise ValueError(
                "each adaptive tensor covariate must contain distinct values"
            )

        basis_sizes = self._dimension_pair(basis_size, name="basis_size")
        if any(value < 5 for value in basis_sizes):
            raise ValueError("adaptive tensor basis sizes must be at least 5")
        penalty_basis_sizes = self._dimension_pair(
            penalty_basis_size,
            name="penalty_basis_size",
        )
        if any(value < 3 for value in penalty_basis_sizes):
            raise ValueError(
                "adaptive tensor penalty basis sizes must be at least 3"
            )
        penalty_count = math.prod(penalty_basis_sizes)
        difference_count = math.prod(value - 2 for value in basis_sizes)
        if penalty_count > difference_count:
            raise ValueError(
                "adaptive tensor penalty basis is too large for the main basis"
            )
        if (
            smoothing_parameters is not None
            and initial_smoothing_parameters is not None
        ):
            raise ValueError(
                "specify either smoothing_parameters or "
                "initial_smoothing_parameters, not both"
            )
        estimates_smoothing = smoothing_parameters is None
        supplied = (
            initial_smoothing_parameters
            if estimates_smoothing and initial_smoothing_parameters is not None
            else smoothing_parameters
        )
        if supplied is None:
            supplied = (1.0,) * penalty_count
        if isinstance(supplied, Tensor) or not isinstance(supplied, Sequence):
            raise ValueError(
                "adaptive tensor smoothing parameters must be a sequence"
            )
        values = tuple(supplied)
        if len(values) != penalty_count:
            raise ValueError(
                "adaptive tensor smoothing parameters must have one value "
                "per penalty-basis column"
            )
        normalized: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool):
                raise ValueError(
                    f"adaptive tensor smoothing parameter {index} must be numeric"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"adaptive tensor smoothing parameter {index} must be numeric"
                ) from error
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"adaptive tensor smoothing parameter {index} must be "
                    "finite and non-negative"
                )
            if estimates_smoothing and numeric <= 0:
                raise ValueError(
                    "automatic adaptive tensor smoothing parameters must "
                    "start positive"
                )
            normalized.append(numeric)

        main_states = tuple(
            _mgcv_pspline_basis_state(
                training_covariates[:, index],
                basis_sizes[index],
                2,
            )
            for index in range(2)
        )
        main_designs = tuple(
            _mgcv_pspline_basis(
                training_covariates[:, index],
                *main_states[index],
            )
            for index in range(2)
        )
        raw_training_design = row_tensor_product(main_designs)
        (
            drr,
            dcc,
            dcr,
            rr_coordinates,
            cc_coordinates,
            cr_coordinates,
        ) = _adaptive_difference_operators_2d(
            basis_sizes[0],
            basis_sizes[1],
            dtype=training_covariates.dtype,
            device=training_covariates.device,
        )
        penalty_spline_order = min(min(penalty_basis_sizes) - 2, 1)
        penalty_locations = tuple(
            torch.arange(
                1,
                basis_sizes[index] + 1,
                dtype=training_covariates.dtype,
                device=training_covariates.device,
            )
            for index in range(2)
        )
        penalty_states = tuple(
            _mgcv_pspline_basis_state(
                penalty_locations[index],
                penalty_basis_sizes[index],
                penalty_spline_order,
            )
            for index in range(2)
        )

        def penalty_basis(coordinates: Tensor) -> Tensor:
            return row_tensor_product(
                tuple(
                    _mgcv_pspline_basis(
                        coordinates[:, index],
                        *penalty_states[index],
                    )
                    for index in range(2)
                )
            )

        vrr = penalty_basis(rr_coordinates)
        vcc = penalty_basis(cc_coordinates)
        vcr = penalty_basis(cr_coordinates)
        raw_penalties = tuple(
            drr.mT @ (vrr[:, index].unsqueeze(1) * drr)
            + dcc.mT @ (vcc[:, index].unsqueeze(1) * dcc)
            + dcr.mT @ (vcr[:, index].unsqueeze(1) * dcr)
            for index in range(penalty_count)
        )
        centering_transform = _right_null_space(
            raw_training_design.sum(dim=0, keepdim=True)
        )
        penalties: list[Tensor] = []
        penalty_roots: list[Tensor] = []
        for raw_penalty in raw_penalties:
            penalty = centering_transform.mT @ raw_penalty @ centering_transform
            penalty = 0.5 * (penalty + penalty.mT)
            eigenvalues, eigenvectors = torch.linalg.eigh(penalty)
            tolerance = (
                1000.0
                * torch.finfo(training_covariates.dtype).eps
                * max(penalty.shape)
                * max(float(penalty.detach().abs().max()), 1.0)
            )
            if float(eigenvalues.detach().min()) < -tolerance:
                raise RuntimeError(
                    "adaptive tensor construction produced a non-PSD penalty"
                )
            penalties.append(penalty)
            penalty_roots.append(
                eigenvalues.clamp_min(0).sqrt().unsqueeze(1) * eigenvectors.mT
            )

        self.dimension = 2
        self.basis_size = basis_sizes
        self.basis_sizes = basis_sizes
        self.raw_coefficient_shape = basis_sizes
        self.penalty_basis_size = penalty_count
        self.penalty_basis_shape = penalty_basis_sizes
        self.degree = (main_states[0][4], main_states[1][4])
        self.lower_boundary = (main_states[0][2], main_states[1][2])
        self.upper_boundary = (main_states[0][3], main_states[1][3])
        self._estimates_smoothing_parameter = estimates_smoothing
        for index, state in enumerate(main_states):
            self.register_buffer(f"_knots_{index}", state[0])
            self.register_buffer(f"_basis_differences_{index}", state[1])
        self.register_buffer("_centering_transform", centering_transform)
        self.register_buffer("_penalties", torch.stack(penalties))
        self.register_buffer("_penalty_roots", torch.stack(penalty_roots))
        self.register_buffer(
            "_smoothing_parameter_values",
            torch.tensor(
                normalized,
                dtype=training_covariates.dtype,
                device=training_covariates.device,
            ),
        )
        self.coefficients = nn.Parameter(
            torch.zeros(
                math.prod(basis_sizes) - 1,
                dtype=training_covariates.dtype,
                device=training_covariates.device,
            )
        )

    @staticmethod
    def _dimension_pair(
        value: int | Sequence[int],
        *,
        name: str,
    ) -> tuple[int, int]:
        if isinstance(value, int):
            values = (value, value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            values = tuple(value)
        else:
            raise ValueError(f"{name} must be an integer or two integers")
        if (
            len(values) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in values
            )
        ):
            raise ValueError(f"{name} must be an integer or two integers")
        return values

    @classmethod
    def from_data(
        cls,
        covariates: Tensor,
        smoothing_parameters: Sequence[float] | None = None,
        *,
        initial_smoothing_parameters: Sequence[float] | None = None,
        basis_size: int | Sequence[int] = 15,
        penalty_basis_size: int | Sequence[int] = 3,
    ) -> AdaptiveTensorPSpline:
        """Construct a two-dimensional adaptive smooth from training data."""
        return cls(
            covariates,
            smoothing_parameters,
            initial_smoothing_parameters=initial_smoothing_parameters,
            basis_size=basis_size,
            penalty_basis_size=penalty_basis_size,
        )

    def basis(self, covariates: Tensor) -> Tensor:
        if (
            not isinstance(covariates, Tensor)
            or covariates.ndim != 2
            or covariates.shape[1] != 2
        ):
            raise ValueError("adaptive tensor covariates must have shape (n, 2)")
        if (
            covariates.dtype != self.coefficients.dtype
            or covariates.device != self.coefficients.device
            or not torch.isfinite(covariates).all()
        ):
            raise ValueError(
                "adaptive tensor covariates must be finite and match the term "
                "dtype and device"
            )
        marginal_designs = tuple(
            _mgcv_pspline_basis(
                covariates[:, index],
                getattr(self, f"_knots_{index}"),
                getattr(self, f"_basis_differences_{index}"),
                self.lower_boundary[index],
                self.upper_boundary[index],
                self.degree[index],
            )
            for index in range(2)
        )
        return row_tensor_product(marginal_designs) @ self._centering_transform

    @property
    def penalty_nullity(self) -> int:
        combined = self._penalties.sum(dim=0)
        rank = int(torch.linalg.matrix_rank(combined).detach())
        return self.coefficients.numel() - rank


def _polygon_coordinate_tensor(value: object, *, context: str) -> Tensor:
    """Convert one polygon ring to a finite CPU float64 coordinate matrix."""
    try:
        coordinates = (
            value.detach().to(dtype=torch.float64, device="cpu")
            if isinstance(value, Tensor)
            else torch.as_tensor(value, dtype=torch.float64)
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{context} must contain numeric coordinate pairs") from error
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{context} must have shape (vertices, 2)")
    if not torch.isfinite(coordinates).all():
        raise ValueError(f"{context} coordinates must be finite")
    if torch.unique(coordinates, dim=0).shape[0] < 3:
        raise ValueError(f"{context} must contain at least three unique vertices")
    return coordinates


def _polygon_point_set(value: object, *, context: str) -> set[tuple[float, float]]:
    """Normalize one polygon or a sequence of polygon components."""
    try:
        coordinates = (
            value.detach().to(dtype=torch.float64, device="cpu")
            if isinstance(value, Tensor)
            else torch.as_tensor(value, dtype=torch.float64)
        )
    except (TypeError, ValueError, RuntimeError):
        coordinates = None
    if coordinates is not None and coordinates.ndim == 2:
        rings = (_polygon_coordinate_tensor(coordinates, context=context),)
    elif coordinates is not None and coordinates.ndim == 3:
        rings = tuple(
            _polygon_coordinate_tensor(
                coordinates[index],
                context=f"{context} component {index}",
            )
            for index in range(coordinates.shape[0])
        )
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{context} must contain polygon coordinates")
        try:
            components = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                f"{context} must contain polygon coordinates"
            ) from error
        if not components:
            raise ValueError(f"{context} must contain at least one component")
        rings = tuple(
            _polygon_coordinate_tensor(
                component,
                context=f"{context} component {index}",
            )
            for index, component in enumerate(components)
        )
    points: set[tuple[float, float]] = set()
    for ring in rings:
        points.update((float(row[0]), float(row[1])) for row in ring)
    return points


def polygon_adjacency(
    polygons: Mapping[object, object],
) -> dict[object, tuple[object, ...]]:
    """Return ``mgcv::pol2nb`` adjacency from exactly shared polygon vertices.

    Mapping keys establish the region order. A value may be one ``(n, 2)``
    coordinate sequence or a sequence of such components. As in ``mgcv``, any
    exactly shared vertex makes two regions neighbors; touching only at a
    corner therefore counts as adjacency.
    """
    if not isinstance(polygons, Mapping) or len(polygons) < 2:
        raise ValueError("polygons must map at least two region labels")
    labels = tuple(polygons)
    point_sets: list[set[tuple[float, float]]] = []
    seen_labels: set[object] = set()
    for index, label in enumerate(labels):
        if isinstance(label, bool):
            raise ValueError(f"polygon region label {index} must not be boolean")
        try:
            hash(label)
        except TypeError as error:
            raise ValueError(
                f"polygon region label {index} must be hashable"
            ) from error
        if label in seen_labels:
            raise ValueError("polygon region labels must be unique")
        seen_labels.add(label)
        point_sets.append(
            _polygon_point_set(
                polygons[label],
                context=f"polygon {label!r}",
            )
        )
    neighbors: dict[object, list[object]] = {label: [] for label in labels}
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            if point_sets[first].intersection(point_sets[second]):
                neighbors[labels[first]].append(labels[second])
                neighbors[labels[second]].append(labels[first])
    return {label: tuple(values) for label, values in neighbors.items()}


class MarkovRandomField(SmoothTerm):
    """Full- or low-rank areal graph smoother compatible with ``mgcv::mrf``."""

    def __init__(
        self,
        training_regions: Tensor,
        edges: (
            Sequence[Sequence[object]]
            | Mapping[object, Sequence[object]]
            | None
        ) = None,
        smoothing_parameter: float | None = None,
        *,
        initial_smoothing_parameter: float = 1.0,
        area_count: int | None = None,
        basis_size: int | None = None,
        region_labels: Sequence[object] | None = None,
        graph_penalty: Tensor | Sequence[Sequence[float]] | None = None,
        polygons: Mapping[object, object] | None = None,
    ) -> None:
        super().__init__()
        spatial_source_count = sum(
            source is not None for source in (edges, graph_penalty, polygons)
        )
        if spatial_source_count != 1:
            raise ValueError(
                "MRF requires exactly one of edges, graph_penalty, or polygons"
            )
        polygon_neighbors: dict[object, tuple[object, ...]] | None = None
        if polygons is not None:
            polygon_neighbors = polygon_adjacency(polygons)
            edges = polygon_neighbors
            if region_labels is None:
                region_labels = tuple(polygons)
        if (
            not isinstance(training_regions, Tensor)
            or training_regions.ndim != 1
            or training_regions.numel() < 2
            or training_regions.dtype not in {torch.float32, torch.float64}
            or not torch.isfinite(training_regions).all()
        ):
            raise ValueError(
                "MRF training regions must be a finite one-dimensional "
                "float32 or float64 tensor with at least two values"
            )
        rounded_regions = training_regions.round()
        if not torch.equal(training_regions, rounded_regions):
            raise ValueError("MRF region codes must be integers")
        if (rounded_regions < 0).any():
            raise ValueError("MRF region codes must be non-negative")
        inferred_area_count = int(rounded_regions.detach().max()) + 1
        supplied_penalty: Tensor | None = None
        if graph_penalty is not None:
            try:
                supplied_penalty = torch.as_tensor(
                    graph_penalty,
                    dtype=training_regions.dtype,
                    device=training_regions.device,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    "graph_penalty must be a finite numeric matrix"
                ) from error
            if (
                supplied_penalty.ndim != 2
                or supplied_penalty.shape[0] < 2
                or supplied_penalty.shape[0] != supplied_penalty.shape[1]
            ):
                raise ValueError("graph_penalty must be a square matrix")
            if not torch.isfinite(supplied_penalty).all():
                raise ValueError("graph_penalty must be finite")
        normalized_labels: tuple[object, ...] | None = None
        if region_labels is not None:
            if (
                isinstance(region_labels, (str, bytes, Tensor))
                or not isinstance(region_labels, Sequence)
            ):
                raise ValueError("region_labels must be a sequence of labels")
            normalized_labels = tuple(region_labels)
            if len(normalized_labels) < 2:
                raise ValueError("region_labels must contain at least two labels")
            label_codes: dict[object, int] = {}
            for index, label in enumerate(normalized_labels):
                if isinstance(label, bool):
                    raise ValueError(
                        f"MRF region label {index} must not be boolean"
                    )
                try:
                    hash(label)
                except TypeError as error:
                    raise ValueError(
                        f"MRF region label {index} must be hashable"
                    ) from error
                if isinstance(label, float) and not math.isfinite(label):
                    raise ValueError(
                        f"MRF region label {index} must be finite"
                    )
                if label in label_codes:
                    raise ValueError("MRF region labels must be unique")
                label_codes[label] = index
        if area_count is None:
            area_count = (
                len(normalized_labels)
                if normalized_labels is not None
                else (
                    supplied_penalty.shape[0]
                    if supplied_penalty is not None
                    else inferred_area_count
                )
            )
        if (
            isinstance(area_count, bool)
            or not isinstance(area_count, int)
            or area_count < 2
        ):
            raise ValueError("area_count must be an integer of at least 2")
        if inferred_area_count > area_count:
            raise ValueError("MRF training region code exceeds area_count")
        if (
            supplied_penalty is not None
            and supplied_penalty.shape != (area_count, area_count)
        ):
            raise ValueError("graph_penalty dimension must equal area_count")
        if (
            normalized_labels is not None
            and len(normalized_labels) != area_count
        ):
            raise ValueError(
                "region_labels must contain exactly area_count labels"
            )
        if normalized_labels is None:
            normalized_labels = tuple(range(area_count))
            label_codes = {
                label: index for index, label in enumerate(normalized_labels)
            }
        if basis_size is None:
            basis_size = area_count
        if (
            isinstance(basis_size, bool)
            or not isinstance(basis_size, int)
            or not 2 <= basis_size <= area_count
        ):
            raise ValueError(
                "basis_size must be between 2 and area_count inclusive"
            )
        edge_pairs: list[tuple[object, object]] = []
        if isinstance(edges, Mapping):
            for first, neighbors in edges.items():
                if (
                    isinstance(neighbors, (str, bytes, Tensor))
                    or not isinstance(neighbors, Sequence)
                ):
                    raise ValueError(
                        "MRF adjacency values must be sequences of region labels"
                    )
                edge_pairs.extend((first, second) for second in neighbors)
        elif edges is not None and (
            isinstance(edges, Tensor) or not isinstance(edges, Sequence)
        ):
            raise ValueError(
                "MRF edges must be pairs or an adjacency mapping"
            )
        elif edges is not None:
            for index, edge in enumerate(edges):
                if (
                    isinstance(edge, (str, bytes))
                    or not isinstance(edge, Sequence)
                    or len(edge) != 2
                ):
                    raise ValueError(
                        f"MRF edge {index} must contain two regions"
                    )
                edge_pairs.append((edge[0], edge[1]))
        normalized_edges: set[tuple[int, int]] = set()
        for index, (first_label, second_label) in enumerate(edge_pairs):
            if isinstance(first_label, bool) or isinstance(second_label, bool):
                raise ValueError(
                    f"MRF edge {index} contains an unknown region"
                )
            try:
                first = label_codes[first_label]
                second = label_codes[second_label]
            except (KeyError, TypeError):
                raise ValueError(f"MRF edge {index} contains an unknown region")
            if first == second:
                raise ValueError(f"MRF edge {index} is a self-loop")
            normalized_edges.add(
                (first, second) if first < second else (second, first)
            )
        if edges is not None and not normalized_edges:
            raise ValueError("MRF requires at least one adjacency edge")
        if smoothing_parameter is not None:
            if (
                isinstance(smoothing_parameter, bool)
                or not math.isfinite(float(smoothing_parameter))
                or float(smoothing_parameter) < 0
            ):
                raise ValueError(
                    "smoothing_parameter must be finite and non-negative"
                )
        if (
            isinstance(initial_smoothing_parameter, bool)
            or not math.isfinite(float(initial_smoothing_parameter))
            or float(initial_smoothing_parameter) <= 0
        ):
            raise ValueError(
                "initial_smoothing_parameter must be finite and positive"
            )

        raw_design = (
            rounded_regions.to(torch.long).unsqueeze(1)
            == torch.arange(
                area_count,
                device=training_regions.device,
            ).unsqueeze(0)
        ).to(training_regions.dtype)
        if supplied_penalty is None:
            raw_penalty = torch.zeros(
                (area_count, area_count),
                dtype=training_regions.dtype,
                device=training_regions.device,
            )
            for first, second in sorted(normalized_edges):
                raw_penalty[first, first] += 1.0
                raw_penalty[second, second] += 1.0
                raw_penalty[first, second] -= 1.0
                raw_penalty[second, first] -= 1.0
        else:
            raw_penalty = supplied_penalty.clone()
        raw_penalty_scale = max(
            float(raw_penalty.detach().abs().max()),
            1.0,
        )
        raw_penalty_tolerance = (
            1000.0
            * torch.finfo(training_regions.dtype).eps
            * area_count
            * raw_penalty_scale
        )
        if not torch.allclose(
            raw_penalty,
            raw_penalty.mT,
            rtol=0.0,
            atol=raw_penalty_tolerance,
        ):
            raise ValueError("graph_penalty must be symmetric")
        raw_penalty = 0.5 * (raw_penalty + raw_penalty.mT)
        raw_eigenvalues = torch.linalg.eigvalsh(raw_penalty)
        raw_psd_tolerance = (
            torch.finfo(training_regions.dtype).eps**0.8
            * raw_penalty_scale
        )
        if float(raw_eigenvalues.detach().min()) < -raw_psd_tolerance:
            raise ValueError("graph_penalty must be positive semidefinite")
        basis_transform = torch.eye(
            area_count,
            dtype=training_regions.dtype,
            device=training_regions.device,
        )
        basis_penalty = raw_penalty
        training_design = raw_design
        if basis_size < area_count:
            missing_regions = torch.nonzero(
                raw_design.sum(dim=0) == 0,
                as_tuple=False,
            ).flatten()
            augmented_design = raw_design
            if missing_regions.numel():
                dummy_design = torch.zeros(
                    (missing_regions.numel(), area_count),
                    dtype=training_regions.dtype,
                    device=training_regions.device,
                )
                dummy_design[
                    torch.arange(missing_regions.numel(), device=raw_design.device),
                    missing_regions,
                ] = 1.0
                augmented_design = torch.cat(
                    (dummy_design, raw_design),
                    dim=0,
                )
            _, triangular = torch.linalg.qr(
                augmented_design,
                mode="reduced",
            )
            triangular_inverse = torch.linalg.inv(triangular)
            natural_penalty = (
                triangular_inverse.mT
                @ raw_penalty
                @ triangular_inverse
            )
            natural_penalty = 0.5 * (
                natural_penalty + natural_penalty.mT
            )
            natural_values, natural_vectors = torch.linalg.eigh(
                natural_penalty
            )
            natural_tolerance = (
                torch.finfo(training_regions.dtype).eps**0.8
                * max(float(natural_values.detach().abs().max()), 1.0)
            )
            if float(natural_values.detach().min()) < -natural_tolerance:
                raise RuntimeError(
                    "MRF natural parameterization produced a non-PSD penalty"
                )
            boundary_tolerance = (
                1000.0
                * torch.finfo(training_regions.dtype).eps
                * area_count
                * max(float(natural_values.detach().abs().max()), 1.0)
            )
            if (
                basis_size < area_count
                and abs(
                    float(
                        natural_values[basis_size]
                        - natural_values[basis_size - 1]
                    )
                )
                <= boundary_tolerance
            ):
                raise ValueError(
                    "basis_size cuts through a repeated MRF eigenvalue; "
                    "choose a size that retains the complete eigenspace"
                )
            scaling = math.sqrt(augmented_design.shape[0])
            basis_transform = (
                triangular_inverse
                @ natural_vectors[:, :basis_size]
                * scaling
            )
            training_design = raw_design @ basis_transform
            basis_penalty = torch.diag(
                natural_values[:basis_size].clamp_min(0)
                * augmented_design.shape[0]
            )
        centering_transform = _right_null_space(
            training_design.sum(dim=0, keepdim=True)
        )
        penalty = centering_transform.mT @ basis_penalty @ centering_transform
        penalty = 0.5 * (penalty + penalty.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(penalty)
        tolerance = (
            1000.0
            * torch.finfo(training_regions.dtype).eps
            * max(penalty.shape)
            * max(float(penalty.detach().abs().max()), 1.0)
        )
        if float(eigenvalues.detach().min()) < -tolerance:
            raise RuntimeError("MRF construction produced a non-PSD penalty")

        self.area_count = area_count
        self.basis_size = basis_size
        self.edges = tuple(sorted(normalized_edges))
        self.region_labels = normalized_labels
        self.label_edges = tuple(
            (normalized_labels[first], normalized_labels[second])
            for first, second in self.edges
        )
        self.uses_supplied_penalty = supplied_penalty is not None
        self.uses_polygon_adjacency = polygon_neighbors is not None
        self.polygon_neighbors = polygon_neighbors
        self._region_label_codes = label_codes
        self._estimates_smoothing_parameter = smoothing_parameter is None
        smoothing_value = (
            initial_smoothing_parameter
            if smoothing_parameter is None
            else smoothing_parameter
        )
        self.register_buffer("_basis_transform", basis_transform)
        self.register_buffer("_centering_transform", centering_transform)
        self.register_buffer("_penalty", penalty)
        self.register_buffer(
            "_penalty_root",
            eigenvalues.clamp_min(0).sqrt().unsqueeze(1) * eigenvectors.mT,
        )
        self.register_buffer(
            "_smoothing_parameter_value",
            torch.tensor(
                float(smoothing_value),
                dtype=training_regions.dtype,
                device=training_regions.device,
            ),
        )
        self.coefficients = nn.Parameter(
            torch.zeros(
                basis_size - 1,
                dtype=training_regions.dtype,
                device=training_regions.device,
            )
        )

    @classmethod
    def from_data(
        cls,
        regions: Tensor,
        edges: (
            Sequence[Sequence[object]]
            | Mapping[object, Sequence[object]]
            | None
        ) = None,
        smoothing_parameter: float | None = None,
        *,
        initial_smoothing_parameter: float = 1.0,
        area_count: int | None = None,
        basis_size: int | None = None,
        region_labels: Sequence[object] | None = None,
        graph_penalty: Tensor | Sequence[Sequence[float]] | None = None,
        polygons: Mapping[object, object] | None = None,
    ) -> MarkovRandomField:
        """Construct an MRF from integer codes and graph spatial information."""
        return cls(
            regions,
            edges,
            smoothing_parameter,
            initial_smoothing_parameter=initial_smoothing_parameter,
            area_count=area_count,
            basis_size=basis_size,
            region_labels=region_labels,
            graph_penalty=graph_penalty,
            polygons=polygons,
        )

    @classmethod
    def from_labels(
        cls,
        regions: Sequence[object],
        adjacency: (
            Sequence[Sequence[object]]
            | Mapping[object, Sequence[object]]
            | None
        ) = None,
        smoothing_parameter: float | None = None,
        *,
        initial_smoothing_parameter: float = 1.0,
        levels: Sequence[object] | None = None,
        basis_size: int | None = None,
        graph_penalty: Tensor | Sequence[Sequence[float]] | None = None,
        polygons: Mapping[object, object] | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> MarkovRandomField:
        """Construct an MRF from categorical labels and named adjacency."""
        if isinstance(regions, (str, bytes, Tensor)):
            raise ValueError("MRF labelled regions must be a one-dimensional sequence")
        try:
            region_values = tuple(regions)
        except TypeError as error:
            raise ValueError(
                "MRF labelled regions must be a one-dimensional sequence"
            ) from error
        if len(region_values) < 2:
            raise ValueError("MRF labelled regions must contain at least two values")

        spatial_source_count = sum(
            source is not None
            for source in (adjacency, graph_penalty, polygons)
        )
        if spatial_source_count == 0:
            raise ValueError(
                "MRF requires adjacency, graph_penalty, or polygons"
            )
        if spatial_source_count > 1:
            raise ValueError(
                "MRF accepts exactly one of adjacency, graph_penalty, or polygons"
            )
        if graph_penalty is not None and levels is None:
            raise ValueError(
                "MRF levels are required with a supplied graph_penalty"
            )

        adjacency_for_levels = (
            polygon_adjacency(polygons)
            if polygons is not None
            else adjacency
        )
        if levels is None:
            inferred_labels: list[object] = []
            inferred_codes: dict[object, int] = {}

            def add_label(label: object) -> None:
                try:
                    hash(label)
                except TypeError as error:
                    raise ValueError("MRF region labels must be hashable") from error
                if label not in inferred_codes:
                    inferred_codes[label] = len(inferred_labels)
                    inferred_labels.append(label)

            if isinstance(adjacency_for_levels, Mapping):
                for first in adjacency_for_levels:
                    add_label(first)
                for neighbors in adjacency_for_levels.values():
                    if isinstance(neighbors, (str, bytes, Tensor)):
                        raise ValueError(
                            "MRF adjacency values must be sequences of region labels"
                        )
                    try:
                        tuple(neighbors)
                    except TypeError as error:
                        raise ValueError(
                            "MRF adjacency values must be sequences of region labels"
                        ) from error
            else:
                assert adjacency_for_levels is not None
                if isinstance(adjacency_for_levels, (str, bytes, Tensor)):
                    raise ValueError(
                        "MRF adjacency must contain region pairs or named neighbors"
                    )
                try:
                    for edge in adjacency_for_levels:
                        if (
                            isinstance(edge, (str, bytes))
                            or not isinstance(edge, Sequence)
                            or len(edge) != 2
                        ):
                            raise ValueError(
                                "MRF adjacency must contain two-region pairs"
                            )
                        add_label(edge[0])
                        add_label(edge[1])
                except TypeError as error:
                    raise ValueError(
                        "MRF adjacency must contain region pairs or named neighbors"
                    ) from error
                for label in region_values:
                    add_label(label)
            normalized_levels = tuple(inferred_labels)
        else:
            if isinstance(levels, (str, bytes, Tensor)):
                raise ValueError("MRF levels must be a sequence of labels")
            try:
                normalized_levels = tuple(levels)
            except TypeError as error:
                raise ValueError("MRF levels must be a sequence of labels") from error

        label_codes: dict[object, int] = {}
        for index, label in enumerate(normalized_levels):
            try:
                hash(label)
            except TypeError as error:
                raise ValueError(f"MRF level {index} must be hashable") from error
            if label in label_codes:
                raise ValueError("MRF levels must be unique")
            label_codes[label] = index
        try:
            encoded_values = [label_codes[label] for label in region_values]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "MRF observations contain an unknown region label"
            ) from error
        encoded_regions = torch.tensor(
            encoded_values,
            dtype=dtype,
            device=device,
        )
        return cls(
            encoded_regions,
            adjacency,
            smoothing_parameter,
            initial_smoothing_parameter=initial_smoothing_parameter,
            area_count=len(normalized_levels),
            basis_size=basis_size,
            region_labels=normalized_levels,
            graph_penalty=graph_penalty,
            polygons=polygons,
        )

    @classmethod
    def from_polygons(
        cls,
        regions: Sequence[object],
        polygons: Mapping[object, object],
        smoothing_parameter: float | None = None,
        *,
        initial_smoothing_parameter: float = 1.0,
        levels: Sequence[object] | None = None,
        basis_size: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> MarkovRandomField:
        """Construct an MRF whose neighbors share exact polygon vertices."""
        return cls.from_labels(
            regions,
            smoothing_parameter=smoothing_parameter,
            initial_smoothing_parameter=initial_smoothing_parameter,
            levels=levels,
            basis_size=basis_size,
            polygons=polygons,
            dtype=dtype,
            device=device,
        )

    def encode_regions(self, regions: Sequence[object]) -> Tensor:
        """Encode categorical region labels on the term dtype and device."""
        if isinstance(regions, (str, bytes, Tensor)):
            raise ValueError("MRF region labels must be a one-dimensional sequence")
        try:
            values = tuple(regions)
            encoded = [self._region_label_codes[label] for label in values]
        except (KeyError, TypeError) as error:
            raise ValueError("MRF regions contain an unknown region label") from error
        return self.coefficients.new_tensor(encoded)

    def basis_labels(self, regions: Sequence[object]) -> Tensor:
        """Return the MRF design matrix for categorical region labels."""
        return self.basis(self.encode_regions(regions))

    def predict_design_labels(self, regions: Sequence[object]) -> Tensor:
        """Return the prediction design for categorical region labels."""
        return self.basis_labels(regions)

    @property
    def smoothing_parameter(self) -> float:
        return float(self._smoothing_parameter_value)

    @property
    def estimates_smoothing_parameter(self) -> bool:
        return self._estimates_smoothing_parameter

    @property
    def smoothing_method(self) -> str | None:
        return "LAML" if self.estimates_smoothing_parameter else None

    def _set_fitted_smoothing_parameter(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "fitted smoothing parameter must be finite and non-negative"
            )
        if not self.estimates_smoothing_parameter and value != self.smoothing_parameter:
            raise RuntimeError("This MRF smooth has a fixed smoothing parameter")
        self._smoothing_parameter_value.fill_(value)

    def basis(self, regions: Tensor) -> Tensor:
        if (
            not isinstance(regions, Tensor)
            or regions.ndim != 1
            or regions.dtype != self.coefficients.dtype
            or regions.device != self.coefficients.device
            or not torch.isfinite(regions).all()
        ):
            raise ValueError(
                "MRF regions must be a finite one-dimensional tensor matching "
                "the term dtype and device"
            )
        rounded_regions = regions.round()
        if (
            not torch.equal(regions, rounded_regions)
            or (rounded_regions < 0).any()
            or (rounded_regions >= self.area_count).any()
        ):
            raise ValueError(
                "MRF region codes must be integers between zero and "
                "area_count minus one"
            )
        raw_design = (
            rounded_regions.to(torch.long).unsqueeze(1)
            == torch.arange(
                self.area_count,
                device=regions.device,
            ).unsqueeze(0)
        ).to(regions.dtype)
        return (
            raw_design
            @ self._basis_transform
            @ self._centering_transform
        )

    def penalty_matrix(self) -> Tensor:
        return self._penalty_root

    def penalty_matrices(self) -> tuple[Tensor, ...]:
        return (self._penalty,)

    def constraints(self, covariates: Tensor) -> Tensor:
        design = self.design(covariates)
        return design.new_empty((0, design.shape[1]))

    @property
    def penalty_nullity(self) -> int:
        rank = int(torch.linalg.matrix_rank(self._penalty).detach())
        return self.coefficients.numel() - rank

    @property
    def linear_design_overlap(self) -> int:
        return 0


class ThinPlateSpline(SmoothTerm):
    """Low-rank isotropic thin-plate regression spline.

    The construction follows ``mgcv``: retain the leading-magnitude
    eigenvectors of the full radial basis, impose the polynomial side
    conditions, preserve the penalty null space, scale design columns to unit
    root-mean-square size, and absorb the empirical sum-to-zero constraint.
    """

    def __init__(
        self,
        training_covariates: Tensor,
        smoothing_parameter: float | None = None,
        *,
        initial_smoothing_parameter: float = 1.0,
        basis_size: int | None = None,
        penalty_order: int = 0,
        max_knots: int = 500,
    ) -> None:
        super().__init__()
        covariates = self._as_covariate_matrix(
            training_covariates,
            expected_dimension=None,
            context="training covariates",
        )
        if covariates.shape[0] < 2:
            raise ValueError("training covariates must contain at least two rows")
        dimension = covariates.shape[1]
        if isinstance(penalty_order, bool) or not isinstance(penalty_order, int):
            raise ValueError("penalty_order must be an integer")
        if penalty_order < 0:
            raise ValueError("penalty_order must be non-negative")
        effective_order = penalty_order
        if effective_order == 0:
            effective_order = 1
            while 2 * effective_order < dimension + 2:
                effective_order += 1
        if 2 * effective_order <= dimension:
            raise ValueError("thin-plate splines require 2 * penalty_order > dimension")
        if smoothing_parameter is not None:
            if not math.isfinite(smoothing_parameter) or smoothing_parameter < 0:
                raise ValueError(
                    "smoothing_parameter must be finite and non-negative"
                )
        if (
            not math.isfinite(initial_smoothing_parameter)
            or initial_smoothing_parameter <= 0
        ):
            raise ValueError(
                "initial_smoothing_parameter must be finite and positive"
            )
        if isinstance(max_knots, bool) or not isinstance(max_knots, int):
            raise ValueError("max_knots must be an integer")
        if max_knots < 2:
            raise ValueError("max_knots must be at least 2")

        polynomial_powers = _thin_plate_polynomial_powers(
            dimension,
            effective_order,
        )
        nullity = len(polynomial_powers)
        if basis_size is None:
            default_extra = (8, 27, 100)[min(dimension, 3) - 1]
            effective_basis_size = nullity + default_extra
        else:
            if isinstance(basis_size, bool) or not isinstance(basis_size, int):
                raise ValueError("basis_size must be an integer")
            effective_basis_size = basis_size
        if effective_basis_size <= nullity:
            raise ValueError(
                "basis_size must exceed the thin-plate penalty nullity"
            )
        if max_knots < effective_basis_size:
            raise ValueError("max_knots must be at least basis_size")

        coordinate_shift = covariates.mean(dim=0)
        centered = covariates - coordinate_shift
        knots = torch.unique(centered, dim=0)
        if knots.shape[0] < effective_basis_size:
            raise ValueError(
                "thin-plate term has fewer unique covariate combinations "
                "than basis_size"
            )
        if knots.shape[0] > max_knots:
            indices = torch.linspace(
                0,
                knots.shape[0] - 1,
                max_knots,
                dtype=covariates.dtype,
                device=covariates.device,
            ).round().to(dtype=torch.long)
            knots = knots.index_select(0, indices)

        radial_penalty = _thin_plate_kernel(knots, knots, effective_order)
        eigenvalues, eigenvectors = torch.linalg.eigh(radial_penalty)
        retained = torch.argsort(eigenvalues.abs(), descending=True)[
            :effective_basis_size
        ]
        eigenvalues = eigenvalues[retained]
        eigenvectors = eigenvectors[:, retained]
        knot_polynomials = _thin_plate_polynomial_basis(
            knots,
            polynomial_powers,
        )
        side_conditions = knot_polynomials.mT @ eigenvectors
        _, singular_values, right_vectors = torch.linalg.svd(
            side_conditions,
            full_matrices=True,
        )
        side_tolerance = (
            100.0
            * torch.finfo(covariates.dtype).eps
            * max(side_conditions.shape)
            * max(float(singular_values.detach().max()), 1.0)
        )
        if singular_values.numel() < nullity or float(
            singular_values[-1].detach()
        ) <= side_tolerance:
            raise ValueError("thin-plate polynomial side conditions are rank deficient")
        radial_transform = right_vectors.mT[:, nullity:]

        transform = covariates.new_zeros(
            (knots.shape[0] + nullity, effective_basis_size)
        )
        penalized_size = effective_basis_size - nullity
        transform[: knots.shape[0], :penalized_size] = (
            eigenvectors @ radial_transform
        )
        transform[knots.shape[0] :, penalized_size:] = torch.eye(
            nullity,
            dtype=covariates.dtype,
            device=covariates.device,
        )
        raw_penalty = covariates.new_zeros(
            (effective_basis_size, effective_basis_size)
        )
        raw_penalty[:penalized_size, :penalized_size] = (
            radial_transform.mT
            @ torch.diag(eigenvalues)
            @ radial_transform
        )

        full_training_basis = torch.cat(
            (
                _thin_plate_kernel(centered, knots, effective_order),
                _thin_plate_polynomial_basis(centered, polynomial_powers),
            ),
            dim=1,
        )
        raw_design = full_training_basis @ transform
        column_scale = raw_design.square().mean(dim=0).sqrt()
        scale_tolerance = torch.finfo(covariates.dtype).eps ** 0.5
        if bool((column_scale <= scale_tolerance).any()):
            raise ValueError("thin-plate construction produced a degenerate column")
        raw_design = raw_design / column_scale
        transform = transform / column_scale
        raw_penalty = raw_penalty / column_scale[:, None]
        raw_penalty = raw_penalty / column_scale[None, :]

        centering_constraint = raw_design.sum(dim=0, keepdim=True)
        _, _, constraint_right_vectors = torch.linalg.svd(
            centering_constraint,
            full_matrices=True,
        )
        centering_transform = constraint_right_vectors.mT[:, 1:]
        transform = transform @ centering_transform
        penalty = centering_transform.mT @ raw_penalty @ centering_transform
        penalty = 0.5 * (penalty + penalty.mT)

        penalty_eigenvalues, penalty_eigenvectors = torch.linalg.eigh(penalty)
        penalty_scale = max(float(penalty.detach().abs().max()), 1.0)
        penalty_tolerance = (
            1000.0
            * torch.finfo(covariates.dtype).eps
            * max(penalty.shape)
            * penalty_scale
        )
        if float(penalty_eigenvalues.detach().min()) < -penalty_tolerance:
            raise RuntimeError("thin-plate construction produced a non-PSD penalty")
        penalty_root = (
            penalty_eigenvalues.clamp_min(0).sqrt().unsqueeze(1)
            * penalty_eigenvectors.mT
        )

        self.dimension = dimension
        self.basis_size = effective_basis_size
        self.penalty_order = effective_order
        self.max_knots = max_knots
        self._penalty_nullity = nullity - 1
        self._estimates_smoothing_parameter = smoothing_parameter is None
        smoothing_value = (
            initial_smoothing_parameter
            if smoothing_parameter is None
            else smoothing_parameter
        )
        self.polynomial_powers = polynomial_powers
        self.register_buffer("coordinate_shift", coordinate_shift)
        self.register_buffer("knots", knots)
        self.register_buffer("_basis_transform", transform)
        self.register_buffer("_penalty", penalty)
        self.register_buffer("_penalty_root", penalty_root)
        self.register_buffer(
            "_smoothing_parameter_value",
            torch.tensor(
                smoothing_value,
                dtype=covariates.dtype,
                device=covariates.device,
            ),
        )
        self.coefficients = nn.Parameter(
            torch.zeros(
                effective_basis_size - 1,
                dtype=covariates.dtype,
                device=covariates.device,
            )
        )

    @classmethod
    def from_data(
        cls,
        covariates: Tensor,
        smoothing_parameter: float | None = None,
        *,
        initial_smoothing_parameter: float = 1.0,
        basis_size: int | None = None,
        penalty_order: int = 0,
        max_knots: int = 500,
    ) -> ThinPlateSpline:
        """Construct a thin-plate basis from its training locations."""
        return cls(
            covariates,
            smoothing_parameter,
            initial_smoothing_parameter=initial_smoothing_parameter,
            basis_size=basis_size,
            penalty_order=penalty_order,
            max_knots=max_knots,
        )

    @staticmethod
    def _as_covariate_matrix(
        covariates: Tensor,
        *,
        expected_dimension: int | None,
        context: str,
    ) -> Tensor:
        if not isinstance(covariates, Tensor):
            raise ValueError(f"{context} must be a tensor")
        matrix = covariates.unsqueeze(1) if covariates.ndim == 1 else covariates
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(
                f"{context} must contain at least one row and one column"
            )
        if expected_dimension is not None and matrix.shape[1] != expected_dimension:
            raise ValueError(
                f"{context} must contain {expected_dimension} covariate columns"
            )
        if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
            raise ValueError(f"{context} must be finite and floating-point")
        return matrix

    @property
    def smoothing_parameter(self) -> float:
        return float(self._smoothing_parameter_value)

    @property
    def estimates_smoothing_parameter(self) -> bool:
        return self._estimates_smoothing_parameter

    @property
    def smoothing_method(self) -> str | None:
        return "LAML" if self.estimates_smoothing_parameter else None

    def _set_fitted_smoothing_parameter(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "fitted smoothing parameter must be finite and non-negative"
            )
        if not self.estimates_smoothing_parameter and value != self.smoothing_parameter:
            raise RuntimeError("This thin-plate spline has a fixed smoothing parameter")
        self._smoothing_parameter_value.fill_(value)

    def basis(self, covariates: Tensor) -> Tensor:
        matrix = self._as_covariate_matrix(
            covariates,
            expected_dimension=self.dimension,
            context="thin-plate covariates",
        )
        if (
            matrix.dtype != self.knots.dtype
            or matrix.device != self.knots.device
        ):
            raise ValueError(
                "thin-plate covariates must match the term dtype and device"
            )
        centered = matrix - self.coordinate_shift
        full_basis = torch.cat(
            (
                _thin_plate_kernel(centered, self.knots, self.penalty_order),
                _thin_plate_polynomial_basis(
                    centered,
                    self.polynomial_powers,
                ),
            ),
            dim=1,
        )
        return full_basis @ self._basis_transform

    def penalty_matrix(self) -> Tensor:
        return self._penalty_root

    def penalty_matrices(self) -> tuple[Tensor, ...]:
        return (self._penalty,)

    def constraints(self, covariates: Tensor) -> Tensor:
        design = self.design(covariates)
        return design.new_empty((0, design.shape[1]))

    @property
    def penalty_nullity(self) -> int:
        return self._penalty_nullity

    @property
    def linear_design_overlap(self) -> int:
        return 0


class _MarginalSmoothBasis(nn.Module):
    """Parameter-free copy of a marginal smooth's basis and penalty state."""

    def __init__(self, term: SmoothTerm) -> None:
        super().__init__()
        self.coefficient_count = term.coefficients.numel()
        self.term = copy.deepcopy(term)
        coefficients = self.term.coefficients.detach().clone()
        del self.term.coefficients
        self.term.register_buffer(
            "coefficients",
            coefficients,
            persistent=False,
        )

    def design(self, covariate: Tensor) -> Tensor:
        return self.term.design(covariate)

    def predict_design(self, covariate: Tensor) -> Tensor:
        return self.term.predict_design(covariate)

    def penalty(self) -> Tensor:
        return self.term.penalty_matrices()[0]


class TensorProductSmooth(SmoothTerm):
    """Tensor product of single-penalty marginal smooth bases.

    The design is the row-wise Kronecker product of the marginal designs.
    There is one embedded coefficient-space penalty per margin. With
    ``center=True``, the term exposes the global sum-to-zero constraint used
    to separate a full tensor smooth from the model intercept. Supplying
    ``training_covariates`` absorbs that constraint into the coefficient
    parametrization and stores the resulting prediction mapping.
    """

    def __init__(
        self,
        marginals: Sequence[SmoothTerm],
        *,
        smoothing_parameters: Sequence[float] | None = None,
        estimate_smoothing: Sequence[bool] | bool = False,
        center: bool = True,
        training_covariates: Tensor | None = None,
        _interaction_covariates: Tensor | None = None,
    ) -> None:
        super().__init__()
        if isinstance(marginals, SmoothTerm):
            raise ValueError("marginals must be a sequence of smooth terms")
        marginal_terms = tuple(marginals)
        if len(marginal_terms) < 2:
            raise ValueError("a tensor product requires at least two marginals")
        if any(not isinstance(term, SmoothTerm) for term in marginal_terms):
            raise ValueError("every marginal must be a SmoothTerm")
        if not isinstance(center, bool):
            raise ValueError("center must be a boolean")
        reference = marginal_terms[0].coefficients
        marginal_penalties: list[Tensor] = []
        default_smoothing_parameters: list[float] = []
        coefficient_counts: list[int] = []
        for index, term in enumerate(marginal_terms):
            if (
                term.coefficients.dtype != reference.dtype
                or term.coefficients.device != reference.device
            ):
                raise ValueError(
                    f"marginal {index} must match the first dtype and device"
                )
            penalties = term.penalty_matrices()
            if len(penalties) != 1:
                raise ValueError(
                    "tensor marginals must each expose exactly one penalty"
                )
            penalty = penalties[0]
            coefficient_count = term.coefficients.numel()
            if penalty.shape != (coefficient_count, coefficient_count):
                raise ValueError(
                    f"marginal {index} penalty has an invalid shape"
                )
            marginal_penalties.append(penalty.detach().clone())
            default_smoothing_parameters.append(term.smoothing_parameters[0])
            coefficient_counts.append(coefficient_count)

        values = (
            tuple(default_smoothing_parameters)
            if smoothing_parameters is None
            else tuple(smoothing_parameters)
        )
        if len(values) != len(marginal_terms):
            raise ValueError(
                "smoothing_parameters must have one value per margin"
            )
        normalized_values: list[float] = []
        for index, value in enumerate(values):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"smoothing parameter {index} must be scalar"
                ) from error
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(
                    f"smoothing parameter {index} must be finite and non-negative"
                )
            normalized_values.append(numeric_value)
        if isinstance(estimate_smoothing, bool):
            estimated_smoothing = (estimate_smoothing,) * len(values)
        else:
            estimated_smoothing = tuple(estimate_smoothing)
            if (
                len(estimated_smoothing) != len(values)
                or any(
                    not isinstance(value, bool)
                    for value in estimated_smoothing
                )
            ):
                raise ValueError(
                    "estimate_smoothing must contain one boolean per margin"
                )
        if any(
            estimated and value <= 0
            for estimated, value in zip(
                estimated_smoothing,
                normalized_values,
                strict=True,
            )
        ):
            raise ValueError(
                "estimated smoothing parameters must start from positive values"
            )

        self.marginals = nn.ModuleList(
            _MarginalSmoothBasis(term) for term in marginal_terms
        )
        self.marginal_coefficient_counts = tuple(coefficient_counts)
        self.center = center
        self.interaction = _interaction_covariates is not None
        transforms = self._build_marginal_transforms(
            _interaction_covariates,
            marginal_penalties,
            reference,
        )
        self._transform_names: list[str] = []
        for index, transform in enumerate(transforms):
            name = f"_marginal_transform_{index}"
            self.register_buffer(name, transform)
            self._transform_names.append(name)
        reduced_counts = tuple(transform.shape[1] for transform in transforms)
        self.coefficient_shape = reduced_counts
        raw_coefficient_count = math.prod(reduced_counts)
        coefficient_transform = torch.eye(
            raw_coefficient_count,
            dtype=reference.dtype,
            device=reference.device,
        )
        self.constraint_absorbed = training_covariates is not None and center
        if training_covariates is not None:
            _validate_tensor_covariates(
                training_covariates,
                len(marginal_terms),
                reference,
            )
            if center:
                training_design = row_tensor_product(
                    tuple(
                        marginal.design(training_covariates[:, index])
                        @ transforms[index]
                        for index, marginal in enumerate(self.marginals)
                    )
                )
                coefficient_transform = _right_null_space(
                    training_design.sum(dim=0, keepdim=True)
                )
        self.register_buffer(
            "_coefficient_transform",
            coefficient_transform,
        )
        self.coefficients = nn.Parameter(
            torch.zeros(
                coefficient_transform.shape[1],
                dtype=reference.dtype,
                device=reference.device,
            )
        )
        self.register_buffer(
            "_smoothing_parameter_values",
            torch.tensor(
                normalized_values,
                dtype=reference.dtype,
                device=reference.device,
            ),
        )
        self._estimated_smoothing_parameters = estimated_smoothing

    def _build_marginal_transforms(
        self,
        covariates: Tensor | None,
        penalties: Sequence[Tensor],
        reference: Tensor,
    ) -> tuple[Tensor, ...]:
        if covariates is None:
            return tuple(
                torch.eye(
                    penalty.shape[0],
                    dtype=reference.dtype,
                    device=reference.device,
                )
                for penalty in penalties
            )
        _validate_tensor_covariates(
            covariates,
            len(penalties),
            reference,
        )
        transforms = []
        for index, marginal in enumerate(self.marginals):
            design = marginal.design(covariates[:, index])
            constraint = design.sum(dim=0, keepdim=True)
            transforms.append(_right_null_space(constraint))
        return tuple(transforms)

    def _marginal_transform(self, index: int) -> Tensor:
        return getattr(self, self._transform_names[index])

    def marginal_designs(
        self,
        covariates: Tensor,
        *,
        prediction: bool = False,
    ) -> tuple[Tensor, ...]:
        """Return transformed marginal designs used by this tensor term."""
        _validate_tensor_covariates(
            covariates,
            len(self.marginals),
            self.coefficients,
        )
        designs = []
        for index, marginal in enumerate(self.marginals):
            covariate = covariates[:, index]
            design = (
                marginal.predict_design(covariate)
                if prediction
                else marginal.design(covariate)
            )
            designs.append(design @ self._marginal_transform(index))
        return tuple(designs)

    def basis(self, covariate: Tensor) -> Tensor:
        return self.design(covariate)

    def design(self, covariates: Tensor) -> Tensor:
        return (
            row_tensor_product(self.marginal_designs(covariates))
            @ self._coefficient_transform
        )

    def predict_design(self, new_covariates: Tensor) -> Tensor:
        return (
            row_tensor_product(
                self.marginal_designs(new_covariates, prediction=True)
            )
            @ self._coefficient_transform
        )

    def penalty_matrix(self) -> Tensor:
        raise RuntimeError(
            "tensor smooths have multiple penalties; use penalty_matrices()"
        )

    def penalty_matrices(self) -> tuple[Tensor, ...]:
        marginal_penalties = tuple(
            self._marginal_transform(index).mT
            @ marginal.penalty()
            @ self._marginal_transform(index)
            for index, marginal in enumerate(self.marginals)
        )
        return tuple(
            self._coefficient_transform.mT
            @ penalty
            @ self._coefficient_transform
            for penalty in tensor_product_penalties(marginal_penalties)
        )

    @property
    def smoothing_parameter(self) -> float:
        raise RuntimeError(
            "tensor smooths have multiple penalties and smoothing parameters; "
            "use smoothing_parameters"
        )

    @property
    def smoothing_parameters(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._smoothing_parameter_values)

    @property
    def estimates_smoothing_parameter(self) -> bool:
        return any(self._estimated_smoothing_parameters)

    @property
    def estimated_smoothing_parameters(self) -> tuple[bool, ...]:
        return self._estimated_smoothing_parameters

    def _set_fitted_smoothing_parameters(
        self,
        values: Sequence[float],
    ) -> None:
        normalized = tuple(float(value) for value in values)
        if len(normalized) != len(self.smoothing_parameters):
            raise ValueError(
                "tensor smooth requires one smoothing parameter per penalty"
            )
        for index, (value, estimated, current) in enumerate(
            zip(
                normalized,
                self.estimated_smoothing_parameters,
                self.smoothing_parameters,
                strict=True,
            )
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"smoothing parameter {index} must be finite and positive"
                )
            if not estimated and not math.isclose(
                value,
                current,
                rel_tol=1e-12,
                abs_tol=1e-14,
            ):
                raise RuntimeError(
                    f"tensor smoothing parameter {index} is fixed"
                )
        self._smoothing_parameter_values.copy_(
            self._smoothing_parameter_values.new_tensor(normalized)
        )

    @property
    def penalty_nullity(self) -> int:
        combined = sum(
            self.penalty_matrices(),
            torch.zeros(
                (self.coefficients.numel(), self.coefficients.numel()),
                dtype=self.coefficients.dtype,
                device=self.coefficients.device,
            ),
        )
        rank = int(torch.linalg.matrix_rank(combined).detach())
        return self.coefficients.numel() - rank

    def constraints(self, covariates: Tensor) -> Tensor:
        design = self.design(covariates)
        if self.interaction or not self.center or self.constraint_absorbed:
            return design.new_empty((0, design.shape[1]))
        return design.sum(dim=0, keepdim=True)

    def quadratic_penalty(self) -> Tensor:
        return sum(
            (
                smoothing_parameter
                * (
                    self.coefficients
                    @ penalty
                    @ self.coefficients
                )
                for smoothing_parameter, penalty in zip(
                    self._smoothing_parameter_values,
                    self.penalty_matrices(),
                    strict=True,
                )
            ),
            self.coefficients.new_zeros(()),
        )

    def effective_degrees_of_freedom(
        self,
        covariate: Tensor,
        weights: Tensor,
    ) -> Tensor:
        design = self.design(covariate)
        if weights.ndim != 1 or weights.shape[0] != design.shape[0]:
            raise ValueError("weights must have one value per tensor row")
        if weights.dtype != design.dtype or weights.device != design.device:
            raise ValueError("weights must match tensor smooth dtype and device")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("at least one tensor smooth weight must be positive")
        transform = _right_null_space(
            self.constraints(covariate),
            allow_empty=True,
        )
        reduced_design = design @ transform
        gram = reduced_design.mT @ (
            weights.unsqueeze(-1) * reduced_design
        )
        combined_penalty = sum(
            (
                smoothing_parameter
                * (transform.mT @ penalty @ transform)
                for smoothing_parameter, penalty in zip(
                    self._smoothing_parameter_values,
                    self.penalty_matrices(),
                    strict=True,
                )
            ),
            torch.zeros_like(gram),
        )
        system = 0.5 * (
            gram + combined_penalty + (gram + combined_penalty).mT
        )
        return torch.trace(torch.linalg.pinv(system) @ gram)


class TensorInteractionSmooth(TensorProductSmooth):
    """Highest-order tensor interaction with marginal main effects removed."""

    def __init__(
        self,
        marginals: Sequence[SmoothTerm],
        training_covariates: Tensor,
        *,
        smoothing_parameters: Sequence[float] | None = None,
        estimate_smoothing: Sequence[bool] | bool = False,
    ) -> None:
        super().__init__(
            marginals,
            smoothing_parameters=smoothing_parameters,
            estimate_smoothing=estimate_smoothing,
            center=False,
            _interaction_covariates=training_covariates,
        )


def _validate_tensor_covariates(
    covariates: Tensor,
    marginal_count: int,
    reference: Tensor,
) -> None:
    if (
        covariates.ndim != 2
        or covariates.shape[0] < 1
        or covariates.shape[1] != marginal_count
    ):
        raise ValueError(
            f"tensor covariates must have shape (n, {marginal_count})"
        )
    if covariates.dtype != reference.dtype or covariates.device != reference.device:
        raise ValueError(
            "tensor covariates must match the smooth dtype and device"
        )
    if not torch.isfinite(covariates).all():
        raise ValueError("tensor covariates must be finite")


def _right_null_space(
    constraints: Tensor,
    *,
    allow_empty: bool = False,
) -> Tensor:
    coefficient_count = constraints.shape[1]
    if constraints.shape[0] == 0:
        if not allow_empty:
            raise ValueError("a marginal interaction constraint is empty")
        return torch.eye(
            coefficient_count,
            dtype=constraints.dtype,
            device=constraints.device,
        )
    _, singular_values, right_vectors = torch.linalg.svd(
        constraints,
        full_matrices=True,
    )
    largest = float(singular_values.detach().max())
    tolerance = (
        max(constraints.shape)
        * torch.finfo(constraints.dtype).eps
        * largest
    )
    rank = int((singular_values.detach() > tolerance).sum())
    if rank >= coefficient_count:
        raise ValueError("constraints leave no tensor coefficients")
    return right_vectors[rank:].mT
