"""Smooth terms for additive GAMLSS predictors."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

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
