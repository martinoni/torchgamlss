"""Generic penalized weighted least-squares systems for smooth terms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PenalizedLeastSquaresResult:
    """Result of a constrained, multiply penalized least-squares solve."""

    coefficients: Tensor
    fitted_values: Tensor
    effective_degrees_of_freedom: Tensor
    combined_penalty_matrix: Tensor
    constraint_null_space: Tensor
    penalty_ranks: tuple[int, ...]
    constraint_rank: int
    reduced_system_condition_number: Tensor


def solve_penalized_least_squares(
    design: Tensor,
    response: Tensor,
    weights: Tensor,
    penalty_matrices: Sequence[Tensor],
    smoothing_parameters: Sequence[float | Tensor],
    *,
    constraints: Tensor | None = None,
) -> PenalizedLeastSquaresResult:
    """Solve a weighted least-squares problem with generic smooth penalties.

    The minimized criterion is

    ``sum_i w_i (y_i - X_i beta)^2 + beta.T @ S_lambda @ beta``

    subject to ``C @ beta = 0``, where

    ``S_lambda = sum_j lambda_j S_j``.

    Each ``S_j`` must be a finite, symmetric positive-semidefinite matrix in
    coefficient space and every ``lambda_j`` must be finite and non-negative.
    Constraints are imposed through a null-space reparameterization rather than
    through a large numerical penalty.

    This is the generic dense solver. The GAMLSS-compatible scalar ``PSpline``
    fitter intentionally retains its existing square-root augmented
    least-squares path for numerical parity with ``gamlss::pb()``.
    """
    _validate_observation_system(design, response, weights)
    components = _validate_penalty_components(
        design,
        penalty_matrices,
        smoothing_parameters,
    )
    _, constraint_rank, null_space = _constraint_null_space(
        design,
        constraints,
    )

    combined_penalty = torch.zeros(
        (design.shape[1], design.shape[1]),
        dtype=design.dtype,
        device=design.device,
    )
    penalty_ranks: list[int] = []
    for matrix, smoothing_parameter, rank in components:
        combined_penalty = combined_penalty + smoothing_parameter * matrix
        penalty_ranks.append(rank)
    if not bool(torch.isfinite(combined_penalty).all().detach()):
        raise ValueError("combined penalty matrix must be finite")
    combined_penalty = _symmetrize(combined_penalty)

    reduced_design = design @ null_space
    reduced_penalty = _symmetrize(
        null_space.mT @ combined_penalty @ null_space
    )
    penalty_root = _positive_semidefinite_root(reduced_penalty)
    square_root_weights = weights.sqrt()
    weighted_reduced_design = (
        reduced_design * square_root_weights.unsqueeze(-1)
    )
    augmented_design = torch.cat(
        (
            weighted_reduced_design,
            penalty_root,
        )
    )
    augmented_response = torch.cat(
        (
            response * square_root_weights,
            torch.zeros(
                penalty_root.shape[0],
                dtype=response.dtype,
                device=response.device,
            ),
        )
    )
    reduced_coefficient_count = reduced_design.shape[1]
    augmented_rank = int(torch.linalg.matrix_rank(augmented_design).detach())
    if augmented_rank < reduced_coefficient_count:
        raise ValueError(
            "penalized system is rank deficient after applying constraints"
        )

    reduced_coefficients = torch.linalg.lstsq(
        augmented_design,
        augmented_response,
    ).solution
    coefficients = null_space @ reduced_coefficients
    fitted_values = design @ coefficients

    gram = reduced_design.mT @ (weights.unsqueeze(-1) * reduced_design)
    reduced_system = _symmetrize(gram + reduced_penalty)
    influence_rhs = torch.cat(
        (
            weighted_reduced_design,
            torch.zeros_like(penalty_root),
        )
    )
    coefficient_influence = torch.linalg.lstsq(
        augmented_design,
        influence_rhs,
    ).solution
    effective_degrees_of_freedom = torch.trace(coefficient_influence)
    condition_number = torch.linalg.cond(reduced_system)

    return PenalizedLeastSquaresResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        effective_degrees_of_freedom=effective_degrees_of_freedom,
        combined_penalty_matrix=combined_penalty,
        constraint_null_space=null_space,
        penalty_ranks=tuple(penalty_ranks),
        constraint_rank=constraint_rank,
        reduced_system_condition_number=condition_number,
    )


def _validate_observation_system(
    design: Tensor,
    response: Tensor,
    weights: Tensor,
) -> None:
    if design.ndim != 2 or design.shape[0] < 1 or design.shape[1] < 1:
        raise ValueError("design must be a non-empty two-dimensional tensor")
    if not design.is_floating_point():
        raise ValueError("design must use a floating-point dtype")
    if not torch.isfinite(design).all():
        raise ValueError("design must be finite")
    if response.ndim != 1 or response.shape[0] != design.shape[0]:
        raise ValueError("response must have one value per design row")
    if response.dtype != design.dtype or response.device != design.device:
        raise ValueError("response must match the design dtype and device")
    if not torch.isfinite(response).all():
        raise ValueError("response must be finite")
    if weights.ndim != 1 or weights.shape[0] != design.shape[0]:
        raise ValueError("weights must have one value per design row")
    if weights.dtype != design.dtype or weights.device != design.device:
        raise ValueError("weights must match the design dtype and device")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("at least one weight must be positive")


def _validate_penalty_components(
    design: Tensor,
    penalty_matrices: Sequence[Tensor],
    smoothing_parameters: Sequence[float | Tensor],
) -> tuple[tuple[Tensor, Tensor, int], ...]:
    if isinstance(penalty_matrices, Tensor):
        raise ValueError("penalty_matrices must be a sequence of square tensors")
    if isinstance(smoothing_parameters, Tensor):
        raise ValueError(
            "smoothing_parameters must be a sequence of scalar values"
        )
    matrices = tuple(penalty_matrices)
    parameters = tuple(smoothing_parameters)
    if not matrices:
        raise ValueError("at least one penalty matrix is required")
    if len(matrices) != len(parameters):
        raise ValueError(
            "penalty_matrices and smoothing_parameters must have equal lengths"
        )

    coefficient_count = design.shape[1]
    validated: list[tuple[Tensor, Tensor, int]] = []
    for index, (matrix, value) in enumerate(zip(matrices, parameters, strict=True)):
        if not isinstance(matrix, Tensor):
            raise ValueError(f"penalty matrix {index} must be a tensor")
        if matrix.ndim != 2 or matrix.shape != (
            coefficient_count,
            coefficient_count,
        ):
            raise ValueError(
                f"penalty matrix {index} must have shape "
                f"({coefficient_count}, {coefficient_count})"
            )
        if matrix.dtype != design.dtype or matrix.device != design.device:
            raise ValueError(
                f"penalty matrix {index} must match the design dtype and device"
            )
        if not torch.isfinite(matrix).all():
            raise ValueError(f"penalty matrix {index} must be finite")

        scale = max(float(matrix.detach().abs().max()), 1.0)
        tolerance = _matrix_tolerance(matrix, scale)
        asymmetry = float((matrix - matrix.mT).detach().abs().max())
        if asymmetry > tolerance:
            raise ValueError(f"penalty matrix {index} must be symmetric")
        symmetric_matrix = _symmetrize(matrix)
        eigenvalues = torch.linalg.eigvalsh(symmetric_matrix)
        if float(eigenvalues.detach().min()) < -tolerance:
            raise ValueError(
                f"penalty matrix {index} must be positive semidefinite"
            )

        smoothing_parameter = _as_smoothing_parameter(
            design,
            value,
            index=index,
        )
        rank = int((eigenvalues.detach() > tolerance).sum())
        validated.append((symmetric_matrix, smoothing_parameter, rank))
    return tuple(validated)


def _as_smoothing_parameter(
    reference: Tensor,
    value: float | Tensor,
    *,
    index: int,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError(f"smoothing parameter {index} must be scalar")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(
                f"smoothing parameter {index} must match the design dtype and device"
            )
        parameter = value
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"smoothing parameter {index} must be a scalar value"
            ) from error
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"smoothing parameter {index} must be finite and non-negative"
            )
        parameter = reference.new_tensor(numeric_value)
    if not bool(torch.isfinite(parameter).detach()) or bool(
        (parameter < 0).detach()
    ):
        raise ValueError(
            f"smoothing parameter {index} must be finite and non-negative"
        )
    return parameter


def _constraint_null_space(
    design: Tensor,
    constraints: Tensor | None,
) -> tuple[Tensor, int, Tensor]:
    coefficient_count = design.shape[1]
    if constraints is None:
        constraint_matrix = design.new_empty((0, coefficient_count))
    else:
        if not isinstance(constraints, Tensor):
            raise ValueError("constraints must be a tensor or None")
        constraint_matrix = constraints
    if constraint_matrix.ndim != 2 or constraint_matrix.shape[1] != coefficient_count:
        raise ValueError(
            f"constraints must have shape (q, {coefficient_count})"
        )
    if (
        constraint_matrix.dtype != design.dtype
        or constraint_matrix.device != design.device
    ):
        raise ValueError("constraints must match the design dtype and device")
    if not torch.isfinite(constraint_matrix).all():
        raise ValueError("constraints must be finite")
    if constraint_matrix.shape[0] == 0:
        return (
            constraint_matrix,
            0,
            torch.eye(
                coefficient_count,
                dtype=design.dtype,
                device=design.device,
            ),
        )

    _, singular_values, right_vectors = torch.linalg.svd(
        constraint_matrix,
        full_matrices=True,
    )
    largest = (
        float(singular_values.detach().max())
        if singular_values.numel()
        else 0.0
    )
    rank_tolerance = (
        max(constraint_matrix.shape)
        * torch.finfo(design.dtype).eps
        * largest
    )
    constraint_rank = int(
        (singular_values.detach() > rank_tolerance).sum()
    )
    if constraint_rank >= coefficient_count:
        raise ValueError("constraints must leave at least one coefficient free")
    null_space = right_vectors[constraint_rank:].mT
    return constraint_matrix, constraint_rank, null_space


def _positive_semidefinite_root(matrix: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(_symmetrize(matrix))
    scale = max(float(matrix.detach().abs().max()), 1.0)
    tolerance = _matrix_tolerance(matrix, scale)
    if float(eigenvalues.detach().min()) < -tolerance:
        raise RuntimeError("combined reduced penalty is not positive semidefinite")
    retained = torch.where(
        eigenvalues > tolerance,
        eigenvalues,
        torch.zeros_like(eigenvalues),
    )
    square_roots = retained.sqrt()
    return square_roots.unsqueeze(-1) * eigenvectors.mT


def _matrix_tolerance(matrix: Tensor, scale: float) -> float:
    return 100.0 * torch.finfo(matrix.dtype).eps * max(matrix.shape) * scale


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.mT)
