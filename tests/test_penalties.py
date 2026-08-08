import pytest
import torch

from torchgamlss import PSpline, solve_penalized_least_squares


def _weighted_system(
    design: torch.Tensor,
    weights: torch.Tensor,
    penalty: torch.Tensor,
) -> torch.Tensor:
    return design.mT @ (weights.unsqueeze(-1) * design) + penalty


def test_multiple_penalties_match_direct_normal_equation_solution():
    design = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [1.0, -0.2, -0.3],
            [1.0, 0.4, 0.8],
            [1.0, 1.2, -0.4],
            [1.0, 1.8, 0.2],
        ],
        dtype=torch.float64,
    )
    response = torch.tensor([0.2, -0.1, 1.4, 1.1, 2.3], dtype=torch.float64)
    weights = torch.tensor([1.0, 0.5, 2.0, 1.5, 0.75], dtype=torch.float64)
    first_penalty = torch.diag(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    second_penalty = torch.diag(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        (first_penalty, second_penalty),
        (2.0, 5.0),
    )

    combined_penalty = 2.0 * first_penalty + 5.0 * second_penalty
    gram = design.mT @ (weights.unsqueeze(-1) * design)
    system = gram + combined_penalty
    expected_coefficients = torch.linalg.solve(
        system,
        design.mT @ (weights * response),
    )
    expected_edf = torch.trace(torch.linalg.solve(system, gram))

    torch.testing.assert_close(result.coefficients, expected_coefficients)
    torch.testing.assert_close(
        result.fitted_values,
        design @ expected_coefficients,
    )
    torch.testing.assert_close(
        result.combined_penalty_matrix,
        combined_penalty,
    )
    torch.testing.assert_close(
        result.effective_degrees_of_freedom,
        expected_edf,
    )
    assert result.penalty_ranks == (1, 1)
    assert result.constraint_rank == 0
    torch.testing.assert_close(
        result.constraint_null_space,
        torch.eye(3, dtype=torch.float64),
    )
    assert torch.isfinite(result.reduced_system_condition_number)


def test_constraints_use_null_space_reparameterization_and_allow_redundancy():
    design = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [1.0, 0.0, -0.2],
            [1.0, 0.7, 0.3],
            [1.0, 1.5, 1.0],
        ],
        dtype=torch.float64,
    )
    response = torch.tensor([0.0, 0.4, 1.0, 1.8], dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    penalty = torch.diag(torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64))
    constraint = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=torch.float64,
    )

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        (penalty,),
        (3.0,),
        constraints=constraint,
    )

    combined_penalty = 3.0 * penalty
    system = _weighted_system(design, weights, combined_penalty)
    kkt = torch.cat(
        (
            torch.cat((system, constraint[:1].mT), dim=1),
            torch.cat(
                (
                    constraint[:1],
                    torch.zeros((1, 1), dtype=torch.float64),
                ),
                dim=1,
            ),
        )
    )
    right_hand_side = torch.cat(
        (
            design.mT @ (weights * response),
            torch.zeros(1, dtype=torch.float64),
        )
    )
    expected = torch.linalg.solve(kkt, right_hand_side)[: design.shape[1]]

    torch.testing.assert_close(result.coefficients, expected)
    torch.testing.assert_close(
        constraint @ result.coefficients,
        torch.zeros(2, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        constraint @ result.constraint_null_space,
        torch.zeros((2, 2), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert result.constraint_rank == 1
    assert result.constraint_null_space.shape == (3, 2)
    assert 0.0 < result.effective_degrees_of_freedom < 2.0


def test_rank_deficient_penalty_and_extreme_lambda_remain_finite():
    covariate = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
    design = torch.column_stack(
        (
            torch.ones_like(covariate),
            covariate,
            covariate.square(),
        )
    )
    response = 0.5 + 1.2 * covariate - 0.4 * covariate.square()
    weights = torch.ones_like(response)
    difference = torch.tensor([[1.0, -2.0, 1.0]], dtype=torch.float64)
    rank_deficient_penalty = difference.mT @ difference

    moderate = solve_penalized_least_squares(
        design,
        response,
        weights,
        (rank_deficient_penalty,),
        (10.0,),
    )
    extreme = solve_penalized_least_squares(
        design,
        response,
        weights,
        (rank_deficient_penalty,),
        (1e12,),
    )

    assert moderate.penalty_ranks == (1,)
    assert torch.isfinite(moderate.coefficients).all()
    assert torch.isfinite(extreme.coefficients).all()
    assert torch.isfinite(extreme.effective_degrees_of_freedom)
    assert (
        extreme.effective_degrees_of_freedom
        < moderate.effective_degrees_of_freedom
    )
    assert float(extreme.effective_degrees_of_freedom) == pytest.approx(
        2.0,
        abs=1e-8,
    )
    assert float(
        extreme.coefficients @ rank_deficient_penalty @ extreme.coefficients
    ) < 1e-18


def test_generic_solver_consumes_pspline_coefficient_space_contract():
    covariate = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    response = torch.sin(2.0 * covariate) + 0.1 * covariate
    weights = torch.linspace(0.5, 1.5, covariate.numel(), dtype=torch.float64)
    term = PSpline.from_data(
        covariate,
        smoothing_parameter=12.0,
        intervals=10,
    )
    design = term.design(covariate)

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        term.penalty_matrices(),
        term.smoothing_parameters,
        constraints=term.constraints(covariate),
    )

    penalty_root = term.penalty_matrix()
    square_root_weights = weights.sqrt()
    augmented_design = torch.cat(
        (
            design * square_root_weights.unsqueeze(-1),
            (term.smoothing_parameter**0.5) * penalty_root,
        )
    )
    augmented_response = torch.cat(
        (
            response * square_root_weights,
            torch.zeros(
                penalty_root.shape[0],
                dtype=torch.float64,
            ),
        )
    )
    expected_coefficients = torch.linalg.lstsq(
        augmented_design,
        augmented_response,
    ).solution

    torch.testing.assert_close(
        result.coefficients,
        expected_coefficients,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        result.effective_degrees_of_freedom,
        term.effective_degrees_of_freedom(covariate, weights),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    ("penalty", "message"),
    [
        (
            torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
            "symmetric",
        ),
        (
            torch.tensor([[1.0, 0.0], [0.0, -0.1]], dtype=torch.float64),
            "positive semidefinite",
        ),
        (
            torch.tensor(
                [[1.0, float("nan")], [float("nan"), 1.0]],
                dtype=torch.float64,
            ),
            "finite",
        ),
    ],
)
def test_invalid_penalty_matrices_are_rejected(penalty, message):
    design = torch.eye(2, dtype=torch.float64)
    response = torch.ones(2, dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)

    with pytest.raises(ValueError, match=message):
        solve_penalized_least_squares(
            design,
            response,
            weights,
            (penalty,),
            (1.0,),
        )


@pytest.mark.parametrize(
    "smoothing_parameters",
    [
        (),
        (-1.0,),
        (float("inf"),),
        (torch.ones(2, dtype=torch.float64),),
    ],
)
def test_invalid_smoothing_parameters_are_rejected(smoothing_parameters):
    design = torch.eye(2, dtype=torch.float64)
    response = torch.ones(2, dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    penalties = (
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
    )
    selected_penalties = penalties if not smoothing_parameters else penalties[:1]

    with pytest.raises(ValueError, match="smoothing|equal lengths"):
        solve_penalized_least_squares(
            design,
            response,
            weights,
            selected_penalties,
            smoothing_parameters,
        )


def test_invalid_constraints_and_unidentified_system_are_rejected():
    design = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    response = torch.ones(2, dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    zero_penalty = torch.zeros((2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="rank deficient"):
        solve_penalized_least_squares(
            design,
            response,
            weights,
            (zero_penalty,),
            (0.0,),
        )
    with pytest.raises(ValueError, match="leave at least one"):
        solve_penalized_least_squares(
            design,
            response,
            weights,
            (zero_penalty,),
            (0.0,),
            constraints=torch.eye(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="shape"):
        solve_penalized_least_squares(
            design,
            response,
            weights,
            (zero_penalty,),
            (0.0,),
            constraints=torch.ones((1, 3), dtype=torch.float64),
        )


def test_overflowing_combined_penalty_is_rejected():
    design = torch.eye(2, dtype=torch.float64)
    response = torch.ones(2, dtype=torch.float64)
    weights = torch.ones(2, dtype=torch.float64)
    penalty = 2.0 * torch.eye(2, dtype=torch.float64)

    with pytest.raises(ValueError, match="combined penalty matrix must be finite"):
        solve_penalized_least_squares(
            design,
            response,
            weights,
            (penalty,),
            (torch.finfo(torch.float64).max,),
        )


def test_fixed_tensor_lambdas_retain_autograd_for_future_outer_criteria():
    design = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    response = torch.tensor([0.0, 1.0, 2.0, 2.5], dtype=torch.float64)
    weights = torch.ones(4, dtype=torch.float64)
    first = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    second = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    first_lambda = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    second_lambda = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        (first, second),
        (first_lambda, second_lambda),
    )
    criterion = (
        result.coefficients.square().sum()
        + result.effective_degrees_of_freedom
    )
    gradients = torch.autograd.grad(
        criterion,
        (first_lambda, second_lambda),
    )

    assert all(torch.isfinite(gradient) for gradient in gradients)
    assert all(gradient.abs() > 0 for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_multiple_penalty_solver_runs_on_cuda_float64():
    device = torch.device("cuda")
    design = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [1.0, 0.0, -0.5],
            [1.0, 0.5, 0.2],
            [1.0, 1.0, 0.8],
        ],
        dtype=torch.float64,
        device=device,
    )
    response = torch.tensor(
        [0.1, 0.4, 1.0, 1.7],
        dtype=torch.float64,
        device=device,
    )
    weights = torch.tensor(
        [1.0, 2.0, 0.5, 1.5],
        dtype=torch.float64,
        device=device,
    )
    first = torch.diag(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    )
    second = torch.diag(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)
    )
    constraints = torch.tensor(
        [[0.0, 1.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        (first, second),
        (2.0, 3.0),
        constraints=constraints,
    )

    assert result.coefficients.device.type == "cuda"
    assert result.combined_penalty_matrix.device.type == "cuda"
    assert torch.isfinite(result.coefficients).all()
    assert torch.isfinite(result.effective_degrees_of_freedom)
    torch.testing.assert_close(
        constraints @ result.coefficients,
        torch.zeros(1, dtype=torch.float64, device=device),
        atol=1e-12,
        rtol=0.0,
    )
