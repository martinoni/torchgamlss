from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    Normal,
    PSpline,
    RSControl,
    TensorInteractionSmooth,
    TensorProductSmooth,
    row_tensor_product,
    solve_penalized_least_squares,
    tensor_product_penalties,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _training_grid(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, 9, dtype=dtype, device=device)
    z = torch.linspace(0.0, 2.0, 8, dtype=dtype, device=device)
    return torch.cartesian_prod(x, z)


def _marginals(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
):
    return (
        PSpline(
            -1.0,
            1.0,
            2.0,
            intervals=3,
            dtype=dtype,
            device=device,
        ),
        PSpline(
            0.0,
            2.0,
            5.0,
            intervals=2,
            dtype=dtype,
            device=device,
        ),
    )


def test_low_level_tensor_construction_matches_mgcv_reference():
    first_design = torch.tensor(
        [
            [1.0, -0.5, 0.2],
            [1.0, -0.1, 0.7],
            [1.0, 0.3, -0.4],
            [1.0, 0.8, 0.5],
            [1.0, 1.2, -0.2],
        ],
        dtype=torch.float64,
    )
    second_design = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
        dtype=torch.float64,
    )
    first_difference = torch.tensor(
        [[1.0, -2.0, 1.0]],
        dtype=torch.float64,
    )
    second_difference = torch.tensor(
        [[1.0, -1.0]],
        dtype=torch.float64,
    )
    expected_design = pd.read_csv(
        REFERENCE_DIR / "mgcv_tensor_design_reference.csv"
    ).filter(regex=r"^coefficient_")
    penalty_rows = pd.read_csv(
        REFERENCE_DIR / "mgcv_tensor_penalty_reference.csv"
    )

    design = row_tensor_product((first_design, second_design))
    penalties = tensor_product_penalties(
        (
            first_difference.mT @ first_difference,
            second_difference.mT @ second_difference,
        )
    )

    torch.testing.assert_close(
        design,
        torch.tensor(expected_design.to_numpy(), dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    for index, penalty in enumerate(penalties, start=1):
        rows = penalty_rows[penalty_rows["penalty"] == index]
        expected = torch.zeros_like(penalty)
        expected[
            torch.tensor(rows["row"].to_numpy() - 1),
            torch.tensor(rows["column"].to_numpy() - 1),
        ] = torch.tensor(rows["value"].to_numpy(), dtype=torch.float64)
        torch.testing.assert_close(penalty, expected, rtol=0.0, atol=0.0)


def test_low_level_tensor_helpers_validate_first_inputs():
    with pytest.raises(ValueError, match="marginal design 0"):
        row_tensor_product(([1.0, 2.0], torch.ones((2, 1))))
    with pytest.raises(ValueError, match="marginal penalty 0"):
        tensor_product_penalties(([1.0, 2.0], torch.eye(2)))


def test_tensor_product_exposes_row_kronecker_penalties_and_constraint():
    covariates = _training_grid()
    first, second = _marginals()
    with torch.no_grad():
        first.coefficients.fill_(3.0)
        second.coefficients.fill_(-2.0)
    term = TensorProductSmooth((first, second))
    first_design = first.design(covariates[:, 0])
    second_design = second.design(covariates[:, 1])
    expected_design = torch.einsum(
        "ni,nj->nij",
        first_design,
        second_design,
    ).reshape(covariates.shape[0], -1)
    first_penalty, second_penalty = term.penalty_matrices()

    assert term.coefficient_shape == (
        first.coefficients.numel(),
        second.coefficients.numel(),
    )
    assert term.smoothing_parameters == (2.0, 5.0)
    assert tuple(dict(term.named_parameters())) == ("coefficients",)
    assert torch.count_nonzero(term.coefficients) == 0
    torch.testing.assert_close(term.design(covariates), expected_design)
    torch.testing.assert_close(
        term.predict_design(covariates),
        expected_design,
    )
    torch.testing.assert_close(
        first_penalty,
        torch.kron(
            first.penalty_matrices()[0],
            torch.eye(
                second.coefficients.numel(),
                dtype=torch.float64,
            ),
        ),
    )
    torch.testing.assert_close(
        second_penalty,
        torch.kron(
            torch.eye(
                first.coefficients.numel(),
                dtype=torch.float64,
            ),
            second.penalty_matrices()[0],
        ),
    )
    constraint = term.constraints(covariates)
    assert constraint.shape == (1, term.coefficients.numel())
    torch.testing.assert_close(
        constraint,
        expected_design.sum(dim=0, keepdim=True),
    )
    assert term.penalty_nullity == 4

    with torch.no_grad():
        term.coefficients.copy_(
            torch.linspace(
                -0.3,
                0.4,
                term.coefficients.numel(),
                dtype=torch.float64,
            )
        )
    expected_penalty = sum(
        smoothing_parameter
        * (term.coefficients @ penalty @ term.coefficients)
        for smoothing_parameter, penalty in zip(
            term._smoothing_parameter_values,
            term.penalty_matrices(),
            strict=True,
        )
    )
    torch.testing.assert_close(term.quadratic_penalty(), expected_penalty)


def test_generic_solver_consumes_tensor_product_penalties_and_constraint():
    covariates = _training_grid()
    term = TensorProductSmooth(_marginals())
    design = term.design(covariates)
    response = (
        torch.sin(2.0 * covariates[:, 0])
        + 0.4 * covariates[:, 0] * covariates[:, 1]
        + 0.2 * covariates[:, 1].square()
    )
    weights = torch.linspace(
        0.5,
        1.5,
        response.numel(),
        dtype=torch.float64,
    )
    constraints = term.constraints(covariates)

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        term.penalty_matrices(),
        term.smoothing_parameters,
        constraints=constraints,
    )
    with torch.no_grad():
        term.coefficients.copy_(result.coefficients)

    torch.testing.assert_close(term(covariates), result.fitted_values)
    torch.testing.assert_close(
        constraints @ result.coefficients,
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )
    torch.testing.assert_close(
        term.effective_degrees_of_freedom(covariates, weights),
        result.effective_degrees_of_freedom,
        rtol=1e-9,
        atol=1e-9,
    )
    assert result.penalty_ranks == (
        (first_rank := torch.linalg.matrix_rank(term.penalty_matrices()[0]).item()),
        torch.linalg.matrix_rank(term.penalty_matrices()[1]).item(),
    )
    assert first_rank > 0


def test_rs_backfitting_consumes_external_tensor_constraint():
    covariates = _training_grid()
    term = TensorProductSmooth(_marginals())
    response = (
        0.6
        + torch.sin(2.0 * covariates[:, 0])
        + 0.4 * covariates[:, 0] * covariates[:, 1]
        + 0.2 * covariates[:, 1].square()
        + 0.03
        * torch.cos(
            torch.arange(covariates.shape[0], dtype=torch.float64)
        )
    )
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"surface": term}},
        dtype=torch.float64,
    )
    design_matrices = {
        "mu": torch.ones((response.numel(), 1), dtype=torch.float64),
        "sigma": torch.ones((response.numel(), 1), dtype=torch.float64),
    }

    result = model.fit_rs(
        response,
        design_matrices,
        smooth_covariates={"mu": {"surface": covariates}},
        control=RSControl(
            outer_tolerance=1e-7,
            max_outer_iterations=60,
            inner_tolerance=1e-7,
            max_inner_iterations=60,
            backfitting_tolerance=1e-7,
            max_backfitting_iterations=60,
        ),
    )

    assert result.converged
    assert result.smoothing_parameters["mu"]["surface"] == (2.0, 5.0)
    torch.testing.assert_close(
        term.constraints(covariates) @ term.coefficients,
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )
    assert result.parameter_effective_degrees_of_freedom[
        "mu"
    ] == pytest.approx(
        1.0 + result.smooth_effective_degrees_of_freedom["mu"]["surface"]
    )
    joint = model.smooth_joint_inference(
        response,
        design_matrices,
        smooth_covariates={"mu": {"surface": covariates}},
    )
    coefficient_slice = joint.smooth_coefficient_slices[("mu", "surface")]
    constraint = term.constraints(covariates)
    covariance = joint.coefficient_covariance_matrix
    torch.testing.assert_close(
        constraint @ covariance[coefficient_slice],
        torch.zeros(
            (1, covariance.shape[1]),
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1e-11,
    )
    torch.testing.assert_close(
        covariance[:, coefficient_slice] @ constraint.mT,
        torch.zeros(
            (covariance.shape[0], 1),
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1e-11,
    )


def test_tensor_interaction_removes_marginal_main_effect_directions():
    covariates = _training_grid()
    first, second = _marginals()
    term = TensorInteractionSmooth((first, second), covariates)
    first_design, second_design = term.marginal_designs(covariates)
    interaction_design = term.design(covariates)

    assert term.interaction
    assert term.coefficient_shape == (
        first.coefficients.numel() - 1,
        second.coefficients.numel() - 1,
    )
    assert term.constraints(covariates).shape == (
        0,
        term.coefficients.numel(),
    )
    torch.testing.assert_close(
        first_design.sum(dim=0),
        torch.zeros(first_design.shape[1], dtype=torch.float64),
        rtol=0.0,
        atol=2e-14,
    )
    torch.testing.assert_close(
        second_design.sum(dim=0),
        torch.zeros(second_design.shape[1], dtype=torch.float64),
        rtol=0.0,
        atol=2e-14,
    )
    torch.testing.assert_close(
        first_design.mT @ interaction_design,
        torch.zeros(
            (first_design.shape[1], interaction_design.shape[1]),
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=2e-13,
    )
    torch.testing.assert_close(
        second_design.mT @ interaction_design,
        torch.zeros(
            (second_design.shape[1], interaction_design.shape[1]),
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=2e-13,
    )
    assert len(term.penalty_matrices()) == 2
    assert term.penalty_nullity == 1


def test_tensor_term_state_and_prediction_transforms_round_trip():
    covariates = _training_grid()
    evaluation = torch.tensor(
        [[-0.8, 0.2], [-0.1, 0.7], [0.4, 1.3], [0.9, 1.8]],
        dtype=torch.float64,
    )
    original = TensorInteractionSmooth(
        _marginals(),
        covariates,
        smoothing_parameters=(3.0, 7.0),
    )
    with torch.no_grad():
        original.coefficients.copy_(
            torch.linspace(
                -0.2,
                0.3,
                original.coefficients.numel(),
                dtype=torch.float64,
            )
        )
    restored = TensorInteractionSmooth(
        _marginals(),
        covariates,
        smoothing_parameters=(3.0, 7.0),
    )
    restored.load_state_dict(original.state_dict())

    torch.testing.assert_close(
        restored.predict_design(evaluation),
        original.predict_design(evaluation),
    )
    torch.testing.assert_close(restored(evaluation), original(evaluation))
    assert restored.smoothing_parameters == (3.0, 7.0)


def test_tensor_term_moves_basis_penalties_and_transforms_between_dtypes():
    covariates = _training_grid()
    term = TensorInteractionSmooth(_marginals(), covariates).to(torch.float32)
    float_covariates = covariates.to(torch.float32)

    assert term.design(float_covariates).dtype == torch.float32
    assert term.constraints(float_covariates).dtype == torch.float32
    assert all(
        penalty.dtype == torch.float32 for penalty in term.penalty_matrices()
    )
    assert tuple(dict(term.named_parameters())) == ("coefficients",)


def test_tensor_pspline_design_is_invariant_to_linear_unit_changes():
    covariates = _training_grid()
    transformed_covariates = torch.column_stack(
        (
            1000.0 + 20.0 * covariates[:, 0],
            -0.3 + 0.001 * covariates[:, 1],
        )
    )
    original = TensorProductSmooth(_marginals(), center=False)
    transformed = TensorProductSmooth(
        (
            PSpline(
                980.0,
                1020.0,
                2.0,
                intervals=3,
                dtype=torch.float64,
            ),
            PSpline(
                -0.3,
                -0.298,
                5.0,
                intervals=2,
                dtype=torch.float64,
            ),
        ),
        center=False,
    )

    torch.testing.assert_close(
        transformed.design(transformed_covariates),
        original.design(covariates),
        rtol=2e-10,
        atol=2e-10,
    )
    for transformed_penalty, original_penalty in zip(
        transformed.penalty_matrices(),
        original.penalty_matrices(),
        strict=True,
    ):
        torch.testing.assert_close(
            transformed_penalty,
            original_penalty,
            rtol=0.0,
            atol=0.0,
        )


def test_tensor_design_and_penalty_retain_autograd():
    covariates = torch.tensor(
        [
            [-0.87, 0.13],
            [-0.31, 0.62],
            [0.18, 1.16],
            [0.73, 1.71],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    term = TensorProductSmooth(_marginals(), center=False)
    with torch.no_grad():
        term.coefficients.copy_(
            torch.linspace(
                -0.1,
                0.2,
                term.coefficients.numel(),
                dtype=torch.float64,
            )
        )
    objective = term(covariates).square().sum() + term.quadratic_penalty()
    covariate_gradient, coefficient_gradient = torch.autograd.grad(
        objective,
        (covariates, term.coefficients),
    )

    assert torch.isfinite(covariate_gradient).all()
    assert torch.isfinite(coefficient_gradient).all()
    assert covariate_gradient.abs().sum() > 0
    assert coefficient_gradient.abs().sum() > 0


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: TensorProductSmooth((_marginals()[0],)), "at least two"),
        (
            lambda: TensorProductSmooth(
                _marginals(),
                smoothing_parameters=(1.0,),
            ),
            "one value per margin",
        ),
        (
            lambda: TensorProductSmooth(
                _marginals(),
                smoothing_parameters=(1.0, -1.0),
            ),
            "non-negative",
        ),
        (
            lambda: TensorProductSmooth(
                _marginals(),
                estimate_smoothing=(True,),
            ),
            "one boolean per margin",
        ),
        (
            lambda: TensorProductSmooth(_marginals(), center=1),
            "boolean",
        ),
        (
            lambda: TensorInteractionSmooth(
                _marginals(),
                torch.ones((5, 1), dtype=torch.float64),
            ),
            "shape",
        ),
    ],
)
def test_invalid_tensor_configuration_is_rejected(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()


def test_tensor_product_tracks_and_updates_penalty_selection_flags():
    covariates = _training_grid()
    term = TensorProductSmooth(
        _marginals(),
        smoothing_parameters=(2.0, 5.0),
        estimate_smoothing=(True, False),
        training_covariates=covariates,
    )

    assert term.estimates_smoothing_parameter
    assert term.estimated_smoothing_parameters == (True, False)
    term._set_fitted_smoothing_parameters((3.0, 5.0))
    assert term.smoothing_parameters == (3.0, 5.0)
    with pytest.raises(RuntimeError, match="is fixed"):
        term._set_fitted_smoothing_parameters((3.0, 6.0))


def test_tensor_legacy_scalar_penalty_access_is_explicitly_rejected():
    term = TensorProductSmooth(_marginals())

    with pytest.raises(RuntimeError, match="multiple penalties"):
        _ = term.smoothing_parameter
    with pytest.raises(RuntimeError, match="multiple penalties"):
        term.penalty_matrix()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tensor_product_solver_runs_on_cuda_float64():
    device = torch.device("cuda")
    covariates = _training_grid(device=device)
    term = TensorProductSmooth(_marginals(device=device))
    design = term.design(covariates)
    response = (
        torch.sin(2.0 * covariates[:, 0])
        + 0.3 * covariates[:, 0] * covariates[:, 1]
    )
    weights = torch.ones_like(response)

    result = solve_penalized_least_squares(
        design,
        response,
        weights,
        term.penalty_matrices(),
        term.smoothing_parameters,
        constraints=term.constraints(covariates),
    )

    assert result.coefficients.device.type == "cuda"
    assert result.combined_penalty_matrix.device.type == "cuda"
    assert torch.isfinite(result.coefficients).all()
    assert torch.isfinite(result.effective_degrees_of_freedom)
