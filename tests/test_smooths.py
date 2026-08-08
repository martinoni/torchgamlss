from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    AdaptivePSpline,
    AdaptiveTensorPSpline,
    CyclicPSpline,
    MarkovRandomField,
    PSpline,
    ShrinkagePSpline,
    ThinPlateSpline,
    polygon_adjacency,
    solve_penalized_least_squares,
)

REFERENCE_DIR = Path(__file__).parent / "reference"
MRF_EDGES = (
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
)
MRF_LABELS = (
    "north_west",
    "north",
    "north_east",
    "west",
    "center",
    "east",
    "south_west",
    "south",
    "south_east",
)
MRF_ADJACENCY = {
    label: tuple(
        MRF_LABELS[second if first == index else first]
        for first, second in MRF_EDGES
        if index in (first, second)
    )
    for index, label in enumerate(MRF_LABELS)
}
MRF_EDGE_WEIGHTS = (
    0.7,
    1.1,
    0.9,
    1.4,
    0.8,
    1.3,
    1.05,
    0.65,
    1.55,
    1.2,
    0.95,
    1.35,
)
MRF_POLYGONS = {
    f"area_{index}": (
        (float(index), 0.0),
        (float(index + 1), 0.0),
        (float(index) + 0.5, 0.8),
    )
    for index in range(6)
}


def _mrf_graph_penalty() -> torch.Tensor:
    penalty = torch.zeros((9, 9), dtype=torch.float64)
    for (first, second), weight in zip(MRF_EDGES, MRF_EDGE_WEIGHTS, strict=True):
        penalty[first, first] += weight
        penalty[second, second] += weight
        penalty[first, second] -= weight
        penalty[second, first] -= weight
    return penalty


def test_pspline_basis_and_penalty_match_r_pb():
    covariate = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    term = PSpline.from_data(covariate, smoothing_parameter=10.0, intervals=10)

    expected_basis = torch.tensor(
        [
            [
                0.12229584900729593,
                0.6575261400215593,
                0.2200209572487208,
                0.0001570537224245078,
                0.0,
            ],
            [
                0.04589308253335528,
                0.5659024932010855,
                0.38109394598770485,
                0.00711047827785337,
                0.0,
            ],
            [
                0.01060244205290356,
                0.4141356821567531,
                0.5391168851939412,
                0.0361449905964001,
                0.0,
            ],
        ],
        dtype=torch.float64,
    )
    expected_penalty = torch.tensor(
        [
            [1.0, -2.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, -2.0, 1.0],
        ],
        dtype=torch.float64,
    )

    assert term.coefficients.numel() == 13
    assert not term.estimates_smoothing_parameter
    assert term.smoothing_method is None
    torch.testing.assert_close(
        term.basis(covariate)[:3, :5], expected_basis, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        term.penalty_matrix()[:3, :5], expected_penalty, rtol=0.0, atol=0.0
    )


def test_pspline_forward_penalty_and_effective_degrees_of_freedom():
    covariate = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
    term = PSpline.from_data(covariate, smoothing_parameter=4.0, intervals=8)
    with torch.no_grad():
        term.coefficients.copy_(
            torch.arange(term.coefficients.numel(), dtype=torch.float64).square()
        )

    assert term(covariate).shape == covariate.shape
    expected_penalty = 4.0 * (term.penalty_matrix() @ term.coefficients).square().sum()
    torch.testing.assert_close(term.quadratic_penalty(), expected_penalty)
    edf = term.effective_degrees_of_freedom(covariate, torch.ones_like(covariate))
    assert 2.0 < float(edf) < term.coefficients.numel()


def test_pspline_exposes_generic_basis_penalty_constraint_contract():
    covariate = torch.linspace(-1.0, 1.0, 21, dtype=torch.float64)
    term = PSpline.from_data(
        covariate,
        smoothing_parameter=4.0,
        intervals=8,
    )

    torch.testing.assert_close(term.design(covariate), term.basis(covariate))
    torch.testing.assert_close(
        term.predict_design(covariate),
        term.basis(covariate),
    )
    penalty_root = term.penalty_matrix()
    penalty_matrices = term.penalty_matrices()
    assert len(penalty_matrices) == 1
    torch.testing.assert_close(
        penalty_matrices[0],
        penalty_root.mT @ penalty_root,
    )
    assert term.smoothing_parameters == (4.0,)
    constraints = term.constraints(covariate)
    assert constraints.shape == (0, term.coefficients.numel())
    assert constraints.dtype == covariate.dtype
    assert constraints.device == covariate.device


def test_cyclic_pspline_basis_penalty_and_periodicity_match_r_pbc():
    data = pd.read_csv(REFERENCE_DIR / "no_pbc_fit_data.csv")
    covariate = torch.tensor(data["x"].to_numpy(), dtype=torch.float64)
    expected_basis = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "no_pbc_basis_reference.csv").to_numpy(),
        dtype=torch.float64,
    )
    expected_penalty = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "no_pbc_penalty_reference.csv").to_numpy(),
        dtype=torch.float64,
    )
    term = CyclicPSpline.from_data(
        covariate,
        smoothing_parameter=12.0,
        intervals=12,
    )

    assert term.coefficients.numel() == 12
    assert term.penalty_nullity == 2
    assert term.linear_design_overlap == 1
    torch.testing.assert_close(
        term.basis(covariate),
        expected_basis,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        term.penalty_matrix(),
        expected_penalty,
        rtol=1e-12,
        atol=1e-12,
    )
    period = term.upper_bound - term.lower_bound
    probes = covariate[7:12]
    torch.testing.assert_close(term.basis(probes), term.basis(probes + period))
    torch.testing.assert_close(term.basis(probes), term.basis(probes - period))


def test_cyclic_pspline_target_df_and_state_follow_r_pbc_semantics():
    covariate = torch.linspace(0.0, 2.0 * torch.pi, 120, dtype=torch.float64)
    term = CyclicPSpline.from_data(covariate, degrees_of_freedom=4.0, intervals=12)

    assert term.estimates_smoothing_parameter
    assert term.smoothing_method == "DF"
    assert term.target_effective_degrees_of_freedom == pytest.approx(5.0)
    term._set_fitted_smoothing_parameter(3.25)
    restored = CyclicPSpline.from_data(
        covariate,
        degrees_of_freedom=4.0,
        intervals=12,
    )
    restored.load_state_dict(term.state_dict())

    assert restored.smoothing_parameter == pytest.approx(3.25)


def test_shrinkage_pspline_exposes_pbz_double_penalty_and_constraint():
    covariate = torch.linspace(-1.0, 1.0, 120, dtype=torch.float64)
    term = ShrinkagePSpline.from_data(
        covariate,
        smoothing_parameters=(4.0, 2.0),
        estimate_smoothing=False,
        intervals=8,
    )
    reference = PSpline.from_data(
        covariate,
        smoothing_parameter=4.0,
        intervals=8,
    )
    penalties = term.penalty_matrices()
    first_difference = torch.diff(
        torch.eye(term.coefficients.numel(), dtype=torch.float64),
        dim=0,
    )

    assert term.smoothing_parameters == (4.0, 2.0)
    assert term.estimated_smoothing_parameters == (False, False)
    assert term.penalty_nullity == 1
    assert term.linear_design_overlap == 0
    assert len(penalties) == 2
    torch.testing.assert_close(
        penalties[0],
        reference.penalty_matrices()[0],
    )
    torch.testing.assert_close(
        penalties[1],
        first_difference.mT @ first_difference,
    )
    constraint = term.constraints(covariate)
    assert constraint.shape == (1, term.coefficients.numel())
    torch.testing.assert_close(
        constraint,
        term.design(covariate).sum(dim=0, keepdim=True),
    )


def test_thin_plate_fixed_smoother_and_prediction_match_mgcv():
    data = pd.read_csv(REFERENCE_DIR / "tp_fit_data.csv")
    prediction_data = pd.read_csv(REFERENCE_DIR / "tp_prediction_data.csv")
    expected_fit = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "tp_fixed_fitted_reference.csv")[
            "fitted"
        ].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "tp_fixed_prediction_reference.csv")[
            "fitted"
        ].to_numpy(),
        dtype=torch.float64,
    )
    covariates = torch.tensor(
        data[["x", "z"]].to_numpy(),
        dtype=torch.float64,
    )
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    new_covariates = torch.tensor(
        prediction_data[["x", "z"]].to_numpy(),
        dtype=torch.float64,
    )
    term = ThinPlateSpline.from_data(
        covariates,
        smoothing_parameter=2.5,
        basis_size=15,
        penalty_order=2,
    )
    smooth_design = term.design(covariates)
    design = torch.cat(
        (
            torch.ones((covariates.shape[0], 1), dtype=torch.float64),
            smooth_design,
        ),
        dim=1,
    )
    penalty = torch.zeros((design.shape[1], design.shape[1]), dtype=torch.float64)
    penalty[1:, 1:] = term.penalty_matrices()[0]
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        (penalty,),
        (2.5,),
    )
    prediction_design = torch.cat(
        (
            torch.ones((new_covariates.shape[0], 1), dtype=torch.float64),
            term.predict_design(new_covariates),
        ),
        dim=1,
    )

    assert term.coefficients.numel() == 14
    assert term.penalty_nullity == 2
    assert term.linear_design_overlap == 0
    assert term.smoothing_method is None
    assert term.constraints(covariates).shape == (0, 14)
    assert float(smooth_design.sum(dim=0).abs().max()) < 1e-12
    assert float(torch.linalg.eigvalsh(term.penalty_matrices()[0]).min()) > -1e-12
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fit,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-11,
        atol=1e-11,
    )


def test_thin_plate_univariate_defaults_state_and_single_row_prediction():
    covariate = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    term = ThinPlateSpline.from_data(covariate)

    assert term.dimension == 1
    assert term.basis_size == 10
    assert term.penalty_order == 2
    assert term.penalty_nullity == 1
    assert term.estimates_smoothing_parameter
    assert term.smoothing_method == "LAML"
    assert term.design(covariate).shape == (80, 9)
    assert term.predict_design(covariate[:1]).shape == (1, 9)

    term._set_fitted_smoothing_parameter(3.25)
    restored = ThinPlateSpline.from_data(covariate)
    restored.load_state_dict(term.state_dict())
    assert restored.smoothing_parameter == pytest.approx(3.25)


def test_adaptive_pspline_fixed_smoother_and_prediction_match_mgcv():
    data = pd.read_csv(REFERENCE_DIR / "adaptive_fit_data.csv")
    prediction_data = pd.read_csv(
        REFERENCE_DIR / "adaptive_prediction_data.csv"
    )
    expected_fit = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "adaptive_fixed_fitted_reference.csv")[
            "fitted"
        ].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / "adaptive_fixed_prediction_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    covariate = torch.tensor(data["x"].to_numpy(), dtype=torch.float64)
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    new_covariate = torch.tensor(
        prediction_data["x"].to_numpy(),
        dtype=torch.float64,
    )
    smoothing_parameters = (0.4, 1.2, 4.0, 1.8, 0.6)
    term = AdaptivePSpline.from_data(
        covariate,
        smoothing_parameters=smoothing_parameters,
        basis_size=20,
        penalty_basis_size=5,
    )
    smooth_design = term.design(covariate)
    design = torch.cat(
        (
            torch.ones((covariate.numel(), 1), dtype=torch.float64),
            smooth_design,
        ),
        dim=1,
    )
    penalties = []
    for smooth_penalty in term.penalty_matrices():
        penalty = torch.zeros(
            (design.shape[1], design.shape[1]),
            dtype=torch.float64,
        )
        penalty[1:, 1:] = smooth_penalty
        penalties.append(penalty)
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        penalties,
        smoothing_parameters,
    )
    prediction_design = torch.cat(
        (
            torch.ones((new_covariate.numel(), 1), dtype=torch.float64),
            term.predict_design(new_covariate),
        ),
        dim=1,
    )

    assert term.coefficients.numel() == 19
    assert len(term.penalty_matrices()) == 5
    assert term.smoothing_parameters == smoothing_parameters
    assert term.estimated_smoothing_parameters == (False,) * 5
    assert term.penalty_nullity == 1
    assert term.linear_design_overlap == 0
    assert term.constraints(covariate).shape == (0, 19)
    assert float(smooth_design.sum(dim=0).abs().max()) < 1e-12
    effective_degrees_of_freedom = term.effective_degrees_of_freedom(
        covariate,
        torch.ones_like(covariate),
    )
    assert 1.0 < float(effective_degrees_of_freedom) < 19.0
    for penalty in term.penalty_matrices():
        assert float(torch.linalg.eigvalsh(penalty).min()) > -1e-12
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fit,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-10,
        atol=1e-10,
    )


def test_adaptive_pspline_automatic_state_and_validation():
    covariate = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    term = AdaptivePSpline.from_data(
        covariate,
        initial_smoothing_parameters=(1.0, 2.0, 3.0),
        basis_size=12,
        penalty_basis_size=3,
    )

    assert term.estimates_smoothing_parameter
    assert term.smoothing_parameters == (1.0, 2.0, 3.0)
    assert term.estimated_smoothing_parameters == (True, True, True)
    term._set_fitted_smoothing_parameters((2.5, 4.0, 6.0))
    restored = AdaptivePSpline.from_data(
        covariate,
        initial_smoothing_parameters=(1.0, 2.0, 3.0),
        basis_size=12,
        penalty_basis_size=3,
    )
    restored.load_state_dict(term.state_dict())
    assert restored.smoothing_parameters == pytest.approx((2.5, 4.0, 6.0))

    with pytest.raises(ValueError, match="one value per"):
        AdaptivePSpline.from_data(
            covariate,
            smoothing_parameters=(1.0, 2.0),
            basis_size=12,
            penalty_basis_size=3,
        )


def test_adaptive_tensor_pspline_fixed_smoother_and_prediction_match_mgcv():
    data = pd.read_csv(REFERENCE_DIR / "adaptive_2d_fit_data.csv")
    prediction_data = pd.read_csv(
        REFERENCE_DIR / "adaptive_2d_prediction_data.csv"
    )
    reference = pd.read_csv(
        REFERENCE_DIR / "adaptive_2d_fixed_reference.csv"
    ).iloc[0]
    expected_fitted = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / "adaptive_2d_fixed_fitted_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / "adaptive_2d_fixed_prediction_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    covariates = torch.tensor(
        data[["x", "z"]].to_numpy(),
        dtype=torch.float64,
    )
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    new_covariates = torch.tensor(
        prediction_data[["x", "z"]].to_numpy(),
        dtype=torch.float64,
    )
    smoothing_parameters = tuple(
        float(reference[f"lambda_{index}"])
        for index in range(int(reference["penalty_count"]))
    )
    term = AdaptiveTensorPSpline.from_data(
        covariates,
        smoothing_parameters=smoothing_parameters,
        basis_size=(7, 6),
        penalty_basis_size=(3, 3),
    )
    smooth_design = term.design(covariates)
    design = torch.cat(
        (
            torch.ones((covariates.shape[0], 1), dtype=torch.float64),
            smooth_design,
        ),
        dim=1,
    )
    penalties = []
    for smooth_penalty in term.penalty_matrices():
        penalty = torch.zeros(
            (design.shape[1], design.shape[1]),
            dtype=torch.float64,
        )
        penalty[1:, 1:] = smooth_penalty
        penalties.append(penalty)
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        tuple(penalties),
        smoothing_parameters,
    )
    prediction_design = torch.cat(
        (
            torch.ones((new_covariates.shape[0], 1), dtype=torch.float64),
            term.predict_design(new_covariates),
        ),
        dim=1,
    )

    assert term.basis_sizes == (7, 6)
    assert term.penalty_basis_shape == (3, 3)
    assert smooth_design.shape == (covariates.shape[0], 41)
    assert len(term.penalty_matrices()) == 9
    assert term.penalty_nullity == int(reference["penalty_nullity"]) - 1
    torch.testing.assert_close(
        smooth_design.sum(dim=0),
        torch.zeros(41, dtype=torch.float64),
        rtol=0,
        atol=5e-13,
    )
    for penalty in term.penalty_matrices():
        assert float(torch.linalg.eigvalsh(penalty).min()) >= -1e-12
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fitted,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-10,
        atol=1e-10,
    )


def test_adaptive_tensor_pspline_automatic_state_and_validation():
    x = torch.linspace(-1.0, 1.0, 70, dtype=torch.float64)
    covariates = torch.column_stack(
        (x, torch.sin(torch.arange(70, dtype=torch.float64)))
    )
    term = AdaptiveTensorPSpline.from_data(
        covariates,
        initial_smoothing_parameters=(1.0,) * 9,
        basis_size=(7, 6),
        penalty_basis_size=(3, 3),
    )

    assert term.estimates_smoothing_parameter
    assert term.estimated_smoothing_parameters == (True,) * 9
    term._set_fitted_smoothing_parameters((2.0,) * 9)
    restored = AdaptiveTensorPSpline.from_data(
        covariates,
        initial_smoothing_parameters=(1.0,) * 9,
        basis_size=(7, 6),
        penalty_basis_size=(3, 3),
    )
    restored.load_state_dict(term.state_dict())
    assert restored.smoothing_parameters == pytest.approx((2.0,) * 9)

    with pytest.raises(ValueError, match="one value per"):
        AdaptiveTensorPSpline.from_data(
            covariates,
            smoothing_parameters=(1.0,) * 8,
            basis_size=(7, 6),
            penalty_basis_size=(3, 3),
        )


def test_mrf_fixed_smoother_and_prediction_match_mgcv():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    prediction_data = pd.read_csv(REFERENCE_DIR / "mrf_prediction_data.csv")
    reference = pd.read_csv(REFERENCE_DIR / "mrf_fixed_reference.csv").iloc[0]
    expected_fitted = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "mrf_fixed_fitted_reference.csv")[
            "fitted"
        ].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(REFERENCE_DIR / "mrf_fixed_prediction_reference.csv")[
            "fitted"
        ].to_numpy(),
        dtype=torch.float64,
    )
    regions = torch.tensor(data["region"].to_numpy(), dtype=torch.float64)
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    new_regions = torch.tensor(
        prediction_data["region"].to_numpy(),
        dtype=torch.float64,
    )
    smoothing_parameter = float(reference["lambda"])
    term = MarkovRandomField.from_data(
        regions,
        MRF_EDGES,
        smoothing_parameter=smoothing_parameter,
        area_count=9,
    )
    smooth_design = term.design(regions)
    design = torch.cat(
        (
            torch.ones((regions.numel(), 1), dtype=torch.float64),
            smooth_design,
        ),
        dim=1,
    )
    penalty = torch.zeros(
        (design.shape[1], design.shape[1]),
        dtype=torch.float64,
    )
    penalty[1:, 1:] = term.penalty_matrices()[0]
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        (penalty,),
        (smoothing_parameter,),
    )
    prediction_design = torch.cat(
        (
            torch.ones((new_regions.numel(), 1), dtype=torch.float64),
            term.predict_design(new_regions),
        ),
        dim=1,
    )

    assert term.area_count == int(reference["area_count"])
    assert len(term.edges) == int(reference["edge_count"])
    assert smooth_design.shape == (regions.numel(), 8)
    assert term.penalty_nullity == int(reference["penalty_nullity"])
    torch.testing.assert_close(
        smooth_design.sum(dim=0),
        torch.zeros(8, dtype=torch.float64),
        rtol=0,
        atol=1e-13,
    )
    assert float(torch.linalg.eigvalsh(term.penalty_matrices()[0]).min()) > 0
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fitted,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mrf_low_rank_smoother_and_prediction_match_mgcv():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    prediction_data = pd.read_csv(REFERENCE_DIR / "mrf_prediction_data.csv")
    reference = pd.read_csv(
        REFERENCE_DIR / "mrf_low_rank_fixed_reference.csv"
    ).iloc[0]
    expected_fitted = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / "mrf_low_rank_fixed_fitted_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / "mrf_low_rank_fixed_prediction_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    regions = torch.tensor(data["region"].to_numpy(), dtype=torch.float64)
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    new_regions = torch.tensor(
        prediction_data["region"].to_numpy(),
        dtype=torch.float64,
    )
    smoothing_parameter = float(reference["lambda"])
    term = MarkovRandomField.from_data(
        regions,
        MRF_EDGES,
        smoothing_parameter=smoothing_parameter,
        area_count=9,
        basis_size=int(reference["requested_basis_size"]),
    )
    design = torch.cat(
        (
            torch.ones((regions.numel(), 1), dtype=torch.float64),
            term.design(regions),
        ),
        dim=1,
    )
    penalty = torch.zeros(
        (design.shape[1], design.shape[1]),
        dtype=torch.float64,
    )
    penalty[1:, 1:] = term.penalty_matrices()[0]
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        (penalty,),
        (smoothing_parameter,),
    )
    prediction_design = torch.cat(
        (
            torch.ones((new_regions.numel(), 1), dtype=torch.float64),
            term.predict_design(new_regions),
        ),
        dim=1,
    )

    assert term.basis_size == int(reference["requested_basis_size"])
    assert term.coefficients.numel() == int(reference["basis_size"])
    assert term.penalty_nullity == int(reference["penalty_nullity"])
    assert int(torch.linalg.matrix_rank(term.penalty_matrices()[0])) == int(
        reference["penalty_rank"]
    )
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fitted,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mrf_named_adjacency_preserves_numeric_mgcv_parameterization():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    codes = torch.tensor(data["region"].to_numpy(), dtype=torch.float64)
    labels = tuple(MRF_LABELS[int(code)] for code in data["region"])
    numeric = MarkovRandomField.from_data(
        codes,
        MRF_EDGES,
        smoothing_parameter=1.7,
        area_count=9,
        basis_size=6,
    )
    labelled = MarkovRandomField.from_labels(
        labels,
        MRF_ADJACENCY,
        smoothing_parameter=1.7,
        basis_size=6,
    )

    assert labelled.region_labels == MRF_LABELS
    assert labelled.label_edges == tuple(
        (MRF_LABELS[first], MRF_LABELS[second])
        for first, second in sorted(MRF_EDGES)
    )
    torch.testing.assert_close(
        labelled.basis_labels(labels),
        numeric.design(codes),
    )
    torch.testing.assert_close(
        labelled.penalty_matrices()[0],
        numeric.penalty_matrices()[0],
    )
    prediction_labels = tuple(reversed(MRF_LABELS))
    prediction_codes = torch.arange(8, -1, -1, dtype=torch.float64)
    torch.testing.assert_close(
        labelled.predict_design_labels(prediction_labels),
        numeric.predict_design(prediction_codes),
    )
    with pytest.raises(ValueError, match="unknown region label"):
        labelled.encode_regions(("center", "off_graph"))
    with pytest.raises(ValueError, match="unknown region"):
        MarkovRandomField.from_labels(
            ("north", "south"),
            {"north": ("outside",), "south": ("north",)},
        )
    with pytest.raises(ValueError, match="unique"):
        MarkovRandomField.from_labels(
            ("north", "south"),
            (("north", "south"),),
            levels=("north", "north"),
        )


@pytest.mark.parametrize(
    ("basis_size", "reference_prefix"),
    ((9, "mrf_penalty"), (6, "mrf_penalty_low_rank")),
)
def test_mrf_supplied_penalty_matches_mgcv(
    basis_size,
    reference_prefix,
):
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    prediction_data = pd.read_csv(REFERENCE_DIR / "mrf_prediction_data.csv")
    reference = pd.read_csv(
        REFERENCE_DIR / f"{reference_prefix}_fixed_reference.csv"
    ).iloc[0]
    expected_fitted = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / f"{reference_prefix}_fixed_fitted_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / f"{reference_prefix}_fixed_prediction_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    regions = torch.tensor(data["region"].to_numpy(), dtype=torch.float64)
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    new_regions = torch.tensor(
        prediction_data["region"].to_numpy(),
        dtype=torch.float64,
    )
    smoothing_parameter = float(reference["lambda"])
    term = MarkovRandomField.from_data(
        regions,
        graph_penalty=_mrf_graph_penalty(),
        smoothing_parameter=smoothing_parameter,
        basis_size=basis_size,
    )
    design = torch.cat(
        (
            torch.ones((regions.numel(), 1), dtype=torch.float64),
            term.design(regions),
        ),
        dim=1,
    )
    penalty = torch.zeros(
        (design.shape[1], design.shape[1]),
        dtype=torch.float64,
    )
    penalty[1:, 1:] = term.penalty_matrices()[0]
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        (penalty,),
        (smoothing_parameter,),
    )
    prediction_design = torch.cat(
        (
            torch.ones((new_regions.numel(), 1), dtype=torch.float64),
            term.predict_design(new_regions),
        ),
        dim=1,
    )

    assert term.uses_supplied_penalty
    assert term.edges == ()
    assert term.basis_size == int(reference["requested_basis_size"])
    assert term.coefficients.numel() == int(reference["basis_size"])
    assert term.penalty_nullity == int(reference["penalty_nullity"])
    assert int(torch.linalg.matrix_rank(term.penalty_matrices()[0])) == int(
        reference["penalty_rank"]
    )
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fitted,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mrf_supplied_penalty_labels_and_validation():
    penalty = _mrf_graph_penalty()
    labels = tuple(MRF_LABELS[index] for index in range(9)) * 2
    labelled = MarkovRandomField.from_labels(
        labels,
        graph_penalty=penalty,
        levels=MRF_LABELS,
        basis_size=6,
    )
    numeric = MarkovRandomField.from_data(
        torch.arange(9, dtype=torch.float64).repeat(2),
        graph_penalty=penalty,
        basis_size=6,
    )

    assert labelled.region_labels == MRF_LABELS
    assert labelled.uses_supplied_penalty
    torch.testing.assert_close(
        labelled.basis_labels(labels),
        numeric.design(torch.arange(9, dtype=torch.float64).repeat(2)),
    )
    with pytest.raises(ValueError, match="exactly one"):
        MarkovRandomField.from_data(
            torch.arange(9, dtype=torch.float64),
            MRF_EDGES,
            graph_penalty=penalty,
        )
    with pytest.raises(ValueError, match="symmetric"):
        asymmetric = penalty.clone()
        asymmetric[0, 1] -= 0.2
        MarkovRandomField.from_data(
            torch.arange(9, dtype=torch.float64),
            graph_penalty=asymmetric,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        indefinite = penalty.clone()
        indefinite[0, 0] = -10.0
        MarkovRandomField.from_data(
            torch.arange(9, dtype=torch.float64),
            graph_penalty=indefinite,
        )


def test_polygon_adjacency_matches_mgcv_and_supports_multiple_components():
    expected = pd.read_csv(
        REFERENCE_DIR / "mrf_polygon_adjacency_reference.csv"
    )
    adjacency = polygon_adjacency(MRF_POLYGONS)
    actual_edges = tuple(
        (first, second)
        for first, neighbors in adjacency.items()
        for second in neighbors
        if first < second
    )

    assert actual_edges == tuple(expected.itertuples(index=False, name=None))
    assert adjacency["area_0"] == ("area_1",)
    assert adjacency["area_5"] == ("area_4",)

    components = {
        "islands": (
            ((10.0, 0.0), (11.0, 0.0), (10.5, 0.8)),
            ((0.0, 0.0), (1.0, 0.0), (0.5, 0.8), (0.0, 0.0)),
        ),
        "mainland": ((1.0, 0.0), (2.0, 0.0), (1.5, 0.8)),
    }
    assert polygon_adjacency(components) == {
        "islands": ("mainland",),
        "mainland": ("islands",),
    }

    with pytest.raises(ValueError, match="three unique vertices"):
        polygon_adjacency(
            {
                "invalid": ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
                "valid": ((1.0, 0.0), (2.0, 0.0), (1.5, 0.8)),
            }
        )


@pytest.mark.parametrize(
    ("basis_size", "reference_prefix"),
    ((6, "mrf_polygon"), (4, "mrf_polygon_low_rank")),
)
def test_mrf_polygon_smoother_and_prediction_match_mgcv(
    basis_size,
    reference_prefix,
):
    data = pd.read_csv(REFERENCE_DIR / "mrf_polygon_fit_data.csv")
    prediction_data = pd.read_csv(
        REFERENCE_DIR / "mrf_polygon_prediction_data.csv"
    )
    reference = pd.read_csv(
        REFERENCE_DIR / f"{reference_prefix}_fixed_reference.csv"
    ).iloc[0]
    expected_fitted = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / f"{reference_prefix}_fixed_fitted_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    expected_prediction = torch.tensor(
        pd.read_csv(
            REFERENCE_DIR / f"{reference_prefix}_fixed_prediction_reference.csv"
        )["fitted"].to_numpy(),
        dtype=torch.float64,
    )
    smoothing_parameter = float(reference["lambda"])
    term = MarkovRandomField.from_polygons(
        tuple(data["region"]),
        MRF_POLYGONS,
        smoothing_parameter=smoothing_parameter,
        basis_size=basis_size,
    )
    regions = term.encode_regions(tuple(data["region"]))
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    design = torch.cat(
        (
            torch.ones((regions.numel(), 1), dtype=torch.float64),
            term.design(regions),
        ),
        dim=1,
    )
    penalty = torch.zeros(
        (design.shape[1], design.shape[1]),
        dtype=torch.float64,
    )
    penalty[1:, 1:] = term.penalty_matrices()[0]
    fit = solve_penalized_least_squares(
        design,
        response,
        torch.ones_like(response),
        (penalty,),
        (smoothing_parameter,),
    )
    prediction_design = torch.cat(
        (
            torch.ones((len(prediction_data), 1), dtype=torch.float64),
            term.predict_design_labels(tuple(prediction_data["region"])),
        ),
        dim=1,
    )

    assert term.uses_polygon_adjacency
    assert not term.uses_supplied_penalty
    assert term.region_labels == tuple(MRF_POLYGONS)
    assert len(term.edges) == int(reference["edge_count"])
    assert term.coefficients.numel() == int(reference["basis_size"])
    assert term.penalty_nullity == int(reference["penalty_nullity"])
    torch.testing.assert_close(
        design @ fit.coefficients,
        expected_fitted,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        prediction_design @ fit.coefficients,
        expected_prediction,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mrf_automatic_state_and_edge_validation():
    regions = torch.arange(9, dtype=torch.float64).repeat(3)
    term = MarkovRandomField.from_data(
        regions,
        MRF_EDGES,
        initial_smoothing_parameter=2.0,
        area_count=9,
    )

    assert term.estimates_smoothing_parameter
    assert term.smoothing_parameter == pytest.approx(2.0)
    term._set_fitted_smoothing_parameter(3.5)
    restored = MarkovRandomField.from_data(
        regions,
        MRF_EDGES,
        initial_smoothing_parameter=2.0,
        area_count=9,
    )
    restored.load_state_dict(term.state_dict())
    assert restored.smoothing_parameter == pytest.approx(3.5)

    with pytest.raises(ValueError, match="self-loop"):
        MarkovRandomField.from_data(regions, ((0, 0),), area_count=9)
    with pytest.raises(ValueError, match="repeated MRF eigenvalue"):
        MarkovRandomField.from_data(
            regions,
            MRF_EDGES,
            area_count=9,
            basis_size=5,
        )

    observed_regions = torch.arange(8, dtype=torch.float64).repeat(3)
    low_rank_with_missing = MarkovRandomField.from_data(
        observed_regions,
        MRF_EDGES,
        area_count=9,
        basis_size=6,
    )
    missing_prediction = low_rank_with_missing.predict_design(
        torch.tensor([8.0], dtype=torch.float64)
    )
    assert missing_prediction.shape == (1, 5)
    assert torch.isfinite(missing_prediction).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_bound": 1.0, "upper_bound": 1.0, "smoothing_parameter": 1.0},
        {"lower_bound": 0.0, "upper_bound": 1.0, "smoothing_parameter": -1.0},
        {
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "smoothing_parameter": 1.0,
            "intervals": 0,
        },
    ],
)
def test_invalid_pspline_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PSpline(**kwargs)


def test_pspline_from_data_rejects_a_constant_covariate():
    with pytest.raises(ValueError, match="distinct"):
        PSpline.from_data(torch.ones(10, dtype=torch.float64), 1.0)


def test_automatic_pspline_smoothing_state_is_serialized():
    covariate = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
    term = PSpline.from_data(covariate)
    assert term.smoothing_parameter == pytest.approx(10.0)
    term._set_fitted_smoothing_parameter(3.25)
    restored = PSpline.from_data(covariate)

    restored.load_state_dict(term.state_dict())

    assert restored.estimates_smoothing_parameter
    assert restored.smoothing_method == "ML"
    assert restored.smoothing_parameter == pytest.approx(3.25)


@pytest.mark.parametrize("method", ["GAIC", "GCV"])
def test_pspline_supports_criterion_smoothing_selection(method):
    covariate = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
    term = PSpline.from_data(
        covariate,
        smoothing_method=method,
        criterion_penalty=3.0,
    )

    assert term.estimates_smoothing_parameter
    assert term.smoothing_method == method
    assert term.criterion_penalty == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_smoothing_parameter": 0.0},
        {"initial_smoothing_parameter": float("inf")},
        {"smoothing_method": "REML"},
        {"criterion_penalty": 0.0},
        {"criterion_penalty": float("inf")},
    ],
)
def test_invalid_automatic_smoothing_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PSpline(0.0, 1.0, None, **kwargs)


def test_pspline_target_degrees_of_freedom_uses_r_pb_semantics():
    covariate = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    term = PSpline.from_data(covariate, degrees_of_freedom=3.0)

    assert term.estimates_smoothing_parameter
    assert term.smoothing_method == "DF"
    assert term.target_effective_degrees_of_freedom == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothing_parameter": 1.0, "degrees_of_freedom": 3.0},
        {"degrees_of_freedom": -1.0},
        {"degrees_of_freedom": float("inf")},
        {"degrees_of_freedom": 21.0},
    ],
)
def test_invalid_target_degrees_of_freedom_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PSpline(0.0, 1.0, **kwargs)
