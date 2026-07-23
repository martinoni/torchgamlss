import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import GAMLSS, Normal, PSpline, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _read_rows(name: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return list(csv.DictReader(data_file))


def _column(rows: list[dict[str, str]], name: str) -> torch.Tensor:
    return torch.tensor([float(row[name]) for row in rows], dtype=torch.float64)


def _weighted_offset_problem():
    rows = _read_rows("no_rs_fit_data.csv")
    reference = _read_rows("no_rs_reference.csv")[0]
    x = _column(rows, "x")
    z = _column(rows, "z")
    response = _column(rows, "y")
    weights = _column(rows, "weight")
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.column_stack((torch.ones_like(z), z)),
    }
    offsets = {
        "mu": _column(rows, "mu_offset"),
        "sigma": _column(rows, "sigma_offset"),
    }
    return response, weights, design, offsets, reference


def test_rs_with_weights_and_offsets_matches_r_gamlss():
    response, weights, design, offsets, reference = _weighted_offset_problem()
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 2}, dtype=torch.float64)

    result = model.fit_rs(
        response,
        design,
        weights=weights,
        offsets=offsets,
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert all(
        current <= previous + 1e-12
        for previous, current in zip(
            result.deviance_history, result.deviance_history[1:]
        )
    )
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor(
            [
                float(reference["sigma_intercept"]),
                float(reference["sigma_z"]),
            ],
            dtype=torch.float64,
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-11, abs=1e-11
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-11, abs=1e-11
    )


def test_lbfgs_with_weights_and_offsets_matches_r_gamlss():
    response, weights, design, offsets, reference = _weighted_offset_problem()
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 2}, dtype=torch.float64)

    result = model.fit(
        response,
        design,
        weights=weights,
        offsets=offsets,
        max_iter=300,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    assert result.converged
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor(
            [
                float(reference["sigma_intercept"]),
                float(reference["sigma_z"]),
            ],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-10, abs=1e-10
    )


def test_rs_fixed_pspline_matches_r_gamlss_pb():
    rows = _read_rows("no_pb_fit_data.csv")
    reference = _read_rows("no_pb_reference.csv")[0]
    fitted_reference = _read_rows("no_pb_fitted_reference.csv")
    coefficient_reference = _read_rows("no_pb_coefficient_reference.csv")
    x = _column(rows, "x")
    response = _column(rows, "y")
    term = PSpline.from_data(x, smoothing_parameter=12.0, intervals=10)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }
    smooth_covariates = {"mu": {"x": x}}

    result = model.fit_rs(
        response,
        design,
        smooth_covariates=smooth_covariates,
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert result.backfitting_iterations["mu"] > 0
    assert result.smoothing_parameters["mu"]["x"] == pytest.approx(12.0)
    assert result.smoothing_iterations["mu"]["x"] == 0
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor([float(reference["sigma_intercept"])], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        term.coefficients,
        _column(coefficient_reference, "coefficient"),
        rtol=2e-7,
        atol=2e-7,
    )
    predictors = model.linear_predictors(design, smooth_covariates=smooth_covariates)
    parameters = model.family.parameters_from_predictors(predictors)
    torch.testing.assert_close(
        parameters["mu"],
        _column(fitted_reference, "mu"),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        parameters["sigma"],
        _column(fitted_reference, "sigma"),
        rtol=1e-7,
        atol=1e-8,
    )
    torch.testing.assert_close(
        term(x),
        _column(fitted_reference, "mu_smooth"),
        rtol=2e-7,
        atol=2e-7,
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-10, abs=1e-10
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-10, abs=1e-10
    )


def test_rs_ml_pspline_smoothing_parameter_matches_r_gamlss_pb():
    rows = _read_rows("no_pb_fit_data.csv")
    reference = _read_rows("no_pb_ml_reference.csv")[0]
    fitted_reference = _read_rows("no_pb_ml_fitted_reference.csv")
    coefficient_reference = _read_rows("no_pb_ml_coefficient_reference.csv")
    x = _column(rows, "x")
    response = _column(rows, "y")
    term = PSpline.from_data(x)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }
    smooth_covariates = {"mu": {"x": x}}

    result = model.fit_rs(
        response,
        design,
        smooth_covariates=smooth_covariates,
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert result.smoothing_iterations["mu"]["x"] > 0
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor([float(reference["sigma_intercept"])], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        term.coefficients,
        _column(coefficient_reference, "coefficient"),
        rtol=2e-7,
        atol=2e-7,
    )
    predictors = model.linear_predictors(design, smooth_covariates=smooth_covariates)
    parameters = model.family.parameters_from_predictors(predictors)
    torch.testing.assert_close(
        parameters["mu"],
        _column(fitted_reference, "mu"),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        term(x),
        _column(fitted_reference, "mu_smooth"),
        rtol=2e-7,
        atol=2e-7,
    )
    expected_smoothing_parameter = float(reference["smoothing_parameter"])
    assert term.smoothing_parameter == pytest.approx(
        expected_smoothing_parameter, rel=1e-8, abs=1e-8
    )
    assert result.smoothing_parameters["mu"]["x"] == pytest.approx(
        expected_smoothing_parameter, rel=1e-8, abs=1e-8
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-8, abs=1e-8
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-9, abs=1e-9
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-9, abs=1e-9
    )


def test_rs_target_edf_pspline_matches_r_gamlss_pb():
    rows = _read_rows("no_pb_fit_data.csv")
    reference = _read_rows("no_pb_df_reference.csv")[0]
    fitted_reference = _read_rows("no_pb_df_fitted_reference.csv")
    coefficient_reference = _read_rows("no_pb_df_coefficient_reference.csv")
    x = _column(rows, "x")
    response = _column(rows, "y")
    term = PSpline.from_data(x, degrees_of_freedom=float(reference["requested_df"]))
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }
    smooth_covariates = {"mu": {"x": x}}

    result = model.fit_rs(
        response,
        design,
        smooth_covariates=smooth_covariates,
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    assert result.converged
    assert reference["converged"] == "TRUE"
    assert result.smoothing_iterations["mu"]["x"] > 0
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor([float(reference["sigma_intercept"])], dtype=torch.float64),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        term.coefficients,
        _column(coefficient_reference, "coefficient"),
        rtol=3e-7,
        atol=3e-7,
    )
    predictors = model.linear_predictors(design, smooth_covariates=smooth_covariates)
    parameters = model.family.parameters_from_predictors(predictors)
    torch.testing.assert_close(
        parameters["mu"],
        _column(fitted_reference, "mu"),
        rtol=3e-7,
        atol=3e-7,
    )
    expected_lambda = float(reference["smoothing_parameter"])
    assert term.smoothing_parameter == pytest.approx(
        expected_lambda, rel=1e-8, abs=2e-7
    )
    assert result.smoothing_parameters["mu"]["x"] == pytest.approx(
        expected_lambda, rel=1e-8, abs=2e-7
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["target_edf"]), rel=1e-7, abs=1e-7
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-8, abs=1e-7
    )


def test_rs_supports_multiple_smooth_terms_for_one_parameter():
    x = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    z = torch.cos(torch.linspace(0.0, 3.0 * torch.pi, 80, dtype=torch.float64))
    response = (
        0.5
        + torch.sin(torch.pi * x)
        + 0.4 * torch.cos(torch.pi * z)
        + 0.08 * torch.sin(19.0 * x)
    )
    x_term = PSpline.from_data(x, smoothing_parameter=8.0)
    z_term = PSpline.from_data(z, smoothing_parameter=8.0)
    model = GAMLSS(
        Normal(),
        {"mu": 3, "sigma": 1},
        smooth_terms={"mu": {"x": x_term, "z": z_term}},
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x, z)),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }

    result = model.fit_rs(
        response,
        design,
        smooth_covariates={"mu": {"x": x, "z": z}},
    )

    assert result.converged
    assert set(result.smooth_effective_degrees_of_freedom["mu"]) == {"x", "z"}
    assert torch.isfinite(x_term.coefficients).all()
    assert torch.isfinite(z_term.coefficients).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_tolerance": 0.0},
        {"inner_tolerance": 0.0},
        {"max_outer_iterations": 0},
        {"max_inner_iterations": 0},
        {"backfitting_tolerance": 0.0},
        {"max_backfitting_iterations": 0},
        {"smoothing_tolerance": 0.0},
        {"max_smoothing_iterations": 0},
        {"edf_tolerance": 0.0},
        {"max_edf_iterations": 0},
        {"step": 0.0},
        {"step": 1.1},
        {"deviance_tolerance": -1.0},
    ],
)
def test_invalid_rs_control_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RSControl(**kwargs)


def test_rs_rejects_rank_deficient_design_matrix():
    response = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    repeated = torch.ones((3, 2), dtype=torch.float64)
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=torch.float64)
    design = {
        "mu": repeated,
        "sigma": torch.ones((3, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="rank deficient"):
        model.fit_rs(response, design)
