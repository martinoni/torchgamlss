import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import GAMLSS, Gamma, PSpline, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _read_rows(name: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return list(csv.DictReader(data_file))


def _column(rows: list[dict[str, str]], name: str) -> torch.Tensor:
    return torch.tensor([float(row[name]) for row in rows], dtype=torch.float64)


def _gamma_problem():
    rows = _read_rows("ga_fit_data.csv")
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
    return rows, x, response, weights, design, offsets


def _expected_coefficients(reference: dict[str, str]) -> dict[str, torch.Tensor]:
    return {
        "mu": torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        "sigma": torch.tensor(
            [
                float(reference["sigma_intercept"]),
                float(reference["sigma_z"]),
            ],
            dtype=torch.float64,
        ),
    }


def test_ga_weighted_rs_fit_with_offsets_matches_r_gamlss():
    _, _, response, weights, design, offsets = _gamma_problem()
    reference = _read_rows("ga_rs_reference.csv")[0]
    expected = _expected_coefficients(reference)
    model = GAMLSS(Gamma(), {"mu": 2, "sigma": 2}, dtype=torch.float64)

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
    torch.testing.assert_close(model.coefficients["mu"], expected["mu"])
    torch.testing.assert_close(model.coefficients["sigma"], expected["sigma"])
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-12, abs=1e-12
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-12, abs=1e-12
    )


def test_ga_joint_lbfgs_fit_matches_r_gamlss():
    _, _, response, weights, design, offsets = _gamma_problem()
    reference = _read_rows("ga_rs_reference.csv")[0]
    expected = _expected_coefficients(reference)
    model = GAMLSS(Gamma(), {"mu": 2, "sigma": 2}, dtype=torch.float64)

    result = model.fit(
        response,
        design,
        weights=weights,
        offsets=offsets,
        max_iter=500,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    assert result.converged
    assert result.gradient_max < 1e-6
    torch.testing.assert_close(
        model.coefficients["mu"], expected["mu"], rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        model.coefficients["sigma"], expected["sigma"], rtol=1e-6, atol=1e-7
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-12, abs=1e-12
    )


def test_ga_fixed_pspline_rs_fit_matches_r_gamlss():
    _, x, response, weights, design, offsets = _gamma_problem()
    reference = _read_rows("ga_pb_reference.csv")[0]
    fitted_reference = _read_rows("ga_pb_fitted_reference.csv")
    coefficient_reference = _read_rows("ga_pb_coefficient_reference.csv")
    expected = _expected_coefficients(reference)
    term = PSpline.from_data(x, smoothing_parameter=12.0)
    model = GAMLSS(
        Gamma(),
        {"mu": 2, "sigma": 2},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    smooth_covariates = {"mu": {"x": x}}

    result = model.fit_rs(
        response,
        design,
        weights=weights,
        offsets=offsets,
        smooth_covariates=smooth_covariates,
        control=RSControl(
            outer_tolerance=1e-9,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    torch.testing.assert_close(
        model.coefficients["mu"], expected["mu"], rtol=1e-5, atol=5e-6
    )
    torch.testing.assert_close(model.coefficients["sigma"], expected["sigma"])
    torch.testing.assert_close(
        term.coefficients,
        _column(coefficient_reference, "coefficient"),
        rtol=1e-4,
        atol=1e-5,
    )
    parameters = model.predict(
        design,
        offsets,
        smooth_covariates=smooth_covariates,
        type="response",
    )
    torch.testing.assert_close(
        parameters["mu"],
        _column(fitted_reference, "mu"),
        rtol=1e-10,
        atol=2e-12,
    )
    torch.testing.assert_close(
        parameters["sigma"],
        _column(fitted_reference, "sigma"),
        rtol=1e-10,
        atol=2e-12,
    )
    assert result.smoothing_parameters["mu"]["x"] == pytest.approx(12.0)
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-12, abs=1e-12
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-12, abs=1e-11
    )
