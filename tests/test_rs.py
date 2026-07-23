import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import GAMLSS, Normal, RSControl

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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_tolerance": 0.0},
        {"inner_tolerance": 0.0},
        {"max_outer_iterations": 0},
        {"max_inner_iterations": 0},
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
