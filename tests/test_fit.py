import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import GAMLSS, Normal

REFERENCE_DIR = Path(__file__).parent / "reference"


def _read_fit_data() -> tuple[torch.Tensor, torch.Tensor]:
    with (REFERENCE_DIR / "no_fit_data.csv").open(
        newline="", encoding="utf-8"
    ) as data_file:
        rows = list(csv.DictReader(data_file))
    x = torch.tensor([float(row["x"]) for row in rows], dtype=torch.float64)
    y = torch.tensor([float(row["y"]) for row in rows], dtype=torch.float64)
    return x, y


def _read_fit_reference() -> dict[str, float]:
    with (REFERENCE_DIR / "no_fit_reference.csv").open(
        newline="", encoding="utf-8"
    ) as reference_file:
        row = next(csv.DictReader(reference_file))
    return {
        key: float(row[key])
        for key in (
            "mu_intercept",
            "mu_x",
            "sigma_intercept",
            "negative_log_likelihood",
        )
    }


def test_joint_lbfgs_fit_matches_r_gamlss_no():
    x, response = _read_fit_data()
    reference = _read_fit_reference()
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=torch.float64)
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }

    result = model.fit(
        response,
        design,
        max_iter=200,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    assert result.converged
    assert result.gradient_max < 1e-6
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [reference["mu_intercept"], reference["mu_x"]],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor([reference["sigma_intercept"]], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result.negative_log_likelihood == pytest.approx(
        reference["negative_log_likelihood"], rel=1e-10, abs=1e-10
    )


def test_fit_rejects_non_positive_iteration_limit():
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)
    response = torch.tensor([0.0], dtype=torch.float64)
    design = {
        "mu": torch.ones((1, 1), dtype=torch.float64),
        "sigma": torch.ones((1, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="max_iter"):
        model.fit(response, design, max_iter=0)
