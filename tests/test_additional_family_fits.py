import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import GAMLSS, Beta, NegativeBinomial, Poisson, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference_row(name: str) -> dict[str, str]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return next(csv.DictReader(data_file))


@pytest.mark.parametrize(
    ("family", "prefix", "formulas", "tolerance"),
    [
        (
            Poisson(),
            "po",
            {"mu": "y ~ x + offset(mu_offset)"},
            1e-10,
        ),
        (
            NegativeBinomial(),
            "nbi",
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            2e-7,
        ),
        (
            Beta(),
            "be",
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            2e-7,
        ),
    ],
)
def test_formula_rs_fits_match_r_gamlss(family, prefix, formulas, tolerance):
    data = pd.read_csv(REFERENCE_DIR / f"{prefix}_fit_data.csv")
    reference = _reference_row(f"{prefix}_rs_reference.csv")
    model = GAMLSS.from_formula(family, formulas, data)

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-9 if prefix != "po" else 1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-9 if prefix != "po" else 1e-10,
            max_inner_iterations=200,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=tolerance,
        atol=tolerance,
    )
    if "sigma" in family.parameter_names:
        torch.testing.assert_close(
            model.coefficients["sigma"],
            torch.tensor(
                [
                    float(reference["sigma_intercept"]),
                    float(reference["sigma_z"]),
                ],
                dtype=torch.float64,
            ),
            rtol=tolerance,
            atol=tolerance,
        )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=tolerance,
        abs=tolerance,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]),
        rel=tolerance,
        abs=tolerance,
    )
    prediction = model.predict_data(data)
    assert set(prediction) == set(family.parameter_names)
    assert all(torch.isfinite(value).all() for value in prediction.values())
