import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    Beta,
    NegativeBinomial,
    Normal,
    Poisson,
    PSpline,
    RSControl,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _table_rows(family: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / "inference_table_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return [row for row in csv.DictReader(data_file) if row["family"] == family]


def _covariance(family: str, size: int) -> torch.Tensor:
    covariance = torch.zeros((size, size), dtype=torch.float64)
    with (REFERENCE_DIR / "inference_covariance_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        for row in csv.DictReader(data_file):
            if row["family"] == family:
                covariance[int(row["row_index"]), int(row["column_index"])] = float(
                    row["covariance"]
                )
    return covariance


@pytest.mark.parametrize(
    ("family_code", "prefix", "family", "formulas", "tolerance"),
    [
        (
            "NO",
            "no_rs",
            Normal(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-10,
        ),
        (
            "PO",
            "po",
            Poisson(),
            {"mu": "y ~ x + offset(mu_offset)"},
            1e-10,
        ),
        (
            "NBI",
            "nbi",
            NegativeBinomial(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-9,
        ),
        (
            "BE",
            "be",
            Beta(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-9,
        ),
    ],
)
def test_full_hessian_inference_matches_r_gamlss(
    family_code, prefix, family, formulas, tolerance
):
    data = pd.read_csv(REFERENCE_DIR / f"{prefix}_fit_data.csv")
    model = GAMLSS.from_formula(family, formulas, data)
    model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=tolerance,
            max_outer_iterations=200,
            inner_tolerance=tolerance,
            max_inner_iterations=200,
        ),
    )

    result = model.inference_data(data, weights="weight")
    rows = _table_rows(family_code)
    expected = {
        column: torch.tensor(
            [float(row[column]) for row in rows],
            dtype=torch.float64,
        )
        for column in (
            "estimate",
            "standard_error",
            "statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        )
    }

    assert result.coefficient_names == tuple(row["coefficient"] for row in rows)
    assert result.degrees_of_freedom == pytest.approx(
        float(rows[0]["degrees_of_freedom"])
    )
    torch.testing.assert_close(
        result.estimates,
        expected["estimate"],
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.covariance_matrix,
        _covariance(family_code, len(rows)),
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.statistics,
        expected["statistic"],
        rtol=1e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.p_values,
        expected["p_value"],
        rtol=2e-4,
        atol=3e-7,
    )
    torch.testing.assert_close(
        result.confidence_intervals,
        torch.column_stack((expected["ci_lower"], expected["ci_upper"])),
        rtol=5e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result.correlation_matrix,
        result.correlation_matrix.mT,
    )
    torch.testing.assert_close(
        torch.diagonal(result.correlation_matrix),
        torch.ones(len(rows), dtype=torch.float64),
    )

    split_estimates = result.by_parameter(result.estimates)
    assert tuple(split_estimates) == family.parameter_names
    assert sum(values.numel() for values in split_estimates.values()) == len(rows)
    table = result.to_dataframe()
    assert tuple(table.index) == result.coefficient_names
    assert tuple(table.columns) == (
        "estimate",
        "standard_error",
        "statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
    )


def test_low_level_inference_uses_stable_positional_names():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    design = {"mu": torch.column_stack((torch.ones_like(x), x))}
    model = GAMLSS(Poisson(), {"mu": 2}, dtype=torch.float64)
    model.fit_rs(response, design)

    result = model.inference(response, design)

    assert result.coefficient_names == ("mu[0]", "mu[1]")


def test_inference_degrees_of_freedom_can_be_overridden():
    response = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    weights = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    model.fit_rs(response, design, weights=weights)

    automatic = model.inference(response, design, weights=weights)
    overridden = model.inference(
        response,
        design,
        weights=weights,
        degrees_of_freedom=2.5,
    )

    assert automatic.degrees_of_freedom == pytest.approx(5.0)
    assert overridden.degrees_of_freedom == pytest.approx(2.5)
    torch.testing.assert_close(
        automatic.covariance_matrix,
        overridden.covariance_matrix,
    )
    assert not torch.equal(automatic.p_values, overridden.p_values)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, float("nan")])
def test_inference_rejects_invalid_confidence_levels(confidence_level):
    response = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="confidence_level"):
        model.inference(
            response,
            design,
            confidence_level=confidence_level,
        )


def test_inference_rejects_nonpositive_residual_degrees_of_freedom():
    response = torch.tensor([1.0], dtype=torch.float64)
    design = {"mu": torch.ones((1, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="degrees of freedom"):
        model.inference(response, design)


def test_inference_rejects_singular_hessian():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    design = {
        "mu": torch.column_stack(
            (torch.ones_like(x), torch.ones_like(x)),
        )
    }
    model = GAMLSS(Poisson(), {"mu": 2}, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="positive definite"):
        model.inference(response, design)


def test_inference_rejects_smooth_models_until_joint_uncertainty_is_supported():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    term = PSpline.from_data(x, smoothing_parameter=12.0)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="without smooth"):
        model.inference(torch.sin(x), design)


def test_inference_result_rejects_misaligned_parameter_split():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    design = {"mu": torch.ones((4, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    model.fit_rs(response, design)
    result = model.inference(response, design)

    with pytest.raises(ValueError, match="one row"):
        result.by_parameter(torch.ones(2, dtype=torch.float64))
