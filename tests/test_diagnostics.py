import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    Beta,
    Gamma,
    ModelDiagnostics,
    NegativeBinomial,
    Normal,
    Poisson,
    PSpline,
    RSControl,
    compare_models,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _rows(name: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return list(csv.DictReader(data_file))


def _diagnostic_row(family: str) -> dict[str, str]:
    return next(
        row
        for row in _rows("model_diagnostics_reference.csv")
        if row["family"] == family
    )


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
            "GA",
            "ga",
            Gamma(),
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
def test_model_diagnostics_match_r_gamlss(
    family_code, prefix, family, formulas, tolerance
):
    data = pd.read_csv(REFERENCE_DIR / f"{prefix}_fit_data.csv")
    model = GAMLSS.from_formula(family, formulas, data)
    fit = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=tolerance,
            max_outer_iterations=200,
            inner_tolerance=tolerance,
            max_inner_iterations=200,
        ),
    )
    result = model.diagnostics_data(data, weights="weight")
    reference = _diagnostic_row(family_code)

    assert fit.effective_degrees_of_freedom == pytest.approx(
        float(reference["effective_df"])
    )
    assert result.observation_count == int(reference["observation_count"])
    assert result.effective_observation_count == pytest.approx(
        float(reference["effective_observation_count"])
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["effective_df"])
    )
    assert result.residual_degrees_of_freedom == pytest.approx(
        float(reference["residual_df"])
    )
    assert result.log_likelihood == pytest.approx(
        float(reference["log_likelihood"]), rel=2e-7, abs=2e-7
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=2e-7, abs=2e-7
    )
    assert result.aic == pytest.approx(float(reference["aic"]), rel=2e-7, abs=2e-7)
    assert result.aicc == pytest.approx(float(reference["aicc"]), rel=2e-7, abs=2e-7)
    assert result.gaic(3) == pytest.approx(
        float(reference["gaic3"]), rel=2e-7, abs=2e-7
    )
    assert result.bic == pytest.approx(float(reference["sbc"]), rel=2e-7, abs=2e-7)
    assert result.sbc == result.bic


def test_smooth_model_effective_degrees_of_freedom_match_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    x = torch.tensor(data["x"].to_numpy(), dtype=torch.float64)
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
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
    fit = model.fit_rs(
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
    reference = _diagnostic_row("NO_PB")

    assert fit.parameter_effective_degrees_of_freedom["mu"] == pytest.approx(
        fit.smooth_effective_degrees_of_freedom["mu"]["x"]
    )
    assert fit.parameter_effective_degrees_of_freedom["sigma"] == 1.0
    assert fit.effective_degrees_of_freedom == pytest.approx(
        float(reference["effective_df"]), rel=1e-9, abs=1e-9
    )
    diagnostics = model.diagnostics(
        response,
        design,
        smooth_covariates=smooth_covariates,
        degrees_of_freedom=fit.effective_degrees_of_freedom,
    )
    assert diagnostics.residual_degrees_of_freedom == pytest.approx(
        float(reference["residual_df"]), rel=1e-9, abs=1e-9
    )
    assert diagnostics.aic == pytest.approx(float(reference["aic"]), rel=1e-9, abs=1e-9)
    assert diagnostics.aicc == pytest.approx(
        float(reference["aicc"]), rel=1e-9, abs=1e-9
    )
    assert diagnostics.bic == pytest.approx(float(reference["sbc"]), rel=1e-9, abs=1e-9)

    with pytest.raises(ValueError, match="degrees_of_freedom"):
        model.diagnostics(
            response,
            design,
            smooth_covariates=smooth_covariates,
        )


@pytest.mark.parametrize(
    ("family_code", "reference_name", "family"),
    [
        ("NO", "no_reference.csv", Normal()),
        ("GA", "ga_reference.csv", Gamma()),
        ("PO", "po_reference.csv", Poisson()),
        ("NBI", "nbi_reference.csv", NegativeBinomial()),
        ("BE", "be_reference.csv", Beta()),
    ],
)
def test_quantile_residuals_match_r_gamlss(family_code, reference_name, family):
    family_rows = _rows(reference_name)
    residual_rows = [
        row
        for row in _rows("quantile_residual_reference.csv")
        if row["family"] == family_code
    ]
    response = torch.tensor(
        [float(row["y"]) for row in family_rows],
        dtype=torch.float64,
    )
    parameters = {
        parameter: torch.tensor(
            [float(row[parameter]) for row in family_rows],
            dtype=torch.float64,
        )
        for parameter in family.parameter_names
    }
    expected_probability = torch.tensor(
        [float(row["probability"]) for row in residual_rows],
        dtype=torch.float64,
    )
    expected_residual = torch.tensor(
        [float(row["residual"]) for row in residual_rows],
        dtype=torch.float64,
    )
    observation_count = response.numel()
    model = GAMLSS(
        family,
        {parameter: observation_count for parameter in family.parameter_names},
        dtype=torch.float64,
    )
    design = {
        parameter: torch.eye(observation_count, dtype=torch.float64)
        for parameter in family.parameter_names
    }
    with torch.no_grad():
        for parameter in family.parameter_names:
            model.coefficients[parameter].copy_(
                family.links[parameter](parameters[parameter])
            )

    if family.is_discrete:
        uniforms = torch.tensor(
            [float(row["uniform"]) for row in residual_rows],
            dtype=torch.float64,
        )
        lower = family.cdf(response - 1, parameters)
        upper = family.cdf(response, parameters)
        probability = lower + uniforms * (upper - lower)
    else:
        uniforms = None
        probability = family.cdf(response, parameters)

    torch.testing.assert_close(
        probability,
        expected_probability,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        model.quantile_residuals(response, design, uniforms=uniforms),
        expected_residual,
        rtol=2e-11,
        atol=2e-11,
    )


def test_formula_quantile_residuals_accept_uniform_column():
    data = pd.DataFrame(
        {
            "y": [0.0, 1.0, 3.0, 5.0],
            "u": [0.1, 0.3, 0.7, 0.9],
        }
    )
    model = GAMLSS.from_formula(Poisson(), {"mu": "y ~ 1"}, data)
    model.fit_rs_data(data)

    from_column = model.quantile_residuals_data(data, uniforms="u")
    from_values = model.quantile_residuals_data(data, uniforms=data["u"])

    torch.testing.assert_close(from_column, from_values)


def test_discrete_quantile_residual_generator_is_reproducible():
    response = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    first = model.quantile_residuals(
        response,
        design,
        generator=torch.Generator().manual_seed(2026),
    )
    second = model.quantile_residuals(
        response,
        design,
        generator=torch.Generator().manual_seed(2026),
    )

    torch.testing.assert_close(first, second)


def test_quantile_residuals_validate_randomization_inputs():
    response = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    discrete = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    continuous = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        dtype=torch.float64,
    )
    normal_design = {"mu": design["mu"], "sigma": design["mu"]}

    with pytest.raises(ValueError, match="either uniforms"):
        discrete.quantile_residuals(
            response,
            design,
            uniforms=torch.full_like(response, 0.5),
            generator=torch.Generator(),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        discrete.quantile_residuals(
            response,
            design,
            uniforms=torch.tensor([0.2, 1.1, 0.5], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="discrete"):
        continuous.quantile_residuals(
            response,
            normal_design,
            uniforms=torch.full_like(response, 0.5),
        )
    with pytest.raises(ValueError, match="discrete"):
        continuous.quantile_residuals(
            response,
            normal_design,
            generator=torch.Generator(),
        )


def test_model_comparison_ranks_models_and_calculates_weights():
    diagnostics = {
        "larger": ModelDiagnostics(-45.0, 90.0, 5.0, 95.0, 100, 100.0),
        "smaller": ModelDiagnostics(-46.0, 92.0, 2.0, 98.0, 100, 100.0),
    }

    table = compare_models(diagnostics)

    assert tuple(table.index) == ("smaller", "larger")
    assert table.loc["smaller", "delta"] == 0
    assert table["weight"].sum() == pytest.approx(1.0)
    assert tuple(compare_models(diagnostics, criterion="gaic", penalty=3).index) == (
        "smaller",
        "larger",
    )


def test_diagnostics_and_model_comparison_reject_invalid_inputs():
    result = ModelDiagnostics(-1.0, 2.0, 2.0, 1.0, 3, 3.0)
    incomparable = ModelDiagnostics(-1.0, 2.0, 2.0, 2.0, 4, 4.0)

    assert result.aicc == float("inf")
    with pytest.raises(ValueError, match="penalty"):
        result.gaic(float("nan"))
    with pytest.raises(ValueError, match="at least one"):
        compare_models({})
    with pytest.raises(ValueError, match="comparable"):
        compare_models({"a": result, "b": incomparable})
    with pytest.raises(ValueError, match="finite"):
        compare_models({"a": result}, criterion="aicc")
