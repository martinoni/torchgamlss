import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    BCCG,
    BCPE,
    BCT,
    GAMLSS,
    Beta,
    CGControl,
    NegativeBinomial,
    Normal,
    Poisson,
    PSpline,
    RSControl,
    SmoothInferenceResult,
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


def _conditional_table_rows(case: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / "conditional_inference_table_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return [row for row in csv.DictReader(data_file) if row["case"] == case]


def _conditional_covariance(case: str, size: int) -> torch.Tensor:
    covariance = torch.zeros((size, size), dtype=torch.float64)
    with (REFERENCE_DIR / "conditional_inference_covariance_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        for row in csv.DictReader(data_file):
            if row["case"] == case:
                covariance[int(row["row_index"]), int(row["column_index"])] = float(
                    row["covariance"]
                )
    return covariance


def _smooth_table_rows(
    case: str,
    parameter: str,
    term: str,
) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_DIR / "smooth_inference_reference.csv")
    return reference.loc[
        (reference["case"] == case)
        & (reference["parameter"] == parameter)
        & (reference["term"] == term)
    ].sort_values("observation_index")


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
        (
            "BCCG",
            "bccg",
            BCCG(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
            },
            1e-9,
        ),
        (
            "BCT",
            "bct",
            BCT(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
                "tau": "~ 1",
            },
            1e-7,
        ),
        (
            "BCPE",
            "bcpe",
            BCPE(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
                "tau": "~ 1",
            },
            1e-7,
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
    # R's optimHess uses finite differences. Its BCPE Hessian is slightly less
    # accurate than our autograd Hessian even though the fitted coefficients
    # agree to machine precision.
    if family_code == "BCPE":
        covariance_tolerances = {"rtol": 8e-3, "atol": 1e-4}
        derived_tolerances = {"rtol": 8e-3, "atol": 2e-6}
        p_value_tolerances = {"rtol": 2e-3, "atol": 3e-7}
    else:
        covariance_tolerances = {"rtol": 5e-6, "atol": 5e-7}
        derived_tolerances = {"rtol": 5e-6, "atol": 5e-7}
        p_value_tolerances = {"rtol": 2e-4, "atol": 3e-7}
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
        **covariance_tolerances,
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        **derived_tolerances,
    )
    torch.testing.assert_close(
        result.statistics,
        expected["statistic"],
        rtol=max(1e-5, derived_tolerances["rtol"]),
        atol=max(2e-6, derived_tolerances["atol"]),
    )
    torch.testing.assert_close(
        result.p_values,
        expected["p_value"],
        **p_value_tolerances,
    )
    torch.testing.assert_close(
        result.confidence_intervals,
        torch.column_stack((expected["ci_lower"], expected["ci_upper"])),
        rtol=derived_tolerances["rtol"],
        atol=max(1e-6, derived_tolerances["atol"]),
    )
    torch.testing.assert_close(
        result.correlation_matrix,
        result.correlation_matrix.mT,
    )
    torch.testing.assert_close(
        torch.diagonal(result.correlation_matrix),
        torch.ones(len(rows), dtype=torch.float64),
    )
    assert not result.conditional_on_smooths

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


@pytest.mark.parametrize("case", ["NO_FIXED_RS", "NO_ML_RS", "BE_FIXED_CG"])
def test_conditional_smooth_inference_matches_r_gamlss_vcov(case):
    if case == "BE_FIXED_CG":
        data = pd.read_csv(REFERENCE_DIR / "be_fit_data.csv")
        model = GAMLSS.from_formula(
            Beta(),
            {
                "mu": "y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            data,
        )
        model.fit_cg_data(
            data,
            weights="weight",
            control=CGControl(
                outer_tolerance=1e-7,
                max_outer_iterations=300,
                inner_tolerance=1e-7,
                max_inner_iterations=300,
                backfitting_tolerance=1e-7,
            ),
        )
        weights = "weight"
    else:
        data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
        mu_formula = (
            "y ~ pb(x, smoothing_parameter=12)"
            if case == "NO_FIXED_RS"
            else "y ~ pb(x)"
        )
        model = GAMLSS.from_formula(
            Normal(),
            {"mu": mu_formula, "sigma": "~ 1"},
            data,
        )
        model.fit_rs_data(
            data,
            control=RSControl(
                outer_tolerance=1e-10,
                max_outer_iterations=200,
                inner_tolerance=1e-10,
                max_inner_iterations=200,
                backfitting_tolerance=1e-10,
                max_backfitting_iterations=200,
            ),
        )
        weights = None

    rows = _conditional_table_rows(case)
    degrees_of_freedom = float(rows[0]["degrees_of_freedom"])
    result = model.inference_data(
        data,
        weights=weights,
        conditional_on_smooths=True,
        degrees_of_freedom=degrees_of_freedom,
    )
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

    assert result.conditional_on_smooths
    expected_names = {
        "NO_FIXED_RS": (
            "mu.Intercept",
            "mu.pb(x, smoothing_parameter=12)",
            "sigma.Intercept",
        ),
        "NO_ML_RS": ("mu.Intercept", "mu.pb(x)", "sigma.Intercept"),
        "BE_FIXED_CG": (
            "mu.Intercept",
            "mu.pb(x, smoothing_parameter=12)",
            "sigma.Intercept",
            "sigma.z",
        ),
    }
    assert result.coefficient_names == expected_names[case]
    assert result.degrees_of_freedom == pytest.approx(degrees_of_freedom)
    torch.testing.assert_close(
        result.estimates,
        expected["estimate"],
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.covariance_matrix,
        _conditional_covariance(case, len(rows)),
        rtol=1e-5,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        rtol=1e-5,
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
        rtol=1e-5,
        atol=1e-6,
    )


def test_inference_requires_explicit_conditioning_for_smooth_models():
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

    with pytest.raises(ValueError, match="conditional_on_smooths"):
        model.inference(torch.sin(x), design)


def test_conditional_smooth_inference_requires_residual_degrees_of_freedom():
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

    with pytest.raises(ValueError, match="degrees_of_freedom is required"):
        model.inference(
            torch.sin(x),
            design,
            smooth_covariates={"mu": {"x": x}},
            conditional_on_smooths=True,
        )


@pytest.mark.parametrize(
    ("case", "formula"),
    [
        ("NO_FIXED_RS", "y ~ pb(x, smoothing_parameter=12)"),
        ("NO_ML_RS", "y ~ pb(x)"),
    ],
)
def test_smooth_curve_inference_matches_r_gamlss_pb(case, formula):
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": formula, "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    result = model.smooth_inference_data(data)["mu"]["x"]
    reference = _smooth_table_rows(case, "mu", "x")
    expected = {
        column: torch.tensor(reference[column].to_numpy(), dtype=torch.float64)
        for column in (
            "estimate",
            "variance",
            "standard_error",
            "ci_lower",
            "ci_upper",
        )
    }

    assert isinstance(result, SmoothInferenceResult)
    assert result.parameter == "mu"
    assert result.term == "x"
    assert result.confidence_level == pytest.approx(0.95)
    assert result.smoothing_parameter == pytest.approx(
        float(reference["smoothing_parameter"].iloc[0]),
        rel=1e-8,
        abs=1e-8,
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["smooth_edf"].iloc[0]),
        rel=1e-8,
        abs=1e-8,
    )
    torch.testing.assert_close(
        result.estimates,
        expected["estimate"],
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        result.variances,
        expected["variance"],
        rtol=2e-7,
        atol=2e-10,
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        rtol=2e-7,
        atol=2e-9,
    )
    torch.testing.assert_close(
        result.confidence_intervals,
        torch.column_stack((expected["ci_lower"], expected["ci_upper"])),
        rtol=2e-7,
        atol=2e-7,
    )
    table = result.to_dataframe()
    assert tuple(table.columns) == (
        "covariate",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )
    assert len(table) == len(data)


def test_smooth_curve_inference_supports_formula_new_data():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x, smoothing_parameter=12)", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)
    new_data = data.iloc[::4].drop(columns="y")

    result = model.smooth_inference_data(data, new_data=new_data)["mu"]["x"]
    contribution = model.predict_data(new_data, type="terms")["mu"].smooth["x"]

    torch.testing.assert_close(result.estimates, contribution)
    assert result.standard_errors.shape == (len(new_data),)
    assert result.confidence_intervals.shape == (len(new_data), 2)
    assert torch.all(result.confidence_intervals[:, 0] <= result.estimates)
    assert torch.all(result.confidence_intervals[:, 1] >= result.estimates)


def test_smooth_curve_inference_rejects_parametric_models():
    response = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="at least one smooth"):
        model.smooth_inference(
            response,
            design,
            smooth_covariates={"mu": {}},
        )


def test_inference_result_rejects_misaligned_parameter_split():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    design = {"mu": torch.ones((4, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    model.fit_rs(response, design)
    result = model.inference(response, design)

    with pytest.raises(ValueError, match="one row"):
        result.by_parameter(torch.ones(2, dtype=torch.float64))
