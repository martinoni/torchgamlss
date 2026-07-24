import csv
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import BCT, GAMLSS, Beta, CGControl, Normal, Poisson

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference(name: str) -> dict[str, str]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return next(csv.DictReader(data_file))


def _case_rows(name: str, case: str) -> pd.DataFrame:
    rows = pd.read_csv(REFERENCE_DIR / name)
    return rows.loc[rows["case"] == case].reset_index(drop=True)


def _assert_coefficients(model, reference, columns):
    for parameter, parameter_columns in columns.items():
        torch.testing.assert_close(
            model.coefficients[parameter],
            torch.tensor(
                [float(reference[column]) for column in parameter_columns],
                dtype=torch.float64,
            ),
            rtol=2e-10,
            atol=2e-10,
        )


def test_cg_beta_formula_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "be_fit_data.csv")
    reference = _reference("be_cg_reference.csv")
    model = GAMLSS.from_formula(
        Beta(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    result = model.fit_cg_data(
        data,
        weights="weight",
        control=CGControl(
            outer_tolerance=1e-9,
            max_outer_iterations=200,
            inner_tolerance=1e-9,
            max_inner_iterations=200,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert len(result.inner_iterations) == result.outer_iterations
    assert all(iterations >= 1 for iterations in result.inner_iterations)
    assert result.effective_degrees_of_freedom == pytest.approx(4.0)
    assert all(
        current <= previous + 1e-12
        for previous, current in zip(
            result.deviance_history,
            result.deviance_history[1:],
        )
    )
    _assert_coefficients(
        model,
        reference,
        {
            "mu": ("mu_intercept", "mu_x"),
            "sigma": ("sigma_intercept", "sigma_z"),
        },
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=2e-12,
        abs=2e-12,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]),
        rel=2e-12,
        abs=2e-12,
    )


def test_cg_four_parameter_bct_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "bct_fit_data.csv")
    reference = _reference("bct_cg_reference.csv")
    model = GAMLSS.from_formula(
        BCT(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
            "nu": "~ w + offset(nu_offset)",
            "tau": "~ 1",
        },
        data,
    )
    initial_parameters = {
        "mu": 3.0 + 0.7 * data["x"] + data["mu_offset"],
        "sigma": (-1.5 + 0.18 * data["z"] + data["sigma_offset"]).map(math.exp),
        "nu": 0.35 + 0.2 * data["w"] + data["nu_offset"],
        "tau": 4.0,
    }

    result = model.fit_cg_data(
        data,
        weights="weight",
        initial_parameters=initial_parameters,
        control=CGControl(
            outer_tolerance=1e-9,
            max_outer_iterations=200,
            inner_tolerance=1e-9,
            max_inner_iterations=200,
            autostep=False,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert len(result.inner_iterations) == result.outer_iterations
    assert result.effective_degrees_of_freedom == pytest.approx(7.0)
    _assert_coefficients(
        model,
        reference,
        {
            "mu": ("mu_intercept", "mu_x"),
            "sigma": ("sigma_intercept", "sigma_z"),
            "nu": ("nu_intercept", "nu_w"),
            "tau": ("tau_intercept",),
        },
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=2e-11,
        abs=2e-10,
    )
    predictions = model.predict_data(data)
    assert tuple(predictions) == ("mu", "sigma", "nu", "tau")
    assert all(values.shape == (len(data),) for values in predictions.values())


def test_cg_supports_a_one_parameter_family():
    response = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((4, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    result = model.fit_cg(
        response,
        design,
        control=CGControl(
            outer_tolerance=1e-10,
            inner_tolerance=1e-10,
            max_outer_iterations=100,
            max_inner_iterations=100,
        ),
    )

    assert result.converged
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor([math.log(1.5)], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_tolerance": 0.0},
        {"outer_tolerance": float("nan")},
        {"inner_tolerance": 0.0},
        {"inner_tolerance": float("inf")},
        {"backfitting_tolerance": 0.0},
        {"backfitting_tolerance": float("nan")},
        {"smoothing_tolerance": 0.0},
        {"smoothing_tolerance": float("inf")},
        {"edf_tolerance": 0.0},
        {"edf_tolerance": float("nan")},
        {"criterion_tolerance": 0.0},
        {"criterion_tolerance": float("inf")},
        {"max_outer_iterations": 0},
        {"max_inner_iterations": 0},
        {"max_smoothing_iterations": 0},
        {"max_edf_iterations": 0},
        {"max_criterion_iterations": 0},
        {"mu_step": 0.0},
        {"sigma_step": 1.1},
        {"nu_step": -0.1},
        {"tau_step": 2.0},
        {"deviance_tolerance": -1.0},
        {"deviance_tolerance": float("nan")},
    ],
)
def test_cg_control_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CGControl(**kwargs)


def test_cg_fixed_pspline_with_cross_derivatives_matches_r_gamlss_pb():
    case = "BE_FIXED"
    data = pd.read_csv(REFERENCE_DIR / "be_fit_data.csv")
    reference = _case_rows("cg_smooth_reference.csv", case).iloc[0]
    linear_reference = _case_rows("cg_smooth_linear_reference.csv", case)
    fitted_reference = _case_rows("cg_smooth_fitted_reference.csv", case)
    coefficient_reference = _case_rows("cg_smooth_coefficient_reference.csv", case)
    model = GAMLSS.from_formula(
        Beta(),
        {
            "mu": "y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    result = model.fit_cg_data(
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

    term = model.smooth_terms["mu"]["x"]
    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert result.backfitting_iterations["mu"] > 0
    assert result.smoothing_iterations["mu"]["x"] == 0
    assert result.smoothing_parameters["mu"]["x"] == pytest.approx(12.0)
    for parameter, rows in linear_reference.groupby("parameter", sort=False):
        expected = torch.tensor(
            rows.sort_values("coefficient_index")["coefficient"].to_numpy(),
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            model.coefficients[parameter], expected, rtol=1e-8, atol=1e-8
        )
    torch.testing.assert_close(
        term.coefficients,
        torch.tensor(
            coefficient_reference["coefficient"].to_numpy(), dtype=torch.float64
        ),
        rtol=1e-8,
        atol=1e-8,
    )
    predictions = model.predict_data(data)
    torch.testing.assert_close(
        predictions["mu"],
        torch.tensor(fitted_reference["mu"].to_numpy(), dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        predictions["sigma"],
        torch.tensor(fitted_reference["sigma"].to_numpy(), dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    x = torch.tensor(data["x"].to_numpy(), dtype=torch.float64)
    torch.testing.assert_close(
        term(x),
        torch.tensor(fitted_reference["mu_smooth"].to_numpy(), dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-8, abs=1e-8
    )
    assert result.parameter_effective_degrees_of_freedom["mu"] == pytest.approx(
        float(reference["mu_df"]), rel=1e-8, abs=1e-8
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["total_df"]), rel=1e-8, abs=1e-8
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-10, abs=1e-9
    )


@pytest.mark.parametrize(
    (
        "case",
        "mu_formula",
        "lambda_tolerance",
        "deviance_tolerance",
        "same_outer_iterations",
    ),
    [
        ("NO_FIXED", "y ~ pb(x, smoothing_parameter=12)", 1e-8, 1e-8, True),
        ("NO_ML", "y ~ pb(x)", 1e-7, 1e-7, True),
        (
            "NO_DF",
            "y ~ pb(x, degrees_of_freedom=3)",
            1e-5,
            2e-6,
            False,
        ),
        (
            "NO_GAIC",
            "y ~ pb(x, smoothing_method='GAIC', criterion_penalty=2)",
            2e-5,
            2e-6,
            False,
        ),
        (
            "NO_GCV",
            "y ~ pb(x, smoothing_method='GCV', criterion_penalty=2)",
            2e-4,
            2e-5,
            False,
        ),
    ],
)
def test_cg_normal_pspline_selection_matches_r_gamlss_pb(
    case,
    mu_formula,
    lambda_tolerance,
    deviance_tolerance,
    same_outer_iterations,
):
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    reference = _case_rows("cg_smooth_reference.csv", case).iloc[0]
    linear_reference = _case_rows("cg_smooth_linear_reference.csv", case)
    fitted_reference = _case_rows("cg_smooth_fitted_reference.csv", case)
    coefficient_reference = _case_rows("cg_smooth_coefficient_reference.csv", case)
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": mu_formula, "sigma": "~ 1"},
        data,
    )

    result = model.fit_cg_data(
        data,
        control=CGControl(
            outer_tolerance=1e-8,
            max_outer_iterations=300,
            inner_tolerance=1e-8,
            max_inner_iterations=300,
            backfitting_tolerance=1e-8,
        ),
    )

    term = model.smooth_terms["mu"]["x"]
    assert result.converged
    if same_outer_iterations:
        assert result.outer_iterations == int(reference["outer_iterations"])
    assert result.backfitting_iterations["mu"] > 0
    if case == "NO_FIXED":
        assert result.smoothing_iterations["mu"]["x"] == 0
    else:
        assert result.smoothing_iterations["mu"]["x"] > 0
    assert result.smoothing_parameters["mu"]["x"] == pytest.approx(
        float(reference["smoothing_parameter"]),
        rel=5e-7,
        abs=lambda_tolerance,
    )
    assert term.smoothing_parameter == pytest.approx(
        float(reference["smoothing_parameter"]),
        rel=5e-7,
        abs=lambda_tolerance,
    )
    for parameter, rows in linear_reference.groupby("parameter", sort=False):
        expected = torch.tensor(
            rows.sort_values("coefficient_index")["coefficient"].to_numpy(),
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            model.coefficients[parameter], expected, rtol=5e-6, atol=5e-6
        )
    torch.testing.assert_close(
        term.coefficients,
        torch.tensor(
            coefficient_reference["coefficient"].to_numpy(), dtype=torch.float64
        ),
        rtol=5e-6,
        atol=5e-6,
    )
    predictions = model.predict_data(data)
    torch.testing.assert_close(
        predictions["mu"],
        torch.tensor(fitted_reference["mu"].to_numpy(), dtype=torch.float64),
        rtol=5e-6,
        atol=5e-6,
    )
    torch.testing.assert_close(
        predictions["sigma"],
        torch.tensor(fitted_reference["sigma"].to_numpy(), dtype=torch.float64),
        rtol=5e-6,
        atol=5e-6,
    )
    x = torch.tensor(data["x"].to_numpy(), dtype=torch.float64)
    torch.testing.assert_close(
        term(x),
        torch.tensor(fitted_reference["mu_smooth"].to_numpy(), dtype=torch.float64),
        rtol=5e-6,
        atol=5e-6,
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=5e-7, abs=5e-6
    )
    assert result.parameter_effective_degrees_of_freedom["mu"] == pytest.approx(
        float(reference["mu_df"]), rel=5e-7, abs=5e-6
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["total_df"]), rel=5e-7, abs=5e-6
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=5e-7,
        abs=deviance_tolerance,
    )


@pytest.mark.parametrize(
    ("case", "data_name", "formulas", "fit_tolerance", "iteration_limit"),
    [
        (
            "BOTH_PARAMETERS",
            "be_fit_data.csv",
            {
                "mu": "y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)",
                "sigma": ("~ pb(z, smoothing_parameter=15) + offset(sigma_offset)"),
            },
            1e-9,
            500,
        ),
        (
            "TWO_MU_TERMS",
            "be_fit_data.csv",
            {
                "mu": (
                    "y ~ pb(x, smoothing_parameter=12)"
                    " + pb(z, smoothing_parameter=15) + offset(mu_offset)"
                ),
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-9,
            500,
        ),
        (
            "BOTH_ML",
            "cg_multi_smooth_fit_data.csv",
            {
                "mu": "y ~ pb(x) + offset(mu_offset)",
                "sigma": "~ pb(z) + offset(sigma_offset)",
            },
            1e-8,
            300,
        ),
    ],
    ids=["smooths-in-mu-and-sigma", "two-mu-smooths", "two-ml-smooths"],
)
def test_cg_multiple_psplines_match_r_gamlss_additive_fit(
    case,
    data_name,
    formulas,
    fit_tolerance,
    iteration_limit,
):
    data = pd.read_csv(REFERENCE_DIR / data_name)
    reference = _case_rows("cg_multi_smooth_reference.csv", case).iloc[0]
    linear_reference = _case_rows("cg_multi_smooth_linear_reference.csv", case)
    fitted_reference = _case_rows("cg_multi_smooth_fitted_reference.csv", case)
    term_reference = _case_rows("cg_multi_smooth_term_reference.csv", case)
    coefficient_reference = _case_rows(
        "cg_multi_smooth_coefficient_reference.csv", case
    )
    contribution_reference = _case_rows(
        "cg_multi_smooth_contribution_reference.csv", case
    )
    model = GAMLSS.from_formula(Beta(), formulas, data)

    result = model.fit_cg_data(
        data,
        weights="weight",
        control=CGControl(
            outer_tolerance=fit_tolerance,
            max_outer_iterations=iteration_limit,
            inner_tolerance=fit_tolerance,
            max_inner_iterations=iteration_limit,
            backfitting_tolerance=fit_tolerance,
        ),
    )

    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    for parameter in model.family.parameter_names:
        expected_terms = set(
            term_reference.loc[
                term_reference["parameter"] == parameter, "term"
            ].tolist()
        )
        assert set(result.smoothing_parameters[parameter]) == expected_terms
        if expected_terms:
            assert result.backfitting_iterations[parameter] > 0
        else:
            assert result.backfitting_iterations[parameter] == 0

    for parameter, rows in linear_reference.groupby("parameter", sort=False):
        expected = torch.tensor(
            rows.sort_values("coefficient_index")["coefficient"].to_numpy(),
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            model.coefficients[parameter], expected, rtol=2e-6, atol=2e-6
        )

    for term_row in term_reference.itertuples():
        term = model.smooth_terms[term_row.parameter][term_row.term]
        assert result.smoothing_parameters[term_row.parameter][
            term_row.term
        ] == pytest.approx(float(term_row.smoothing_parameter), rel=1e-7, abs=1e-7)
        if term_row.selection == "FIXED":
            assert result.smoothing_iterations[term_row.parameter][term_row.term] == 0
        else:
            assert result.smoothing_iterations[term_row.parameter][term_row.term] > 0
        assert result.smooth_effective_degrees_of_freedom[term_row.parameter][
            term_row.term
        ] == pytest.approx(float(term_row.smooth_edf), rel=1e-8, abs=1e-8)

        expected_coefficients = coefficient_reference.loc[
            (coefficient_reference["parameter"] == term_row.parameter)
            & (coefficient_reference["term"] == term_row.term)
        ].sort_values("coefficient_index")
        torch.testing.assert_close(
            term.coefficients,
            torch.tensor(
                expected_coefficients["coefficient"].to_numpy(),
                dtype=torch.float64,
            ),
            rtol=2e-6,
            atol=2e-6,
        )

        expected_contribution = contribution_reference.loc[
            (contribution_reference["parameter"] == term_row.parameter)
            & (contribution_reference["term"] == term_row.term)
        ].sort_values("observation_index")
        covariate = torch.tensor(data[term_row.term].to_numpy(), dtype=torch.float64)
        torch.testing.assert_close(
            term(covariate),
            torch.tensor(
                expected_contribution["contribution"].to_numpy(),
                dtype=torch.float64,
            ),
            rtol=2e-6,
            atol=2e-6,
        )

    predictions = model.predict_data(data)
    torch.testing.assert_close(
        predictions["mu"],
        torch.tensor(fitted_reference["mu"].to_numpy(), dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        predictions["sigma"],
        torch.tensor(fitted_reference["sigma"].to_numpy(), dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert result.parameter_effective_degrees_of_freedom["mu"] == pytest.approx(
        float(reference["mu_df"]), rel=1e-8, abs=1e-8
    )
    assert result.parameter_effective_degrees_of_freedom["sigma"] == pytest.approx(
        float(reference["sigma_df"]), rel=1e-8, abs=1e-8
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["total_df"]), rel=1e-8, abs=1e-8
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-10, abs=1e-9
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-10, abs=1e-9
    )
