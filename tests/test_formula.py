import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import GAMLSS, Gamma, Normal, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference_row(name: str) -> dict[str, str]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return next(csv.DictReader(data_file))


def test_formula_weighted_normal_rs_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_rs_fit_data.csv")
    reference = _reference_row("no_rs_reference.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
        ),
    )

    assert model.formula_response_name == "y"
    assert model.formula_column_names == {
        "mu": ("Intercept", "x"),
        "sigma": ("Intercept", "z"),
    }
    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
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
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-12, abs=1e-12
    )


def test_formula_normal_lbfgs_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_fit_data.csv")
    reference = _reference_row("no_fit_reference.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )

    result = model.fit_data(
        data,
        max_iter=200,
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
            [float(reference["sigma_intercept"])],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-10, abs=1e-10
    )


def test_formula_gamma_pspline_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "ga_fit_data.csv")
    reference = _reference_row("ga_pb_reference.csv")
    model = GAMLSS.from_formula(
        Gamma(),
        {
            "mu": "y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-9,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    term = model.smooth_terms["mu"]["x"]
    assert result.converged
    assert term.smoothing_parameter == pytest.approx(12.0)
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-12, abs=1e-12
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-12, abs=1e-11
    )
    predicted = model.predict_data(data)
    assert torch.isfinite(predicted["mu"]).all()
    assert torch.isfinite(predicted["sigma"]).all()


def test_formula_automatic_ml_pspline_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    reference = _reference_row("no_pb_ml_reference.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x)", "sigma": "~ 1"},
        data,
    )

    result = model.fit_rs_data(
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

    expected_lambda = float(reference["smoothing_parameter"])
    assert result.converged
    assert model.smooth_terms["mu"]["x"].smoothing_parameter == pytest.approx(
        expected_lambda, rel=1e-8, abs=1e-8
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-8, abs=1e-8
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-9, abs=1e-9
    )


def test_formula_reuses_categorical_encoding_for_new_data():
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5, 2.0, 2.4],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "group": ["a", "b", "c", "a", "b", "c"],
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x + C(group)", "sigma": "~ 1"},
        data,
    )
    with torch.no_grad():
        model.coefficients["mu"].copy_(
            torch.tensor([0.5, 0.8, 0.2, -0.3], dtype=torch.float64)
        )
        model.coefficients["sigma"].fill_(-1.0)
    new_data = pd.DataFrame(
        {
            "x": [-0.5, 0.5, 0.8],
            "group": ["c", "a", "b"],
        }
    )

    prepared = model.prepare_formula_data(new_data)
    prediction = model.predict_data(new_data, type="link")

    assert model.formula_column_names["mu"] == (
        "Intercept",
        "x",
        "C(group)[T.b]",
        "C(group)[T.c]",
    )
    torch.testing.assert_close(
        prepared.design_matrices["mu"],
        torch.tensor(
            [
                [1.0, -0.5, 0.0, 1.0],
                [1.0, 0.5, 0.0, 0.0],
                [1.0, 0.8, 1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        prediction["mu"],
        prepared.design_matrices["mu"] @ model.coefficients["mu"],
    )


def test_formula_rejects_unseen_categorical_levels():
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5],
            "group": ["a", "b", "a", "b"],
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ C(group)", "sigma": "~ 1"},
        data,
    )

    with pytest.raises(ValueError, match="new formula data"):
        model.predict_data(pd.DataFrame({"group": ["unseen"]}))


def test_formula_pb_options_create_the_requested_smoother():
    data = pd.DataFrame(
        {
            "y": torch.linspace(0.2, 1.2, 30).numpy(),
            "x": torch.linspace(-1.0, 1.0, 30).numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pb(x, df=3, name='trend')",
            "sigma": "~ 1",
        },
        data,
    )

    term = model.smooth_terms["mu"]["trend"]
    assert set(model.smooth_terms["mu"]) == {"trend"}
    assert term.smoothing_method == "DF"
    assert term.target_effective_degrees_of_freedom == pytest.approx(5.0)
    assert "pb(x, df=3, name='trend')" in model.formula_column_names["mu"]


def test_formula_pb_aliases_and_model_dtype_are_preserved():
    data = pd.DataFrame(
        {
            "y": torch.linspace(0.2, 1.2, 120).numpy(),
            "x": torch.linspace(-1.0, 1.0, 120).numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pb(x, method='GAIC', k=3, inter=8)",
            "sigma": "~ 1",
        },
        data,
        dtype=torch.float32,
    )

    term = model.smooth_terms["mu"]["x"]
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    assert term.smoothing_method == "GAIC"
    assert term.criterion_penalty == pytest.approx(3.0)
    assert term.intervals == 8
    assert prepared.design_matrices["mu"].dtype == torch.float32
    assert prepared.smooth_covariates["mu"]["x"].dtype == torch.float32


def test_formula_rejects_missing_values():
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5],
            "x": [-1.0, -0.3, float("nan"), 1.0],
        }
    )

    with pytest.raises(ValueError, match="materialize formula"):
        GAMLSS.from_formula(
            Normal(),
            {"mu": "y ~ x", "sigma": "~ 1"},
            data,
        )


@pytest.mark.parametrize(
    ("formulas", "match"),
    [
        ({"mu": "y ~ x"}, "Formulas do not match"),
        ({"mu": "~ x", "sigma": "~ 1"}, "must contain a response"),
        (
            {"mu": "y ~ x", "sigma": "y ~ 1"},
            "Only the first",
        ),
        (
            {"mu": "y ~ pb(x):x", "sigma": "~ 1"},
            "interactions",
        ),
        (
            {"mu": "y ~ pb(x, unsupported=1)", "sigma": "~ 1"},
            "unsupported",
        ),
    ],
)
def test_invalid_formula_configurations_are_rejected(formulas, match):
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5],
            "x": [-1.0, -0.3, 0.4, 1.0],
        }
    )

    with pytest.raises(ValueError, match=match):
        GAMLSS.from_formula(Normal(), formulas, data)


def test_formula_methods_require_a_formula_model():
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="from_formula"):
        model.predict_data({"x": [1.0]})
    with pytest.raises(RuntimeError, match="from_formula"):
        model.fit_cg_data({"y": [1.0]})
