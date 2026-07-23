import csv
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import BCT, GAMLSS, Beta, CGControl, Normal, Poisson, PSpline

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference(name: str) -> dict[str, str]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return next(csv.DictReader(data_file))


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
        {"max_outer_iterations": 0},
        {"max_inner_iterations": 0},
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


def test_cg_rejects_smooth_terms_until_joint_backfitting_is_implemented():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    response = 1.0 + x.square()
    smooth = PSpline.from_data(x, smoothing_parameter=10.0, intervals=6)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"x": smooth}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.ones((x.numel(), 1), dtype=torch.float64),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }

    with pytest.raises(NotImplementedError, match="parametric predictors only"):
        model.fit_cg(response, design)
