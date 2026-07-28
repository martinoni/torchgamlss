import csv
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.distributions import Normal as TorchNormal

from torchgamlss import GAMLSS, PE, PowerExponential, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference() -> dict[str, torch.Tensor]:
    with (REFERENCE_DIR / "pe_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        rows = list(csv.DictReader(data_file))
    return {
        column: torch.tensor(
            [float(row[column]) for row in rows],
            dtype=torch.float64,
        )
        for column in rows[0]
        if column != "gamlss_dist_version"
    }


def _fit_reference() -> dict[str, str]:
    with (REFERENCE_DIR / "pe_rs_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return next(csv.DictReader(data_file))


def test_pe_density_cdf_links_derivatives_and_initial_values_match_r():
    reference = _reference()
    family = PowerExponential()
    parameters = {
        parameter: reference[parameter] for parameter in family.parameter_names
    }

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=3e-12,
        atol=3e-12,
    )
    torch.testing.assert_close(
        family.cdf(reference["y"], parameters),
        reference["cdf"],
        rtol=3e-12,
        atol=3e-12,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            family.links[parameter](reference[parameter]),
            reference[f"eta_{parameter}"],
        )
    scores = family.score(reference["y"], parameters)
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            scores[parameter],
            reference[f"dld{parameter}"],
            rtol=3e-11,
            atol=3e-11,
        )
    second = family.expected_second_derivatives(reference["y"], parameters)
    for pair, column in {
        ("mu", "mu"): "d2ldmu2",
        ("sigma", "sigma"): "d2ldsigma2",
        ("nu", "nu"): "d2ldnu2",
        ("mu", "sigma"): "d2ldmudsigma",
        ("mu", "nu"): "d2ldmudnu",
        ("sigma", "nu"): "d2ldsigmadnu",
    }.items():
        tolerance = 1e-8 if pair == ("nu", "nu") else 3e-11
        torch.testing.assert_close(
            second[pair],
            reference[column],
            rtol=tolerance,
            atol=tolerance,
        )
    initial = family.initial_parameters(reference["y"])
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            initial[parameter],
            reference[f"initial_{parameter}"],
        )
    assert PE is PowerExponential


def test_pe_autograd_matches_scores_after_default_links():
    reference = _reference()
    family = PE()
    predictors = {
        parameter: reference[f"eta_{parameter}"].clone().requires_grad_()
        for parameter in family.parameter_names
    }
    parameters = family.parameters_from_predictors(predictors)

    gradients = torch.autograd.grad(
        family.log_prob(reference["y"], parameters).sum(),
        tuple(predictors.values()),
    )
    scores = family.score(reference["y"], parameters)

    torch.testing.assert_close(gradients[0], scores["mu"], rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(
        gradients[1],
        scores["sigma"] * parameters["sigma"],
        rtol=3e-11,
        atol=3e-11,
    )
    torch.testing.assert_close(
        gradients[2],
        scores["nu"] * parameters["nu"],
        rtol=3e-11,
        atol=3e-11,
    )


def test_pe_mean_variance_and_nu_two_normal_identity():
    family = PE()
    parameters = {
        "mu": torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64),
        "sigma": torch.tensor([0.4, 1.2, 2.0], dtype=torch.float64),
        "nu": torch.full((3,), 2.0, dtype=torch.float64),
    }
    distribution = family.distribution(parameters)
    response = torch.tensor([-2.0, 0.25, 4.0], dtype=torch.float64)
    normal = TorchNormal(parameters["mu"], parameters["sigma"])

    torch.testing.assert_close(distribution.mean, parameters["mu"])
    torch.testing.assert_close(distribution.variance, parameters["sigma"].square())
    torch.testing.assert_close(
        family.log_prob(response, parameters),
        normal.log_prob(response),
    )
    torch.testing.assert_close(family.cdf(response, parameters), normal.cdf(response))


def test_pe_sampling_is_reproducible_and_has_uniform_pit():
    observation_count = 6000
    family = PE()
    parameters = {
        "mu": torch.full((observation_count,), 1.5, dtype=torch.float64),
        "sigma": torch.full((observation_count,), 0.7, dtype=torch.float64),
        "nu": torch.full((observation_count,), 1.25, dtype=torch.float64),
    }

    first = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2026),
    )
    second = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2026),
    )
    probabilities = family.cdf(first, parameters)

    torch.testing.assert_close(first, second)
    assert probabilities.mean() == pytest.approx(0.5, abs=0.015)
    assert probabilities.var(correction=1) == pytest.approx(1.0 / 12.0, abs=0.005)


def test_pe_formula_rs_fit_and_predictions_match_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "pe_fit_data.csv")
    reference = _fit_reference()
    model = GAMLSS.from_formula(
        PE(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
            "nu": "~ w + offset(nu_offset)",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=300,
            inner_tolerance=1e-8,
            max_inner_iterations=300,
        ),
    )

    assert result.converged
    for parameter, columns in {
        "mu": ("mu_intercept", "mu_x"),
        "sigma": ("sigma_intercept", "sigma_z"),
        "nu": ("nu_intercept", "nu_w"),
    }.items():
        torch.testing.assert_close(
            model.coefficients[parameter],
            torch.tensor(
                [float(reference[column]) for column in columns],
                dtype=torch.float64,
            ),
            rtol=3e-6,
            atol=3e-6,
        )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=3e-6,
        abs=3e-6,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]),
        rel=3e-6,
        abs=3e-6,
    )
    predictions = model.predict_data(data)
    assert tuple(predictions) == ("mu", "sigma", "nu")
    assert all(values.shape == (len(data),) for values in predictions.values())
    assert torch.isfinite(predictions["mu"]).all()
    assert (predictions["sigma"] > 0).all()
    assert (predictions["nu"] > 0).all()
