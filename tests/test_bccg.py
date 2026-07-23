import csv
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.distributions import LogNormal

from torchgamlss import BCCG, GAMLSS, BoxCoxColeGreen, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference() -> dict[str, torch.Tensor]:
    with (REFERENCE_DIR / "bccg_reference.csv").open(
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
    with (REFERENCE_DIR / "bccg_rs_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return next(csv.DictReader(data_file))


def test_bccg_density_cdf_links_derivatives_and_initial_values_match_r():
    reference = _reference()
    family = BoxCoxColeGreen()
    parameters = {
        "mu": reference["mu"],
        "sigma": reference["sigma"],
        "nu": reference["nu"],
    }

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        family.cdf(reference["y"], parameters),
        reference["cdf"],
        rtol=2e-11,
        atol=2e-13,
    )
    torch.testing.assert_close(
        family.links["mu"](reference["mu"]),
        reference["eta_mu"],
    )
    torch.testing.assert_close(
        family.links["sigma"](reference["sigma"]),
        reference["eta_sigma"],
    )
    torch.testing.assert_close(
        family.links["nu"](reference["nu"]),
        reference["eta_nu"],
    )
    scores = family.score(reference["y"], parameters)
    torch.testing.assert_close(scores["mu"], reference["dldmu"])
    torch.testing.assert_close(scores["sigma"], reference["dldsigma"])
    torch.testing.assert_close(scores["nu"], reference["dldnu"])
    second = family.expected_second_derivatives(reference["y"], parameters)
    for pair, column in {
        ("mu", "mu"): "d2ldmu2",
        ("sigma", "sigma"): "d2ldsigma2",
        ("nu", "nu"): "d2ldnu2",
        ("mu", "sigma"): "d2ldmudsigma",
        ("mu", "nu"): "d2ldmudnu",
        ("sigma", "nu"): "d2ldsigmadnu",
    }.items():
        torch.testing.assert_close(second[pair], reference[column])
    initial = family.initial_parameters(reference["y"])
    torch.testing.assert_close(initial["mu"], reference["initial_mu"])
    torch.testing.assert_close(initial["sigma"], reference["initial_sigma"])
    torch.testing.assert_close(initial["nu"], reference["initial_nu"])
    assert BCCG is BoxCoxColeGreen


def test_bccg_autograd_matches_r_scores_after_default_links():
    reference = _reference()
    family = BoxCoxColeGreen()
    predictors = {
        "mu": reference["eta_mu"].clone().requires_grad_(),
        "sigma": reference["eta_sigma"].clone().requires_grad_(),
        "nu": reference["eta_nu"].clone().requires_grad_(),
    }
    parameters = family.parameters_from_predictors(predictors)

    gradients = torch.autograd.grad(
        family.log_prob(reference["y"], parameters).sum(),
        tuple(predictors.values()),
    )
    scores = family.score(reference["y"], parameters)

    torch.testing.assert_close(gradients[0], scores["mu"], rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        gradients[1],
        scores["sigma"] * parameters["sigma"],
        rtol=2e-11,
        atol=2e-11,
    )
    torch.testing.assert_close(gradients[2], scores["nu"], rtol=2e-11, atol=2e-11)


def test_bccg_nu_zero_is_the_lognormal_limit_with_finite_gradient():
    family = BoxCoxColeGreen()
    response = torch.tensor([0.8, 2.8, 7.0], dtype=torch.float64)
    mu = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(
        [0.15, 0.3, 0.55],
        dtype=torch.float64,
        requires_grad=True,
    )
    nu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    parameters = {"mu": mu, "sigma": sigma, "nu": nu}
    lognormal = LogNormal(mu.log(), sigma)

    log_density = family.log_prob(response, parameters)
    torch.testing.assert_close(log_density, lognormal.log_prob(response))
    torch.testing.assert_close(
        family.cdf(response, parameters), lognormal.cdf(response)
    )
    gradients = torch.autograd.grad(log_density.sum(), (mu, sigma, nu))
    scores = family.score(response, parameters)

    for gradient, parameter in zip(gradients, family.parameter_names, strict=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, scores[parameter])

    nearby_nu = torch.tensor([-1e-8, 1e-8], dtype=torch.float64)
    nearby_parameters = {
        "mu": torch.full_like(nearby_nu, 3.0),
        "sigma": torch.full_like(nearby_nu, 0.3),
        "nu": nearby_nu,
    }
    nearby_density = family.log_prob(
        torch.full_like(nearby_nu, 2.8),
        nearby_parameters,
    )
    torch.testing.assert_close(
        nearby_density,
        log_density[1].expand_as(nearby_density),
        rtol=1e-7,
        atol=1e-8,
    )


def test_bccg_formula_rs_fit_and_predictions_match_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "bccg_fit_data.csv")
    reference = _fit_reference()
    model = GAMLSS.from_formula(
        BoxCoxColeGreen(),
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
            outer_tolerance=1e-9,
            max_outer_iterations=300,
            inner_tolerance=1e-9,
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
            rtol=2e-6,
            atol=2e-6,
        )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=2e-6,
        abs=2e-6,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]),
        rel=2e-6,
        abs=2e-6,
    )
    predictions = model.predict_data(data)
    assert tuple(predictions) == ("mu", "sigma", "nu")
    assert all(values.shape == (len(data),) for values in predictions.values())
    assert (predictions["mu"] > 0).all()
    assert (predictions["sigma"] > 0).all()
    assert torch.isfinite(predictions["nu"]).all()


@pytest.mark.parametrize(
    "response",
    [
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.tensor([float("nan"), 1.0], dtype=torch.float64),
    ],
)
def test_bccg_rejects_invalid_responses(response):
    family = BoxCoxColeGreen()
    parameters = {
        "mu": torch.ones(2, dtype=torch.float64),
        "sigma": torch.full((2,), 0.2, dtype=torch.float64),
        "nu": torch.zeros(2, dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="strictly positive"):
        family.log_prob(response, parameters)
