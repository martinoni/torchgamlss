import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import Gamma

REFERENCE_PATH = Path(__file__).parent / "reference" / "ga_reference.csv"


def _reference() -> dict[str, torch.Tensor]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
        rows = list(csv.DictReader(reference_file))

    numeric_columns = [
        "y",
        "mu",
        "sigma",
        "eta_mu",
        "eta_sigma",
        "log_density",
        "dldmu",
        "dldsigma",
        "d2ldmu2",
        "d2ldsigma2",
        "d2ldmudsigma",
    ]
    return {
        column: torch.tensor([float(row[column]) for row in rows], dtype=torch.float64)
        for column in numeric_columns
    }


def test_ga_log_density_links_mean_and_variance_match_r_gamlss_dist():
    reference = _reference()
    family = Gamma()
    parameters = {"mu": reference["mu"], "sigma": reference["sigma"]}
    distribution = family.distribution(parameters)

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["mu"](parameters["mu"]),
        reference["eta_mu"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["sigma"](parameters["sigma"]),
        reference["eta_sigma"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(distribution.mean, reference["mu"])
    torch.testing.assert_close(
        distribution.variance,
        reference["sigma"].square() * reference["mu"].square(),
    )


def test_ga_analytic_derivatives_match_r_gamlss_dist():
    reference = _reference()
    family = Gamma()
    parameters = {"mu": reference["mu"], "sigma": reference["sigma"]}

    score = family.score(reference["y"], parameters)
    second = family.expected_second_derivatives(reference["y"], parameters)

    torch.testing.assert_close(score["mu"], reference["dldmu"])
    torch.testing.assert_close(score["sigma"], reference["dldsigma"])
    torch.testing.assert_close(second[("mu", "mu")], reference["d2ldmu2"])
    torch.testing.assert_close(second[("sigma", "sigma")], reference["d2ldsigma2"])
    torch.testing.assert_close(second[("mu", "sigma")], reference["d2ldmudsigma"])


def test_ga_autograd_score_matches_r_after_log_link_chain_rule():
    reference = _reference()
    family = Gamma()
    eta_mu = reference["eta_mu"].clone().requires_grad_()
    eta_sigma = reference["eta_sigma"].clone().requires_grad_()
    parameters = family.parameters_from_predictors({"mu": eta_mu, "sigma": eta_sigma})

    log_likelihood = family.log_prob(reference["y"], parameters).sum()
    gradient_mu, gradient_sigma = torch.autograd.grad(
        log_likelihood, (eta_mu, eta_sigma)
    )

    torch.testing.assert_close(
        gradient_mu,
        reference["dldmu"] * reference["mu"],
    )
    torch.testing.assert_close(
        gradient_sigma,
        reference["dldsigma"] * reference["sigma"],
    )


def test_ga_initial_parameters_match_r_expressions():
    response = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)

    initial = Gamma().initial_parameters(response)

    torch.testing.assert_close(
        initial["mu"],
        (response + response.mean()) / 2.0,
    )
    torch.testing.assert_close(initial["sigma"], torch.ones_like(response))


@pytest.mark.parametrize(
    "response",
    [
        torch.tensor([], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.tensor([float("nan"), 1.0], dtype=torch.float64),
    ],
)
def test_ga_initial_parameters_reject_invalid_responses(response):
    with pytest.raises(ValueError, match="GA initialization"):
        Gamma().initial_parameters(response)


@pytest.mark.parametrize(
    "response",
    [
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([-1.0], dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float64),
    ],
)
def test_ga_log_prob_rejects_values_outside_the_family_support(response):
    parameters = {
        "mu": torch.ones_like(response),
        "sigma": torch.ones_like(response),
    }

    with pytest.raises(ValueError, match="strictly positive"):
        Gamma().log_prob(response, parameters)
