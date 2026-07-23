import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import Beta, NegativeBinomial, Poisson

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference(name: str, columns: list[str]) -> dict[str, torch.Tensor]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        rows = list(csv.DictReader(data_file))
    return {
        column: torch.tensor(
            [float(row[column]) for row in rows],
            dtype=torch.float64,
        )
        for column in columns
    }


def test_poisson_density_links_derivatives_and_initial_values_match_r():
    reference = _reference(
        "po_reference.csv",
        [
            "y",
            "mu",
            "eta_mu",
            "log_density",
            "dldmu",
            "d2ldmu2",
            "initial_mu",
        ],
    )
    family = Poisson()
    parameters = {"mu": reference["mu"]}

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["mu"](reference["mu"]),
        reference["eta_mu"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.score(reference["y"], parameters)["mu"],
        reference["dldmu"],
    )
    torch.testing.assert_close(
        family.expected_second_derivatives(reference["y"], parameters)[("mu", "mu")],
        reference["d2ldmu2"],
    )
    torch.testing.assert_close(
        family.initial_parameters(reference["y"])["mu"],
        reference["initial_mu"],
    )
    torch.testing.assert_close(family.distribution(parameters).mean, reference["mu"])
    torch.testing.assert_close(
        family.distribution(parameters).variance,
        reference["mu"],
    )


def test_poisson_autograd_matches_r_score_after_log_link():
    reference = _reference(
        "po_reference.csv",
        ["y", "mu", "eta_mu", "dldmu"],
    )
    family = Poisson()
    eta_mu = reference["eta_mu"].clone().requires_grad_()
    parameters = family.parameters_from_predictors({"mu": eta_mu})

    gradient = torch.autograd.grad(
        family.log_prob(reference["y"], parameters).sum(),
        eta_mu,
    )[0]

    torch.testing.assert_close(gradient, reference["dldmu"] * reference["mu"])


def test_nbi_density_links_derivatives_and_initial_values_match_r():
    columns = [
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
        "initial_mu",
        "initial_sigma",
    ]
    reference = _reference("nbi_reference.csv", columns)
    family = NegativeBinomial()
    parameters = {"mu": reference["mu"], "sigma": reference["sigma"]}
    distribution = family.distribution(parameters)

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["mu"](reference["mu"]),
        reference["eta_mu"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["sigma"](reference["sigma"]),
        reference["eta_sigma"],
        rtol=1e-12,
        atol=1e-12,
    )
    score = family.score(reference["y"], parameters)
    second = family.expected_second_derivatives(reference["y"], parameters)
    torch.testing.assert_close(score["mu"], reference["dldmu"])
    torch.testing.assert_close(score["sigma"], reference["dldsigma"])
    torch.testing.assert_close(second[("mu", "mu")], reference["d2ldmu2"])
    torch.testing.assert_close(second[("sigma", "sigma")], reference["d2ldsigma2"])
    torch.testing.assert_close(
        second[("mu", "sigma")],
        reference["d2ldmudsigma"],
    )
    initial = family.initial_parameters(reference["y"])
    torch.testing.assert_close(initial["mu"], reference["initial_mu"])
    torch.testing.assert_close(initial["sigma"], reference["initial_sigma"])
    torch.testing.assert_close(distribution.mean, reference["mu"])
    torch.testing.assert_close(
        distribution.variance,
        reference["mu"] + reference["sigma"] * reference["mu"].square(),
    )


def test_nbi_autograd_matches_r_scores_after_log_links():
    reference = _reference(
        "nbi_reference.csv",
        ["y", "mu", "sigma", "eta_mu", "eta_sigma", "dldmu", "dldsigma"],
    )
    family = NegativeBinomial()
    eta_mu = reference["eta_mu"].clone().requires_grad_()
    eta_sigma = reference["eta_sigma"].clone().requires_grad_()
    parameters = family.parameters_from_predictors({"mu": eta_mu, "sigma": eta_sigma})

    gradient_mu, gradient_sigma = torch.autograd.grad(
        family.log_prob(reference["y"], parameters).sum(),
        (eta_mu, eta_sigma),
    )

    torch.testing.assert_close(gradient_mu, reference["dldmu"] * reference["mu"])
    torch.testing.assert_close(
        gradient_sigma,
        reference["dldsigma"] * reference["sigma"],
    )


def test_beta_density_links_derivatives_and_initial_values_match_r():
    columns = [
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
        "initial_mu",
        "initial_sigma",
    ]
    reference = _reference("be_reference.csv", columns)
    family = Beta()
    parameters = {"mu": reference["mu"], "sigma": reference["sigma"]}
    distribution = family.distribution(parameters)

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["mu"](reference["mu"]),
        reference["eta_mu"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        family.links["sigma"](reference["sigma"]),
        reference["eta_sigma"],
        rtol=1e-12,
        atol=1e-12,
    )
    score = family.score(reference["y"], parameters)
    second = family.expected_second_derivatives(reference["y"], parameters)
    torch.testing.assert_close(score["mu"], reference["dldmu"])
    torch.testing.assert_close(score["sigma"], reference["dldsigma"])
    torch.testing.assert_close(second[("mu", "mu")], reference["d2ldmu2"])
    torch.testing.assert_close(second[("sigma", "sigma")], reference["d2ldsigma2"])
    torch.testing.assert_close(
        second[("mu", "sigma")],
        reference["d2ldmudsigma"],
    )
    initial = family.initial_parameters(reference["y"])
    torch.testing.assert_close(initial["mu"], reference["initial_mu"])
    torch.testing.assert_close(initial["sigma"], reference["initial_sigma"])
    torch.testing.assert_close(distribution.mean, reference["mu"])
    torch.testing.assert_close(
        distribution.variance,
        reference["sigma"].square() * reference["mu"] * (1.0 - reference["mu"]),
    )


def test_beta_autograd_matches_r_scores_after_logit_links():
    reference = _reference(
        "be_reference.csv",
        ["y", "mu", "sigma", "eta_mu", "eta_sigma", "dldmu", "dldsigma"],
    )
    family = Beta()
    eta_mu = reference["eta_mu"].clone().requires_grad_()
    eta_sigma = reference["eta_sigma"].clone().requires_grad_()
    parameters = family.parameters_from_predictors({"mu": eta_mu, "sigma": eta_sigma})

    gradient_mu, gradient_sigma = torch.autograd.grad(
        family.log_prob(reference["y"], parameters).sum(),
        (eta_mu, eta_sigma),
    )

    torch.testing.assert_close(
        gradient_mu,
        reference["dldmu"] * reference["mu"] * (1.0 - reference["mu"]),
    )
    torch.testing.assert_close(
        gradient_sigma,
        reference["dldsigma"] * reference["sigma"] * (1.0 - reference["sigma"]),
    )


@pytest.mark.parametrize(
    ("family", "response", "message"),
    [
        (Poisson(), torch.tensor([0.5], dtype=torch.float64), "integer counts"),
        (Poisson(), torch.tensor([-1.0], dtype=torch.float64), "integer counts"),
        (
            NegativeBinomial(),
            torch.tensor([1.5], dtype=torch.float64),
            "integer counts",
        ),
        (Beta(), torch.tensor([0.0], dtype=torch.float64), "between zero and one"),
        (Beta(), torch.tensor([1.0], dtype=torch.float64), "between zero and one"),
    ],
)
def test_new_families_reject_responses_outside_their_support(family, response, message):
    parameters = {
        parameter: torch.full_like(response, 0.5)
        for parameter in family.parameter_names
    }

    with pytest.raises(ValueError, match=message):
        family.log_prob(response, parameters)
