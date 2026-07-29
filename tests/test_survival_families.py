import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import LOGNO, WEI, LogNormal, Weibull

REFERENCE_PATH = Path(__file__).parent / "reference" / "survival_family_reference.csv"
FAMILY_FACTORIES = {"WEI": Weibull, "LOGNO": LogNormal}
SECOND_DERIVATIVE_COLUMNS = {
    ("mu", "mu"): "d2ldmu2",
    ("sigma", "sigma"): "d2ldsigma2",
    ("mu", "sigma"): "d2ldmudsigma",
}


def _rows(family_name: str) -> list[dict[str, str]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
        return [
            row
            for row in csv.DictReader(reference_file)
            if row["family"] == family_name
        ]


def _tensor(rows: list[dict[str, str]], column: str) -> torch.Tensor:
    return torch.tensor(
        [float(row[column]) for row in rows],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("family_name", tuple(FAMILY_FACTORIES))
def test_survival_family_distribution_functions_match_r_gamlss_dist(family_name):
    rows = _rows(family_name)
    family = FAMILY_FACTORIES[family_name]()
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }
    distribution = family.distribution(parameters)

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        _tensor(rows, "log_density"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        _tensor(rows, "cdf"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        family.survival(response, parameters),
        _tensor(rows, "survival"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        family.hazard(response, parameters),
        _tensor(rows, "hazard"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        family.cumulative_hazard(response, parameters),
        _tensor(rows, "cumulative_hazard"),
        rtol=2e-12,
        atol=2e-12,
    )
    quantiles = family.quantile(_tensor(rows, "probability"), parameters)
    torch.testing.assert_close(
        torch.diagonal(quantiles),
        _tensor(rows, "quantile"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        distribution.mean,
        _tensor(rows, "mean"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        distribution.variance,
        _tensor(rows, "variance"),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("family_name", tuple(FAMILY_FACTORIES))
def test_survival_family_derivatives_and_initial_values_match_r(family_name):
    rows = _rows(family_name)
    family = FAMILY_FACTORIES[family_name]()
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }

    scores = family.score(response, parameters)
    second = family.expected_second_derivatives(response, parameters)
    initial = family.initial_parameters(response)

    for parameter in family.parameter_names:
        torch.testing.assert_close(
            scores[parameter],
            _tensor(rows, f"dld{parameter}"),
            rtol=2e-12,
            atol=2e-12,
        )
        torch.testing.assert_close(
            initial[parameter],
            _tensor(rows, f"initial_{parameter}"),
            rtol=2e-12,
            atol=2e-12,
        )
    for pair, column in SECOND_DERIVATIVE_COLUMNS.items():
        torch.testing.assert_close(
            second[pair],
            _tensor(rows, column),
            rtol=2e-12,
            atol=2e-12,
        )


@pytest.mark.parametrize("family_name", tuple(FAMILY_FACTORIES))
def test_survival_family_autograd_obeys_link_chain_rule(family_name):
    rows = _rows(family_name)
    family = FAMILY_FACTORIES[family_name]()
    response = _tensor(rows, "y")
    predictors = {
        parameter: family.links[parameter](_tensor(rows, parameter)).requires_grad_()
        for parameter in family.parameter_names
    }
    parameters = family.parameters_from_predictors(predictors)
    gradients = torch.autograd.grad(
        family.log_prob(response, parameters).sum(),
        tuple(predictors.values()),
    )
    scores = family.score(response, parameters)

    for parameter, gradient in zip(
        family.parameter_names,
        gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            scores[parameter]
            * family.links[parameter].inverse_derivative(predictors[parameter]),
            rtol=2e-12,
            atol=2e-12,
        )


def test_survival_family_aliases_and_default_links():
    assert WEI is Weibull
    assert LOGNO is LogNormal
    assert Weibull().links["mu"].name == "log"
    assert Weibull().links["sigma"].name == "log"
    assert LogNormal().links["mu"].name == "identity"
    assert LogNormal().links["sigma"].name == "log"


@pytest.mark.parametrize("family", [Weibull(), LogNormal()])
@pytest.mark.parametrize(
    "response",
    [
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([-1.0], dtype=torch.float64),
        torch.tensor([torch.nan], dtype=torch.float64),
    ],
)
def test_survival_families_reject_responses_outside_support(family, response):
    parameters = {
        "mu": torch.ones_like(response),
        "sigma": torch.ones_like(response),
    }
    with pytest.raises(ValueError):
        family.log_prob(response, parameters)
