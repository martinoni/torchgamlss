import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    GG,
    IG,
    LOGNO,
    WEI,
    GeneralizedGamma,
    InverseGaussian,
    LogNormal,
    RSControl,
    Weibull,
)

REFERENCE_PATH = Path(__file__).parent / "reference" / "survival_family_reference.csv"
FAMILY_FACTORIES = {
    "WEI": Weibull,
    "LOGNO": LogNormal,
    "IG": InverseGaussian,
    "GG": GeneralizedGamma,
}
SECOND_DERIVATIVE_COLUMNS = {
    ("mu", "mu"): "d2ldmu2",
    ("sigma", "sigma"): "d2ldsigma2",
    ("nu", "nu"): "d2ldnu2",
    ("mu", "sigma"): "d2ldmudsigma",
    ("mu", "nu"): "d2ldmudnu",
    ("sigma", "nu"): "d2ldsigmadnu",
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
    quantile_tolerance = 2e-5 if family_name == "IG" else 2e-12
    torch.testing.assert_close(
        torch.diagonal(quantiles),
        _tensor(rows, "quantile"),
        rtol=quantile_tolerance,
        atol=quantile_tolerance,
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
    second_tolerance = 3e-8 if family_name == "GG" else 2e-12
    for pair, column in SECOND_DERIVATIVE_COLUMNS.items():
        if pair not in second:
            continue
        torch.testing.assert_close(
            second[pair],
            _tensor(rows, column),
            rtol=second_tolerance,
            atol=second_tolerance,
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
            rtol=2e-10,
            atol=2e-10,
        )


def test_survival_family_aliases_and_default_links():
    assert WEI is Weibull
    assert LOGNO is LogNormal
    assert IG is InverseGaussian
    assert GG is GeneralizedGamma
    assert Weibull().links["mu"].name == "log"
    assert Weibull().links["sigma"].name == "log"
    assert LogNormal().links["mu"].name == "identity"
    assert LogNormal().links["sigma"].name == "log"
    assert InverseGaussian().links["mu"].name == "log"
    assert InverseGaussian().links["sigma"].name == "log"
    assert GeneralizedGamma().links["mu"].name == "log"
    assert GeneralizedGamma().links["sigma"].name == "log"
    assert GeneralizedGamma().links["nu"].name == "identity"


@pytest.mark.parametrize(
    "family",
    [Weibull(), LogNormal(), InverseGaussian(), GeneralizedGamma()],
)
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
        parameter: torch.ones_like(response) for parameter in family.parameter_names
    }
    with pytest.raises(ValueError):
        family.log_prob(response, parameters)


def test_generalized_gamma_log_normal_limit_is_continuous_and_differentiable():
    response = torch.tensor([0.2, 0.7, 1.0, 2.0, 8.0], dtype=torch.float64)
    mu = torch.full_like(response, 1.3, requires_grad=True)
    sigma = torch.full_like(response, 0.7, requires_grad=True)
    nu = torch.zeros_like(response, requires_grad=True)
    parameters = {"mu": mu, "sigma": sigma, "nu": nu}
    family = GeneralizedGamma()
    log_normal = LogNormal()
    log_normal_parameters = {"mu": torch.log(mu), "sigma": sigma}

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        log_normal.log_prob(response, log_normal_parameters),
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        log_normal.cdf(response, log_normal_parameters),
    )
    distribution = family.distribution(parameters)
    log_normal_distribution = log_normal.distribution(log_normal_parameters)
    torch.testing.assert_close(distribution.mean, log_normal_distribution.mean)
    torch.testing.assert_close(distribution.variance, log_normal_distribution.variance)

    gradients = torch.autograd.grad(
        family.log_prob(response, parameters).sum(),
        (mu, sigma, nu),
    )
    scores = family.score(response, parameters)
    for parameter, gradient in zip(
        family.parameter_names,
        gradients,
        strict=True,
    ):
        torch.testing.assert_close(gradient, scores[parameter])

    near_parameters = {
        "mu": mu.detach(),
        "sigma": sigma.detach(),
        "nu": torch.full_like(response, 0.03),
    }
    near_predictors = {
        parameter: family.links[parameter](value).requires_grad_()
        for parameter, value in near_parameters.items()
    }
    linked_parameters = family.parameters_from_predictors(near_predictors)
    near_gradients = torch.autograd.grad(
        family.log_prob(response, linked_parameters).sum(),
        tuple(near_predictors.values()),
    )
    near_scores = family.score(response, linked_parameters)
    for parameter, gradient in zip(
        family.parameter_names,
        near_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            gradient,
            near_scores[parameter]
            * family.links[parameter].inverse_derivative(near_predictors[parameter]),
            rtol=2e-12,
            atol=2e-12,
        )


@pytest.mark.parametrize("nu", [-0.05, -0.03, 0.0, 0.03, 0.05])
def test_generalized_gamma_near_limit_quantiles_invert_cdf(nu):
    family = GeneralizedGamma()
    parameters = {
        "mu": torch.tensor([1.3], dtype=torch.float64),
        "sigma": torch.tensor([0.7], dtype=torch.float64),
        "nu": torch.tensor([nu], dtype=torch.float64),
    }
    probabilities = torch.tensor(
        [0.01, 0.1, 0.5, 0.9, 0.99],
        dtype=torch.float64,
    )
    quantiles = family.quantile(probabilities, parameters).squeeze(0)
    expanded_parameters = {
        parameter: value.expand_as(quantiles) for parameter, value in parameters.items()
    }

    torch.testing.assert_close(
        family.cdf(quantiles, expanded_parameters),
        probabilities,
        rtol=2e-11,
        atol=2e-11,
    )


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        (
            InverseGaussian(),
            {"mu": 2.0, "sigma": 0.6},
        ),
        (
            GeneralizedGamma(),
            {"mu": 2.0, "sigma": 0.6, "nu": -0.7},
        ),
        (
            GeneralizedGamma(),
            {"mu": 2.0, "sigma": 0.6, "nu": 0.03},
        ),
    ],
)
def test_new_survival_family_samples_have_uniform_probability_integral_transform(
    family,
    parameters,
):
    count = 8_000
    tensor_parameters = {
        parameter: torch.full((count,), value, dtype=torch.float64)
        for parameter, value in parameters.items()
    }
    samples = family.sample(
        tensor_parameters,
        generator=torch.Generator().manual_seed(99173),
    )
    probabilities = family.cdf(samples, tensor_parameters)

    assert torch.isfinite(samples).all()
    assert bool((samples > 0).all())
    assert float(probabilities.mean()) == pytest.approx(0.5, abs=0.015)
    assert float(probabilities.var(correction=1)) == pytest.approx(
        1.0 / 12.0,
        abs=0.01,
    )


@pytest.mark.parametrize(
    ("family", "parameter_values"),
    [
        (InverseGaussian(), {"sigma": 0.55}),
        (GeneralizedGamma(), {"sigma": 0.55, "nu": 0.7}),
    ],
)
def test_new_survival_families_fit_and_predict_with_rs_and_lbfgs(
    family,
    parameter_values,
):
    count = 320
    covariate = torch.linspace(-1.0, 1.0, count, dtype=torch.float64)
    parameters = {
        "mu": torch.exp(0.5 + 0.35 * covariate),
        **{
            parameter: torch.full_like(covariate, value)
            for parameter, value in parameter_values.items()
        },
    }
    response = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(11891),
    )
    data = pd.DataFrame({"y": response.numpy(), "x": covariate.numpy()})
    formulas = {
        "mu": "y ~ x",
        "sigma": "~ 1",
    }
    if "nu" in family.parameter_names:
        formulas["nu"] = "~ 1"

    rs_model = GAMLSS.from_formula(family, formulas, data)
    rs_result = rs_model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=200,
            inner_tolerance=1e-8,
            max_inner_iterations=200,
        ),
    )
    lbfgs_model = GAMLSS.from_formula(family, formulas, data)
    lbfgs_result = lbfgs_model.fit_data(
        data,
        max_iter=500,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    assert rs_result.converged
    assert lbfgs_result.converged
    assert rs_result.negative_log_likelihood == pytest.approx(
        lbfgs_result.negative_log_likelihood,
        rel=2e-8,
        abs=2e-8,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            rs_model.coefficients[parameter],
            lbfgs_model.coefficients[parameter],
            rtol=4e-5,
            atol=4e-5,
        )
    predictions = rs_model.predict_data(data.iloc[:8])
    quantiles = rs_model.predict_quantiles_data(
        data.iloc[:8],
        probabilities=[0.1, 0.5, 0.9],
    )
    curves = rs_model.predict_survival_data(
        data.iloc[:8],
        times=[0.5, 1.0, 2.0, 4.0],
    )
    assert all(torch.isfinite(value).all() for value in predictions.values())
    assert quantiles.quantiles.shape == (8, 3)
    assert curves.survival.shape == (8, 4)
    assert bool((torch.diff(curves.survival, dim=-1) <= 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("family", "parameter_values"),
    [
        (InverseGaussian(), {"mu": 2.0, "sigma": 0.6}),
        (GeneralizedGamma(), {"mu": 2.0, "sigma": 0.6, "nu": 0.03}),
    ],
)
def test_new_survival_families_run_with_gradients_and_sampling_on_cuda(
    family,
    parameter_values,
):
    device = torch.device("cuda")
    response = torch.tensor([0.4, 1.0, 2.5, 6.0], device=device)
    predictors = {
        parameter: family.links[parameter](
            torch.full_like(response, value)
        ).requires_grad_()
        for parameter, value in parameter_values.items()
    }
    parameters = family.parameters_from_predictors(predictors)
    objective = (
        family.log_prob(response, parameters)
        + torch.log(family.cdf(response, parameters))
        + torch.log(family.survival(response, parameters))
    ).sum()
    gradients = torch.autograd.grad(objective, tuple(predictors.values()))
    samples = family.sample(
        parameters,
        generator=torch.Generator(device=device).manual_seed(431),
    )

    assert objective.device.type == "cuda"
    assert torch.isfinite(objective)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert samples.device.type == "cuda"
    assert torch.isfinite(samples).all()
