import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    MX,
    FiniteMixture,
    Gamma,
    MixtureControl,
    Normal,
    Poisson,
)

REFERENCE_PATH = Path(__file__).parent / "reference" / "mixture_reference.csv"
FIT_DATA_PATH = Path(__file__).parent / "reference" / "mixture_fit_data.csv"
FIT_REFERENCE_PATH = (
    Path(__file__).parent / "reference" / "mixture_fit_reference.csv"
)
FIT_POSTERIOR_PATH = (
    Path(__file__).parent
    / "reference"
    / "mixture_fit_posterior_reference.csv"
)
FAMILY_FACTORIES = {
    "NO": Normal,
    "GA": Gamma,
    "PO": Poisson,
}


def _reference_rows(case: str) -> list[dict[str, str]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
        return [
            row
            for row in csv.DictReader(reference_file)
            if row["case"] == case
        ]


def _tensor(rows: list[dict[str, str]], column: str) -> torch.Tensor:
    return torch.tensor(
        [float(row[column]) for row in rows],
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    "case",
    ("normal_normal", "gamma_gamma", "poisson_poisson"),
)
def test_mixture_density_cdf_posterior_moments_match_r_gamlss_mx(case):
    rows = _reference_rows(case)
    family = FiniteMixture(
        [
            FAMILY_FACTORIES[rows[0]["family_1"]](),
            FAMILY_FACTORIES[rows[0]["family_2"]](),
        ]
    )
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter)
        for parameter in family.parameter_names
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
        family.component_weights(parameters)[..., 0],
        _tensor(rows, "pi_1"),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        family.posterior_probabilities(response, parameters),
        torch.stack(
            [
                _tensor(rows, "posterior_1"),
                _tensor(rows, "posterior_2"),
            ],
            dim=-1,
        ),
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


def test_reference_log_odds_are_stable_for_extreme_mixture_predictors():
    family = FiniteMixture([Normal(), Normal(), Normal()])
    response = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float32)
    predictors = {
        "component_1_mu": torch.full_like(response, -100.0),
        "component_1_sigma": torch.full_like(response, -20.0),
        "component_2_mu": torch.zeros_like(response),
        "component_2_sigma": torch.full_like(response, -20.0),
        "component_3_mu": torch.full_like(response, 100.0),
        "component_3_sigma": torch.full_like(response, -20.0),
        "mixing_1": torch.full_like(response, 1_000.0),
        "mixing_2": torch.full_like(response, -1_000.0),
    }
    parameters = family.parameters_from_predictors(predictors)

    log_prob = family.log_prob(response, parameters)
    posterior = family.posterior_probabilities(response, parameters)

    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(posterior).all()
    torch.testing.assert_close(
        posterior.sum(dim=-1),
        torch.ones_like(response),
    )


def test_continuous_mixture_quantiles_invert_the_cdf():
    family = FiniteMixture([Normal(), Gamma()])
    parameters = {
        "component_1_mu": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "component_1_sigma": torch.tensor([1.0, 0.7], dtype=torch.float64),
        "component_2_mu": torch.tensor([2.0, 3.0], dtype=torch.float64),
        "component_2_sigma": torch.tensor([0.5, 0.3], dtype=torch.float64),
        "mixing_1": torch.tensor([2.0, 0.5], dtype=torch.float64),
    }
    probabilities = torch.tensor([0.05, 0.25, 0.5, 0.9], dtype=torch.float64)

    quantiles = family.quantile(probabilities, parameters)
    expanded_parameters = {
        parameter: value.unsqueeze(-1).expand_as(quantiles)
        for parameter, value in parameters.items()
    }

    torch.testing.assert_close(
        family.cdf(quantiles, expanded_parameters),
        probabilities.expand_as(quantiles),
        rtol=2e-10,
        atol=2e-10,
    )


def test_component_specific_and_shared_predictors_work_in_gamlss():
    family = FiniteMixture(
        [Normal(), Normal()],
        shared_parameters=("sigma",),
    )
    assert family.parameter_names == (
        "component_1_mu",
        "sigma",
        "component_2_mu",
        "mixing_1",
    )
    design = {
        "component_1_mu": torch.tensor(
            [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float64,
        ),
        "sigma": torch.ones((3, 1), dtype=torch.float64),
        "component_2_mu": torch.ones((3, 1), dtype=torch.float64),
        "mixing_1": torch.tensor(
            [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float64,
        ),
    }
    model = GAMLSS(
        family,
        {parameter: matrix.shape[1] for parameter, matrix in design.items()},
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.coefficients["component_1_mu"].copy_(
            torch.tensor([-1.0, 0.25], dtype=torch.float64)
        )
        model.coefficients["component_2_mu"].fill_(2.0)
        model.coefficients["sigma"].fill_(-0.5)
        model.coefficients["mixing_1"].copy_(
            torch.tensor([0.2, -0.4], dtype=torch.float64)
        )
    response = torch.tensor([-1.0, 0.5, 2.5], dtype=torch.float64)

    parameters = model.predict(design)
    assert isinstance(parameters, dict)
    grouped = family.component_parameters(parameters)
    torch.testing.assert_close(grouped[0]["sigma"], grouped[1]["sigma"])
    loss = model.negative_log_likelihood(response, design)
    loss.backward()

    assert torch.isfinite(loss)
    for coefficient in model.coefficients.values():
        assert coefficient.grad is not None
        assert torch.isfinite(coefficient.grad).all()


def test_intercept_only_em_fit_matches_r_gamlss_mx():
    with FIT_DATA_PATH.open(newline="", encoding="utf-8") as data_file:
        response = torch.tensor(
            [float(row["y"]) for row in csv.DictReader(data_file)],
            dtype=torch.float64,
        )
    with FIT_REFERENCE_PATH.open(
        newline="",
        encoding="utf-8",
    ) as reference_file:
        reference = next(csv.DictReader(reference_file))
    with FIT_POSTERIOR_PATH.open(
        newline="",
        encoding="utf-8",
    ) as posterior_file:
        posterior_reference = tuple(csv.DictReader(posterior_file))

    family = FiniteMixture([Normal(), Normal()])
    design = {
        parameter: torch.ones((response.numel(), 1), dtype=response.dtype)
        for parameter in family.parameter_names
    }
    model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=response.dtype,
    )
    result = model.fit_mixture(
        response,
        design,
        control=MixtureControl(
            tolerance=1e-10,
            max_iterations=100,
            m_step_max_iter=100,
        ),
    )
    parameters = model.predict(design)
    assert isinstance(parameters, dict)

    assert result.converged
    assert result.iterations < 100
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=2e-9,
        abs=2e-9,
    )
    assert all(
        right <= left + 1e-8 * (1.0 + abs(left))
        for left, right in zip(
            result.deviance_history[:-1],
            result.deviance_history[1:],
            strict=True,
        )
    )
    for parameter in (
        "component_1_mu",
        "component_1_sigma",
        "component_2_mu",
        "component_2_sigma",
    ):
        torch.testing.assert_close(
            parameters[parameter],
            torch.full_like(
                response,
                float(reference[parameter]),
            ),
            rtol=2e-8,
            atol=2e-8,
        )
    expected_probabilities = torch.tensor(
        [
            float(reference["pi_1"]),
            float(reference["pi_2"]),
        ],
        dtype=response.dtype,
    )
    torch.testing.assert_close(
        family.component_weights(parameters)[0],
        expected_probabilities,
        rtol=2e-8,
        atol=2e-8,
    )
    expected_posterior = torch.tensor(
        [
            [
                float(row["posterior_1"]),
                float(row["posterior_2"]),
            ]
            for row in posterior_reference
        ],
        dtype=response.dtype,
    )
    torch.testing.assert_close(
        model.posterior_probabilities(response, design),
        expected_posterior,
        rtol=2e-6,
        atol=2e-8,
    )
    diagnostics = model.component_diagnostics(response, design)
    torch.testing.assert_close(
        diagnostics.posterior_probabilities,
        result.posterior_probabilities,
    )


def test_formula_mixture_fit_uses_component_specific_parameter_names():
    with FIT_DATA_PATH.open(newline="", encoding="utf-8") as data_file:
        data = pd.DataFrame(csv.DictReader(data_file)).astype(float)
    family = FiniteMixture(
        [Normal(), Normal()],
        shared_parameters=("sigma",),
    )
    formulas = {
        parameter: "y ~ 1" if index == 0 else "~ 1"
        for index, parameter in enumerate(family.parameter_names)
    }

    model = GAMLSS.from_formula(family, formulas, data)
    result = model.fit_mixture_data(
        data,
        control=MixtureControl(
            tolerance=1e-8,
            max_iterations=100,
            m_step_max_iter=100,
        ),
    )

    assert result.converged
    parameters = model.predict_data(data)
    assert isinstance(parameters, dict)
    grouped = family.component_parameters(parameters)
    torch.testing.assert_close(grouped[0]["sigma"], grouped[1]["sigma"])
    assert bool((grouped[0]["mu"] < grouped[1]["mu"]).all())


def test_initialization_and_canonical_labels_are_deterministic():
    family = FiniteMixture([Normal(), Normal()])
    response = torch.tensor(
        [-3.0, -2.0, -1.0, 4.0, 5.0, 6.0],
        dtype=torch.float64,
    )

    first = family.initial_parameters(response)
    second = family.initial_parameters(response)

    for parameter in family.parameter_names:
        torch.testing.assert_close(first[parameter], second[parameter])
    assert bool((first["component_1_mu"] < first["component_2_mu"]).all())
    torch.testing.assert_close(first["mixing_1"], torch.zeros_like(response))

    switched = {
        "component_1_mu": torch.full_like(response, 5.0),
        "component_1_sigma": torch.full_like(response, 2.0),
        "component_2_mu": torch.full_like(response, -2.0),
        "component_2_sigma": torch.full_like(response, 0.5),
        "mixing_1": torch.full_like(response, 3.0),
    }
    canonical = family.canonicalize_parameters(switched)
    assert family.component_order(switched) == (1, 0)
    torch.testing.assert_close(
        canonical["component_1_mu"],
        switched["component_2_mu"],
    )
    torch.testing.assert_close(
        canonical["component_2_mu"],
        switched["component_1_mu"],
    )
    torch.testing.assert_close(
        family.component_weights(canonical),
        family.component_weights(switched)[..., [1, 0]],
    )


def test_posterior_diagnostics_respect_case_weights():
    family = FiniteMixture([Normal(), Normal()])
    response = torch.tensor([-2.0, -1.0, 2.0, 3.0], dtype=torch.float64)
    parameters = {
        "component_1_mu": torch.full_like(response, -1.5),
        "component_1_sigma": torch.full_like(response, 0.5),
        "component_2_mu": torch.full_like(response, 2.5),
        "component_2_sigma": torch.full_like(response, 0.5),
        "mixing_1": torch.zeros_like(response),
    }
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=response.dtype)

    result = family.diagnostics(response, parameters, weights=weights)

    torch.testing.assert_close(
        result.posterior_probabilities.sum(dim=-1),
        torch.ones_like(response),
    )
    torch.testing.assert_close(
        result.effective_counts.sum(),
        weights.sum(),
    )
    torch.testing.assert_close(
        result.effective_proportions.sum(),
        response.new_tensor(1.0),
    )
    assert result.classification.tolist() == [0, 0, 1, 1]
    assert 0 <= float(result.mean_entropy) <= torch.log(response.new_tensor(2.0))
    assert 0.5 <= float(result.mean_max_posterior) <= 1.0


def test_mixture_score_matches_autograd_on_the_parameter_scale():
    family = FiniteMixture([Normal(), Normal()])
    response = torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float64)
    parameters = {
        "component_1_mu": torch.tensor([-0.5, 0.0, 0.5], dtype=response.dtype),
        "component_1_sigma": torch.tensor([0.8, 1.0, 1.2], dtype=response.dtype),
        "component_2_mu": torch.tensor([2.0, 2.5, 3.0], dtype=response.dtype),
        "component_2_sigma": torch.tensor([1.2, 1.0, 0.8], dtype=response.dtype),
        "mixing_1": torch.tensor([0.5, 1.0, 2.0], dtype=response.dtype),
    }
    differentiable = {
        parameter: value.clone().requires_grad_()
        for parameter, value in parameters.items()
    }
    gradients = torch.autograd.grad(
        family.log_prob(response, differentiable).sum(),
        tuple(differentiable.values()),
    )
    scores = family.score(response, parameters)

    for parameter, gradient in zip(
        family.parameter_names,
        gradients,
        strict=True,
    ):
        torch.testing.assert_close(scores[parameter], gradient)


def test_sampling_has_the_configured_mixture_moments():
    observation_count = 20_000
    family = FiniteMixture([Normal(), Normal()])
    parameters = {
        "component_1_mu": torch.tensor(0.0, dtype=torch.float64),
        "component_1_sigma": torch.tensor(1.0, dtype=torch.float64),
        "component_2_mu": torch.tensor(4.0, dtype=torch.float64),
        "component_2_sigma": torch.tensor(0.5, dtype=torch.float64),
        "mixing_1": torch.tensor(2.0 / 3.0, dtype=torch.float64).log(),
    }
    distribution = family.distribution(parameters)

    with torch.random.fork_rng():
        torch.manual_seed(2026)
        samples = distribution.sample((observation_count,))

    assert float(samples.mean()) == pytest.approx(float(distribution.mean), abs=0.04)
    assert float(samples.var()) == pytest.approx(
        float(distribution.variance),
        abs=0.08,
    )


def test_mixture_configuration_and_parameter_errors_are_explicit():
    assert MX is FiniteMixture
    with pytest.raises(ValueError, match="at least two"):
        FiniteMixture([Normal()])
    with pytest.raises(ValueError, match="all continuous or all discrete"):
        FiniteMixture([Normal(), Poisson()])
    with pytest.raises(ValueError, match="exist in every"):
        FiniteMixture([Normal(), Normal()], shared_parameters=("nu",))

    family = FiniteMixture([Normal(), Gamma()])
    positive_response = torch.tensor([1.0, 2.0], dtype=torch.float64)
    parameters = family.initial_parameters(positive_response)
    assert torch.isfinite(family.log_prob(positive_response, parameters)).all()
    with pytest.raises(ValueError, match="exchangeable"):
        family.component_order(parameters)
    with pytest.raises(NotImplementedError, match="mixture EM"):
        family.expected_second_derivatives(positive_response, parameters)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixture_likelihood_and_gradients_run_on_cuda():
    device = torch.device("cuda")
    family = FiniteMixture([Normal(), Normal()])
    response = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32, device=device)
    predictors = {
        "component_1_mu": torch.full_like(response, -1.0, requires_grad=True),
        "component_1_sigma": torch.zeros_like(response, requires_grad=True),
        "component_2_mu": torch.full_like(response, 2.0, requires_grad=True),
        "component_2_sigma": torch.zeros_like(response, requires_grad=True),
        "mixing_1": torch.zeros_like(response, requires_grad=True),
    }
    parameters = family.parameters_from_predictors(predictors)

    loss = -family.log_prob(response, parameters).sum()
    gradients = torch.autograd.grad(loss, tuple(predictors.values()))

    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
    assert all(
        gradient.device.type == "cuda" and torch.isfinite(gradient).all()
        for gradient in gradients
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixture_em_fit_runs_on_cuda():
    device = torch.device("cuda")
    response = torch.cat(
        [
            torch.linspace(-2.5, -1.5, 20, device=device),
            torch.linspace(2.0, 4.0, 30, device=device),
        ]
    )
    family = FiniteMixture([Normal(), Normal()])
    design = {
        parameter: torch.ones(
            (response.numel(), 1),
            dtype=response.dtype,
            device=device,
        )
        for parameter in family.parameter_names
    }
    model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=response.dtype,
        device=device,
    )

    result = model.fit_mixture(
        response,
        design,
        control=MixtureControl(
            tolerance=1e-6,
            max_iterations=50,
            m_step_max_iter=50,
        ),
    )

    assert result.converged
    assert result.posterior_probabilities.device.type == "cuda"
    assert result.effective_counts.device.type == "cuda"
    torch.testing.assert_close(
        result.effective_proportions,
        torch.tensor([0.4, 0.6], device=device),
        rtol=2e-3,
        atol=2e-3,
    )
