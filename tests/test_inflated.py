import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    BEINF,
    BEINF0,
    BEINF1,
    BEOI,
    BEZI,
    GAMLSS,
    ZINBI,
    ZIP,
    Beta,
    PointMassFamily,
    RSControl,
)

REFERENCE_DIR = Path(__file__).parent / "reference"
FAMILY_FACTORIES = {
    "ZIP": ZIP,
    "ZINBI": ZINBI,
    "BEZI": BEZI,
    "BEOI": BEOI,
    "BEINF": BEINF,
    "BEINF0": BEINF0,
    "BEINF1": BEINF1,
}
SCORE_COLUMNS = {
    "mu": "dldmu",
    "sigma": "dldsigma",
    "nu": "dldnu",
    "tau": "dldtau",
}
SECOND_DERIVATIVE_COLUMNS = {
    ("mu", "mu"): "d2ldmu2",
    ("sigma", "sigma"): "d2ldsigma2",
    ("nu", "nu"): "d2ldnu2",
    ("tau", "tau"): "d2ldtau2",
    ("mu", "sigma"): "d2ldmudsigma",
    ("mu", "nu"): "d2ldmunu",
    ("mu", "tau"): "d2ldmutau",
    ("sigma", "nu"): "d2ldsigmanu",
    ("sigma", "tau"): "d2ldsigmatau",
    ("nu", "tau"): "d2ldnutau",
}


def _rows(name: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return list(csv.DictReader(data_file))


def _case_rows(name: str, value: str) -> list[dict[str, str]]:
    key = "family" if name.startswith("inflated") else "case"
    return [row for row in _rows(name) if row[key] == value]


def _tensor(rows: list[dict[str, str]], column: str) -> torch.Tensor:
    return torch.tensor(
        [float(row[column]) for row in rows],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("family_code", tuple(FAMILY_FACTORIES))
def test_inflated_family_distribution_and_derivatives_match_r(family_code):
    rows = _case_rows("inflated_family_reference.csv", family_code)
    family = FAMILY_FACTORIES[family_code]()
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        _tensor(rows, "log_density"),
        rtol=2e-11,
        atol=2e-11,
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        _tensor(rows, "cdf"),
        rtol=2e-9,
        atol=2e-10,
    )
    quantiles = family.quantile(_tensor(rows, "probability"), parameters)
    torch.testing.assert_close(
        torch.diagonal(quantiles),
        _tensor(rows, "quantile"),
        rtol=2e-8,
        atol=2e-9,
    )

    scores = family.score(response, parameters)
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            scores[parameter],
            _tensor(rows, SCORE_COLUMNS[parameter]),
            rtol=2e-9,
            atol=2e-10,
        )
        torch.testing.assert_close(
            family.links[parameter](parameters[parameter]),
            _tensor(rows, f"eta_{parameter}"),
            rtol=2e-12,
            atol=2e-12,
        )

    second = family.expected_second_derivatives(response, parameters)
    for pair, column in SECOND_DERIVATIVE_COLUMNS.items():
        if not all(parameter in family.parameter_names for parameter in pair):
            continue
        torch.testing.assert_close(
            second[pair],
            _tensor(rows, column),
            rtol=5e-8,
            atol=5e-9,
        )

    initial = family.initial_parameters(response)
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            initial[parameter],
            _tensor(rows, f"initial_{parameter}"),
            rtol=2e-12,
            atol=2e-12,
        )


@pytest.mark.parametrize("family_code", tuple(FAMILY_FACTORIES))
def test_inflated_family_autograd_matches_parameter_scores(family_code):
    rows = _case_rows("inflated_family_reference.csv", family_code)
    family = FAMILY_FACTORIES[family_code]()
    response = _tensor(rows, "y")
    predictors = {
        parameter: _tensor(rows, f"eta_{parameter}").requires_grad_()
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
            rtol=2e-8,
            atol=2e-9,
        )


@pytest.mark.parametrize("family_code", tuple(FAMILY_FACTORIES))
def test_inflated_intercept_only_rs_fit_matches_r(family_code):
    data = pd.read_csv(REFERENCE_DIR / "inflated_fit_data.csv")
    data = data.loc[data["family"] == family_code].reset_index(drop=True)
    reference = (
        pd.read_csv(REFERENCE_DIR / "inflated_fit_reference.csv")
        .set_index("family")
        .loc[family_code]
    )
    family = FAMILY_FACTORIES[family_code]()
    model = GAMLSS.from_formula(
        family,
        {
            parameter: "y ~ 1" if parameter == "mu" else "~ 1"
            for parameter in family.parameter_names
        },
        data,
        dtype=torch.float64,
    )

    fit = model.fit_rs_data(
        data,
        control=RSControl(
            max_outer_iterations=500,
            outer_tolerance=1e-10,
        ),
    )
    fitted_parameters = model.predict_data(data)

    assert fit.converged
    assert reference["converged"]
    assert fit.outer_iterations == reference["iterations"]
    assert fit.global_deviance == pytest.approx(
        reference["global_deviance"],
        rel=2e-11,
        abs=2e-11,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            model.coefficients[parameter].detach()[0],
            torch.tensor(
                reference[f"coefficient_{parameter}"],
                dtype=torch.float64,
            ),
            rtol=2e-8,
            atol=2e-9,
        )
        torch.testing.assert_close(
            fitted_parameters[parameter].detach()[0],
            torch.tensor(
                reference[f"fitted_{parameter}"],
                dtype=torch.float64,
            ),
            rtol=2e-8,
            atol=2e-9,
        )


def _generic_family(case: str) -> PointMassFamily:
    if case == "beta_zero_probability":
        return PointMassFamily(
            Beta(),
            points=(0.0,),
            mass_parameter_names=("xi0",),
            parameterization="probability",
        )
    if case == "beta_one_probability":
        return PointMassFamily(
            Beta(),
            points=(1.0,),
            mass_parameter_names=("xi1",),
            parameterization="probability",
        )
    if case == "beta_zero_one_odds":
        return PointMassFamily(
            Beta(),
            points=(0.0, 1.0),
            mass_parameter_names=("xi0", "xi1"),
            parameterization="odds",
        )
    raise AssertionError(f"unknown generic point-mass case {case}")


@pytest.mark.parametrize(
    "case",
    (
        "beta_zero_probability",
        "beta_one_probability",
        "beta_zero_one_odds",
    ),
)
def test_generic_point_mass_distribution_matches_gamlss_inf(case):
    rows = _case_rows("point_mass_reference.csv", case)
    family = _generic_family(case)
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        _tensor(rows, "log_density"),
        rtol=2e-10,
        atol=2e-11,
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        _tensor(rows, "cdf"),
        rtol=2e-9,
        atol=2e-10,
    )
    torch.testing.assert_close(
        family.cdf_left(response, parameters),
        _tensor(rows, "cdf_left"),
        rtol=2e-9,
        atol=2e-10,
    )
    quantiles = family.quantile(_tensor(rows, "probability"), parameters)
    torch.testing.assert_close(
        torch.diagonal(quantiles),
        _tensor(rows, "quantile"),
        rtol=2e-8,
        atol=2e-9,
    )


@pytest.mark.parametrize(
    "case",
    (
        "beta_zero_probability",
        "beta_one_probability",
        "beta_zero_one_odds",
    ),
)
def test_point_mass_quantile_residuals_randomize_only_at_atoms(case):
    rows = _case_rows("point_mass_reference.csv", case)
    family = _generic_family(case)
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }
    observation_count = response.numel()
    model = GAMLSS(
        family,
        {parameter: observation_count for parameter in family.parameter_names},
        dtype=torch.float64,
    )
    design = {
        parameter: torch.eye(observation_count, dtype=torch.float64)
        for parameter in family.parameter_names
    }
    with torch.no_grad():
        for parameter in family.parameter_names:
            model.coefficients[parameter].copy_(
                family.links[parameter](parameters[parameter])
            )

    uniforms = _tensor(rows, "uniform")
    lower = family.cdf_left(response, parameters)
    upper = family.cdf(response, parameters)
    probability = lower + uniforms * (upper - lower)
    torch.testing.assert_close(
        probability,
        _tensor(rows, "randomized_probability"),
        rtol=2e-10,
        atol=2e-11,
    )
    torch.testing.assert_close(
        model.quantile_residuals(response, design, uniforms=uniforms),
        _tensor(rows, "randomized_residual"),
        rtol=2e-9,
        atol=2e-10,
    )


def test_point_mass_family_validates_configuration_and_response_support():
    with pytest.raises(ValueError, match="exactly one"):
        PointMassFamily(
            Beta(),
            points=(0.0, 1.0),
            parameterization="probability",
        )
    with pytest.raises(ValueError, match="increasing"):
        PointMassFamily(Beta(), points=(1.0, 0.0))
    with pytest.raises(ValueError, match="collide"):
        PointMassFamily(
            Beta(),
            points=(0.0,),
            mass_parameter_names=("mu",),
        )

    zero = BEZI()
    zero.validate_response(torch.tensor([0.0, 0.5], dtype=torch.float64))
    with pytest.raises(ValueError, match="between zero and one"):
        zero.validate_response(torch.tensor([0.0, 1.0], dtype=torch.float64))

    one = BEOI()
    one.validate_response(torch.tensor([0.5, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="between zero and one"):
        one.validate_response(torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_point_mass_sampling_is_reproducible_and_preserves_moments():
    family = ZIP()
    parameters = {
        "mu": torch.full((20_000,), 2.5, dtype=torch.float64),
        "sigma": torch.full((20_000,), 0.35, dtype=torch.float64),
    }
    distribution = family.distribution(parameters)
    expected_mean = (1.0 - parameters["sigma"]) * parameters["mu"]
    expected_variance = (
        parameters["mu"]
        * (1.0 - parameters["sigma"])
        * (1.0 + parameters["mu"] * parameters["sigma"])
    )
    torch.testing.assert_close(distribution.mean, expected_mean)
    torch.testing.assert_close(distribution.variance, expected_variance)

    first = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(20260729),
    )
    second = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(20260729),
    )
    torch.testing.assert_close(first, second)
    assert float(first.mean()) == pytest.approx(1.625, abs=0.04)
    assert float(first.var()) == pytest.approx(3.046875, abs=0.12)


@pytest.mark.parametrize(
    ("family", "values", "observation_count"),
    [
        (ZIP(), {"mu": 2.4, "sigma": 0.32}, 500),
        (
            ZINBI(),
            {"mu": 2.8, "sigma": 0.45, "nu": 0.28},
            650,
        ),
        (
            BEINF(),
            {"mu": 0.58, "sigma": 0.35, "nu": 0.18, "tau": 0.25},
            650,
        ),
    ],
)
def test_inflated_intercept_only_rs_and_lbfgs_fits_agree(
    family,
    values,
    observation_count,
):
    parameters = {
        parameter: torch.full(
            (observation_count,),
            value,
            dtype=torch.float64,
        )
        for parameter, value in values.items()
    }
    response = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(29072026 + observation_count),
    )
    design = {
        parameter: torch.ones(
            (observation_count, 1),
            dtype=torch.float64,
        )
        for parameter in family.parameter_names
    }
    rs_model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=torch.float64,
    )
    lbfgs_model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=torch.float64,
    )

    rs_fit = rs_model.fit_rs(
        response,
        design,
        control=RSControl(
            max_outer_iterations=100,
            outer_tolerance=1e-7,
        ),
    )
    lbfgs_fit = lbfgs_model.fit(
        response,
        design,
        max_iter=120,
    )

    assert rs_fit.converged
    assert lbfgs_fit.converged
    rs_parameters = rs_model.predict(design)
    lbfgs_parameters = lbfgs_model.predict(design)
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            rs_parameters[parameter],
            lbfgs_parameters[parameter],
            rtol=2e-4,
            atol=2e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("family", "response", "parameters"),
    [
        (
            ZIP(),
            [0.0, 1.0, 4.0],
            {"mu": [0.7, 2.0, 5.0], "sigma": [0.1, 0.3, 0.6]},
        ),
        (
            ZINBI(),
            [0.0, 2.0, 6.0],
            {
                "mu": [0.8, 2.5, 6.0],
                "sigma": [0.2, 0.6, 1.2],
                "nu": [0.1, 0.3, 0.55],
            },
        ),
        (
            BEINF(),
            [0.0, 0.45, 1.0],
            {
                "mu": [0.2, 0.55, 0.8],
                "sigma": [0.25, 0.45, 0.65],
                "nu": [0.1, 0.3, 0.8],
                "tau": [0.2, 0.5, 1.1],
            },
        ),
    ],
)
def test_inflated_likelihood_cdf_and_gradients_run_on_cuda(
    family,
    response,
    parameters,
):
    response_tensor = torch.tensor(
        response,
        dtype=torch.float64,
        device="cuda",
    )
    predictors = {
        parameter: family.links[parameter](
            torch.tensor(values, dtype=torch.float64, device="cuda")
        ).requires_grad_()
        for parameter, values in parameters.items()
    }
    linked = family.parameters_from_predictors(predictors)
    log_prob = family.log_prob(response_tensor, linked)
    cdf = family.cdf(response_tensor, linked)
    gradients = torch.autograd.grad(log_prob.sum(), tuple(predictors.values()))

    assert log_prob.device.type == "cuda"
    assert cdf.device.type == "cuda"
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(cdf).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
