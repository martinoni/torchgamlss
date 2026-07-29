import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    BCCG,
    BCPE,
    BCT,
    GAMLSS,
    PE,
    TF,
    Beta,
    Gamma,
    NegativeBinomial,
    Normal,
    Poisson,
    RSControl,
    TruncatedFamily,
)

REFERENCE_PATH = Path(__file__).parent / "reference" / "truncated_reference.csv"
FAMILY_FACTORIES = {
    "NO": Normal,
    "GA": Gamma,
    "PO": Poisson,
    "NBI": NegativeBinomial,
    "BE": Beta,
    "BCCG": BCCG,
    "TF": TF,
    "PE": PE,
    "BCT": BCT,
    "BCPE": BCPE,
}
with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
    CASES = tuple(dict.fromkeys(row["case"] for row in csv.DictReader(reference_file)))

CATALOG_VARYING_CASES = (
    "normal_varying_both",
    "gamma_varying_both",
    "poisson_varying_both",
    "nbi_varying_both",
    "beta_varying_both",
    "bccg_varying_both",
    "tf_varying_both",
    "pe_varying_both",
    "bct_varying_both",
    "bcpe_varying_both",
)
SECOND_DERIVATIVE_COLUMNS = {
    ("mu", "mu"): "d2ldmu2",
    ("sigma", "sigma"): "d2ldsigma2",
    ("nu", "nu"): "d2ldnu2",
    ("tau", "tau"): "d2ldtau2",
    ("mu", "sigma"): "d2ldmudsigma",
    ("mu", "nu"): "d2ldmudnu",
    ("mu", "tau"): "d2ldmudtau",
    ("sigma", "nu"): "d2ldsigmadnu",
    ("sigma", "tau"): "d2ldsigmadtau",
    ("nu", "tau"): "d2ldnudtau",
}


def _case_rows(case: str) -> list[dict[str, str]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
        return [row for row in csv.DictReader(reference_file) if row["case"] == case]


def _tensor(rows: list[dict[str, str]], column: str) -> torch.Tensor:
    return torch.tensor(
        [float(row[column]) for row in rows],
        dtype=torch.float64,
    )


def _family(rows: list[dict[str, str]]) -> TruncatedFamily:
    row = rows[0]
    base = FAMILY_FACTORIES[row["family"]]()
    varying = row["varying"] == "TRUE"

    def bound(column: str) -> float | torch.Tensor | None:
        values = [entry[column] for entry in rows]
        if not values[0]:
            return None
        if varying:
            return torch.tensor(
                [float(value) for value in values],
                dtype=torch.float64,
            )
        return float(values[0])

    lower = bound("lower")
    upper = bound("upper")
    return TruncatedFamily(base, lower=lower, upper=upper)


@pytest.mark.parametrize("case", CASES)
def test_truncated_density_cdf_quantile_and_derivatives_match_r(case):
    rows = _case_rows(case)
    family = _family(rows)
    response = _tensor(rows, "y")
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        _tensor(rows, "log_density"),
        rtol=5e-10,
        atol=5e-11,
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        _tensor(rows, "cdf"),
        rtol=5e-9,
        atol=5e-10,
    )
    quantiles = family.quantile(
        _tensor(rows, "probability"),
        parameters,
    )
    torch.testing.assert_close(
        torch.diagonal(quantiles),
        _tensor(rows, "quantile"),
        rtol=5e-9,
        atol=5e-10,
    )
    assert quantiles.shape == (len(rows), len(rows))

    scores = family.score(response, parameters)
    for parameter in family.parameter_names:
        column = f"dld{parameter}"
        assert torch.isfinite(scores[parameter]).all()
        if not all(row[column] for row in rows):
            continue
        torch.testing.assert_close(
            scores[parameter],
            _tensor(rows, column),
            rtol=2e-5,
            atol=1e-6,
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


@pytest.mark.parametrize("case", CATALOG_VARYING_CASES)
def test_truncated_log_likelihood_autograd_matches_parameter_scores(
    case,
):
    rows = _case_rows(case)
    family = _family(rows)
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
        inverse_derivative = family.links[parameter].inverse_derivative(
            predictors[parameter]
        )
        torch.testing.assert_close(
            gradient,
            scores[parameter] * inverse_derivative,
            rtol=2e-7,
            atol=2e-8,
        )


def test_continuous_and_discrete_bounds_follow_gamlss_tr_conventions():
    continuous = TruncatedFamily(Normal(), lower=0.0, upper=2.0)
    continuous.validate_response(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))

    discrete = TruncatedFamily(Poisson(), lower=0, upper=4)
    discrete.validate_response(torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="open interval"):
        discrete.validate_response(torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="open interval"):
        discrete.validate_response(torch.tensor([3.0, 4.0]))

    parameters = {"mu": torch.tensor([2.0], dtype=torch.float64)}
    torch.testing.assert_close(
        discrete.cdf(torch.tensor([-1.0]), parameters),
        torch.tensor([0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        discrete.cdf(torch.tensor([4.0]), parameters),
        torch.tensor([1.0], dtype=torch.float64),
    )


def test_truncated_family_validates_bounds_and_preserves_base_metadata():
    base = Normal()
    family = TruncatedFamily(base, lower=0)

    assert family.family is base
    assert family.name == "NOtr"
    assert family.parameter_names == base.parameter_names
    assert family.links == base.links
    assert not family.is_discrete
    assert not family.varying

    varying = TruncatedFamily(
        base,
        lower=torch.tensor([-1.0, 0.0]),
        upper=2.0,
    )
    assert varying.varying
    torch.testing.assert_close(
        varying.lower,
        torch.tensor([-1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="at least one"):
        TruncatedFamily(base)
    with pytest.raises(ValueError, match="less than"):
        TruncatedFamily(base, lower=2, upper=1)
    with pytest.raises(ValueError, match="integer"):
        TruncatedFamily(Poisson(), lower=0.5)
    with pytest.raises(ValueError, match="one-dimensional"):
        TruncatedFamily(base, lower=torch.ones(2, 2))
    with pytest.raises(ValueError, match="cannot require gradients"):
        TruncatedFamily(base, lower=torch.tensor([0.0], requires_grad=True))
    with pytest.raises(ValueError, match="finite"):
        TruncatedFamily(base, lower=torch.tensor([0.0, torch.inf]))
    with pytest.raises(ValueError, match="same observation count"):
        TruncatedFamily(
            base,
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
    with pytest.raises(ValueError, match="every observation"):
        TruncatedFamily(
            base,
            lower=torch.tensor([0.0, 2.0]),
            upper=torch.tensor([1.0, 1.5]),
        )
    with pytest.raises(ValueError, match="integer"):
        TruncatedFamily(Poisson(), lower=torch.tensor([0.0, 1.5]))


def test_observation_specific_bounds_validate_rows_and_support_description():
    family = TruncatedFamily(
        Normal(),
        lower=torch.tensor([-1.0, 0.0, 1.0]),
        upper=torch.tensor([0.5, 1.5, 3.0]),
    )
    family.validate_response(torch.tensor([-0.5, 0.8, 2.5]))

    with pytest.raises(ValueError, match="one value per observation"):
        family.validate_response(torch.tensor([-0.5, 0.8]))
    with pytest.raises(ValueError, match="observation-specific"):
        family.validate_response(torch.tensor([-1.1, 0.8, 2.5]))

    parameters = {
        "mu": torch.tensor([-0.3, 0.5, 1.8], dtype=torch.float64),
        "sigma": torch.tensor([0.8, 1.0, 0.7], dtype=torch.float64),
    }
    quantiles = family.quantile([0.2, 0.5, 0.8], parameters)
    assert quantiles.shape == (3, 3)
    assert (quantiles >= family.lower.to(dtype=torch.float64).unsqueeze(-1)).all()
    assert (quantiles <= family.upper.to(dtype=torch.float64).unsqueeze(-1)).all()

    with pytest.raises(ValueError, match="has 3 rows"):
        family.distribution(
            {
                "mu": torch.zeros(2, dtype=torch.float64),
                "sigma": torch.ones(2, dtype=torch.float64),
            }
        )


def test_truncated_sampling_is_reproducible_and_respects_support():
    observation_count = 6000
    normal = TruncatedFamily(Normal(), lower=0.0, upper=2.0)
    normal_parameters = {
        "mu": torch.full((observation_count,), 0.7, dtype=torch.float64),
        "sigma": torch.full((observation_count,), 0.8, dtype=torch.float64),
    }
    first = normal.sample(
        normal_parameters,
        generator=torch.Generator().manual_seed(2027),
    )
    second = normal.sample(
        normal_parameters,
        generator=torch.Generator().manual_seed(2027),
    )
    probabilities = normal.cdf(first, normal_parameters)

    torch.testing.assert_close(first, second)
    assert ((first >= 0) & (first <= 2)).all()
    assert probabilities.mean() == pytest.approx(0.5, abs=0.015)
    assert probabilities.var(correction=1) == pytest.approx(
        1.0 / 12.0,
        abs=0.005,
    )

    poisson = TruncatedFamily(Poisson(), lower=0, upper=5)
    poisson_parameters = {"mu": torch.full((1000,), 2.5, dtype=torch.float64)}
    counts = poisson.sample(
        poisson_parameters,
        generator=torch.Generator().manual_seed(2028),
    )
    assert ((counts > 0) & (counts < 5)).all()
    assert (counts == torch.floor(counts)).all()


def test_observation_specific_sampling_respects_each_interval():
    observation_count = 4000
    lower = torch.linspace(-1.0, 1.0, observation_count, dtype=torch.float64)
    upper = lower + 2.0
    family = TruncatedFamily(Normal(), lower=lower, upper=upper)
    parameters = {
        "mu": lower + 0.8,
        "sigma": torch.full_like(lower, 0.7),
    }
    samples = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2031),
    )
    probabilities = family.cdf(samples, parameters)

    assert ((samples >= lower) & (samples <= upper)).all()
    assert probabilities.mean() == pytest.approx(0.5, abs=0.015)
    assert probabilities.var(correction=1) == pytest.approx(
        1.0 / 12.0,
        abs=0.005,
    )


@pytest.mark.parametrize("case", CATALOG_VARYING_CASES)
def test_observation_specific_sampling_runs_across_family_catalog(case):
    rows = _case_rows(case)
    family = _family(rows)
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }
    with torch.random.fork_rng():
        torch.manual_seed(2034)
        samples = family.distribution(parameters).sample(torch.Size([128]))
    probabilities = family.cdf(samples, parameters)

    assert samples.shape == (128, len(rows))
    assert torch.isfinite(samples).all()
    assert torch.isfinite(probabilities).all()
    if family.lower is not None:
        lower = torch.as_tensor(family.lower, dtype=samples.dtype)
        comparison = samples > lower if family.is_discrete else samples >= lower
        assert comparison.all()
    if family.upper is not None:
        upper = torch.as_tensor(family.upper, dtype=samples.dtype)
        comparison = samples < upper if family.is_discrete else samples <= upper
        assert comparison.all()


@pytest.mark.parametrize(
    ("family", "parameters", "observation_count"),
    (
        (
            TruncatedFamily(Normal(), lower=0),
            {"mu": 0.6, "sigma": 0.8},
            120,
        ),
        (
            TruncatedFamily(Poisson(), lower=0),
            {"mu": 2.4},
            180,
        ),
    ),
)
def test_truncated_family_fits_with_rs_and_differentiable_lbfgs(
    family,
    parameters,
    observation_count,
):
    parameter_rows = {
        parameter: torch.full(
            (observation_count,),
            value,
            dtype=torch.float64,
        )
        for parameter, value in parameters.items()
    }
    response = family.sample(
        parameter_rows,
        generator=torch.Generator().manual_seed(77),
    )
    designs = {
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
    rs_result = rs_model.fit_rs(
        response,
        designs,
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=200,
            inner_tolerance=1e-8,
            max_inner_iterations=200,
        ),
    )
    lbfgs_model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=torch.float64,
    )
    lbfgs_result = lbfgs_model.fit(
        response,
        designs,
        max_iter=300,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    assert rs_result.converged
    assert lbfgs_result.converged
    assert rs_result.negative_log_likelihood == pytest.approx(
        lbfgs_result.negative_log_likelihood,
        rel=1e-7,
        abs=1e-7,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            rs_model.coefficients[parameter],
            lbfgs_model.coefficients[parameter],
            rtol=1e-4,
            atol=1e-4,
        )


def test_observation_specific_bounds_fit_with_rs_and_lbfgs():
    observation_count = 120
    lower = torch.linspace(-0.5, 0.5, observation_count, dtype=torch.float64)
    upper = lower + 2.5
    family = TruncatedFamily(Normal(), lower=lower, upper=upper)
    parameters = {
        "mu": torch.full_like(lower, 0.8),
        "sigma": torch.full_like(lower, 0.7),
    }
    response = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2032),
    )
    designs = {
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
    rs_result = rs_model.fit_rs(
        response,
        designs,
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=200,
            inner_tolerance=1e-8,
            max_inner_iterations=200,
        ),
    )
    lbfgs_model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=torch.float64,
    )
    lbfgs_result = lbfgs_model.fit(
        response,
        designs,
        max_iter=300,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    assert rs_result.converged
    assert lbfgs_result.converged
    assert rs_result.negative_log_likelihood == pytest.approx(
        lbfgs_result.negative_log_likelihood,
        rel=1e-7,
        abs=1e-7,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            rs_model.coefficients[parameter],
            lbfgs_model.coefficients[parameter],
            rtol=1e-4,
            atol=1e-4,
        )


def test_observation_specific_bounds_fit_and_predict_from_formula_data():
    observation_count = 80
    x = torch.linspace(-1.0, 1.0, observation_count, dtype=torch.float64)
    lower = -0.5 + 0.2 * x
    upper = lower + 2.4
    family = TruncatedFamily(Normal(), lower=lower, upper=upper)
    response = family.sample(
        {
            "mu": 0.7 + 0.3 * x,
            "sigma": torch.full_like(x, 0.6),
        },
        generator=torch.Generator().manual_seed(2033),
    )
    data = pd.DataFrame({"x": x.numpy(), "y": response.numpy()})
    model = GAMLSS.from_formula(
        family,
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )

    result = model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=200,
            inner_tolerance=1e-8,
            max_inner_iterations=200,
        ),
    )
    quantiles = model.predict_quantiles_data(
        data,
        probabilities=[0.1, 0.5, 0.9],
    ).quantiles

    assert result.converged
    assert quantiles.shape == (observation_count, 3)
    assert (quantiles >= lower.unsqueeze(-1)).all()
    assert (quantiles <= upper.unsqueeze(-1)).all()


@pytest.mark.parametrize("case", CATALOG_VARYING_CASES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_truncated_catalog_likelihoods_run_on_cuda(case):
    device = torch.device("cuda")
    rows = _case_rows(case)
    family = _family(rows)
    response = _tensor(rows, "y").to(device=device, dtype=torch.float32)
    predictors = {
        parameter: family.links[parameter](
            _tensor(rows, parameter).to(device=device, dtype=torch.float32)
        ).requires_grad_()
        for parameter in family.parameter_names
    }
    linked_parameters = family.parameters_from_predictors(predictors)
    loss = -family.log_prob(response, linked_parameters).sum()
    gradients = torch.autograd.grad(loss, tuple(predictors.values()))

    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
    assert all(
        gradient.device.type == "cuda" and torch.isfinite(gradient).all()
        for gradient in gradients
    )
