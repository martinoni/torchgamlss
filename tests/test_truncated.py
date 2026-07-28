import csv
from pathlib import Path

import pytest
import torch

from torchgamlss import GAMLSS, Normal, Poisson, RSControl, TruncatedFamily

REFERENCE_PATH = Path(__file__).parent / "reference" / "truncated_reference.csv"
CASES = (
    "normal_left",
    "normal_right",
    "normal_both",
    "poisson_left",
    "poisson_right",
    "poisson_both",
)


def _case_rows(case: str) -> list[dict[str, str]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
        return [
            row for row in csv.DictReader(reference_file) if row["case"] == case
        ]


def _tensor(rows: list[dict[str, str]], column: str) -> torch.Tensor:
    return torch.tensor(
        [float(row[column]) for row in rows],
        dtype=torch.float64,
    )


def _family(rows: list[dict[str, str]]) -> TruncatedFamily:
    row = rows[0]
    base = Normal() if row["family"] == "NO" else Poisson()
    lower = float(row["lower"]) if row["lower"] else None
    upper = float(row["upper"]) if row["upper"] else None
    return TruncatedFamily(base, lower=lower, upper=upper)


@pytest.mark.parametrize("case", CASES)
def test_truncated_density_cdf_quantile_and_derivatives_match_r(case):
    rows = _case_rows(case)
    family = _family(rows)
    response = _tensor(rows, "y")
    parameters = {"mu": _tensor(rows, "mu")}
    if rows[0]["family"] == "NO":
        parameters["sigma"] = _tensor(rows, "sigma")

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        _tensor(rows, "log_density"),
        rtol=5e-13,
        atol=5e-13,
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        _tensor(rows, "cdf"),
        rtol=5e-13,
        atol=5e-13,
    )
    quantiles = family.quantile(
        _tensor(rows, "probability"),
        parameters,
    )
    torch.testing.assert_close(
        torch.diagonal(quantiles),
        _tensor(rows, "quantile"),
        rtol=5e-13,
        atol=5e-13,
    )
    assert quantiles.shape == (len(rows), len(rows))

    scores = family.score(response, parameters)
    torch.testing.assert_close(
        scores["mu"],
        _tensor(rows, "dldmu"),
        rtol=8e-8,
        atol=8e-8,
    )
    if "sigma" in parameters:
        torch.testing.assert_close(
            scores["sigma"],
            _tensor(rows, "dldsigma"),
            rtol=8e-8,
            atol=8e-8,
        )

    second = family.expected_second_derivatives(response, parameters)
    torch.testing.assert_close(
        second[("mu", "mu")],
        _tensor(rows, "d2ldmu2"),
        rtol=5e-13,
        atol=5e-13,
    )
    if "sigma" in parameters:
        torch.testing.assert_close(
            second[("sigma", "sigma")],
            _tensor(rows, "d2ldsigma2"),
            rtol=5e-13,
            atol=5e-13,
        )
        torch.testing.assert_close(
            second[("mu", "sigma")],
            _tensor(rows, "d2ldmudsigma"),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("family", "response", "predictors"),
    (
        (
            TruncatedFamily(Normal(), lower=-0.5, upper=1.8),
            torch.tensor([-0.2, 0.7, 1.5], dtype=torch.float64),
            {
                "mu": torch.tensor([-0.1, 0.4, 1.0], dtype=torch.float64),
                "sigma": torch.tensor([-0.4, 0.1, 0.3], dtype=torch.float64),
            },
        ),
        (
            TruncatedFamily(Poisson(), lower=0, upper=7),
            torch.tensor([1.0, 3.0, 6.0], dtype=torch.float64),
            {
                "mu": torch.tensor([-0.3, 0.9, 1.6], dtype=torch.float64),
            },
        ),
    ),
)
def test_truncated_log_likelihood_autograd_matches_parameter_scores(
    family,
    response,
    predictors,
):
    predictors = {
        parameter: value.requires_grad_()
        for parameter, value in predictors.items()
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
            rtol=2e-11,
            atol=2e-11,
        )


def test_continuous_and_discrete_bounds_follow_gamlss_tr_conventions():
    continuous = TruncatedFamily(Normal(), lower=0.0, upper=2.0)
    continuous.validate_response(
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    )

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

    with pytest.raises(ValueError, match="at least one"):
        TruncatedFamily(base)
    with pytest.raises(ValueError, match="less than"):
        TruncatedFamily(base, lower=2, upper=1)
    with pytest.raises(ValueError, match="integer"):
        TruncatedFamily(Poisson(), lower=0.5)
    with pytest.raises(ValueError, match="fixed scalar"):
        TruncatedFamily(base, lower=torch.tensor([0.0, 1.0]))


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
    poisson_parameters = {
        "mu": torch.full((1000,), 2.5, dtype=torch.float64)
    }
    counts = poisson.sample(
        poisson_parameters,
        generator=torch.Generator().manual_seed(2028),
    )
    assert ((counts > 0) & (counts < 5)).all()
    assert (counts == torch.floor(counts)).all()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_truncated_normal_and_poisson_likelihoods_run_on_cuda():
    device = torch.device("cuda")
    cases = (
        (
            TruncatedFamily(Normal(), lower=0.0),
            torch.tensor([0.2, 1.4], device=device),
            {
                "mu": torch.tensor([-0.3, 0.8], device=device),
                "sigma": torch.tensor([0.7, 1.1], device=device),
            },
        ),
        (
            TruncatedFamily(Poisson(), lower=0, upper=7),
            torch.tensor([1.0, 5.0], device=device),
            {"mu": torch.tensor([1.2, 4.5], device=device)},
        ),
    )
    for family, response, parameters in cases:
        predictors = {
            parameter: family.links[parameter](value).requires_grad_()
            for parameter, value in parameters.items()
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
