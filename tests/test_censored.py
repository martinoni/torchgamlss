import csv
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    CensoredFamily,
    CensoredResponse,
    Censoring,
    GeneralizedGamma,
    InverseGaussian,
    LogNormal,
    Poisson,
    RSControl,
    Weibull,
)

REFERENCE_PATH = Path(__file__).parent / "reference" / "censored_reference.csv"
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
with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
    CASES = tuple(dict.fromkeys(row["case"] for row in csv.DictReader(reference_file)))


def _case_rows(case: str) -> list[dict[str, str]]:
    with REFERENCE_PATH.open(newline="", encoding="utf-8") as reference_file:
        return [row for row in csv.DictReader(reference_file) if row["case"] == case]


def _tensor(
    rows: list[dict[str, str]],
    column: str,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return torch.tensor(
        [float(row[column]) for row in rows],
        dtype=dtype,
        device=device,
    )


def _family(
    rows: list[dict[str, str]],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> CensoredFamily:
    observed = _tensor(rows, "observed", dtype=dtype, device=device)
    status = torch.tensor(
        [int(row["status"]) for row in rows],
        dtype=torch.int64,
        device=device,
    )
    upper = None
    if bool((status == int(Censoring.INTERVAL)).any()):
        upper = torch.tensor(
            [
                float(row["upper"]) if row["upper"] else float(row["observed"])
                for row in rows
            ],
            dtype=dtype,
            device=device,
        )
    response = CensoredResponse(observed, status, upper)
    return CensoredFamily(FAMILY_FACTORIES[rows[0]["family"]](), response)


@pytest.mark.parametrize("case", CASES)
def test_censored_likelihood_and_derivatives_match_r_gamlss_cens(case):
    rows = _case_rows(case)
    family = _family(rows)
    response = family.response.observed
    parameters = {
        parameter: _tensor(rows, parameter) for parameter in family.parameter_names
    }

    torch.testing.assert_close(
        family.log_prob(response, parameters),
        _tensor(rows, "log_likelihood"),
        rtol=2e-10,
        atol=2e-10,
    )
    scores = family.score(response, parameters)
    for parameter in family.parameter_names:
        assert torch.isfinite(scores[parameter]).all()
        torch.testing.assert_close(
            scores[parameter],
            _tensor(rows, f"dld{parameter}"),
            rtol=3e-5,
            atol=3e-6,
        )
    second = family.expected_second_derivatives(response, parameters)
    for pair, column in SECOND_DERIVATIVE_COLUMNS.items():
        if pair not in second:
            continue
        torch.testing.assert_close(
            second[pair],
            _tensor(rows, column),
            rtol=3e-8,
            atol=3e-8,
        )


@pytest.mark.parametrize("case", CASES)
def test_censored_score_matches_link_scale_autograd(case):
    rows = _case_rows(case)
    family = _family(rows)
    response = family.response.observed
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


def test_censored_likelihood_uses_all_four_surv_status_codes():
    observed = torch.tensor([1.0, 1.5, 0.7, 1.2], dtype=torch.float64)
    upper = torch.tensor([1.0, 1.5, 0.7, 2.0], dtype=torch.float64)
    status = torch.tensor(
        [
            Censoring.EXACT,
            Censoring.RIGHT,
            Censoring.LEFT,
            Censoring.INTERVAL,
        ]
    )
    response = CensoredResponse(observed, status, upper)
    base = Weibull()
    family = CensoredFamily(base, response)
    parameters = {
        "mu": torch.full_like(observed, 1.8),
        "sigma": torch.full_like(observed, 1.3),
    }
    distribution = base.distribution(parameters)
    expected = torch.stack(
        [
            distribution.log_prob(observed)[0],
            torch.log1p(-distribution.cdf(observed)[1]),
            torch.log(distribution.cdf(observed)[2]),
            torch.log(distribution.cdf(upper)[3] - distribution.cdf(observed)[3]),
        ]
    )

    torch.testing.assert_close(family.log_prob(observed, parameters), expected)
    torch.testing.assert_close(
        family.cdf(observed, parameters), base.cdf(observed, parameters)
    )
    torch.testing.assert_close(
        family.survival(observed, parameters),
        base.survival(observed, parameters),
    )
    torch.testing.assert_close(
        family.hazard(observed, parameters),
        base.hazard(observed, parameters),
    )
    torch.testing.assert_close(
        family.cumulative_hazard(observed, parameters),
        base.cumulative_hazard(observed, parameters),
    )
    probabilities = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    torch.testing.assert_close(
        family.quantile(probabilities, parameters),
        base.quantile(probabilities, parameters),
    )


def test_censored_response_constructors_and_validation():
    time = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    right = CensoredResponse.right(time, torch.tensor([1, 0, 1]))
    left = CensoredResponse.left(time, torch.tensor([False, True, False]))
    interval = CensoredResponse.interval(time, time + 0.5)

    assert right.status.tolist() == [1, 0, 1]
    assert left.status.tolist() == [2, 1, 2]
    assert interval.status.tolist() == [3, 3, 3]
    assert not right.observed.requires_grad
    with pytest.raises(ValueError, match="between 0 and 3"):
        CensoredResponse(time, torch.tensor([0, 1, 4]))
    with pytest.raises(ValueError, match="upper endpoint"):
        CensoredResponse(time, torch.tensor([1, 3, 1]))
    with pytest.raises(ValueError, match="exceed lower"):
        CensoredResponse.interval(time, time)
    with pytest.raises(ValueError, match="binary"):
        CensoredResponse.right(time, torch.tensor([1, 2, 0]))
    with pytest.raises(ValueError, match="continuous"):
        CensoredFamily(Poisson(), right)

    family = CensoredFamily(Weibull(), right)
    with pytest.raises(ValueError, match="stored"):
        family.log_prob(
            time + 0.1,
            {"mu": torch.ones_like(time), "sigma": torch.ones_like(time)},
        )


@pytest.mark.parametrize(
    ("base", "observed", "expected", "mu", "sigma"),
    [
        (Weibull(), 100.0, -10000.0, 1.0, 2.0),
        (
            LogNormal(),
            float(torch.exp(torch.tensor(10.0, dtype=torch.float64))),
            float(torch.special.log_ndtr(torch.tensor(-10.0, dtype=torch.float64))),
            0.0,
            1.0,
        ),
    ],
)
def test_right_censored_log_likelihood_remains_stable_in_far_tail(
    base,
    observed,
    expected,
    mu,
    sigma,
):
    time = torch.tensor([observed], dtype=torch.float64)
    family = CensoredFamily(
        base,
        CensoredResponse.right(time, torch.tensor([0])),
    )
    parameters = {
        "mu": torch.tensor([mu], dtype=torch.float64, requires_grad=True),
        "sigma": torch.tensor([sigma], dtype=torch.float64),
    }
    log_likelihood = family.log_prob(time, parameters)
    gradient = torch.autograd.grad(log_likelihood.sum(), parameters["mu"])[0]

    torch.testing.assert_close(
        log_likelihood,
        torch.tensor([expected], dtype=torch.float64),
    )
    assert torch.isfinite(gradient).all()


def test_right_censored_weibull_fits_with_rs_lbfgs_and_formula_data():
    count = 240
    base = Weibull()
    parameters = {
        "mu": torch.full((count,), 2.0, dtype=torch.float64),
        "sigma": torch.full((count,), 1.5, dtype=torch.float64),
    }
    event_time = base.sample(
        parameters,
        generator=torch.Generator().manual_seed(8675309),
    )
    censor_time = torch.full_like(event_time, 2.2)
    event = event_time <= censor_time
    observed = torch.minimum(event_time, censor_time)
    family = CensoredFamily(
        base,
        CensoredResponse.right(observed, event),
    )
    data = pd.DataFrame({"time": observed.numpy()})
    formulas = {"mu": "time ~ 1", "sigma": "~ 1"}

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
        max_iter=300,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    assert 0 < int((~event).sum()) < count
    assert rs_result.converged
    assert lbfgs_result.converged
    assert rs_result.negative_log_likelihood == pytest.approx(
        lbfgs_result.negative_log_likelihood,
        rel=2e-5,
        abs=2e-5,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            rs_model.coefficients[parameter],
            lbfgs_model.coefficients[parameter],
            rtol=2e-4,
            atol=2e-4,
        )
    prediction = rs_model.predict_data(data)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    curves = rs_model.predict_survival_data(
        data.iloc[:4],
        times=[0.5, 1.0, 2.0, 4.0],
    )
    assert curves.family == "WEIcens"
    assert curves.survival.shape == (4, 4)
    assert curves.hazard.shape == (4, 4)
    assert curves.cumulative_hazard.shape == (4, 4)
    assert (torch.diff(curves.survival, dim=-1) <= 0).all()
    assert (torch.diff(curves.cumulative_hazard, dim=-1) >= 0).all()
    torch.testing.assert_close(
        curves.cumulative_hazard,
        -torch.log(curves.survival),
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_censored_likelihoods_and_gradients_run_on_cuda(case):
    device = torch.device("cuda")
    rows = _case_rows(case)
    family = _family(rows, dtype=torch.float32, device=device)
    response = family.response.observed
    predictors = {
        parameter: family.links[parameter](
            _tensor(rows, parameter, dtype=torch.float32, device=device)
        ).requires_grad_()
        for parameter in family.parameter_names
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
