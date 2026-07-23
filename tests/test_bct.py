import csv
from pathlib import Path

import pandas as pd
import pytest
import torch
from scipy.stats import t as scipy_student_t
from torch.distributions import StudentT

from torchgamlss import BCCG, BCT, GAMLSS, BoxCoxT, RSControl
from torchgamlss.families.bct import _student_t_cdf

REFERENCE_DIR = Path(__file__).parent / "reference"


def _reference() -> dict[str, torch.Tensor]:
    with (REFERENCE_DIR / "bct_reference.csv").open(
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
    with (REFERENCE_DIR / "bct_rs_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return next(csv.DictReader(data_file))


def test_bct_density_cdf_links_derivatives_and_initial_values_match_r():
    reference = _reference()
    family = BoxCoxT()
    parameters = {
        parameter: reference[parameter] for parameter in family.parameter_names
    }

    torch.testing.assert_close(
        family.log_prob(reference["y"], parameters),
        reference["log_density"],
        rtol=3e-12,
        atol=3e-12,
    )
    torch.testing.assert_close(
        family.cdf(reference["y"], parameters),
        reference["cdf"],
        rtol=3e-12,
        atol=3e-12,
    )
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            family.links[parameter](reference[parameter]),
            reference[f"eta_{parameter}"],
        )
    scores = family.score(reference["y"], parameters)
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            scores[parameter],
            reference[f"dld{parameter}"],
            rtol=2e-10,
            atol=2e-10,
        )
    second = family.expected_second_derivatives(reference["y"], parameters)
    for pair, column in {
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
    }.items():
        torch.testing.assert_close(second[pair], reference[column])
    initial = family.initial_parameters(reference["y"])
    for parameter in family.parameter_names:
        torch.testing.assert_close(
            initial[parameter],
            reference[f"initial_{parameter}"],
        )
    assert BCT is BoxCoxT


def test_bct_autograd_matches_scores_after_default_links():
    reference = _reference()
    family = BoxCoxT()
    predictors = {
        parameter: reference[f"eta_{parameter}"].clone().requires_grad_()
        for parameter in family.parameter_names
    }
    parameters = family.parameters_from_predictors(predictors)

    gradients = torch.autograd.grad(
        family.log_prob(reference["y"], parameters).sum(),
        tuple(predictors.values()),
    )
    scores = family.score(reference["y"], parameters)

    torch.testing.assert_close(gradients[0], scores["mu"], rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(
        gradients[1],
        scores["sigma"] * parameters["sigma"],
        rtol=3e-11,
        atol=3e-11,
    )
    torch.testing.assert_close(gradients[2], scores["nu"], rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(
        gradients[3],
        scores["tau"] * parameters["tau"],
        rtol=5e-3,
        atol=2e-5,
    )


def test_bct_student_t_cdf_matches_scipy_and_has_finite_df_gradient():
    values = torch.tensor(
        [-10.0, -2.0, -1.0, 0.0, 0.5, 3.0, 10.0],
        dtype=torch.float64,
    )
    degrees_of_freedom = torch.tensor(
        [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    probabilities = _student_t_cdf(values, degrees_of_freedom)
    expected = torch.tensor(
        scipy_student_t.cdf(
            values.numpy(),
            degrees_of_freedom.detach().numpy(),
        ),
        dtype=torch.float64,
    )

    torch.testing.assert_close(probabilities, expected, rtol=3e-13, atol=3e-13)
    gradient = torch.autograd.grad(probabilities.sum(), degrees_of_freedom)[0]
    assert torch.isfinite(gradient).all()


def test_bct_nu_zero_is_log_student_t_with_finite_shape_gradients():
    family = BoxCoxT()
    response = torch.tensor([0.8, 2.8, 7.0], dtype=torch.float64)
    mu = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(
        [0.15, 0.3, 0.55],
        dtype=torch.float64,
        requires_grad=True,
    )
    nu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor([2.5, 7.0, 20.0], dtype=torch.float64, requires_grad=True)
    parameters = {"mu": mu, "sigma": sigma, "nu": nu, "tau": tau}
    z = torch.log(response / mu) / sigma
    expected_log_density = (
        StudentT(tau).log_prob(z) - torch.log(response) - torch.log(sigma)
    )

    log_density = family.log_prob(response, parameters)
    torch.testing.assert_close(log_density, expected_log_density)
    expected_cdf = torch.tensor(
        scipy_student_t.cdf(
            z.detach().numpy(),
            tau.detach().numpy(),
        ),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        family.cdf(response, parameters),
        expected_cdf,
        rtol=3e-13,
        atol=3e-13,
    )
    gradients = torch.autograd.grad(log_density.sum(), (mu, sigma, nu, tau))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_bct_large_tau_log_density_uses_bccg_limit():
    response = torch.tensor([0.8, 2.8, 7.0], dtype=torch.float64)
    parameters = {
        "mu": torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64),
        "sigma": torch.tensor([0.15, 0.3, 0.55], dtype=torch.float64),
        "nu": torch.tensor([-0.5, 0.0, 0.8], dtype=torch.float64),
        "tau": torch.full((3,), 1e6 + 1.0, dtype=torch.float64),
    }

    torch.testing.assert_close(
        BCT().log_prob(response, parameters),
        BCCG().log_prob(
            response,
            {parameter: parameters[parameter] for parameter in ("mu", "sigma", "nu")},
        ),
    )


def test_bct_formula_rs_fit_and_predictions_match_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "bct_fit_data.csv")
    reference = _fit_reference()
    model = GAMLSS.from_formula(
        BCT(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
            "nu": "~ w + offset(nu_offset)",
            "tau": "~ 1",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-7,
            max_outer_iterations=200,
            inner_tolerance=1e-7,
            max_inner_iterations=200,
        ),
    )

    assert result.converged
    for parameter, columns in {
        "mu": ("mu_intercept", "mu_x"),
        "sigma": ("sigma_intercept", "sigma_z"),
        "nu": ("nu_intercept", "nu_w"),
        "tau": ("tau_intercept",),
    }.items():
        torch.testing.assert_close(
            model.coefficients[parameter],
            torch.tensor(
                [float(reference[column]) for column in columns],
                dtype=torch.float64,
            ),
            rtol=3e-6,
            atol=3e-6,
        )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]),
        rel=3e-6,
        abs=3e-6,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]),
        rel=3e-6,
        abs=3e-6,
    )
    predictions = model.predict_data(data)
    assert tuple(predictions) == ("mu", "sigma", "nu", "tau")
    assert all(values.shape == (len(data),) for values in predictions.values())
    assert (predictions["mu"] > 0).all()
    assert (predictions["sigma"] > 0).all()
    assert torch.isfinite(predictions["nu"]).all()
    assert (predictions["tau"] > 0).all()


@pytest.mark.parametrize(
    "response",
    [
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.tensor([float("nan"), 1.0], dtype=torch.float64),
    ],
)
def test_bct_rejects_invalid_responses(response):
    family = BCT()
    parameters = {
        "mu": torch.ones(2, dtype=torch.float64),
        "sigma": torch.full((2,), 0.2, dtype=torch.float64),
        "nu": torch.zeros(2, dtype=torch.float64),
        "tau": torch.full((2,), 5.0, dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="strictly positive"):
        family.log_prob(response, parameters)
