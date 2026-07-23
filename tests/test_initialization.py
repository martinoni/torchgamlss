import pandas as pd
import pytest
import torch

from torchgamlss import GAMLSS, NegativeBinomial, Normal, Poisson, RSControl


def test_family_initial_parameters_accept_partial_scalar_and_vector_overrides():
    response = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    sigma = torch.tensor([0.5, 0.75, 1.0], dtype=torch.float64)

    initial = Normal().initial_parameters(
        response,
        {"mu": 3.0, "sigma": sigma},
    )

    torch.testing.assert_close(initial["mu"], torch.full_like(response, 3.0))
    torch.testing.assert_close(initial["sigma"], sigma)
    assert initial["sigma"].data_ptr() != sigma.data_ptr()


def test_partial_user_initial_parameters_can_replace_an_unavailable_default():
    response = torch.ones(3, dtype=torch.float64)

    initial = Normal().initial_parameters(
        response,
        {"sigma": 0.5},
    )

    torch.testing.assert_close(initial["mu"], torch.ones_like(response))
    torch.testing.assert_close(initial["sigma"], torch.full_like(response, 0.5))


def test_partial_nbi_override_avoids_an_unneeded_variance_default():
    response = torch.tensor([2.0], dtype=torch.float64)

    initial = NegativeBinomial().initial_parameters(
        response,
        {"sigma": 0.5},
    )

    torch.testing.assert_close(initial["mu"], response)
    torch.testing.assert_close(initial["sigma"], torch.full_like(response, 0.5))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"unknown": 1.0}, "unknown"),
        ({"mu": [1.0, 2.0]}, "one value per observation"),
        ({"mu": float("nan")}, "finite"),
        ({"sigma": 0.0}, "link domain"),
    ],
)
def test_initial_parameter_validation_rejects_invalid_overrides(values, match):
    response = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    with pytest.raises(ValueError, match=match):
        Normal().initial_parameters(response, values)


def test_rs_uses_partial_user_initial_parameters_on_parameter_scale():
    response = torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=torch.float64)
    design = {
        "mu": torch.ones((4, 1), dtype=torch.float64),
        "sigma": torch.ones((4, 1), dtype=torch.float64),
    }
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)
    expected_parameters = Normal().initial_parameters(response, {"sigma": 2.0})
    expected_deviance = float(
        -2.0 * Normal().log_prob(response, expected_parameters).sum()
    )

    result = model.fit_rs(
        response,
        design,
        initial_parameters={"sigma": 2.0},
        control=RSControl(max_outer_iterations=1),
    )

    assert result.deviance_history[0] == pytest.approx(expected_deviance)


def test_formula_rs_accepts_initial_parameters_from_columns():
    data = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 4.0],
            "mu_start": [0.75, 1.0, 1.5, 2.5],
        }
    )
    model = GAMLSS.from_formula(Poisson(), {"mu": "y ~ 1"}, data)
    expected = Poisson().initial_parameters(
        torch.tensor(data["y"].to_numpy(), dtype=torch.float64),
        {
            "mu": torch.tensor(
                data["mu_start"].to_numpy(),
                dtype=torch.float64,
            )
        },
    )
    expected_deviance = float(
        -2.0
        * Poisson()
        .log_prob(
            torch.tensor(data["y"].to_numpy(), dtype=torch.float64),
            expected,
        )
        .sum()
    )

    result = model.fit_rs_data(
        data,
        initial_parameters={"mu": "mu_start"},
        control=RSControl(max_outer_iterations=1),
    )

    assert result.deviance_history[0] == pytest.approx(expected_deviance)
