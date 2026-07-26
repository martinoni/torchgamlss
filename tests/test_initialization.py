import pandas as pd
import pytest
import torch

from torchgamlss import (
    BCCG,
    GAMLSS,
    MiniBatchControl,
    NegativeBinomial,
    Normal,
    Poisson,
    RSControl,
)


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


def test_minibatch_initial_parameters_make_box_cox_zero_model_safe():
    response = torch.tensor(
        [1.5, 2.0, 2.5, 3.5, 5.0],
        dtype=torch.float64,
    )
    designs = {
        parameter: torch.ones((response.numel(), 1), dtype=response.dtype)
        for parameter in BCCG().parameter_names
    }
    starts = {"mu": 2.75, "sigma": 0.2, "nu": 0.5}
    family = BCCG()
    expected_parameters = family.initial_parameters(response, starts)
    expected_objective = float(
        -family.log_prob(response, expected_parameters).mean()
    )
    model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        dtype=response.dtype,
    )

    result = model.fit_minibatch(
        response,
        designs,
        initial_parameters=starts,
        control=MiniBatchControl(
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            shuffle=False,
            minimum_epochs=1,
        ),
    )

    assert result.objective_history[0] == pytest.approx(
        expected_objective,
        rel=1e-12,
        abs=1e-12,
    )
    assert torch.isfinite(
        torch.tensor(result.negative_log_likelihood)
    )


def test_formula_minibatch_accepts_parameter_starts_from_columns():
    data = pd.DataFrame(
        {
            "y": [1.5, 2.0, 3.0, 4.0],
            "mu_start": [2.0, 2.5, 3.0, 3.5],
        }
    )
    family = BCCG()
    model = GAMLSS.from_formula(
        family,
        {"mu": "y ~ 1", "sigma": "~ 1", "nu": "~ 1"},
        data,
    )
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    parameter_values = {
        "mu": torch.full_like(
            response,
            float(data["mu_start"].mean()),
        ),
        "sigma": torch.full_like(response, 0.2),
        "nu": torch.full_like(response, 0.5),
    }
    expected_objective = float(
        -family.log_prob(response, parameter_values).mean()
    )

    result = model.fit_minibatch_data(
        data,
        initial_parameters={
            "mu": "mu_start",
            "sigma": 0.2,
            "nu": 0.5,
        },
        control=MiniBatchControl(
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            shuffle=False,
            minimum_epochs=1,
        ),
    )

    assert result.objective_history[0] == pytest.approx(
        expected_objective,
        rel=1e-12,
        abs=1e-12,
    )


def test_minibatch_initial_parameters_require_an_explicit_intercept():
    response = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    designs = {
        "mu": torch.arange(1.0, 4.0, dtype=torch.float64).unsqueeze(-1),
        "sigma": torch.ones((3, 1), dtype=torch.float64),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="intercept containing only ones"):
        model.fit_minibatch(
            response,
            designs,
            initial_parameters={"mu": 2.0, "sigma": 1.0},
            control=MiniBatchControl(epochs=1, minimum_epochs=1),
        )


def test_minibatch_starts_center_existing_terms_with_case_weights():
    response = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    offset = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float64)
    designs = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((3, 1), dtype=torch.float64),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.coefficients["mu"][1] = 2.0
    expected_intercept = (
        weights * (4.0 - 2.0 * x - offset)
    ).sum() / weights.sum()
    expected_parameters = {
        "mu": expected_intercept + 2.0 * x + offset,
        "sigma": torch.ones_like(response),
    }
    expected_objective = float(
        (
            -Normal().log_prob(response, expected_parameters) * weights
        ).sum()
        / weights.sum()
    )

    result = model.fit_minibatch(
        response,
        designs,
        weights=weights,
        offsets={"mu": offset},
        initial_parameters={"mu": 4.0, "sigma": 1.0},
        control=MiniBatchControl(
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            shuffle=False,
            minimum_epochs=1,
        ),
    )

    assert result.objective_history[0] == pytest.approx(
        expected_objective,
        rel=1e-12,
        abs=1e-12,
    )
