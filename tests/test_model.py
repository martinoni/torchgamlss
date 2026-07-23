import torch

from torchgamlss import GAMLSS, Normal


def _example():
    dtype = torch.float64
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=dtype)
    design = {
        "mu": torch.tensor([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=dtype),
        "sigma": torch.ones((3, 1), dtype=dtype),
    }
    return model, design


def test_normal_parameters_follow_their_predictors():
    model, design = _example()
    with torch.no_grad():
        model.coefficients["mu"].copy_(torch.tensor([2.0, 0.5]))
        model.coefficients["sigma"].copy_(torch.tensor([0.0]))

    distribution = model(design)

    torch.testing.assert_close(
        distribution.loc,
        torch.tensor([1.5, 2.0, 2.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        distribution.scale,
        torch.ones(3, dtype=torch.float64),
    )


def test_negative_log_likelihood_is_differentiable():
    model, design = _example()
    response = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    loss = model.negative_log_likelihood(response, design)
    loss.backward()

    assert loss.ndim == 0
    for coefficient in model.coefficients.values():
        assert coefficient.grad is not None
        assert torch.isfinite(coefficient.grad).all()


def test_invalid_reduction_is_rejected():
    model, design = _example()
    response = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    try:
        model.negative_log_likelihood(response, design, reduction="median")
    except ValueError as error:
        assert "reduction" in str(error)
    else:
        raise AssertionError("invalid reduction was accepted")


def test_weights_and_offsets_are_applied_to_the_likelihood():
    dtype = torch.float64
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=dtype)
    design = {
        "mu": torch.ones((3, 1), dtype=dtype),
        "sigma": torch.ones((3, 1), dtype=dtype),
    }
    offsets = {
        "mu": torch.tensor([0.5, -0.5, 1.0], dtype=dtype),
        "sigma": torch.log(torch.tensor([2.0, 1.0, 0.5], dtype=dtype)),
    }
    response = torch.tensor([1.0, -1.0, 2.0], dtype=dtype)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    expected = (
        -torch.distributions.Normal(offsets["mu"], offsets["sigma"].exp()).log_prob(
            response
        )
        * weights
    )

    actual = model.negative_log_likelihood(
        response,
        design,
        weights=weights,
        offsets=offsets,
        reduction="none",
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        model.negative_log_likelihood(
            response, design, weights=weights, offsets=offsets
        ),
        expected.sum(),
    )
    torch.testing.assert_close(
        model.negative_log_likelihood(
            response,
            design,
            weights=weights,
            offsets=offsets,
            reduction="mean",
        ),
        expected.sum() / weights.sum(),
    )


def test_invalid_likelihood_weights_are_rejected():
    model, design = _example()
    response = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    for weights in (
        torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([1.0, float("nan"), 1.0], dtype=torch.float64),
    ):
        try:
            model.negative_log_likelihood(response, design, weights=weights)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid weights were accepted")


def test_unknown_parameter_offset_is_rejected():
    model, design = _example()
    response = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    try:
        model.negative_log_likelihood(
            response,
            design,
            offsets={"nu": torch.zeros(3, dtype=torch.float64)},
        )
    except ValueError as error:
        assert "Offsets" in str(error)
    else:
        raise AssertionError("unknown offset was accepted")
