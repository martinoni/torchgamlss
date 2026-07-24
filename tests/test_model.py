import pytest
import torch

from torchgamlss import BCCG, BCPE, BCT, GAMLSS, Normal, PSpline


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


def test_family_sampling_with_generator_is_reproducible_and_isolated():
    family = Normal()
    parameters = {
        "mu": torch.zeros(4, dtype=torch.float64),
        "sigma": torch.ones(4, dtype=torch.float64),
    }
    global_state = torch.random.get_rng_state().clone()

    first = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2026),
    )
    second = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2026),
    )

    torch.testing.assert_close(first, second)
    assert torch.equal(torch.random.get_rng_state(), global_state)


@pytest.mark.parametrize(
    ("family", "tau"),
    [
        (BCCG(), None),
        (BCT(), 4.0),
        (BCPE(), 1.5),
    ],
)
def test_box_cox_family_sampling_has_uniform_probability_integral_transform(
    family,
    tau,
):
    observation_count = 3_000
    parameters = {
        "mu": torch.full((observation_count,), 2.0, dtype=torch.float64),
        "sigma": torch.full((observation_count,), 0.25, dtype=torch.float64),
        "nu": torch.tensor([-0.7, 0.0, 0.7], dtype=torch.float64).repeat(
            observation_count // 3
        ),
    }
    if tau is not None:
        parameters["tau"] = torch.full(
            (observation_count,),
            tau,
            dtype=torch.float64,
        )

    response = family.sample(
        parameters,
        generator=torch.Generator().manual_seed(2026),
    )
    probabilities = family.cdf(response, parameters)

    assert response.shape == (observation_count,)
    assert torch.isfinite(response).all()
    assert torch.all(response > 0)
    assert float(probabilities.mean()) == pytest.approx(0.5, abs=0.02)
    assert float(probabilities.var()) == pytest.approx(1.0 / 12.0, abs=0.01)


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


def test_smooth_terms_contribute_to_predictors_and_likelihood():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    term = PSpline.from_data(x, smoothing_parameter=5.0)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
    )
    design = {
        "mu": torch.ones((x.numel(), 1), dtype=x.dtype),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }
    with torch.no_grad():
        model.coefficients["mu"].fill_(0.75)
        term.coefficients.copy_(torch.linspace(-0.5, 0.5, term.coefficients.numel()))

    predictors = model.linear_predictors(design, smooth_covariates={"mu": {"x": x}})

    torch.testing.assert_close(predictors["mu"], 0.75 + term(x))
    assert torch.isfinite(
        model.negative_log_likelihood(
            torch.zeros_like(x),
            design,
            smooth_covariates={"mu": {"x": x}},
        )
    )
    assert model.smooth_penalty() > 0


def test_missing_or_extra_smooth_covariates_are_rejected():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    term = PSpline.from_data(x, smoothing_parameter=5.0)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
    )
    design = {
        "mu": torch.ones((x.numel(), 1), dtype=x.dtype),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }

    with pytest.raises(ValueError, match="missing"):
        model.linear_predictors(design)
    with pytest.raises(ValueError, match="extra"):
        model.linear_predictors(design, smooth_covariates={"mu": {"x": x, "other": x}})


def test_lbfgs_rejects_automatic_smoothing_parameter_selection():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"x": PSpline.from_data(x)}},
    )
    design = {
        "mu": torch.ones((x.numel(), 1), dtype=x.dtype),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }

    with pytest.raises(ValueError, match="fit_rs"):
        model.fit(
            torch.sin(x),
            design,
            smooth_covariates={"mu": {"x": x}},
        )
