import pytest
import torch

from torchgamlss import GAMLSS, Gamma, Normal, PSpline, TermContributions


def test_predict_returns_link_and_response_scales_for_normal():
    dtype = torch.float64
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=dtype)
    design = {
        "mu": torch.tensor(
            [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=dtype,
        ),
        "sigma": torch.ones((3, 1), dtype=dtype),
    }
    offsets = {
        "mu": torch.tensor([0.1, 0.2, 0.3], dtype=dtype),
        "sigma": torch.tensor(-0.2, dtype=dtype),
    }
    with torch.no_grad():
        model.coefficients["mu"].copy_(torch.tensor([1.5, 0.4], dtype=dtype))
        model.coefficients["sigma"].fill_(torch.log(torch.tensor(2.0)))

    link = model.predict(design, offsets, type="link")
    response = model.predict(design, offsets, type="response")

    expected_mu = design["mu"] @ model.coefficients["mu"] + offsets["mu"]
    expected_sigma_link = (
        design["sigma"] @ model.coefficients["sigma"] + offsets["sigma"]
    )
    torch.testing.assert_close(link["mu"], expected_mu)
    torch.testing.assert_close(link["sigma"], expected_sigma_link)
    torch.testing.assert_close(response["mu"], expected_mu)
    torch.testing.assert_close(response["sigma"], expected_sigma_link.exp())


def test_term_predictions_reconstruct_each_link_predictor():
    dtype = torch.float64
    training_x = torch.linspace(-1.0, 1.0, 30, dtype=dtype)
    term = PSpline.from_data(training_x, smoothing_parameter=8.0)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=dtype,
    )
    with torch.no_grad():
        model.coefficients["mu"].copy_(torch.tensor([0.7, 0.2], dtype=dtype))
        model.coefficients["sigma"].fill_(-1.0)
        term.coefficients.copy_(
            torch.linspace(-0.3, 0.4, term.coefficients.numel(), dtype=dtype)
        )

    new_x = torch.tensor([-0.9, -0.35, 0.0, 0.45, 0.8], dtype=dtype)
    design = {
        "mu": torch.column_stack((torch.ones_like(new_x), new_x)),
        "sigma": torch.ones((new_x.numel(), 1), dtype=dtype),
    }
    offsets = {"mu": torch.linspace(0.0, 0.1, new_x.numel(), dtype=dtype)}
    smooth_covariates = {"mu": {"x": new_x}}

    terms = model.predict(
        design,
        offsets,
        smooth_covariates=smooth_covariates,
        type="terms",
    )
    link = model.predict(
        design,
        offsets,
        smooth_covariates=smooth_covariates,
        type="link",
    )

    assert isinstance(terms["mu"], TermContributions)
    assert terms["mu"].linear.shape == (new_x.numel(), 2)
    assert set(terms["mu"].smooth) == {"x"}
    torch.testing.assert_close(
        terms["mu"].linear,
        design["mu"] * model.coefficients["mu"],
    )
    torch.testing.assert_close(terms["mu"].smooth["x"], term(new_x))
    torch.testing.assert_close(terms["mu"].offset, offsets["mu"])
    torch.testing.assert_close(terms["mu"].total, link["mu"])
    torch.testing.assert_close(terms["sigma"].total, link["sigma"])
    torch.testing.assert_close(
        terms["sigma"].offset,
        torch.zeros_like(new_x),
    )


def test_predict_response_uses_gamma_parameterization():
    dtype = torch.float64
    model = GAMLSS(Gamma(), {"mu": 1, "sigma": 1}, dtype=dtype)
    design = {
        "mu": torch.ones((4, 1), dtype=dtype),
        "sigma": torch.ones((4, 1), dtype=dtype),
    }
    with torch.no_grad():
        model.coefficients["mu"].fill_(torch.log(torch.tensor(2.5)))
        model.coefficients["sigma"].fill_(torch.log(torch.tensor(0.4)))

    response = model.predict(design, type="response")
    distribution = model.distribution(design)

    torch.testing.assert_close(
        response["mu"],
        torch.full((4,), 2.5, dtype=dtype),
    )
    torch.testing.assert_close(
        response["sigma"],
        torch.full((4,), 0.4, dtype=dtype),
    )
    torch.testing.assert_close(distribution.mean, response["mu"])
    torch.testing.assert_close(
        distribution.variance,
        response["sigma"].square() * response["mu"].square(),
    )


def test_predict_preserves_autograd():
    dtype = torch.float64
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=dtype)
    design = {
        "mu": torch.ones((3, 1), dtype=dtype),
        "sigma": torch.ones((3, 1), dtype=dtype),
    }

    prediction = model.predict(design, type="response")
    prediction["mu"].sum().backward()

    assert model.coefficients["mu"].grad is not None
    torch.testing.assert_close(
        model.coefficients["mu"].grad,
        torch.tensor([3.0], dtype=dtype),
    )


def test_predict_rejects_an_unknown_type():
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="type"):
        model.predict({}, type="distribution")


@pytest.mark.parametrize(
    "design",
    [
        {
            "mu": torch.ones((3, 1), dtype=torch.float64),
            "sigma": torch.ones((2, 1), dtype=torch.float64),
        },
        {
            "mu": torch.ones((3, 1), dtype=torch.float32),
            "sigma": torch.ones((3, 1), dtype=torch.float64),
        },
        {
            "mu": torch.tensor(
                [[1.0], [float("nan")], [1.0]],
                dtype=torch.float64,
            ),
            "sigma": torch.ones((3, 1), dtype=torch.float64),
        },
    ],
)
def test_predict_rejects_incompatible_design_matrices(design):
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)

    with pytest.raises(ValueError):
        model.predict(design)


def test_predict_rejects_an_incompatible_offset():
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)
    design = {
        "mu": torch.ones((3, 1), dtype=torch.float64),
        "sigma": torch.ones((3, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="offset"):
        model.predict(
            design,
            offsets={"mu": torch.ones(3, dtype=torch.float32)},
        )
