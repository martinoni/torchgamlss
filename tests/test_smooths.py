import pytest
import torch

from torchgamlss import PSpline


def test_pspline_basis_and_penalty_match_r_pb():
    covariate = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    term = PSpline.from_data(covariate, smoothing_parameter=10.0, intervals=10)

    expected_basis = torch.tensor(
        [
            [
                0.12229584900729593,
                0.6575261400215593,
                0.2200209572487208,
                0.0001570537224245078,
                0.0,
            ],
            [
                0.04589308253335528,
                0.5659024932010855,
                0.38109394598770485,
                0.00711047827785337,
                0.0,
            ],
            [
                0.01060244205290356,
                0.4141356821567531,
                0.5391168851939412,
                0.0361449905964001,
                0.0,
            ],
        ],
        dtype=torch.float64,
    )
    expected_penalty = torch.tensor(
        [
            [1.0, -2.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, -2.0, 1.0],
        ],
        dtype=torch.float64,
    )

    assert term.coefficients.numel() == 13
    assert not term.estimates_smoothing_parameter
    assert term.smoothing_method is None
    torch.testing.assert_close(
        term.basis(covariate)[:3, :5], expected_basis, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        term.penalty_matrix()[:3, :5], expected_penalty, rtol=0.0, atol=0.0
    )


def test_pspline_forward_penalty_and_effective_degrees_of_freedom():
    covariate = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
    term = PSpline.from_data(covariate, smoothing_parameter=4.0, intervals=8)
    with torch.no_grad():
        term.coefficients.copy_(
            torch.arange(term.coefficients.numel(), dtype=torch.float64).square()
        )

    assert term(covariate).shape == covariate.shape
    expected_penalty = 4.0 * (term.penalty_matrix() @ term.coefficients).square().sum()
    torch.testing.assert_close(term.quadratic_penalty(), expected_penalty)
    edf = term.effective_degrees_of_freedom(covariate, torch.ones_like(covariate))
    assert 2.0 < float(edf) < term.coefficients.numel()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_bound": 1.0, "upper_bound": 1.0, "smoothing_parameter": 1.0},
        {"lower_bound": 0.0, "upper_bound": 1.0, "smoothing_parameter": -1.0},
        {
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "smoothing_parameter": 1.0,
            "intervals": 0,
        },
    ],
)
def test_invalid_pspline_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PSpline(**kwargs)


def test_pspline_from_data_rejects_a_constant_covariate():
    with pytest.raises(ValueError, match="distinct"):
        PSpline.from_data(torch.ones(10, dtype=torch.float64), 1.0)


def test_automatic_pspline_smoothing_state_is_serialized():
    covariate = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64)
    term = PSpline.from_data(covariate)
    assert term.smoothing_parameter == pytest.approx(10.0)
    term._set_fitted_smoothing_parameter(3.25)
    restored = PSpline.from_data(covariate)

    restored.load_state_dict(term.state_dict())

    assert restored.estimates_smoothing_parameter
    assert restored.smoothing_method == "ML"
    assert restored.smoothing_parameter == pytest.approx(3.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_smoothing_parameter": 0.0},
        {"initial_smoothing_parameter": float("inf")},
        {"smoothing_method": "REML"},
    ],
)
def test_invalid_automatic_smoothing_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PSpline(0.0, 1.0, None, **kwargs)


def test_pspline_target_degrees_of_freedom_uses_r_pb_semantics():
    covariate = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    term = PSpline.from_data(covariate, degrees_of_freedom=3.0)

    assert term.estimates_smoothing_parameter
    assert term.smoothing_method == "DF"
    assert term.target_effective_degrees_of_freedom == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothing_parameter": 1.0, "degrees_of_freedom": 3.0},
        {"degrees_of_freedom": -1.0},
        {"degrees_of_freedom": float("inf")},
        {"degrees_of_freedom": 21.0},
    ],
)
def test_invalid_target_degrees_of_freedom_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PSpline(0.0, 1.0, **kwargs)
