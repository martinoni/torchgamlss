import pytest
import torch

from torchgamlss.links import IdentityLink, InverseLink, LogitLink, LogLink


@pytest.mark.parametrize(
    ("link", "values"),
    [
        (IdentityLink(), torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)),
        (InverseLink(), torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64)),
        (LogLink(), torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64)),
        (LogitLink(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)),
    ],
)
def test_link_round_trip(link, values):
    recovered = link.inverse(link(values))
    torch.testing.assert_close(recovered, values)


@pytest.mark.parametrize(
    ("link", "predictors", "expected"),
    [
        (
            IdentityLink(),
            torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
        ),
        (
            InverseLink(),
            torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64),
            -torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64),
        ),
        (
            LogLink(),
            torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
            torch.exp(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)),
        ),
        (
            LogitLink(),
            torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
            torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
            * (
                1.0 - torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
            ),
        ),
    ],
)
def test_inverse_link_derivative(link, predictors, expected):
    torch.testing.assert_close(link.inverse_derivative(predictors), expected)
