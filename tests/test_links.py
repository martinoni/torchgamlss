import pytest
import torch

from torchgamlss.links import IdentityLink, LogitLink, LogLink


@pytest.mark.parametrize(
    ("link", "values"),
    [
        (IdentityLink(), torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)),
        (LogLink(), torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64)),
        (LogitLink(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)),
    ],
)
def test_link_round_trip(link, values):
    recovered = link.inverse(link(values))
    torch.testing.assert_close(recovered, values)
