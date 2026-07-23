"""Normal GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor
from torch.distributions import Normal as TorchNormal

from torchgamlss.families.base import Family
from torchgamlss.links import IdentityLink, Link, LogLink


class Normal(Family):
    """Normal family using the GAMLSS ``NO(mu, sigma)`` parameterization."""

    name = "NO"
    parameter_names = ("mu", "sigma")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or IdentityLink(),
            "sigma": sigma_link or LogLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchNormal:
        return TorchNormal(
            loc=parameters["mu"],
            scale=parameters["sigma"],
            validate_args=True,
        )
