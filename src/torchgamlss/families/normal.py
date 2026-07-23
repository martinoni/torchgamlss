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

    def score(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Match ``NO()$dldm`` and ``NO()$dldd`` from ``gamlss.dist``."""
        mu, sigma, response = self._broadcast(response, parameters)
        residual = response - mu
        return {
            "mu": residual / sigma.square(),
            "sigma": (residual.square() - sigma.square()) / sigma.pow(3),
        }

    def expected_second_derivatives(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[tuple[str, str], Tensor]:
        """Match the expected second derivatives supplied by R's ``NO`` family.

        These are the quantities used by the GAMLSS fitting algorithm. The
        sigma-sigma entry is not the observation-wise Hessian of ``dNO``.
        """
        _, sigma, response = self._broadcast(response, parameters)
        inverse_variance = sigma.square().reciprocal()
        return {
            ("mu", "mu"): -inverse_variance,
            ("sigma", "sigma"): -2.0 * inverse_variance,
            ("mu", "sigma"): response.new_zeros(response.shape),
        }

    def _broadcast(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        return (
            distribution.loc.broadcast_to(response.shape),
            distribution.scale.broadcast_to(response.shape),
            response,
        )
