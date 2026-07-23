"""Beta GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.stats import beta as scipy_beta
from torch import Tensor
from torch.distributions import Beta as TorchBeta

from torchgamlss.families._scipy import scipy_cdf
from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogitLink


class Beta(Family):
    """Beta family compatible with ``gamlss.dist::BE(mu, sigma)``.

    ``mu`` is the mean and ``Var(Y) = sigma**2 * mu * (1 - mu)``.
    """

    name = "BE"
    parameter_names = ("mu", "sigma")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or LogitLink(),
            "sigma": sigma_link or LogitLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchBeta:
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        precision = sigma.square().reciprocal() - 1.0
        return TorchBeta(
            concentration1=mu * precision,
            concentration0=(1.0 - mu) * precision,
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        mu, sigma, response = self._broadcast(response, parameters)
        precision = sigma.square().reciprocal() - 1.0
        concentration1 = mu * precision
        concentration0 = (1.0 - mu) * precision
        return scipy_cdf(
            response,
            scipy_beta.cdf,
            response,
            concentration1,
            concentration0,
        )

    def score(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        precision = sigma.square().reciprocal() - 1.0
        concentration1 = mu * precision
        concentration0 = (1.0 - mu) * precision
        concentration = concentration1 + concentration0
        return {
            "mu": precision
            * (
                -torch.digamma(concentration1)
                + torch.digamma(concentration0)
                + torch.log(response)
                - torch.log1p(-response)
            ),
            "sigma": -(2.0 / sigma.pow(3))
            * (
                mu
                * (
                    -torch.digamma(concentration1)
                    + torch.digamma(concentration)
                    + torch.log(response)
                )
                + (1.0 - mu)
                * (
                    -torch.digamma(concentration0)
                    + torch.digamma(concentration)
                    + torch.log1p(-response)
                )
            ),
        }

    def expected_second_derivatives(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[tuple[str, str], Tensor]:
        mu, sigma, response = self._broadcast(response, parameters)
        precision = sigma.square().reciprocal() - 1.0
        concentration1 = mu * precision
        concentration0 = (1.0 - mu) * precision
        concentration = concentration1 + concentration0
        trigamma1 = torch.special.polygamma(1, concentration1)
        trigamma0 = torch.special.polygamma(1, concentration0)
        return {
            ("mu", "mu"): -precision.square() * (trigamma1 + trigamma0),
            ("sigma", "sigma"): -(4.0 / sigma.pow(6))
            * (
                mu.square() * trigamma1
                + (1.0 - mu).square() * trigamma0
                - torch.special.polygamma(1, concentration)
            ),
            ("mu", "sigma"): (2.0 * precision / sigma.pow(3))
            * (mu * trigamma1 - (1.0 - mu) * trigamma0),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::BE``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.5)
        return defaults

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        if (
            response.ndim != 1
            or response.numel() < 1
            or not torch.isfinite(response).all()
            or (response <= 0).any()
            or (response >= 1).any()
        ):
            raise ValueError(
                f"BE {context} requires a finite response strictly between zero and one"
            )

    def _broadcast(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        concentration = distribution.concentration1 + distribution.concentration0
        mu = distribution.concentration1 / concentration
        sigma = (concentration + 1.0).rsqrt()
        return torch.broadcast_tensors(mu, sigma, response)
