"""Gamma GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.stats import gamma as scipy_gamma
from torch import Tensor
from torch.distributions import Gamma as TorchGamma

from torchgamlss.families._scipy import scipy_cdf
from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogLink


class Gamma(Family):
    """Gamma family using the GAMLSS ``GA(mu, sigma)`` parameterization.

    ``mu`` is the mean and ``sigma`` is the coefficient of variation, so the
    variance is ``sigma**2 * mu**2``.
    """

    name = "GA"
    parameter_names = ("mu", "sigma")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or LogLink(),
            "sigma": sigma_link or LogLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchGamma:
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        inverse_variance = sigma.square().reciprocal()
        return TorchGamma(
            concentration=inverse_variance,
            rate=inverse_variance / mu,
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        mu, sigma, response = self._broadcast(response, parameters)
        shape = sigma.square().reciprocal()
        scale = mu * sigma.square()
        zeros = torch.zeros_like(response)
        return scipy_cdf(
            response,
            scipy_gamma.cdf,
            response,
            shape,
            zeros,
            scale,
        )

    def score(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Match ``GA()$dldm`` and ``GA()$dldd`` from ``gamlss.dist``."""
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        sigma_squared = sigma.square()
        inverse_variance = sigma_squared.reciprocal()
        return {
            "mu": (response - mu) / (sigma_squared * mu.square()),
            "sigma": (2.0 / sigma.pow(3))
            * (
                response / mu
                - torch.log(response)
                + torch.log(mu)
                + torch.log(sigma_squared)
                - 1.0
                + torch.digamma(inverse_variance)
            ),
        }

    def expected_second_derivatives(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[tuple[str, str], Tensor]:
        """Match the Fisher-scoring derivatives supplied by R's ``GA``."""
        mu, sigma, response = self._broadcast(response, parameters)
        inverse_variance = sigma.square().reciprocal()
        return {
            ("mu", "mu"): -(inverse_variance / mu.square()),
            ("sigma", "sigma"): 4.0 / sigma.pow(4)
            - 4.0 * torch.special.polygamma(1, inverse_variance) / sigma.pow(6),
            ("mu", "sigma"): response.new_zeros(response.shape),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::GA``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.ones_like(response)
        return defaults

    def _broadcast(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        mu = distribution.concentration / distribution.rate
        sigma = distribution.concentration.rsqrt()
        mu, sigma, response = torch.broadcast_tensors(mu, sigma, response)
        return mu, sigma, response

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
        ):
            raise ValueError(
                f"GA {context} requires a finite strictly positive response"
            )
