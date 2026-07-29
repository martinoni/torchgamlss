"""Log-normal GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.distributions import LogNormal as TorchLogNormal

from torchgamlss.families.base import Family
from torchgamlss.links import IdentityLink, Link, LogLink


class LogNormal(Family):
    """Log-normal family compatible with ``gamlss.dist::LOGNO``."""

    name = "LOGNO"
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

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchLogNormal:
        return TorchLogNormal(
            loc=parameters["mu"],
            scale=parameters["sigma"],
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def _log_cdf(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        standardized = (torch.log(response) - mu) / sigma
        return torch.special.log_ndtr(standardized)

    def _log_survival(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        standardized = (torch.log(response) - mu) / sigma
        return torch.special.log_ndtr(-standardized)

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        difference = torch.log(response) - mu
        return {
            "mu": difference / sigma.square(),
            "sigma": (difference.square() - sigma.square()) / sigma.pow(3),
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        self.validate_response(response)
        _, sigma, response = self._broadcast(response, parameters)
        return {
            ("mu", "mu"): -sigma.square().reciprocal(),
            ("sigma", "sigma"): -2.0 / sigma.square(),
            ("mu", "sigma"): torch.zeros_like(response),
        }

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        super().validate_response(response, context=context)
        if bool((response <= 0).any()):
            raise ValueError(f"LOGNO {context} requires strictly positive responses")

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        log_response = torch.log(response)
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (log_response + log_response.mean()) / 2.0
        if "sigma" in parameters:
            sigma = log_response.std(correction=1)
            defaults["sigma"] = sigma.expand_as(response)
        return defaults

    def _broadcast(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        return torch.broadcast_tensors(
            distribution.loc,
            distribution.scale,
            response,
        )


LOGNO = LogNormal
