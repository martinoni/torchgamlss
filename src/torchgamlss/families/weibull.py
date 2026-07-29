"""Weibull GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.distributions import Weibull as TorchWeibull

from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogLink


class Weibull(Family):
    """Weibull family compatible with ``gamlss.dist::WEI``."""

    name = "WEI"
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

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchWeibull:
        return TorchWeibull(
            scale=parameters["mu"],
            concentration=parameters["sigma"],
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
        powered = (response / mu).pow(sigma)
        return torch.log(-torch.expm1(-powered))

    def _log_survival(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        return -(response / mu).pow(sigma)

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        ratio = response / mu
        powered = ratio.pow(sigma)
        return {
            "mu": (powered - 1.0) * sigma / mu,
            "sigma": sigma.reciprocal() - torch.log(ratio) * (powered - 1.0),
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        return {
            ("mu", "mu"): -sigma.square() / mu.square(),
            ("sigma", "sigma"): -1.82368 / sigma.square(),
            ("mu", "sigma"): 0.422784 / mu,
        }

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        super().validate_response(response, context=context)
        if bool((response <= 0).any()):
            raise ValueError(f"WEI {context} requires strictly positive responses")

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        log_response = torch.log(response)
        variance = log_response.var(correction=1)
        tiny = torch.finfo(response.dtype).tiny
        sigma = 1.283 / variance.clamp_min(tiny).sqrt()
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = torch.exp(log_response + 0.5772 / sigma)
        if "sigma" in parameters:
            defaults["sigma"] = sigma.expand_as(response)
        return defaults

    def _broadcast(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        return torch.broadcast_tensors(
            distribution.scale,
            distribution.concentration,
            response,
        )


WEI = Weibull
