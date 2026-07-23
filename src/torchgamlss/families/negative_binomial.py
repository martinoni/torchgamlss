"""Negative-binomial type I GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.stats import nbinom as scipy_negative_binomial
from torch import Tensor
from torch.distributions import NegativeBinomial as TorchNegativeBinomial

from torchgamlss.families._scipy import scipy_cdf
from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogLink


class NegativeBinomial(Family):
    """Negative binomial with the ``gamlss.dist::NBI(mu, sigma)`` convention.

    ``mu`` is the mean and ``Var(Y) = mu + sigma * mu**2``.
    """

    name = "NBI"
    parameter_names = ("mu", "sigma")
    is_discrete = True

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

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchNegativeBinomial:
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return TorchNegativeBinomial(
            total_count=sigma.reciprocal(),
            logits=torch.log(mu * sigma),
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        mu, sigma, response = self._broadcast(response, parameters)
        total_count = sigma.reciprocal()
        success_probability = (1.0 + mu * sigma).reciprocal()
        return scipy_cdf(
            response,
            scipy_negative_binomial.cdf,
            response,
            total_count,
            success_probability,
        )

    def score(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        denominator = 1.0 + mu * sigma
        sigma_score = -sigma.pow(-2) * (
            torch.digamma(response + sigma.reciprocal())
            - torch.digamma(sigma.reciprocal())
            - torch.log(denominator)
            - (response - mu) * sigma / denominator
        )
        return {
            "mu": (response - mu) / (mu * denominator),
            "sigma": sigma_score,
        }

    def expected_second_derivatives(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[tuple[str, str], Tensor]:
        mu, sigma, response = self._broadcast(response, parameters)
        sigma_score = self.score(response, {"mu": mu, "sigma": sigma})["sigma"]
        return {
            ("mu", "mu"): -1.0 / (mu * (1.0 + mu * sigma)),
            ("sigma", "sigma"): torch.minimum(
                -sigma_score.square(),
                response.new_full(response.shape, -1e-15),
            ),
            ("mu", "sigma"): response.new_zeros(response.shape),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::NBI``."""
        mean = response.mean()
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + mean) / 2.0
        if "sigma" in parameters:
            dispersion = torch.clamp(
                (response.var(correction=1) - mean) / mean.square(),
                min=0.1,
            )
            defaults["sigma"] = dispersion.expand_as(response).clone()
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
            or (response < 0).any()
            or (response != torch.floor(response)).any()
        ):
            raise ValueError(
                f"NBI {context} requires finite non-negative integer counts"
            )

    def _broadcast(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        sigma = distribution.total_count.reciprocal()
        mu = distribution.mean
        return torch.broadcast_tensors(mu, sigma, response)
