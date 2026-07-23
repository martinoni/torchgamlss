"""Poisson GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.stats import poisson as scipy_poisson
from torch import Tensor
from torch.distributions import Poisson as TorchPoisson

from torchgamlss.families._scipy import scipy_cdf
from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogLink


class Poisson(Family):
    """Poisson family compatible with ``gamlss.dist::PO(mu)``."""

    name = "PO"
    parameter_names = ("mu",)
    is_discrete = True

    def __init__(self, *, mu_link: Link | None = None) -> None:
        self._links = {"mu": mu_link or LogLink()}

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchPoisson:
        return TorchPoisson(rate=parameters["mu"], validate_args=True)

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        mu, response = torch.broadcast_tensors(parameters["mu"], response)
        return scipy_cdf(
            response,
            scipy_poisson.cdf,
            response,
            mu,
        )

    def score(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, response = torch.broadcast_tensors(parameters["mu"], response)
        return {"mu": (response - mu) / mu}

    def expected_second_derivatives(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[tuple[str, str], Tensor]:
        self.validate_response(response)
        mu, _ = torch.broadcast_tensors(parameters["mu"], response)
        return {("mu", "mu"): -mu.reciprocal()}

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expression in ``gamlss.dist::PO``."""
        return {"mu": (response + response.mean()) / 2.0} if "mu" in parameters else {}

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
                f"PO {context} requires finite non-negative integer counts"
            )
