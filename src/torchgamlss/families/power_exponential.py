"""Power-exponential location-scale-shape GAMLSS family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from torchgamlss.families.base import Family
from torchgamlss.families.bcpe import (
    _log_scale,
    _power_exponential_cdf,
    _power_exponential_icdf,
    _power_exponential_log_prob,
    _sample_power_exponential,
)
from torchgamlss.links import IdentityLink, Link, LogLink

_LOG_TWO = math.log(2.0)


class PowerExponentialDistribution(Distribution):
    """Standard-deviation parameterized power-exponential distribution."""

    arg_constraints = {
        "mu": constraints.real,
        "sigma": constraints.positive,
        "nu": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        nu: Tensor,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.sigma, self.nu = broadcast_all(mu, sigma, nu)
        super().__init__(
            batch_shape=self.mu.size(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def variance(self) -> Tensor:
        return self.sigma.square()

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        standardized = (value - mu) / sigma
        return _power_exponential_log_prob(standardized, nu) - torch.log(sigma)

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        mu = self.mu.expand(shape)
        sigma = self.sigma.expand(shape)
        nu = self.nu.expand(shape)
        return mu + sigma * _sample_power_exponential(nu)

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        return _power_exponential_cdf((value - mu) / sigma, nu)

    def icdf(self, probability: Tensor) -> Tensor:
        mu, sigma, nu, probability = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            probability,
        )
        return mu + sigma * _power_exponential_icdf(probability, nu)


class PowerExponential(Family):
    """Power-exponential family compatible with ``gamlss.dist::PE``."""

    name = "PE"
    parameter_names = ("mu", "sigma", "nu")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or IdentityLink(),
            "sigma": sigma_link or LogLink(),
            "nu": nu_link or LogLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> PowerExponentialDistribution:
        return PowerExponentialDistribution(
            parameters["mu"],
            parameters["sigma"],
            parameters["nu"],
            validate_args=True,
        )

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Match the parameter-scale scores supplied by R's ``PE``."""
        self.validate_response(response)
        mu, sigma, nu, response = self._broadcast(response, parameters)
        standardized = (response - mu) / sigma
        scale = torch.exp(_log_scale(nu))
        scaled_absolute = standardized.abs() / scale
        power = scaled_absolute.pow(nu)
        safe_absolute = standardized.abs().clamp_min(
            torch.finfo(standardized.dtype).tiny
        )
        mu_score = (
            standardized.sign() * nu * power / (2.0 * sigma * safe_absolute)
        )
        dlog_scale = (
            2.0 * _LOG_TWO
            - torch.digamma(1.0 / nu)
            + 3.0 * torch.digamma(3.0 / nu)
        ) / (2.0 * nu.square())
        log_scaled_absolute = torch.log(
            scaled_absolute.clamp_min(torch.finfo(standardized.dtype).tiny)
        )
        return {
            "mu": mu_score,
            "sigma": ((nu / 2.0) * power - 1.0) / sigma,
            "nu": (
                1.0 / nu
                - 0.5 * log_scaled_absolute * power
                + (_LOG_TWO + torch.digamma(1.0 / nu)) / nu.square()
                + ((nu / 2.0) * power - 1.0) * dlog_scale
            ),
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        """Match the Fisher-scoring derivatives supplied by R's ``PE``."""
        _, sigma, nu, response = self._broadcast(response, parameters)
        mu_score = self.score(response, parameters)["mu"]
        safe_nu = torch.where(
            nu < 1.05,
            torch.full_like(nu, 1.05),
            nu,
        )
        mu_information = torch.exp(
            2.0 * torch.log(safe_nu)
            + torch.lgamma(2.0 - 1.0 / safe_nu)
            + torch.lgamma(3.0 / safe_nu)
            - 2.0 * torch.lgamma(1.0 / safe_nu)
        ) / sigma.square()
        mu_second = torch.where(
            nu < 1.05,
            -mu_score.square(),
            -mu_information,
        )

        ratio = (nu + 1.0) / nu
        dlog_scale = (
            2.0 * _LOG_TWO
            - torch.digamma(1.0 / nu)
            + 3.0 * torch.digamma(3.0 / nu)
        ) / (2.0 * nu.square())
        part1 = (
            ratio * torch.special.polygamma(1, ratio)
            + 2.0 * torch.digamma(ratio).square()
        )
        part2 = torch.digamma(ratio) * (
            _LOG_TWO + 3.0 - 3.0 * torch.digamma(3.0 / nu) - nu
        )
        part3 = -3.0 * torch.digamma(3.0 / nu) * (1.0 + _LOG_TWO)
        part4 = -(nu + _LOG_TWO) * _LOG_TWO
        part5 = -nu + nu.pow(4) * dlog_scale.square()
        nu_second = -(part1 + part2 + part3 + part4 + part5) / nu.pow(3)
        nu_second = torch.minimum(
            nu_second,
            response.new_full(response.shape, -1e-15),
        )
        digamma_difference = (
            torch.digamma(1.0 / nu) - torch.digamma(3.0 / nu)
        )
        zeros = torch.zeros_like(response)
        return {
            ("mu", "mu"): mu_second,
            ("sigma", "sigma"): -nu / sigma.square(),
            ("nu", "nu"): nu_second,
            ("mu", "sigma"): zeros,
            ("mu", "nu"): zeros,
            ("sigma", "nu"): (
                3.0 * digamma_difference / nu + 2.0 + 2.0 / nu
            )
            / (2.0 * sigma),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::PE``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = (
                (response - response.mean()).abs()
                + response.std(correction=1)
            ) / 2.0
        if "nu" in parameters:
            defaults["nu"] = torch.full_like(response, 1.8)
        return defaults

    def _broadcast(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        return torch.broadcast_tensors(
            distribution.mu,
            distribution.sigma,
            distribution.nu,
            response,
        )


PE = PowerExponential
