"""Generalized-gamma GAMLSS family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from scipy.stats import gamma as scipy_gamma
from torch import Size, Tensor
from torch.distributions import Distribution, Gamma, LogNormal, constraints
from torch.distributions.utils import broadcast_all

from torchgamlss.families._scipy import scipy_call
from torchgamlss.families.base import Family
from torchgamlss.families.bcpe import (
    _regularized_gamma_p,
    _regularized_gamma_q,
)
from torchgamlss.links import IdentityLink, Link, LogLink

_LOG_TWO_PI = math.log(2.0 * math.pi)


def _near_log_normal(nu: Tensor) -> Tensor:
    return nu.abs() <= 5e-2


def _log_moment_limit_series(
    mu: Tensor,
    sigma: Tensor,
    nu: Tensor,
    order: float,
) -> Tensor:
    order_tensor = torch.as_tensor(order, dtype=mu.dtype, device=mu.device)
    sigma_squared = sigma.square()
    return (
        order_tensor * torch.log(mu)
        + order_tensor.square() * sigma_squared / 2.0
        + nu
        * (
            -order_tensor.pow(3) * sigma_squared.square() / 6.0
            - order_tensor * sigma_squared / 2.0
        )
        + nu.square()
        * (
            order_tensor.pow(4) * sigma_squared.pow(3) / 12.0
            + order_tensor.square() * sigma_squared.square() / 4.0
        )
        + nu.pow(3)
        * (
            -order_tensor.pow(5) * sigma_squared.pow(4) / 20.0
            - order_tensor.pow(3) * sigma_squared.pow(3) / 6.0
            - order_tensor * sigma_squared.square() / 12.0
        )
        + nu.pow(4)
        * (
            order_tensor.pow(6) * sigma_squared.pow(5) / 30.0
            + order_tensor.pow(4) * sigma_squared.pow(4) / 8.0
            + order_tensor.square() * sigma_squared.pow(3) / 12.0
        )
    )


class GeneralizedGammaDistribution(Distribution):
    """Lopatatzidis-Green generalized gamma used by ``gamlss.dist::GG``."""

    arg_constraints = {
        "mu": constraints.positive,
        "sigma": constraints.positive,
        "nu": constraints.real,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        nu: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.sigma, self.nu = broadcast_all(mu, sigma, nu)
        super().__init__(
            batch_shape=self.mu.shape,
            validate_args=validate_args,
        )

    def _components(
        self,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        near_zero = _near_log_normal(nu)
        safe_nu = torch.where(near_zero, torch.ones_like(nu), nu)
        log_ratio = torch.log(value / mu)
        log_z = safe_nu * log_ratio
        theta = 1.0 / (sigma.square() * safe_nu.square())
        gamma_argument = theta * torch.exp(log_z)
        return near_zero, safe_nu, log_ratio, log_z, theta, gamma_argument

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        near_zero, safe_nu, log_ratio, log_z, theta, gamma_argument = self._components(
            value
        )
        _, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        general = (
            theta * torch.log(theta)
            - torch.lgamma(theta)
            + theta * log_z
            - gamma_argument
            + torch.log(safe_nu.abs())
            - torch.log(value)
        )
        inverse_variance = sigma.square().reciprocal()
        log_normal = (
            -torch.log(value)
            - 0.5 * _LOG_TWO_PI
            - torch.log(sigma)
            - 0.5 * log_ratio.square() * inverse_variance
        )
        limit_series = (
            log_normal
            - nu * log_ratio.pow(3) * inverse_variance / 6.0
            - nu.square()
            * (log_ratio.pow(4) * inverse_variance / 24.0 + sigma.square() / 12.0)
            - nu.pow(3) * log_ratio.pow(5) * inverse_variance / 120.0
        )
        return torch.where(near_zero, limit_series, general)

    def _probabilities(self, value: Tensor) -> tuple[Tensor, Tensor]:
        near_zero, _, log_ratio, _, theta, gamma_argument = self._components(value)
        _, sigma, nu, _ = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        gamma_lower = _regularized_gamma_p(theta, gamma_argument)
        gamma_upper = _regularized_gamma_q(theta, gamma_argument)
        general_cdf = torch.where(nu > 0, gamma_lower, gamma_upper)
        general_survival = torch.where(nu > 0, gamma_upper, gamma_lower)

        standardized = log_ratio / sigma
        density = torch.exp(-0.5 * standardized.square()) / math.sqrt(2.0 * math.pi)
        first_correction = nu * sigma * (standardized.square() + 2.0) * density / 6.0
        second_correction = (
            -nu.square()
            * sigma.square()
            * (standardized.pow(5) + 2.0 * standardized.pow(3) + 6.0 * standardized)
            * density
            / 72.0
        )
        correction = first_correction + second_correction
        normal_cdf = torch.special.ndtr(standardized)
        normal_survival = torch.special.ndtr(-standardized)
        limit_cdf = normal_cdf + correction
        limit_survival = normal_survival - correction
        finfo = torch.finfo(value.dtype)
        return (
            torch.where(near_zero, limit_cdf, general_cdf).clamp(
                min=finfo.tiny,
                max=1.0,
            ),
            torch.where(near_zero, limit_survival, general_survival).clamp(
                min=finfo.tiny,
                max=1.0,
            ),
        )

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        cdf, _ = self._probabilities(value)
        return cdf

    def log_cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        cdf, _ = self._probabilities(value)
        return torch.log(cdf)

    def log_survival(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        _, survival = self._probabilities(value)
        return torch.log(survival)

    @property
    def mean(self) -> Tensor:
        near_zero = _near_log_normal(self.nu)
        safe_nu = torch.where(near_zero, torch.ones_like(self.nu), self.nu)
        theta = 1.0 / (self.sigma.square() * safe_nu.square())
        moment_shape = theta + safe_nu.reciprocal()
        valid = (safe_nu > 0) | (
            (safe_nu < 0) & (self.sigma.square() * safe_nu.abs() < 1.0)
        )
        safe_shape = torch.where(valid, moment_shape, torch.ones_like(moment_shape))
        log_mean = (
            torch.log(self.mu)
            + torch.lgamma(safe_shape)
            - torch.lgamma(theta)
            - torch.log(theta) / safe_nu
        )
        general = torch.where(
            valid,
            torch.exp(log_mean),
            torch.full_like(log_mean, torch.inf),
        )
        limit = torch.exp(
            _log_moment_limit_series(
                self.mu,
                self.sigma,
                self.nu,
                1.0,
            )
        )
        return torch.where(near_zero, limit, general)

    @property
    def variance(self) -> Tensor:
        near_zero = _near_log_normal(self.nu)
        safe_nu = torch.where(near_zero, torch.ones_like(self.nu), self.nu)
        theta = 1.0 / (self.sigma.square() * safe_nu.square())
        first_shape = theta + safe_nu.reciprocal()
        second_shape = theta + 2.0 / safe_nu
        valid = (safe_nu > 0) | (
            (safe_nu < 0) & (self.sigma.square() * safe_nu.abs() < 0.5)
        )
        safe_first = torch.where(valid, first_shape, torch.ones_like(first_shape))
        safe_second = torch.where(valid, second_shape, torch.ones_like(second_shape))
        log_first = (
            torch.log(self.mu)
            + torch.lgamma(safe_first)
            - torch.lgamma(theta)
            - torch.log(theta) / safe_nu
        )
        log_second = (
            2.0 * torch.log(self.mu)
            + torch.lgamma(safe_second)
            - torch.lgamma(theta)
            - 2.0 * torch.log(theta) / safe_nu
        )
        general_variance = torch.exp(2.0 * log_first) * torch.expm1(
            log_second - 2.0 * log_first
        ).clamp_min(0.0)
        general = torch.where(
            valid,
            general_variance,
            torch.full_like(general_variance, torch.inf),
        )
        limit_log_first = _log_moment_limit_series(
            self.mu,
            self.sigma,
            self.nu,
            1.0,
        )
        limit_log_second = _log_moment_limit_series(
            self.mu,
            self.sigma,
            self.nu,
            2.0,
        )
        limit = torch.exp(2.0 * limit_log_first) * torch.expm1(
            limit_log_second - 2.0 * limit_log_first
        )
        return torch.where(near_zero, limit, general)

    @torch.no_grad()
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        mu = self.mu.expand(shape)
        sigma = self.sigma.expand(shape)
        nu = self.nu.expand(shape)
        sampling_limit = math.sqrt(torch.finfo(nu.dtype).eps)
        near_zero = nu.abs() <= sampling_limit
        safe_nu = torch.where(near_zero, torch.ones_like(nu), nu)
        theta = 1.0 / (sigma.square() * safe_nu.square())
        gamma_sample = Gamma(theta, theta).sample()
        general = mu * torch.exp(torch.log(gamma_sample) / safe_nu)
        log_normal = LogNormal(torch.log(mu), sigma).sample()
        return torch.where(near_zero, log_normal, general)


class GeneralizedGamma(Family):
    """Generalized-gamma family compatible with ``gamlss.dist::GG``."""

    name = "GG"
    parameter_names = ("mu", "sigma", "nu")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or LogLink(),
            "sigma": sigma_link or LogLink(),
            "nu": nu_link or IdentityLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> GeneralizedGammaDistribution:
        return GeneralizedGammaDistribution(
            parameters["mu"],
            parameters["sigma"],
            parameters["nu"],
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return self.distribution(parameters).cdf(response)

    def _differentiable_cdf(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        return self.cdf(response, parameters)

    def _log_cdf(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        return self.distribution(parameters).log_cdf(response)

    def _log_survival(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        return self.distribution(parameters).log_survival(response)

    def _quantile(
        self,
        probabilities: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        nu = parameters["nu"]
        near_zero = _near_log_normal(nu)
        safe_nu = torch.where(near_zero, torch.ones_like(nu), nu)
        theta = 1.0 / (sigma.square() * safe_nu.square())
        gamma_probabilities = torch.where(
            safe_nu > 0,
            probabilities,
            1.0 - probabilities,
        )
        gamma_quantile = scipy_call(
            probabilities,
            scipy_gamma.ppf,
            gamma_probabilities,
            theta,
            torch.zeros_like(probabilities),
            theta.reciprocal(),
        )
        general = mu * torch.exp(torch.log(gamma_quantile) / safe_nu)
        standardized = torch.special.ndtri(probabilities)
        for _ in range(4):
            density = torch.exp(-0.5 * standardized.square()) / math.sqrt(2.0 * math.pi)
            first_correction = (
                nu * sigma * (standardized.square() + 2.0) * density / 6.0
            )
            second_correction = (
                -nu.square()
                * sigma.square()
                * (standardized.pow(5) + 2.0 * standardized.pow(3) + 6.0 * standardized)
                * density
                / 72.0
            )
            approximate_cdf = (
                torch.special.ndtr(standardized) + first_correction + second_correction
            )
            derivative = density * (
                1.0
                - nu * sigma * standardized.pow(3) / 6.0
                + nu.square()
                * sigma.square()
                * (standardized.pow(6) - 3.0 * standardized.pow(4) - 6.0)
                / 72.0
            )
            standardized = standardized - (approximate_cdf - probabilities) / derivative
        log_normal = mu * torch.exp(sigma * standardized)
        return torch.where(near_zero, log_normal, general)

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, nu, response = self._broadcast(response, parameters)
        near_zero = _near_log_normal(nu)
        safe_nu = torch.where(near_zero, torch.ones_like(nu), nu)
        log_ratio = torch.log(response / mu)
        z = torch.exp(safe_nu * log_ratio)
        theta = 1.0 / (sigma.square() * safe_nu.square())
        log_theta = torch.log(theta)
        general_mu = (z - 1.0) * theta * safe_nu / mu
        general_sigma = (
            -2.0
            * theta
            * (log_theta + 1.0 + torch.log(z) - z - torch.digamma(theta))
            / sigma
        )
        general_nu = (
            1.0
            + 2.0
            * theta
            * (
                torch.digamma(theta)
                + z
                - log_theta
                - 1.0
                - 0.5 * (z + 1.0) * torch.log(z)
            )
        ) / safe_nu
        return {
            "mu": torch.where(
                near_zero,
                (
                    log_ratio
                    + nu * log_ratio.square() / 2.0
                    + nu.square() * log_ratio.pow(3) / 6.0
                    + nu.pow(3) * log_ratio.pow(4) / 24.0
                )
                / (mu * sigma.square()),
                general_mu,
            ),
            "sigma": torch.where(
                near_zero,
                -sigma.reciprocal()
                + log_ratio.square() / sigma.pow(3)
                + nu * log_ratio.pow(3) / (3.0 * sigma.pow(3))
                + nu.square() * (log_ratio.pow(4) / (12.0 * sigma.pow(3)) - sigma / 6.0)
                + nu.pow(3) * log_ratio.pow(5) / (60.0 * sigma.pow(3)),
                general_sigma,
            ),
            "nu": torch.where(
                near_zero,
                -log_ratio.pow(3) / (6.0 * sigma.square())
                - nu
                * (log_ratio.pow(4) / (12.0 * sigma.square()) + sigma.square() / 6.0)
                - nu.square() * log_ratio.pow(5) / (40.0 * sigma.square()),
                general_nu,
            ),
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        self.validate_response(response)
        mu, sigma, nu, response = self._broadcast(response, parameters)
        near_zero = _near_log_normal(nu)
        safe_nu = torch.where(near_zero, torch.ones_like(nu), nu)
        theta = 1.0 / (sigma.square() * safe_nu.square())
        trigamma = torch.special.polygamma(1, theta)
        log_theta = torch.log(theta)
        digamma = torch.digamma(theta)
        general_sigma_second = 4.0 * theta / sigma.square() * (1.0 - theta * trigamma)
        general_nu_second = -(theta / safe_nu.square()) * (
            trigamma * (1.0 + 4.0 * theta)
            - (4.0 + 3.0 / theta)
            - log_theta * (2.0 / theta - log_theta)
            + digamma * (digamma + 2.0 / theta - 2.0 * log_theta)
        )
        general_mu_nu = theta / mu * (digamma + theta.reciprocal() - log_theta)
        general_sigma_nu = (
            -2.0
            * safe_nu.sign()
            * theta.pow(1.5)
            * (2.0 * theta * trigamma - theta.reciprocal() - 2.0)
        )
        zeros = torch.zeros_like(response)
        return {
            ("mu", "mu"): -1.0 / (mu.square() * sigma.square()),
            ("sigma", "sigma"): torch.where(
                near_zero,
                -2.0 / sigma.square() - 2.0 * nu.square() / 3.0,
                general_sigma_second,
            ),
            ("nu", "nu"): torch.where(
                near_zero,
                -5.0 * sigma.square() / 12.0 - sigma.pow(4) * nu.square() / 12.0,
                general_nu_second,
            ),
            ("mu", "sigma"): zeros,
            ("mu", "nu"): torch.where(
                near_zero,
                0.5 / mu - sigma.square() * nu.square() / (12.0 * mu),
                general_mu_nu,
            ),
            ("sigma", "nu"): torch.where(
                near_zero,
                -2.0 * sigma * nu / 3.0,
                general_sigma_nu,
            ),
        }

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        super().validate_response(response, context=context)
        if bool((response <= 0).any()):
            raise ValueError(f"GG {context} requires strictly positive responses")

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.ones_like(response)
        if "nu" in parameters:
            defaults["nu"] = torch.ones_like(response)
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


GG = GeneralizedGamma
