"""Inverse-Gaussian GAMLSS family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from scipy.stats import invgauss as scipy_inverse_gaussian
from torch import Size, Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from torchgamlss.families._scipy import scipy_call
from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogLink

_LOG_TWO_PI = math.log(2.0 * math.pi)


class InverseGaussianDistribution(Distribution):
    """Inverse Gaussian with mean ``mu`` and shape ``1 / sigma**2``."""

    arg_constraints = {
        "mu": constraints.positive,
        "sigma": constraints.positive,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.sigma = broadcast_all(mu, sigma)
        super().__init__(
            batch_shape=self.mu.shape,
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def variance(self) -> Tensor:
        return self.sigma.square() * self.mu.pow(3)

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, value = torch.broadcast_tensors(self.mu, self.sigma, value)
        return (
            -0.5 * _LOG_TWO_PI
            - torch.log(sigma)
            - 1.5 * torch.log(value)
            - (value - mu).square() / (2.0 * sigma.square() * mu.square() * value)
        )

    def _standardized_terms(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, sigma, value = torch.broadcast_tensors(self.mu, self.sigma, value)
        denominator = sigma * torch.sqrt(value)
        first = (value / mu - 1.0) / denominator
        second = -(value / mu + 1.0) / denominator
        exponent = 2.0 / (mu * sigma.square())
        return first, second, exponent

    def log_cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        first, second, exponent = self._standardized_terms(value)
        result = torch.logaddexp(
            torch.special.log_ndtr(first),
            exponent + torch.special.log_ndtr(second),
        )
        return result.clamp_max(0.0)

    def log_survival(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        first, second, exponent = self._standardized_terms(value)
        log_larger = torch.special.log_ndtr(-first)
        log_smaller = exponent + torch.special.log_ndtr(second)
        epsilon = torch.finfo(value.dtype).eps
        log_ratio = (log_smaller - log_larger).clamp_max(-epsilon)
        return log_larger + torch.log1p(-torch.exp(log_ratio))

    def cdf(self, value: Tensor) -> Tensor:
        return torch.exp(self.log_cdf(value))

    @torch.no_grad()
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        mu = self.mu.expand(shape)
        sigma = self.sigma.expand(shape)
        normal_square = torch.randn(
            shape,
            dtype=mu.dtype,
            device=mu.device,
        ).square()
        inverse_shape = sigma.square()
        scaled = 0.5 * mu * inverse_shape * normal_square
        candidate = mu / (1.0 + scaled + torch.sqrt(scaled * (scaled + 2.0)))
        uniform = torch.rand(
            shape,
            dtype=mu.dtype,
            device=mu.device,
        )
        reciprocal_candidate = mu.square() / candidate
        return torch.where(
            uniform <= mu / (mu + candidate),
            candidate,
            reciprocal_candidate,
        )


class InverseGaussian(Family):
    """Inverse-Gaussian family compatible with ``gamlss.dist::IG``."""

    name = "IG"
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

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> InverseGaussianDistribution:
        return InverseGaussianDistribution(
            parameters["mu"],
            parameters["sigma"],
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
        inverse_shape = sigma.square().reciprocal()
        return scipy_call(
            probabilities,
            scipy_inverse_gaussian.ppf,
            probabilities,
            mu / inverse_shape,
            torch.zeros_like(probabilities),
            inverse_shape,
        )

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        return {
            "mu": (response - mu) / (sigma.square() * mu.pow(3)),
            "sigma": -sigma.reciprocal()
            + (response - mu).square() / (response * sigma.pow(3) * mu.square()),
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        self.validate_response(response)
        mu, sigma, response = self._broadcast(response, parameters)
        return {
            ("mu", "mu"): -1.0 / (mu.pow(3) * sigma.square()),
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
            raise ValueError(f"IG {context} requires strictly positive responses")

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            sigma = response.std(correction=1) / response.mean().pow(1.5)
            defaults["sigma"] = sigma.expand_as(response)
        return defaults

    def _broadcast(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        return torch.broadcast_tensors(
            distribution.mu,
            distribution.sigma,
            response,
        )


IG = InverseGaussian
