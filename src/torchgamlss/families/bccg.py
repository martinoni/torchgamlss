"""Box-Cox Cole-Green GAMLSS family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from torchgamlss.families.base import Family
from torchgamlss.links import IdentityLink, Link, LogLink

_SMALL_ARGUMENT = 1e-5
_SMALL_NU = 1e-7
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _exprel(value: Tensor) -> Tensor:
    """Return ``expm1(value) / value`` with a stable zero limit."""
    use_series = value.abs() < _SMALL_ARGUMENT
    safe_value = torch.where(use_series, torch.ones_like(value), value)
    regular = torch.expm1(value) / safe_value
    series = (
        1.0
        + value / 2.0
        + value.square() / 6.0
        + value.pow(3) / 24.0
        + value.pow(4) / 120.0
    )
    return torch.where(use_series, series, regular)


def _exprel_derivative(value: Tensor) -> Tensor:
    """Return the derivative of ``expm1(value) / value`` at and near zero."""
    use_series = value.abs() < _SMALL_ARGUMENT
    safe_value = torch.where(use_series, torch.ones_like(value), value)
    regular = ((value - 1.0) * torch.exp(value) + 1.0) / safe_value.square()
    series = (
        0.5
        + value / 3.0
        + value.square() / 8.0
        + value.pow(3) / 30.0
        + value.pow(4) / 144.0
    )
    return torch.where(use_series, series, regular)


def _box_cox_z(
    response: Tensor,
    mu: Tensor,
    sigma: Tensor,
    nu: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the Box-Cox normal score and ``log(response / mu)``."""
    log_ratio = torch.log(response / mu)
    transformed = nu * log_ratio
    z = log_ratio * _exprel(transformed) / sigma
    return z, log_ratio


def _truncation_terms(sigma: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
    """Return log normalizer and inverse-Mills ratio for BCCG truncation."""
    near_zero = nu.abs() < _SMALL_NU
    safe_absolute_nu = torch.where(near_zero, torch.ones_like(nu), nu.abs())
    boundary = 1.0 / (sigma * safe_absolute_nu)
    log_normalizer = torch.special.log_ndtr(boundary)
    log_mills_ratio = -0.5 * boundary.square() - _LOG_SQRT_TWO_PI - log_normalizer
    mills_ratio = torch.exp(log_mills_ratio)
    zeros = torch.zeros_like(nu)
    return (
        torch.where(near_zero, zeros, log_normalizer),
        torch.where(near_zero, zeros, mills_ratio),
    )


class BoxCoxColeGreenDistribution(Distribution):
    """Torch distribution for the BCCG parameterization used by GAMLSS."""

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
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.sigma, self.nu = broadcast_all(mu, sigma, nu)
        super().__init__(
            batch_shape=self.mu.size(),
            validate_args=validate_args,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        z, log_ratio = _box_cox_z(value, mu, sigma, nu)
        log_normalizer, _ = _truncation_terms(sigma, nu)
        return (
            nu * log_ratio
            - torch.log(sigma)
            - 0.5 * z.square()
            - torch.log(value)
            - _LOG_SQRT_TWO_PI
            - log_normalizer
        )

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        z, _ = _box_cox_z(value, mu, sigma, nu)
        zero_nu = nu == 0
        safe_absolute_nu = torch.where(zero_nu, torch.ones_like(nu), nu.abs())
        boundary = 1.0 / (sigma * safe_absolute_nu)
        boundary = torch.where(
            zero_nu,
            torch.full_like(boundary, float("inf")),
            boundary,
        )
        denominator = torch.special.ndtr(boundary)
        unadjusted = torch.special.ndtr(z)
        lower_truncation = torch.special.ndtr(-boundary)
        probabilities = torch.where(
            nu > 0,
            (unadjusted - lower_truncation) / denominator,
            unadjusted / denominator,
        )
        return probabilities.clamp(0.0, 1.0)


class BoxCoxColeGreen(Family):
    """Box-Cox Cole-Green family compatible with ``gamlss.dist::BCCG``."""

    name = "BCCG"
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
            "nu": nu_link or IdentityLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> BoxCoxColeGreenDistribution:
        return BoxCoxColeGreenDistribution(
            parameters["mu"],
            parameters["sigma"],
            parameters["nu"],
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Match the parameter-scale scores supplied by R's ``BCCG``."""
        self.validate_response(response)
        mu, sigma, nu, response = self._broadcast(response, parameters)
        z, log_ratio = _box_cox_z(response, mu, sigma, nu)
        _, mills_ratio = _truncation_terms(sigma, nu)
        absolute_nu = nu.abs()
        near_zero = absolute_nu < _SMALL_NU
        safe_absolute_nu = torch.where(
            near_zero,
            torch.ones_like(nu),
            absolute_nu,
        )
        transformed = nu * log_ratio
        z_derivative = log_ratio.square() * _exprel_derivative(transformed) / sigma
        sigma_correction = mills_ratio / (sigma.square() * safe_absolute_nu)
        nu_correction = nu.sign() * mills_ratio / (sigma * safe_absolute_nu.square())
        zeros = torch.zeros_like(nu)
        sigma_correction = torch.where(near_zero, zeros, sigma_correction)
        nu_correction = torch.where(near_zero, zeros, nu_correction)
        return {
            "mu": ((z / sigma) + nu * (z.square() - 1.0)) / mu,
            "sigma": (z.square() - 1.0) / sigma + sigma_correction,
            "nu": log_ratio - z * z_derivative + nu_correction,
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        """Match the Fisher-scoring derivatives supplied by R's ``BCCG``."""
        mu, sigma, nu, response = self._broadcast(response, parameters)
        return {
            ("mu", "mu"): -(1.0 + 2.0 * nu.square() * sigma.square())
            / (mu.square() * sigma.square()),
            ("sigma", "sigma"): -2.0 / sigma.square(),
            ("nu", "nu"): -7.0 * sigma.square() / 4.0,
            ("mu", "sigma"): -2.0 * nu / (mu * sigma),
            ("mu", "nu"): 1.0 / (2.0 * mu),
            ("sigma", "nu"): -sigma * nu,
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::BCCG``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.1)
        if "nu" in parameters:
            defaults["nu"] = torch.full_like(response, 0.5)
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
        ):
            raise ValueError(
                f"BCCG {context} requires a finite strictly positive response"
            )

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


BCCG = BoxCoxColeGreen
