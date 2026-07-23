"""Box-Cox power-exponential GAMLSS family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from torchgamlss.families.base import Family
from torchgamlss.families.bccg import _box_cox_z, _exprel_derivative
from torchgamlss.links import IdentityLink, Link, LogLink

_MAX_GAMMA_ITERATIONS = 300
_SMALL_NU = 1e-7
_TAU_DIFFERENCE = 0.001
_LOG_TWO = math.log(2.0)


def _nonzero(value: Tensor, tiny: float) -> Tensor:
    replacement = torch.where(
        value < 0,
        value.new_full((), -tiny),
        value.new_full((), tiny),
    )
    return torch.where(value.abs() < tiny, replacement, value)


def _gamma_series(shape: Tensor, argument: Tensor) -> Tensor:
    """Evaluate regularized lower gamma by its convergent power series."""
    finfo = torch.finfo(argument.dtype)
    tolerance = 32.0 * finfo.eps
    denominator = shape
    term = shape.reciprocal()
    series = term
    for _ in range(_MAX_GAMMA_ITERATIONS):
        denominator = denominator + 1.0
        term = term * argument / denominator
        series = series + term
        if bool(torch.all(term.abs() <= series.abs() * tolerance).detach()):
            break
    else:
        raise RuntimeError("incomplete-gamma series did not converge")
    return series * torch.exp(
        -argument + shape * torch.log(argument) - torch.lgamma(shape)
    )


def _gamma_complement_fraction(shape: Tensor, argument: Tensor) -> Tensor:
    """Evaluate regularized upper gamma by a continued fraction."""
    finfo = torch.finfo(argument.dtype)
    tolerance = 32.0 * finfo.eps
    tiny = 1024.0 * finfo.tiny
    offset = argument + 1.0 - shape
    c = argument.new_full(argument.shape, 1.0 / tiny)
    d = _nonzero(offset, tiny).reciprocal()
    fraction = d
    for iteration in range(1, _MAX_GAMMA_ITERATIONS + 1):
        coefficient = -iteration * (iteration - shape)
        offset = offset + 2.0
        d = _nonzero(coefficient * d + offset, tiny).reciprocal()
        c = _nonzero(offset + coefficient / c, tiny)
        change = d * c
        fraction = fraction * change
        if bool(torch.all((change - 1.0).abs() <= tolerance).detach()):
            break
    else:
        raise RuntimeError("incomplete-gamma continued fraction did not converge")
    return (
        torch.exp(-argument + shape * torch.log(argument) - torch.lgamma(shape))
        * fraction
    )


def _regularized_gamma_p(shape: Tensor, argument: Tensor) -> Tensor:
    """Return the regularized lower gamma with shape differentiation."""
    shape, argument = torch.broadcast_tensors(shape, argument)
    finfo = torch.finfo(argument.dtype)
    finite_argument = torch.where(
        torch.isposinf(argument),
        shape + 1000.0,
        argument,
    )
    safe_argument = finite_argument.clamp_min(finfo.tiny)
    use_series = safe_argument < shape + 1.0
    series_argument = torch.where(
        use_series,
        safe_argument,
        0.5 * (shape + 1.0),
    )
    fraction_argument = torch.where(
        use_series,
        shape + 2.0,
        safe_argument,
    )
    series = _gamma_series(shape, series_argument)
    complement = _gamma_complement_fraction(shape, fraction_argument)
    result = torch.where(use_series, series, 1.0 - complement)
    result = torch.where(argument <= 0, torch.zeros_like(result), result)
    result = torch.where(
        torch.isposinf(argument),
        torch.ones_like(result),
        result,
    )
    return result.clamp(0.0, 1.0)


def _log_scale(tau: Tensor) -> Tensor:
    return 0.5 * (
        -(2.0 / tau) * _LOG_TWO + torch.lgamma(1.0 / tau) - torch.lgamma(3.0 / tau)
    )


def _power_exponential_log_prob(value: Tensor, tau: Tensor) -> Tensor:
    log_scale = _log_scale(tau)
    scaled_absolute = value.abs() * torch.exp(-log_scale)
    return (
        torch.log(tau)
        - log_scale
        - 0.5 * scaled_absolute.pow(tau)
        - (1.0 + 1.0 / tau) * _LOG_TWO
        - torch.lgamma(1.0 / tau)
    )


def _power_exponential_cdf(value: Tensor, tau: Tensor) -> Tensor:
    value, tau = torch.broadcast_tensors(value, tau)
    log_scale = _log_scale(tau)
    gamma_argument = 0.5 * (value.abs() * torch.exp(-log_scale)).pow(tau)
    gamma_probability = _regularized_gamma_p(
        1.0 / tau,
        gamma_argument,
    )
    return 0.5 * (1.0 + gamma_probability * value.sign())


def _truncation_terms(
    sigma: Tensor,
    nu: Tensor,
    tau: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return log normalizer and inverse-Mills ratio for BCPE truncation."""
    near_zero = nu.abs() < _SMALL_NU
    safe_absolute_nu = torch.where(near_zero, torch.ones_like(nu), nu.abs())
    boundary = 1.0 / (sigma * safe_absolute_nu)
    normalizer = _power_exponential_cdf(boundary, tau)
    log_normalizer = torch.log(normalizer)
    mills_ratio = torch.exp(_power_exponential_log_prob(boundary, tau) - log_normalizer)
    zeros = torch.zeros_like(nu)
    return (
        torch.where(near_zero, zeros, log_normalizer),
        torch.where(near_zero, zeros, mills_ratio),
    )


class BoxCoxPowerExponentialDistribution(Distribution):
    """Torch distribution for the BCPE parameterization used by GAMLSS."""

    arg_constraints = {
        "mu": constraints.positive,
        "sigma": constraints.positive,
        "nu": constraints.real,
        "tau": constraints.positive,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(
        self,
        mu: Tensor,
        sigma: Tensor,
        nu: Tensor,
        tau: Tensor,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.mu, self.sigma, self.nu, self.tau = broadcast_all(
            mu,
            sigma,
            nu,
            tau,
        )
        super().__init__(
            batch_shape=self.mu.size(),
            validate_args=validate_args,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, tau, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            self.tau,
            value,
        )
        z, _ = _box_cox_z(value, mu, sigma, nu)
        log_normalizer, _ = _truncation_terms(sigma, nu, tau)
        return (
            (nu - 1.0) * torch.log(value)
            - nu * torch.log(mu)
            - torch.log(sigma)
            + _power_exponential_log_prob(z, tau)
            - log_normalizer
        )

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, tau, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            self.tau,
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
        denominator = _power_exponential_cdf(boundary, tau)
        unadjusted = _power_exponential_cdf(z, tau)
        lower_truncation = _power_exponential_cdf(-boundary, tau)
        probabilities = torch.where(
            nu > 0,
            (unadjusted - lower_truncation) / denominator,
            unadjusted / denominator,
        )
        return probabilities.clamp(0.0, 1.0)


class BoxCoxPowerExponential(Family):
    """Box-Cox power-exponential family matching ``gamlss.dist::BCPE``."""

    name = "BCPE"
    parameter_names = ("mu", "sigma", "nu", "tau")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
        tau_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or IdentityLink(),
            "sigma": sigma_link or LogLink(),
            "nu": nu_link or IdentityLink(),
            "tau": tau_link or LogLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> BoxCoxPowerExponentialDistribution:
        return BoxCoxPowerExponentialDistribution(
            parameters["mu"],
            parameters["sigma"],
            parameters["nu"],
            parameters["tau"],
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
        """Match the parameter-scale scores supplied by R's ``BCPE``."""
        self.validate_response(response)
        mu, sigma, nu, tau, response = self._broadcast(response, parameters)
        z, log_ratio = _box_cox_z(response, mu, sigma, nu)
        log_scale = _log_scale(tau)
        scale = torch.exp(log_scale)
        scaled_absolute = z.abs() / scale
        power = scaled_absolute.pow(tau)
        z_density_derivative = (
            -(tau / (2.0 * scale)) * z.sign() * scaled_absolute.pow(tau - 1.0)
        )
        _, mills_ratio = _truncation_terms(sigma, nu, tau)
        absolute_nu = nu.abs()
        near_zero = absolute_nu < _SMALL_NU
        safe_absolute_nu = torch.where(
            near_zero,
            torch.ones_like(nu),
            absolute_nu,
        )
        transformed = nu * log_ratio
        z_nu_derivative = log_ratio.square() * _exprel_derivative(transformed) / sigma
        sigma_correction = mills_ratio / (sigma.square() * safe_absolute_nu)
        nu_correction = nu.sign() * mills_ratio / (sigma * safe_absolute_nu.square())
        zeros = torch.zeros_like(nu)
        sigma_correction = torch.where(near_zero, zeros, sigma_correction)
        nu_correction = torch.where(near_zero, zeros, nu_correction)

        boundary = 1.0 / (sigma * safe_absolute_nu)
        current_log_cdf = torch.log(_power_exponential_cdf(boundary, tau))
        incremented_log_cdf = torch.log(
            _power_exponential_cdf(boundary, tau + _TAU_DIFFERENCE)
        )
        cdf_tau_derivative = (incremented_log_cdf - current_log_cdf) / _TAU_DIFFERENCE
        cdf_tau_derivative = torch.where(
            near_zero,
            zeros,
            cdf_tau_derivative,
        )
        dlog_scale = (
            2.0 * _LOG_TWO - torch.digamma(1.0 / tau) + 3.0 * torch.digamma(3.0 / tau)
        ) / (2.0 * tau.square())
        log_scaled_absolute = torch.log(
            scaled_absolute.clamp_min(torch.finfo(z.dtype).tiny)
        )
        tau_score = (
            1.0 / tau
            - 0.5 * log_scaled_absolute * power
            + (_LOG_TWO + torch.digamma(1.0 / tau)) / tau.square()
            + ((tau / 2.0) * power - 1.0) * dlog_scale
            - cdf_tau_derivative
        )
        return {
            "mu": z_density_derivative * (-(1.0 + sigma * nu * z) / (mu * sigma))
            - nu / mu,
            "sigma": (tau * power / 2.0 - 1.0) / sigma + sigma_correction,
            "nu": log_ratio + z_density_derivative * z_nu_derivative + nu_correction,
            "tau": tau_score,
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        """Match the Fisher-scoring derivatives supplied by R's ``BCPE``."""
        mu, sigma, nu, tau, response = self._broadcast(response, parameters)
        score_mu = self.score(
            response,
            {"mu": mu, "sigma": sigma, "nu": nu, "tau": tau},
        )["mu"]
        safe_tau = torch.where(
            tau < 1.05,
            torch.full_like(tau, 1.05),
            tau,
        )
        mu_information = torch.exp(
            2.0 * torch.log(safe_tau)
            + torch.lgamma(2.0 - 1.0 / safe_tau)
            + torch.lgamma(3.0 / safe_tau)
            - 2.0 * torch.lgamma(1.0 / safe_tau)
        ) / (mu.square() * sigma.square())
        mu_second = -mu_information - safe_tau * nu.square() / mu.square()
        mu_second = torch.where(tau < 1.05, -score_mu.square(), mu_second)

        ratio = (tau + 1.0) / tau
        dlog_scale = (
            2.0 * _LOG_TWO - torch.digamma(1.0 / tau) + 3.0 * torch.digamma(3.0 / tau)
        ) / (2.0 * tau.square())
        part1 = (
            ratio * torch.special.polygamma(1, ratio)
            + 2.0 * torch.digamma(ratio).square()
        )
        part2 = torch.digamma(ratio) * (
            _LOG_TWO + 3.0 - 3.0 * torch.digamma(3.0 / tau) - tau
        )
        part3 = -3.0 * torch.digamma(3.0 / tau) * (1.0 + _LOG_TWO)
        part4 = -(tau + _LOG_TWO) * _LOG_TWO
        part5 = -tau + tau.pow(4) * dlog_scale.square()
        tau_second = -(part1 + part2 + part3 + part4 + part5) / tau.pow(3)
        tau_second = torch.minimum(
            tau_second,
            response.new_full(response.shape, -1e-15),
        )
        digamma_difference = torch.digamma(1.0 / tau) - torch.digamma(3.0 / tau)
        return {
            ("mu", "mu"): mu_second,
            ("sigma", "sigma"): -tau / sigma.square(),
            ("nu", "nu"): -sigma.square() * (3.0 * tau + 1.0) / 4.0,
            ("tau", "tau"): tau_second,
            ("mu", "sigma"): -nu * tau / (mu * sigma),
            ("mu", "nu"): (
                2.0 * (tau - 1.0) - (tau + 1.0) * sigma.square() * nu.square()
            )
            / (4.0 * mu),
            ("mu", "tau"): (nu / (mu * tau)) * (1.0 + tau + 1.5 * digamma_difference),
            ("sigma", "nu"): -sigma * nu * tau / 2.0,
            ("sigma", "tau"): (1.0 + tau + 1.5 * digamma_difference) / (sigma * tau),
            ("nu", "tau"): (
                sigma.square()
                * nu
                / (2.0 * tau)
                * (1.0 + tau / 3.0 + 0.5 * digamma_difference)
            ),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::BCPE``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.1)
        if "nu" in parameters:
            defaults["nu"] = torch.ones_like(response)
        if "tau" in parameters:
            defaults["tau"] = torch.full_like(response, 2.0)
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
                f"BCPE {context} requires a finite strictly positive response"
            )

    def _broadcast(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        distribution = self.distribution(parameters)
        return torch.broadcast_tensors(
            distribution.mu,
            distribution.sigma,
            distribution.nu,
            distribution.tau,
            response,
        )


BCPE = BoxCoxPowerExponential
