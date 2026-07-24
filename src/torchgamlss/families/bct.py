"""Box-Cox t GAMLSS family."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.distributions import Distribution, StudentT, constraints
from torch.distributions.utils import broadcast_all

from torchgamlss.families.base import Family
from torchgamlss.families.bccg import (
    BoxCoxColeGreenDistribution,
    _box_cox_response,
    _box_cox_z,
    _exprel_derivative,
    _valid_box_cox_score,
)
from torchgamlss.links import IdentityLink, Link, LogLink

_MAX_BETA_ITERATIONS = 200
_SMALL_NU = 1e-7
_LARGE_TAU = 1e6
_TAU_DIFFERENCE = 0.01
_LOG_SQRT_PI = 0.5 * math.log(math.pi)


def _nonzero(value: Tensor, tiny: float) -> Tensor:
    replacement = torch.where(
        value < 0,
        value.new_full((), -tiny),
        value.new_full((), tiny),
    )
    return torch.where(value.abs() < tiny, replacement, value)


def _beta_continued_fraction(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Evaluate the incomplete-beta continued fraction with Lentz's method."""
    finfo = torch.finfo(x.dtype)
    tolerance = 32.0 * finfo.eps
    tiny = 1024.0 * finfo.tiny
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = torch.ones_like(x)
    d = _nonzero(1.0 - qab * x / qap, tiny).reciprocal()
    result = d

    for iteration in range(1, _MAX_BETA_ITERATIONS + 1):
        doubled = 2.0 * iteration
        coefficient = (
            iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        )
        d = _nonzero(1.0 + coefficient * d, tiny).reciprocal()
        c = _nonzero(1.0 + coefficient / c, tiny)
        result = result * d * c
        coefficient = (
            -(a + iteration) * (qab + iteration) * x / ((a + doubled) * (qap + doubled))
        )
        d = _nonzero(1.0 + coefficient * d, tiny).reciprocal()
        c = _nonzero(1.0 + coefficient / c, tiny)
        change = d * c
        result = result * change
        if bool(torch.all((change - 1.0).abs() <= tolerance).detach()):
            break
    else:
        raise RuntimeError("incomplete-beta continued fraction did not converge")

    return result


def _regularized_incomplete_beta(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Return the regularized incomplete beta using differentiable Torch ops."""
    x, a, b = torch.broadcast_tensors(x, a, b)
    finfo = torch.finfo(x.dtype)
    safe_x = x.clamp(finfo.tiny, 1.0 - finfo.eps)
    front = torch.exp(
        torch.lgamma(a + b)
        - torch.lgamma(a)
        - torch.lgamma(b)
        + a * torch.log(safe_x)
        + b * torch.log1p(-safe_x)
    )
    use_lower = safe_x < (a + 1.0) / (a + b + 2.0)
    lower_x = torch.where(
        use_lower,
        safe_x,
        0.5 * (a + 1.0) / (a + b + 2.0),
    )
    upper_x = torch.where(
        use_lower,
        0.5 * (b + 1.0) / (a + b + 2.0),
        1.0 - safe_x,
    )
    lower = front * _beta_continued_fraction(a, b, lower_x) / a
    upper = 1.0 - (front * _beta_continued_fraction(b, a, upper_x) / b)
    result = torch.where(use_lower, lower, upper)
    result = torch.where(x <= 0, torch.zeros_like(result), result)
    result = torch.where(x >= 1, torch.ones_like(result), result)
    return result.clamp(0.0, 1.0)


def _student_t_log_prob(value: Tensor, degrees_of_freedom: Tensor) -> Tensor:
    half_df = degrees_of_freedom / 2.0
    return (
        torch.lgamma((degrees_of_freedom + 1.0) / 2.0)
        - torch.lgamma(half_df)
        - 0.5 * torch.log(degrees_of_freedom)
        - _LOG_SQRT_PI
        - ((degrees_of_freedom + 1.0) / 2.0)
        * torch.log1p(value.square() / degrees_of_freedom)
    )


def _student_t_cdf(value: Tensor, degrees_of_freedom: Tensor) -> Tensor:
    value, degrees_of_freedom = torch.broadcast_tensors(
        value,
        degrees_of_freedom,
    )
    beta_argument = degrees_of_freedom / (degrees_of_freedom + value.square())
    beta = _regularized_incomplete_beta(
        beta_argument,
        degrees_of_freedom / 2.0,
        torch.full_like(degrees_of_freedom, 0.5),
    )
    probabilities = torch.where(
        value < 0,
        0.5 * beta,
        1.0 - 0.5 * beta,
    )
    return torch.where(
        value == 0,
        torch.full_like(probabilities, 0.5),
        probabilities,
    )


def _truncation_terms(
    sigma: Tensor,
    nu: Tensor,
    tau: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return log normalizer and inverse-Mills ratio for BCT truncation."""
    near_zero = nu.abs() < _SMALL_NU
    safe_absolute_nu = torch.where(near_zero, torch.ones_like(nu), nu.abs())
    boundary = 1.0 / (sigma * safe_absolute_nu)
    normalizer = _student_t_cdf(boundary, tau)
    log_normalizer = torch.log(normalizer)
    mills_ratio = torch.exp(_student_t_log_prob(boundary, tau) - log_normalizer)
    zeros = torch.zeros_like(nu)
    return (
        torch.where(near_zero, zeros, log_normalizer),
        torch.where(near_zero, zeros, mills_ratio),
    )


class BoxCoxTDistribution(Distribution):
    """Torch distribution for the BCT parameterization used by GAMLSS."""

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
        use_normal_limit = tau > _LARGE_TAU
        safe_tau = torch.where(
            use_normal_limit,
            torch.full_like(tau, 10.0),
            tau,
        )
        z, _ = _box_cox_z(value, mu, sigma, nu)
        log_normalizer, _ = _truncation_terms(sigma, nu, safe_tau)
        log_density = (
            (nu - 1.0) * torch.log(value)
            - nu * torch.log(mu)
            - torch.log(sigma)
            + _student_t_log_prob(z, safe_tau)
            - log_normalizer
        )
        normal_density = BoxCoxColeGreenDistribution(
            mu,
            sigma,
            nu,
            validate_args=False,
        ).log_prob(value)
        return torch.where(use_normal_limit, normal_density, log_density)

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw from the truncated Box-Cox Student-t representation."""
        shape = self._extended_shape(sample_shape)
        mu = self.mu.expand(shape)
        sigma = self.sigma.expand(shape)
        nu = self.nu.expand(shape)
        tau = self.tau.expand(shape)
        score_distribution = StudentT(tau)
        score = score_distribution.sample()
        valid = _valid_box_cox_score(score, sigma, nu)
        for _ in range(100):
            if bool(valid.all()):
                break
            replacement = score_distribution.sample()
            score = torch.where(valid, score, replacement)
            valid = _valid_box_cox_score(score, sigma, nu)
        else:
            raise RuntimeError("BCT latent-score rejection sampler did not converge")
        return _box_cox_response(score, mu, sigma, nu)

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
        denominator = _student_t_cdf(boundary, tau)
        unadjusted = _student_t_cdf(z, tau)
        lower_truncation = _student_t_cdf(-boundary, tau)
        probabilities = torch.where(
            nu > 0,
            (unadjusted - lower_truncation) / denominator,
            unadjusted / denominator,
        )
        return probabilities.clamp(0.0, 1.0)


class BoxCoxT(Family):
    """Box-Cox t family compatible with ``gamlss.dist::BCT``."""

    name = "BCT"
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

    def distribution(self, parameters: Mapping[str, Tensor]) -> BoxCoxTDistribution:
        return BoxCoxTDistribution(
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
        """Match the parameter-scale scores supplied by R's ``BCT``."""
        self.validate_response(response)
        mu, sigma, nu, tau, response = self._broadcast(response, parameters)
        z, log_ratio = _box_cox_z(response, mu, sigma, nu)
        weight = (tau + 1.0) / (tau + z.square())
        _, mills_ratio = _truncation_terms(sigma, nu, tau)
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

        boundary = 1.0 / (sigma * safe_absolute_nu)
        current_log_cdf = torch.log(_student_t_cdf(boundary, tau))
        incremented_log_cdf = torch.log(_student_t_cdf(boundary, tau + _TAU_DIFFERENCE))
        cdf_tau_derivative = (incremented_log_cdf - current_log_cdf) / _TAU_DIFFERENCE
        cdf_tau_derivative = torch.where(
            near_zero,
            zeros,
            cdf_tau_derivative,
        )
        tau_score = (
            -0.5 * torch.log1p(z.square() / tau)
            + weight * z.square() / (2.0 * tau)
            + 0.5 * torch.digamma((tau + 1.0) / 2.0)
            - 0.5 * torch.digamma(tau / 2.0)
            - 1.0 / (2.0 * tau)
            - cdf_tau_derivative
        )
        return {
            "mu": weight * z / (mu * sigma) + (nu / mu) * (weight * z.square() - 1.0),
            "sigma": (weight * z.square() - 1.0) / sigma + sigma_correction,
            "nu": log_ratio - weight * z * z_derivative + nu_correction,
            "tau": tau_score,
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        """Match the Fisher-scoring derivatives supplied by R's ``BCT``."""
        mu, sigma, nu, tau, response = self._broadcast(response, parameters)
        tau_second = (
            torch.special.polygamma(1, (tau + 1.0) / 2.0)
            - torch.special.polygamma(1, tau / 2.0)
            + 2.0 * (tau + 5.0) / (tau * (tau + 1.0) * (tau + 3.0))
        ) / 4.0
        tau_second = torch.minimum(
            tau_second,
            response.new_full(response.shape, -1e-15),
        )
        return {
            ("mu", "mu"): -(tau + 2.0 * nu.square() * sigma.square() * tau + 1.0)
            / ((tau + 3.0) * mu.square() * sigma.square()),
            ("sigma", "sigma"): -2.0 * tau / (sigma.square() * (tau + 3.0)),
            ("nu", "nu"): -7.0 * sigma.square() / 4.0,
            ("tau", "tau"): tau_second,
            ("mu", "sigma"): -2.0 * nu * tau / (mu * sigma * (tau + 3.0)),
            ("mu", "nu"): (tau - 3.0) / (2.0 * mu * (tau + 3.0)),
            ("mu", "tau"): 2.0 * nu / (mu * (tau + 1.0) * (tau + 3.0)),
            ("sigma", "nu"): -sigma * nu * tau / (tau + 3.0),
            ("sigma", "tau"): 2.0 / (sigma * (tau + 1.0) * (tau + 3.0)),
            ("nu", "tau"): 2.0 * sigma.square() * nu / tau.square(),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::BCT``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.1)
        if "nu" in parameters:
            defaults["nu"] = torch.full_like(response, 0.5)
        if "tau" in parameters:
            defaults["tau"] = torch.full_like(response, 10.0)
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
                f"BCT {context} requires a finite strictly positive response"
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


BCT = BoxCoxT
