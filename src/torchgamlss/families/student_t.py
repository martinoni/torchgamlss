"""Student-t location-scale GAMLSS family."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.stats import t as scipy_student_t
from torch import Tensor
from torch.distributions import (
    Distribution,
    constraints,
)
from torch.distributions import (
    Normal as TorchNormal,
)
from torch.distributions import (
    StudentT as TorchStudentT,
)
from torch.distributions.utils import broadcast_all

from torchgamlss.families._scipy import scipy_call
from torchgamlss.families.base import Family
from torchgamlss.families.bct import _student_t_cdf, _student_t_log_prob
from torchgamlss.links import IdentityLink, Link, LogLink

_LARGE_NU = 1e6


class StudentTDistribution(Distribution):
    """Location-scale Student-t distribution with GAMLSS ``TF`` semantics."""

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
        return torch.where(
            self.nu > 1.0,
            self.mu,
            torch.full_like(self.mu, float("nan")),
        )

    @property
    def variance(self) -> Tensor:
        finite_variance = (
            self.sigma.square() * self.nu / (self.nu - 2.0)
        )
        return torch.where(
            self.nu > 2.0,
            finite_variance,
            torch.full_like(finite_variance, float("inf")),
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
        use_normal_limit = nu > _LARGE_NU
        safe_nu = torch.where(
            use_normal_limit,
            torch.full_like(nu, 10.0),
            nu,
        )
        standardized = (value - mu) / sigma
        student_log_density = (
            _student_t_log_prob(standardized, safe_nu) - torch.log(sigma)
        )
        normal_log_density = TorchNormal(
            mu,
            sigma,
            validate_args=False,
        ).log_prob(value)
        return torch.where(
            use_normal_limit,
            normal_log_density,
            student_log_density,
        )

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        mu = self.mu.expand(shape)
        sigma = self.sigma.expand(shape)
        nu = self.nu.expand(shape)
        use_normal_limit = nu > _LARGE_NU
        safe_nu = torch.where(
            use_normal_limit,
            torch.full_like(nu, 10.0),
            nu,
        )
        student_sample = TorchStudentT(
            safe_nu,
            loc=mu,
            scale=sigma,
            validate_args=False,
        ).sample()
        normal_sample = TorchNormal(
            mu,
            sigma,
            validate_args=False,
        ).sample()
        return torch.where(use_normal_limit, normal_sample, student_sample)

    def cdf(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma, nu, value = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            value,
        )
        use_normal_limit = nu > _LARGE_NU
        safe_nu = torch.where(
            use_normal_limit,
            torch.full_like(nu, 10.0),
            nu,
        )
        standardized = (value - mu) / sigma
        student_probability = _student_t_cdf(standardized, safe_nu)
        normal_probability = torch.special.ndtr(standardized)
        return torch.where(
            use_normal_limit,
            normal_probability,
            student_probability,
        )

    def icdf(self, probability: Tensor) -> Tensor:
        mu, sigma, nu, probability = torch.broadcast_tensors(
            self.mu,
            self.sigma,
            self.nu,
            probability,
        )
        use_normal_limit = nu > _LARGE_NU
        safe_nu = torch.where(
            use_normal_limit,
            torch.full_like(nu, 10.0),
            nu,
        )
        student_quantile = scipy_call(
            probability,
            scipy_student_t.ppf,
            probability,
            safe_nu,
        )
        normal_quantile = torch.special.ndtri(probability)
        standardized = torch.where(
            use_normal_limit,
            normal_quantile,
            student_quantile,
        )
        return mu + sigma * standardized


class StudentT(Family):
    """Student-t family compatible with ``gamlss.dist::TF``."""

    name = "TF"
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
    ) -> StudentTDistribution:
        return StudentTDistribution(
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
        """Match the parameter-scale scores supplied by R's ``TF``."""
        self.validate_response(response)
        mu, sigma, nu, response = self._broadcast(response, parameters)
        squared_residual = (response - mu).square() / sigma.square()
        weight = (nu + 1.0) / (nu + squared_residual)
        return {
            "mu": weight * (response - mu) / sigma.square(),
            "sigma": (weight * squared_residual - 1.0) / sigma,
            "nu": (
                -torch.log1p(squared_residual / nu)
                + (weight * squared_residual - 1.0) / nu
                + torch.digamma((nu + 1.0) / 2.0)
                - torch.digamma(nu / 2.0)
            )
            / 2.0,
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        """Match the Fisher-scoring derivatives supplied by R's ``TF``."""
        _, sigma, nu, response = self._broadcast(response, parameters)
        nu_second = (
            torch.special.polygamma(1, (nu + 1.0) / 2.0)
            - torch.special.polygamma(1, nu / 2.0)
            + 2.0 * (nu + 5.0) / (nu * (nu + 1.0) * (nu + 3.0))
        ) / 4.0
        nu_second = torch.minimum(
            nu_second,
            response.new_full(response.shape, -1e-15),
        )
        zeros = torch.zeros_like(response)
        return {
            ("mu", "mu"): -(nu + 1.0) / ((nu + 3.0) * sigma.square()),
            ("sigma", "sigma"): -2.0 * nu / ((nu + 3.0) * sigma.square()),
            ("nu", "nu"): nu_second,
            ("mu", "sigma"): zeros,
            ("mu", "nu"): zeros,
            ("sigma", "nu"): 2.0 / (sigma * (nu + 3.0) * (nu + 1.0)),
        }

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Match the starting expressions in ``gamlss.dist::TF``."""
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(
                response,
                response.std(correction=1),
            )
        if "nu" in parameters:
            defaults["nu"] = torch.full_like(response, 10.0)
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


TF = StudentT
