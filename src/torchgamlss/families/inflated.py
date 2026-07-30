"""Inflated count and beta families compatible with ``gamlss.dist``."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from scipy.stats import beta as scipy_beta
from torch import Tensor
from torch.distributions import Beta as TorchBeta

from torchgamlss.families._scipy import scipy_call, scipy_cdf
from torchgamlss.families.base import Family
from torchgamlss.families.bct import _regularized_incomplete_beta
from torchgamlss.families.beta import Beta
from torchgamlss.families.negative_binomial import NegativeBinomial
from torchgamlss.families.point_mass import PointMassFamily
from torchgamlss.families.poisson import Poisson
from torchgamlss.links import Link, LogitLink, LogLink


class _BetaMeanPrecision(Family):
    """Beta law parameterized by mean and total concentration."""

    name = "BE_precision"
    parameter_names = ("mu", "sigma")

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
    ) -> None:
        self._links = {
            "mu": mu_link or LogitLink(),
            "sigma": sigma_link or LogLink(),
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def distribution(self, parameters: Mapping[str, Tensor]) -> TorchBeta:
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return TorchBeta(
            concentration1=mu * sigma,
            concentration0=(1.0 - mu) * sigma,
            validate_args=True,
        )

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        self.validate_response(response)
        return super().log_prob(response, parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        mu, sigma, response = torch.broadcast_tensors(
            parameters["mu"],
            parameters["sigma"],
            response,
        )
        return scipy_cdf(
            response,
            scipy_beta.cdf,
            response,
            mu * sigma,
            (1.0 - mu) * sigma,
        )

    def _differentiable_cdf(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        mu, sigma, response = torch.broadcast_tensors(
            parameters["mu"],
            parameters["sigma"],
            response,
        )
        return _regularized_incomplete_beta(
            response,
            mu * sigma,
            (1.0 - mu) * sigma,
        )

    def _quantile(
        self,
        probabilities: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        mu = parameters["mu"]
        sigma = parameters["sigma"]
        return scipy_call(
            probabilities,
            scipy_beta.ppf,
            probabilities,
            mu * sigma,
            (1.0 - mu) * sigma,
        )

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        mu, sigma, response = torch.broadcast_tensors(
            parameters["mu"],
            parameters["sigma"],
            response,
        )
        alpha = mu * sigma
        beta = (1.0 - mu) * sigma
        logit_response = torch.log(response) - torch.log1p(-response)
        digamma_difference = torch.digamma(alpha) - torch.digamma(beta)
        return {
            "mu": sigma * (logit_response - digamma_difference),
            "sigma": (
                mu * (logit_response - digamma_difference)
                + torch.log1p(-response)
                - torch.digamma(beta)
                + torch.digamma(sigma)
            ),
        }

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        mu, sigma, response = torch.broadcast_tensors(
            parameters["mu"],
            parameters["sigma"],
            response,
        )
        alpha = mu * sigma
        beta = (1.0 - mu) * sigma
        trigamma_alpha = torch.special.polygamma(1, alpha)
        trigamma_beta = torch.special.polygamma(1, beta)
        return {
            ("mu", "mu"): -sigma.square() * (trigamma_alpha + trigamma_beta),
            ("sigma", "sigma"): -(
                mu.square() * trigamma_alpha
                + (1.0 - mu).square() * trigamma_beta
                - torch.special.polygamma(1, sigma)
            ),
            # Preserve the working cross derivative used by gamlss.dist.
            ("mu", "sigma"): (
                -sigma * mu * trigamma_alpha - (1.0 - mu) * trigamma_beta
            ),
        }

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
            or bool((response <= 0.0).any())
            or bool((response >= 1.0).any())
        ):
            raise ValueError(
                f"beta {context} requires a finite response strictly "
                "between zero and one"
            )


class ZeroInflatedPoisson(PointMassFamily):
    """Poisson with direct zero-inflation probability ``sigma`` (``ZIP``)."""

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
    ) -> None:
        super().__init__(
            Poisson(mu_link=mu_link),
            points=(0.0,),
            mass_parameter_names=("sigma",),
            parameterization="probability",
            mass_links={"sigma": sigma_link or LogitLink()},
            name="ZIP",
        )

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.1)
        return defaults


class ZeroInflatedNegativeBinomial(PointMassFamily):
    """NBI with direct zero-inflation probability ``nu`` (``ZINBI``)."""

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        super().__init__(
            NegativeBinomial(
                mu_link=mu_link,
                sigma_link=sigma_link,
            ),
            points=(0.0,),
            mass_parameter_names=("nu",),
            parameterization="probability",
            mass_links={"nu": nu_link or LogitLink()},
            name="ZINBI",
        )

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults = self.family._default_initial_parameters(
            response,
            parameters.intersection(self.family.parameter_names),
        )
        if "nu" in parameters:
            zero_fraction = (response == 0.0).to(response.dtype).mean()
            defaults["nu"] = ((zero_fraction + 0.01) / 2.0).expand_as(response).clone()
        return defaults


class _DirectProbabilityBeta(PointMassFamily):
    """Shared expected derivatives for ``BEZI`` and ``BEOI``."""

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        base_parameters = {
            name: parameters[name] for name in self.family.parameter_names
        }
        base_second = self.family.expected_second_derivatives(
            response,
            base_parameters,
        )
        atom = response == self.points[0]
        result = {
            pair: torch.where(atom, torch.zeros_like(value), value)
            for pair, value in base_second.items()
        }
        nu, _ = torch.broadcast_tensors(parameters["nu"], response)
        result[("nu", "nu")] = -1.0 / (nu * (1.0 - nu))
        zero = torch.zeros_like(response)
        result[("mu", "nu")] = zero
        result[("sigma", "nu")] = zero
        return result

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
            defaults["nu"] = torch.full_like(response, 0.3)
        return defaults


class BetaZeroInflated(_DirectProbabilityBeta):
    """Zero-inflated beta with mean/precision base parameters (``BEZI``)."""

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        super().__init__(
            _BetaMeanPrecision(
                mu_link=mu_link,
                sigma_link=sigma_link,
            ),
            points=(0.0,),
            mass_parameter_names=("nu",),
            parameterization="probability",
            mass_links={"nu": nu_link or LogitLink()},
            name="BEZI",
        )


class BetaOneInflated(_DirectProbabilityBeta):
    """One-inflated beta with mean/precision base parameters (``BEOI``)."""

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        super().__init__(
            _BetaMeanPrecision(
                mu_link=mu_link,
                sigma_link=sigma_link,
            ),
            points=(1.0,),
            mass_parameter_names=("nu",),
            parameterization="probability",
            mass_links={"nu": nu_link or LogitLink()},
            name="BEOI",
        )


class _OddsBeta(PointMassFamily):
    """Shared R-compatible working derivatives for ``BEINF*`` families."""

    def _base_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        base_parameters = {
            name: parameters[name] for name in self.family.parameter_names
        }
        base_second = self.family.expected_second_derivatives(
            response,
            base_parameters,
        )
        atom = torch.zeros_like(response, dtype=torch.bool)
        for point in self.points:
            atom = atom | (response == point)
        return {
            pair: torch.where(atom, torch.zeros_like(value), value)
            for pair, value in base_second.items()
        }


class BetaInflated(_OddsBeta):
    """Beta distribution inflated at zero and one (``BEINF``)."""

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
        tau_link: Link | None = None,
    ) -> None:
        super().__init__(
            Beta(
                mu_link=mu_link,
                sigma_link=sigma_link,
            ),
            points=(0.0, 1.0),
            mass_parameter_names=("nu", "tau"),
            parameterization="odds",
            mass_links={
                "nu": nu_link or LogLink(),
                "tau": tau_link or LogLink(),
            },
            name="BEINF",
        )

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        result = self._base_second_derivatives(response, parameters)
        nu, tau, _ = torch.broadcast_tensors(
            parameters["nu"],
            parameters["tau"],
            response,
        )
        denominator = 1.0 + nu + tau
        result[("nu", "nu")] = -(1.0 + tau) / (nu * denominator.square())
        result[("tau", "tau")] = -(1.0 + nu) / (tau * denominator.square())
        zero = torch.zeros_like(response)
        result[("mu", "nu")] = zero
        result[("mu", "tau")] = zero
        result[("sigma", "nu")] = zero
        result[("sigma", "tau")] = zero
        result[("nu", "tau")] = denominator.square().reciprocal()
        return result

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.5)
        if "nu" in parameters:
            defaults["nu"] = torch.full_like(response, 0.3)
        if "tau" in parameters:
            defaults["tau"] = torch.full_like(response, 0.3)
        return defaults


class _SingleOddsBeta(_OddsBeta):
    """Shared derivatives and starts for one-boundary ``BEINF`` variants."""

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        result = self._base_second_derivatives(response, parameters)
        nu, _ = torch.broadcast_tensors(parameters["nu"], response)
        result[("nu", "nu")] = -1.0 / (nu * (1.0 + nu).square())
        zero = torch.zeros_like(response)
        result[("mu", "nu")] = zero
        result[("sigma", "nu")] = zero
        return result

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults = {}
        if "mu" in parameters:
            defaults["mu"] = (response + response.mean()) / 2.0
        if "sigma" in parameters:
            defaults["sigma"] = torch.full_like(response, 0.5)
        if "nu" in parameters:
            defaults["nu"] = torch.full_like(response, self._nu_start)
        return defaults


class BetaInflatedZero(_SingleOddsBeta):
    """Beta distribution inflated at zero with odds ``nu`` (``BEINF0``)."""

    _nu_start = 0.3

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        super().__init__(
            Beta(
                mu_link=mu_link,
                sigma_link=sigma_link,
            ),
            points=(0.0,),
            mass_parameter_names=("nu",),
            parameterization="odds",
            mass_links={"nu": nu_link or LogLink()},
            name="BEINF0",
        )


class BetaInflatedOne(_SingleOddsBeta):
    """Beta distribution inflated at one with odds ``nu`` (``BEINF1``)."""

    _nu_start = 0.1

    def __init__(
        self,
        *,
        mu_link: Link | None = None,
        sigma_link: Link | None = None,
        nu_link: Link | None = None,
    ) -> None:
        super().__init__(
            Beta(
                mu_link=mu_link,
                sigma_link=sigma_link,
            ),
            points=(1.0,),
            mass_parameter_names=("nu",),
            parameterization="odds",
            mass_links={"nu": nu_link or LogLink()},
            name="BEINF1",
        )


ZIP = ZeroInflatedPoisson
ZINBI = ZeroInflatedNegativeBinomial
BEZI = BetaZeroInflated
BEOI = BetaOneInflated
BEINF = BetaInflated
BEINF0 = BetaInflatedZero
BEINF1 = BetaInflatedOne


__all__ = [
    "BEINF",
    "BEINF0",
    "BEINF1",
    "BEOI",
    "BEZI",
    "BetaInflated",
    "BetaInflatedOne",
    "BetaInflatedZero",
    "BetaOneInflated",
    "BetaZeroInflated",
    "ZINBI",
    "ZIP",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]
