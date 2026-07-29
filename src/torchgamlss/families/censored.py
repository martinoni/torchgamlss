"""Censored response representation and likelihood composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor
from torch.distributions import Distribution

from torchgamlss.families.base import Family
from torchgamlss.links import Link


class Censoring(IntEnum):
    """Observation status codes compatible with ``survival::Surv``."""

    RIGHT = 0
    EXACT = 1
    LEFT = 2
    INTERVAL = 3


@dataclass(frozen=True)
class CensoredResponse:
    """Fixed row-aligned censoring metadata for a one-dimensional response.

    ``observed`` contains the event time, censoring threshold, or lower
    interval endpoint. ``upper`` is required only when at least one row is
    interval-censored.
    """

    observed: Tensor
    status: Tensor
    upper: Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observed, Tensor)
            or not self.observed.is_floating_point()
            or self.observed.ndim != 1
            or self.observed.numel() < 1
            or not torch.isfinite(self.observed).all()
        ):
            raise ValueError(
                "censored observations must be a non-empty finite "
                "one-dimensional floating-point tensor"
            )
        if (
            not isinstance(self.status, Tensor)
            or self.status.ndim != 1
            or self.status.shape != self.observed.shape
            or self.status.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError(
                "censoring status must be a one-dimensional integer tensor "
                "with one value per observation"
            )
        if not bool(
            (
                (self.status >= int(Censoring.RIGHT))
                & (self.status <= int(Censoring.INTERVAL))
            ).all()
        ):
            raise ValueError("censoring status values must be between 0 and 3")

        observed = self.observed.detach().clone()
        status = (
            self.status.detach()
            .to(
                dtype=torch.int64,
                device=observed.device,
            )
            .clone()
        )
        upper = self.upper
        interval = status == int(Censoring.INTERVAL)
        if bool(interval.any()):
            if not isinstance(upper, Tensor):
                raise ValueError(
                    "an upper endpoint tensor is required for interval-censored rows"
                )
        if upper is not None:
            if (
                not isinstance(upper, Tensor)
                or not upper.is_floating_point()
                or upper.shape != observed.shape
                or not torch.isfinite(upper).all()
            ):
                raise ValueError(
                    "upper censoring endpoints must be a finite floating-point "
                    "tensor with one value per observation"
                )
            upper = (
                upper.detach()
                .to(
                    dtype=observed.dtype,
                    device=observed.device,
                )
                .clone()
            )
            if bool((upper[interval] <= observed[interval]).any()):
                raise ValueError(
                    "upper endpoints must exceed lower endpoints for "
                    "interval-censored rows"
                )

        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def right(cls, time: Tensor, event: Tensor) -> CensoredResponse:
        """Create exact/right-censored observations from an event indicator."""
        event = cls._event_indicator(time, event)
        status = torch.where(
            event,
            torch.full_like(event, int(Censoring.EXACT), dtype=torch.int64),
            torch.full_like(event, int(Censoring.RIGHT), dtype=torch.int64),
        )
        return cls(time, status)

    @classmethod
    def left(cls, time: Tensor, event: Tensor) -> CensoredResponse:
        """Create exact/left-censored observations from an event indicator."""
        event = cls._event_indicator(time, event)
        status = torch.where(
            event,
            torch.full_like(event, int(Censoring.EXACT), dtype=torch.int64),
            torch.full_like(event, int(Censoring.LEFT), dtype=torch.int64),
        )
        return cls(time, status)

    @classmethod
    def interval(cls, lower: Tensor, upper: Tensor) -> CensoredResponse:
        """Create observations known only to lie in ``(lower, upper]``."""
        if not isinstance(lower, Tensor):
            raise ValueError("interval lower endpoints must be a tensor")
        status = torch.full(
            lower.shape,
            int(Censoring.INTERVAL),
            dtype=torch.int64,
            device=lower.device,
        )
        return cls(lower, status, upper)

    @staticmethod
    def _event_indicator(time: Tensor, event: Tensor) -> Tensor:
        if (
            not isinstance(time, Tensor)
            or not isinstance(event, Tensor)
            or event.ndim != 1
            or event.shape != time.shape
        ):
            raise ValueError(
                "event must be a one-dimensional tensor with one value per time"
            )
        if event.dtype == torch.bool:
            return event.to(device=time.device)
        if event.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ) or not bool(((event == 0) | (event == 1)).all()):
            raise ValueError("event indicators must be boolean or binary integers")
        return event.to(device=time.device, dtype=torch.bool)


class CensoredFamily(Family):
    """Compose a base family with fixed censoring metadata.

    Status codes follow ``survival::Surv`` interval semantics: 0 is right
    censored, 1 exact, 2 left censored, and 3 interval censored. First
    derivatives use the censored likelihood; RS/CG working second derivatives
    are inherited from the base family, matching ``gamlss.cens``.
    """

    def __init__(self, family: Family, response: CensoredResponse) -> None:
        if not isinstance(family, Family):
            raise TypeError("family must be a torchgamlss Family instance")
        if family.is_discrete:
            raise ValueError(
                "censored likelihoods currently require a continuous base family"
            )
        if not isinstance(response, CensoredResponse):
            raise TypeError("response must be a CensoredResponse instance")
        self.family = family
        self.response = response
        self.name = f"{family.name}cens"
        self.parameter_names = family.parameter_names
        self.is_discrete = family.is_discrete

    @property
    def links(self) -> Mapping[str, Link]:
        return self.family.links

    def distribution(self, parameters: Mapping[str, Tensor]) -> Distribution:
        return self.family.distribution(parameters)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        """Evaluate the base event-time CDF."""
        return self.family.cdf(response, parameters)

    def survival(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        """Evaluate the base event-time survival function."""
        return self.family.survival(response, parameters)

    def cumulative_hazard(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        """Evaluate the base event-time cumulative hazard."""
        return self.family.cumulative_hazard(response, parameters)

    def hazard(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        """Evaluate the base event-time hazard."""
        return self.family.hazard(response, parameters)

    def log_prob(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        status = self.response.status.to(device=response.device)
        lower_cdf = self.family._differentiable_cdf(response, parameters)
        log_cdf = self.family._log_cdf(response, parameters)
        log_survival = self.family._log_survival(response, parameters)

        upper = response
        if self.response.upper is not None:
            upper = self.response.upper.to(
                dtype=response.dtype,
                device=response.device,
            )
        log_upper_cdf = self.family._log_cdf(upper, parameters)
        log_upper_survival = self.family._log_survival(upper, parameters)
        interval_from_cdf = self._log_difference(log_upper_cdf, log_cdf)
        interval_from_survival = self._log_difference(
            log_survival,
            log_upper_survival,
        )
        log_interval = torch.where(
            lower_cdf <= 0.5,
            interval_from_cdf,
            interval_from_survival,
        )
        exact = self.family.log_prob(response, parameters)

        result = torch.where(
            status == int(Censoring.RIGHT),
            log_survival,
            exact,
        )
        result = torch.where(
            status == int(Censoring.LEFT),
            log_cdf,
            result,
        )
        return torch.where(
            status == int(Censoring.INTERVAL),
            log_interval,
            result,
        )

    @staticmethod
    def _log_difference(log_larger: Tensor, log_smaller: Tensor) -> Tensor:
        epsilon = torch.finfo(log_larger.dtype).eps
        log_ratio = (log_smaller - log_larger).clamp_max(-epsilon)
        return log_larger + torch.log1p(-torch.exp(log_ratio))

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        create_graph = torch.is_grad_enabled() and any(
            parameters[parameter].requires_grad for parameter in self.parameter_names
        )
        with torch.enable_grad():
            values = []
            for parameter in self.parameter_names:
                value = parameters[parameter]
                if not value.requires_grad:
                    value = value.detach().requires_grad_(True)
                values.append(value)
            broadcast_values = torch.broadcast_tensors(*values, response)[: len(values)]
            differentiable_parameters = dict(
                zip(self.parameter_names, broadcast_values, strict=True)
            )
            log_prob = self.log_prob(response, differentiable_parameters)
            gradients = torch.autograd.grad(
                log_prob.sum(),
                broadcast_values,
                create_graph=create_graph,
            )
        return dict(zip(self.parameter_names, gradients, strict=True))

    def expected_second_derivatives(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[tuple[str, str], Tensor]:
        self.validate_response(response)
        return self.family.expected_second_derivatives(response, parameters)

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        self.family.validate_response(response, context=context)
        expected = self.response.observed.to(
            dtype=response.dtype,
            device=response.device,
        )
        if response.shape != expected.shape or not torch.equal(response, expected):
            raise ValueError(
                f"{self.name} {context} requires the observations stored in "
                "its CensoredResponse"
            )
        interval = self.response.status == int(Censoring.INTERVAL)
        if bool(interval.any()):
            assert self.response.upper is not None
            upper = self.response.upper[interval]
            self.family.validate_response(
                upper,
                context="interval upper endpoints",
            )

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        return self.family._default_initial_parameters(response, parameters)
