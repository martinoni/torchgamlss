"""Fixed-bound truncated GAMLSS families."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch
from torch import Size, Tensor
from torch.distributions import Distribution

from torchgamlss.families.base import Family
from torchgamlss.links import Link


class _TruncatedDistribution(Distribution):
    """Distribution adapter used by :class:`TruncatedFamily`."""

    arg_constraints: dict[str, Any] = {}
    has_rsample = False

    def __init__(
        self,
        family: Family,
        parameters: Mapping[str, Tensor],
        *,
        lower: float | None,
        upper: float | None,
        validate_args: bool | None = None,
    ) -> None:
        self.family = family
        self.parameters = dict(parameters)
        self.lower = lower
        self.upper = upper
        self.base_distribution = family.distribution(parameters)

        parameter_values = [
            parameters[parameter] for parameter in family.parameter_names
        ]
        broadcast_parameters = torch.broadcast_tensors(*parameter_values)
        self._batch_reference = broadcast_parameters[0]
        self._lower_mass, self._upper_mass = self._boundary_masses()
        raw_normalization = self._upper_mass - self._lower_mass
        tiny = torch.finfo(self._batch_reference.dtype).tiny
        self._normalization = raw_normalization.clamp_min(tiny)

        super().__init__(
            batch_shape=self.base_distribution.batch_shape,
            event_shape=self.base_distribution.event_shape,
            validate_args=validate_args,
        )

    def _boundary_masses(self) -> tuple[Tensor, Tensor]:
        if self.lower is None:
            lower_mass = torch.zeros_like(self._batch_reference)
        else:
            lower = self._batch_reference.new_tensor(self.lower)
            lower_mass = self.family._differentiable_cdf(
                lower,
                self.parameters,
            )

        if self.upper is None:
            upper_mass = torch.ones_like(self._batch_reference)
        else:
            upper = self._batch_reference.new_tensor(self.upper)
            if self.family.is_discrete:
                upper = upper - 1.0
            upper_mass = self.family._differentiable_cdf(
                upper,
                self.parameters,
            )
        return lower_mass, upper_mass

    def _inside_support(self, value: Tensor) -> Tensor:
        inside = torch.ones_like(value, dtype=torch.bool)
        if self.lower is not None:
            if self.family.is_discrete:
                inside = inside & (value > self.lower)
            else:
                inside = inside & (value >= self.lower)
        if self.upper is not None:
            if self.family.is_discrete:
                inside = inside & (value < self.upper)
            else:
                inside = inside & (value <= self.upper)
        return inside

    def log_prob(self, value: Tensor) -> Tensor:
        inside = self._inside_support(value)
        if self._validate_args and not bool(inside.all()):
            raise ValueError(
                f"response contains values outside the support of {self.family.name}"
            )
        log_prob = self.base_distribution.log_prob(value)
        result = log_prob - self._normalization.log()
        return torch.where(
            inside,
            result,
            torch.full_like(result, -torch.inf),
        )

    def cdf(self, value: Tensor) -> Tensor:
        base_cdf = self.family._differentiable_cdf(value, self.parameters)
        result = (base_cdf - self._lower_mass) / self._normalization
        if self.lower is not None:
            result = torch.where(value < self.lower, torch.zeros_like(result), result)
        if self.upper is not None:
            result = torch.where(value >= self.upper, torch.ones_like(result), result)
        return result.clamp(0.0, 1.0)

    def icdf(self, value: Tensor) -> Tensor:
        if self._validate_args and (
            not torch.isfinite(value).all()
            or not bool(((value >= 0) & (value <= 1)).all())
        ):
            raise ValueError("truncated-distribution probabilities must be in [0, 1]")
        base_probabilities = (
            self._lower_mass + value * self._normalization
        )
        return self.family._quantile(base_probabilities, self.parameters)

    @torch.no_grad()
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        probabilities = torch.rand(
            shape,
            dtype=self._batch_reference.dtype,
            device=self._batch_reference.device,
        )
        epsilon = torch.finfo(probabilities.dtype).eps
        probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
        sample_dimensions = (1,) * len(sample_shape)
        expanded_parameters = {
            parameter: value.reshape(sample_dimensions + value.shape).expand(shape)
            for parameter, value in self.parameters.items()
        }
        lower_mass = self._lower_mass.reshape(
            sample_dimensions + self._lower_mass.shape
        ).expand(shape)
        normalization = self._normalization.reshape(
            sample_dimensions + self._normalization.shape
        ).expand(shape)
        base_probabilities = lower_mass + probabilities * normalization
        return self.family._quantile(base_probabilities, expanded_parameters)


class TruncatedFamily(Family):
    """Create a fixed-bound truncation of an existing response family.

    Continuous bounds are closed. Discrete bounds follow ``gamlss.tr`` and
    are open: ``lower < y < upper``. At least one finite scalar bound is
    required.

    The first derivatives include the truncation normalizer. For classical
    RS/CG fitting, expected second derivatives are inherited from the base
    family, matching the working approximation used by ``gamlss.tr``.
    """

    def __init__(
        self,
        family: Family,
        *,
        lower: Real | Tensor | None = None,
        upper: Real | Tensor | None = None,
    ) -> None:
        if not isinstance(family, Family):
            raise TypeError("family must be a torchgamlss Family instance")
        self.family = family
        self.lower = self._validate_bound(lower, "lower")
        self.upper = self._validate_bound(upper, "upper")
        if self.lower is None and self.upper is None:
            raise ValueError("at least one truncation bound is required")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("lower truncation bound must be less than upper")
        if family.is_discrete:
            for label, bound in (("lower", self.lower), ("upper", self.upper)):
                if bound is not None and not bound.is_integer():
                    raise ValueError(
                        f"{label} truncation bound must be an integer for "
                        "a discrete family"
                    )

        self.name = f"{family.name}tr"
        self.parameter_names = family.parameter_names
        self.is_discrete = family.is_discrete

    @staticmethod
    def _validate_bound(
        bound: Real | Tensor | None,
        label: str,
    ) -> float | None:
        if bound is None:
            return None
        if isinstance(bound, Tensor):
            if bound.ndim != 0 or bound.requires_grad:
                raise ValueError(
                    f"{label} truncation bound must be a fixed scalar"
                )
            bound = float(bound.detach().cpu())
        elif isinstance(bound, bool) or not isinstance(bound, Real):
            raise TypeError(f"{label} truncation bound must be a real scalar")
        value = float(bound)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{label} truncation bound must be finite")
        return value

    @property
    def links(self) -> Mapping[str, Link]:
        return self.family.links

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> _TruncatedDistribution:
        missing = set(self.parameter_names).difference(parameters)
        extra = set(parameters).difference(self.parameter_names)
        if missing or extra:
            raise ValueError(
                "Truncated-family parameters do not match the base family: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return _TruncatedDistribution(
            self.family,
            parameters,
            lower=self.lower,
            upper=self.upper,
            validate_args=True,
        )

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_response(response)
        create_graph = torch.is_grad_enabled() and any(
            parameters[parameter].requires_grad
            for parameter in self.parameter_names
        )
        with torch.enable_grad():
            values = []
            for parameter in self.parameter_names:
                value = parameters[parameter]
                if not value.requires_grad:
                    value = value.detach().requires_grad_(True)
                values.append(value)
            broadcast_values = torch.broadcast_tensors(*values, response)[
                : len(values)
            ]
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
        outside = torch.zeros_like(response, dtype=torch.bool)
        if self.lower is not None:
            if self.is_discrete:
                outside = outside | (response <= self.lower)
            else:
                outside = outside | (response < self.lower)
        if self.upper is not None:
            if self.is_discrete:
                outside = outside | (response >= self.upper)
            else:
                outside = outside | (response > self.upper)
        if bool(outside.any()):
            interval = self._support_description()
            raise ValueError(
                f"{self.name} {context} requires responses in {interval}"
            )

    def _support_description(self) -> str:
        left = "-inf" if self.lower is None else f"{self.lower:g}"
        right = "inf" if self.upper is None else f"{self.upper:g}"
        if self.is_discrete:
            return f"the open interval ({left}, {right})"
        return f"the closed interval [{left}, {right}]"

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        return self.family._default_initial_parameters(response, parameters)
