"""Scalar- and observation-bound truncated GAMLSS families."""

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
        lower: float | Tensor | None,
        upper: float | Tensor | None,
        validate_args: bool | None = None,
    ) -> None:
        self.family = family
        parameter_values = [
            parameters[parameter] for parameter in family.parameter_names
        ]
        broadcast_parameters = torch.broadcast_tensors(*parameter_values)
        parameter_shape = broadcast_parameters[0].shape
        reference = broadcast_parameters[0]
        aligned_lower = self._align_bound(
            lower,
            reference,
            parameter_shape,
            "lower",
        )
        aligned_upper = self._align_bound(
            upper,
            reference,
            parameter_shape,
            "upper",
        )
        bounds = [
            bound for bound in (aligned_lower, aligned_upper) if bound is not None
        ]
        try:
            broadcast_values = torch.broadcast_tensors(
                *broadcast_parameters,
                *bounds,
            )
        except RuntimeError as error:
            raise ValueError(
                "truncation bounds cannot be broadcast with family parameters"
            ) from error
        parameter_count = len(family.parameter_names)
        broadcast_parameters = broadcast_values[:parameter_count]
        bound_index = parameter_count
        self.lower = None
        if aligned_lower is not None:
            self.lower = broadcast_values[bound_index]
            bound_index += 1
        self.upper = None
        if aligned_upper is not None:
            self.upper = broadcast_values[bound_index]
        self.parameters = dict(
            zip(
                family.parameter_names,
                broadcast_parameters,
                strict=True,
            )
        )
        self.base_distribution = family.distribution(self.parameters)
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

    @staticmethod
    def _align_bound(
        bound: float | Tensor | None,
        reference: Tensor,
        parameter_shape: torch.Size,
        label: str,
    ) -> Tensor | None:
        if bound is None:
            return None
        if isinstance(bound, Tensor):
            value = bound.to(dtype=reference.dtype, device=reference.device)
        else:
            value = reference.new_tensor(bound)
        if value.ndim == 0 or not parameter_shape:
            return value
        observation_count = value.shape[0]
        parameter_observations = parameter_shape[0]
        if parameter_observations not in (1, observation_count):
            raise ValueError(
                f"{label} truncation bound has {observation_count} rows but "
                f"family parameters have {parameter_observations}"
            )
        return value.reshape((observation_count,) + (1,) * (len(parameter_shape) - 1))

    def _boundary_masses(self) -> tuple[Tensor, Tensor]:
        if self.lower is None:
            lower_mass = torch.zeros_like(self._batch_reference)
        else:
            lower_mass = self.family._differentiable_cdf(
                self.lower,
                self.parameters,
            )

        if self.upper is None:
            upper_mass = torch.ones_like(self._batch_reference)
        else:
            upper = self.upper
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
        base_probabilities = self._lower_mass + value * self._normalization
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
    """Create a scalar- or observation-bound truncation of a response family.

    Continuous bounds are closed. Discrete bounds follow ``gamlss.tr`` and
    are open: ``lower < y < upper``. Each supplied bound may be a finite
    scalar or a fixed one-dimensional tensor with one value per observation.
    At least one bound is required.

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
        self._validate_bound_pair()
        if family.is_discrete:
            for label, bound in (("lower", self.lower), ("upper", self.upper)):
                if bound is not None and not self._is_integer_bound(bound):
                    raise ValueError(
                        f"{label} truncation bound must be an integer for "
                        "a discrete family"
                    )

        self.name = f"{family.name}tr"
        self.parameter_names = family.parameter_names
        self.is_discrete = family.is_discrete
        self.varying = isinstance(self.lower, Tensor) or isinstance(
            self.upper,
            Tensor,
        )

    @staticmethod
    def _validate_bound(
        bound: Real | Tensor | None,
        label: str,
    ) -> float | Tensor | None:
        if bound is None:
            return None
        if isinstance(bound, Tensor):
            if bound.requires_grad:
                raise ValueError(
                    f"{label} truncation bound must be fixed and cannot "
                    "require gradients"
                )
            if bound.ndim not in (0, 1) or bound.numel() < 1:
                raise ValueError(
                    f"{label} truncation bound must be a scalar or a "
                    "non-empty one-dimensional tensor"
                )
            if bound.dtype == torch.bool or bound.is_complex():
                raise TypeError(f"{label} truncation bound must be real")
            if not torch.isfinite(bound).all():
                raise ValueError(f"{label} truncation bound must be finite")
            if bound.ndim == 0:
                return float(bound.detach().cpu())
            return bound.detach().clone()
        elif isinstance(bound, bool) or not isinstance(bound, Real):
            raise TypeError(f"{label} truncation bound must be a real scalar")
        value = float(bound)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{label} truncation bound must be finite")
        return value

    def _validate_bound_pair(self) -> None:
        if self.lower is None or self.upper is None:
            return
        lower = torch.as_tensor(self.lower, dtype=torch.float64, device="cpu")
        upper = torch.as_tensor(self.upper, dtype=torch.float64, device="cpu")
        try:
            lower, upper = torch.broadcast_tensors(lower, upper)
        except RuntimeError as error:
            raise ValueError(
                "lower and upper truncation bounds must have the same observation count"
            ) from error
        if bool((lower >= upper).any()):
            raise ValueError(
                "lower truncation bound must be less than upper for every observation"
            )

    @staticmethod
    def _is_integer_bound(bound: float | Tensor) -> bool:
        if isinstance(bound, Tensor):
            return bool((bound == torch.floor(bound)).all())
        return bound.is_integer()

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
        outside = torch.zeros_like(response, dtype=torch.bool)
        lower = self._bound_for_response(self.lower, response, "lower")
        upper = self._bound_for_response(self.upper, response, "upper")
        if lower is not None:
            if self.is_discrete:
                outside = outside | (response <= lower)
            else:
                outside = outside | (response < lower)
        if upper is not None:
            if self.is_discrete:
                outside = outside | (response >= upper)
            else:
                outside = outside | (response > upper)
        if bool(outside.any()):
            interval = self._support_description()
            raise ValueError(f"{self.name} {context} requires responses in {interval}")

    @staticmethod
    def _bound_for_response(
        bound: float | Tensor | None,
        response: Tensor,
        label: str,
    ) -> Tensor | None:
        if bound is None:
            return None
        if isinstance(bound, Tensor):
            if bound.shape[0] != response.shape[0]:
                raise ValueError(
                    f"{label} truncation bound must have one value per "
                    f"observation: expected {response.shape[0]}, got "
                    f"{bound.shape[0]}"
                )
            return bound.to(dtype=response.dtype, device=response.device)
        return response.new_tensor(bound)

    def _support_description(self) -> str:
        if self.varying:
            return "the observation-specific truncation intervals"
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
