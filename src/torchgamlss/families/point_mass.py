"""Generic point-mass compositions for inflated and adjusted families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Literal

import torch
from torch import Size, Tensor
from torch.distributions import Distribution

from torchgamlss.families.base import Family
from torchgamlss.links import Link, LogitLink, LogLink

MassParameterization = Literal["probability", "odds"]


class _PointMassDistribution(Distribution):
    """Distribution adapter for a base law plus fixed response atoms."""

    arg_constraints: dict[str, Any] = {}
    has_rsample = False

    def __init__(
        self,
        family: PointMassFamily,
        parameters: Mapping[str, Tensor],
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.family = family
        values = [parameters[name] for name in family.parameter_names]
        try:
            broadcast_values = torch.broadcast_tensors(*values)
        except RuntimeError as error:
            raise ValueError(
                "point-mass family parameters cannot be broadcast together"
            ) from error
        self.parameters = dict(
            zip(family.parameter_names, broadcast_values, strict=True)
        )
        self.base_parameters = {
            name: self.parameters[name] for name in family.family.parameter_names
        }
        self.base_distribution = family.family.distribution(self.base_parameters)
        self.base_weight, self.atom_weights = family._component_probabilities(
            self.parameters
        )
        self._reference = broadcast_values[0]

        super().__init__(
            batch_shape=self._reference.shape,
            event_shape=torch.Size(),
            validate_args=validate_args,
        )

    def _atom_mask(self, value: Tensor) -> Tensor:
        mask = torch.zeros_like(value, dtype=torch.bool)
        for point in self.family.points:
            mask = mask | (value == point)
        return mask

    def _safe_base_value(self, value: Tensor) -> Tensor:
        if self.family.family.is_discrete:
            return value
        atom = self._atom_mask(value)
        if not bool(atom.any()):
            return value
        base_mean, value = torch.broadcast_tensors(
            self.base_distribution.mean.detach(),
            value,
        )
        fallback = torch.ones_like(base_mean)
        safe_value = torch.where(torch.isfinite(base_mean), base_mean, fallback)
        return torch.where(atom, safe_value, value)

    def log_prob(self, value: Tensor) -> Tensor:
        safe_value = self._safe_base_value(value)
        base_log_prob = self.base_distribution.log_prob(safe_value)
        result = torch.log(self.base_weight) + base_log_prob
        for point, weight in zip(
            self.family.points,
            self.atom_weights,
            strict=True,
        ):
            at_point = value == point
            atom_log_prob = torch.log(weight)
            if self.family.family.is_discrete:
                atom_log_prob = torch.logaddexp(result, atom_log_prob)
            result = torch.where(at_point, atom_log_prob, result)
        return result

    def _base_cdf(self, value: Tensor) -> Tensor:
        return self.family.family._differentiable_cdf(
            value,
            self.base_parameters,
        )

    def _base_cdf_left(self, value: Tensor) -> Tensor:
        if self.family.family.is_discrete:
            value = value - 1.0
        return self._base_cdf(value)

    def cdf(self, value: Tensor) -> Tensor:
        result = self.base_weight * self._base_cdf(value)
        for point, weight in zip(
            self.family.points,
            self.atom_weights,
            strict=True,
        ):
            result = result + weight * (value >= point).to(value.dtype)
        return result.clamp(0.0, 1.0)

    def cdf_left(self, value: Tensor) -> Tensor:
        result = self.base_weight * self._base_cdf_left(value)
        for point, weight in zip(
            self.family.points,
            self.atom_weights,
            strict=True,
        ):
            result = result + weight * (value > point).to(value.dtype)
        return result.clamp(0.0, 1.0)

    def icdf(self, probability: Tensor) -> Tensor:
        if self._validate_args and (
            not torch.isfinite(probability).all()
            or not bool(((probability >= 0.0) & (probability <= 1.0)).all())
        ):
            raise ValueError("point-mass probabilities must lie in [0, 1]")

        finfo = torch.finfo(probability.dtype)

        def base_quantile(adjusted: Tensor) -> Tensor:
            return self.family.family._quantile(
                adjusted.clamp(finfo.eps, 1.0 - finfo.eps),
                self.base_parameters,
            )

        result = base_quantile(probability / self.base_weight)
        preceding_weight = torch.zeros_like(self.base_weight)
        for point, weight in zip(
            self.family.points,
            self.atom_weights,
            strict=True,
        ):
            point_value = torch.full_like(probability, point)
            left = (
                self.base_weight * self._base_cdf_left(point_value) + preceding_weight
            )
            preceding_weight = preceding_weight + weight
            right = self.base_weight * self._base_cdf(point_value) + preceding_weight
            in_jump = (probability > left) & (probability <= right)
            result = torch.where(in_jump, point_value, result)
            after = probability > right
            adjusted = (probability - preceding_weight) / self.base_weight
            result = torch.where(after, base_quantile(adjusted), result)
        return result

    @torch.no_grad()
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        probability = torch.rand(
            self._extended_shape(sample_shape),
            dtype=self._reference.dtype,
            device=self._reference.device,
        )
        finfo = torch.finfo(probability.dtype)
        return self.icdf(probability.clamp(finfo.eps, 1.0 - finfo.eps))

    @property
    def mean(self) -> Tensor:
        result = self.base_weight * self.base_distribution.mean
        for point, weight in zip(
            self.family.points,
            self.atom_weights,
            strict=True,
        ):
            result = result + point * weight
        return result

    @property
    def variance(self) -> Tensor:
        base_mean = self.base_distribution.mean
        second_moment = self.base_weight * (
            self.base_distribution.variance + base_mean.square()
        )
        for point, weight in zip(
            self.family.points,
            self.atom_weights,
            strict=True,
        ):
            second_moment = second_moment + point**2 * weight
        return (second_moment - self.mean.square()).clamp_min(0.0)


class PointMassFamily(Family):
    """Compose a base family with fixed response probability masses.

    A single mass may be parameterized directly as a probability in ``(0, 1)``
    or as positive odds relative to the base distribution. Multiple masses
    use positive odds, with the base component assigned reference weight one.
    This covers both the direct-probability and multinomial-odds conventions
    used by ``gamlss.dist`` and ``gamlss.inf``.
    """

    has_point_masses = True

    def __init__(
        self,
        family: Family,
        *,
        points: Sequence[Real],
        mass_parameter_names: Sequence[str] | None = None,
        parameterization: MassParameterization = "odds",
        mass_links: Mapping[str, Link] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(family, Family):
            raise TypeError("family must be a torchgamlss Family instance")
        try:
            normalized_points = tuple(float(point) for point in points)
        except (TypeError, ValueError) as error:
            raise TypeError("point masses must be supplied as real values") from error
        if (
            not normalized_points
            or len(set(normalized_points)) != len(normalized_points)
            or any(
                not torch.isfinite(torch.tensor(point)) for point in normalized_points
            )
        ):
            raise ValueError("point masses must be distinct finite values")
        if tuple(sorted(normalized_points)) != normalized_points:
            raise ValueError("point masses must be supplied in increasing order")
        if parameterization not in {"probability", "odds"}:
            raise ValueError("parameterization must be 'probability' or 'odds'")
        if parameterization == "probability" and len(normalized_points) != 1:
            raise ValueError(
                "direct probability parameterization supports exactly one mass"
            )

        if mass_parameter_names is None:
            mass_parameter_names = tuple(
                "xi0" if point == 0.0 else "xi1" if point == 1.0 else f"xi{index}"
                for index, point in enumerate(normalized_points)
            )
        else:
            mass_parameter_names = tuple(mass_parameter_names)
        if (
            len(mass_parameter_names) != len(normalized_points)
            or any(
                not isinstance(parameter, str) or not parameter
                for parameter in mass_parameter_names
            )
            or len(set(mass_parameter_names)) != len(mass_parameter_names)
        ):
            raise ValueError(
                "mass_parameter_names must contain one distinct name per point"
            )
        collisions = set(family.parameter_names).intersection(mass_parameter_names)
        if collisions:
            raise ValueError(
                "mass parameter names collide with base-family parameters: "
                f"{sorted(collisions)}"
            )

        default_link: Link = (
            LogitLink() if parameterization == "probability" else LogLink()
        )
        supplied_links = dict(mass_links or {})
        unknown_links = set(supplied_links).difference(mass_parameter_names)
        if unknown_links:
            raise ValueError(
                f"mass links contain unknown parameters: {sorted(unknown_links)}"
            )
        if any(not isinstance(link, Link) for link in supplied_links.values()):
            raise TypeError("mass links must be torchgamlss Link instances")

        self.family = family
        self.points = normalized_points
        self.mass_parameter_names = tuple(mass_parameter_names)
        self.parameterization = parameterization
        self.name = name or f"{family.name}PointMass"
        self.parameter_names = family.parameter_names + self.mass_parameter_names
        self.is_discrete = family.is_discrete
        self._links = {
            **family.links,
            **{
                parameter: supplied_links.get(parameter, default_link)
                for parameter in self.mass_parameter_names
            },
        }

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    def _component_probabilities(
        self,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        weights = tuple(parameters[name] for name in self.mass_parameter_names)
        if self.parameterization == "probability":
            probability = weights[0]
            if not torch.isfinite(probability).all() or bool(
                ((probability <= 0.0) | (probability >= 1.0)).any()
            ):
                raise ValueError("point-mass probability must lie strictly in (0, 1)")
            return 1.0 - probability, (probability,)

        if any(
            not torch.isfinite(weight).all() or bool((weight <= 0.0).any())
            for weight in weights
        ):
            raise ValueError("point-mass odds must be finite and strictly positive")
        denominator = torch.ones_like(weights[0])
        for weight in weights:
            denominator = denominator + weight
        return denominator.reciprocal(), tuple(
            weight / denominator for weight in weights
        )

    def mass_probabilities(
        self,
        parameters: Mapping[str, Tensor],
    ) -> dict[float, Tensor]:
        """Return normalized probability for every configured response mass."""
        distribution = self.distribution(parameters)
        return dict(zip(self.points, distribution.atom_weights, strict=True))

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> _PointMassDistribution:
        missing = set(self.parameter_names).difference(parameters)
        extra = set(parameters).difference(self.parameter_names)
        if missing or extra:
            raise ValueError(
                "Point-mass parameters do not match the family: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return _PointMassDistribution(self, parameters, validate_args=True)

    def log_prob(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        self.validate_response(response)
        return self.distribution(parameters).log_prob(response)

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        return self.distribution(parameters).cdf(response)

    def cdf_left(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        return self.distribution(parameters).cdf_left(response)

    def _differentiable_cdf(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        return self.cdf(response, parameters)

    def _quantile(
        self,
        probabilities: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        return self.distribution(parameters).icdf(probabilities)

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
        """Return the score-outer-product working approximation."""
        scores = self.score(response, parameters)
        result = {}
        for index, left in enumerate(self.parameter_names):
            for right in self.parameter_names[index:]:
                value = -scores[left] * scores[right]
                if left == right:
                    value = torch.minimum(
                        value,
                        response.new_full(response.shape, -1e-15),
                    )
                result[(left, right)] = value
        return result

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        super().validate_response(response, context=context)
        if self.family.is_discrete:
            self.family.validate_response(response, context=context)
            return

        atom = torch.zeros_like(response, dtype=torch.bool)
        for point in self.points:
            atom = atom | (response == point)
        if bool((~atom).any()):
            self.family.validate_response(response[~atom], context=context)

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        defaults: dict[str, Tensor] = {}
        base_parameters = parameters.intersection(self.family.parameter_names)
        if base_parameters:
            base_response = response
            if not self.family.is_discrete:
                atom = torch.zeros_like(response, dtype=torch.bool)
                for point in self.points:
                    atom = atom | (response == point)
                if bool(atom.any()):
                    replacement = (
                        response[~atom].mean()
                        if bool((~atom).any())
                        else response.new_tensor(0.5)
                    )
                    base_response = torch.where(atom, replacement, response)
            defaults.update(
                self.family._default_initial_parameters(
                    base_response,
                    base_parameters,
                )
            )
        for parameter in self.mass_parameter_names:
            if parameter in parameters:
                defaults[parameter] = torch.full_like(response, 0.1)
        return defaults
