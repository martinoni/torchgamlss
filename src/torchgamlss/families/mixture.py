"""Differentiable finite mixtures of GAMLSS response families."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Size, Tensor
from torch.distributions import Categorical, Distribution

from torchgamlss.families.base import Family
from torchgamlss.links import IdentityLink, Link


@dataclass(frozen=True)
class MixtureDiagnostics:
    """Observation- and component-level summaries of a finite mixture."""

    prior_probabilities: Tensor
    posterior_probabilities: Tensor
    classification: Tensor
    effective_counts: Tensor
    effective_proportions: Tensor
    entropy: Tensor
    mean_entropy: Tensor
    mean_max_posterior: Tensor


class _FiniteMixtureDistribution(Distribution):
    """Scalar-event distribution adapter for heterogeneous finite mixtures."""

    arg_constraints: dict[str, Any] = {}
    has_rsample = False

    def __init__(
        self,
        family: FiniteMixture,
        parameters: Mapping[str, Tensor],
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.family = family
        self.parameters = family._validated_parameters(parameters)
        self.component_parameters = family._component_parameters(self.parameters)
        self.component_distributions = tuple(
            component.distribution(component_parameters)
            for component, component_parameters in zip(
                family.components,
                self.component_parameters,
                strict=True,
            )
        )
        self.logits = family._component_log_weights(self.parameters)
        self.probs = self.logits.exp()
        reference = self.parameters[family.parameter_names[0]]
        batch_shape = reference.shape
        for component_distribution in self.component_distributions:
            if component_distribution.event_shape:
                raise ValueError(
                    "finite mixtures currently require scalar-event families"
                )
            if component_distribution.batch_shape != batch_shape:
                raise ValueError(
                    "component distributions must share one broadcast batch shape"
                )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=torch.Size(),
            validate_args=validate_args,
        )

    def component_log_probabilities(self, value: Tensor) -> Tensor:
        """Return one unweighted log density or log mass per component."""
        return torch.stack(
            [
                component_distribution.log_prob(value)
                for component_distribution in self.component_distributions
            ],
            dim=-1,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        component_log_probabilities = self.component_log_probabilities(value)
        return torch.logsumexp(
            self.logits + component_log_probabilities,
            dim=-1,
        )

    def cdf(self, value: Tensor) -> Tensor:
        component_cdfs = torch.stack(
            [
                component._differentiable_cdf(value, component_parameters)
                for component, component_parameters in zip(
                    self.family.components,
                    self.component_parameters,
                    strict=True,
                )
            ],
            dim=-1,
        )
        return (self.probs * component_cdfs).sum(dim=-1).clamp(0.0, 1.0)

    def icdf(self, value: Tensor) -> Tensor:
        return self.family._quantile(value, self.parameters)

    @property
    def mean(self) -> Tensor:
        component_means = torch.stack(
            [
                component_distribution.mean
                for component_distribution in self.component_distributions
            ],
            dim=-1,
        )
        return (self.probs * component_means).sum(dim=-1)

    @property
    def variance(self) -> Tensor:
        component_means = torch.stack(
            [
                component_distribution.mean
                for component_distribution in self.component_distributions
            ],
            dim=-1,
        )
        component_variances = torch.stack(
            [
                component_distribution.variance
                for component_distribution in self.component_distributions
            ],
            dim=-1,
        )
        mixture_mean = (self.probs * component_means).sum(dim=-1, keepdim=True)
        return (
            self.probs
            * (component_variances + (component_means - mixture_mean).square())
        ).sum(dim=-1)

    @torch.no_grad()
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        component_samples = torch.stack(
            [
                component_distribution.sample(sample_shape)
                for component_distribution in self.component_distributions
            ],
            dim=-1,
        )
        component_indices = Categorical(probs=self.probs).sample(sample_shape)
        return component_samples.gather(
            dim=-1,
            index=component_indices.unsqueeze(-1),
        ).squeeze(-1)


class FiniteMixture(Family):
    """Compose two or more scalar response families into a finite mixture.

    The first ``K - 1`` mixing parameters are reference-category log-odds
    against component ``K``, named ``mixing_1`` through ``mixing_{K-1}``.
    Call :meth:`component_weights` to obtain normalized probabilities.

    Component parameters are named ``component_1_mu``, ``component_2_mu``,
    and so on. Names listed in ``shared_parameters`` instead use one common
    predictor, such as ``sigma``, across every component.
    """

    def __init__(
        self,
        components: Sequence[Family],
        *,
        shared_parameters: Sequence[str] = (),
        ordering_parameter: str | None = None,
    ) -> None:
        if isinstance(components, (str, bytes)) or not isinstance(
            components,
            Sequence,
        ):
            raise TypeError("components must be a sequence of Family instances")
        normalized_components = tuple(components)
        if len(normalized_components) < 2:
            raise ValueError("a finite mixture requires at least two components")
        if any(
            not isinstance(component, Family)
            for component in normalized_components
        ):
            raise TypeError("every mixture component must be a Family instance")
        if len({component.is_discrete for component in normalized_components}) != 1:
            raise ValueError(
                "mixture components must be either all continuous or all discrete"
            )

        try:
            normalized_shared = tuple(shared_parameters)
        except TypeError as error:
            raise TypeError(
                "shared_parameters must be a sequence of parameter names"
            ) from error
        if (
            any(
                not isinstance(parameter, str) or not parameter
                for parameter in normalized_shared
            )
            or len(set(normalized_shared)) != len(normalized_shared)
        ):
            raise ValueError(
                "shared_parameters must contain distinct non-empty names"
            )

        common_parameters = set(normalized_components[0].parameter_names)
        for component in normalized_components[1:]:
            common_parameters.intersection_update(component.parameter_names)
        unknown_shared = set(normalized_shared).difference(common_parameters)
        if unknown_shared:
            raise ValueError(
                "shared parameters must exist in every component: "
                f"{sorted(unknown_shared)}"
            )
        for parameter in normalized_shared:
            component_links = [
                component.links[parameter] for component in normalized_components
            ]
            first_link = component_links[0]
            if any(
                type(link) is not type(first_link) or link.name != first_link.name
                for link in component_links[1:]
            ):
                raise ValueError(
                    f"shared parameter {parameter!r} must use the same link "
                    "in every component"
                )

        if ordering_parameter is None:
            ordering_parameter = (
                "mu"
                if "mu" in common_parameters and "mu" not in normalized_shared
                else None
            )
        elif (
            ordering_parameter not in common_parameters
            or ordering_parameter in normalized_shared
        ):
            raise ValueError(
                "ordering_parameter must be an unshared parameter present "
                "in every component"
            )

        self.components = normalized_components
        self.shared_parameters = normalized_shared
        self.ordering_parameter = ordering_parameter
        self.is_discrete = normalized_components[0].is_discrete
        self.name = "MX[" + ",".join(
            component.name for component in normalized_components
        ) + "]"

        parameter_maps: list[dict[str, str]] = []
        parameter_names: list[str] = []
        links: dict[str, Link] = {}
        for component_index, component in enumerate(normalized_components, start=1):
            component_map = {}
            for parameter in component.parameter_names:
                public_name = (
                    parameter
                    if parameter in normalized_shared
                    else f"component_{component_index}_{parameter}"
                )
                component_map[parameter] = public_name
                if public_name not in links:
                    parameter_names.append(public_name)
                    links[public_name] = component.links[parameter]
            parameter_maps.append(component_map)

        mixing_names = tuple(
            f"mixing_{component_index}"
            for component_index in range(1, len(normalized_components))
        )
        for mixing_name in mixing_names:
            parameter_names.append(mixing_name)
            links[mixing_name] = IdentityLink()

        self.parameter_names = tuple(parameter_names)
        self._links = links
        self._parameter_maps = tuple(parameter_maps)
        self.mixing_parameter_names = mixing_names

    @property
    def links(self) -> Mapping[str, Link]:
        return self._links

    @property
    def component_count(self) -> int:
        """Return the number of mixture components."""
        return len(self.components)

    def distribution(
        self,
        parameters: Mapping[str, Tensor],
    ) -> _FiniteMixtureDistribution:
        return _FiniteMixtureDistribution(
            self,
            parameters,
            validate_args=True,
        )

    def component_parameters(
        self,
        parameters: Mapping[str, Tensor],
    ) -> tuple[dict[str, Tensor], ...]:
        """Return normalized parameters grouped by component family."""
        validated = self._validated_parameters(parameters)
        return self._component_parameters(validated)

    def component_weights(self, parameters: Mapping[str, Tensor]) -> Tensor:
        """Return normalized prior component probabilities on the final axis."""
        validated = self._validated_parameters(parameters)
        return self._component_weights(validated)

    def component_log_weights(self, parameters: Mapping[str, Tensor]) -> Tensor:
        """Return normalized log prior probabilities on the final axis."""
        validated = self._validated_parameters(parameters)
        return self._component_log_weights(validated)

    def component_log_probabilities(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        """Return one unweighted log density or log mass per component."""
        self.validate_response(response)
        return self.distribution(parameters).component_log_probabilities(response)

    def _quantile(
        self,
        probabilities: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        if self.is_discrete:
            raise NotImplementedError(
                "finite-mixture quantiles currently require continuous components"
            )
        validated = self._validated_parameters(parameters)
        grouped = self._component_parameters(validated)
        component_quantiles = torch.stack(
            [
                component._quantile(probabilities, component_parameters)
                for component, component_parameters in zip(
                    self.components,
                    grouped,
                    strict=True,
                )
            ],
            dim=-1,
        )
        lower = component_quantiles.min(dim=-1).values
        upper = component_quantiles.max(dim=-1).values
        for _ in range(64):
            midpoint = (lower + upper) / 2.0
            midpoint_cdf = self._differentiable_cdf(midpoint, validated)
            lower = torch.where(midpoint_cdf < probabilities, midpoint, lower)
            upper = torch.where(midpoint_cdf >= probabilities, midpoint, upper)
        return (lower + upper) / 2.0

    def posterior_probabilities(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        """Return posterior component probabilities for every observation."""
        distribution = self.distribution(parameters)
        component_log_probabilities = distribution.component_log_probabilities(
            response
        )
        log_joint = distribution.logits + component_log_probabilities
        log_normalizer = torch.logsumexp(log_joint, dim=-1, keepdim=True)
        if not torch.isfinite(log_normalizer).all():
            raise FloatingPointError(
                "mixture likelihood is zero or non-finite for at least one response"
            )
        return torch.exp(log_joint - log_normalizer)

    def diagnostics(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
    ) -> MixtureDiagnostics:
        """Summarize prior weights, posterior separation, and effective counts."""
        posterior = self.posterior_probabilities(response, parameters)
        prior = self.component_weights(parameters)
        if prior.ndim == 1:
            prior = prior.expand(response.shape + (self.component_count,))
        else:
            prior = torch.broadcast_to(
                prior,
                response.shape + (self.component_count,),
            )
        if weights is None:
            case_weights = torch.ones_like(response)
        else:
            if not isinstance(weights, Tensor):
                raise ValueError("mixture diagnostic weights must be a tensor")
            try:
                case_weights = torch.broadcast_to(weights, response.shape)
            except RuntimeError as error:
                raise ValueError(
                    "mixture diagnostic weights are not broadcastable to the response"
                ) from error
            if (
                case_weights.dtype != response.dtype
                or case_weights.device != response.device
                or not torch.isfinite(case_weights).all()
                or bool((case_weights < 0).any())
                or not bool(case_weights.sum() > 0)
            ):
                raise ValueError(
                    "mixture diagnostic weights must be finite, non-negative, "
                    "positive in total, and match the response dtype and device"
                )
        total_weight = case_weights.sum()
        effective_counts = (posterior * case_weights.unsqueeze(-1)).sum(dim=0)
        entropy = -(
            posterior * posterior.clamp_min(torch.finfo(posterior.dtype).tiny).log()
        ).sum(dim=-1)
        return MixtureDiagnostics(
            prior_probabilities=prior,
            posterior_probabilities=posterior,
            classification=posterior.argmax(dim=-1),
            effective_counts=effective_counts,
            effective_proportions=effective_counts / total_weight,
            entropy=entropy,
            mean_entropy=(entropy * case_weights).sum() / total_weight,
            mean_max_posterior=(
                posterior.max(dim=-1).values * case_weights
            ).sum()
            / total_weight,
        )

    def score(
        self,
        response: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Return exact first derivatives of the mixture log-likelihood."""
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
        raise NotImplementedError(
            "finite mixtures do not have the component-family RS/CG expected "
            "derivatives; use Torch joint optimization or the mixture EM fitter"
        )

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        super().validate_response(response, context=context)
        for component in self.components:
            component.validate_response(response, context=context)

    def component_order(
        self,
        parameters: Mapping[str, Tensor],
    ) -> tuple[int, ...]:
        """Return the stable zero-based component order used for labels."""
        self._require_exchangeable_components()
        if self.ordering_parameter is None:
            raise ValueError(
                "component ordering requires an unshared ordering_parameter"
            )
        validated = self._validated_parameters(parameters)
        grouped = self._component_parameters(validated)
        ordering_values = torch.stack(
            [
                component_parameters[self.ordering_parameter].detach().mean()
                for component_parameters in grouped
            ]
        )
        return tuple(
            int(index)
            for index in torch.argsort(ordering_values, stable=True).cpu().tolist()
        )

    def canonicalize_parameters(
        self,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Relabel exchangeable components by the configured ordering parameter."""
        validated = self._validated_parameters(parameters)
        order = self.component_order(validated)
        result: dict[str, Tensor] = {
            parameter: validated[parameter]
            for parameter in self.shared_parameters
        }
        for target_index, source_index in enumerate(order):
            source_map = self._parameter_maps[source_index]
            target_map = self._parameter_maps[target_index]
            for local_parameter, target_parameter in target_map.items():
                if local_parameter not in self.shared_parameters:
                    result[target_parameter] = validated[
                        source_map[local_parameter]
                    ]
        ordered_weights = self._component_weights(validated)[..., list(order)]
        reference_weight = ordered_weights[..., -1]
        for index, mixing_parameter in enumerate(self.mixing_parameter_names):
            result[mixing_parameter] = torch.log(
                ordered_weights[..., index] / reference_weight
            )
        return {
            parameter: result[parameter] for parameter in self.parameter_names
        }

    def _validated_parameters(
        self,
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        if not isinstance(parameters, Mapping):
            raise ValueError("mixture parameters must be supplied as a mapping")
        missing = set(self.parameter_names).difference(parameters)
        extra = set(parameters).difference(self.parameter_names)
        if missing or extra:
            raise ValueError(
                "Mixture parameters do not match the configured predictors: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        values = [parameters[parameter] for parameter in self.parameter_names]
        reference = values[0]
        if not isinstance(reference, Tensor) or not reference.is_floating_point():
            raise ValueError("mixture parameters must be floating-point tensors")
        if any(
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not torch.isfinite(value).all()
            for value in values
        ):
            raise ValueError(
                "mixture parameters must be finite floating-point tensors "
                "with one common dtype and device"
            )
        try:
            broadcast_values = torch.broadcast_tensors(*values)
        except RuntimeError as error:
            raise ValueError(
                "mixture parameters cannot be broadcast together"
            ) from error
        validated = dict(
            zip(self.parameter_names, broadcast_values, strict=True)
        )
        return validated

    def _component_parameters(
        self,
        parameters: Mapping[str, Tensor],
    ) -> tuple[dict[str, Tensor], ...]:
        return tuple(
            {
                local_parameter: parameters[public_parameter]
                for local_parameter, public_parameter in parameter_map.items()
            }
            for parameter_map in self._parameter_maps
        )

    def _component_weights(
        self,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        return self._component_log_weights(parameters).exp()

    def _component_log_weights(
        self,
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        reference = parameters[self.parameter_names[0]]
        log_unnormalized = torch.stack(
            [
                *[
                    parameters[parameter]
                    for parameter in self.mixing_parameter_names
                ],
                torch.zeros_like(reference),
            ],
            dim=-1,
        )
        return torch.log_softmax(log_unnormalized, dim=-1)

    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        if response.numel() < self.component_count:
            raise ValueError(
                "finite-mixture initialization requires at least one response "
                "per component"
            )
        defaults: dict[str, Tensor] = {}
        for mixing_parameter in self.mixing_parameter_names:
            if mixing_parameter in parameters:
                defaults[mixing_parameter] = torch.zeros_like(response)

        for shared_parameter in self.shared_parameters:
            if shared_parameter not in parameters:
                continue
            component = self.components[0]
            value = component._default_initial_parameters(
                response,
                {shared_parameter},
            )[shared_parameter]
            defaults[shared_parameter] = self._constant_start(
                component,
                shared_parameter,
                value,
                response,
            )

        sorted_indices = torch.argsort(response, stable=True)
        component_indices = torch.tensor_split(
            sorted_indices,
            self.component_count,
        )
        for component, parameter_map, indices in zip(
            self.components,
            self._parameter_maps,
            component_indices,
            strict=True,
        ):
            component_response = response[indices]
            for local_parameter, public_parameter in parameter_map.items():
                if (
                    public_parameter not in parameters
                    or local_parameter in self.shared_parameters
                ):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    candidate = component._default_initial_parameters(
                        component_response,
                        {local_parameter},
                    )[local_parameter]
                try:
                    defaults[public_parameter] = self._constant_start(
                        component,
                        local_parameter,
                        candidate,
                        response,
                    )
                except ValueError:
                    fallback = component._default_initial_parameters(
                        response,
                        {local_parameter},
                    )[local_parameter]
                    defaults[public_parameter] = self._constant_start(
                        component,
                        local_parameter,
                        fallback,
                        response,
                    )
        return defaults

    @staticmethod
    def _constant_start(
        component: Family,
        parameter: str,
        values: Tensor,
        response: Tensor,
    ) -> Tensor:
        value = torch.as_tensor(
            values,
            dtype=response.dtype,
            device=response.device,
        ).median()
        if not torch.isfinite(value):
            raise ValueError("component start is not finite")
        expanded = value.expand_as(response)
        if not torch.isfinite(component.links[parameter](expanded)).all():
            raise ValueError("component start is outside its link domain")
        return expanded

    def _require_exchangeable_components(self) -> None:
        first = self.components[0]
        first_signature = (
            type(first),
            first.name,
            first.parameter_names,
            tuple(type(first.links[name]) for name in first.parameter_names),
        )
        for component in self.components[1:]:
            signature = (
                type(component),
                component.name,
                component.parameter_names,
                tuple(
                    type(component.links[name])
                    for name in component.parameter_names
                ),
            )
            if signature != first_signature:
                raise ValueError(
                    "automatic label ordering requires exchangeable components "
                    "with the same family and links"
                )


MX = FiniteMixture
