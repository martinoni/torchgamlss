"""Base interface for GAMLSS response families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Distribution

from torchgamlss.links import Link


class Family(ABC):
    """Describe a response distribution and links for all its parameters."""

    name: str
    parameter_names: tuple[str, ...]
    is_discrete = False

    @property
    @abstractmethod
    def links(self) -> Mapping[str, Link]:
        """Return the link assigned to each distribution parameter."""

    def parameters_from_predictors(
        self, predictors: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Apply inverse links to a complete set of parameter predictors."""
        missing = set(self.parameter_names).difference(predictors)
        extra = set(predictors).difference(self.parameter_names)
        if missing or extra:
            raise ValueError(
                "Predictors do not match family parameters: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        return {
            parameter: self.links[parameter].inverse(predictors[parameter])
            for parameter in self.parameter_names
        }

    @abstractmethod
    def distribution(self, parameters: Mapping[str, Tensor]) -> Distribution:
        """Construct the response distribution from linked parameters."""

    def log_prob(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        """Evaluate the observation-wise log density or log mass."""
        return self.distribution(parameters).log_prob(response)

    def sample(
        self,
        parameters: Mapping[str, Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw one response per parameter row.

        PyTorch distributions use the global random-number generator. When a
        generator is supplied, a seed is drawn from it inside an isolated RNG
        context so sampling is reproducible without changing global RNG state.
        """
        distribution = self.distribution(parameters)
        try:
            if generator is None:
                return distribution.sample()

            generator_device = torch.device(generator.device)
            seed = int(
                torch.randint(
                    torch.iinfo(torch.int64).max,
                    (),
                    dtype=torch.int64,
                    device=generator_device,
                    generator=generator,
                )
            )
            parameter_device = next(iter(parameters.values())).device
            devices = [] if parameter_device.type == "cpu" else [parameter_device]
            with torch.random.fork_rng(
                devices=devices,
                device_type=parameter_device.type,
            ):
                torch.manual_seed(seed)
                return distribution.sample()
        except NotImplementedError as error:
            raise NotImplementedError(
                f"response sampling is not implemented for the {self.name} family"
            ) from error

    def cdf(self, response: Tensor, parameters: Mapping[str, Tensor]) -> Tensor:
        """Evaluate the response CDF for quantile diagnostics."""
        try:
            return self.distribution(parameters).cdf(response)
        except NotImplementedError as error:
            raise NotImplementedError(
                f"CDF is not implemented for the {self.name} family"
            ) from error

    @abstractmethod
    def score(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Return first log-likelihood derivatives on the parameter scale."""

    @abstractmethod
    def expected_second_derivatives(
        self, response: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[tuple[str, str], Tensor]:
        """Return GAMLSS expected second derivatives on the parameter scale."""

    def initial_parameters(
        self,
        response: Tensor,
        values: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Return validated parameter-scale starts, with optional overrides.

        User values may override any subset of the family defaults. Scalars
        are expanded to the response shape; vectors must already have exactly
        one value per observation.
        """
        self.validate_response(response, context="initialization")
        if values is not None and not isinstance(values, Mapping):
            raise ValueError("initial parameters must be supplied as a mapping")
        values = values or {}
        extra = set(values).difference(self.parameter_names)
        if extra:
            raise ValueError(
                f"Initial parameters contain unknown names: {sorted(extra)}"
            )

        missing = set(self.parameter_names).difference(values)
        defaults = (
            self._default_initial_parameters(response, missing) if missing else {}
        )
        if set(defaults) != missing:
            raise RuntimeError(
                "Family default initial parameters do not match the requested names"
            )

        result = {}
        for parameter in self.parameter_names:
            raw_value = (
                values[parameter] if parameter in values else defaults[parameter]
            )
            try:
                value = torch.as_tensor(
                    raw_value,
                    dtype=response.dtype,
                    device=response.device,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"initial parameter {parameter!r} cannot be converted to a tensor"
                ) from error
            if value.ndim == 0:
                value = value.expand_as(response)
            elif value.shape != response.shape:
                raise ValueError(
                    f"initial parameter {parameter!r} must be scalar or have "
                    "one value per observation"
                )
            value = value.detach().clone()
            if not torch.isfinite(value).all():
                raise ValueError(f"initial parameter {parameter!r} must be finite")
            predictor = self.links[parameter](value)
            if not torch.isfinite(predictor).all():
                raise ValueError(
                    f"initial parameter {parameter!r} is outside its link domain"
                )
            result[parameter] = value

        try:
            self.distribution(result)
        except (ValueError, RuntimeError) as error:
            raise ValueError(
                f"initial parameters are invalid for the {self.name} family"
            ) from error
        return result

    def validate_response(
        self,
        response: Tensor,
        *,
        context: str = "family",
    ) -> None:
        """Validate the response support shared by family operations."""
        if (
            response.ndim != 1
            or response.numel() < 1
            or not torch.isfinite(response).all()
        ):
            raise ValueError(
                f"{self.name} {context} requires a non-empty finite "
                "one-dimensional response"
            )

    @abstractmethod
    def _default_initial_parameters(
        self,
        response: Tensor,
        parameters: set[str],
    ) -> dict[str, Tensor]:
        """Return the family-specific default starts used by R GAMLSS."""
