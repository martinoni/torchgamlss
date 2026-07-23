"""Base interface for GAMLSS response families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import Tensor
from torch.distributions import Distribution

from torchgamlss.links import Link


class Family(ABC):
    """Describe a response distribution and links for all its parameters."""

    name: str
    parameter_names: tuple[str, ...]

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

    @abstractmethod
    def initial_parameters(self, response: Tensor) -> dict[str, Tensor]:
        """Return the family-specific starting values used by R GAMLSS."""
