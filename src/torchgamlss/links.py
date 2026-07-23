"""Link functions for distribution parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Link(ABC):
    """Map a constrained distribution parameter to an unconstrained predictor."""

    name: str

    @abstractmethod
    def forward(self, parameter: Tensor) -> Tensor:
        """Transform a distribution parameter to the predictor scale."""

    @abstractmethod
    def inverse(self, predictor: Tensor) -> Tensor:
        """Transform a predictor to the distribution-parameter scale."""

    def __call__(self, parameter: Tensor) -> Tensor:
        return self.forward(parameter)


class IdentityLink(Link):
    """Identity link for parameters with real support."""

    name = "identity"

    def forward(self, parameter: Tensor) -> Tensor:
        return parameter

    def inverse(self, predictor: Tensor) -> Tensor:
        return predictor


class LogLink(Link):
    """Log link for strictly positive parameters."""

    name = "log"

    def forward(self, parameter: Tensor) -> Tensor:
        return torch.log(parameter)

    def inverse(self, predictor: Tensor) -> Tensor:
        return torch.exp(predictor)


class LogitLink(Link):
    """Logit link for parameters in the open unit interval."""

    name = "logit"

    def forward(self, parameter: Tensor) -> Tensor:
        return torch.logit(parameter)

    def inverse(self, predictor: Tensor) -> Tensor:
        return torch.sigmoid(predictor)
