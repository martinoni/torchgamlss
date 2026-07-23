"""Core differentiable GAMLSS model."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from torchgamlss.families import Family


class GAMLSS(nn.Module):
    """A minimal multi-parameter distributional regression model.

    The caller supplies one design matrix per distribution parameter. Formula
    handling and smooth-term construction intentionally live outside this first
    vertical slice.
    """

    def __init__(
        self,
        family: Family,
        design_sizes: Mapping[str, int],
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        expected = set(family.parameter_names)
        received = set(design_sizes)
        if expected != received:
            raise ValueError(
                "Design sizes do not match family parameters: "
                f"missing={sorted(expected - received)}, "
                f"extra={sorted(received - expected)}"
            )
        if any(size < 1 for size in design_sizes.values()):
            raise ValueError("Every design matrix must contain at least one column")

        self.family = family
        self.coefficients = nn.ParameterDict(
            {
                parameter: nn.Parameter(torch.zeros(size, dtype=dtype))
                for parameter, size in design_sizes.items()
            }
        )

    def linear_predictors(
        self, design_matrices: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Calculate one linear predictor for each distribution parameter."""
        expected = set(self.family.parameter_names)
        received = set(design_matrices)
        if expected != received:
            raise ValueError(
                "Design matrices do not match family parameters: "
                f"missing={sorted(expected - received)}, "
                f"extra={sorted(received - expected)}"
            )

        return {
            parameter: design_matrices[parameter] @ self.coefficients[parameter]
            for parameter in self.family.parameter_names
        }

    def distribution(self, design_matrices: Mapping[str, Tensor]) -> Distribution:
        """Build the fitted conditional response distribution."""
        predictors = self.linear_predictors(design_matrices)
        parameters = self.family.parameters_from_predictors(predictors)
        return self.family.distribution(parameters)

    def forward(self, design_matrices: Mapping[str, Tensor]) -> Distribution:
        return self.distribution(design_matrices)

    def negative_log_likelihood(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        reduction: str = "sum",
    ) -> Tensor:
        """Return the negative log-likelihood with sum, mean, or no reduction."""
        losses = -self.distribution(design_matrices).log_prob(response)
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.mean()
        if reduction == "none":
            return losses
        raise ValueError("reduction must be one of: 'sum', 'mean', 'none'")
