"""Core differentiable GAMLSS model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from torchgamlss.families import Family


@dataclass(frozen=True)
class FitResult:
    """Summary of a full-batch optimization run."""

    negative_log_likelihood: float
    iterations: int
    function_evaluations: int
    gradient_max: float
    converged: bool


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

    def fit(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        max_iter: int = 100,
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
    ) -> FitResult:
        """Fit all distribution parameters jointly with full-batch L-BFGS.

        This Torch-native optimizer is the first parametric fitting path. It is
        intended as a numerical baseline; the classical GAMLSS RS and CG
        algorithms remain separate planned implementations.
        """
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        parameters = list(self.parameters())
        optimizer = torch.optim.LBFGS(
            parameters,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(response, design_matrices)
            if not torch.isfinite(loss):
                raise FloatingPointError("negative log-likelihood is not finite")
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = closure().detach()
        gradient_max = max(
            float(parameter.grad.detach().abs().max())
            for parameter in parameters
            if parameter.grad is not None
        )
        state = optimizer.state[parameters[0]]
        iterations = int(state.get("n_iter", 0))
        function_evaluations = int(state.get("func_evals", 0))

        return FitResult(
            negative_log_likelihood=float(final_loss),
            iterations=iterations,
            function_evaluations=function_evaluations,
            gradient_max=gradient_max,
            converged=bool(torch.isfinite(final_loss) and iterations < max_iter),
        )
