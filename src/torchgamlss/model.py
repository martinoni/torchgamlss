"""Core differentiable GAMLSS model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from torchgamlss.families import Family
from torchgamlss.fitting import RSControl, RSFitResult, fit_rs
from torchgamlss.smooths import SmoothTerm


@dataclass(frozen=True)
class FitResult:
    """Summary of a full-batch optimization run."""

    negative_log_likelihood: float
    iterations: int
    function_evaluations: int
    gradient_max: float
    converged: bool


@dataclass(frozen=True)
class TermContributions:
    """Additive contributions to one parameter predictor on the link scale."""

    linear: Tensor
    smooth: Mapping[str, Tensor]
    offset: Tensor

    @property
    def total(self) -> Tensor:
        """Reconstruct the complete link-scale predictor."""
        total = self.linear.sum(dim=-1)
        for contribution in self.smooth.values():
            total = total + contribution
        return total + self.offset


class GAMLSS(nn.Module):
    """A minimal multi-parameter distributional regression model.

    The caller supplies one design matrix per distribution parameter and may
    attach named smooth terms. Formula handling remains outside the core model.
    """

    def __init__(
        self,
        family: Family,
        design_sizes: Mapping[str, int],
        *,
        smooth_terms: Mapping[str, Mapping[str, SmoothTerm]] | None = None,
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
        smooth_terms = smooth_terms or {}
        extra_smooth_parameters = set(smooth_terms).difference(expected)
        if extra_smooth_parameters:
            raise ValueError(
                "Smooth terms contain unknown parameters: "
                f"{sorted(extra_smooth_parameters)}"
            )
        self.smooth_terms = nn.ModuleDict()
        for parameter in family.parameter_names:
            parameter_terms = smooth_terms.get(parameter, {})
            if any(not name or "." in name for name in parameter_terms):
                raise ValueError(
                    "Smooth-term names must be non-empty and contain no dots"
                )
            if any(
                term.coefficients.dtype != dtype for term in parameter_terms.values()
            ):
                raise ValueError("Smooth terms must match the model dtype")
            self.smooth_terms[parameter] = nn.ModuleDict(dict(parameter_terms))

    def linear_predictors(
        self,
        design_matrices: Mapping[str, Tensor],
        offsets: Mapping[str, Tensor] | None = None,
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Calculate one linear predictor for each distribution parameter."""
        return {
            parameter: contributions.total
            for parameter, contributions in self.term_contributions(
                design_matrices,
                offsets,
                smooth_covariates=smooth_covariates,
            ).items()
        }

    def term_contributions(
        self,
        design_matrices: Mapping[str, Tensor],
        offsets: Mapping[str, Tensor] | None = None,
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    ) -> dict[str, TermContributions]:
        """Decompose parameter predictors into linear, smooth, and offset terms."""
        expected = set(self.family.parameter_names)
        received = set(design_matrices)
        if expected != received:
            raise ValueError(
                "Design matrices do not match family parameters: "
                f"missing={sorted(expected - received)}, "
                f"extra={sorted(received - expected)}"
            )

        offsets = offsets or {}
        extra_offsets = set(offsets).difference(expected)
        if extra_offsets:
            raise ValueError(
                f"Offsets contain unknown parameters: {sorted(extra_offsets)}"
            )

        contributions = {}
        observation_count: int | None = None
        model_parameter = next(self.parameters())
        for parameter in self.family.parameter_names:
            design_matrix = design_matrices[parameter]
            if (
                design_matrix.ndim != 2
                or design_matrix.shape[1] != self.coefficients[parameter].numel()
            ):
                raise ValueError(
                    f"design matrix for {parameter!r} has an invalid shape"
                )
            if observation_count is None:
                observation_count = design_matrix.shape[0]
            elif design_matrix.shape[0] != observation_count:
                raise ValueError(
                    "design matrices must contain the same number of observations"
                )
            if (
                design_matrix.dtype != model_parameter.dtype
                or design_matrix.device != model_parameter.device
            ):
                raise ValueError(
                    f"design matrix for {parameter!r} must match model dtype and device"
                )
            if not torch.isfinite(design_matrix).all():
                raise ValueError(f"design matrix for {parameter!r} must be finite")

            linear = design_matrix * self.coefficients[parameter]
            linear_total = linear.sum(dim=-1)
            smooth = {}
            for term_name, covariate in self._validated_smooth_covariates(
                parameter, linear_total, smooth_covariates
            ).items():
                smooth[term_name] = self.smooth_terms[parameter][term_name](covariate)
            if parameter in offsets:
                raw_offset = offsets[parameter]
                if (
                    raw_offset.dtype != model_parameter.dtype
                    or raw_offset.device != model_parameter.device
                ):
                    raise ValueError(
                        f"offset for {parameter!r} must match model dtype and device"
                    )
                try:
                    offset = torch.broadcast_to(raw_offset, linear_total.shape)
                except RuntimeError as error:
                    raise ValueError(
                        f"offset for {parameter!r} cannot be broadcast to its predictor"
                    ) from error
                if not torch.isfinite(offset).all():
                    raise ValueError(f"offset for {parameter!r} must be finite")
            else:
                offset = torch.zeros_like(linear_total)
            contributions[parameter] = TermContributions(
                linear=linear,
                smooth=smooth,
                offset=offset,
            )
        return contributions

    def predict(
        self,
        design_matrices: Mapping[str, Tensor],
        offsets: Mapping[str, Tensor] | None = None,
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        type: Literal["link", "response", "terms"] = "response",
    ) -> dict[str, Tensor] | dict[str, TermContributions]:
        """Predict family parameters, link predictors, or additive terms."""
        if type not in {"link", "response", "terms"}:
            raise ValueError("type must be one of: 'link', 'response', 'terms'")
        if type == "terms":
            return self.term_contributions(
                design_matrices,
                offsets,
                smooth_covariates=smooth_covariates,
            )
        predictors = self.linear_predictors(
            design_matrices,
            offsets,
            smooth_covariates=smooth_covariates,
        )
        if type == "link":
            return predictors
        if type == "response":
            return self.family.parameters_from_predictors(predictors)
        raise AssertionError("unreachable prediction type")

    def distribution(
        self,
        design_matrices: Mapping[str, Tensor],
        offsets: Mapping[str, Tensor] | None = None,
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    ) -> Distribution:
        """Build the fitted conditional response distribution."""
        predictors = self.linear_predictors(
            design_matrices, offsets, smooth_covariates=smooth_covariates
        )
        parameters = self.family.parameters_from_predictors(predictors)
        return self.family.distribution(parameters)

    def forward(
        self,
        design_matrices: Mapping[str, Tensor],
        offsets: Mapping[str, Tensor] | None = None,
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
    ) -> Distribution:
        return self.distribution(
            design_matrices, offsets, smooth_covariates=smooth_covariates
        )

    def negative_log_likelihood(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        reduction: str = "sum",
    ) -> Tensor:
        """Return the negative log-likelihood with sum, mean, or no reduction."""
        losses = -self.distribution(
            design_matrices,
            offsets,
            smooth_covariates=smooth_covariates,
        ).log_prob(response)
        observation_weights = self._validated_weights(losses, weights)
        losses = losses * observation_weights
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.sum() / observation_weights.sum()
        if reduction == "none":
            return losses
        raise ValueError("reduction must be one of: 'sum', 'mean', 'none'")

    def fit(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        max_iter: int = 100,
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
    ) -> FitResult:
        """Fit all distribution parameters jointly with full-batch L-BFGS.

        This Torch-native optimizer is the first parametric fitting path. It is
        intended as a numerical baseline; the classical GAMLSS RS path is
        implemented separately and the CG algorithm remains planned work.
        """
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if any(
            term.estimates_smoothing_parameter
            for terms in self.smooth_terms.values()
            for term in terms.values()
        ):
            raise ValueError(
                "automatic smoothing-parameter estimation requires fit_rs()"
            )

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
            loss = self.negative_log_likelihood(
                response,
                design_matrices,
                weights=weights,
                offsets=offsets,
                smooth_covariates=smooth_covariates,
            )
            loss = loss + 0.5 * self.smooth_penalty()
            if not torch.isfinite(loss):
                raise FloatingPointError("negative log-likelihood is not finite")
            loss.backward()
            return loss

        optimizer.step(closure)
        closure()
        final_loss = self.negative_log_likelihood(
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
        ).detach()
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

    def fit_rs(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        control: RSControl | None = None,
    ) -> RSFitResult:
        """Fit linear or additive predictors with Rigby-Stasinopoulos cycles."""
        return fit_rs(
            self,
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            control=control,
        )

    def smooth_penalty(self) -> Tensor:
        """Return the sum of all fixed quadratic smoothness penalties."""
        return sum(
            (
                term.quadratic_penalty()
                for terms in self.smooth_terms.values()
                for term in terms.values()
            ),
            torch.zeros(
                (),
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device,
            ),
        )

    def _validated_smooth_covariates(
        self,
        parameter: str,
        predictor: Tensor,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None,
    ) -> Mapping[str, Tensor]:
        smooth_covariates = smooth_covariates or {}
        extra_parameters = set(smooth_covariates).difference(
            self.family.parameter_names
        )
        if extra_parameters:
            raise ValueError(
                "Smooth covariates contain unknown parameters: "
                f"{sorted(extra_parameters)}"
            )
        supplied = smooth_covariates.get(parameter, {})
        expected_terms = set(self.smooth_terms[parameter])
        supplied_terms = set(supplied)
        if expected_terms != supplied_terms:
            raise ValueError(
                f"Smooth covariates for {parameter!r} do not match configured terms: "
                f"missing={sorted(expected_terms - supplied_terms)}, "
                f"extra={sorted(supplied_terms - expected_terms)}"
            )
        for term_name, covariate in supplied.items():
            if covariate.shape != predictor.shape:
                raise ValueError(
                    f"smooth covariate {term_name!r} for {parameter!r} "
                    "must have one value per predictor"
                )
        return supplied

    @staticmethod
    def _validated_weights(losses: Tensor, weights: Tensor | None) -> Tensor:
        if weights is None:
            return torch.ones_like(losses)
        if weights.device != losses.device:
            raise ValueError("weights must be on the same device as the response")
        try:
            observation_weights = torch.broadcast_to(weights, losses.shape)
        except RuntimeError as error:
            raise ValueError("weights are not broadcastable to the response") from error
        if not torch.isfinite(observation_weights).all():
            raise ValueError("weights must be finite")
        if (observation_weights < 0).any():
            raise ValueError("weights must be non-negative")
        if observation_weights.sum() <= 0:
            raise ValueError("at least one observation weight must be positive")
        return observation_weights
