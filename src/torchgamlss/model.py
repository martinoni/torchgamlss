"""Core differentiable GAMLSS model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from torchgamlss.diagnostics import (
    ModelDiagnostics,
    model_diagnostics,
    quantile_residuals,
)
from torchgamlss.families import Family
from torchgamlss.fitting import (
    CGControl,
    CGFitResult,
    RSControl,
    RSFitResult,
    fit_cg,
    fit_rs,
)
from torchgamlss.formula import FormulaData, FormulaEncoder
from torchgamlss.inference import (
    InferenceResult,
    SmoothBootstrapResult,
    SmoothInferenceResult,
    SmoothJointBootstrapResult,
    coefficient_inference,
    smooth_term_bootstrap,
    smooth_term_inference,
)
from torchgamlss.smooths import PSpline, SmoothTerm


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
        device: torch.device | str | None = None,
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
                parameter: nn.Parameter(torch.zeros(size, dtype=dtype, device=device))
                for parameter, size in design_sizes.items()
            }
        )
        model_device = next(iter(self.coefficients.values())).device
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
                term.coefficients.dtype != dtype
                or term.coefficients.device != model_device
                for term in parameter_terms.values()
            ):
                raise ValueError("Smooth terms must match the model dtype and device")
            self.smooth_terms[parameter] = nn.ModuleDict(dict(parameter_terms))
        self._formula_encoder: FormulaEncoder | None = None

    @classmethod
    def from_formula(
        cls,
        family: Family,
        formulas: Mapping[str, str],
        data: Any,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> GAMLSS:
        """Construct a model and fitted tabular encodings from formulas."""
        encoder = FormulaEncoder.fit(family.parameter_names, formulas, data)
        prepared = encoder.transform(
            data,
            dtype=dtype,
            device=device,
            include_response=True,
        )
        smooth_terms = {
            parameter: {
                spec.name: PSpline.from_data(
                    prepared.smooth_covariates[parameter][spec.name],
                    **dict(spec.options),
                )
                for spec in encoder.smooth_specs[parameter]
            }
            for parameter in family.parameter_names
        }
        model = cls(
            family,
            {
                parameter: design.shape[1]
                for parameter, design in prepared.design_matrices.items()
            },
            smooth_terms=smooth_terms,
            dtype=dtype,
            device=device,
        )
        model._formula_encoder = encoder
        return model

    @property
    def formula_column_names(self) -> Mapping[str, tuple[str, ...]]:
        """Return fitted design-matrix column names for a formula model."""
        return dict(self._require_formula_encoder().design_columns)

    @property
    def formula_response_name(self) -> str:
        """Return the untransformed response column used by a formula model."""
        return self._require_formula_encoder().response_name

    def prepare_formula_data(
        self,
        data: Any,
        *,
        include_response: bool = False,
    ) -> FormulaData:
        """Materialize stored formulas into model-compatible tensors."""
        model_parameter = next(self.parameters())
        return self._require_formula_encoder().transform(
            data,
            dtype=model_parameter.dtype,
            device=model_parameter.device,
            include_response=include_response,
        )

    def fit_rs_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        initial_parameters: Mapping[str, Any] | None = None,
        control: RSControl | None = None,
    ) -> RSFitResult:
        """Fit a formula model from tabular data with RS cycles."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        parameter_starts = self._formula_initial_parameters(data, initial_parameters)
        return self.fit_rs(
            prepared.response,
            prepared.design_matrices,
            weights=case_weights,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            initial_parameters=parameter_starts,
            control=control,
        )

    def fit_cg_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        initial_parameters: Mapping[str, Any] | None = None,
        control: CGControl | None = None,
    ) -> CGFitResult:
        """Fit a linear or additive formula model from tabular data with CG cycles."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        parameter_starts = self._formula_initial_parameters(data, initial_parameters)
        return self.fit_cg(
            prepared.response,
            prepared.design_matrices,
            weights=case_weights,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            initial_parameters=parameter_starts,
            control=control,
        )

    def fit_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        max_iter: int = 100,
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
    ) -> FitResult:
        """Fit a formula model from tabular data with Torch L-BFGS."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        return self.fit(
            prepared.response,
            prepared.design_matrices,
            weights=case_weights,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
        )

    def predict_data(
        self,
        data: Any,
        *,
        type: Literal["link", "response", "terms"] = "response",
    ) -> dict[str, Tensor] | dict[str, TermContributions]:
        """Predict from tabular data using the fitted formula encodings."""
        prepared = self.prepare_formula_data(data)
        return self.predict(
            prepared.design_matrices,
            prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            type=type,
        )

    def inference_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        conditional_on_smooths: bool = False,
        confidence_level: float = 0.95,
        degrees_of_freedom: float | None = None,
    ) -> InferenceResult:
        """Infer formula-model linear coefficients from the fitted state."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        return self.inference(
            prepared.response,
            prepared.design_matrices,
            weights=case_weights,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            conditional_on_smooths=conditional_on_smooths,
            confidence_level=confidence_level,
            degrees_of_freedom=degrees_of_freedom,
        )

    def smooth_inference_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        new_data: Any = None,
        confidence_level: float = 0.95,
    ) -> dict[str, dict[str, SmoothInferenceResult]]:
        """Infer fitted smooth contributions, conditional on their lambdas."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        evaluation_covariates = (
            prepared.smooth_covariates
            if new_data is None
            else self.prepare_formula_data(new_data).smooth_covariates
        )
        return self.smooth_inference(
            prepared.response,
            prepared.design_matrices,
            weights=case_weights,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            evaluation_smooth_covariates=evaluation_covariates,
            confidence_level=confidence_level,
        )

    def smooth_bootstrap_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        new_data: Any = None,
        replicates: int = 999,
        max_attempts: int | None = None,
        algorithm: Literal["rs", "cg"] = "rs",
        control: RSControl | CGControl | None = None,
        confidence_level: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> dict[str, dict[str, SmoothBootstrapResult]]:
        """Bootstrap fitted smooths while repeating lambda selection."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        evaluation_covariates = (
            prepared.smooth_covariates
            if new_data is None
            else self.prepare_formula_data(new_data).smooth_covariates
        )
        return self.smooth_bootstrap(
            prepared.response,
            prepared.design_matrices,
            smooth_covariates=prepared.smooth_covariates,
            weights=case_weights,
            offsets=prepared.offsets,
            evaluation_smooth_covariates=evaluation_covariates,
            replicates=replicates,
            max_attempts=max_attempts,
            algorithm=algorithm,
            control=control,
            confidence_level=confidence_level,
            generator=generator,
        )

    def smooth_joint_bootstrap_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        new_data: Any = None,
        replicates: int = 999,
        max_attempts: int | None = None,
        algorithm: Literal["rs", "cg"] = "rs",
        control: RSControl | CGControl | None = None,
        confidence_level: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> SmoothJointBootstrapResult:
        """Bootstrap all fitted smooths in one aligned joint result."""
        curves = self.smooth_bootstrap_data(
            data,
            weights=weights,
            new_data=new_data,
            replicates=replicates,
            max_attempts=max_attempts,
            algorithm=algorithm,
            control=control,
            confidence_level=confidence_level,
            generator=generator,
        )
        return SmoothJointBootstrapResult._from_curves(curves)

    def diagnostics_data(
        self,
        data: Any,
        *,
        weights: Any = None,
        degrees_of_freedom: float | None = None,
    ) -> ModelDiagnostics:
        """Evaluate model-selection diagnostics from formula data."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        case_weights = self._formula_tensor(data, weights, context="weights")
        return self.diagnostics(
            prepared.response,
            prepared.design_matrices,
            weights=case_weights,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            degrees_of_freedom=degrees_of_freedom,
        )

    def quantile_residuals_data(
        self,
        data: Any,
        *,
        uniforms: Any = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return quantile residuals using stored formula encodings."""
        prepared = self.prepare_formula_data(data, include_response=True)
        assert prepared.response is not None
        uniform_tensor = self._formula_tensor(
            data,
            uniforms,
            context="uniforms",
        )
        return self.quantile_residuals(
            prepared.response,
            prepared.design_matrices,
            offsets=prepared.offsets,
            smooth_covariates=prepared.smooth_covariates,
            uniforms=uniform_tensor,
            generator=generator,
        )

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

        This Torch-native optimizer is intended as a numerical baseline. The
        classical GAMLSS RS and CG algorithms are implemented separately.
        """
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if any(
            term.estimates_smoothing_parameter
            for terms in self.smooth_terms.values()
            for term in terms.values()
        ):
            raise ValueError(
                "automatic smoothing-parameter estimation requires fit_rs() or fit_cg()"
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
        initial_parameters: Mapping[str, Any] | None = None,
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
            initial_parameters=initial_parameters,
            control=control,
        )

    def fit_cg(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        initial_parameters: Mapping[str, Any] | None = None,
        control: CGControl | None = None,
    ) -> CGFitResult:
        """Fit linear or additive predictors with Cole-Green joint cycles."""
        return fit_cg(
            self,
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            initial_parameters=initial_parameters,
            control=control,
        )

    def inference(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        conditional_on_smooths: bool = False,
        confidence_level: float = 0.95,
        degrees_of_freedom: float | None = None,
    ) -> InferenceResult:
        """Return full-Hessian covariance and t-based Wald inference."""
        return coefficient_inference(
            self,
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            conditional_on_smooths=conditional_on_smooths,
            confidence_level=confidence_level,
            degrees_of_freedom=degrees_of_freedom,
        )

    def smooth_inference(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]],
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        evaluation_smooth_covariates: (
            Mapping[str, Mapping[str, Tensor]] | None
        ) = None,
        confidence_level: float = 0.95,
    ) -> dict[str, dict[str, SmoothInferenceResult]]:
        """Infer smooth curves conditional on fitted smoothing parameters."""
        return smooth_term_inference(
            self,
            response,
            design_matrices,
            smooth_covariates=smooth_covariates,
            weights=weights,
            offsets=offsets,
            evaluation_smooth_covariates=evaluation_smooth_covariates,
            confidence_level=confidence_level,
        )

    def smooth_bootstrap(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]],
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        evaluation_smooth_covariates: (
            Mapping[str, Mapping[str, Tensor]] | None
        ) = None,
        replicates: int = 999,
        max_attempts: int | None = None,
        algorithm: Literal["rs", "cg"] = "rs",
        control: RSControl | CGControl | None = None,
        confidence_level: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> dict[str, dict[str, SmoothBootstrapResult]]:
        """Bootstrap smooth curves while repeating classical fitting."""
        return smooth_term_bootstrap(
            self,
            response,
            design_matrices,
            smooth_covariates=smooth_covariates,
            weights=weights,
            offsets=offsets,
            evaluation_smooth_covariates=evaluation_smooth_covariates,
            replicates=replicates,
            max_attempts=max_attempts,
            algorithm=algorithm,
            control=control,
            confidence_level=confidence_level,
            generator=generator,
        )

    def smooth_joint_bootstrap(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]],
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        evaluation_smooth_covariates: (
            Mapping[str, Mapping[str, Tensor]] | None
        ) = None,
        replicates: int = 999,
        max_attempts: int | None = None,
        algorithm: Literal["rs", "cg"] = "rs",
        control: RSControl | CGControl | None = None,
        confidence_level: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> SmoothJointBootstrapResult:
        """Bootstrap all smooth curves with replicate alignment preserved."""
        curves = self.smooth_bootstrap(
            response,
            design_matrices,
            smooth_covariates=smooth_covariates,
            weights=weights,
            offsets=offsets,
            evaluation_smooth_covariates=evaluation_smooth_covariates,
            replicates=replicates,
            max_attempts=max_attempts,
            algorithm=algorithm,
            control=control,
            confidence_level=confidence_level,
            generator=generator,
        )
        return SmoothJointBootstrapResult._from_curves(curves)

    def diagnostics(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        weights: Tensor | None = None,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        degrees_of_freedom: float | None = None,
    ) -> ModelDiagnostics:
        """Return deviance and information criteria for the current model."""
        return model_diagnostics(
            self,
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            degrees_of_freedom=degrees_of_freedom,
        )

    def quantile_residuals(
        self,
        response: Tensor,
        design_matrices: Mapping[str, Tensor],
        *,
        offsets: Mapping[str, Tensor] | None = None,
        smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None = None,
        uniforms: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return normal quantile residuals for the current fitted model."""
        return quantile_residuals(
            self,
            response,
            design_matrices,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            uniforms=uniforms,
            generator=generator,
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

    def _require_formula_encoder(self) -> FormulaEncoder:
        if self._formula_encoder is None:
            raise RuntimeError(
                "This operation requires a model constructed with from_formula()"
            )
        return self._formula_encoder

    def _formula_tensor(
        self,
        data: Any,
        value: Any,
        *,
        context: str,
    ) -> Tensor | None:
        model_parameter = next(self.parameters())
        return self._require_formula_encoder().tensor(
            data,
            value,
            dtype=model_parameter.dtype,
            device=model_parameter.device,
            context=context,
        )

    def _formula_initial_parameters(
        self,
        data: Any,
        values: Mapping[str, Any] | None,
    ) -> Mapping[str, Tensor] | None:
        if values is None:
            return None
        if not isinstance(values, Mapping):
            raise ValueError("initial parameters must be supplied as a mapping")
        extra = set(values).difference(self.family.parameter_names)
        if extra:
            raise ValueError(
                f"Initial parameters contain unknown names: {sorted(extra)}"
            )
        return {
            parameter: self._formula_tensor(
                data,
                value,
                context=f"initial parameter {parameter!r}",
            )
            for parameter, value in values.items()
        }

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
