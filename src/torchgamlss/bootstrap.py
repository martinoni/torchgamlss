"""Shared parametric-bootstrap refit helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, TypeAlias

from torch import Tensor

from torchgamlss.fitting import CGControl, CGFitResult, RSControl, RSFitResult
from torchgamlss.laml import (
    GAMLSSLAMLResult,
    LAMLControl,
    NormalLAMLResult,
    fit_gamlss_model_laml,
)

if TYPE_CHECKING:
    from torchgamlss.model import GAMLSS


BootstrapAlgorithm: TypeAlias = Literal["rs", "cg", "laml"]
BootstrapControl: TypeAlias = RSControl | CGControl | LAMLControl
BootstrapFitResult: TypeAlias = (
    RSFitResult | CGFitResult | NormalLAMLResult | GAMLSSLAMLResult
)


def validate_bootstrap_refit(
    model: GAMLSS,
    algorithm: str,
    control: BootstrapControl | None,
) -> BootstrapAlgorithm:
    """Validate a bootstrap refit algorithm and its numerical controls."""
    if algorithm not in {"rs", "cg", "laml"}:
        raise ValueError("algorithm must be 'rs', 'cg', or 'laml'")

    normalized: BootstrapAlgorithm = algorithm
    expected_control = {
        "rs": RSControl,
        "cg": CGControl,
        "laml": LAMLControl,
    }[normalized]
    if control is not None and not isinstance(control, expected_control):
        raise ValueError(
            f"control must be {expected_control.__name__} when algorithm={normalized!r}"
        )

    if normalized == "laml":
        from torchgamlss.families import (
            Beta,
            BoxCoxColeGreen,
            BoxCoxT,
            Gamma,
            NegativeBinomial,
            Normal,
            Poisson,
            StudentT,
        )
        from torchgamlss.links import IdentityLink, LogitLink, LogLink

        is_normal = isinstance(model.family, Normal)
        is_poisson = isinstance(model.family, Poisson)
        is_nbi = isinstance(model.family, NegativeBinomial)
        is_gamma = isinstance(model.family, Gamma)
        is_beta = isinstance(model.family, Beta)
        is_student_t = isinstance(model.family, StudentT)
        is_bccg = isinstance(model.family, BoxCoxColeGreen)
        is_bct = isinstance(model.family, BoxCoxT)
        if not (
            is_normal
            or is_poisson
            or is_nbi
            or is_gamma
            or is_beta
            or is_student_t
            or is_bccg
            or is_bct
        ):
            raise ValueError(
                "LAML bootstrap currently supports Normal, Poisson, NBI, "
                "Gamma, Beta, Student-t, BCCG, and BCT families"
            )
        if is_normal and (
            not isinstance(model.family.links["mu"], IdentityLink)
            or not isinstance(model.family.links["sigma"], LogLink)
        ):
            raise ValueError(
                "Normal LAML bootstrap requires identity mu and log sigma links"
            )
        if is_poisson and not isinstance(model.family.links["mu"], LogLink):
            raise ValueError("Poisson LAML bootstrap requires a log mu link")
        if is_nbi and (
            not isinstance(model.family.links["mu"], LogLink)
            or not isinstance(model.family.links["sigma"], LogLink)
        ):
            raise ValueError(
                "NBI LAML bootstrap requires log mu and log sigma links"
            )
        if is_gamma and (
            not isinstance(model.family.links["mu"], LogLink)
            or not isinstance(model.family.links["sigma"], LogLink)
        ):
            raise ValueError("Gamma LAML bootstrap requires log mu and log sigma links")
        if is_beta and (
            not isinstance(model.family.links["mu"], LogitLink)
            or not isinstance(model.family.links["sigma"], LogitLink)
        ):
            raise ValueError(
                "Beta LAML bootstrap requires logit mu and logit sigma links"
            )
        if is_student_t and (
            not isinstance(model.family.links["mu"], IdentityLink)
            or not isinstance(model.family.links["sigma"], LogLink)
            or not isinstance(model.family.links["nu"], LogLink)
        ):
            raise ValueError(
                "Student-t LAML bootstrap requires identity mu, log sigma, "
                "and log nu links"
            )
        if is_bccg and (
            not isinstance(model.family.links["mu"], IdentityLink)
            or not isinstance(model.family.links["sigma"], LogLink)
            or not isinstance(model.family.links["nu"], IdentityLink)
        ):
            raise ValueError(
                "BCCG LAML bootstrap requires identity mu, log sigma, "
                "and identity nu links"
            )
        if is_bct and (
            not isinstance(model.family.links["mu"], IdentityLink)
            or not isinstance(model.family.links["sigma"], LogLink)
            or not isinstance(model.family.links["nu"], IdentityLink)
            or not isinstance(model.family.links["tau"], LogLink)
        ):
            raise ValueError(
                "BCT LAML bootstrap requires identity mu, log sigma, "
                "identity nu, and log tau links"
            )
        if not any(model.smooth_terms.values()):
            raise ValueError("LAML bootstrap requires at least one smooth term")
        if model.neural_predictors or model.shared_predictor is not None:
            raise ValueError(
                "LAML bootstrap does not support neural or shared predictors"
            )
    return normalized


def fit_bootstrap_model(
    model: GAMLSS,
    response: Tensor,
    design_matrices: Mapping[str, Tensor],
    *,
    weights: Tensor,
    offsets: Mapping[str, Tensor] | None,
    smooth_covariates: Mapping[str, Mapping[str, Tensor]] | None,
    initial_parameters: Mapping[str, Tensor],
    algorithm: BootstrapAlgorithm,
    control: BootstrapControl | None,
) -> BootstrapFitResult:
    """Refit one bootstrap response with the selected whole-model estimator."""
    if algorithm == "rs":
        return model.fit_rs(
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            initial_parameters=initial_parameters,
            control=control,
        )
    if algorithm == "cg":
        return model.fit_cg(
            response,
            design_matrices,
            weights=weights,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
            initial_parameters=initial_parameters,
            control=control,
        )
    if smooth_covariates is None:
        raise ValueError(
            "LAML bootstrap requires smooth_covariates for every smooth term"
        )
    return fit_gamlss_model_laml(
        model,
        response,
        design_matrices,
        weights=weights,
        offsets=offsets,
        smooth_covariates=smooth_covariates,
        control=control,
        warm_start=True,
    )


def bootstrap_fit_converged(
    result: BootstrapFitResult,
    algorithm: BootstrapAlgorithm,
) -> bool:
    """Return the convergence state shared by classical and LAML fits."""
    if algorithm == "laml":
        assert isinstance(result, (NormalLAMLResult, GAMLSSLAMLResult))
        return result.outer_converged and result.inner_converged
    return result.converged
