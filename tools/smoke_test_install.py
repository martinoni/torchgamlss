"""End-to-end smoke test executed against an installed wheel."""

from __future__ import annotations

import importlib.metadata

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torchgamlss
from torchgamlss import (
    BCCG,
    BCPE,
    BCT,
    GAMLSS,
    GG,
    IG,
    LOGNO,
    PE,
    TF,
    WEI,
    Beta,
    CensoredFamily,
    CensoredResponse,
    GAMLSSLAMLResult,
    Gamma,
    GeneralizedGamma,
    InverseGaussian,
    LAMLControl,
    LogNormal,
    NegativeBinomial,
    Normal,
    NormalLAMLResult,
    PenalizedLeastSquaresResult,
    PowerExponential,
    PSpline,
    StudentT,
    TensorProductSmooth,
    TruncatedFamily,
    Weibull,
    solve_penalized_least_squares,
)


def main() -> None:
    """Exercise the installed metadata and representative public APIs."""
    installed_version = importlib.metadata.version("torchgamlss")
    if torchgamlss.__version__ != installed_version:
        raise RuntimeError(
            f"runtime version {torchgamlss.__version__!r} does not match "
            f"installed metadata {installed_version!r}"
        )
    if TF is not StudentT or TF().name != "TF":
        raise RuntimeError("installed Student-t family exports are invalid")
    if PE is not PowerExponential or PE().name != "PE":
        raise RuntimeError("installed power-exponential family exports are invalid")
    if WEI is not Weibull or LOGNO is not LogNormal:
        raise RuntimeError("installed survival-family exports are invalid")
    if IG is not InverseGaussian or GG is not GeneralizedGamma:
        raise RuntimeError("installed extended survival-family exports are invalid")

    penalty_fit = solve_penalized_least_squares(
        torch.eye(2, dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        (torch.eye(2, dtype=torch.float64),),
        (2.0,),
        constraints=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
    )
    if not isinstance(penalty_fit, PenalizedLeastSquaresResult):
        raise RuntimeError("installed generic penalty solver result is invalid")
    if not torch.isfinite(penalty_fit.coefficients).all():
        raise RuntimeError("installed generic penalty solver is non-finite")
    if not torch.allclose(
        penalty_fit.coefficients[:1],
        penalty_fit.coefficients[1:],
    ):
        raise RuntimeError("installed generic penalty constraints are invalid")

    tensor_covariates = torch.tensor(
        [
            [-1.0, 0.0],
            [-0.2, 0.5],
            [0.4, 1.2],
            [1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    tensor_term = TensorProductSmooth(
        (
            PSpline(-1.0, 1.0, 2.0, intervals=2, dtype=torch.float64),
            PSpline(0.0, 2.0, 3.0, intervals=2, dtype=torch.float64),
        )
    )
    if tensor_term.design(tensor_covariates).shape[0] != len(tensor_covariates):
        raise RuntimeError("installed tensor-product design is invalid")
    if len(tensor_term.penalty_matrices()) != 2:
        raise RuntimeError("installed tensor-product penalties are invalid")
    if tensor_term.constraints(tensor_covariates).shape[0] != 1:
        raise RuntimeError("installed tensor-product constraint is invalid")

    data = pd.DataFrame(
        {
            "x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "z": [0.2, 0.8, -0.4, 1.1, -0.7, 0.5],
            "y": [-2.4, -0.7, 1.4, 2.7, 5.2, 6.5],
        }
    )
    tensor_observation_count = 30
    tensor_x = torch.linspace(
        -1.0,
        1.0,
        tensor_observation_count,
        dtype=torch.float64,
    )
    tensor_z = torch.sin(
        torch.linspace(
            0.15,
            5.6,
            tensor_observation_count,
            dtype=torch.float64,
        )
    )
    tensor_y = (
        0.7
        + torch.sin(2.0 * tensor_x)
        + 0.5 * tensor_x * tensor_z
        + 0.2 * tensor_z.square()
        + 0.05
        * torch.cos(
            torch.arange(tensor_observation_count, dtype=torch.float64)
        )
    )
    tensor_formula_frame = pd.DataFrame(
        {
            "x": tensor_x.numpy(),
            "z": tensor_z.numpy(),
            "y": tensor_y.numpy(),
        }
    )
    tensor_formula_model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ te(x, z, lambda_=(2, 3), intervals=(2, 2), "
                "name='surface')"
            ),
            "sigma": "~ 1",
        },
        tensor_formula_frame,
    )
    tensor_formula_fit = tensor_formula_model.fit_rs_data(
        tensor_formula_frame
    )
    tensor_formula_data = tensor_formula_model.prepare_formula_data(
        tensor_formula_frame
    )
    tensor_formula_covariates = tensor_formula_data.smooth_covariates["mu"][
        "surface"
    ]
    tensor_formula_term = tensor_formula_model.smooth_terms["mu"]["surface"]
    if tensor_formula_model.formula_column_names["mu"] != ("Intercept",):
        raise RuntimeError("installed te() formula design is invalid")
    if tensor_formula_covariates.shape != (tensor_observation_count, 2):
        raise RuntimeError("installed te() formula covariates are invalid")
    if not torch.allclose(
        tensor_formula_term.design(tensor_formula_covariates).sum(dim=0),
        torch.zeros_like(tensor_formula_term.coefficients),
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("installed te() formula constraint is invalid")
    if not tensor_formula_fit.converged:
        raise RuntimeError("installed te() RS fit did not converge")
    if tensor_formula_fit.smoothing_parameters["mu"]["surface"] != (
        2.0,
        3.0,
    ):
        raise RuntimeError("installed te() RS smoothing parameters are invalid")
    tensor_formula_inference = tensor_formula_model.smooth_inference_data(
        tensor_formula_frame
    )["mu"]["surface"]
    if tensor_formula_inference.smoothing_parameter != (2.0, 3.0):
        raise RuntimeError("installed te() inference lambdas are invalid")
    if not torch.isfinite(tensor_formula_inference.standard_errors).all():
        raise RuntimeError("installed te() inference is non-finite")
    if tensor_formula_inference.to_dataframe().columns[:2].tolist() != [
        "covariate_0",
        "covariate_1",
    ]:
        raise RuntimeError("installed te() inference table is invalid")
    tensor_formula_bootstrap = (
        tensor_formula_model.smooth_joint_bootstrap_data(
            tensor_formula_frame,
            replicates=10,
            generator=torch.Generator().manual_seed(2026),
        )
    )
    tensor_surface_bootstrap = tensor_formula_bootstrap["mu"]["surface"]
    if tensor_surface_bootstrap.bootstrap_smoothing_parameters.shape != (
        10,
        2,
    ):
        raise RuntimeError("installed te() bootstrap lambda shape is invalid")
    if tensor_formula_bootstrap.smoothing_parameter_labels != (
        ("mu", "surface", 0),
        ("mu", "surface", 1),
    ):
        raise RuntimeError("installed te() bootstrap lambda labels are invalid")
    if not torch.allclose(
        tensor_surface_bootstrap.bootstrap_smoothing_parameters,
        torch.tensor([[2.0, 3.0]] * 10, dtype=torch.float64),
    ):
        raise RuntimeError("installed te() bootstrap lambdas are invalid")
    if tensor_surface_bootstrap.to_dataframe().columns[:2].tolist() != [
        "covariate_0",
        "covariate_1",
    ]:
        raise RuntimeError("installed te() bootstrap table is invalid")
    tensor_formula_prediction = tensor_formula_model.predict_data(
        tensor_formula_frame
    )
    if not torch.isfinite(tensor_formula_prediction["mu"]).all():
        raise RuntimeError("installed te() formula prediction is non-finite")
    tensor_laml_model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ te(x, z, intervals=(2, 2), name='surface')"
            ),
            "sigma": "~ 1",
        },
        tensor_formula_frame,
    )
    tensor_laml_fit = tensor_laml_model.fit_laml_data(
        tensor_formula_frame,
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=5e-5,
        ),
    )
    if not isinstance(tensor_laml_fit, NormalLAMLResult):
        raise RuntimeError("installed formula LAML result is invalid")
    if not tensor_laml_fit.outer_converged:
        raise RuntimeError("installed te() LAML fit did not converge")
    if tensor_laml_fit.smoothing_parameter_labels != (
        ("mu", "surface", 0),
        ("mu", "surface", 1),
    ):
        raise RuntimeError("installed te() LAML lambda labels are invalid")
    if not torch.allclose(
        tensor_laml_fit.smoothing_parameters,
        tensor_laml_fit.smoothing_parameters.new_tensor(
            tensor_laml_model.smooth_terms["mu"]["surface"].smoothing_parameters
        ),
    ):
        raise RuntimeError("installed te() LAML state update is invalid")

    beta_x = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
    beta_mu = torch.sigmoid(-0.2 + 0.9 * torch.sin(torch.pi * beta_x))
    beta_sigma = torch.full_like(beta_mu, 0.28)
    beta_y = Beta().sample(
        {"mu": beta_mu, "sigma": beta_sigma},
        generator=torch.Generator().manual_seed(20260803),
    )
    beta_frame = pd.DataFrame({"x": beta_x.numpy(), "y": beta_y.numpy()})
    beta_laml_model = GAMLSS.from_formula(
        Beta(),
        {"mu": "y ~ pb(x, intervals=4)", "sigma": "~ 1"},
        beta_frame,
    )
    beta_laml_fit = beta_laml_model.fit_laml_data(
        beta_frame,
        control=LAMLControl(
            inner_gradient_tolerance=1e-6,
            outer_max_iterations=25,
            outer_gradient_tolerance=5e-4,
        ),
    )
    if not isinstance(beta_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed Beta LAML result is invalid")
    if not beta_laml_fit.outer_converged:
        raise RuntimeError("installed Beta LAML fit did not converge")
    if not torch.isfinite(beta_laml_fit.outer_hessian).all():
        raise RuntimeError("installed Beta LAML Hessian is non-finite")

    nbi_x = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
    nbi_mu = torch.exp(0.3 + 0.8 * torch.sin(torch.pi * nbi_x))
    nbi_sigma = torch.full_like(nbi_mu, 0.3)
    nbi_y = NegativeBinomial().sample(
        {"mu": nbi_mu, "sigma": nbi_sigma},
        generator=torch.Generator().manual_seed(20260804),
    )
    nbi_frame = pd.DataFrame({"x": nbi_x.numpy(), "y": nbi_y.numpy()})
    nbi_laml_model = GAMLSS.from_formula(
        NegativeBinomial(),
        {"mu": "y ~ pb(x, intervals=4)", "sigma": "~ 1"},
        nbi_frame,
    )
    nbi_laml_fit = nbi_laml_model.fit_laml_data(
        nbi_frame,
        control=LAMLControl(
            inner_gradient_tolerance=1e-6,
            outer_max_iterations=25,
            outer_gradient_tolerance=5e-4,
        ),
    )
    if not isinstance(nbi_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed NBI LAML result is invalid")
    if not nbi_laml_fit.outer_converged:
        raise RuntimeError("installed NBI LAML fit did not converge")
    if not torch.isfinite(nbi_laml_fit.outer_hessian).all():
        raise RuntimeError("installed NBI LAML Hessian is non-finite")

    tf_x = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
    tf_mu = 0.3 + 0.9 * torch.sin(torch.pi * tf_x)
    tf_sigma = torch.full_like(tf_mu, 0.5)
    tf_nu = torch.full_like(tf_mu, 7.0)
    tf_y = StudentT().sample(
        {"mu": tf_mu, "sigma": tf_sigma, "nu": tf_nu},
        generator=torch.Generator().manual_seed(20260805),
    )
    tf_frame = pd.DataFrame({"x": tf_x.numpy(), "y": tf_y.numpy()})
    tf_laml_model = GAMLSS.from_formula(
        StudentT(),
        {
            "mu": "y ~ pb(x, intervals=4)",
            "sigma": "~ 1",
            "nu": "~ 1",
        },
        tf_frame,
    )
    tf_laml_fit = tf_laml_model.fit_laml_data(
        tf_frame,
        control=LAMLControl(
            inner_gradient_tolerance=1e-6,
            outer_max_iterations=25,
            outer_gradient_tolerance=5e-4,
        ),
    )
    if not isinstance(tf_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed Student-t LAML result is invalid")
    if not tf_laml_fit.outer_converged:
        raise RuntimeError("installed Student-t LAML fit did not converge")
    if not torch.isfinite(tf_laml_fit.outer_hessian).all():
        raise RuntimeError("installed Student-t LAML Hessian is non-finite")

    bccg_x = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
    bccg_mu = 3.0 + 0.5 * torch.sin(torch.pi * bccg_x)
    bccg_sigma = torch.full_like(bccg_mu, 0.18)
    bccg_nu = torch.full_like(bccg_mu, 0.35)
    bccg_y = BCCG().sample(
        {"mu": bccg_mu, "sigma": bccg_sigma, "nu": bccg_nu},
        generator=torch.Generator().manual_seed(20260806),
    )
    bccg_frame = pd.DataFrame({"x": bccg_x.numpy(), "y": bccg_y.numpy()})
    bccg_laml_model = GAMLSS.from_formula(
        BCCG(),
        {
            "mu": "y ~ pb(x, intervals=4)",
            "sigma": "~ 1",
            "nu": "~ 1",
        },
        bccg_frame,
    )
    bccg_laml_fit = bccg_laml_model.fit_laml_data(
        bccg_frame,
        control=LAMLControl(
            inner_gradient_tolerance=1e-6,
            outer_max_iterations=30,
            outer_gradient_tolerance=5e-4,
        ),
    )
    if not isinstance(bccg_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed BCCG LAML result is invalid")
    if not bccg_laml_fit.outer_converged:
        raise RuntimeError("installed BCCG LAML fit did not converge")
    if not torch.isfinite(bccg_laml_fit.outer_hessian).all():
        raise RuntimeError("installed BCCG LAML Hessian is non-finite")

    bct_x = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    bct_mu = 3.0 + 0.45 * torch.sin(torch.pi * bct_x) + 0.1 * bct_x
    bct_sigma = torch.full_like(bct_mu, 0.18)
    bct_nu = torch.full_like(bct_mu, 0.3)
    bct_tau = torch.full_like(bct_mu, 4.0)
    bct_probability = (
        torch.arange(80, dtype=torch.float64).mul(37).remainder(80) + 0.5
    ) / 80
    bct_y = BCT().distribution(
        {"mu": bct_mu, "sigma": bct_sigma, "nu": bct_nu, "tau": bct_tau}
    ).icdf(bct_probability)
    bct_frame = pd.DataFrame({"x": bct_x.numpy(), "y": bct_y.numpy()})
    bct_laml_model = GAMLSS.from_formula(
        BCT(),
        {
            "mu": "y ~ pb(x, intervals=3, lambda_=10)",
            "sigma": "~ 1",
            "nu": "~ 1",
            "tau": "~ 1",
        },
        bct_frame,
    )
    bct_rs_fit = bct_laml_model.fit_rs_data(bct_frame)
    bct_laml_fit = bct_laml_model.fit_laml_data(bct_frame, warm_start=True)
    if not bct_rs_fit.converged:
        raise RuntimeError("installed BCT RS warm start did not converge")
    if not isinstance(bct_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed BCT LAML result is invalid")
    if not bct_laml_fit.outer_converged:
        raise RuntimeError("installed BCT LAML fit did not converge")
    if not torch.isfinite(bct_laml_fit.outer_hessian).all():
        raise RuntimeError("installed BCT LAML Hessian is non-finite")

    bcpe_x = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    bcpe_mu = 3.0 + 0.45 * torch.sin(torch.pi * bcpe_x) + 0.1 * bcpe_x
    bcpe_sigma = torch.full_like(bcpe_mu, 0.18)
    bcpe_nu = torch.full_like(bcpe_mu, 0.3)
    bcpe_tau = torch.full_like(bcpe_mu, 1.5)
    bcpe_probability = (
        torch.arange(80, dtype=torch.float64).mul(37).remainder(80) + 0.5
    ) / 80
    bcpe_y = BCPE().distribution(
        {
            "mu": bcpe_mu,
            "sigma": bcpe_sigma,
            "nu": bcpe_nu,
            "tau": bcpe_tau,
        }
    ).icdf(bcpe_probability)
    bcpe_frame = pd.DataFrame({"x": bcpe_x.numpy(), "y": bcpe_y.numpy()})
    bcpe_laml_model = GAMLSS.from_formula(
        BCPE(),
        {
            "mu": "y ~ pb(x, intervals=3, lambda_=10)",
            "sigma": "~ 1",
            "nu": "~ 1",
            "tau": "~ 1",
        },
        bcpe_frame,
    )
    bcpe_rs_fit = bcpe_laml_model.fit_rs_data(bcpe_frame)
    bcpe_laml_fit = bcpe_laml_model.fit_laml_data(bcpe_frame, warm_start=True)
    if not bcpe_rs_fit.converged:
        raise RuntimeError("installed BCPE RS warm start did not converge")
    if not isinstance(bcpe_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed BCPE LAML result is invalid")
    if not bcpe_laml_fit.outer_converged:
        raise RuntimeError("installed BCPE LAML fit did not converge")
    if not torch.isfinite(bcpe_laml_fit.outer_hessian).all():
        raise RuntimeError("installed BCPE LAML Hessian is non-finite")

    pe_x = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    pe_mu = 1.5 + 0.45 * torch.sin(torch.pi * pe_x) + 0.1 * pe_x
    pe_sigma = torch.full_like(pe_mu, 0.7)
    pe_nu = torch.full_like(pe_mu, 1.6)
    pe_probability = (
        torch.arange(80, dtype=torch.float64).mul(37).remainder(80) + 0.5
    ) / 80
    pe_y = PE().distribution(
        {"mu": pe_mu, "sigma": pe_sigma, "nu": pe_nu}
    ).icdf(pe_probability)
    pe_frame = pd.DataFrame({"x": pe_x.numpy(), "y": pe_y.numpy()})
    pe_laml_model = GAMLSS.from_formula(
        PE(),
        {
            "mu": "y ~ pb(x, intervals=3, lambda_=10)",
            "sigma": "~ 1",
            "nu": "~ 1",
        },
        pe_frame,
    )
    pe_rs_fit = pe_laml_model.fit_rs_data(pe_frame)
    pe_laml_fit = pe_laml_model.fit_laml_data(pe_frame, warm_start=True)
    if not pe_rs_fit.converged:
        raise RuntimeError("installed PE RS warm start did not converge")
    if not isinstance(pe_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed PE LAML result is invalid")
    if not pe_laml_fit.outer_converged:
        raise RuntimeError("installed PE LAML fit did not converge")
    if not torch.isfinite(pe_laml_fit.outer_hessian).all():
        raise RuntimeError("installed PE LAML Hessian is non-finite")

    gg_x = torch.linspace(-1.0, 1.0, 80, dtype=torch.float64)
    gg_mu = torch.exp(0.5 + 0.3 * torch.sin(torch.pi * gg_x) + 0.1 * gg_x)
    gg_sigma = torch.full_like(gg_mu, 0.58)
    gg_nu = torch.full_like(gg_mu, 0.8)
    gg_probability = (
        torch.arange(80, dtype=torch.float64).mul(37).remainder(80) + 0.5
    ) / 80
    gg_y = GG().quantile(
        gg_probability,
        {"mu": gg_mu, "sigma": gg_sigma, "nu": gg_nu},
    ).diagonal()
    gg_frame = pd.DataFrame({"x": gg_x.numpy(), "y": gg_y.numpy()})
    gg_laml_model = GAMLSS.from_formula(
        GG(),
        {
            "mu": "y ~ pb(x, intervals=3, lambda_=10)",
            "sigma": "~ 1",
            "nu": "~ 1",
        },
        gg_frame,
    )
    gg_rs_fit = gg_laml_model.fit_rs_data(gg_frame)
    gg_laml_fit = gg_laml_model.fit_laml_data(gg_frame, warm_start=True)
    if not gg_rs_fit.converged:
        raise RuntimeError("installed GG RS warm start did not converge")
    if not isinstance(gg_laml_fit, GAMLSSLAMLResult):
        raise RuntimeError("installed GG LAML result is invalid")
    if not gg_laml_fit.outer_converged:
        raise RuntimeError("installed GG LAML fit did not converge")
    if not torch.isfinite(gg_laml_fit.outer_hessian).all():
        raise RuntimeError("installed GG LAML Hessian is non-finite")

    bootstrap_x = torch.linspace(-1.0, 1.0, 40, dtype=torch.float64)
    bootstrap_generator = torch.Generator().manual_seed(44)
    bootstrap_y = (
        0.4
        + torch.sin(2.5 * bootstrap_x)
        + 0.16
        * torch.randn(
            bootstrap_x.numel(),
            dtype=torch.float64,
            generator=bootstrap_generator,
        )
    )
    laml_bootstrap_frame = pd.DataFrame(
        {"x": bootstrap_x.numpy(), "y": bootstrap_y.numpy()}
    )
    laml_bootstrap_model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x, intervals=4)", "sigma": "~ 1"},
        laml_bootstrap_frame,
    )
    laml_bootstrap_control = LAMLControl(
        outer_max_iterations=25,
        outer_gradient_tolerance=1e-4,
    )
    laml_bootstrap_model.fit_laml_data(
        laml_bootstrap_frame,
        control=laml_bootstrap_control,
    )
    laml_bootstrap = laml_bootstrap_model.smooth_bootstrap_data(
        laml_bootstrap_frame,
        new_data=laml_bootstrap_frame.iloc[::10].drop(columns="y"),
        replicates=10,
        max_attempts=20,
        algorithm="laml",
        control=laml_bootstrap_control,
        generator=torch.Generator().manual_seed(99),
    )["mu"]["x"]
    if laml_bootstrap.algorithm != "laml":
        raise RuntimeError("installed LAML bootstrap algorithm is invalid")
    if laml_bootstrap.bootstrap_smoothing_parameters.std() <= 0:
        raise RuntimeError("installed LAML bootstrap did not reselect lambda")
    if not torch.isfinite(laml_bootstrap.bootstrap_estimates).all():
        raise RuntimeError("installed LAML bootstrap estimates are non-finite")

    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )
    fit = model.fit_rs_data(data)
    if not fit.converged:
        raise RuntimeError("wheel smoke-test model did not converge")

    parameters = model.predict_data(data)
    if set(parameters) != {"mu", "sigma"}:
        raise RuntimeError("wheel smoke-test prediction returned wrong parameters")
    if not all(torch.isfinite(value).all() for value in parameters.values()):
        raise RuntimeError("wheel smoke-test prediction is non-finite")

    varying = TruncatedFamily(
        Normal(),
        lower=torch.tensor([-1.0, -0.5, 0.0]),
        upper=torch.tensor([1.0, 1.5, 2.0]),
    )
    varying_parameters = {
        "mu": torch.tensor([0.0, 0.5, 1.0]),
        "sigma": torch.tensor([1.0, 1.0, 1.0]),
    }
    varying_quantiles = varying.quantile(
        torch.tensor([0.25, 0.5, 0.75]),
        varying_parameters,
    )
    if varying_quantiles.shape != (3, 3):
        raise RuntimeError("installed varying-truncation quantiles have wrong shape")
    if not torch.isfinite(varying_quantiles).all():
        raise RuntimeError("installed varying-truncation quantiles are non-finite")

    gamma_truncation = TruncatedFamily(Gamma(), lower=0.2, upper=4.0)
    gamma_predictors = {
        "mu": torch.tensor([0.0, 0.5], requires_grad=True),
        "sigma": torch.tensor([-0.8, -0.3], requires_grad=True),
    }
    gamma_parameters = gamma_truncation.parameters_from_predictors(gamma_predictors)
    gamma_loss = -gamma_truncation.log_prob(
        torch.tensor([0.8, 2.0]),
        gamma_parameters,
    ).sum()
    gamma_gradients = torch.autograd.grad(
        gamma_loss,
        tuple(gamma_predictors.values()),
    )
    if not torch.isfinite(gamma_loss) or not all(
        torch.isfinite(gradient).all() for gradient in gamma_gradients
    ):
        raise RuntimeError("installed Gamma truncation gradients are invalid")

    survival_time = torch.tensor([0.8, 1.5, 2.0], dtype=torch.float64)
    event = torch.tensor([1, 0, 1])
    censored = CensoredFamily(
        Weibull(),
        CensoredResponse.right(survival_time, event),
    )
    survival_data = pd.DataFrame(
        {
            "time": survival_time.numpy(),
            "x": [-1.0, 0.0, 1.0],
        }
    )
    survival_model = GAMLSS.from_formula(
        censored,
        {"mu": "time ~ x", "sigma": "~ 1"},
        survival_data,
    )
    censored_loss = survival_model.negative_log_likelihood(
        survival_time,
        survival_model.prepare_formula_data(survival_data).design_matrices,
    )
    curves = survival_model.predict_survival_data(
        survival_data,
        times=[0.5, 1.0, 2.0],
    )
    if not torch.isfinite(censored_loss):
        raise RuntimeError("installed censored likelihood is non-finite")
    if curves.survival.shape != (3, 3):
        raise RuntimeError("installed survival prediction has wrong shape")
    if not torch.isfinite(curves.hazard).all():
        raise RuntimeError("installed survival prediction is non-finite")

    residual_plot = model.plot_data(data)
    worm_plot = model.wp_data(data)
    bucket_plot = model.bp_data(data, bootstrap=False)
    if residual_plot.summary.observation_count != len(data):
        raise RuntimeError("installed plot() API returned an invalid result")
    if worm_plot.residuals.numel() != len(data):
        raise RuntimeError("installed wp() API returned an invalid result")
    if bucket_plot.panels[0].statistics.observation_count != len(data):
        raise RuntimeError("installed bp() API returned an invalid result")
    plt.close(residual_plot.figure)
    plt.close(worm_plot.figure)
    plt.close(bucket_plot.figure)

    print(f"TorchGAMLSS {installed_version} wheel smoke test passed")


if __name__ == "__main__":
    main()
