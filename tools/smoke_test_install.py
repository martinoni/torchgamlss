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
    GAMLSS,
    GG,
    IG,
    LOGNO,
    PE,
    TF,
    WEI,
    CensoredFamily,
    CensoredResponse,
    Gamma,
    GeneralizedGamma,
    InverseGaussian,
    LAMLControl,
    LogNormal,
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
