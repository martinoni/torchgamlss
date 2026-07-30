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
    MX,
    PE,
    TF,
    WEI,
    CensoredFamily,
    CensoredResponse,
    FiniteMixture,
    Gamma,
    GeneralizedGamma,
    InverseGaussian,
    LogNormal,
    Normal,
    PowerExponential,
    StudentT,
    TruncatedFamily,
    Weibull,
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
    if MX is not FiniteMixture:
        raise RuntimeError("installed finite-mixture exports are invalid")

    data = pd.DataFrame(
        {
            "x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "y": [-2.4, -0.7, 1.4, 2.7, 5.2, 6.5],
        }
    )
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

    mixture = FiniteMixture([Normal(), Normal()])
    mixture_response = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    mixture_parameters = {
        "component_1_mu": torch.tensor([-1.0, -1.0], dtype=torch.float64),
        "component_1_sigma": torch.ones(2, dtype=torch.float64),
        "component_2_mu": torch.tensor([2.0, 2.0], dtype=torch.float64),
        "component_2_sigma": torch.ones(2, dtype=torch.float64),
        "mixing_1": torch.zeros(2, dtype=torch.float64),
    }
    mixture_loss = -mixture.log_prob(
        mixture_response,
        mixture_parameters,
    ).sum()
    mixture_posterior = mixture.posterior_probabilities(
        mixture_response,
        mixture_parameters,
    )
    if not torch.isfinite(mixture_loss):
        raise RuntimeError("installed finite-mixture likelihood is non-finite")
    if not torch.allclose(
        mixture_posterior.sum(dim=-1),
        torch.ones_like(mixture_response),
    ):
        raise RuntimeError("installed finite-mixture posterior is invalid")

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
