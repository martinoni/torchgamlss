"""Fit and export the TorchGAMLSS side of the Normal location-scale example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import torchgamlss
from torchgamlss import GAMLSS, Normal, RSControl

PROBABILITIES = (0.03, 0.50, 0.97)


def run(data_path: Path, output_dir: Path) -> None:
    """Fit the example and write standardized parity artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )
    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
        ),
    )
    diagnostics = model.diagnostics_data(
        data,
        weights="weight",
        degrees_of_freedom=result.effective_degrees_of_freedom,
    )
    fit = pd.DataFrame(
        {
            "converged": [result.converged],
            "outer_iterations": [result.outer_iterations],
            "global_deviance": [result.global_deviance],
            "negative_log_likelihood": [result.negative_log_likelihood],
            "effective_degrees_of_freedom": [
                result.effective_degrees_of_freedom
            ],
            "effective_observation_count": [
                diagnostics.effective_observation_count
            ],
            "aic": [diagnostics.aic],
        }
    )
    coefficient_rows = []
    for parameter in model.family.parameter_names:
        names = model.formula_column_names[parameter]
        estimates = model.coefficients[parameter].detach().cpu().tolist()
        coefficient_rows.extend(
            {
                "parameter": parameter,
                "term": name,
                "estimate": estimate,
            }
            for name, estimate in zip(names, estimates, strict=True)
        )
    coefficients = pd.DataFrame(coefficient_rows)

    parameters = model.predict_data(data)
    fitted = pd.DataFrame(
        {
            "observation": np.arange(len(data)),
            "mu": parameters["mu"].detach().cpu().numpy(),
            "sigma": parameters["sigma"].detach().cpu().numpy(),
        }
    )
    quantile_prediction = model.predict_quantiles_data(
        data,
        probabilities=PROBABILITIES,
    )
    quantiles = quantile_prediction.to_dataframe()[
        ["observation", "probability", "centile", "quantile"]
    ]
    residuals = pd.DataFrame(
        {
            "observation": np.arange(len(data)),
            "quantile_residual": model.quantile_residuals_data(data)
            .detach()
            .cpu()
            .numpy(),
        }
    )
    metadata = pd.DataFrame(
        {
            "case": ["normal_location_scale_rs"],
            "implementation": ["TorchGAMLSS"],
            "family": ["NO"],
            "algorithm": ["RS"],
            "torchgamlss_version": [torchgamlss.__version__],
            "torch_version": [torch.__version__],
        }
    )

    _write_csv(fit, output_dir / "fit.csv")
    _write_csv(coefficients, output_dir / "coefficients.csv")
    _write_csv(fitted, output_dir / "fitted.csv")
    _write_csv(quantiles, output_dir / "quantiles.csv")
    _write_csv(residuals, output_dir / "residuals.csv")
    _write_csv(metadata, output_dir / "metadata.csv")
    _plot_example(data, fitted, quantile_prediction.quantiles, residuals, output_dir)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _plot_example(
    data: pd.DataFrame,
    fitted: pd.DataFrame,
    quantiles: torch.Tensor,
    residuals: pd.DataFrame,
    output_dir: Path,
) -> None:
    order = np.argsort(data["x"].to_numpy())
    quantile_values = quantiles.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(data["x"], data["y"], color="black", label="observed")
    axes[0].plot(
        data["x"].to_numpy()[order],
        fitted["mu"].to_numpy()[order],
        color="tab:blue",
        label="fitted mu",
    )
    axes[0].fill_between(
        data["x"].to_numpy()[order],
        quantile_values[order, 0],
        quantile_values[order, 2],
        color="tab:blue",
        alpha=0.18,
        label="3%-97%",
    )
    axes[0].set(xlabel="x", ylabel="y", title="Location")
    axes[0].legend()

    axes[1].scatter(data["z"], fitted["sigma"], color="tab:orange")
    axes[1].set(xlabel="z", ylabel="fitted sigma", title="Scale")

    axes[2].axhline(0.0, color="0.5", linewidth=1)
    axes[2].scatter(
        residuals["observation"],
        residuals["quantile_residual"],
        color="tab:green",
    )
    axes[2].set(
        xlabel="observation",
        ylabel="quantile residual",
        title="Residuals",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "location_scale_fit.png", dpi=140)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(arguments.data, arguments.output_dir)
