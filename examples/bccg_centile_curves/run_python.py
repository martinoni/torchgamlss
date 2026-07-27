"""Fit and export BCCG fetal-growth centile curves with TorchGAMLSS."""

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
from torchgamlss import BCCG, GAMLSS, RSControl

CENTILES = (0.4, 2.0, 9.0, 25.0, 50.0, 75.0, 91.0, 98.0, 99.6)
GRID_SIZE = 121
SMOOTHING_PARAMETER = 10.0
PARAMETERS = ("mu", "sigma", "nu")


def run(data_path: Path, output_dir: Path) -> None:
    """Fit the example and write standardized parity artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    smooth = f"pb(x, smoothing_parameter={SMOOTHING_PARAMETER:g})"
    model = GAMLSS.from_formula(
        BCCG(),
        {
            parameter: f"y ~ {smooth}" if parameter == "mu" else f"~ {smooth}"
            for parameter in PARAMETERS
        },
        data,
    )
    control = RSControl(
        outer_tolerance=1e-8,
        max_outer_iterations=300,
        inner_tolerance=1e-8,
        max_inner_iterations=300,
        backfitting_tolerance=1e-8,
        max_backfitting_iterations=300,
    )
    result = model.fit_rs_data(data, control=control)
    diagnostics = model.diagnostics_data(
        data,
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
            "residual_degrees_of_freedom": [
                diagnostics.residual_degrees_of_freedom
            ],
            "observation_count": [diagnostics.observation_count],
            "effective_observation_count": [
                diagnostics.effective_observation_count
            ],
            "aic": [diagnostics.aic],
        }
    )
    coefficients = _coefficient_table(model)
    smoothing = pd.DataFrame(
        {
            "parameter": PARAMETERS,
            "smoothing_parameter": [
                result.smoothing_parameters[parameter]["x"]
                for parameter in PARAMETERS
            ],
            "effective_degrees_of_freedom": [
                result.smooth_effective_degrees_of_freedom[parameter]["x"]
                for parameter in PARAMETERS
            ],
        }
    )

    parameters = model.predict_data(data)
    fitted = pd.DataFrame(
        {
            "observation": np.arange(len(data)),
            "age": data["x"],
            **{
                parameter: parameters[parameter].detach().cpu().numpy()
                for parameter in PARAMETERS
            },
        }
    )
    grid = pd.DataFrame(
        {
            "x": np.linspace(
                data["x"].min(),
                data["x"].max(),
                GRID_SIZE,
            )
        }
    )
    grid_parameters = model.predict_data(grid)
    centile_prediction = model.predict_centiles_data(
        grid,
        centiles=CENTILES,
    )
    centiles = centile_prediction.to_dataframe().rename(
        columns={"observation": "grid_index"}
    )
    centiles.insert(
        1,
        "age",
        np.repeat(grid["x"].to_numpy(), len(CENTILES)),
    )
    residuals = pd.DataFrame(
        {
            "observation": np.arange(len(data)),
            "age": data["x"],
            "quantile_residual": model.quantile_residuals_data(data)
            .detach()
            .cpu()
            .numpy(),
        }
    )
    metadata = pd.DataFrame(
        {
            "case": ["bccg_centile_curves"],
            "implementation": ["TorchGAMLSS"],
            "family": ["BCCG"],
            "algorithm": ["RS"],
            "grid_size": [GRID_SIZE],
            "torchgamlss_version": [torchgamlss.__version__],
            "torch_version": [torch.__version__],
        }
    )

    _write_csv(fit, output_dir / "fit.csv")
    _write_csv(coefficients, output_dir / "coefficients.csv")
    _write_csv(smoothing, output_dir / "smoothing.csv")
    _write_csv(fitted, output_dir / "fitted.csv")
    _write_csv(centiles, output_dir / "centiles.csv")
    _write_csv(residuals, output_dir / "residuals.csv")
    _write_csv(metadata, output_dir / "metadata.csv")
    _plot_example(data, grid, grid_parameters, centiles, output_dir)


def _coefficient_table(model: GAMLSS) -> pd.DataFrame:
    rows = []
    semantic_terms = ("Intercept", "x_linear")
    for parameter in PARAMETERS:
        estimates = model.coefficients[parameter].detach().cpu().tolist()
        if len(estimates) != len(semantic_terms):
            raise RuntimeError(
                f"unexpected {parameter} coefficient count: {len(estimates)}"
            )
        rows.extend(
            {
                "parameter": parameter,
                "term": term,
                "estimate": estimate,
            }
            for term, estimate in zip(semantic_terms, estimates, strict=True)
        )
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _plot_example(
    data: pd.DataFrame,
    grid: pd.DataFrame,
    parameters: dict[str, torch.Tensor],
    centiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(13, 8.5))
    layout = figure.add_gridspec(2, 3, height_ratios=(1.7, 1.0))
    centile_axis = figure.add_subplot(layout[0, :])
    parameter_axes = [
        figure.add_subplot(layout[1, index]) for index in range(3)
    ]

    centile_axis.scatter(
        data["x"],
        data["y"],
        color="0.45",
        alpha=0.34,
        s=13,
        label="observed",
    )
    for centile in CENTILES:
        curve = centiles[np.isclose(centiles["centile"], centile)]
        emphasized = centile in {0.4, 50.0, 99.6}
        centile_axis.plot(
            curve["age"],
            curve["quantile"],
            color="tab:blue" if centile == 50.0 else "tab:orange",
            linewidth=2.1 if emphasized else 1.0,
            alpha=1.0 if emphasized else 0.78,
            label=f"{centile:g}th" if emphasized else None,
        )
    centile_axis.set(
        xlabel="gestational age (weeks)",
        ylabel="abdominal circumference (mm)",
        title="BCCG fetal-growth centile curves",
    )
    centile_axis.legend(ncol=4)

    colors = {"mu": "tab:blue", "sigma": "tab:green", "nu": "tab:red"}
    labels = {
        "mu": "location μ",
        "sigma": "scale σ",
        "nu": "shape ν",
    }
    for axis, parameter in zip(parameter_axes, PARAMETERS, strict=True):
        axis.plot(
            grid["x"],
            parameters[parameter].detach().cpu().numpy(),
            color=colors[parameter],
            linewidth=2,
        )
        axis.set(
            xlabel="gestational age (weeks)",
            ylabel=parameter,
            title=labels[parameter],
        )

    figure.tight_layout()
    figure.savefig(output_dir / "bccg_centile_curves.png", dpi=140)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(arguments.data, arguments.output_dir)
