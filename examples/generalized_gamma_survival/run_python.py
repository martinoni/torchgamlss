"""Fit and export the TorchGAMLSS side of the censored GG example."""

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
from torchgamlss import (
    GAMLSS,
    CensoredFamily,
    CensoredResponse,
    GeneralizedGamma,
)

PROBABILITIES = (0.1, 0.5, 0.9)
PROFILE_X = (-0.75, 0.0, 0.75)
TIMES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def run(data_path: Path, output_dir: Path) -> None:
    """Fit the right-censored model and write standardized artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    time = torch.tensor(data["time"].to_numpy(), dtype=torch.float64)
    event = torch.tensor(data["event"].to_numpy(), dtype=torch.int64)
    family = CensoredFamily(
        GeneralizedGamma(),
        CensoredResponse.right(time, event),
    )
    model = GAMLSS.from_formula(
        family,
        {
            "mu": "time ~ x",
            "sigma": "~ 1",
            "nu": "~ 1",
        },
        data,
    )
    result = model.fit_data(
        data,
        max_iter=500,
        tolerance_grad=1e-10,
        tolerance_change=1e-13,
    )
    effective_degrees_of_freedom = sum(
        coefficients.numel() for coefficients in model.coefficients.values()
    )

    fit = pd.DataFrame(
        {
            "converged": [result.converged],
            "global_deviance": [2.0 * result.negative_log_likelihood],
            "negative_log_likelihood": [result.negative_log_likelihood],
            "effective_degrees_of_freedom": [effective_degrees_of_freedom],
            "observation_count": [len(data)],
            "event_count": [int(event.sum())],
            "censored_count": [int((event == 0).sum())],
        }
    )
    coefficient_rows = []
    for parameter in model.family.parameter_names:
        coefficient_rows.extend(
            {
                "parameter": parameter,
                "term": term,
                "estimate": estimate,
            }
            for term, estimate in zip(
                model.formula_column_names[parameter],
                model.coefficients[parameter].detach().cpu().tolist(),
                strict=True,
            )
        )
    coefficients = pd.DataFrame(coefficient_rows)

    fitted_parameters = model.predict_data(data)
    fitted = pd.DataFrame(
        {
            "observation": np.arange(len(data)),
            **{
                parameter: values.detach().cpu().numpy()
                for parameter, values in fitted_parameters.items()
            },
        }
    )
    profiles = pd.DataFrame({"x": PROFILE_X})
    curves = model.predict_survival_data(profiles, times=TIMES)
    curve_rows = []
    for profile, x_value in enumerate(PROFILE_X):
        curve_rows.extend(
            {
                "profile": profile,
                "x": x_value,
                "time": time_value,
                "survival": float(curves.survival[profile, time_index].detach()),
                "hazard": float(curves.hazard[profile, time_index].detach()),
                "cumulative_hazard": float(
                    curves.cumulative_hazard[profile, time_index].detach()
                ),
            }
            for time_index, time_value in enumerate(TIMES)
        )
    survival = pd.DataFrame(curve_rows)
    quantiles = model.predict_quantiles_data(
        profiles,
        probabilities=PROBABILITIES,
    ).to_dataframe()[["observation", "probability", "centile", "quantile"]]
    quantiles = quantiles.rename(columns={"observation": "profile"})
    quantiles.insert(1, "x", np.repeat(PROFILE_X, len(PROBABILITIES)))
    metadata = pd.DataFrame(
        {
            "case": ["generalized_gamma_right_censored_mle"],
            "implementation": ["TorchGAMLSS"],
            "family": ["GGcens"],
            "algorithm": ["L-BFGS"],
            "torchgamlss_version": [torchgamlss.__version__],
            "torch_version": [torch.__version__],
        }
    )

    _write_csv(fit, output_dir / "fit.csv")
    _write_csv(coefficients, output_dir / "coefficients.csv")
    _write_csv(fitted, output_dir / "fitted.csv")
    _write_csv(survival, output_dir / "survival.csv")
    _write_csv(quantiles, output_dir / "quantiles.csv")
    _write_csv(metadata, output_dir / "metadata.csv")
    _plot(data, survival, quantiles, output_dir)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _plot(
    data: pd.DataFrame,
    survival: pd.DataFrame,
    quantiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].scatter(
        data.loc[data["event"] == 1, "x"],
        data.loc[data["event"] == 1, "time"],
        color="tab:blue",
        label="event",
    )
    axes[0].scatter(
        data.loc[data["event"] == 0, "x"],
        data.loc[data["event"] == 0, "time"],
        marker="+",
        color="tab:orange",
        label="right-censored",
    )
    for probability, group in quantiles.groupby("probability"):
        axes[0].plot(
            group["x"],
            group["quantile"],
            marker="o",
            label=f"q={probability:g}",
        )
    axes[0].set(xlabel="x", ylabel="time", title="Observed times and quantiles")
    axes[0].legend(fontsize="small")

    for x_value, group in survival.groupby("x"):
        axes[1].plot(
            group["time"],
            group["survival"],
            marker="o",
            label=f"x={x_value:g}",
        )
    axes[1].set(
        xlabel="time",
        ylabel="survival probability",
        ylim=(0.0, 1.0),
        title="Fitted GG survival",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "generalized_gamma_survival.png", dpi=140)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(arguments.data, arguments.output_dir)
