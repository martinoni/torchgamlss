"""Compare Poisson and NBI count regressions with TorchGAMLSS."""

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
    NegativeBinomial,
    Poisson,
    RSControl,
    compare_models,
)

PROBABILITIES = (0.05, 0.50, 0.95)


def run(data_path: Path, output_dir: Path) -> None:
    """Fit both count models and write standardized parity artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path)
    data["uniform"] = _residual_uniforms(len(data))
    models = {
        "PO": GAMLSS.from_formula(
            Poisson(),
            {"mu": "y ~ x + offset(log_exposure)"},
            data,
        ),
        "NBI": GAMLSS.from_formula(
            NegativeBinomial(),
            {
                "mu": "y ~ x + offset(log_exposure)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            data,
        ),
    }
    controls = {
        "PO": RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
        ),
        "NBI": RSControl(
            outer_tolerance=1e-9,
            max_outer_iterations=200,
            inner_tolerance=1e-9,
            max_inner_iterations=200,
        ),
    }

    fit_results = {}
    diagnostics = {}
    for name, model in models.items():
        result = model.fit_rs_data(
            data,
            weights="weight",
            control=controls[name],
        )
        fit_results[name] = result
        diagnostics[name] = model.diagnostics_data(
            data,
            weights="weight",
            degrees_of_freedom=result.effective_degrees_of_freedom,
        )

    fit = _fit_table(data, models, fit_results, diagnostics)
    comparison = compare_models(diagnostics, criterion="aic").reset_index()
    comparison.insert(1, "rank", np.arange(1, len(comparison) + 1))
    coefficients = _coefficient_table(models)
    fitted = _fitted_table(data, models)
    quantiles = _quantile_table(data, models)
    residuals = _residual_table(data, models)
    metadata = pd.DataFrame(
        {
            "case": ["count_model_comparison"] * len(models),
            "model": list(models),
            "implementation": ["TorchGAMLSS"] * len(models),
            "algorithm": ["RS"] * len(models),
            "torchgamlss_version": [torchgamlss.__version__] * len(models),
            "torch_version": [torch.__version__] * len(models),
        }
    )

    _write_csv(fit, output_dir / "fit.csv")
    _write_csv(comparison, output_dir / "model_comparison.csv")
    _write_csv(coefficients, output_dir / "coefficients.csv")
    _write_csv(fitted, output_dir / "fitted.csv")
    _write_csv(
        quantiles[quantiles["observation"] % 7 == 0],
        output_dir / "quantiles.csv",
    )
    _write_csv(residuals, output_dir / "residuals.csv")
    _write_csv(metadata, output_dir / "metadata.csv")
    _plot_example(data, fitted, quantiles, residuals, comparison, output_dir)


def _residual_uniforms(observation_count: int) -> np.ndarray:
    indices = np.arange(1, observation_count + 1)
    return ((indices * 29) % observation_count + 0.5) / observation_count


def _fit_table(
    data: pd.DataFrame,
    models: dict[str, GAMLSS],
    fit_results: dict,
    diagnostics: dict,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        result = fit_results[name]
        diagnostic = diagnostics[name]
        parameters = model.predict_data(data)
        variance = (
            model.family.distribution(parameters).variance.detach().cpu().numpy()
        )
        pearson = np.sum(
            data["weight"].to_numpy()
            * np.square(data["y"].to_numpy() - parameters["mu"].detach().cpu().numpy())
            / variance
        ) / diagnostic.residual_degrees_of_freedom
        rows.append(
            {
                "model": name,
                "converged": result.converged,
                "outer_iterations": result.outer_iterations,
                "global_deviance": result.global_deviance,
                "negative_log_likelihood": result.negative_log_likelihood,
                "effective_degrees_of_freedom": (
                    result.effective_degrees_of_freedom
                ),
                "residual_degrees_of_freedom": (
                    diagnostic.residual_degrees_of_freedom
                ),
                "observation_count": diagnostic.observation_count,
                "effective_observation_count": (
                    diagnostic.effective_observation_count
                ),
                "aic": diagnostic.aic,
                "aicc": diagnostic.aicc,
                "bic": diagnostic.bic,
                "pearson_dispersion": pearson,
            }
        )
    return pd.DataFrame(rows)


def _coefficient_table(models: dict[str, GAMLSS]) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        for parameter in model.family.parameter_names:
            names = model.formula_column_names[parameter]
            estimates = model.coefficients[parameter].detach().cpu().tolist()
            rows.extend(
                {
                    "model": model_name,
                    "parameter": parameter,
                    "term": name,
                    "estimate": estimate,
                }
                for name, estimate in zip(names, estimates, strict=True)
            )
    return pd.DataFrame(rows)


def _fitted_table(
    data: pd.DataFrame,
    models: dict[str, GAMLSS],
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        parameters = model.predict_data(data)
        mu = parameters["mu"].detach().cpu().numpy()
        sigma = (
            parameters["sigma"].detach().cpu().numpy()
            if "sigma" in parameters
            else np.zeros(len(data))
        )
        variance = (
            model.family.distribution(parameters).variance.detach().cpu().numpy()
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "observation": np.arange(len(data)),
                    "mu": mu,
                    "sigma": sigma,
                    "variance": variance,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _quantile_table(
    data: pd.DataFrame,
    models: dict[str, GAMLSS],
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        table = model.predict_quantiles_data(
            data,
            probabilities=PROBABILITIES,
        ).to_dataframe()
        table.insert(0, "model", name)
        rows.append(
            table[
                ["model", "observation", "probability", "centile", "quantile"]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _residual_table(
    data: pd.DataFrame,
    models: dict[str, GAMLSS],
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        residual = (
            model.quantile_residuals_data(data, uniforms="uniform")
            .detach()
            .cpu()
            .numpy()
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "observation": np.arange(len(data)),
                    "uniform": data["uniform"],
                    "quantile_residual": residual,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _plot_example(
    data: pd.DataFrame,
    fitted: pd.DataFrame,
    quantiles: pd.DataFrame,
    residuals: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {"PO": "tab:orange", "NBI": "tab:blue"}
    order = np.argsort(data["x"].to_numpy())
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].scatter(data["x"], data["y"], color="black", s=18, label="observed")
    for name in ("PO", "NBI"):
        model_fitted = fitted[fitted["model"] == name]
        axes[0, 0].plot(
            data["x"].to_numpy()[order],
            model_fitted["mu"].to_numpy()[order],
            color=colors[name],
            label=f"{name} mean",
        )
    axes[0, 0].set(xlabel="x", ylabel="count", title="Conditional mean")
    axes[0, 0].legend()

    axes[0, 1].scatter(data["x"], data["y"], color="black", s=14, alpha=0.5)
    for name in ("PO", "NBI"):
        model_quantiles = quantiles[quantiles["model"] == name]
        lower = model_quantiles[model_quantiles["probability"] == 0.05][
            "quantile"
        ].to_numpy()
        median = model_quantiles[model_quantiles["probability"] == 0.50][
            "quantile"
        ].to_numpy()
        upper = model_quantiles[model_quantiles["probability"] == 0.95][
            "quantile"
        ].to_numpy()
        axes[0, 1].fill_between(
            data["x"].to_numpy()[order],
            lower[order],
            upper[order],
            color=colors[name],
            alpha=0.13,
            label=f"{name} 5%-95%",
        )
        axes[0, 1].plot(
            data["x"].to_numpy()[order],
            median[order],
            color=colors[name],
        )
    axes[0, 1].set(xlabel="x", ylabel="count", title="Predictive quantiles")
    axes[0, 1].legend()

    axes[1, 0].axhline(0.0, color="0.5", linewidth=1)
    for name in ("PO", "NBI"):
        model_residuals = residuals[residuals["model"] == name]
        axes[1, 0].scatter(
            data["x"],
            model_residuals["quantile_residual"],
            color=colors[name],
            s=18,
            alpha=0.72,
            label=name,
        )
    axes[1, 0].set(
        xlabel="x",
        ylabel="randomized quantile residual",
        title="Dunn-Smyth residuals",
    )
    axes[1, 0].legend()

    axes[1, 1].bar(
        comparison["model"],
        comparison["weight"],
        color=[colors[name] for name in comparison["model"]],
    )
    axes[1, 1].set(
        ylim=(0, 1.05),
        ylabel="Akaike weight",
        title="Model evidence",
    )
    for index, row in comparison.iterrows():
        axes[1, 1].text(
            index,
            min(row["weight"] + 0.03, 1.01),
            f"ΔAIC={row['delta']:.2f}",
            ha="center",
        )

    figure.tight_layout()
    figure.savefig(output_dir / "count_model_comparison.png", dpi=140)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(arguments.data, arguments.output_dir)
