"""End-to-end smoke test executed against an installed wheel."""

from __future__ import annotations

import importlib.metadata

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torchgamlss
from torchgamlss import GAMLSS, TF, Normal, StudentT


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
