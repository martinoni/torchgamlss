import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import GAMLSS, MiniBatchControl, Normal

REFERENCE_DIR = Path(__file__).parent / "reference"


def _normal_fit_inputs():
    with (REFERENCE_DIR / "no_fit_data.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        rows = list(csv.DictReader(data_file))
    x = torch.tensor([float(row["x"]) for row in rows], dtype=torch.float64)
    response = torch.tensor(
        [float(row["y"]) for row in rows],
        dtype=torch.float64,
    )
    design_matrices = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }
    with (REFERENCE_DIR / "no_fit_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as reference_file:
        reference = next(csv.DictReader(reference_file))
    return x, response, design_matrices, reference


def test_minibatch_adam_matches_r_normal_fit():
    _, response, design_matrices, reference = _normal_fit_inputs()
    copies = 20
    response = response.repeat(copies)
    design_matrices = {
        parameter: design.repeat((copies, 1))
        for parameter, design in design_matrices.items()
    }
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=torch.float64)
    result = model.fit_minibatch(
        response,
        design_matrices,
        control=MiniBatchControl(
            batch_size=16,
            epochs=500,
            learning_rate=0.05,
            learning_rate_decay=0.97,
            minimum_epochs=300,
            patience=20,
            tolerance_change=1e-9,
            tolerance_gradient=1e-4,
        ),
        generator=torch.Generator().manual_seed(2026),
    )

    assert result.converged
    assert result.stop_reason in {"loss_change", "gradient"}
    assert result.epochs <= 500
    assert result.updates >= result.epochs
    assert result.batch_size == min(16, response.numel())
    assert result.evaluation_epochs[0] == 0
    assert result.evaluation_epochs[-1] == result.epochs
    assert len(result.objective_history) == len(result.evaluation_epochs)
    assert result.gradient_max < 1e-2
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [
                float(reference["mu_intercept"]),
                float(reference["mu_x"]),
            ],
            dtype=torch.float64,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor(
            [float(reference["sigma_intercept"])],
            dtype=torch.float64,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    assert result.negative_log_likelihood == pytest.approx(
        copies * float(reference["negative_log_likelihood"]),
        rel=2e-5,
        abs=2e-6,
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_minibatch_shuffle_is_reproducible_with_a_seeded_generator():
    _, response, design_matrices, _ = _normal_fit_inputs()
    response = response.repeat(10)
    design_matrices = {
        parameter: design.repeat((10, 1))
        for parameter, design in design_matrices.items()
    }
    control = MiniBatchControl(
        batch_size=7,
        epochs=40,
        learning_rate=0.03,
        minimum_epochs=40,
        patience=2,
        tolerance_change=0.0,
        tolerance_gradient=0.0,
        evaluation_frequency=4,
    )
    models = [
        GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=torch.float64) for _ in range(2)
    ]
    results = [
        model.fit_minibatch(
            response,
            design_matrices,
            control=control,
            generator=torch.Generator().manual_seed(8128),
        )
        for model in models
    ]

    for parameter in models[0].family.parameter_names:
        torch.testing.assert_close(
            models[0].coefficients[parameter],
            models[1].coefficients[parameter],
            rtol=0.0,
            atol=0.0,
        )
    assert results[0] == results[1]
    assert results[0].stop_reason == "max_epochs"
    assert not results[0].converged


def test_formula_minibatches_support_zero_weights_offsets_and_fixed_smooths():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    data["mu_offset"] = (
        0.05
        * torch.sin(torch.tensor(data["x"].to_numpy(), dtype=torch.float64)).numpy()
    )
    data["weight"] = 1.0
    data.loc[:7, "weight"] = 0.0
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": ("y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)"),
            "sigma": "~ 1",
        },
        data,
    )
    coefficients_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    result = model.fit_minibatch_data(
        data,
        weights="weight",
        control=MiniBatchControl(
            batch_size=8,
            epochs=6,
            learning_rate=0.01,
            shuffle=False,
            minimum_epochs=6,
            patience=10,
            evaluation_frequency=2,
        ),
    )

    assert result.batch_size == 8
    assert result.epochs == 6
    assert result.updates == math.ceil(len(data) / 8) * 6
    assert result.evaluation_epochs == (0, 2, 4, 6)
    assert math.isfinite(result.negative_log_likelihood)
    assert math.isfinite(result.penalized_objective)
    assert any(
        not torch.equal(parameter, coefficients_before[name])
        for name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize(
    "keyword,value,match",
    [
        ("batch_size", 0, "batch_size"),
        ("epochs", 0, "epochs"),
        ("learning_rate", 0.0, "learning_rate"),
        ("learning_rate_decay", 0.0, "learning_rate_decay"),
        ("minimum_epochs", 101, "minimum_epochs"),
        ("patience", 0, "patience"),
        ("evaluation_frequency", 0, "evaluation_frequency"),
        ("clip_gradient_norm", 0.0, "clip_gradient_norm"),
    ],
)
def test_minibatch_control_validation(keyword, value, match):
    arguments = {"epochs": 100, keyword: value}
    with pytest.raises(ValueError, match=match):
        MiniBatchControl(**arguments)


def test_minibatch_rejects_automatic_smoothing_selection():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x)", "sigma": "~ 1"},
        data,
    )

    with pytest.raises(ValueError, match="automatic smoothing"):
        model.fit_minibatch_data(
            data,
            control=MiniBatchControl(epochs=1, minimum_epochs=1),
        )


def test_minibatch_benchmark_cli_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_minibatch.py",
            "--rows",
            "128",
            "--features",
            "2",
            "--batch-size",
            "64",
            "--epochs",
            "1",
            "--evaluation-frequency",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float64",
        ],
        cwd=REFERENCE_DIR.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["device"] == "cpu"
    assert report["rows"] == 128
    assert report["batch_size"] == 64
    assert report["updates"] == 2
    assert report["epochs"] == 1
