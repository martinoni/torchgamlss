import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    MiniBatchControl,
    MiniBatchValidationData,
    Normal,
    SharedMLPPredictor,
)

PROJECT_ROOT = Path(__file__).parent.parent


def _linear_data(
    observation_count: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(observation_count, dtype=dtype, generator=generator)
    response = (
        0.4
        + 1.3 * x
        + 0.35
        * torch.randn(
            observation_count,
            dtype=dtype,
            generator=generator,
        )
    )
    designs = {
        "mu": torch.column_stack((torch.ones_like(x), x)).to(device),
        "sigma": torch.ones(
            (observation_count, 1),
            dtype=dtype,
            device=device,
        ),
    }
    return x.to(device), response.to(device), designs


def test_validation_early_stopping_restores_the_exact_initial_state():
    _, response, designs = _linear_data(96, seed=101)
    _, validation_response, validation_designs = _linear_data(48, seed=202)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    initial_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    initial_training_nll = float(
        model.negative_log_likelihood(response, designs).detach()
    )
    validation = MiniBatchValidationData(
        response=validation_response,
        design_matrices=validation_designs,
    )

    result = model.fit_minibatch(
        response,
        designs,
        validation=validation,
        control=MiniBatchControl(
            batch_size=24,
            epochs=20,
            learning_rate=0.05,
            minimum_epochs=3,
            validation_patience=2,
            validation_minimum_delta=1e6,
            evaluation_frequency=1,
        ),
        generator=torch.Generator().manual_seed(303),
    )

    assert result.stop_reason == "validation"
    assert result.converged
    assert result.epochs == 3
    assert result.best_epoch == 0
    assert result.restored_best_parameters
    assert result.validation_epochs == (0, 1, 2, 3)
    assert len(result.validation_history) == 4
    assert result.best_validation_loss == result.validation_history[0]
    assert result.negative_log_likelihood == pytest.approx(
        initial_training_nll,
        rel=0.0,
        abs=1e-12,
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            initial_state[name],
            rtol=0.0,
            atol=0.0,
        )


def test_validation_history_tracks_weighted_holdout_likelihood():
    _, response, designs = _linear_data(256, seed=410)
    _, validation_response, validation_designs = _linear_data(128, seed=411)
    validation_weights = torch.linspace(
        0.5,
        1.5,
        validation_response.numel(),
        dtype=torch.float64,
    )
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    validation = MiniBatchValidationData(
        response=validation_response,
        design_matrices=validation_designs,
        weights=validation_weights,
    )

    result = model.fit_minibatch(
        response,
        designs,
        validation=validation,
        control=MiniBatchControl(
            batch_size=32,
            epochs=40,
            learning_rate=0.04,
            learning_rate_decay=0.97,
            minimum_epochs=40,
            validation_patience=40,
            evaluation_frequency=5,
        ),
        generator=torch.Generator().manual_seed(412),
    )
    final_validation_nll = float(
        model.negative_log_likelihood(
            validation_response,
            validation_designs,
            weights=validation_weights,
        ).detach()
    )

    assert result.validation_epochs == (0, 5, 10, 15, 20, 25, 30, 35, 40)
    assert len(result.validation_history) == len(result.validation_epochs)
    assert result.best_validation_loss == min(result.validation_history)
    best_index = result.validation_history.index(result.best_validation_loss)
    assert result.best_epoch == result.validation_epochs[best_index]
    assert result.best_epoch is not None and result.best_epoch > 0
    assert result.validation_negative_log_likelihood == pytest.approx(
        final_validation_nll,
        rel=1e-12,
        abs=1e-12,
    )
    assert (
        result.validation_negative_log_likelihood
        / float(validation_weights.sum())
    ) == pytest.approx(
        result.best_validation_loss,
        rel=1e-12,
        abs=1e-12,
    )


def test_validation_can_keep_the_last_parameters_instead_of_restoring():
    _, response, designs = _linear_data(96, seed=450)
    _, validation_response, validation_designs = _linear_data(48, seed=451)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    validation = MiniBatchValidationData(
        response=validation_response,
        design_matrices=validation_designs,
    )

    result = model.fit_minibatch(
        response,
        designs,
        validation=validation,
        control=MiniBatchControl(
            batch_size=24,
            epochs=10,
            learning_rate=0.05,
            minimum_epochs=2,
            validation_patience=1,
            validation_minimum_delta=1e6,
            restore_best_parameters=False,
        ),
        generator=torch.Generator().manual_seed(452),
    )
    final_validation_nll = float(
        model.negative_log_likelihood(
            validation_response,
            validation_designs,
        ).detach()
    )

    assert result.stop_reason == "validation"
    assert result.best_epoch == 0
    assert not result.restored_best_parameters
    assert result.validation_negative_log_likelihood == pytest.approx(
        final_validation_nll,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.validation_negative_log_likelihood / len(
        validation_response
    ) != pytest.approx(result.best_validation_loss)


def test_formula_minibatch_materializes_validation_data_with_same_columns():
    x, response, _ = _linear_data(96, seed=501)
    validation_x, validation_response, _ = _linear_data(48, seed=502)
    training_frame = pd.DataFrame(
        {
            "y": response.numpy(),
            "x": x.numpy(),
            "weight": torch.linspace(0.8, 1.2, 96).numpy(),
        }
    )
    validation_frame = pd.DataFrame(
        {
            "y": validation_response.numpy(),
            "x": validation_x.numpy(),
            "weight": torch.linspace(0.9, 1.1, 48).numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        training_frame,
    )

    result = model.fit_minibatch_data(
        training_frame,
        validation_data=validation_frame,
        weights="weight",
        control=MiniBatchControl(
            batch_size=24,
            epochs=5,
            minimum_epochs=5,
            validation_patience=10,
            evaluation_frequency=1,
        ),
    )

    assert result.validation_epochs == (0, 1, 2, 3, 4, 5)
    assert result.validation_negative_log_likelihood is not None
    assert math.isfinite(result.validation_negative_log_likelihood)


@pytest.mark.parametrize(
    "keyword,value,match",
    [
        ("validation_patience", 0, "validation_patience"),
        ("validation_minimum_delta", -1.0, "validation_minimum_delta"),
        ("restore_best_parameters", 1, "restore_best_parameters"),
        ("amp_dtype", "float32", "amp_dtype"),
    ],
)
def test_validation_control_is_validated(keyword, value, match):
    with pytest.raises(ValueError, match=match):
        MiniBatchControl(**{keyword: value})


def test_validation_argument_and_tensors_are_validated_before_updates():
    _, response, designs = _linear_data(32, seed=610)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="MiniBatchValidationData"):
        model.fit_minibatch(response, designs, validation={})
    invalid_validation = MiniBatchValidationData(
        response=response.to(torch.float32),
        design_matrices=designs,
    )
    with pytest.raises(ValueError, match="validation data"):
        model.fit_minibatch(
            response,
            designs,
            validation=invalid_validation,
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            state_before[name],
            rtol=0.0,
            atol=0.0,
        )


def test_shared_benchmark_cli_supports_validation():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_shared.py",
            "--rows",
            "128",
            "--validation-rows",
            "64",
            "--features",
            "2",
            "--hidden-size",
            "4",
            "--hidden-layers",
            "1",
            "--batch-size",
            "64",
            "--epochs",
            "2",
            "--minimum-epochs",
            "1",
            "--evaluation-frequency",
            "1",
            "--validation-patience",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float64",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["validation_rows"] == 64
    assert report["validation_negative_log_likelihood"] is not None
    assert report["best_epoch"] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shared_validation_runs_on_cuda():
    device = torch.device("cuda")
    dtype = torch.float32
    train_x, response, designs = _linear_data(
        256,
        seed=701,
        device=device,
        dtype=dtype,
    )
    validation_x, validation_response, validation_designs = _linear_data(
        128,
        seed=702,
        device=device,
        dtype=dtype,
    )
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            1,
            ("mu",),
            (8,),
        ),
        dtype=dtype,
        device=device,
    )
    validation = MiniBatchValidationData(
        response=validation_response,
        design_matrices={
            "mu": validation_designs["mu"][:, :1],
            "sigma": validation_designs["sigma"],
        },
        shared_input=validation_x.unsqueeze(-1),
    )

    result = model.fit_minibatch(
        response,
        {
            "mu": designs["mu"][:, :1],
            "sigma": designs["sigma"],
        },
        shared_input=train_x.unsqueeze(-1),
        validation=validation,
        control=MiniBatchControl(
            batch_size=64,
            epochs=2,
            minimum_epochs=2,
            validation_patience=2,
        ),
    )

    assert result.validation_negative_log_likelihood is not None
    assert math.isfinite(result.validation_negative_log_likelihood)
    assert next(model.parameters()).device.type == "cuda"
