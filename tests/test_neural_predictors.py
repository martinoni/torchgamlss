import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from torchgamlss import (
    GAMLSS,
    MiniBatchControl,
    MLPPredictor,
    Normal,
    PSpline,
)

PROJECT_ROOT = Path(__file__).parent.parent


def _nonlinear_problem(
    observation_count: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(2026)
    x = 2.0 * torch.rand(observation_count, generator=generator, dtype=dtype) - 1.0
    location = 0.35 + torch.sin(math.pi * x)
    response = location + 0.12 * torch.randn(
        observation_count,
        generator=generator,
        dtype=dtype,
    )
    designs = {
        "mu": torch.ones((observation_count, 1), dtype=dtype),
        "sigma": torch.ones((observation_count, 1), dtype=dtype),
    }
    return x, response, designs


def test_mlp_predictor_validates_configuration_and_shape():
    predictor = MLPPredictor(
        2,
        (5, 3),
        activation="tanh",
        dropout=0.1,
    ).double()
    inputs = torch.randn((7, 2), dtype=torch.float64)

    assert predictor(inputs).shape == (7,)
    assert predictor.input_size == 2
    assert predictor.hidden_sizes == (5, 3)

    with pytest.raises(ValueError, match="shape"):
        predictor(torch.ones((7, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="input_size"):
        MLPPredictor(0)
    with pytest.raises(ValueError, match="hidden_sizes"):
        MLPPredictor(1, (4, 0))
    with pytest.raises(ValueError, match="activation"):
        MLPPredictor(1, activation="gelu")
    with pytest.raises(ValueError, match="dropout"):
        MLPPredictor(1, dropout=1.0)


def test_neural_predictor_configuration_is_validated():
    with pytest.raises(ValueError, match="unknown"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            neural_predictors={"nu": MLPPredictor(1)},
        )
    with pytest.raises(ValueError, match="nn.Module"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            neural_predictors={"mu": object()},
        )
    with pytest.raises(ValueError, match="mapping"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            neural_predictors=[MLPPredictor(1)],
        )


def test_neural_contribution_is_added_on_the_link_scale():
    dtype = torch.float64
    network = nn.Linear(2, 1, bias=False)
    smooth_x = torch.linspace(-1.0, 1.0, 20, dtype=dtype)
    smooth = PSpline.from_data(smooth_x, smoothing_parameter=5.0)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"trend": smooth}},
        neural_predictors={"mu": network},
        dtype=dtype,
    )
    x = torch.tensor(
        [[-1.0, 0.5], [0.0, 1.0], [2.0, -0.5]],
        dtype=dtype,
    )
    designs = {
        "mu": torch.ones((3, 1), dtype=dtype),
        "sigma": torch.ones((3, 1), dtype=dtype),
    }
    offsets = {"mu": torch.tensor([0.1, 0.2, 0.3], dtype=dtype)}
    with torch.no_grad():
        model.coefficients["mu"].fill_(0.7)
        model.neural_predictors["mu"].weight.copy_(
            torch.tensor([[0.4, -0.2]], dtype=dtype)
        )
        smooth.coefficients.copy_(
            torch.linspace(
                -0.2,
                0.3,
                smooth.coefficients.numel(),
                dtype=dtype,
            )
        )
    evaluation_smooth_x = torch.tensor([-0.8, 0.0, 0.7], dtype=dtype)

    terms = model.predict(
        designs,
        offsets,
        smooth_covariates={"mu": {"trend": evaluation_smooth_x}},
        neural_inputs={"mu": x},
        type="terms",
    )
    link = model.predict(
        designs,
        offsets,
        smooth_covariates={"mu": {"trend": evaluation_smooth_x}},
        neural_inputs={"mu": x},
        type="link",
    )

    assert model.neural_predictors["mu"].weight.dtype == dtype
    torch.testing.assert_close(terms["mu"].neural, (x @ network.weight.mT).squeeze(-1))
    torch.testing.assert_close(
        terms["mu"].smooth["trend"],
        smooth(evaluation_smooth_x),
    )
    torch.testing.assert_close(terms["sigma"].neural, torch.zeros(3, dtype=dtype))
    torch.testing.assert_close(terms["mu"].total, link["mu"])


def test_minibatch_trains_mu_network_and_linear_sigma():
    x, response, designs = _nonlinear_problem()
    torch.manual_seed(731)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={
            "mu": MLPPredictor(1, (16, 16), activation="tanh")
        },
        dtype=torch.float64,
    )
    initial_loss = float(
        model.negative_log_likelihood(
            response,
            designs,
            neural_inputs={"mu": x.unsqueeze(-1)},
            reduction="mean",
        ).detach()
    )

    result = model.fit_minibatch(
        response,
        designs,
        neural_inputs={"mu": x.unsqueeze(-1)},
        control=MiniBatchControl(
            batch_size=64,
            epochs=120,
            learning_rate=0.015,
            learning_rate_decay=0.985,
            minimum_epochs=120,
            patience=120,
            evaluation_frequency=20,
        ),
        generator=torch.Generator().manual_seed(8128),
    )
    model.eval()
    with torch.no_grad():
        parameters = model.predict(
            designs,
            neural_inputs={"mu": x.unsqueeze(-1)},
            type="response",
        )
    truth = 0.35 + torch.sin(math.pi * x)
    mean_squared_error = float((parameters["mu"] - truth).square().mean())

    assert result.negative_log_likelihood / response.numel() < initial_loss
    assert mean_squared_error < 0.01
    assert float(parameters["sigma"].mean()) == pytest.approx(0.12, abs=0.035)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_batch_lbfgs_trains_neural_parameters():
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, 80, dtype=dtype).unsqueeze(-1)
    truth = 0.4 + 1.7 * x.squeeze(-1)
    response = truth + 0.15 * torch.randn(
        x.shape[0],
        dtype=dtype,
        generator=torch.Generator().manual_seed(319),
    )
    designs = {
        "mu": torch.ones((x.shape[0], 1), dtype=dtype),
        "sigma": torch.ones((x.shape[0], 1), dtype=dtype),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={"mu": MLPPredictor(1, ())},
        dtype=dtype,
    )

    result = model.fit(
        response,
        designs,
        neural_inputs={"mu": x},
        max_iter=100,
    )
    with torch.no_grad():
        prediction = model.predict(
            designs,
            neural_inputs={"mu": x},
            type="response",
        )

    assert result.negative_log_likelihood < 0
    assert float((prediction["mu"] - truth).square().mean()) < 0.001
    assert float(prediction["sigma"].mean()) == pytest.approx(0.15, abs=0.04)


def test_formula_data_accepts_one_or_multiple_neural_columns():
    x, response, _ = _nonlinear_problem(64)
    data = pd.DataFrame(
        {
            "y": response.numpy(),
            "x": x.numpy(),
            "x2": x.square().numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ 1", "sigma": "~ 1"},
        data,
        neural_predictors={"mu": MLPPredictor(2, ())},
    )
    with torch.no_grad():
        model.neural_predictors["mu"].network[-1].weight.copy_(
            torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        )
        model.neural_predictors["mu"].network[-1].bias.zero_()

    prediction = model.predict_data(
        data,
        neural_inputs={"mu": ["x", "x2"]},
        type="link",
    )
    expected = x - 0.5 * x.square()

    torch.testing.assert_close(prediction["mu"], expected)


@pytest.mark.parametrize(
    "neural_inputs,match",
    [
        ({}, "missing"),
        ({"sigma": torch.ones((4, 1), dtype=torch.float64)}, "missing"),
        ({"mu": torch.ones((3, 1), dtype=torch.float64)}, "one row"),
        ({"mu": torch.ones((4, 1), dtype=torch.float32)}, "dtype"),
    ],
)
def test_neural_inputs_are_validated(neural_inputs, match):
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={"mu": MLPPredictor(1, ())},
        dtype=torch.float64,
    )
    designs = {
        "mu": torch.ones((4, 1), dtype=torch.float64),
        "sigma": torch.ones((4, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match=match):
        model.predict(designs, neural_inputs=neural_inputs)


def test_neural_predictor_must_return_one_finite_value_per_row():
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={"mu": nn.Linear(1, 2)},
        dtype=torch.float64,
    )
    designs = {
        "mu": torch.ones((4, 1), dtype=torch.float64),
        "sigma": torch.ones((4, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="one value"):
        model.predict(
            designs,
            neural_inputs={"mu": torch.ones((4, 1), dtype=torch.float64)},
        )


def test_classical_fitting_and_inference_reject_neural_predictors():
    _, response, designs = _nonlinear_problem(32)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={"mu": MLPPredictor(1, ())},
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="fit_minibatch"):
        model.fit_rs(response, designs)
    with pytest.raises(ValueError, match="fit_minibatch"):
        model.fit_cg(response, designs)
    with pytest.raises(ValueError, match="fit_minibatch"):
        model.inference(response, designs)


def test_neural_diagnostics_require_explicit_degrees_of_freedom():
    x, response, designs = _nonlinear_problem(32)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={"mu": MLPPredictor(1, ())},
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="degrees_of_freedom"):
        model.diagnostics(
            response,
            designs,
            neural_inputs={"mu": x.unsqueeze(-1)},
        )
    result = model.diagnostics(
        response,
        designs,
        neural_inputs={"mu": x.unsqueeze(-1)},
        degrees_of_freedom=4,
    )

    assert math.isfinite(result.aic)


def test_neural_benchmark_cli_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_neural.py",
            "--rows",
            "128",
            "--features",
            "2",
            "--hidden-size",
            "4",
            "--hidden-layers",
            "1",
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
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["device"] == "cpu"
    assert report["rows"] == 128
    assert report["features"] == 2
    assert report["updates"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_neural_minibatch_runs_on_cuda():
    device = torch.device("cuda")
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(617)
    x = torch.randn((256, 3), generator=generator, dtype=dtype).to(device)
    response = (0.2 + torch.sin(x[:, 0])).to(device)
    designs = {
        "mu": torch.ones((256, 1), dtype=dtype, device=device),
        "sigma": torch.ones((256, 1), dtype=dtype, device=device),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        neural_predictors={"mu": MLPPredictor(3, (8,))},
        dtype=dtype,
        device=device,
    )

    result = model.fit_minibatch(
        response,
        designs,
        neural_inputs={"mu": x},
        control=MiniBatchControl(
            batch_size=64,
            epochs=2,
            minimum_epochs=2,
            patience=2,
        ),
        generator=generator,
    )
    prediction = model.predict(
        designs,
        neural_inputs={"mu": x},
        type="response",
    )

    assert result.updates == 8
    assert prediction["mu"].device.type == "cuda"
    assert next(model.neural_predictors["mu"].parameters()).device.type == "cuda"
