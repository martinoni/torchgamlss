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
    Normal,
    SharedMLPPredictor,
)

PROJECT_ROOT = Path(__file__).parent.parent


class _KnownSharedPredictor(nn.Module):
    parameter_names = ("mu", "sigma")

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return {
            "mu": self.scale * inputs[:, 0],
            "sigma": self.scale * inputs[:, 1:2],
        }


class _BadSharedPredictor(nn.Module):
    def __init__(self, output) -> None:
        super().__init__()
        self.output = output

    def forward(self, inputs):
        if callable(self.output):
            return self.output(inputs)
        return self.output


class _ModeRecordingSharedPredictor(nn.Module):
    parameter_names = ("mu",)

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 1)
        self.training_modes = []

    def forward(self, inputs):
        self.training_modes.append(self.training)
        return {"mu": self.head(inputs).squeeze(-1)}


def _heteroscedastic_problem(
    observation_count: int = 768,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    dtype = torch.float64
    generator = torch.Generator().manual_seed(404)
    inputs = (
        2.0
        * torch.rand(
            (observation_count, 2),
            dtype=dtype,
            generator=generator,
        )
        - 1.0
    )
    latent = torch.sin(math.pi * inputs[:, 0]) + 0.4 * inputs[:, 1]
    location = 0.3 + latent
    log_scale = -1.5 + 0.35 * latent
    scale = log_scale.exp()
    response = location + scale * torch.randn(
        observation_count,
        dtype=dtype,
        generator=generator,
    )
    designs = {
        "mu": torch.ones((observation_count, 1), dtype=dtype),
        "sigma": torch.ones((observation_count, 1), dtype=dtype),
    }
    return inputs, response, designs, location, scale


def test_shared_mlp_has_one_backbone_and_named_heads():
    predictor = SharedMLPPredictor(
        3,
        ("mu", "sigma"),
        (7, 5),
        activation="tanh",
        dropout=0.1,
    ).double()
    outputs = predictor(torch.randn((11, 3), dtype=torch.float64))

    assert set(outputs) == {"mu", "sigma"}
    assert outputs["mu"].shape == (11,)
    assert outputs["sigma"].shape == (11,)
    assert predictor.parameter_names == ("mu", "sigma")
    assert predictor.hidden_sizes == (7, 5)
    assert set(predictor.heads) == {"mu", "sigma"}

    with pytest.raises(ValueError, match="shape"):
        predictor(torch.ones((11, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="input_size"):
        SharedMLPPredictor(0, ("mu",))
    with pytest.raises(ValueError, match="parameter_names"):
        SharedMLPPredictor(1, ())
    with pytest.raises(ValueError, match="parameter_names"):
        SharedMLPPredictor(1, ("mu", "mu"))
    with pytest.raises(ValueError, match="parameter_names"):
        SharedMLPPredictor(1, ("bad.name",))


def test_shared_predictor_configuration_is_validated():
    with pytest.raises(ValueError, match="nn.Module"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            shared_predictor=object(),
        )
    with pytest.raises(ValueError, match="requires a shared_predictor"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            shared_parameters=("mu",),
        )
    with pytest.raises(ValueError, match="required"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            shared_predictor=nn.Linear(2, 2),
        )
    with pytest.raises(ValueError, match="unknown"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            shared_predictor=nn.Linear(2, 2),
            shared_parameters=("mu", "nu"),
        )
    with pytest.raises(ValueError, match="match"):
        GAMLSS(
            Normal(),
            {"mu": 1, "sigma": 1},
            shared_predictor=SharedMLPPredictor(2, ("mu", "sigma")),
            shared_parameters=("mu",),
        )


def test_shared_contributions_are_evaluated_once_and_added_by_parameter():
    dtype = torch.float64
    predictor = _KnownSharedPredictor()
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=predictor,
        dtype=dtype,
    )
    shared_input = torch.tensor(
        [[-1.0, 0.2], [0.5, -0.4], [2.0, 0.7]],
        dtype=dtype,
    )
    designs = {
        "mu": torch.ones((3, 1), dtype=dtype),
        "sigma": torch.ones((3, 1), dtype=dtype),
    }
    with torch.no_grad():
        model.coefficients["mu"].fill_(0.25)
        model.coefficients["sigma"].fill_(-0.8)

    terms = model.term_contributions(
        designs,
        shared_input=shared_input,
    )

    assert predictor.calls == 1
    torch.testing.assert_close(terms["mu"].shared, shared_input[:, 0])
    torch.testing.assert_close(terms["sigma"].shared, shared_input[:, 1])
    torch.testing.assert_close(
        terms["mu"].total,
        0.25 + shared_input[:, 0],
    )
    torch.testing.assert_close(
        terms["sigma"].total,
        -0.8 + shared_input[:, 1],
    )


def test_formula_data_accepts_shared_feature_columns():
    inputs, response, _, _, _ = _heteroscedastic_problem(64)
    data = pd.DataFrame(
        {
            "y": response.numpy(),
            "x1": inputs[:, 0].numpy(),
            "x2": inputs[:, 1].numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ 1", "sigma": "~ 1"},
        data,
        shared_predictor=SharedMLPPredictor(
            2,
            ("mu", "sigma"),
            (),
        ),
    )
    with torch.no_grad():
        model.shared_predictor.heads["mu"].weight.copy_(
            torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        )
        model.shared_predictor.heads["mu"].bias.zero_()
        model.shared_predictor.heads["sigma"].weight.copy_(
            torch.tensor([[0.2, 0.3]], dtype=torch.float64)
        )
        model.shared_predictor.heads["sigma"].bias.zero_()

    link = model.predict_data(
        data,
        shared_input=["x1", "x2"],
        type="link",
    )

    torch.testing.assert_close(
        link["mu"],
        inputs[:, 0] - 0.5 * inputs[:, 1],
    )
    torch.testing.assert_close(
        link["sigma"],
        0.2 * inputs[:, 0] + 0.3 * inputs[:, 1],
    )


@pytest.mark.parametrize(
    "shared_input,match",
    [
        (None, "required"),
        (torch.ones((3, 2), dtype=torch.float64), "one row"),
        (torch.ones((4, 2), dtype=torch.float32), "dtype"),
    ],
)
def test_shared_input_is_validated(shared_input, match):
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(2, ("mu", "sigma"), ()),
        dtype=torch.float64,
    )
    designs = {
        "mu": torch.ones((4, 1), dtype=torch.float64),
        "sigma": torch.ones((4, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match=match):
        model.predict(designs, shared_input=shared_input)


@pytest.mark.parametrize(
    "output,parameters,match",
    [
        (torch.ones(4), ("mu",), "mapping"),
        ({"mu": torch.ones(4)}, ("mu", "sigma"), "missing"),
        (
            lambda inputs: {
                "mu": torch.ones((inputs.shape[0], 2), dtype=inputs.dtype),
            },
            ("mu",),
            "one value",
        ),
        (
            lambda inputs: {
                "mu": torch.full(
                    (inputs.shape[0],),
                    float("nan"),
                    dtype=inputs.dtype,
                ),
            },
            ("mu",),
            "finite",
        ),
    ],
)
def test_shared_outputs_are_validated(output, parameters, match):
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=_BadSharedPredictor(output),
        shared_parameters=parameters,
        dtype=torch.float64,
    )
    designs = {
        "mu": torch.ones((4, 1), dtype=torch.float64),
        "sigma": torch.ones((4, 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match=match):
        model.predict(
            designs,
            shared_input=torch.ones((4, 2), dtype=torch.float64),
        )


def test_minibatch_shared_backbone_learns_location_and_scale():
    inputs, response, designs, true_location, true_scale = (
        _heteroscedastic_problem()
    )
    torch.manual_seed(937)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            2,
            ("mu", "sigma"),
            (24, 24),
            activation="tanh",
        ),
        dtype=torch.float64,
    )
    initial_loss = float(
        model.negative_log_likelihood(
            response,
            designs,
            shared_input=inputs,
            reduction="mean",
        ).detach()
    )

    result = model.fit_minibatch(
        response,
        designs,
        shared_input=inputs,
        control=MiniBatchControl(
            batch_size=96,
            epochs=140,
            learning_rate=0.012,
            learning_rate_decay=0.987,
            minimum_epochs=140,
            patience=140,
            evaluation_frequency=20,
        ),
        generator=torch.Generator().manual_seed(510),
    )
    model.eval()
    with torch.no_grad():
        fitted = model.predict(
            designs,
            shared_input=inputs,
            type="response",
        )

    location_mse = float((fitted["mu"] - true_location).square().mean())
    log_scale_mse = float(
        (fitted["sigma"].log() - true_scale.log()).square().mean()
    )
    assert result.negative_log_likelihood / response.numel() < initial_loss
    assert location_mse < 0.012
    assert log_scale_mse < 0.035
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_batch_lbfgs_trains_shared_head():
    dtype = torch.float64
    inputs = torch.linspace(-1.0, 1.0, 80, dtype=dtype).unsqueeze(-1)
    truth = 0.25 + 1.4 * inputs.squeeze(-1)
    response = truth + 0.18 * torch.randn(
        inputs.shape[0],
        dtype=dtype,
        generator=torch.Generator().manual_seed(208),
    )
    designs = {
        "mu": torch.ones((inputs.shape[0], 1), dtype=dtype),
        "sigma": torch.ones((inputs.shape[0], 1), dtype=dtype),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(1, ("mu",), ()),
        dtype=dtype,
    )

    result = model.fit(
        response,
        designs,
        shared_input=inputs,
        max_iter=100,
    )
    with torch.no_grad():
        fitted = model.predict(
            designs,
            shared_input=inputs,
            type="response",
        )

    assert result.negative_log_likelihood < 10
    assert float((fitted["mu"] - truth).square().mean()) < 0.002


def test_classical_paths_and_default_diagnostics_reject_shared_predictor():
    inputs, response, designs, _, _ = _heteroscedastic_problem(48)
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(2, ("mu", "sigma"), ()),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="shared"):
        model.fit_rs(response, designs)
    with pytest.raises(ValueError, match="shared"):
        model.fit_cg(response, designs)
    with pytest.raises(ValueError, match="shared"):
        model.inference(response, designs)
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        model.diagnostics(
            response,
            designs,
            shared_input=inputs,
        )
    result = model.diagnostics(
        response,
        designs,
        shared_input=inputs,
        degrees_of_freedom=8,
    )
    assert math.isfinite(result.aic)


def test_minibatch_uses_training_updates_and_evaluation_diagnostics():
    dtype = torch.float64
    predictor = _ModeRecordingSharedPredictor()
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=predictor,
        dtype=dtype,
    )
    inputs = torch.linspace(-1.0, 1.0, 16, dtype=dtype).unsqueeze(-1)
    response = 0.2 + inputs.squeeze(-1)
    designs = {
        "mu": torch.ones((16, 1), dtype=dtype),
        "sigma": torch.ones((16, 1), dtype=dtype),
    }
    model.eval()

    model.fit_minibatch(
        response,
        designs,
        shared_input=inputs,
        control=MiniBatchControl(
            batch_size=8,
            epochs=1,
            minimum_epochs=1,
            patience=1,
        ),
    )

    assert predictor.training_modes[0] is False
    assert True in predictor.training_modes
    assert predictor.training_modes[-1] is False
    assert not model.training


def test_shared_benchmark_cli_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_shared.py",
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
    assert report["shared_parameters"] == ["mu", "sigma"]
    assert report["updates"] == 2
    assert report["trainable_parameters"] < report["independent_equivalent_parameters"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shared_minibatch_runs_on_cuda():
    device = torch.device("cuda")
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(806)
    inputs = torch.randn((256, 3), dtype=dtype, generator=generator).to(device)
    response = (0.2 + torch.sin(inputs[:, 0])).to(device)
    designs = {
        "mu": torch.ones((256, 1), dtype=dtype, device=device),
        "sigma": torch.ones((256, 1), dtype=dtype, device=device),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            3,
            ("mu", "sigma"),
            (8,),
        ),
        dtype=dtype,
        device=device,
    )

    result = model.fit_minibatch(
        response,
        designs,
        shared_input=inputs,
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
        shared_input=inputs,
        type="response",
    )

    assert result.updates == 8
    assert prediction["sigma"].device.type == "cuda"
    assert next(model.shared_predictor.parameters()).device.type == "cuda"
