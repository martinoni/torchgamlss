import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from torchgamlss import (
    GAMLSS,
    MiniBatchControl,
    MLPPredictor,
    Normal,
    PSpline,
    SharedMLPPredictor,
)

PROJECT_ROOT = Path(__file__).parent.parent


def _linear_data(
    observation_count: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(
        observation_count,
        dtype=dtype,
        generator=generator,
    )
    response = (
        0.5
        + 1.2 * x
        + 0.3
        * torch.randn(
            observation_count,
            dtype=dtype,
            generator=generator,
        )
    )
    designs = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((observation_count, 1), dtype=dtype),
    }
    return x, response, designs


class _RowDataset(Dataset):
    def __init__(
        self,
        response,
        design_matrices,
        *,
        weights=None,
        offsets=None,
        smooth_covariates=None,
        neural_inputs=None,
        shared_input=None,
    ):
        self.response = response
        self.design_matrices = design_matrices
        self.weights = weights
        self.offsets = offsets or {}
        self.smooth_covariates = smooth_covariates or {}
        self.neural_inputs = neural_inputs or {}
        self.shared_input = shared_input

    def __len__(self):
        return self.response.numel()

    def __getitem__(self, index):
        item = {
            "response": self.response[index],
            "design_matrices": {
                parameter: design[index]
                for parameter, design in self.design_matrices.items()
            },
        }
        if self.weights is not None:
            item["weights"] = self.weights[index]
        if self.offsets:
            item["offsets"] = {
                parameter: offset[index]
                for parameter, offset in self.offsets.items()
            }
        if self.smooth_covariates:
            item["smooth_covariates"] = {
                parameter: {
                    term: covariate[index]
                    for term, covariate in parameter_covariates.items()
                }
                for parameter, parameter_covariates in (
                    self.smooth_covariates.items()
                )
            }
        if self.neural_inputs:
            item["neural_inputs"] = {
                parameter: inputs[index]
                for parameter, inputs in self.neural_inputs.items()
            }
        if self.shared_input is not None:
            item["shared_input"] = self.shared_input[index]
        return item


def _fixed_epoch_control(*, evaluation_frequency=2):
    return MiniBatchControl(
        batch_size=999,
        epochs=5,
        learning_rate=0.025,
        learning_rate_decay=0.98,
        shuffle=False,
        minimum_epochs=5,
        patience=10,
        tolerance_change=0.0,
        tolerance_gradient=0.0,
        evaluation_frequency=evaluation_frequency,
    )


def test_dataloader_matches_tensor_updates_with_weights_and_unequal_batches():
    _, response, designs = _linear_data(23, seed=101)
    weights = torch.linspace(0.5, 1.5, 23, dtype=torch.float64)
    weights[:7] = 0.0
    offsets = {
        "mu": torch.linspace(-0.1, 0.1, 23, dtype=torch.float64),
    }
    tensor_model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    loader_model = copy.deepcopy(tensor_model)
    control = _fixed_epoch_control()

    tensor_result = tensor_model.fit_minibatch(
        response,
        designs,
        weights=weights,
        offsets=offsets,
        control=MiniBatchControl(
            **{
                **control.__dict__,
                "batch_size": 7,
            }
        ),
    )
    loader = DataLoader(
        _RowDataset(
            response,
            designs,
            weights=weights,
            offsets=offsets,
        ),
        batch_size=7,
        shuffle=False,
    )
    loader_result = loader_model.fit_minibatch_loader(
        loader,
        control=control,
    )

    assert loader_result.batch_size == 7
    assert loader_result.updates == math.ceil(23 / 7) * control.epochs
    assert loader_result.evaluation_epochs == tensor_result.evaluation_epochs
    assert loader_result.objective_history == pytest.approx(
        tensor_result.objective_history,
        rel=1e-14,
        abs=1e-14,
    )
    assert loader_result.negative_log_likelihood == pytest.approx(
        tensor_result.negative_log_likelihood,
        rel=1e-14,
        abs=1e-14,
    )
    for parameter in tensor_model.family.parameter_names:
        torch.testing.assert_close(
            loader_model.coefficients[parameter],
            tensor_model.coefficients[parameter],
            rtol=1e-14,
            atol=1e-14,
        )


def test_dataloader_diagnostics_do_not_change_future_shuffle_order():
    _, response, designs = _linear_data(41, seed=201)
    dataset = _RowDataset(response, designs)
    model_each_epoch = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    model_last_epoch = copy.deepcopy(model_each_epoch)
    loaders = [
        DataLoader(
            dataset,
            batch_size=9,
            shuffle=True,
            generator=torch.Generator().manual_seed(202),
        )
        for _ in range(2)
    ]

    model_each_epoch.fit_minibatch_loader(
        loaders[0],
        control=_fixed_epoch_control(evaluation_frequency=1),
    )
    model_last_epoch.fit_minibatch_loader(
        loaders[1],
        control=_fixed_epoch_control(evaluation_frequency=5),
    )

    for parameter in model_each_epoch.family.parameter_names:
        torch.testing.assert_close(
            model_each_epoch.coefficients[parameter],
            model_last_epoch.coefficients[parameter],
            rtol=0.0,
            atol=0.0,
        )


def test_dataloader_validation_stops_and_restores_epoch_zero():
    _, response, designs = _linear_data(53, seed=301)
    _, validation_response, validation_designs = _linear_data(29, seed=302)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    initial_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    training_loader = DataLoader(
        _RowDataset(response, designs),
        batch_size=11,
        shuffle=True,
        generator=torch.Generator().manual_seed(303),
    )
    validation_loader = DataLoader(
        _RowDataset(validation_response, validation_designs),
        batch_size=7,
        shuffle=False,
    )

    result = model.fit_minibatch_loader(
        training_loader,
        validation_loader=validation_loader,
        control=MiniBatchControl(
            epochs=10,
            learning_rate=0.04,
            minimum_epochs=2,
            evaluation_frequency=1,
            validation_patience=1,
            validation_minimum_delta=1e6,
        ),
    )

    assert result.stop_reason == "validation"
    assert result.epochs == 2
    assert result.best_epoch == 0
    assert result.restored_best_parameters
    assert result.validation_epochs == (0, 1, 2)
    assert result.validation_negative_log_likelihood is not None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            initial_state[name],
            rtol=0.0,
            atol=0.0,
        )


def test_dataloader_supports_nested_smooth_and_neural_inputs():
    x, response, designs = _linear_data(37, seed=351)
    smooth = PSpline.from_data(
        x,
        smoothing_parameter=5.0,
        intervals=6,
    )
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"trend": smooth}},
        neural_predictors={"mu": MLPPredictor(1, (4,))},
        dtype=torch.float64,
    )
    loader = DataLoader(
        _RowDataset(
            response,
            designs,
            smooth_covariates={"mu": {"trend": x}},
            neural_inputs={"mu": x.unsqueeze(-1)},
        ),
        batch_size=8,
        shuffle=False,
    )

    result = model.fit_minibatch_loader(
        loader,
        control=MiniBatchControl(
            epochs=2,
            minimum_epochs=2,
            evaluation_frequency=1,
        ),
    )

    assert result.updates == math.ceil(37 / 8) * 2
    assert math.isfinite(result.penalized_objective)
    assert model.smooth_terms["mu"]["trend"].coefficients.grad is None


def test_dataloader_rejects_drop_last_and_late_invalid_batches_before_updates():
    _, response, designs = _linear_data(8, seed=401)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    dropped_loader = DataLoader(
        _RowDataset(response, designs),
        batch_size=3,
        drop_last=True,
    )
    with pytest.raises(ValueError, match="drop_last=False"):
        model.fit_minibatch_loader(dropped_loader)

    class InvalidLastRow(_RowDataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            if index == len(self) - 1:
                item["design_matrices"]["mu"] = torch.ones(
                    3,
                    dtype=torch.float64,
                )
            return item

    invalid_loader = DataLoader(
        InvalidLastRow(response, designs),
        batch_size=1,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="training loader batch 8"):
        model.fit_minibatch_loader(invalid_loader)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            state_before[name],
            rtol=0.0,
            atol=0.0,
        )


def test_dataloader_api_controls_are_validated():
    _, response, designs = _linear_data(4, seed=450)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    loader = DataLoader(
        _RowDataset(response, designs),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="torch DataLoader"):
        model.fit_minibatch_loader([])
    with pytest.raises(ValueError, match="non_blocking"):
        model.fit_minibatch_loader(loader, non_blocking=1)
    with pytest.raises(ValueError, match="validation loader"):
        model.fit_minibatch_loader(loader, validation_loader=[])
    with pytest.raises(ValueError, match="checkpoint_frequency"):
        model.fit_minibatch_loader(loader, checkpoint_frequency=0)


def test_checkpoint_resume_exactly_matches_uninterrupted_dropout_fit(tmp_path):
    x, response, designs = _linear_data(43, seed=601)
    dataset = _RowDataset(
        response,
        {
            "mu": designs["mu"][:, :1],
            "sigma": designs["sigma"],
        },
        shared_input=x.unsqueeze(-1),
    )
    initial_model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            1,
            ("mu",),
            (8,),
            dropout=0.25,
        ),
        dtype=torch.float64,
    )
    continuous_model = copy.deepcopy(initial_model)
    partial_model = copy.deepcopy(initial_model)
    resumed_model = copy.deepcopy(initial_model)

    def loader(seed):
        return DataLoader(
            dataset,
            batch_size=9,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def control(epochs):
        return MiniBatchControl(
            epochs=epochs,
            learning_rate=0.02,
            learning_rate_decay=0.97,
            minimum_epochs=1,
            patience=20,
            tolerance_change=0.0,
            tolerance_gradient=0.0,
            evaluation_frequency=1,
        )

    torch.manual_seed(602)
    continuous_result = continuous_model.fit_minibatch_loader(
        loader(603),
        control=control(4),
    )
    checkpoint_path = tmp_path / "dropout-checkpoint.pt"
    torch.manual_seed(602)
    partial_model.fit_minibatch_loader(
        loader(603),
        control=control(2),
        checkpoint_path=checkpoint_path,
    )
    resumed_result = resumed_model.fit_minibatch_loader(
        loader(999),
        control=control(4),
        resume_from=checkpoint_path,
    )

    assert checkpoint_path.is_file()
    assert resumed_result == continuous_result
    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_checkpoint_preserves_validation_patience_and_best_state(tmp_path):
    _, response, designs = _linear_data(41, seed=611)
    _, validation_response, validation_designs = _linear_data(23, seed=612)
    dataset = _RowDataset(response, designs)
    validation_dataset = _RowDataset(
        validation_response,
        validation_designs,
    )
    initial_model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    continuous_model = copy.deepcopy(initial_model)
    partial_model = copy.deepcopy(initial_model)
    resumed_model = copy.deepcopy(initial_model)

    def loaders(seed):
        return (
            DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            ),
            DataLoader(validation_dataset, batch_size=6),
        )

    def control(epochs):
        return MiniBatchControl(
            epochs=epochs,
            learning_rate=0.03,
            minimum_epochs=1,
            evaluation_frequency=1,
            validation_patience=3,
            validation_minimum_delta=1e6,
        )

    torch.manual_seed(613)
    training_loader, validation_loader = loaders(614)
    continuous_result = continuous_model.fit_minibatch_loader(
        training_loader,
        validation_loader=validation_loader,
        control=control(10),
    )
    checkpoint_path = tmp_path / "validation-checkpoint.pt"
    torch.manual_seed(613)
    training_loader, validation_loader = loaders(614)
    partial_model.fit_minibatch_loader(
        training_loader,
        validation_loader=validation_loader,
        control=control(1),
        checkpoint_path=checkpoint_path,
    )
    training_loader, validation_loader = loaders(999)
    resumed_result = resumed_model.fit_minibatch_loader(
        training_loader,
        validation_loader=validation_loader,
        control=control(10),
        resume_from=checkpoint_path,
    )

    assert continuous_result.stop_reason == "validation"
    assert continuous_result.epochs == 3
    assert resumed_result == continuous_result
    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_checkpoint_rejects_invalid_schema_and_changed_control(tmp_path):
    _, response, designs = _linear_data(12, seed=621)
    dataset = _RowDataset(response, designs)

    def loader():
        return DataLoader(dataset, batch_size=4, shuffle=False)

    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    malformed_path = tmp_path / "malformed.pt"
    torch.save({"format": "not-a-checkpoint"}, malformed_path)
    with pytest.raises(ValueError, match="invalid schema"):
        model.fit_minibatch_loader(
            loader(),
            control=MiniBatchControl(epochs=1, minimum_epochs=1),
            resume_from=malformed_path,
        )

    checkpoint_path = tmp_path / "valid.pt"
    model.fit_minibatch_loader(
        loader(),
        control=MiniBatchControl(
            epochs=1,
            minimum_epochs=1,
            learning_rate=0.01,
        ),
        checkpoint_path=checkpoint_path,
    )
    resumed_model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="learning_rate"):
        resumed_model.fit_minibatch_loader(
            loader(),
            control=MiniBatchControl(
                epochs=2,
                minimum_epochs=1,
                learning_rate=0.02,
            ),
            resume_from=checkpoint_path,
        )


def test_dataloader_rejects_a_stream_that_changes_between_passes():
    _, response, designs = _linear_data(5, seed=460)

    class ChangingStream(IterableDataset):
        def __init__(self):
            self.passes = 0

        def __iter__(self):
            self.passes += 1
            rows = 5 if self.passes == 1 else 4
            yield {
                "response": response[:rows],
                "design_matrices": {
                    parameter: design[:rows]
                    for parameter, design in designs.items()
                },
            }

    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        dtype=torch.float64,
    )
    loader = DataLoader(ChangingStream(), batch_size=None)

    with pytest.raises(ValueError, match="changed its observation count"):
        model.fit_minibatch_loader(
            loader,
            control=MiniBatchControl(
                epochs=1,
                minimum_epochs=1,
            ),
        )


def test_streaming_benchmark_cli_smoke(tmp_path):
    checkpoint_path = tmp_path / "benchmark.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_dataloader.py",
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
            "32",
            "--epochs",
            "1",
            "--minimum-epochs",
            "1",
            "--checkpoint-path",
            str(checkpoint_path),
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

    assert report["rows"] == 128
    assert report["validation_rows"] == 64
    assert report["batch_size"] == 32
    assert report["updates"] == 4
    assert report["validation_negative_log_likelihood"] is not None
    assert report["checkpoint_enabled"]
    assert report["checkpoint_bytes"] == checkpoint_path.stat().st_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_dataloader_streams_shared_inputs_to_cuda():
    x, response, designs = _linear_data(
        127,
        seed=501,
        dtype=torch.float32,
    )
    validation_x, validation_response, validation_designs = _linear_data(
        61,
        seed=502,
        dtype=torch.float32,
    )
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            1,
            ("mu",),
            (8,),
        ),
        dtype=torch.float32,
        device="cuda",
    )
    training_loader = DataLoader(
        _RowDataset(
            response,
            {
                "mu": designs["mu"][:, :1],
                "sigma": designs["sigma"],
            },
            shared_input=x.unsqueeze(-1),
        ),
        batch_size=31,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(503),
    )
    validation_loader = DataLoader(
        _RowDataset(
            validation_response,
            {
                "mu": validation_designs["mu"][:, :1],
                "sigma": validation_designs["sigma"],
            },
            shared_input=validation_x.unsqueeze(-1),
        ),
        batch_size=17,
        pin_memory=True,
    )

    result = model.fit_minibatch_loader(
        training_loader,
        validation_loader=validation_loader,
        non_blocking=True,
        control=MiniBatchControl(
            epochs=2,
            minimum_epochs=2,
            validation_patience=2,
        ),
    )

    assert math.isfinite(result.negative_log_likelihood)
    assert result.validation_negative_log_likelihood is not None
    assert next(model.parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_checkpoint_restores_device_rng_exactly(tmp_path):
    x, response, designs = _linear_data(
        47,
        seed=701,
        dtype=torch.float32,
    )
    dataset = _RowDataset(
        response,
        {
            "mu": designs["mu"][:, :1],
            "sigma": designs["sigma"],
        },
        shared_input=x.unsqueeze(-1),
    )
    initial_model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            1,
            ("mu",),
            (8,),
            dropout=0.2,
        ),
        dtype=torch.float32,
        device="cuda",
    )
    continuous_model = copy.deepcopy(initial_model)
    partial_model = copy.deepcopy(initial_model)
    resumed_model = copy.deepcopy(initial_model)

    def loader(seed):
        return DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=True,
        )

    def control(epochs):
        return MiniBatchControl(
            epochs=epochs,
            minimum_epochs=1,
            patience=10,
            tolerance_change=0.0,
            tolerance_gradient=0.0,
            evaluation_frequency=1,
        )

    torch.manual_seed(702)
    continuous_result = continuous_model.fit_minibatch_loader(
        loader(703),
        control=control(3),
        non_blocking=True,
    )
    checkpoint_path = tmp_path / "cuda.pt"
    torch.manual_seed(702)
    partial_model.fit_minibatch_loader(
        loader(703),
        control=control(1),
        non_blocking=True,
        checkpoint_path=checkpoint_path,
    )
    resumed_result = resumed_model.fit_minibatch_loader(
        loader(999),
        control=control(3),
        non_blocking=True,
        resume_from=checkpoint_path,
    )

    assert resumed_result == continuous_result
    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
