import copy
import math
from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader

from torchgamlss import (
    BCCG,
    BCPE,
    BCT,
    GAMLSS,
    TF,
    Family,
    Gamma,
    MiniBatchControl,
    MiniBatchFitResult,
    NegativeBinomial,
    SharedMLPPredictor,
)


@dataclass(frozen=True)
class _StressCase:
    family_type: type[Family]
    regimes: tuple[dict[str, float], dict[str, float]]
    starts: dict[str, float]


_STRESS_CASES = {
    "GA": _StressCase(
        Gamma,
        (
            {"mu": 0.2, "sigma": 0.15},
            {"mu": 80.0, "sigma": 1.4},
        ),
        {"mu": 5.0, "sigma": 0.6},
    ),
    "NBI": _StressCase(
        NegativeBinomial,
        (
            {"mu": 0.2, "sigma": 0.05},
            {"mu": 80.0, "sigma": 2.0},
        ),
        {"mu": 5.0, "sigma": 0.8},
    ),
    "BCCG": _StressCase(
        BCCG,
        (
            {"mu": 0.5, "sigma": 0.08, "nu": 1.8},
            {"mu": 50.0, "sigma": 0.7, "nu": -1.8},
        ),
        {"mu": 5.0, "sigma": 0.3, "nu": 0.0},
    ),
    "BCT": _StressCase(
        BCT,
        (
            {"mu": 0.5, "sigma": 0.1, "nu": 1.4, "tau": 30.0},
            {"mu": 50.0, "sigma": 0.6, "nu": -1.4, "tau": 1.5},
        ),
        {"mu": 5.0, "sigma": 0.3, "nu": 0.0, "tau": 5.0},
    ),
    "BCPE": _StressCase(
        BCPE,
        (
            {"mu": 0.5, "sigma": 0.1, "nu": 1.4, "tau": 5.0},
            {"mu": 50.0, "sigma": 0.6, "nu": -1.4, "tau": 0.75},
        ),
        {"mu": 5.0, "sigma": 0.3, "nu": 0.0, "tau": 2.0},
    ),
    "TF": _StressCase(
        TF,
        (
            {"mu": -3.0, "sigma": 0.15, "nu": 30.0},
            {"mu": 20.0, "sigma": 4.0, "nu": 1.2},
        ),
        {"mu": 5.0, "sigma": 1.0, "nu": 5.0},
    ),
}

_TAIL_PROBABILITIES = torch.tensor(
    [
        1e-4,
        1e-3,
        0.01,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        0.99,
        0.999,
        0.9999,
    ],
    dtype=torch.float32,
)


def _stress_response(family: Family, case: _StressCase) -> torch.Tensor:
    regime_responses = []
    for regime in case.regimes:
        parameters = {
            parameter: torch.tensor(value, dtype=torch.float32)
            for parameter, value in regime.items()
        }
        regime_responses.append(
            family.quantile(_TAIL_PROBABILITIES, parameters)
        )
    return torch.cat(regime_responses).repeat(4)


def _stress_model(
    family: Family,
    case: _StressCase,
    *,
    seed: int,
) -> GAMLSS:
    torch.manual_seed(seed)
    model = GAMLSS(
        family,
        {parameter: 1 for parameter in family.parameter_names},
        shared_predictor=SharedMLPPredictor(
            4,
            family.parameter_names,
            (32, 32),
        ),
        dtype=torch.float32,
        device="cuda",
    )
    with torch.no_grad():
        for parameter, start in case.starts.items():
            link_start = family.links[parameter](
                torch.tensor(start, dtype=torch.float32, device="cuda")
            )
            model.coefficients[parameter][0].copy_(link_start)
        assert model.shared_predictor is not None
        for parameter in model.shared_predictor.parameters():
            parameter.mul_(0.05)
    return model


def _fit_stress_mode(
    base_model: GAMLSS,
    response: torch.Tensor,
    designs: dict[str, torch.Tensor],
    features: torch.Tensor,
    amp_dtype: str | None,
) -> tuple[MiniBatchFitResult, dict[str, torch.Tensor]]:
    model = copy.deepcopy(base_model)
    result = model.fit_minibatch(
        response,
        designs,
        shared_input=features,
        control=MiniBatchControl(
            batch_size=22,
            epochs=8,
            minimum_epochs=8,
            learning_rate=1e-3,
            shuffle=True,
            tolerance_change=0.0,
            tolerance_gradient=0.0,
            evaluation_frequency=8,
            clip_gradient_norm=10.0,
            amp_dtype=amp_dtype,
        ),
        generator=torch.Generator().manual_seed(8128),
    )
    prediction = model.predict(
        designs,
        shared_input=features,
        type="response",
    )
    return result, {
        parameter: values.detach()
        for parameter, values in prediction.items()
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("family_name", "case"),
    _STRESS_CASES.items(),
    ids=_STRESS_CASES,
)
def test_cuda_amp_is_stable_for_extreme_family_regimes(family_name, case):
    family = case.family_type()
    response = _stress_response(family, case).cuda()
    observation_count = response.numel()
    feature_generator = torch.Generator().manual_seed(
        9300 + tuple(_STRESS_CASES).index(family_name)
    )
    features = torch.randn(
        (observation_count, 4),
        dtype=torch.float32,
        generator=feature_generator,
    )
    features[:, 0] = torch.tensor(
        [-1.0] * _TAIL_PROBABILITIES.numel()
        + [1.0] * _TAIL_PROBABILITIES.numel(),
        dtype=torch.float32,
    ).repeat(4)
    features = features.cuda()
    designs = {
        parameter: torch.ones(
            (observation_count, 1),
            dtype=torch.float32,
            device="cuda",
        )
        for parameter in family.parameter_names
    }
    base_model = _stress_model(
        family,
        case,
        seed=9400 + tuple(_STRESS_CASES).index(family_name),
    )
    modes = [None, "float16"]
    if torch.cuda.is_bf16_supported():
        modes.append("bfloat16")

    fits = {
        mode: _fit_stress_mode(
            base_model,
            response,
            designs,
            features,
            mode,
        )
        for mode in modes
    }
    baseline_result, baseline_prediction = fits[None]

    assert float(response.max()) > 1_000.0
    assert all(
        math.isfinite(value)
        for value in baseline_result.objective_history
    )
    for mode, (result, prediction) in fits.items():
        assert result.updates == 32
        assert result.updates - result.skipped_updates > 0
        assert math.isfinite(result.negative_log_likelihood)
        assert math.isfinite(result.penalized_objective)
        assert math.isfinite(result.gradient_max)
        assert all(math.isfinite(value) for value in result.objective_history)
        assert result.penalized_objective < result.objective_history[0]
        assert all(torch.isfinite(values).all() for values in prediction.values())
        assert result.objective_history[0] == baseline_result.objective_history[0]
        if mode != "float16":
            assert result.skipped_updates == 0

        relative_nll_difference = abs(
            result.negative_log_likelihood
            - baseline_result.negative_log_likelihood
        ) / max(1.0, abs(baseline_result.negative_log_likelihood))
        tolerance = 0.03 if mode == "float16" else 5e-4
        assert relative_nll_difference < tolerance, (
            mode,
            result.skipped_updates,
            relative_nll_difference,
        )

        for parameter, baseline_values in baseline_prediction.items():
            prediction_scale = max(
                1.0,
                float(baseline_values.abs().max()),
            )
            relative_prediction_difference = float(
                (
                    prediction[parameter] - baseline_values
                ).abs().max()
            ) / prediction_scale
            assert relative_prediction_difference < tolerance, (
                mode,
                parameter,
                relative_prediction_difference,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("data_path", ["tensor", "loader"])
def test_fp16_overflow_only_epoch_does_not_advance_scheduler(data_path):
    family = Gamma()
    observation_count = 16
    torch.manual_seed(9500)
    model = GAMLSS(
        family,
        {"mu": 1, "sigma": 1},
        shared_predictor=SharedMLPPredictor(
            2,
            family.parameter_names,
            (8,),
        ),
        dtype=torch.float32,
        device="cuda",
    )
    with torch.no_grad():
        model.coefficients["mu"][0].copy_(
            torch.log(torch.tensor(5.0, device="cuda"))
        )
        model.coefficients["sigma"][0].copy_(
            torch.log(torch.tensor(0.6, device="cuda"))
        )
        assert model.shared_predictor is not None
        for parameter in model.shared_predictor.parameters():
            parameter.mul_(0.05)
    designs = {
        parameter: torch.ones(
            (observation_count, 1),
            dtype=torch.float32,
            device="cuda",
        )
        for parameter in family.parameter_names
    }

    response = torch.full(
        (observation_count,),
        10_000.0,
        dtype=torch.float32,
        device="cuda",
    )
    shared_input = torch.ones(
        (observation_count, 2),
        dtype=torch.float32,
        device="cuda",
    )
    control = MiniBatchControl(
        batch_size=observation_count,
        epochs=1,
        minimum_epochs=1,
        learning_rate=1e-3,
        learning_rate_decay=0.5,
        tolerance_change=0.0,
        tolerance_gradient=0.0,
        amp_dtype="float16",
    )
    if data_path == "tensor":
        result = model.fit_minibatch(
            response,
            designs,
            shared_input=shared_input,
            control=control,
        )
    else:
        result = model.fit_minibatch_loader(
            DataLoader(
                [
                    {
                        "response": response.cpu(),
                        "design_matrices": {
                            parameter: design.cpu()
                            for parameter, design in designs.items()
                        },
                        "shared_input": shared_input.cpu(),
                    }
                ],
                batch_size=None,
            ),
            control=control,
        )

    assert result.updates == 1
    assert result.skipped_updates == 1
    assert result.final_learning_rate == result.learning_rate
