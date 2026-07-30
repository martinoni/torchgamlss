import math

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    CGControl,
    MiniBatchControl,
    Normal,
    PSpline,
    RSControl,
    TensorInteractionSmooth,
    TensorProductSmooth,
)


def _tensor_data(observation_count: int = 80) -> pd.DataFrame:
    x = torch.linspace(-1.0, 1.0, observation_count, dtype=torch.float64)
    z = torch.sin(
        torch.linspace(0.15, 5.6, observation_count, dtype=torch.float64)
    )
    y = (
        0.7
        + torch.sin(2.0 * x)
        + 0.5 * x * z
        + 0.2 * z.square()
        + 0.05
        * torch.cos(torch.arange(observation_count, dtype=torch.float64))
    )
    return pd.DataFrame({"y": y.numpy(), "x": x.numpy(), "z": z.numpy()})


def _manual_marginals(covariates: torch.Tensor):
    return (
        PSpline(
            float(covariates[:, 0].min()),
            float(covariates[:, 0].max()),
            2.0,
            intervals=3,
            dtype=covariates.dtype,
            device=covariates.device,
        ),
        PSpline(
            float(covariates[:, 1].min()),
            float(covariates[:, 1].max()),
            5.0,
            intervals=2,
            dtype=covariates.dtype,
            device=covariates.device,
        ),
    )


def test_formula_te_matches_manual_constrained_tensor_construction():
    data = _tensor_data()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ te(x, z, lambda_=(2, 5), inter=(3, 2), "
                "name='surface')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data)
    covariates = prepared.smooth_covariates["mu"]["surface"]
    term = model.smooth_terms["mu"]["surface"]
    manual = TensorProductSmooth(
        _manual_marginals(covariates),
        smoothing_parameters=(2.0, 5.0),
        training_covariates=covariates,
    )

    assert model.formula_column_names["mu"] == ("Intercept",)
    assert isinstance(term, TensorProductSmooth)
    assert not isinstance(term, TensorInteractionSmooth)
    assert term.smoothing_parameters == (2.0, 5.0)
    assert term.coefficient_shape == (6, 5)
    assert term.coefficients.numel() == 29
    assert term.constraint_absorbed
    assert term.constraints(covariates).shape == (0, 29)
    torch.testing.assert_close(term.design(covariates), manual.design(covariates))
    torch.testing.assert_close(
        term.design(covariates).sum(dim=0),
        torch.zeros(29, dtype=torch.float64),
        rtol=0.0,
        atol=1e-13,
    )
    for actual, expected in zip(
        term.penalty_matrices(),
        manual.penalty_matrices(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    with torch.no_grad():
        term.coefficients.copy_(
            torch.linspace(
                -0.2,
                0.3,
                term.coefficients.numel(),
                dtype=torch.float64,
            )
        )
    restored = TensorProductSmooth(
        _manual_marginals(covariates),
        smoothing_parameters=(2.0, 5.0),
        training_covariates=covariates,
    )
    restored.load_state_dict(term.state_dict())
    evaluation = torch.tensor(
        [[-0.8, 0.2], [-0.1, -0.6], [0.5, 0.7]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        restored(evaluation),
        term(evaluation),
    )


def test_formula_ti_persists_marginal_constraints_for_new_data():
    data = _tensor_data()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ x + z + ti(x, z, smoothing_parameters=(2, 5), "
                "intervals=(3, 2), name='interaction')"
            ),
            "sigma": "~ 1",
        },
        data,
        dtype=torch.float32,
    )
    training = model.prepare_formula_data(data)
    term = model.smooth_terms["mu"]["interaction"]
    new_data = pd.DataFrame(
        {
            "x": [-0.85, -0.2, 0.35, 0.9],
            "z": [0.1, -0.7, 0.8, -0.2],
        }
    )
    evaluation = model.prepare_formula_data(new_data)
    new_covariates = evaluation.smooth_covariates["mu"]["interaction"]
    with torch.no_grad():
        term.coefficients.copy_(
            torch.linspace(
                -0.2,
                0.3,
                term.coefficients.numel(),
                dtype=torch.float32,
            )
        )
        model.coefficients["mu"].copy_(
            torch.tensor([0.4, 0.2, -0.1], dtype=torch.float32)
        )

    predictions = model.predict_data(new_data, type="terms")

    assert isinstance(term, TensorInteractionSmooth)
    assert term.coefficient_shape == (5, 4)
    assert training.smooth_covariates["mu"]["interaction"].shape == (80, 2)
    assert new_covariates.shape == (4, 2)
    assert new_covariates.dtype == torch.float32
    torch.testing.assert_close(
        predictions["mu"].smooth["interaction"],
        term.predict_design(new_covariates) @ term.coefficients,
    )


def test_formula_te_full_batch_and_minibatch_fits_are_finite():
    data = _tensor_data()
    formulas = {
        "mu": (
            "y ~ te(x, z, smoothing_parameters=(2, 5), "
            "intervals=(3, 2), name='surface')"
        ),
        "sigma": "~ 1",
    }
    full_batch = GAMLSS.from_formula(Normal(), formulas, data)
    prepared = full_batch.prepare_formula_data(data, include_response=True)
    assert prepared.response is not None
    initial_loss = full_batch.negative_log_likelihood(
        prepared.response,
        prepared.design_matrices,
        smooth_covariates=prepared.smooth_covariates,
    ).detach()

    result = full_batch.fit_data(
        data,
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
    )

    assert result.converged
    assert math.isfinite(result.negative_log_likelihood)
    assert result.negative_log_likelihood < float(initial_loss)

    minibatch = GAMLSS.from_formula(Normal(), formulas, data)
    minibatch_result = minibatch.fit_minibatch_data(
        data,
        control=MiniBatchControl(
            batch_size=16,
            epochs=3,
            learning_rate=0.01,
            shuffle=False,
            minimum_epochs=3,
            patience=10,
            evaluation_frequency=1,
        ),
    )

    assert minibatch_result.updates == 15
    assert math.isfinite(minibatch_result.negative_log_likelihood)
    assert math.isfinite(minibatch_result.penalized_objective)


@pytest.mark.parametrize(
    ("mu_formula", "term_name", "linear_size"),
    [
        (
            (
                "y ~ te(x, z, lambda_=(2, 5), intervals=(3, 2), "
                "name='surface')"
            ),
            "surface",
            1,
        ),
        (
            (
                "y ~ x + z + ti(x, z, lambda_=(2, 5), intervals=(3, 2), "
                "name='interaction')"
            ),
            "interaction",
            3,
        ),
    ],
)
def test_formula_tensor_rs_and_cg_fits_agree(
    mu_formula,
    term_name,
    linear_size,
):
    data = _tensor_data()
    formulas = {"mu": mu_formula, "sigma": "~ 1"}
    rs_model = GAMLSS.from_formula(Normal(), formulas, data)
    rs_result = rs_model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-7,
            max_outer_iterations=60,
            inner_tolerance=1e-7,
            max_inner_iterations=60,
            backfitting_tolerance=1e-7,
            max_backfitting_iterations=60,
        ),
    )
    cg_model = GAMLSS.from_formula(Normal(), formulas, data)
    cg_result = cg_model.fit_cg_data(
        data,
        control=CGControl(
            outer_tolerance=1e-7,
            max_outer_iterations=60,
            inner_tolerance=1e-7,
            max_inner_iterations=60,
            backfitting_tolerance=1e-7,
        ),
    )

    assert rs_result.converged
    assert cg_result.converged
    assert rs_result.global_deviance == pytest.approx(
        cg_result.global_deviance,
        rel=1e-9,
        abs=2e-8,
    )
    assert rs_result.smoothing_parameters["mu"][term_name] == (2.0, 5.0)
    assert cg_result.smoothing_parameters["mu"][term_name] == (2.0, 5.0)
    assert rs_result.smoothing_iterations["mu"][term_name] == 0
    assert cg_result.smoothing_iterations["mu"][term_name] == 0
    assert rs_result.parameter_effective_degrees_of_freedom[
        "mu"
    ] == pytest.approx(
        linear_size
        + rs_result.smooth_effective_degrees_of_freedom["mu"][term_name]
    )
    rs_parameters = rs_model.predict_data(data)
    cg_parameters = cg_model.predict_data(data)
    torch.testing.assert_close(
        rs_parameters["mu"],
        cg_parameters["mu"],
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(
        rs_parameters["sigma"],
        cg_parameters["sigma"],
        rtol=1e-8,
        atol=1e-8,
    )
    marginal_inference = rs_model.smooth_inference_data(data)["mu"][term_name]
    joint_inference = rs_model.smooth_joint_inference_data(data)["mu"][term_name]
    assert marginal_inference.smoothing_parameter == (2.0, 5.0)
    assert joint_inference.smoothing_parameter == (2.0, 5.0)
    assert torch.isfinite(marginal_inference.standard_errors).all()
    assert torch.isfinite(joint_inference.standard_errors).all()
    assert (marginal_inference.standard_errors > 0).all()
    assert (joint_inference.standard_errors > 0).all()


def test_formula_tensor_analytic_inference_supports_new_data_and_bands():
    data = _tensor_data()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ te(x, z, lambda_=(2, 5), intervals=(3, 2), "
                "name='surface')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    fit = model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-7,
            max_outer_iterations=60,
            inner_tolerance=1e-7,
            max_inner_iterations=60,
            backfitting_tolerance=1e-7,
            max_backfitting_iterations=60,
        ),
    )
    assert fit.converged
    new_data = data.iloc[[0, 10, 25, 50, 79]].drop(columns="y")

    marginal = model.smooth_inference_data(
        data,
        new_data=new_data,
    )["mu"]["surface"]
    joint_result = model.smooth_joint_inference_data(
        data,
        new_data=new_data,
    )
    joint = joint_result["mu"]["surface"]
    band = joint.simultaneous_confidence_band(
        simulations=100,
        generator=torch.Generator().manual_seed(2026),
    )

    assert marginal.smoothing_parameter == (2.0, 5.0)
    assert joint.smoothing_parameter == (2.0, 5.0)
    assert marginal.covariate.shape == (5, 2)
    assert (marginal.standard_errors > 0).all()
    assert (joint.standard_errors > 0).all()
    torch.testing.assert_close(
        marginal.covariance_matrix,
        marginal.covariance_matrix.mT,
    )
    torch.testing.assert_close(
        joint.covariance_matrix,
        marginal.covariance_matrix,
        rtol=1e-10,
        atol=1e-12,
    )
    assert float(
        torch.linalg.eigvalsh(joint_result.coefficient_covariance_matrix).min()
    ) >= -1e-12
    assert marginal.to_dataframe().columns.tolist() == [
        "covariate_0",
        "covariate_1",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
    ]
    assert band.to_dataframe().columns.tolist() == [
        "covariate_0",
        "covariate_1",
        "estimate",
        "ci_lower",
        "ci_upper",
    ]


@pytest.mark.parametrize(
    ("mu_formula", "term_name", "algorithm"),
    [
        (
            (
                "y ~ te(x, z, lambda_=(2, 5), intervals=(2, 2), "
                "name='surface')"
            ),
            "surface",
            "rs",
        ),
        (
            (
                "y ~ x + z + ti(x, z, lambda_=(2, 5), intervals=(2, 2), "
                "name='interaction')"
            ),
            "interaction",
            "cg",
        ),
    ],
)
def test_formula_tensor_bootstrap_tracks_each_penalty_reproducibly(
    mu_formula,
    term_name,
    algorithm,
):
    data = _tensor_data(observation_count=36)
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": mu_formula, "sigma": "~ 1"},
        data,
    )
    if algorithm == "rs":
        control = RSControl(
            outer_tolerance=1e-5,
            max_outer_iterations=30,
            inner_tolerance=1e-5,
            max_inner_iterations=30,
            backfitting_tolerance=1e-5,
            max_backfitting_iterations=30,
        )
        fit = model.fit_rs_data(data, control=control)
    else:
        control = CGControl(
            outer_tolerance=1e-5,
            max_outer_iterations=30,
            inner_tolerance=1e-5,
            max_inner_iterations=30,
            backfitting_tolerance=1e-5,
        )
        fit = model.fit_cg_data(data, control=control)
    assert fit.converged
    new_data = data.iloc[[0, 8, 20, 35]].drop(columns="y")

    first = model.smooth_joint_bootstrap_data(
        data,
        new_data=new_data,
        replicates=10,
        max_attempts=20,
        algorithm=algorithm,
        control=control,
        generator=torch.Generator().manual_seed(2026),
    )
    second = model.smooth_joint_bootstrap_data(
        data,
        new_data=new_data,
        replicates=10,
        max_attempts=20,
        algorithm=algorithm,
        control=control,
        generator=torch.Generator().manual_seed(2026),
    )
    curve = first["mu"][term_name]
    second_curve = second["mu"][term_name]

    assert first.term_order == (("mu", term_name),)
    assert first.smoothing_parameter_labels == (
        ("mu", term_name, 0),
        ("mu", term_name, 1),
    )
    assert first.smoothing_parameter_slices == {
        ("mu", term_name): slice(0, 2)
    }
    assert curve.smoothing_parameter == (2.0, 5.0)
    assert curve.smoothing_parameter_count == 2
    assert curve.bootstrap_estimates.shape == (10, len(new_data))
    assert curve.bootstrap_smoothing_parameters.shape == (10, 2)
    torch.testing.assert_close(
        curve.bootstrap_smoothing_parameters,
        torch.tensor(
            [[2.0, 5.0]] * 10,
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        curve.bootstrap_estimates,
        second_curve.bootstrap_estimates,
    )
    torch.testing.assert_close(
        curve.bootstrap_smoothing_parameters,
        second_curve.bootstrap_smoothing_parameters,
    )
    torch.testing.assert_close(
        curve.smoothing_parameter_bootstrap_mean,
        torch.tensor([2.0, 5.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        curve.smoothing_parameter_standard_error,
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        curve.smoothing_parameter_bias,
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        curve.smoothing_parameter_confidence_interval,
        torch.tensor([[2.0, 2.0], [5.0, 5.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        first.smoothing_parameters,
        torch.tensor([2.0, 5.0], dtype=torch.float64),
    )
    assert first.bootstrap_smoothing_parameters.shape == (10, 2)
    assert first.smoothing_parameter_covariance_matrix.shape == (2, 2)
    assert (curve.standard_errors > 0).all()
    assert curve.confidence_intervals.shape == (len(new_data), 2)
    assert curve.to_dataframe().columns.tolist() == [
        "covariate_0",
        "covariate_1",
        "estimate",
        "bootstrap_mean",
        "bias",
        "standard_error",
        "ci_lower",
        "ci_upper",
    ]


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        (
            "y ~ te(x, z, lambda_=(2,))",
            "one value per marginal",
        ),
        (
            "y ~ ti(x, z, lambda_=(2, 5), center=True)",
            "does not accept the center",
        ),
        (
            "y ~ te(x, z, lambda_=(2, 5), intervals=(3,))",
            "one value per marginal",
        ),
        (
            "y ~ te(x, z, lambda_=(2, 5), intervals=2.5)",
            "must be integers",
        ),
        (
            (
                "y ~ te(x, z, lambda_=(2, 5), "
                "initial_lambda_=(3, 4))"
            ),
            "either smoothing_parameters or initial_smoothing_parameters",
        ),
        (
            "y ~ te(x, z, initial_lambda_=(0, 4))",
            "must start from positive",
        ),
    ],
)
def test_invalid_tensor_formula_options_are_rejected(formula, match):
    with pytest.raises(ValueError, match=match):
        GAMLSS.from_formula(
            Normal(),
            {"mu": formula, "sigma": "~ 1"},
            _tensor_data(30),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_formula_construction_and_prediction_run_on_cuda():
    data = _tensor_data(30)
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ te(x, z, lambda_=(2, 5), intervals=(3, 2), "
                "name='surface')"
            ),
            "sigma": "~ 1",
        },
        data,
        device="cuda",
    )
    result = model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-6,
            max_outer_iterations=60,
            inner_tolerance=1e-6,
            max_inner_iterations=60,
            backfitting_tolerance=1e-6,
            max_backfitting_iterations=60,
        ),
    )
    prepared = model.prepare_formula_data(data)
    prediction = model.predict_data(data)
    inference = model.smooth_inference_data(data)["mu"]["surface"]
    term = model.smooth_terms["mu"]["surface"]

    assert result.converged
    assert prepared.smooth_covariates["mu"]["surface"].is_cuda
    assert term.coefficients.is_cuda
    assert prediction["mu"].is_cuda
    assert inference.standard_errors.is_cuda
    assert torch.isfinite(prediction["mu"]).all()
    assert torch.isfinite(inference.standard_errors).all()
