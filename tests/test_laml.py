import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    GAMLSSLAMLResult,
    LAMLControl,
    Normal,
    Poisson,
    PSpline,
    RSControl,
    fit_gamlss_laml,
    fit_normal_laml,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _mgcv_system():
    design_frame = pd.read_csv(REFERENCE_DIR / "mgcv_laml_design.csv")
    reference = pd.read_csv(REFERENCE_DIR / "mgcv_laml_reference.csv").iloc[0]
    dtype = torch.float64
    response = torch.tensor(
        design_frame["response"].to_numpy(),
        dtype=dtype,
    )
    weights = torch.tensor(
        design_frame["weight"].to_numpy(),
        dtype=dtype,
    )
    mu_design = torch.tensor(
        design_frame.filter(regex=r"^mu_").to_numpy(),
        dtype=dtype,
    )
    sigma_design = torch.tensor(
        design_frame.filter(regex=r"^sigma_").to_numpy(),
        dtype=dtype,
    )
    coefficient_count = mu_design.shape[1] + sigma_design.shape[1]
    penalty_frame = pd.read_csv(REFERENCE_DIR / "mgcv_laml_penalties.csv")
    penalties = []
    for penalty_index in sorted(penalty_frame["penalty"].unique()):
        rows = penalty_frame[penalty_frame["penalty"] == penalty_index]
        penalty = torch.zeros(
            (coefficient_count, coefficient_count),
            dtype=dtype,
        )
        penalty[
            torch.tensor(rows["row"].to_numpy() - 1),
            torch.tensor(rows["column"].to_numpy() - 1),
        ] = torch.tensor(rows["value"].to_numpy(), dtype=dtype)
        penalties.append(penalty)
    return (
        response,
        weights,
        mu_design,
        sigma_design,
        tuple(penalties),
        reference,
    )


def _mgcv_poisson_system():
    design_frame = pd.read_csv(REFERENCE_DIR / "mgcv_poisson_laml_design.csv")
    reference = pd.read_csv(REFERENCE_DIR / "mgcv_poisson_laml_reference.csv").iloc[0]
    dtype = torch.float64
    response = torch.tensor(
        design_frame["response"].to_numpy(),
        dtype=dtype,
    )
    weights = torch.tensor(
        design_frame["weight"].to_numpy(),
        dtype=dtype,
    )
    design = torch.tensor(
        design_frame.filter(regex=r"^mu_").to_numpy(),
        dtype=dtype,
    )
    penalty_frame = pd.read_csv(REFERENCE_DIR / "mgcv_poisson_laml_penalties.csv")
    penalties = []
    for penalty_index in sorted(penalty_frame["penalty"].unique()):
        rows = penalty_frame[penalty_frame["penalty"] == penalty_index]
        penalty = torch.zeros(
            (design.shape[1], design.shape[1]),
            dtype=dtype,
        )
        penalty[
            torch.tensor(rows["row"].to_numpy() - 1),
            torch.tensor(rows["column"].to_numpy() - 1),
        ] = torch.tensor(rows["value"].to_numpy(), dtype=dtype)
        penalties.append(penalty)
    return response, weights, design, tuple(penalties), reference


def test_poisson_laml_matches_mgcv_reml_reference():
    response, weights, design, penalties, reference = _mgcv_poisson_system()
    result = fit_gamlss_laml(
        Poisson(),
        response,
        {"mu": design},
        penalties,
        (10.0,),
        weights=weights,
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=2e-5,
        ),
    )
    coefficients = pd.read_csv(
        REFERENCE_DIR / "mgcv_poisson_laml_coefficient_reference.csv"
    )
    fitted = pd.read_csv(REFERENCE_DIR / "mgcv_poisson_laml_fitted_reference.csv")

    assert result.outer_converged
    assert result.inner_converged
    assert result.family == "PO"
    assert result.parameter_names == ("mu",)
    assert result.coefficient_slices == {"mu": slice(0, design.shape[1])}
    assert result.boundary_status == ("interior",)
    assert result.penalty_ranks == (6,)
    assert result.combined_penalty_rank == 6
    assert result.unpenalized_dimension == 2
    assert float(result.objective) == pytest.approx(
        float(reference["objective"]),
        rel=2e-8,
        abs=2e-8,
    )
    assert float(result.log_likelihood) == pytest.approx(
        float(reference["log_likelihood"]),
        rel=2e-8,
        abs=5e-6,
    )
    assert float(result.smoothing_parameters[0]) == pytest.approx(
        float(reference["lambda_mu"]),
        rel=2e-5,
    )
    assert float(result.effective_degrees_of_freedom) == pytest.approx(
        float(reference["effective_degrees_of_freedom"]),
        rel=2e-6,
    )
    assert float(result.outer_hessian[0, 0]) == pytest.approx(
        float(reference["hessian_mu_mu"]),
        rel=3e-4,
    )
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor(
            coefficients["coefficient"].to_numpy(),
            dtype=torch.float64,
        ),
        rtol=1e-5,
        atol=5e-6,
    )
    torch.testing.assert_close(
        result.linear_predictors["mu"],
        torch.tensor(fitted["eta_mu"].to_numpy(), dtype=torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.fitted_parameters["mu"],
        torch.tensor(fitted["mu"].to_numpy(), dtype=torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )


def test_normal_location_scale_laml_matches_mgcv_reml_reference():
    (
        response,
        weights,
        mu_design,
        sigma_design,
        penalties,
        reference,
    ) = _mgcv_system()
    result = fit_normal_laml(
        response,
        mu_design,
        sigma_design,
        penalties,
        (10.0, 10.0),
        weights=weights,
        sigma_floor=float(reference["sigma_floor"]),
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=2e-5,
        ),
    )
    coefficients = pd.read_csv(REFERENCE_DIR / "mgcv_laml_coefficient_reference.csv")
    fitted = pd.read_csv(REFERENCE_DIR / "mgcv_laml_fitted_reference.csv")

    assert result.outer_converged
    assert result.inner_converged
    assert result.outer_iterations <= 10
    assert result.boundary_status == ("interior", "interior")
    assert result.penalty_ranks == (6, 5)
    assert result.combined_penalty_rank == 11
    assert result.unpenalized_dimension == 4
    assert result.constraint_rank == 0
    assert float(result.objective) == pytest.approx(
        float(reference["objective"]),
        rel=2e-8,
        abs=2e-8,
    )
    assert float(result.log_likelihood) == pytest.approx(
        float(reference["log_likelihood"]),
        rel=2e-7,
        abs=2e-7,
    )
    torch.testing.assert_close(
        result.smoothing_parameters,
        torch.tensor(
            [reference["lambda_mu"], reference["lambda_sigma"]],
            dtype=torch.float64,
        ),
        rtol=3e-6,
        atol=3e-6,
    )
    assert float(result.effective_degrees_of_freedom) == pytest.approx(
        float(reference["effective_degrees_of_freedom"]),
        rel=3e-7,
        abs=3e-7,
    )
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor(
            coefficients["coefficient"].to_numpy(),
            dtype=torch.float64,
        ),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.fitted_mu,
        torch.tensor(fitted["mu"].to_numpy(), dtype=torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.fitted_sigma,
        torch.tensor(fitted["sigma"].to_numpy(), dtype=torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )
    expected_hessian = torch.tensor(
        [
            [
                reference["hessian_mu_mu"],
                reference["hessian_mu_sigma"],
            ],
            [
                reference["hessian_mu_sigma"],
                reference["hessian_sigma_sigma"],
            ],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        result.outer_hessian,
        expected_hessian,
        rtol=1e-3,
        atol=2e-6,
    )
    assert torch.isfinite(result.outer_hessian_condition_number)
    assert torch.isfinite(result.generalized_log_determinant_penalty)
    assert torch.isfinite(result.log_determinant_penalized_information)
    assert (
        result.reduced_observed_information[
            : mu_design.shape[1],
            mu_design.shape[1] :,
        ]
        .abs()
        .max()
        > 0
    )
    torch.testing.assert_close(
        result.effective_degrees_of_freedom + result.penalty_degrees_of_freedom.sum(),
        response.new_tensor(result.coefficient_transform.shape[1]),
        rtol=1e-10,
        atol=1e-10,
    )
    history_objectives = [entry.objective for entry in result.history]
    assert all(
        later <= earlier
        for earlier, later in zip(
            history_objectives,
            history_objectives[1:],
        )
    )
    assert all(entry.inner_iterations > 0 for entry in result.history)
    assert result.history[-1].projected_gradient_max <= 2e-5


def test_tensor_laml_matches_mgcv_reml_reference():
    design_frame = pd.read_csv(REFERENCE_DIR / "mgcv_tensor_laml_design.csv")
    reference = pd.read_csv(REFERENCE_DIR / "mgcv_tensor_laml_reference.csv").iloc[0]
    dtype = torch.float64
    response = torch.tensor(
        design_frame["response"].to_numpy(),
        dtype=dtype,
    )
    weights = torch.tensor(
        design_frame["weight"].to_numpy(),
        dtype=dtype,
    )
    mu_design = torch.tensor(
        design_frame.filter(regex=r"^mu_").to_numpy(),
        dtype=dtype,
    )
    sigma_design = torch.tensor(
        design_frame.filter(regex=r"^sigma_").to_numpy(),
        dtype=dtype,
    )
    coefficient_count = mu_design.shape[1] + sigma_design.shape[1]
    penalty_frame = pd.read_csv(REFERENCE_DIR / "mgcv_tensor_laml_penalties.csv")
    penalties = []
    for penalty_index in sorted(penalty_frame["penalty"].unique()):
        rows = penalty_frame[penalty_frame["penalty"] == penalty_index]
        penalty = torch.zeros(
            (coefficient_count, coefficient_count),
            dtype=dtype,
        )
        penalty[
            torch.tensor(rows["row"].to_numpy() - 1),
            torch.tensor(rows["column"].to_numpy() - 1),
        ] = torch.tensor(rows["value"].to_numpy(), dtype=dtype)
        penalties.append(penalty)

    result = fit_normal_laml(
        response,
        mu_design,
        sigma_design,
        penalties,
        (10.0, 10.0),
        weights=weights,
        sigma_floor=float(reference["sigma_floor"]),
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=2e-5,
        ),
    )
    coefficients = pd.read_csv(
        REFERENCE_DIR / "mgcv_tensor_laml_coefficient_reference.csv"
    )
    fitted = pd.read_csv(REFERENCE_DIR / "mgcv_tensor_laml_fitted_reference.csv")

    assert result.outer_converged
    assert result.boundary_status == ("interior", "interior")
    assert result.outer_iterations <= 20
    assert float(result.objective) == pytest.approx(
        float(reference["objective"]),
        rel=2e-8,
        abs=2e-8,
    )
    assert float(result.log_likelihood) == pytest.approx(
        float(reference["log_likelihood"]),
        rel=5e-7,
        abs=5e-6,
    )
    torch.testing.assert_close(
        result.smoothing_parameters,
        torch.tensor(
            [reference["lambda_x"], reference["lambda_z"]],
            dtype=dtype,
        ),
        rtol=4e-6,
        atol=4e-6,
    )
    assert float(result.effective_degrees_of_freedom) == pytest.approx(
        float(reference["effective_degrees_of_freedom"]),
        rel=4e-7,
        abs=4e-7,
    )
    torch.testing.assert_close(
        result.coefficients,
        torch.tensor(
            coefficients["coefficient"].to_numpy(),
            dtype=dtype,
        ),
        rtol=3e-6,
        atol=3e-6,
    )
    torch.testing.assert_close(
        result.fitted_mu,
        torch.tensor(fitted["mu"].to_numpy(), dtype=dtype),
        rtol=3e-6,
        atol=3e-6,
    )
    torch.testing.assert_close(
        result.fitted_sigma,
        torch.tensor(fitted["sigma"].to_numpy(), dtype=dtype),
        rtol=3e-6,
        atol=3e-6,
    )
    expected_hessian = torch.tensor(
        [
            [
                reference["hessian_x_x"],
                reference["hessian_x_z"],
            ],
            [
                reference["hessian_x_z"],
                reference["hessian_z_z"],
            ],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(
        result.outer_hessian,
        expected_hessian,
        rtol=7e-3,
        atol=2e-5,
    )
    assert result.penalty_ranks == (15, 15)
    assert result.combined_penalty_rank == 21
    assert result.unpenalized_dimension == 5


def test_fixed_lambda_matches_current_gamlss_rs_pspline_fit():
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, 70, dtype=dtype)
    response = torch.sin(2.6 * x) + 0.28 * torch.sin(17.0 * x)
    intercept = torch.ones((x.numel(), 1), dtype=dtype)
    smoothing_parameter = 7.0
    term = PSpline(
        -1.0,
        1.0,
        smoothing_parameter,
        intervals=6,
        penalty_order=0,
        dtype=dtype,
    )
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=dtype,
    )
    fit = model.fit_rs(
        response,
        {"mu": intercept, "sigma": intercept},
        smooth_covariates={"mu": {"x": x}},
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )
    assert fit.converged

    smooth_design = term.design(x)
    mu_design = torch.cat((intercept, smooth_design), dim=1)
    sigma_design = intercept
    coefficient_count = mu_design.shape[1] + sigma_design.shape[1]
    penalty = torch.zeros(
        (coefficient_count, coefficient_count),
        dtype=dtype,
    )
    penalty[1 : mu_design.shape[1], 1 : mu_design.shape[1]] = term.penalty_matrices()[0]
    result = fit_normal_laml(
        response,
        mu_design,
        sigma_design,
        (penalty,),
        (smoothing_parameter,),
        estimate_smoothing=False,
    )
    expected_coefficients = torch.cat(
        (
            model.coefficients["mu"].detach(),
            term.coefficients.detach(),
            model.coefficients["sigma"].detach(),
        )
    )
    expected = model.predict(
        {"mu": intercept, "sigma": intercept},
        smooth_covariates={"mu": {"x": x}},
    )

    torch.testing.assert_close(
        result.coefficients,
        expected_coefficients,
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        result.fitted_mu,
        expected["mu"],
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        result.fitted_sigma,
        expected["sigma"],
        rtol=1e-7,
        atol=1e-7,
    )
    assert result.outer_converged
    assert result.outer_iterations == 0
    assert result.boundary_status == ("fixed",)
    torch.testing.assert_close(
        result.outer_gradient,
        torch.zeros(1, dtype=dtype),
    )


def test_laml_constraints_use_a_null_space_reparameterization():
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, 50, dtype=dtype)
    mu_design = torch.column_stack((torch.ones_like(x), x, x.square()))
    sigma_design = torch.ones((x.numel(), 1), dtype=dtype)
    response = 0.4 + 1.2 * x - 0.5 * x.square() + 0.15 * torch.sin(9 * x)
    penalty = torch.diag(torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=dtype))
    constraints = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
        dtype=dtype,
    )

    result = fit_normal_laml(
        response,
        mu_design,
        sigma_design,
        (penalty,),
        (4.0,),
        estimate_smoothing=False,
        constraints=constraints,
    )

    assert result.constraint_rank == 1
    assert result.coefficient_transform.shape == (4, 3)
    torch.testing.assert_close(
        constraints @ result.coefficients,
        torch.zeros(2, dtype=dtype),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        constraints @ result.coefficient_transform,
        torch.zeros((2, 3), dtype=dtype),
        rtol=0.0,
        atol=1e-12,
    )
    assert result.combined_penalty_rank == 1
    assert result.unpenalized_dimension == 2


@pytest.mark.parametrize(
    ("smoothing_parameters", "estimate_smoothing", "message"),
    [
        ((0.0,), True, "positive"),
        ((1.0, 2.0), True, "equal lengths"),
        ((1.0,), (True, False), "one boolean"),
    ],
)
def test_laml_rejects_invalid_smoothing_configuration(
    smoothing_parameters,
    estimate_smoothing,
    message,
):
    response = torch.tensor([0.0, 0.2, 0.5], dtype=torch.float64)
    mu_design = torch.ones((3, 1), dtype=torch.float64)
    sigma_design = torch.ones((3, 1), dtype=torch.float64)
    penalty = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))

    with pytest.raises(ValueError, match=message):
        fit_normal_laml(
            response,
            mu_design,
            sigma_design,
            (penalty,),
            smoothing_parameters,
            estimate_smoothing=estimate_smoothing,
        )


def test_laml_rejects_an_estimated_zero_penalty_component():
    response = torch.tensor([0.0, 0.2, 0.5], dtype=torch.float64)
    mu_design = torch.ones((3, 1), dtype=torch.float64)
    sigma_design = torch.ones((3, 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="nonzero penalty"):
        fit_normal_laml(
            response,
            mu_design,
            sigma_design,
            (torch.zeros((2, 2), dtype=torch.float64),),
            (1.0,),
        )


def test_laml_control_rejects_an_invalid_relaxed_gradient_multiplier():
    with pytest.raises(ValueError, match="at least one"):
        LAMLControl(inner_relaxed_gradient_multiplier=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_normal_laml_runs_on_cuda_with_an_estimated_smoothing_parameter():
    device = torch.device("cuda")
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, 60, dtype=dtype, device=device)
    response = torch.sin(2.0 * x) + torch.exp(-1.2 + 0.2 * x) * torch.sin(13.0 * x)
    mu_design = torch.column_stack((torch.ones_like(x), x, x.square(), x.pow(3)))
    sigma_design = torch.column_stack((torch.ones_like(x), x))
    penalty = torch.diag(
        torch.tensor(
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            dtype=dtype,
            device=device,
        )
    )

    result = fit_normal_laml(
        response,
        mu_design,
        sigma_design,
        (penalty,),
        (2.0,),
        control=LAMLControl(outer_max_iterations=20),
    )

    assert result.outer_converged
    assert result.coefficients.device.type == "cuda"
    assert result.outer_hessian.device.type == "cuda"
    assert torch.isfinite(result.objective)
    assert torch.isfinite(result.outer_gradient).all()
    assert torch.isfinite(result.outer_hessian).all()


def _formula_tensor_laml_data(
    observation_count: int = 60,
) -> pd.DataFrame:
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, observation_count, dtype=dtype)
    z = torch.sin(torch.linspace(0.15, 5.6, observation_count, dtype=dtype))
    generator = torch.Generator().manual_seed(2026)
    response = (
        0.7
        + torch.sin(2.0 * x)
        + 0.5 * x * z
        + 0.2 * z.square()
        + 0.18
        * torch.randn(
            observation_count,
            dtype=dtype,
            generator=generator,
        )
    )
    return pd.DataFrame(
        {
            "y": response.numpy(),
            "x": x.numpy(),
            "z": z.numpy(),
        }
    )


def _formula_poisson_laml_data(
    observation_count: int = 80,
) -> pd.DataFrame:
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, observation_count, dtype=dtype)
    mean = torch.exp(0.35 + torch.sin(math.pi * x) + 0.2 * x)
    response = torch.poisson(
        mean,
        generator=torch.Generator().manual_seed(20260801),
    )
    return pd.DataFrame({"y": response.numpy(), "x": x.numpy()})


def test_formula_poisson_laml_selects_lambda_and_bootstraps():
    data = _formula_poisson_laml_data()
    model = GAMLSS.from_formula(
        Poisson(),
        {"mu": "y ~ pb(x, intervals=5)"},
        data,
    )
    control = LAMLControl(
        inner_gradient_tolerance=1e-6,
        outer_max_iterations=25,
        outer_gradient_tolerance=5e-4,
    )

    result = model.fit_laml_data(data, control=control)
    prediction = model.predict_data(data)

    assert isinstance(result, GAMLSSLAMLResult)
    assert result.outer_converged
    assert result.parameter_names == ("mu",)
    assert result.smoothing_parameter_labels == (("mu", "x", 0),)
    assert result.smoothing_parameter_slices == {("mu", "x"): slice(0, 1)}
    assert result.smoothing_parameters[0] > 0
    torch.testing.assert_close(
        prediction["mu"],
        result.fitted_parameters["mu"],
    )

    bootstrap = model.smooth_bootstrap_data(
        data,
        new_data=data.iloc[::20].drop(columns="y"),
        replicates=10,
        max_attempts=20,
        algorithm="laml",
        control=control,
        generator=torch.Generator().manual_seed(718),
    )["mu"]["x"]

    assert bootstrap.algorithm == "laml"
    assert bootstrap.bootstrap_estimates.shape == (10, 4)
    assert bootstrap.bootstrap_smoothing_parameters.shape == (10,)
    assert bootstrap.bootstrap_smoothing_parameters.std() > 0
    assert torch.isfinite(bootstrap.bootstrap_estimates).all()


def test_formula_pb_fixed_laml_matches_rs_and_removes_null_overlap():
    data = _formula_tensor_laml_data(observation_count=70)
    formulas = {
        "mu": "y ~ pb(x, lambda_=7, intervals=5)",
        "sigma": "~ 1",
    }
    laml_model = GAMLSS.from_formula(Normal(), formulas, data)
    laml = laml_model.fit_laml_data(data)
    rs_model = GAMLSS.from_formula(Normal(), formulas, data)
    rs = rs_model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    assert laml.outer_converged
    assert laml.outer_iterations == 0
    assert rs.converged
    assert laml.constraint_rank == 2
    assert laml.smoothing_parameter_labels == (("mu", "x", 0),)
    assert laml.smoothing_parameter_slices == {("mu", "x"): slice(0, 1)}
    assert laml.linear_coefficient_slices["mu"] == slice(0, 2)
    assert laml.smooth_coefficient_slices["mu", "x"].start == 2
    torch.testing.assert_close(
        laml.smoothing_parameters,
        torch.tensor([7.0], dtype=torch.float64),
    )
    laml_parameters = laml_model.predict_data(data)
    rs_parameters = rs_model.predict_data(data)
    torch.testing.assert_close(
        laml_parameters["mu"],
        rs_parameters["mu"],
        rtol=1e-9,
        atol=1e-9,
    )
    torch.testing.assert_close(
        laml_parameters["sigma"],
        rs_parameters["sigma"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_formula_pb_laml_selects_lambda_and_updates_model():
    data = _formula_tensor_laml_data(observation_count=70)
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pb(x, intervals=5)",
            "sigma": "~ 1",
        },
        data,
    )

    result = model.fit_laml_data(data)

    assert result.outer_converged
    assert result.outer_iterations > 0
    assert result.estimated_smoothing_parameters == (True,)
    assert result.smoothing_parameter_labels == (("mu", "x", 0),)
    assert result.smoothing_parameters[0] > 0
    assert model.smooth_terms["mu"]["x"].smoothing_parameter == (
        pytest.approx(float(result.smoothing_parameters[0]))
    )


def test_formula_tensor_laml_selects_each_margin_and_updates_model():
    data = _formula_tensor_laml_data()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": ("y ~ te(x, z, intervals=(2, 2), name='surface')"),
            "sigma": "~ 1",
        },
        data,
    )
    term = model.smooth_terms["mu"]["surface"]

    assert term.smoothing_parameters == (10.0, 10.0)
    assert term.estimated_smoothing_parameters == (True, True)
    with pytest.raises(ValueError, match="whole-model LAML"):
        model.fit_rs_data(data)

    result = model.fit_laml_data(
        data,
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=5e-5,
        ),
    )

    assert result.outer_converged
    assert result.inner_converged
    assert result.outer_iterations > 0
    assert result.boundary_status == ("interior", "interior")
    assert result.estimated_smoothing_parameters == (True, True)
    assert result.smoothing_parameter_labels == (
        ("mu", "surface", 0),
        ("mu", "surface", 1),
    )
    assert result.smoothing_parameter_slices == {("mu", "surface"): slice(0, 2)}
    assert result.smoothing_parameters.shape == (2,)
    assert (result.smoothing_parameters > 0).all()
    assert torch.isfinite(result.outer_gradient).all()
    assert torch.isfinite(result.outer_hessian).all()
    assert result.outer_hessian.shape == (2, 2)
    assert result.penalty_degrees_of_freedom.shape == (2,)
    torch.testing.assert_close(
        result.smoothing_parameters.new_tensor(term.smoothing_parameters),
        result.smoothing_parameters,
    )
    prediction = model.predict_data(data)
    torch.testing.assert_close(prediction["mu"], result.fitted_mu)
    torch.testing.assert_close(prediction["sigma"], result.fitted_sigma)


def test_formula_tensor_laml_bootstrap_reselects_each_margin():
    data = _formula_tensor_laml_data()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": ("y ~ te(x, z, intervals=(2, 2), name='surface')"),
            "sigma": "~ 1",
        },
        data,
    )
    control = LAMLControl(
        inner_gradient_tolerance=1e-6,
        outer_max_iterations=25,
        outer_gradient_tolerance=5e-4,
    )
    fit = model.fit_laml_data(data, control=control)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    original_smoothing_parameters = model.smooth_terms["mu"][
        "surface"
    ].smoothing_parameters
    new_data = data.iloc[[0, 8, 20, 59]].drop(columns="y")

    bootstrap = model.smooth_joint_bootstrap_data(
        data,
        new_data=new_data,
        replicates=10,
        max_attempts=20,
        algorithm="laml",
        control=control,
        generator=torch.Generator().manual_seed(2026),
    )
    curve = bootstrap["mu"]["surface"]

    assert fit.outer_converged
    assert bootstrap.algorithm == "laml"
    assert bootstrap.attempts <= 20
    assert bootstrap.term_order == (("mu", "surface"),)
    assert bootstrap.smoothing_parameter_labels == (
        ("mu", "surface", 0),
        ("mu", "surface", 1),
    )
    assert bootstrap.smoothing_parameter_slices == {("mu", "surface"): slice(0, 2)}
    assert bootstrap.bootstrap_smoothing_parameters.shape == (10, 2)
    assert torch.all(bootstrap.bootstrap_smoothing_parameters.std(dim=0) > 0)
    assert curve.bootstrap_estimates.shape == (10, len(new_data))
    assert curve.bootstrap_smoothing_parameters.shape == (10, 2)
    assert torch.isfinite(curve.bootstrap_estimates).all()
    assert torch.isfinite(curve.bootstrap_smoothing_parameters).all()
    assert (curve.bootstrap_smoothing_parameters > 0).all()
    assert (curve.standard_errors > 0).all()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name])
    assert model.smooth_terms["mu"]["surface"].smoothing_parameters == (
        original_smoothing_parameters
    )


def test_formula_tensor_laml_accepts_explicit_initial_lambdas():
    data = _formula_tensor_laml_data(observation_count=36)
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ te(x, z, initial_lambda_=(4, 7), intervals=(2, 2), name='surface')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    term = model.smooth_terms["mu"]["surface"]

    assert term.smoothing_parameters == (4.0, 7.0)
    assert term.estimated_smoothing_parameters == (True, True)


def test_formula_tensor_interaction_laml_selects_both_margins():
    observation_count = 100
    dtype = torch.float64
    x = torch.linspace(-1.0, 1.0, observation_count, dtype=dtype)
    index = torch.arange(1, observation_count + 1, dtype=dtype)
    z = torch.sin(index * math.sqrt(2.0))
    generator = torch.Generator().manual_seed(99)
    response = (
        0.5
        + 0.7 * x
        - 0.4 * z
        + torch.sin(3.0 * x) * torch.cos(2.5 * z)
        + 0.35
        * torch.randn(
            observation_count,
            dtype=dtype,
            generator=generator,
        )
    )
    data = pd.DataFrame({"y": response.numpy(), "x": x.numpy(), "z": z.numpy()})
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": ("y ~ x + z + ti(x, z, intervals=(2, 2), name='interaction')"),
            "sigma": "~ 1",
        },
        data,
    )

    result = model.fit_laml_data(
        data,
        control=LAMLControl(
            outer_max_iterations=40,
            outer_gradient_tolerance=5e-5,
        ),
    )

    assert result.outer_converged
    assert result.boundary_status == ("interior", "interior")
    assert result.smoothing_parameter_labels == (
        ("mu", "interaction", 0),
        ("mu", "interaction", 1),
    )
    assert (result.smoothing_parameters > 0).all()
    torch.testing.assert_close(
        result.smoothing_parameters.new_tensor(
            model.smooth_terms["mu"]["interaction"].smoothing_parameters
        ),
        result.smoothing_parameters,
    )


def test_formula_laml_rejects_a_model_without_smooth_terms():
    data = _formula_tensor_laml_data(observation_count=30)
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x + z", "sigma": "~ 1"},
        data,
    )

    with pytest.raises(ValueError, match="at least one smooth"):
        model.fit_laml_data(data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_formula_tensor_laml_runs_on_cuda():
    data = _formula_tensor_laml_data()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": ("y ~ te(x, z, intervals=(2, 2), name='surface')"),
            "sigma": "~ 1",
        },
        data,
        device="cuda",
    )

    result = model.fit_laml_data(
        data,
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=5e-5,
        ),
    )

    assert result.outer_converged
    assert result.coefficients.device.type == "cuda"
    assert result.smoothing_parameters.device.type == "cuda"
    assert result.outer_hessian.device.type == "cuda"
    assert model.smooth_terms["mu"]["surface"].coefficients.device.type == ("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_formula_laml_bootstrap_runs_on_cuda():
    data = _formula_tensor_laml_data(observation_count=40)
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pb(x, intervals=4)",
            "sigma": "~ 1",
        },
        data,
        device="cuda",
    )
    control = LAMLControl(
        inner_gradient_tolerance=1e-6,
        outer_max_iterations=25,
        outer_gradient_tolerance=5e-4,
    )
    fit = model.fit_laml_data(data, control=control)

    bootstrap = model.smooth_bootstrap_data(
        data,
        new_data=data.iloc[::10].drop(columns="y"),
        replicates=10,
        max_attempts=20,
        algorithm="laml",
        control=control,
        generator=torch.Generator(device="cuda").manual_seed(718),
    )["mu"]["x"]

    assert fit.outer_converged
    assert bootstrap.algorithm == "laml"
    assert bootstrap.bootstrap_estimates.device.type == "cuda"
    assert bootstrap.bootstrap_smoothing_parameters.device.type == "cuda"
    assert bootstrap.bootstrap_smoothing_parameters.std() > 0
    assert torch.isfinite(bootstrap.bootstrap_estimates).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_poisson_laml_runs_on_cuda():
    response, weights, design, penalties, _ = _mgcv_poisson_system()
    response = response.cuda()
    weights = weights.cuda()
    design = design.cuda()
    penalties = tuple(penalty.cuda() for penalty in penalties)

    result = fit_gamlss_laml(
        Poisson(),
        response,
        {"mu": design},
        penalties,
        (10.0,),
        weights=weights,
        control=LAMLControl(
            outer_max_iterations=30,
            outer_gradient_tolerance=2e-5,
        ),
    )

    assert result.outer_converged
    assert result.coefficients.device.type == "cuda"
    assert result.fitted_parameters["mu"].device.type == "cuda"
    assert result.outer_hessian.device.type == "cuda"
