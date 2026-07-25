from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import GAMLSS, Beta, CGControl, Normal, PSpline, RSControl

REFERENCE_DIR = Path(__file__).parent / "reference"


def _normal_smooth_model():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    x = torch.tensor(data["x"].to_numpy(), dtype=torch.float64)
    response = torch.tensor(data["y"].to_numpy(), dtype=torch.float64)
    term = PSpline.from_data(x, smoothing_parameter=12.0, intervals=10)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design_matrices = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }
    smooth_covariates = {"mu": {"x": x}}
    fit = model.fit_rs(
        response,
        design_matrices,
        smooth_covariates=smooth_covariates,
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
    return model, response, design_matrices, smooth_covariates


def _beta_two_smooth_model():
    data = pd.read_csv(REFERENCE_DIR / "be_fit_data.csv")
    model = GAMLSS.from_formula(
        Beta(),
        {
            "mu": ("y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)"),
            "sigma": ("~ pb(z, smoothing_parameter=15) + offset(sigma_offset)"),
        },
        data,
    )
    fit = model.fit_cg_data(
        data,
        weights="weight",
        control=CGControl(
            outer_tolerance=1e-8,
            max_outer_iterations=500,
            inner_tolerance=1e-8,
            max_inner_iterations=500,
            backfitting_tolerance=1e-8,
        ),
    )
    assert fit.converged
    return model, data


def test_joint_smooth_inference_reduces_to_r_pb_covariance():
    model, response, design_matrices, smooth_covariates = _normal_smooth_model()
    result = model.smooth_joint_inference(
        response,
        design_matrices,
        smooth_covariates=smooth_covariates,
    )
    marginal = model.smooth_inference(
        response,
        design_matrices,
        smooth_covariates=smooth_covariates,
    )["mu"]["x"]
    reference = pd.read_csv(REFERENCE_DIR / "smooth_inference_reference.csv")
    reference = reference.loc[
        (reference["case"] == "NO_FIXED_RS")
        & (reference["parameter"] == "mu")
        & (reference["term"] == "x")
    ].sort_values("observation_index")
    coefficient_covariance_reference = pd.read_csv(
        REFERENCE_DIR / "joint_smooth_coefficient_covariance_reference.csv"
    )
    coefficient_covariance_reference = coefficient_covariance_reference.pivot(
        index="row_index",
        columns="column_index",
        values="covariance",
    )
    curve_covariance_reference = pd.read_csv(
        REFERENCE_DIR / "joint_smooth_curve_covariance_reference.csv"
    )
    curve_covariance_reference = curve_covariance_reference.pivot(
        index="row_index",
        columns="column_index",
        values="covariance",
    )
    curve = result["mu"]["x"]

    assert result.term_order == (("mu", "x"),)
    assert result.linear_coefficient_slices == {
        "mu": slice(0, 2),
        "sigma": slice(15, 16),
    }
    assert result.smooth_coefficient_slices == {("mu", "x"): slice(2, 15)}
    assert result.coefficient_names[:3] == ("mu[0]", "mu[1]", "mu.x[0]")
    assert result.coefficient_names[-1] == "sigma[0]"
    torch.testing.assert_close(
        curve.standard_errors,
        torch.tensor(
            reference["standard_error"].to_numpy(),
            dtype=torch.float64,
        ),
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        curve.covariance_matrix,
        marginal.covariance_matrix,
        rtol=1e-11,
        atol=1e-12,
    )
    torch.testing.assert_close(
        curve.covariance_matrix,
        torch.tensor(
            curve_covariance_reference.to_numpy(),
            dtype=torch.float64,
        ),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.coefficient_covariance_matrix,
        torch.tensor(
            coefficient_covariance_reference.to_numpy(),
            dtype=torch.float64,
        ),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.covariance_matrix,
        result.covariance_block(("mu", "x"), ("mu", "x")),
    )
    torch.testing.assert_close(
        result.coefficient_covariance_matrix,
        result.coefficient_covariance_matrix.mT,
    )
    eigenvalues = torch.linalg.eigvalsh(result.coefficient_covariance_matrix)
    assert float(eigenvalues.min()) >= -1e-12
    assert result.to_dataframe().shape[0] == response.numel()


def test_joint_smooth_inference_retains_cross_parameter_covariance_and_new_data():
    model, data = _beta_two_smooth_model()
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    new_data = data.iloc[[0, 11, 29, 53, 79]].copy()
    result = model.smooth_joint_inference_data(
        data,
        weights="weight",
        new_data=new_data,
    )

    assert result.term_order == (("mu", "x"), ("sigma", "z"))
    assert result.point_labels[0] == ("mu", "x", 0)
    assert result.point_labels[-1] == ("sigma", "z", 4)
    assert result.covariance_matrix.shape == (10, 10)
    cross = result.covariance_block(("mu", "x"), ("sigma", "z"))
    reverse = result.covariance_block(("sigma", "z"), ("mu", "x"))
    assert cross.shape == (5, 5)
    assert float(cross.abs().max()) > 1e-6
    torch.testing.assert_close(cross, reverse.mT)
    torch.testing.assert_close(
        result.covariance_matrix,
        result.covariance_matrix.mT,
    )
    assert (result.standard_errors > 0).all()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name])


def test_joint_gaussian_bands_are_reproducible_and_validate_term_selection():
    model, response, design_matrices, smooth_covariates = _normal_smooth_model()
    result = model.smooth_joint_inference(
        response,
        design_matrices,
        smooth_covariates=smooth_covariates,
    )
    first = result.simultaneous_confidence_bands(
        simulations=500,
        generator=torch.Generator().manual_seed(2026),
    )
    second = result.simultaneous_confidence_bands(
        simulations=500,
        generator=torch.Generator().manual_seed(2026),
    )

    assert first.method == "analytic_joint_gaussian_max_t"
    assert first.term_order == (("mu", "x"),)
    assert first.simulations == 500
    assert first.critical_value == pytest.approx(second.critical_value)
    torch.testing.assert_close(
        first["mu"]["x"].confidence_intervals,
        second["mu"]["x"].confidence_intervals,
    )
    with pytest.raises(ValueError, match="at least 100"):
        result.simultaneous_confidence_bands(simulations=99)
    with pytest.raises(ValueError, match="duplicates"):
        result.simultaneous_confidence_bands(terms=[("mu", "x"), ("mu", "x")])
    with pytest.raises(KeyError, match="unknown smooth term"):
        result.covariance_block(("mu", "missing"), ("mu", "x"))


def test_joint_smooth_inference_requires_a_smooth_term():
    response = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)
    design_matrices = {
        parameter: torch.ones((3, 1), dtype=torch.float64)
        for parameter in model.family.parameter_names
    }

    with pytest.raises(ValueError, match="at least one smooth term"):
        model.smooth_joint_inference(
            response,
            design_matrices,
            smooth_covariates={},
        )
