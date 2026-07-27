import csv
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    BCCG,
    BCPE,
    BCT,
    GAMLSS,
    TF,
    Beta,
    CGControl,
    NegativeBinomial,
    Normal,
    Poisson,
    PSpline,
    RSControl,
    SmoothBootstrapResult,
    SmoothCrossingBootstrapResult,
    SmoothDerivedBandResult,
    SmoothDerivedBootstrapResult,
    SmoothExtremumBootstrapResult,
    SmoothInferenceResult,
    SmoothJointBandResult,
    SmoothJointBootstrapResult,
    SmoothSimultaneousBand,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _table_rows(family: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / "inference_table_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return [row for row in csv.DictReader(data_file) if row["family"] == family]


def _covariance(family: str, size: int) -> torch.Tensor:
    covariance = torch.zeros((size, size), dtype=torch.float64)
    with (REFERENCE_DIR / "inference_covariance_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        for row in csv.DictReader(data_file):
            if row["family"] == family:
                covariance[int(row["row_index"]), int(row["column_index"])] = float(
                    row["covariance"]
                )
    return covariance


def _conditional_table_rows(case: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / "conditional_inference_table_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        return [row for row in csv.DictReader(data_file) if row["case"] == case]


def _conditional_covariance(case: str, size: int) -> torch.Tensor:
    covariance = torch.zeros((size, size), dtype=torch.float64)
    with (REFERENCE_DIR / "conditional_inference_covariance_reference.csv").open(
        newline="",
        encoding="utf-8",
    ) as data_file:
        for row in csv.DictReader(data_file):
            if row["case"] == case:
                covariance[int(row["row_index"]), int(row["column_index"])] = float(
                    row["covariance"]
                )
    return covariance


def _smooth_table_rows(
    case: str,
    parameter: str,
    term: str,
) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_DIR / "smooth_inference_reference.csv")
    return reference.loc[
        (reference["case"] == case)
        & (reference["parameter"] == parameter)
        & (reference["term"] == term)
    ].sort_values("observation_index")


@pytest.mark.parametrize(
    ("family_code", "prefix", "family", "formulas", "tolerance"),
    [
        (
            "NO",
            "no_rs",
            Normal(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-10,
        ),
        (
            "PO",
            "po",
            Poisson(),
            {"mu": "y ~ x + offset(mu_offset)"},
            1e-10,
        ),
        (
            "NBI",
            "nbi",
            NegativeBinomial(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-9,
        ),
        (
            "BE",
            "be",
            Beta(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            1e-9,
        ),
        (
            "BCCG",
            "bccg",
            BCCG(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
            },
            1e-9,
        ),
        (
            "BCT",
            "bct",
            BCT(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
                "tau": "~ 1",
            },
            1e-7,
        ),
        (
            "BCPE",
            "bcpe",
            BCPE(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
                "tau": "~ 1",
            },
            1e-7,
        ),
        (
            "TF",
            "tf",
            TF(),
            {
                "mu": "y ~ x + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
                "nu": "~ w + offset(nu_offset)",
            },
            1e-8,
        ),
    ],
)
def test_full_hessian_inference_matches_r_gamlss(
    family_code, prefix, family, formulas, tolerance
):
    data = pd.read_csv(REFERENCE_DIR / f"{prefix}_fit_data.csv")
    model = GAMLSS.from_formula(family, formulas, data)
    model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=tolerance,
            max_outer_iterations=200,
            inner_tolerance=tolerance,
            max_inner_iterations=200,
        ),
    )

    result = model.inference_data(data, weights="weight")
    rows = _table_rows(family_code)
    # R's optimHess uses finite differences. Its BCPE Hessian is slightly less
    # accurate than our autograd Hessian even though the fitted coefficients
    # agree to machine precision.
    if family_code == "BCPE":
        covariance_tolerances = {"rtol": 8e-3, "atol": 1e-4}
        derived_tolerances = {"rtol": 8e-3, "atol": 2e-6}
        p_value_tolerances = {"rtol": 2e-3, "atol": 3e-7}
    else:
        covariance_tolerances = {"rtol": 5e-6, "atol": 5e-7}
        derived_tolerances = {"rtol": 5e-6, "atol": 5e-7}
        p_value_tolerances = {"rtol": 2e-4, "atol": 3e-7}
    expected = {
        column: torch.tensor(
            [float(row[column]) for row in rows],
            dtype=torch.float64,
        )
        for column in (
            "estimate",
            "standard_error",
            "statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        )
    }

    assert result.coefficient_names == tuple(row["coefficient"] for row in rows)
    assert result.degrees_of_freedom == pytest.approx(
        float(rows[0]["degrees_of_freedom"])
    )
    torch.testing.assert_close(
        result.estimates,
        expected["estimate"],
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.covariance_matrix,
        _covariance(family_code, len(rows)),
        **covariance_tolerances,
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        **derived_tolerances,
    )
    torch.testing.assert_close(
        result.statistics,
        expected["statistic"],
        rtol=max(1e-5, derived_tolerances["rtol"]),
        atol=max(2e-6, derived_tolerances["atol"]),
    )
    torch.testing.assert_close(
        result.p_values,
        expected["p_value"],
        **p_value_tolerances,
    )
    torch.testing.assert_close(
        result.confidence_intervals,
        torch.column_stack((expected["ci_lower"], expected["ci_upper"])),
        rtol=derived_tolerances["rtol"],
        atol=max(1e-6, derived_tolerances["atol"]),
    )
    torch.testing.assert_close(
        result.correlation_matrix,
        result.correlation_matrix.mT,
    )
    torch.testing.assert_close(
        torch.diagonal(result.correlation_matrix),
        torch.ones(len(rows), dtype=torch.float64),
    )
    assert not result.conditional_on_smooths

    split_estimates = result.by_parameter(result.estimates)
    assert tuple(split_estimates) == family.parameter_names
    assert sum(values.numel() for values in split_estimates.values()) == len(rows)
    table = result.to_dataframe()
    assert tuple(table.index) == result.coefficient_names
    assert tuple(table.columns) == (
        "estimate",
        "standard_error",
        "statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
    )


def test_low_level_inference_uses_stable_positional_names():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    design = {"mu": torch.column_stack((torch.ones_like(x), x))}
    model = GAMLSS(Poisson(), {"mu": 2}, dtype=torch.float64)
    model.fit_rs(response, design)

    result = model.inference(response, design)

    assert result.coefficient_names == ("mu[0]", "mu[1]")


def test_inference_degrees_of_freedom_can_be_overridden():
    response = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    weights = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    model.fit_rs(response, design, weights=weights)

    automatic = model.inference(response, design, weights=weights)
    overridden = model.inference(
        response,
        design,
        weights=weights,
        degrees_of_freedom=2.5,
    )

    assert automatic.degrees_of_freedom == pytest.approx(5.0)
    assert overridden.degrees_of_freedom == pytest.approx(2.5)
    torch.testing.assert_close(
        automatic.covariance_matrix,
        overridden.covariance_matrix,
    )
    assert not torch.equal(automatic.p_values, overridden.p_values)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, float("nan")])
def test_inference_rejects_invalid_confidence_levels(confidence_level):
    response = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="confidence_level"):
        model.inference(
            response,
            design,
            confidence_level=confidence_level,
        )


def test_inference_rejects_nonpositive_residual_degrees_of_freedom():
    response = torch.tensor([1.0], dtype=torch.float64)
    design = {"mu": torch.ones((1, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="degrees of freedom"):
        model.inference(response, design)


def test_inference_rejects_singular_hessian():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    design = {
        "mu": torch.column_stack(
            (torch.ones_like(x), torch.ones_like(x)),
        )
    }
    model = GAMLSS(Poisson(), {"mu": 2}, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="positive definite"):
        model.inference(response, design)


@pytest.mark.parametrize("case", ["NO_FIXED_RS", "NO_ML_RS", "BE_FIXED_CG"])
def test_conditional_smooth_inference_matches_r_gamlss_vcov(case):
    if case == "BE_FIXED_CG":
        data = pd.read_csv(REFERENCE_DIR / "be_fit_data.csv")
        model = GAMLSS.from_formula(
            Beta(),
            {
                "mu": "y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)",
                "sigma": "~ z + offset(sigma_offset)",
            },
            data,
        )
        model.fit_cg_data(
            data,
            weights="weight",
            control=CGControl(
                outer_tolerance=1e-7,
                max_outer_iterations=300,
                inner_tolerance=1e-7,
                max_inner_iterations=300,
                backfitting_tolerance=1e-7,
            ),
        )
        weights = "weight"
    else:
        data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
        mu_formula = (
            "y ~ pb(x, smoothing_parameter=12)"
            if case == "NO_FIXED_RS"
            else "y ~ pb(x)"
        )
        model = GAMLSS.from_formula(
            Normal(),
            {"mu": mu_formula, "sigma": "~ 1"},
            data,
        )
        model.fit_rs_data(
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
        weights = None

    rows = _conditional_table_rows(case)
    degrees_of_freedom = float(rows[0]["degrees_of_freedom"])
    result = model.inference_data(
        data,
        weights=weights,
        conditional_on_smooths=True,
        degrees_of_freedom=degrees_of_freedom,
    )
    expected = {
        column: torch.tensor(
            [float(row[column]) for row in rows],
            dtype=torch.float64,
        )
        for column in (
            "estimate",
            "standard_error",
            "statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        )
    }

    assert result.conditional_on_smooths
    expected_names = {
        "NO_FIXED_RS": (
            "mu.Intercept",
            "mu.pb(x, smoothing_parameter=12)",
            "sigma.Intercept",
        ),
        "NO_ML_RS": ("mu.Intercept", "mu.pb(x)", "sigma.Intercept"),
        "BE_FIXED_CG": (
            "mu.Intercept",
            "mu.pb(x, smoothing_parameter=12)",
            "sigma.Intercept",
            "sigma.z",
        ),
    }
    assert result.coefficient_names == expected_names[case]
    assert result.degrees_of_freedom == pytest.approx(degrees_of_freedom)
    torch.testing.assert_close(
        result.estimates,
        expected["estimate"],
        rtol=5e-6,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.covariance_matrix,
        _conditional_covariance(case, len(rows)),
        rtol=1e-5,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        rtol=1e-5,
        atol=5e-7,
    )
    torch.testing.assert_close(
        result.statistics,
        expected["statistic"],
        rtol=1e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.p_values,
        expected["p_value"],
        rtol=2e-4,
        atol=3e-7,
    )
    torch.testing.assert_close(
        result.confidence_intervals,
        torch.column_stack((expected["ci_lower"], expected["ci_upper"])),
        rtol=1e-5,
        atol=1e-6,
    )


def test_inference_requires_explicit_conditioning_for_smooth_models():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    term = PSpline.from_data(x, smoothing_parameter=12.0)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="conditional_on_smooths"):
        model.inference(torch.sin(x), design)


def test_conditional_smooth_inference_requires_residual_degrees_of_freedom():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    term = PSpline.from_data(x, smoothing_parameter=12.0)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }

    with pytest.raises(ValueError, match="degrees_of_freedom is required"):
        model.inference(
            torch.sin(x),
            design,
            smooth_covariates={"mu": {"x": x}},
            conditional_on_smooths=True,
        )


@pytest.mark.parametrize(
    ("case", "formula"),
    [
        ("NO_FIXED_RS", "y ~ pb(x, smoothing_parameter=12)"),
        ("NO_ML_RS", "y ~ pb(x)"),
    ],
)
def test_smooth_curve_inference_matches_r_gamlss_pb(case, formula):
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": formula, "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(
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

    result = model.smooth_inference_data(data)["mu"]["x"]
    reference = _smooth_table_rows(case, "mu", "x")
    expected = {
        column: torch.tensor(reference[column].to_numpy(), dtype=torch.float64)
        for column in (
            "estimate",
            "variance",
            "standard_error",
            "ci_lower",
            "ci_upper",
        )
    }

    assert isinstance(result, SmoothInferenceResult)
    assert result.parameter == "mu"
    assert result.term == "x"
    assert result.confidence_level == pytest.approx(0.95)
    assert result.smoothing_parameter == pytest.approx(
        float(reference["smoothing_parameter"].iloc[0]),
        rel=1e-8,
        abs=1e-8,
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["smooth_edf"].iloc[0]),
        rel=1e-8,
        abs=1e-8,
    )
    torch.testing.assert_close(
        result.estimates,
        expected["estimate"],
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(
        result.variances,
        expected["variance"],
        rtol=2e-7,
        atol=2e-10,
    )
    assert result.covariance_matrix.shape == (len(data), len(data))
    torch.testing.assert_close(
        result.covariance_matrix,
        result.covariance_matrix.mT,
    )
    torch.testing.assert_close(
        torch.diagonal(result.covariance_matrix),
        result.variances,
    )
    assert torch.linalg.eigvalsh(result.covariance_matrix).min() >= -1e-12
    torch.testing.assert_close(
        result.correlation_matrix,
        result.correlation_matrix.mT,
    )
    torch.testing.assert_close(
        torch.diagonal(result.correlation_matrix),
        torch.ones(len(data), dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.standard_errors,
        expected["standard_error"],
        rtol=2e-7,
        atol=2e-9,
    )
    torch.testing.assert_close(
        result.confidence_intervals,
        torch.column_stack((expected["ci_lower"], expected["ci_upper"])),
        rtol=2e-7,
        atol=2e-7,
    )
    table = result.to_dataframe()
    assert tuple(table.columns) == (
        "covariate",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )
    assert len(table) == len(data)


def test_smooth_simultaneous_confidence_band_is_reproducible_and_wider():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x, smoothing_parameter=12)", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)
    result = model.smooth_inference_data(data)["mu"]["x"]

    first = result.simultaneous_confidence_band(
        simulations=2_000,
        generator=torch.Generator().manual_seed(2026),
    )
    second = result.simultaneous_confidence_band(
        simulations=2_000,
        generator=torch.Generator().manual_seed(2026),
    )

    assert isinstance(first, SmoothSimultaneousBand)
    assert first.method == "conditional_gaussian_max_t"
    assert first.parameter == "mu"
    assert first.term == "x"
    assert first.simulations == 2_000
    assert first.confidence_level == pytest.approx(0.95)
    assert first.critical_value == pytest.approx(second.critical_value)
    torch.testing.assert_close(
        first.confidence_intervals,
        second.confidence_intervals,
    )
    assert first.critical_value > 1.96
    assert torch.all(
        first.confidence_intervals[:, 0] <= result.confidence_intervals[:, 0]
    )
    assert torch.all(
        first.confidence_intervals[:, 1] >= result.confidence_intervals[:, 1]
    )
    table = first.to_dataframe()
    assert tuple(table.columns) == (
        "covariate",
        "estimate",
        "ci_lower",
        "ci_upper",
    )
    assert len(table) == len(data)

    with pytest.raises(ValueError, match="at least 100"):
        result.simultaneous_confidence_band(simulations=99)


def test_parametric_smooth_bootstrap_reselects_lambda_reproducibly():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x)", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)
    original_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    new_data = data.iloc[::2].drop(columns="y")

    first = model.smooth_bootstrap_data(
        data,
        new_data=new_data,
        replicates=10,
        generator=torch.Generator().manual_seed(2026),
    )["mu"]["x"]
    second = model.smooth_bootstrap_data(
        data,
        new_data=new_data,
        replicates=10,
        generator=torch.Generator().manual_seed(2026),
    )["mu"]["x"]

    assert isinstance(first, SmoothBootstrapResult)
    assert first.algorithm == "rs"
    assert first.replicates == 10
    assert first.attempts == 10
    assert first.failed_replicates == 0
    assert first.failure_rate == 0
    assert first.confidence_level == pytest.approx(0.95)
    assert first.bootstrap_estimates.shape == (10, len(new_data))
    assert first.bootstrap_smoothing_parameters.shape == (10,)
    assert first.bootstrap_smoothing_parameters.std() > 0
    assert first.smoothing_parameter_standard_error > 0
    assert first.smoothing_parameter_bootstrap_mean > 0
    assert math.isfinite(first.smoothing_parameter_bias)
    assert first.smoothing_parameter_confidence_interval.shape == (2,)
    torch.testing.assert_close(
        first.bootstrap_estimates,
        second.bootstrap_estimates,
    )
    torch.testing.assert_close(
        first.bootstrap_smoothing_parameters,
        second.bootstrap_smoothing_parameters,
    )
    torch.testing.assert_close(
        torch.diagonal(first.covariance_matrix),
        first.standard_errors.square(),
    )
    assert first.confidence_intervals.shape == (len(new_data), 2)
    assert torch.isfinite(first.confidence_intervals).all()
    simultaneous_band = first.simultaneous_confidence_band()
    assert simultaneous_band.method == "parametric_bootstrap_max_t"
    assert simultaneous_band.simulations == 10
    assert math.isfinite(simultaneous_band.critical_value)
    assert simultaneous_band.critical_value > 0
    assert simultaneous_band.confidence_intervals.shape == (len(new_data), 2)
    assert tuple(first.to_dataframe().columns) == (
        "covariate",
        "estimate",
        "bootstrap_mean",
        "bias",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name])

    cg = model.smooth_bootstrap_data(
        data,
        new_data=new_data,
        replicates=10,
        algorithm="cg",
        generator=torch.Generator().manual_seed(7),
    )["mu"]["x"]
    assert cg.algorithm == "cg"
    assert cg.bootstrap_estimates.shape == (10, len(new_data))


def test_joint_smooth_bootstrap_preserves_cross_term_dependence():
    data = pd.read_csv(REFERENCE_DIR / "cg_multi_smooth_fit_data.csv")
    model = GAMLSS.from_formula(
        Beta(),
        {
            "mu": "y ~ pb(x) + offset(mu_offset)",
            "sigma": "~ pb(z) + offset(sigma_offset)",
        },
        data,
    )
    control = RSControl(max_outer_iterations=200)
    model.fit_rs_data(data, weights="weight", control=control)
    new_data = data.iloc[::4].drop(columns="y").copy()
    new_data["z"] = new_data["x"]

    joint = model.smooth_joint_bootstrap_data(
        data,
        weights="weight",
        new_data=new_data,
        replicates=10,
        max_attempts=20,
        control=control,
        generator=torch.Generator().manual_seed(17),
    )

    assert isinstance(joint, SmoothJointBootstrapResult)
    assert joint.term_order == (("mu", "x"), ("sigma", "z"))
    assert joint.algorithm == "rs"
    assert joint.replicates == 10
    assert 10 <= joint.attempts <= 20
    assert joint.failed_replicates == joint.attempts - joint.replicates
    assert joint.failure_rate == pytest.approx(
        joint.failed_replicates / joint.attempts
    )
    assert joint["mu"]["x"].bootstrap_estimates.shape == (10, len(new_data))
    assert joint["sigma"]["z"].bootstrap_estimates.shape == (10, len(new_data))
    assert joint.bootstrap_estimates.shape == (10, 2 * len(new_data))
    assert joint.estimates.shape == (2 * len(new_data),)
    assert len(joint.point_labels) == 2 * len(new_data)

    mu_slice = joint.term_slices[("mu", "x")]
    sigma_slice = joint.term_slices[("sigma", "z")]
    cross_covariance = joint.covariance_block(("mu", "x"), ("sigma", "z"))
    torch.testing.assert_close(
        joint.covariance_matrix[mu_slice, sigma_slice],
        cross_covariance,
    )
    torch.testing.assert_close(
        joint.covariance_matrix[sigma_slice, mu_slice],
        cross_covariance.mT,
    )
    torch.testing.assert_close(
        joint.covariance_matrix[mu_slice, mu_slice],
        joint["mu"]["x"].covariance_matrix,
    )
    torch.testing.assert_close(
        torch.diagonal(joint.correlation_matrix),
        torch.ones(2 * len(new_data), dtype=torch.float64),
    )
    assert joint.bootstrap_smoothing_parameters.shape == (10, 2)
    assert joint.smoothing_parameters.shape == (2,)
    assert torch.all(joint.bootstrap_smoothing_parameters.std(dim=0) > 0)
    torch.testing.assert_close(
        torch.diagonal(joint.smoothing_parameter_covariance_matrix),
        joint.bootstrap_smoothing_parameters.var(dim=0),
    )
    torch.testing.assert_close(
        torch.diagonal(joint.smoothing_parameter_correlation_matrix),
        torch.ones(2, dtype=torch.float64),
    )

    bands = joint.simultaneous_confidence_bands()
    assert isinstance(bands, SmoothJointBandResult)
    assert bands.term_order == joint.term_order
    assert bands.method == "parametric_bootstrap_joint_max_t"
    assert bands.replicates == 10
    assert bands["mu"]["x"].method == "parametric_bootstrap_joint_max_t"
    assert bands["sigma"]["z"].critical_value == pytest.approx(
        bands.critical_value
    )
    mu_band = joint["mu"]["x"].simultaneous_confidence_band()
    assert bands.critical_value >= mu_band.critical_value
    assert tuple(bands.to_dataframe().columns) == (
        "parameter",
        "term",
        "covariate",
        "estimate",
        "ci_lower",
        "ci_upper",
    )
    assert len(bands.to_dataframe()) == 2 * len(new_data)
    assert tuple(joint.to_dataframe().columns) == (
        "parameter",
        "term",
        "covariate",
        "estimate",
        "bootstrap_mean",
        "bias",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )

    mu_only = joint.simultaneous_confidence_bands((("mu", "x"),))
    assert mu_only.term_order == (("mu", "x"),)
    assert mu_only.critical_value == pytest.approx(
        joint["mu"]["x"].simultaneous_confidence_band().critical_value
    )
    with pytest.raises(ValueError, match="at least one"):
        joint.simultaneous_confidence_bands(())
    with pytest.raises(ValueError, match="duplicates"):
        joint.simultaneous_confidence_bands((("mu", "x"), ("mu", "x")))
    with pytest.raises(KeyError, match="unknown smooth term"):
        joint.covariance_block(("mu", "missing"), ("sigma", "z"))

    difference = joint.difference(("mu", "x"), ("sigma", "z"))
    assert isinstance(difference, SmoothDerivedBootstrapResult)
    assert difference.name == "mu.x - sigma.z"
    assert difference.operation == "linear_contrast"
    assert difference.source_terms == joint.term_order
    torch.testing.assert_close(
        difference.estimates,
        joint["mu"]["x"].estimates - joint["sigma"]["z"].estimates,
    )
    torch.testing.assert_close(
        difference.bootstrap_estimates,
        (
            joint["mu"]["x"].bootstrap_estimates
            - joint["sigma"]["z"].bootstrap_estimates
        ),
    )
    torch.testing.assert_close(
        torch.diagonal(difference.covariance_matrix),
        difference.standard_errors.square(),
    )
    assert difference.confidence_intervals.shape == (len(new_data), 2)
    assert tuple(difference.to_dataframe().columns) == (
        "covariate",
        "estimate",
        "bootstrap_mean",
        "bias",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )

    derived_band = difference.simultaneous_confidence_band()
    assert isinstance(derived_band, SmoothDerivedBandResult)
    assert derived_band.method == "parametric_bootstrap_derived_max_t"
    assert derived_band.confidence_intervals.shape == (len(new_data), 2)

    derivative = joint.derivative(("mu", "x"))
    assert derivative.operation == "derivative_1"
    torch.testing.assert_close(
        derivative.estimates,
        torch.gradient(
            joint["mu"]["x"].estimates,
            spacing=(joint["mu"]["x"].covariate,),
        )[0],
    )
    assert joint.derivative(("mu", "x"), order=2).estimates.shape == (
        len(new_data),
    )
    with pytest.raises(ValueError, match="1 or 2"):
        joint.derivative(("mu", "x"), order=3)

    maximum = difference.extremum()
    assert isinstance(maximum, SmoothExtremumBootstrapResult)
    assert maximum.kind == "maximum"
    assert maximum.estimate == pytest.approx(float(difference.estimates.max()))
    assert maximum.bootstrap_estimates.shape == (10,)
    assert maximum.bootstrap_locations.shape == (10,)
    assert maximum.confidence_interval.shape == (2,)
    assert maximum.location_confidence_interval.shape == (2,)
    assert maximum.attempts == joint.attempts
    assert maximum.failure_rate == pytest.approx(joint.failure_rate)
    assert tuple(maximum.to_dataframe()["metric"]) == ("value", "location")

    crossing = difference.crossing(level=0.0, direction="decreasing")
    assert isinstance(crossing, SmoothCrossingBootstrapResult)
    assert crossing.valid_replicates + crossing.missing_replicates == 10
    assert crossing.valid_replicates >= 2
    assert math.isfinite(crossing.estimate)
    assert crossing.confidence_interval.shape == (2,)
    assert 0.0 <= crossing.missing_rate <= 1.0
    assert crossing.attempts == joint.attempts
    assert crossing.failure_rate == pytest.approx(joint.failure_rate)
    assert len(crossing.to_dataframe()) == 1

    contrast = joint.linear_contrast(
        {("mu", "x"): 2.0, ("sigma", "z"): -0.5},
        name="custom",
    )
    assert contrast.name == "custom"
    torch.testing.assert_close(
        contrast.estimates,
        2.0 * joint["mu"]["x"].estimates
        - 0.5 * joint["sigma"]["z"].estimates,
    )
    with pytest.raises(ValueError, match="distinct"):
        joint.difference(("mu", "x"), ("mu", "x"))
    with pytest.raises(ValueError, match="nonzero"):
        joint.linear_contrast({("mu", "x"): 0.0})
    with pytest.raises(KeyError, match="unknown smooth terms"):
        joint.linear_contrast({("mu", "missing"): 1.0})


def test_derived_crossing_tracks_replicates_without_a_root():
    covariate = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    estimates = torch.tensor([-1.0, 0.2, 1.0], dtype=torch.float64)
    bootstrap_estimates = torch.tensor(
        [
            [-1.0, 0.1, 1.0],
            [-2.0, -1.0, 1.0],
            [1.0, 2.0, 3.0],
            [-2.0, -1.0, -0.5],
        ],
        dtype=torch.float64,
    )
    result = SmoothDerivedBootstrapResult(
        name="synthetic",
        operation="test",
        source_terms=(("mu", "x"),),
        covariate=covariate,
        estimates=estimates,
        bootstrap_estimates=bootstrap_estimates,
        standard_errors=bootstrap_estimates.std(dim=0),
        confidence_intervals=torch.zeros((3, 2), dtype=torch.float64),
        confidence_level=0.95,
        replicates=4,
        attempts=4,
        failed_replicates=0,
        algorithm="rs",
    )

    crossing = result.crossing(direction="increasing")

    assert crossing.estimate == pytest.approx(-1.0 / 6.0)
    assert crossing.valid_replicates == 2
    assert crossing.missing_replicates == 2
    assert crossing.missing_rate == pytest.approx(0.5)
    assert torch.isnan(crossing.bootstrap_estimates[2:]).all()
    assert torch.isfinite(crossing.valid_bootstrap_estimates).all()
    with pytest.raises(ValueError, match="does not contain"):
        result.crossing(level=10.0)
    with pytest.raises(ValueError, match="direction"):
        result.crossing(direction="sideways")


def test_smooth_bootstrap_validates_replicates_algorithm_and_control():
    x = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
    response = torch.sin(x)
    term = PSpline.from_data(x)
    model = GAMLSS(
        Normal(),
        {"mu": 2, "sigma": 1},
        smooth_terms={"mu": {"x": term}},
        dtype=torch.float64,
    )
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }
    smooth_covariates = {"mu": {"x": x}, "sigma": {}}

    with pytest.raises(ValueError, match="at least 10"):
        model.smooth_bootstrap(
            response,
            design,
            smooth_covariates=smooth_covariates,
            replicates=9,
        )
    with pytest.raises(ValueError, match="algorithm"):
        model.smooth_bootstrap(
            response,
            design,
            smooth_covariates=smooth_covariates,
            algorithm="other",
        )
    with pytest.raises(ValueError, match="RSControl"):
        model.smooth_bootstrap(
            response,
            design,
            smooth_covariates=smooth_covariates,
            control=CGControl(),
        )
    with pytest.raises(ValueError, match="max_attempts"):
        model.smooth_bootstrap(
            response,
            design,
            smooth_covariates=smooth_covariates,
            replicates=10,
            max_attempts=9,
        )
    with pytest.raises(RuntimeError, match="0 successful fits"):
        model.smooth_bootstrap(
            response,
            design,
            smooth_covariates=smooth_covariates,
            replicates=10,
            max_attempts=10,
            control=RSControl(max_outer_iterations=1),
            generator=torch.Generator().manual_seed(3),
        )


def test_smooth_curve_inference_supports_formula_new_data():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x, smoothing_parameter=12)", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)
    new_data = data.iloc[::4].drop(columns="y")

    result = model.smooth_inference_data(data, new_data=new_data)["mu"]["x"]
    contribution = model.predict_data(new_data, type="terms")["mu"].smooth["x"]

    torch.testing.assert_close(result.estimates, contribution)
    assert result.standard_errors.shape == (len(new_data),)
    assert result.covariance_matrix.shape == (len(new_data), len(new_data))
    assert result.confidence_intervals.shape == (len(new_data), 2)
    assert torch.all(result.confidence_intervals[:, 0] <= result.estimates)
    assert torch.all(result.confidence_intervals[:, 1] >= result.estimates)


def test_smooth_curve_inference_rejects_parametric_models():
    response = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    with pytest.raises(ValueError, match="at least one smooth"):
        model.smooth_inference(
            response,
            design,
            smooth_covariates={"mu": {}},
        )


def test_inference_result_rejects_misaligned_parameter_split():
    response = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    design = {"mu": torch.ones((4, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    model.fit_rs(response, design)
    result = model.inference(response, design)

    with pytest.raises(ValueError, match="one row"):
        result.by_parameter(torch.ones(2, dtype=torch.float64))
