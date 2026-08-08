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
    PE,
    TF,
    Beta,
    Gamma,
    LAMLControl,
    NegativeBinomial,
    Normal,
    Poisson,
    QuantileBandResult,
    QuantileBootstrapResult,
    QuantilePrediction,
)

REFERENCE_DIR = Path(__file__).parent / "reference"

FAMILY_CASES = {
    "NO": ("no_reference.csv", Normal()),
    "GA": ("ga_reference.csv", Gamma()),
    "PO": ("po_reference.csv", Poisson()),
    "NBI": ("nbi_reference.csv", NegativeBinomial()),
    "BE": ("be_reference.csv", Beta()),
    "BCCG": ("bccg_reference.csv", BCCG()),
    "BCT": ("bct_reference.csv", BCT()),
    "BCPE": ("bcpe_reference.csv", BCPE()),
    "TF": ("tf_reference.csv", TF()),
    "PE": ("pe_reference.csv", PE()),
}


@pytest.mark.parametrize("family_code", FAMILY_CASES)
def test_family_quantiles_match_r_gamlss(family_code):
    case_name, family = FAMILY_CASES[family_code]
    cases = pd.read_csv(REFERENCE_DIR / case_name)
    reference = pd.read_csv(
        REFERENCE_DIR / "quantile_prediction_reference.csv"
    )
    reference = reference[reference["family"] == family_code]
    probabilities = torch.tensor(
        reference["probability"].unique(),
        dtype=torch.float64,
    )
    parameters = {
        parameter: torch.tensor(
            cases[parameter].to_numpy(),
            dtype=torch.float64,
        )
        for parameter in family.parameter_names
    }

    actual = family.quantile(probabilities, parameters)
    expected = torch.tensor(
        reference["quantile"].to_numpy().reshape(len(cases), -1),
        dtype=torch.float64,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual.shape == (len(cases), probabilities.numel())
    assert torch.all(torch.diff(actual, dim=1) >= 0)

    expanded_parameters = {
        parameter: value.unsqueeze(1).expand_as(actual)
        for parameter, value in parameters.items()
    }
    expanded_probabilities = probabilities.unsqueeze(0).expand_as(actual)
    if family.is_discrete:
        upper = family.cdf(actual, expanded_parameters)
        lower = family.cdf(actual - 1.0, expanded_parameters)
        assert torch.all(upper >= expanded_probabilities - 1e-12)
        assert torch.all(lower < expanded_probabilities + 1e-12)
    else:
        tolerance = 1e-4 if family_code == "BE" else 1e-8
        torch.testing.assert_close(
            family.cdf(actual, expanded_parameters),
            expanded_probabilities,
            rtol=tolerance,
            atol=tolerance,
        )


def test_family_quantile_validates_probabilities_and_parameters():
    family = Normal()
    parameters = {
        "mu": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "sigma": torch.tensor([1.0, 2.0], dtype=torch.float64),
    }

    for probabilities in ([], [0.0], [1.0], [float("nan")], [[0.5]]):
        with pytest.raises(ValueError, match="probabilities"):
            family.quantile(probabilities, parameters)
    with pytest.raises(ValueError, match="parameters"):
        family.quantile([0.5], {"mu": parameters["mu"]})
    with pytest.raises(ValueError, match="dtype and device"):
        family.quantile(
            [0.5],
            {
                "mu": parameters["mu"],
                "sigma": parameters["sigma"].float(),
            },
        )


def test_model_predicts_quantiles_and_centiles():
    x = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=torch.float64),
    }
    model = GAMLSS(Normal(), {"mu": 2, "sigma": 1}, dtype=torch.float64)
    with torch.no_grad():
        model.coefficients["mu"].copy_(
            torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
        model.coefficients["sigma"].fill_(math.log(0.5))

    prediction = model.predict_quantiles(
        design,
        probabilities=[0.1, 0.5, 0.9],
    )
    centiles = model.predict_centiles(
        design,
        centiles=[10, 50, 90],
    )

    assert isinstance(prediction, QuantilePrediction)
    assert prediction.family == "NO"
    assert prediction.quantiles.shape == (x.numel(), 3)
    torch.testing.assert_close(prediction.quantiles, centiles.quantiles)
    torch.testing.assert_close(
        prediction.centiles,
        torch.tensor([10.0, 50.0, 90.0], dtype=torch.float64),
    )
    torch.testing.assert_close(prediction.at(0.5), 1.0 + 2.0 * x)
    table = prediction.to_dataframe()
    assert tuple(table.columns) == (
        "observation",
        "probability",
        "centile",
        "quantile",
    )
    assert len(table) == 3 * x.numel()
    with pytest.raises(KeyError, match="not uniquely stored"):
        prediction.at(0.25)
    with pytest.raises(ValueError, match="between zero and 100"):
        model.predict_centiles(design, centiles=[0, 50])


def test_quantile_bootstrap_reselects_smoothing_and_is_reproducible():
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
    new_data = data.iloc[::4].drop(columns="y")

    first = model.centile_bootstrap_data(
        data,
        centiles=[10, 50, 90],
        new_data=new_data,
        replicates=10,
        generator=torch.Generator().manual_seed(2030),
    )
    second = model.quantile_bootstrap_data(
        data,
        probabilities=[0.1, 0.5, 0.9],
        new_data=new_data,
        replicates=10,
        generator=torch.Generator().manual_seed(2030),
    )

    assert isinstance(first, QuantileBootstrapResult)
    assert first.family == "NO"
    assert first.algorithm == "rs"
    assert first.replicates == 10
    assert first.failed_replicates == first.attempts - first.replicates
    assert first.failure_rate == pytest.approx(
        first.failed_replicates / first.attempts
    )
    assert first.estimates.shape == (len(new_data), 3)
    assert first.bootstrap_estimates.shape == (10, len(new_data), 3)
    assert first.standard_errors.shape == (len(new_data), 3)
    assert first.confidence_intervals.shape == (len(new_data), 3, 2)
    torch.testing.assert_close(first.bootstrap_estimates, second.bootstrap_estimates)
    torch.testing.assert_close(
        first.estimates,
        model.predict_centiles_data(
            new_data,
            centiles=[10, 50, 90],
        ).quantiles,
    )
    torch.testing.assert_close(
        torch.diagonal(first.covariance_matrix),
        first.standard_errors.flatten().square(),
    )
    assert first.at(0.5).shape == (len(new_data), 2)

    joint_band = first.simultaneous_confidence_bands()
    per_centile_bands = first.simultaneous_confidence_bands(joint=False)
    assert isinstance(joint_band, QuantileBandResult)
    assert joint_band.joint
    assert joint_band.critical_values.shape == (1,)
    assert joint_band.confidence_intervals.shape == (len(new_data), 3, 2)
    assert per_centile_bands.critical_values.shape == (3,)
    assert torch.all(
        joint_band.critical_values[0] >= per_centile_bands.critical_values
    )
    assert len(joint_band.to_dataframe()) == 3 * len(new_data)
    assert tuple(first.to_dataframe().columns) == (
        "observation",
        "probability",
        "centile",
        "estimate",
        "bootstrap_mean",
        "bias",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name])

    cg = model.quantile_bootstrap_data(
        data,
        probabilities=[0.5],
        new_data=new_data,
        replicates=10,
        algorithm="cg",
        generator=torch.Generator().manual_seed(9),
    )
    assert cg.algorithm == "cg"
    assert cg.bootstrap_estimates.shape == (10, len(new_data), 1)


def test_quantile_bootstrap_supports_whole_model_laml_refits():
    observation_count = 40
    x = torch.linspace(-1.0, 1.0, observation_count, dtype=torch.float64)
    generator = torch.Generator().manual_seed(44)
    response = (
        0.4
        + torch.sin(2.5 * x)
        + 0.16
        * torch.randn(
            observation_count,
            dtype=torch.float64,
            generator=generator,
        )
    )
    data = pd.DataFrame({"y": response.numpy(), "x": x.numpy()})
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x, intervals=4)", "sigma": "~ 1"},
        data,
    )
    control = LAMLControl(
        outer_max_iterations=25,
        outer_gradient_tolerance=1e-4,
    )
    fit = model.fit_laml_data(data, control=control)
    original_smoothing_parameter = (
        model.smooth_terms["mu"]["x"].smoothing_parameter
    )
    new_data = data.iloc[::10].drop(columns="y")

    result = model.centile_bootstrap_data(
        data,
        centiles=[10, 50, 90],
        new_data=new_data,
        replicates=10,
        max_attempts=20,
        algorithm="laml",
        control=control,
        generator=torch.Generator().manual_seed(99),
    )

    assert fit.outer_converged
    assert result.algorithm == "laml"
    assert result.bootstrap_estimates.shape == (10, len(new_data), 3)
    assert torch.isfinite(result.bootstrap_estimates).all()
    assert (result.standard_errors > 0).all()
    assert model.smooth_terms["mu"]["x"].smoothing_parameter == pytest.approx(
        original_smoothing_parameter
    )
