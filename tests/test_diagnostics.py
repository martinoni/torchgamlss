import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

from torchgamlss import (
    BCCG,
    BCPE,
    BCT,
    GAMLSS,
    Beta,
    BucketPlotResult,
    BucketStatistics,
    Gamma,
    ModelDiagnostics,
    NegativeBinomial,
    Normal,
    Poisson,
    PSpline,
    QuantileResidualSummary,
    ResidualDiagnosticPlot,
    RSControl,
    WormPlotResult,
    bucket_plot,
    compare_models,
    worm_plot,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _rows(name: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return list(csv.DictReader(data_file))


def _diagnostic_row(family: str) -> dict[str, str]:
    return next(
        row
        for row in _rows("model_diagnostics_reference.csv")
        if row["family"] == family
    )


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
            "GA",
            "ga",
            Gamma(),
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
    ],
)
def test_model_diagnostics_match_r_gamlss(
    family_code, prefix, family, formulas, tolerance
):
    data = pd.read_csv(REFERENCE_DIR / f"{prefix}_fit_data.csv")
    model = GAMLSS.from_formula(family, formulas, data)
    fit = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=tolerance,
            max_outer_iterations=200,
            inner_tolerance=tolerance,
            max_inner_iterations=200,
        ),
    )
    result = model.diagnostics_data(data, weights="weight")
    reference = _diagnostic_row(family_code)

    assert fit.effective_degrees_of_freedom == pytest.approx(
        float(reference["effective_df"])
    )
    assert result.observation_count == int(reference["observation_count"])
    assert result.effective_observation_count == pytest.approx(
        float(reference["effective_observation_count"])
    )
    assert result.effective_degrees_of_freedom == pytest.approx(
        float(reference["effective_df"])
    )
    assert result.residual_degrees_of_freedom == pytest.approx(
        float(reference["residual_df"])
    )
    assert result.log_likelihood == pytest.approx(
        float(reference["log_likelihood"]), rel=2e-7, abs=2e-7
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=2e-7, abs=2e-7
    )
    assert result.aic == pytest.approx(float(reference["aic"]), rel=2e-7, abs=2e-7)
    assert result.aicc == pytest.approx(float(reference["aicc"]), rel=2e-7, abs=2e-7)
    assert result.gaic(3) == pytest.approx(
        float(reference["gaic3"]), rel=2e-7, abs=2e-7
    )
    assert result.bic == pytest.approx(float(reference["sbc"]), rel=2e-7, abs=2e-7)
    assert result.sbc == result.bic


def test_smooth_model_effective_degrees_of_freedom_match_r_gamlss():
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
    design = {
        "mu": torch.column_stack((torch.ones_like(x), x)),
        "sigma": torch.ones((x.numel(), 1), dtype=x.dtype),
    }
    smooth_covariates = {"mu": {"x": x}}
    fit = model.fit_rs(
        response,
        design,
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
    reference = _diagnostic_row("NO_PB")

    assert fit.parameter_effective_degrees_of_freedom["mu"] == pytest.approx(
        fit.smooth_effective_degrees_of_freedom["mu"]["x"]
    )
    assert fit.parameter_effective_degrees_of_freedom["sigma"] == 1.0
    assert fit.effective_degrees_of_freedom == pytest.approx(
        float(reference["effective_df"]), rel=1e-9, abs=1e-9
    )
    diagnostics = model.diagnostics(
        response,
        design,
        smooth_covariates=smooth_covariates,
        degrees_of_freedom=fit.effective_degrees_of_freedom,
    )
    assert diagnostics.residual_degrees_of_freedom == pytest.approx(
        float(reference["residual_df"]), rel=1e-9, abs=1e-9
    )
    assert diagnostics.aic == pytest.approx(float(reference["aic"]), rel=1e-9, abs=1e-9)
    assert diagnostics.aicc == pytest.approx(
        float(reference["aicc"]), rel=1e-9, abs=1e-9
    )
    assert diagnostics.bic == pytest.approx(float(reference["sbc"]), rel=1e-9, abs=1e-9)

    with pytest.raises(ValueError, match="degrees_of_freedom"):
        model.diagnostics(
            response,
            design,
            smooth_covariates=smooth_covariates,
        )


@pytest.mark.parametrize(
    ("family_code", "reference_name", "family"),
    [
        ("NO", "no_reference.csv", Normal()),
        ("GA", "ga_reference.csv", Gamma()),
        ("PO", "po_reference.csv", Poisson()),
        ("NBI", "nbi_reference.csv", NegativeBinomial()),
        ("BE", "be_reference.csv", Beta()),
        ("BCCG", "bccg_reference.csv", BCCG()),
        ("BCT", "bct_reference.csv", BCT()),
        ("BCPE", "bcpe_reference.csv", BCPE()),
    ],
)
def test_quantile_residuals_match_r_gamlss(family_code, reference_name, family):
    family_rows = _rows(reference_name)
    residual_rows = [
        row
        for row in _rows("quantile_residual_reference.csv")
        if row["family"] == family_code
    ]
    response = torch.tensor(
        [float(row["y"]) for row in family_rows],
        dtype=torch.float64,
    )
    parameters = {
        parameter: torch.tensor(
            [float(row[parameter]) for row in family_rows],
            dtype=torch.float64,
        )
        for parameter in family.parameter_names
    }
    expected_probability = torch.tensor(
        [float(row["probability"]) for row in residual_rows],
        dtype=torch.float64,
    )
    expected_residual = torch.tensor(
        [float(row["residual"]) for row in residual_rows],
        dtype=torch.float64,
    )
    observation_count = response.numel()
    model = GAMLSS(
        family,
        {parameter: observation_count for parameter in family.parameter_names},
        dtype=torch.float64,
    )
    design = {
        parameter: torch.eye(observation_count, dtype=torch.float64)
        for parameter in family.parameter_names
    }
    with torch.no_grad():
        for parameter in family.parameter_names:
            model.coefficients[parameter].copy_(
                family.links[parameter](parameters[parameter])
            )

    if family.is_discrete:
        uniforms = torch.tensor(
            [float(row["uniform"]) for row in residual_rows],
            dtype=torch.float64,
        )
        lower = family.cdf(response - 1, parameters)
        upper = family.cdf(response, parameters)
        probability = lower + uniforms * (upper - lower)
    else:
        uniforms = None
        probability = family.cdf(response, parameters)

    torch.testing.assert_close(
        probability,
        expected_probability,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        model.quantile_residuals(response, design, uniforms=uniforms),
        expected_residual,
        rtol=2e-11,
        atol=2e-11,
    )


def test_formula_quantile_residuals_accept_uniform_column():
    data = pd.DataFrame(
        {
            "y": [0.0, 1.0, 3.0, 5.0],
            "u": [0.1, 0.3, 0.7, 0.9],
        }
    )
    model = GAMLSS.from_formula(Poisson(), {"mu": "y ~ 1"}, data)
    model.fit_rs_data(data)

    from_column = model.quantile_residuals_data(data, uniforms="u")
    from_values = model.quantile_residuals_data(data, uniforms=data["u"])

    torch.testing.assert_close(from_column, from_values)


def test_discrete_quantile_residual_generator_is_reproducible():
    response = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    first = model.quantile_residuals(
        response,
        design,
        generator=torch.Generator().manual_seed(2026),
    )
    second = model.quantile_residuals(
        response,
        design,
        generator=torch.Generator().manual_seed(2026),
    )

    torch.testing.assert_close(first, second)


def test_quantile_residuals_validate_randomization_inputs():
    response = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    design = {"mu": torch.ones((3, 1), dtype=torch.float64)}
    discrete = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)
    continuous = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        dtype=torch.float64,
    )
    normal_design = {"mu": design["mu"], "sigma": design["mu"]}

    with pytest.raises(ValueError, match="either uniforms"):
        discrete.quantile_residuals(
            response,
            design,
            uniforms=torch.full_like(response, 0.5),
            generator=torch.Generator(),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        discrete.quantile_residuals(
            response,
            design,
            uniforms=torch.tensor([0.2, 1.1, 0.5], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="discrete"):
        continuous.quantile_residuals(
            response,
            normal_design,
            uniforms=torch.full_like(response, 0.5),
        )
    with pytest.raises(ValueError, match="discrete"):
        continuous.quantile_residuals(
            response,
            normal_design,
            generator=torch.Generator(),
        )


def test_model_comparison_ranks_models_and_calculates_weights():
    diagnostics = {
        "larger": ModelDiagnostics(-45.0, 90.0, 5.0, 95.0, 100, 100.0),
        "smaller": ModelDiagnostics(-46.0, 92.0, 2.0, 98.0, 100, 100.0),
    }

    table = compare_models(diagnostics)

    assert tuple(table.index) == ("smaller", "larger")
    assert table.loc["smaller", "delta"] == 0
    assert table["weight"].sum() == pytest.approx(1.0)
    assert tuple(compare_models(diagnostics, criterion="gaic", penalty=3).index) == (
        "smaller",
        "larger",
    )


def test_diagnostics_and_model_comparison_reject_invalid_inputs():
    result = ModelDiagnostics(-1.0, 2.0, 2.0, 1.0, 3, 3.0)
    incomparable = ModelDiagnostics(-1.0, 2.0, 2.0, 2.0, 4, 4.0)

    assert result.aicc == float("inf")
    with pytest.raises(ValueError, match="penalty"):
        result.gaic(float("nan"))
    with pytest.raises(ValueError, match="at least one"):
        compare_models({})
    with pytest.raises(ValueError, match="comparable"):
        compare_models({"a": result, "b": incomparable})
    with pytest.raises(ValueError, match="finite"):
        compare_models({"a": result}, criterion="aicc")


def test_formula_plot_returns_r_style_four_panel_diagnostics():
    data = pd.read_csv(REFERENCE_DIR / "no_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)

    result = model.plot_data(data, x_variable="x")

    assert isinstance(result, ResidualDiagnosticPlot)
    assert isinstance(result.summary, QuantileResidualSummary)
    assert result.axes == tuple(result.figure.axes)
    assert result.residuals.shape == (len(data),)
    assert result.fitted_values.shape == (len(data),)
    torch.testing.assert_close(
        result.x_values,
        torch.tensor(data["x"].to_numpy(), dtype=torch.float64),
    )
    assert [axis.get_title() for axis in result.axes] == [
        "Residuals vs fitted mu",
        "Residuals vs x",
        "Residual density",
        "Normal Q-Q plot",
    ]
    assert result.summary.observation_count == len(data)
    reference = _rows("residual_plot_summary_reference.csv")[0]
    assert result.summary.observation_count == int(
        reference["observation_count"]
    )
    for statistic in (
        "mean",
        "variance",
        "skewness",
        "kurtosis",
        "filliben_correlation",
    ):
        assert getattr(result.summary, statistic) == pytest.approx(
            float(reference[statistic]),
            rel=1e-12,
            abs=1e-12,
        )
    plt.close(result.figure)


def test_plot_expands_integer_frequency_weights_and_drops_zero_weights():
    response = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    weights = torch.tensor([2.0, 0.0, 1.0, 3.0], dtype=torch.float64)
    x = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
    design = {
        "mu": torch.ones((4, 1), dtype=torch.float64),
        "sigma": torch.ones((4, 1), dtype=torch.float64),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        dtype=torch.float64,
    )

    result = model.plot(
        response,
        design,
        weights=weights,
        x_variable=x,
        x_label="Exposure",
    )

    torch.testing.assert_close(
        result.x_values,
        torch.tensor(
            [10.0, 10.0, 30.0, 40.0, 40.0, 40.0],
            dtype=torch.float64,
        ),
    )
    assert result.summary.observation_count == int(weights.sum())
    assert result.axes[1].get_xlabel() == "Exposure"
    plt.close(result.figure)


def test_discrete_plot_randomization_is_reproducible():
    response = torch.tensor(
        [0.0, 1.0, 1.0, 2.0, 3.0, 5.0],
        dtype=torch.float64,
    )
    design = {"mu": torch.ones((response.numel(), 1), dtype=torch.float64)}
    model = GAMLSS(Poisson(), {"mu": 1}, dtype=torch.float64)

    first = model.plot(
        response,
        design,
        generator=torch.Generator().manual_seed(2026),
    )
    second = model.plot(
        response,
        design,
        generator=torch.Generator().manual_seed(2026),
    )

    torch.testing.assert_close(first.residuals, second.residuals)
    assert first.summary == second.summary
    plt.close(first.figure)
    plt.close(second.figure)


def test_time_series_plot_and_axes_validation():
    response = torch.tensor(
        [
            -0.7,
            -0.1,
            0.4,
            1.1,
            0.5,
            -0.2,
            -1.0,
            -0.4,
            0.3,
            0.9,
        ],
        dtype=torch.float64,
    )
    design = {
        "mu": torch.ones((response.numel(), 1), dtype=torch.float64),
        "sigma": torch.ones((response.numel(), 1), dtype=torch.float64),
    }
    model = GAMLSS(
        Normal(),
        {"mu": 1, "sigma": 1},
        dtype=torch.float64,
    )

    result = model.plot(
        response,
        design,
        time_series=True,
        max_lag=4,
    )

    assert result.axes[0].get_title() == "Residual autocorrelation"
    assert result.axes[1].get_title() == (
        "Residual partial autocorrelation"
    )
    plt.close(result.figure)

    figure, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="exactly four"):
        model.plot(response, design, axes=axes)
    plt.close(figure)
    with pytest.raises(ValueError, match="cannot be supplied"):
        model.plot(
            response,
            design,
            time_series=True,
            x_variable=response,
        )
    with pytest.raises(ValueError, match="max_lag"):
        model.plot(
            response,
            design,
            time_series=True,
            max_lag=6,
        )
    with pytest.raises(ValueError, match="requires time_series"):
        model.plot(response, design, max_lag=2)


def test_standalone_worm_plot_matches_r_coefficients_and_bands():
    observation_count = 40
    probabilities = (
        torch.arange(1, observation_count + 1, dtype=torch.float64) - 0.5
    ) / observation_count
    quantiles = torch.special.ndtri(probabilities)
    residuals = (
        quantiles
        + 0.12
        + 0.08 * quantiles.square()
        - 0.03 * quantiles.pow(3)
    )

    result = worm_plot(residuals)

    assert isinstance(result, WormPlotResult)
    assert result.axes == (result.figure.axes[0],)
    assert result.intervals is None
    assert result.x_values is None
    assert len(result.panels) == 1
    assert result.coefficients is not None
    reference = next(
        row
        for row in _rows("worm_plot_reference.csv")
        if row["scope"] == "global"
    )
    expected = torch.tensor(
        [
            float(reference["intercept"]),
            float(reference["linear"]),
            float(reference["quadratic"]),
            float(reference["cubic"]),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        result.coefficients,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    panel = result.panels[0]
    torch.testing.assert_close(
        panel.deviations,
        panel.residuals - panel.theoretical_quantiles,
    )
    torch.testing.assert_close(
        panel.confidence_upper,
        -panel.confidence_lower,
    )
    assert panel.confidence_grid[1] - panel.confidence_grid[0] == (
        pytest.approx(0.25)
    )
    plt.close(result.figure)


def test_conditioned_worm_plot_matches_r_intervals_and_coefficients():
    observation_count = 40
    probabilities = (
        torch.arange(1, observation_count + 1, dtype=torch.float64) - 0.5
    ) / observation_count
    quantiles = torch.special.ndtri(probabilities)
    residuals = (
        quantiles
        + 0.12
        + 0.08 * quantiles.square()
        - 0.03 * quantiles.pow(3)
    )
    x_variable = torch.linspace(
        -2.0,
        2.0,
        observation_count,
        dtype=torch.float64,
    )

    result = worm_plot(
        residuals,
        x_variable=x_variable,
        x_label="Age",
        n_intervals=4,
        overlap=0.0,
    )

    conditioned = [
        row
        for row in _rows("worm_plot_reference.csv")
        if row["scope"] == "conditioned"
    ]
    expected_intervals = torch.tensor(
        [
            [float(row["lower"]), float(row["upper"])]
            for row in conditioned
        ],
        dtype=torch.float64,
    )
    expected_coefficients = torch.tensor(
        [
            [
                float(row["intercept"]),
                float(row["linear"]),
                float(row["quadratic"]),
                float(row["cubic"]),
            ]
            for row in conditioned
        ],
        dtype=torch.float64,
    )
    assert result.intervals is not None
    assert result.coefficients is not None
    torch.testing.assert_close(
        result.intervals,
        expected_intervals,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.coefficients,
        expected_coefficients,
        rtol=1e-11,
        atol=1e-12,
    )
    assert len(result.panels) == 4
    assert len(result.axes) == 4
    assert all(axis.get_title().startswith("Age:") for axis in result.axes)
    assert [panel.residuals.numel() for panel in result.panels] == [10] * 4
    plt.close(result.figure)


def test_formula_wp_and_worm_plot_validation():
    data = pd.read_csv(REFERENCE_DIR / "no_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)

    result = model.wp_data(data, x_variable="x", n_intervals=2)
    expected_residuals = model.quantile_residuals_data(data).cpu()

    torch.testing.assert_close(result.residuals, expected_residuals)
    assert result.x_values is not None
    assert result.intervals is not None
    assert result.intervals.shape == (2, 2)
    assert result.coefficients is not None
    assert result.coefficients.shape == (2, 4)
    assert torch.isnan(result.coefficients[:, 3]).all()
    plt.close(result.figure)

    no_line = worm_plot(
        [-1.0, -0.2, 0.3, 1.1],
        show_polynomial=False,
    )
    assert no_line.coefficients is None
    assert no_line.panels[0].coefficients is None
    plt.close(no_line.figure)

    with pytest.raises(ValueError, match="at least two"):
        worm_plot([0.0])
    with pytest.raises(ValueError, match="cut_points requires"):
        worm_plot([-1.0, 0.0, 1.0], cut_points=[0.0])
    with pytest.raises(ValueError, match="one value per residual"):
        worm_plot([-1.0, 0.0, 1.0], x_variable=[1.0, 2.0])
    with pytest.raises(ValueError, match="overlap"):
        worm_plot(
            [-1.0, 0.0, 1.0],
            x_variable=[1.0, 2.0, 3.0],
            overlap=1.0,
        )
    figure, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="exactly 2"):
        worm_plot(
            [-1.0, -0.2, 0.3, 1.1],
            x_variable=[1.0, 2.0, 3.0, 4.0],
            n_intervals=2,
            axes=axes,
        )
    plt.close(figure)


def _bucket_reference_inputs():
    observation_count = 80
    probabilities = (
        torch.arange(1, observation_count + 1, dtype=torch.float64) - 0.5
    ) / observation_count
    quantiles = torch.special.ndtri(probabilities)
    residuals = (
        quantiles
        + 0.14 * quantiles.square()
        - 0.025 * quantiles.pow(3)
        + 0.04
        * torch.sin(
            torch.linspace(
                0.0,
                4.0 * torch.pi,
                observation_count,
                dtype=torch.float64,
            )
        )
    )
    weights = torch.tensor(
        [1.0, 2.0, 0.5, 1.5] * (observation_count // 4),
        dtype=torch.float64,
    )
    x_variable = torch.linspace(
        -2.0,
        2.0,
        observation_count,
        dtype=torch.float64,
    )
    return residuals, weights, x_variable


@pytest.mark.parametrize(
    "kind",
    ["moment", "centile.central", "centile.tail"],
)
def test_bucket_plot_statistics_match_r(kind):
    residuals, weights, _ = _bucket_reference_inputs()

    result = bucket_plot(
        residuals,
        weights=weights,
        kind=kind,
        bootstrap=False,
    )

    assert isinstance(result, BucketPlotResult)
    assert result.kind == kind
    assert len(result.panels) == 1
    statistics = result.panels[0].statistics
    assert isinstance(statistics, BucketStatistics)
    reference = next(
        row
        for row in _rows("bucket_plot_reference.csv")
        if row["kind"] == kind
    )
    assert statistics.observation_count == int(
        reference["observation_count"]
    )
    assert statistics.effective_observation_count == pytest.approx(
        float(reference["effective_observation_count"]),
        rel=1e-12,
        abs=1e-12,
    )
    for field in (
        "skewness",
        "transformed_skewness",
        "kurtosis",
        "excess_kurtosis",
        "transformed_kurtosis",
    ):
        assert getattr(statistics, field) == pytest.approx(
            float(reference[field]),
            rel=1e-11,
            abs=1e-12,
        )
    if kind == "moment":
        assert statistics.jarque_bera == pytest.approx(
            float(reference["jarque_bera"]),
            rel=1e-11,
            abs=1e-12,
        )
    else:
        assert statistics.jarque_bera is None
    assert result.panels[0].bootstrap_points.shape == (0, 2)
    plt.close(result.figure)


def test_conditioned_bucket_plot_matches_r_and_bootstrap_is_reproducible():
    residuals, weights, x_variable = _bucket_reference_inputs()

    first = bucket_plot(
        residuals,
        weights=weights,
        x_variable=x_variable,
        x_label="Age",
        bootstrap_replicates=12,
        generator=torch.Generator().manual_seed(2026),
    )
    second = bucket_plot(
        residuals,
        weights=weights,
        x_variable=x_variable,
        x_label="Age",
        bootstrap_replicates=12,
        generator=torch.Generator().manual_seed(2026),
    )

    conditioned = [
        row
        for row in _rows("bucket_plot_reference.csv")
        if row["kind"] == "moment.conditioned"
    ]
    assert first.intervals is not None
    expected_intervals = torch.tensor(
        [
            [float(row["lower"]), float(row["upper"])]
            for row in conditioned
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        first.intervals,
        expected_intervals,
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(first.panels) == 4
    for panel, repeated_panel, reference in zip(
        first.panels,
        second.panels,
        conditioned,
    ):
        torch.testing.assert_close(
            panel.bootstrap_points,
            repeated_panel.bootstrap_points,
        )
        for field in (
            "skewness",
            "transformed_skewness",
            "kurtosis",
            "excess_kurtosis",
            "transformed_kurtosis",
            "jarque_bera",
        ):
            assert getattr(panel.statistics, field) == pytest.approx(
                float(reference[field]),
                rel=1e-11,
                abs=1e-12,
            )
        assert panel.bootstrap_points.shape == (12, 2)
        assert panel.axis.get_title().startswith("Age:")
    plt.close(first.figure)
    plt.close(second.figure)


def test_formula_bp_and_bucket_plot_validation():
    data = pd.read_csv(REFERENCE_DIR / "no_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )
    model.fit_rs_data(data)

    result = model.bp_data(
        data,
        bootstrap=False,
        label="Normal fit",
    )

    torch.testing.assert_close(
        result.residuals,
        model.quantile_residuals_data(data).cpu(),
    )
    assert result.panels[0].statistics.observation_count == len(data)
    assert result.panels[0].axis.get_title() == "Moment bucket plot"
    plt.close(result.figure)

    with pytest.raises(ValueError, match="at least four"):
        bucket_plot([-1.0, 0.0, 1.0], bootstrap=False)
    with pytest.raises(ValueError, match="kind"):
        bucket_plot(
            [-1.0, -0.2, 0.3, 1.1],
            kind="invalid",
            bootstrap=False,
        )
    with pytest.raises(ValueError, match="one value per residual"):
        bucket_plot(
            [-1.0, -0.2, 0.3, 1.1],
            weights=[1.0, 1.0],
            bootstrap=False,
        )
    with pytest.raises(ValueError, match="positive"):
        bucket_plot(
            [-1.0, -0.2, 0.3, 1.1],
            bootstrap=True,
            bootstrap_replicates=0,
        )
    figure, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="exactly 2"):
        bucket_plot(
            [-1.0, -0.2, 0.3, 1.1],
            x_variable=[1.0, 2.0, 3.0, 4.0],
            n_intervals=2,
            bootstrap=False,
            axes=axes,
        )
    plt.close(figure)
