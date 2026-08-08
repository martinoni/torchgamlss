import csv
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgamlss import (
    GAMLSS,
    AdaptivePSpline,
    AdaptiveTensorPSpline,
    CGControl,
    CyclicPSpline,
    Gamma,
    MarkovRandomField,
    MiniBatchControl,
    Normal,
    RSControl,
    ShrinkagePSpline,
    ThinPlateSpline,
)

REFERENCE_DIR = Path(__file__).parent / "reference"
MRF_EDGES = (
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
)
MRF_LABELS = (
    "north_west",
    "north",
    "north_east",
    "west",
    "center",
    "east",
    "south_west",
    "south",
    "south_east",
)
MRF_ADJACENCY = {
    label: tuple(
        MRF_LABELS[second if first == index else first]
        for first, second in MRF_EDGES
        if index in (first, second)
    )
    for index, label in enumerate(MRF_LABELS)
}
MRF_EDGE_WEIGHTS = (
    0.7,
    1.1,
    0.9,
    1.4,
    0.8,
    1.3,
    1.05,
    0.65,
    1.55,
    1.2,
    0.95,
    1.35,
)
MRF_POLYGONS = {
    f"area_{index}": (
        (float(index), 0.0),
        (float(index + 1), 0.0),
        (float(index) + 0.5, 0.8),
    )
    for index in range(6)
}


def _mrf_graph_penalty_tuple() -> tuple[tuple[float, ...], ...]:
    penalty = torch.zeros((9, 9), dtype=torch.float64)
    for (first, second), weight in zip(MRF_EDGES, MRF_EDGE_WEIGHTS, strict=True):
        penalty[first, first] += weight
        penalty[second, second] += weight
        penalty[first, second] -= weight
        penalty[second, first] -= weight
    return tuple(tuple(float(value) for value in row) for row in penalty)


def _reference_row(name: str) -> dict[str, str]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as data_file:
        return next(csv.DictReader(data_file))


def test_formula_weighted_normal_rs_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_rs_fit_data.csv")
    reference = _reference_row("no_rs_reference.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
        ),
    )

    assert model.formula_response_name == "y"
    assert model.formula_column_names == {
        "mu": ("Intercept", "x"),
        "sigma": ("Intercept", "z"),
    }
    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor(
            [
                float(reference["sigma_intercept"]),
                float(reference["sigma_z"]),
            ],
            dtype=torch.float64,
        ),
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-12, abs=1e-12
    )


def test_formula_normal_lbfgs_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_fit_data.csv")
    reference = _reference_row("no_fit_reference.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x", "sigma": "~ 1"},
        data,
    )

    result = model.fit_data(
        data,
        max_iter=200,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    assert result.converged
    torch.testing.assert_close(
        model.coefficients["mu"],
        torch.tensor(
            [float(reference["mu_intercept"]), float(reference["mu_x"])],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    torch.testing.assert_close(
        model.coefficients["sigma"],
        torch.tensor(
            [float(reference["sigma_intercept"])],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result.negative_log_likelihood == pytest.approx(
        float(reference["negative_log_likelihood"]), rel=1e-10, abs=1e-10
    )


def test_formula_gamma_pspline_fit_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "ga_fit_data.csv")
    reference = _reference_row("ga_pb_reference.csv")
    model = GAMLSS.from_formula(
        Gamma(),
        {
            "mu": "y ~ pb(x, smoothing_parameter=12) + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-9,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
            backfitting_tolerance=1e-10,
            max_backfitting_iterations=200,
        ),
    )

    term = model.smooth_terms["mu"]["x"]
    assert result.converged
    assert term.smoothing_parameter == pytest.approx(12.0)
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-12, abs=1e-12
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-12, abs=1e-11
    )
    predicted = model.predict_data(data)
    assert torch.isfinite(predicted["mu"]).all()
    assert torch.isfinite(predicted["sigma"]).all()


def test_formula_automatic_ml_pspline_matches_r_gamlss():
    data = pd.read_csv(REFERENCE_DIR / "no_pb_fit_data.csv")
    reference = _reference_row("no_pb_ml_reference.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ pb(x)", "sigma": "~ 1"},
        data,
    )

    result = model.fit_rs_data(
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

    expected_lambda = float(reference["smoothing_parameter"])
    assert result.converged
    assert model.smooth_terms["mu"]["x"].smoothing_parameter == pytest.approx(
        expected_lambda, rel=1e-8, abs=1e-8
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == pytest.approx(
        float(reference["smooth_edf"]), rel=1e-8, abs=1e-8
    )
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-9, abs=1e-9
    )


def test_formula_fixed_cyclic_pspline_matches_r_gamlss_pbc():
    data = pd.read_csv(REFERENCE_DIR / "no_pbc_fit_data.csv")
    reference = _reference_row("no_pbc_reference.csv")
    fitted_reference = pd.read_csv(
        REFERENCE_DIR / "no_pbc_fitted_reference.csv"
    )
    coefficient_reference = pd.read_csv(
        REFERENCE_DIR / "no_pbc_coefficient_reference.csv"
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pbc(x, lambda_=12, intervals=12)",
            "sigma": "~ 1",
        },
        data,
    )

    result = model.fit_rs_data(
        data,
        control=RSControl(
            outer_tolerance=1e-8,
            max_outer_iterations=200,
            inner_tolerance=1e-8,
            max_inner_iterations=200,
            backfitting_tolerance=1e-8,
            max_backfitting_iterations=1000,
        ),
    )
    term = model.smooth_terms["mu"]["x"]
    predicted = model.predict_data(data)

    assert isinstance(term, CyclicPSpline)
    assert model.formula_column_names["mu"] == ("Intercept",)
    assert result.converged
    assert result.outer_iterations == int(reference["outer_iterations"])
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), rel=1e-9, abs=1e-9
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == (
        pytest.approx(float(reference["smooth_edf"]), rel=1e-9, abs=1e-9)
    )
    torch.testing.assert_close(
        term.coefficients,
        torch.tensor(
            coefficient_reference["coefficient"].to_numpy(),
            dtype=torch.float64,
        ),
        rtol=1e-9,
        atol=1e-9,
    )
    for parameter in ("mu", "sigma"):
        torch.testing.assert_close(
            predicted[parameter],
            torch.tensor(
                fitted_reference[parameter].to_numpy(),
                dtype=torch.float64,
            ),
            rtol=1e-9,
            atol=1e-9,
        )


def test_formula_fixed_shrinkage_pspline_matches_r_gamlss_pbz():
    data = pd.read_csv(REFERENCE_DIR / "no_pbz_fit_data.csv")
    reference = _reference_row("no_pbz_reference.csv")
    fitted_reference = pd.read_csv(
        REFERENCE_DIR / "no_pbz_fitted_reference.csv"
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pbz(x, lambda_=(100000, 50), intervals=20)",
            "sigma": "~ 1",
        },
        data,
    )

    result = model.fit_rs_data(data)
    term = model.smooth_terms["mu"]["x"]
    predicted = model.predict_data(data)

    assert isinstance(term, ShrinkagePSpline)
    assert model.formula_column_names["mu"] == ("Intercept",)
    assert result.converged
    assert term.smoothing_parameters == (100000.0, 50.0)
    assert result.global_deviance == pytest.approx(
        float(reference["global_deviance"]), abs=7e-4
    )
    assert result.smooth_effective_degrees_of_freedom["mu"]["x"] == (
        pytest.approx(float(reference["smooth_edf"]), abs=7e-3)
    )
    torch.testing.assert_close(
        predicted["mu"],
        torch.tensor(fitted_reference["mu"].to_numpy(), dtype=torch.float64),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        predicted["sigma"],
        torch.tensor(fitted_reference["sigma"].to_numpy(), dtype=torch.float64),
        rtol=3e-6,
        atol=3e-6,
    )


def test_formula_reuses_categorical_encoding_for_new_data():
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5, 2.0, 2.4],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "group": ["a", "b", "c", "a", "b", "c"],
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ x + C(group)", "sigma": "~ 1"},
        data,
    )
    with torch.no_grad():
        model.coefficients["mu"].copy_(
            torch.tensor([0.5, 0.8, 0.2, -0.3], dtype=torch.float64)
        )
        model.coefficients["sigma"].fill_(-1.0)
    new_data = pd.DataFrame(
        {
            "x": [-0.5, 0.5, 0.8],
            "group": ["c", "a", "b"],
        }
    )

    prepared = model.prepare_formula_data(new_data)
    prediction = model.predict_data(new_data, type="link")

    assert model.formula_column_names["mu"] == (
        "Intercept",
        "x",
        "C(group)[T.b]",
        "C(group)[T.c]",
    )
    torch.testing.assert_close(
        prepared.design_matrices["mu"],
        torch.tensor(
            [
                [1.0, -0.5, 0.0, 1.0],
                [1.0, 0.5, 0.0, 0.0],
                [1.0, 0.8, 1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        prediction["mu"],
        prepared.design_matrices["mu"] @ model.coefficients["mu"],
    )


def test_formula_rejects_unseen_categorical_levels():
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5],
            "group": ["a", "b", "a", "b"],
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {"mu": "y ~ C(group)", "sigma": "~ 1"},
        data,
    )

    with pytest.raises(ValueError, match="new formula data"):
        model.predict_data(pd.DataFrame({"group": ["unseen"]}))


def test_formula_pb_options_create_the_requested_smoother():
    data = pd.DataFrame(
        {
            "y": torch.linspace(0.2, 1.2, 30).numpy(),
            "x": torch.linspace(-1.0, 1.0, 30).numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pb(x, df=3, name='trend')",
            "sigma": "~ 1",
        },
        data,
    )

    term = model.smooth_terms["mu"]["trend"]
    assert set(model.smooth_terms["mu"]) == {"trend"}
    assert term.smoothing_method == "DF"
    assert term.target_effective_degrees_of_freedom == pytest.approx(5.0)
    assert "pb(x, df=3, name='trend')" in model.formula_column_names["mu"]


def test_formula_pb_aliases_and_model_dtype_are_preserved():
    data = pd.DataFrame(
        {
            "y": torch.linspace(0.2, 1.2, 120).numpy(),
            "x": torch.linspace(-1.0, 1.0, 120).numpy(),
        }
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ pb(x, method='GAIC', k=3, inter=8)",
            "sigma": "~ 1",
        },
        data,
        dtype=torch.float32,
    )

    term = model.smooth_terms["mu"]["x"]
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    assert term.smoothing_method == "GAIC"
    assert term.criterion_penalty == pytest.approx(3.0)
    assert term.intervals == 8
    assert prepared.design_matrices["mu"].dtype == torch.float32
    assert prepared.smooth_covariates["mu"]["x"].dtype == torch.float32


def test_formula_tp_builds_a_multivariate_thin_plate_term():
    data = pd.read_csv(REFERENCE_DIR / "tp_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ tp(x, z, k=15, m=2, lambda_=2.5, name='surface')",
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    term = model.smooth_terms["mu"]["surface"]

    assert isinstance(term, ThinPlateSpline)
    assert term.dimension == 2
    assert term.basis_size == 15
    assert term.penalty_order == 2
    assert term.smoothing_parameter == pytest.approx(2.5)
    assert model.formula_column_names["mu"] == ("Intercept",)
    assert prepared.smooth_covariates["mu"]["surface"].shape == (90, 2)
    assert term.design(prepared.smooth_covariates["mu"]["surface"]).shape == (
        90,
        14,
    )


def test_formula_ad_builds_an_adaptive_pspline_term():
    data = pd.read_csv(REFERENCE_DIR / "adaptive_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ ad(x, k=20, m=5, "
                "lambda_=(0.4, 1.2, 4, 1.8, 0.6), name='trend')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    term = model.smooth_terms["mu"]["trend"]

    assert isinstance(term, AdaptivePSpline)
    assert term.basis_size == 20
    assert term.penalty_basis_size == 5
    assert term.smoothing_parameters == pytest.approx((0.4, 1.2, 4, 1.8, 0.6))
    assert model.formula_column_names["mu"] == ("Intercept",)
    assert prepared.smooth_covariates["mu"]["trend"].shape == (100,)
    assert term.design(prepared.smooth_covariates["mu"]["trend"]).shape == (
        100,
        19,
    )


def test_formula_fixed_adaptive_pspline_runs_all_fitters():
    data = pd.read_csv(REFERENCE_DIR / "adaptive_fit_data.csv")
    formulas = {
        "mu": "y ~ ad(x, k=12, m=3, lambda_=(0.5, 2, 1))",
        "sigma": "~ 1",
    }

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
    full_batch_model = GAMLSS.from_formula(Normal(), formulas, data)
    full_batch_result = full_batch_model.fit_data(
        data,
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
    )
    minibatch_model = GAMLSS.from_formula(Normal(), formulas, data)
    minibatch_result = minibatch_model.fit_minibatch_data(
        data,
        control=MiniBatchControl(
            batch_size=25,
            epochs=2,
            learning_rate=0.01,
            shuffle=False,
            minimum_epochs=2,
            patience=10,
            evaluation_frequency=1,
        ),
    )

    assert rs_result.converged
    assert cg_result.converged
    assert full_batch_result.converged
    assert minibatch_result.updates == 8
    assert all(
        math.isfinite(result.negative_log_likelihood)
        for result in (
            rs_result,
            cg_result,
            full_batch_result,
            minibatch_result,
        )
    )
    assert math.isfinite(minibatch_result.penalized_objective)


def test_formula_ad_builds_a_two_dimensional_adaptive_pspline():
    data = pd.read_csv(REFERENCE_DIR / "adaptive_2d_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                "y ~ ad(x, z, k=(7, 6), m=(3, 3), "
                "lambda_=(0.3, 0.7, 1.5, 3, 2.2, 1.1, 0.8, 1.7, 0.5), "
                "name='surface')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    term = model.smooth_terms["mu"]["surface"]

    assert isinstance(term, AdaptiveTensorPSpline)
    assert term.basis_sizes == (7, 6)
    assert term.penalty_basis_shape == (3, 3)
    assert model.formula_column_names["mu"] == ("Intercept",)
    assert prepared.smooth_covariates["mu"]["surface"].shape == (120, 2)
    assert term.design(prepared.smooth_covariates["mu"]["surface"]).shape == (
        120,
        41,
    )


def test_formula_mrf_builds_an_areal_graph_smooth():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    edges = (
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                f"y ~ mrf(region, edges={edges}, area_count=9, "
                "lambda_=2.4, name='area')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    term = model.smooth_terms["mu"]["area"]

    assert isinstance(term, MarkovRandomField)
    assert term.area_count == 9
    assert len(term.edges) == 12
    assert term.smoothing_parameter == pytest.approx(2.4)
    assert model.formula_column_names["mu"] == ("Intercept",)
    assert prepared.smooth_covariates["mu"]["area"].shape == (135,)
    assert term.design(prepared.smooth_covariates["mu"]["area"]).shape == (
        135,
        8,
    )


def test_formula_mrf_builds_a_low_rank_graph_smooth():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    edges = (
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
    )
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                f"y ~ mrf(region, edges={edges}, area_count=9, k=6, "
                "lambda_=1.7, name='area')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data.drop(columns="y"))
    term = model.smooth_terms["mu"]["area"]

    assert isinstance(term, MarkovRandomField)
    assert term.basis_size == 6
    assert term.coefficients.numel() == 5
    assert term.design(prepared.smooth_covariates["mu"]["area"]).shape == (
        135,
        5,
    )


def test_formula_mrf_encodes_named_adjacency_and_unobserved_graph_regions():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    data["region_label"] = data["region"].map(dict(enumerate(MRF_LABELS)))
    training = data.loc[data["region"] != 8, ["y", "region_label"]]
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                f"y ~ mrf(region_label, adjacency={MRF_ADJACENCY!r}, "
                "k=6, lambda_=1.7, name='area')"
            ),
            "sigma": "~ 1",
        },
        training,
    )
    term = model.smooth_terms["mu"]["area"]
    prediction = pd.DataFrame(
        {"region_label": ["south_east", "center", "north_west"]}
    )
    prepared = model.prepare_formula_data(prediction)
    encoded = prepared.smooth_covariates["mu"]["area"]

    assert model.formula_column_names["mu"] == ("Intercept",)
    assert term.region_labels == MRF_LABELS
    assert term.area_count == 9
    assert term.basis_size == 6
    torch.testing.assert_close(
        encoded,
        torch.tensor([8.0, 4.0, 0.0], dtype=torch.float64),
    )
    assert torch.isfinite(term.predict_design(encoded)).all()

    with pytest.raises(ValueError, match="unknown region"):
        model.prepare_formula_data(
            pd.DataFrame({"region_label": ["outside_graph"]})
        )


def test_formula_mrf_accepts_nb_alias_and_ordered_labelled_edges():
    data = pd.DataFrame(
        {
            "y": [0.1, 0.3, -0.2, 0.4, 0.0, 0.2],
            "region": ["north", "center", "south"] * 2,
        }
    )
    adjacency = {
        "north": ("center",),
        "center": ("north", "south"),
        "south": ("center",),
    }
    levels = ("north", "center", "south")
    edges = (("north", "center"), ("center", "south"))
    named = GAMLSS.from_formula(
        Normal(),
        {
            "mu": f"y ~ mrf(region, nb={adjacency!r}, lambda_=1)",
            "sigma": "~ 1",
        },
        data,
    )
    paired = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                f"y ~ mrf(region, edges={edges!r}, levels={levels!r}, "
                "lambda_=1)"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    named_data = named.prepare_formula_data(data)
    paired_data = paired.prepare_formula_data(data)
    named_term = named.smooth_terms["mu"]["region"]
    paired_term = paired.smooth_terms["mu"]["region"]

    assert named_term.region_labels == levels
    assert paired_term.region_labels == levels
    torch.testing.assert_close(
        named_term.design(named_data.smooth_covariates["mu"]["region"]),
        paired_term.design(paired_data.smooth_covariates["mu"]["region"]),
    )
    torch.testing.assert_close(
        named_term.penalty_matrices()[0],
        paired_term.penalty_matrices()[0],
    )


def test_formula_mrf_accepts_a_labelled_supplied_penalty():
    data = pd.read_csv(REFERENCE_DIR / "mrf_fit_data.csv")
    data["region_label"] = data["region"].map(dict(enumerate(MRF_LABELS)))
    penalty = _mrf_graph_penalty_tuple()
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                f"y ~ mrf(region_label, penalty={penalty!r}, "
                f"levels={MRF_LABELS!r}, k=6, lambda_=1.9, name='area')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data)
    term = model.smooth_terms["mu"]["area"]
    numeric = MarkovRandomField.from_data(
        torch.tensor(data["region"].to_numpy(), dtype=torch.float64),
        graph_penalty=penalty,
        basis_size=6,
        smoothing_parameter=1.9,
    )

    assert model.formula_column_names["mu"] == ("Intercept",)
    assert term.region_labels == MRF_LABELS
    assert term.uses_supplied_penalty
    torch.testing.assert_close(
        term.design(prepared.smooth_covariates["mu"]["area"]),
        numeric.design(
            torch.tensor(data["region"].to_numpy(), dtype=torch.float64)
        ),
    )

    with pytest.raises(ValueError, match="levels are required"):
        GAMLSS.from_formula(
            Normal(),
            {
                "mu": f"y ~ mrf(region_label, penalty={penalty!r})",
                "sigma": "~ 1",
            },
            data,
        )


def test_formula_mrf_builds_adjacency_from_polygons():
    data = pd.read_csv(REFERENCE_DIR / "mrf_polygon_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": (
                f"y ~ mrf(region, polygons={MRF_POLYGONS!r}, k=4, "
                "lambda_=1.3, name='area')"
            ),
            "sigma": "~ 1",
        },
        data,
    )
    prepared = model.prepare_formula_data(data)
    term = model.smooth_terms["mu"]["area"]

    assert model.formula_column_names["mu"] == ("Intercept",)
    assert term.uses_polygon_adjacency
    assert term.region_labels == tuple(MRF_POLYGONS)
    assert term.edges == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
    assert term.design(prepared.smooth_covariates["mu"]["area"]).shape == (
        len(data),
        3,
    )

    with pytest.raises(ValueError, match="exactly one"):
        GAMLSS.from_formula(
            Normal(),
            {
                "mu": (
                    f"y ~ mrf(region, polygons={MRF_POLYGONS!r}, "
                    "adjacency={'area_0': ('area_1',), "
                    "'area_1': ('area_0',)})"
                ),
                "sigma": "~ 1",
            },
            data,
        )


def test_formula_rejects_missing_values():
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5],
            "x": [-1.0, -0.3, float("nan"), 1.0],
        }
    )

    with pytest.raises(ValueError, match="materialize formula"):
        GAMLSS.from_formula(
            Normal(),
            {"mu": "y ~ x", "sigma": "~ 1"},
            data,
        )


@pytest.mark.parametrize(
    ("formulas", "match"),
    [
        ({"mu": "y ~ x"}, "Formulas do not match"),
        ({"mu": "~ x", "sigma": "~ 1"}, "must contain a response"),
        (
            {"mu": "y ~ x", "sigma": "y ~ 1"},
            "Only the first",
        ),
        (
            {"mu": "y ~ pb(x):x", "sigma": "~ 1"},
            "interactions",
        ),
        (
            {"mu": "y ~ pb(x, unsupported=1)", "sigma": "~ 1"},
            "unsupported",
        ),
    ],
)
def test_invalid_formula_configurations_are_rejected(formulas, match):
    data = pd.DataFrame(
        {
            "y": [0.2, 0.7, 1.1, 1.5],
            "x": [-1.0, -0.3, 0.4, 1.0],
        }
    )

    with pytest.raises(ValueError, match=match):
        GAMLSS.from_formula(Normal(), formulas, data)


def test_formula_methods_require_a_formula_model():
    model = GAMLSS(Normal(), {"mu": 1, "sigma": 1}, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="from_formula"):
        model.predict_data({"x": [1.0]})
    with pytest.raises(RuntimeError, match="from_formula"):
        model.fit_cg_data({"y": [1.0]})
