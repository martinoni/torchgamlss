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
    NegativeBinomial,
    Normal,
    Poisson,
    RSControl,
    TruncatedFamily,
)

REFERENCE_DIR = Path(__file__).parent / "reference"


@pytest.mark.parametrize(
    ("family", "code", "parameters", "link_types"),
    [
        (Normal(), "NO", ("mu", "sigma"), ("IdentityLink", "LogLink")),
        (Gamma(), "GA", ("mu", "sigma"), ("LogLink", "LogLink")),
        (Poisson(), "PO", ("mu",), ("LogLink",)),
        (
            NegativeBinomial(),
            "NBI",
            ("mu", "sigma"),
            ("LogLink", "LogLink"),
        ),
        (Beta(), "BE", ("mu", "sigma"), ("LogitLink", "LogitLink")),
        (
            BCCG(),
            "BCCG",
            ("mu", "sigma", "nu"),
            ("IdentityLink", "LogLink", "IdentityLink"),
        ),
        (
            BCT(),
            "BCT",
            ("mu", "sigma", "nu", "tau"),
            ("IdentityLink", "LogLink", "IdentityLink", "LogLink"),
        ),
        (
            BCPE(),
            "BCPE",
            ("mu", "sigma", "nu", "tau"),
            ("IdentityLink", "LogLink", "IdentityLink", "LogLink"),
        ),
        (
            TF(),
            "TF",
            ("mu", "sigma", "nu"),
            ("IdentityLink", "LogLink", "LogLink"),
        ),
        (
            PE(),
            "PE",
            ("mu", "sigma", "nu"),
            ("IdentityLink", "LogLink", "LogLink"),
        ),
    ],
)
def test_r_to_python_family_mapping(family, code, parameters, link_types):
    assert family.name == code
    assert family.parameter_names == parameters
    assert tuple(type(family.links[name]).__name__ for name in parameters) == link_types


def test_r_to_python_truncated_family_mapping():
    continuous = TruncatedFamily(Normal(), lower=0)
    discrete = TruncatedFamily(Poisson(), lower=0, upper=6)

    assert continuous.name == "NOtr"
    assert continuous.parameter_names == ("mu", "sigma")
    assert discrete.name == "POtr"
    assert discrete.parameter_names == ("mu",)
    assert discrete.is_discrete


def test_r_to_python_formula_workflow_smoke():
    data = pd.read_csv(REFERENCE_DIR / "no_rs_fit_data.csv")
    model = GAMLSS.from_formula(
        Normal(),
        {
            "mu": "y ~ x + offset(mu_offset)",
            "sigma": "~ z + offset(sigma_offset)",
        },
        data,
    )

    fit = model.fit_rs_data(
        data,
        weights="weight",
        control=RSControl(
            outer_tolerance=1e-10,
            max_outer_iterations=200,
            inner_tolerance=1e-10,
            max_inner_iterations=200,
        ),
    )

    assert fit.converged
    assert model.formula_column_names == {
        "mu": ("Intercept", "x"),
        "sigma": ("Intercept", "z"),
    }
    coefficients = {
        parameter: dict(
            zip(
                model.formula_column_names[parameter],
                model.coefficients[parameter].detach().cpu().tolist(),
            )
        )
        for parameter in model.family.parameter_names
    }
    assert set(coefficients["mu"]) == {"Intercept", "x"}
    assert set(coefficients["sigma"]) == {"Intercept", "z"}

    parameters = model.predict_data(data)
    predictors = model.predict_data(data, type="link")
    contributions = model.predict_data(data, type="terms")
    for parameter in model.family.parameter_names:
        assert parameters[parameter].shape == (len(data),)
        torch.testing.assert_close(
            contributions[parameter].total,
            predictors[parameter],
        )

    diagnostics = model.diagnostics_data(data, weights="weight")
    assert diagnostics.global_deviance == pytest.approx(fit.global_deviance)
    assert diagnostics.aic == pytest.approx(diagnostics.gaic(2.0))
    assert diagnostics.sbc == diagnostics.bic

    inference = model.inference_data(data, weights="weight")
    assert inference.to_dataframe().index.tolist() == [
        "mu.Intercept",
        "mu.x",
        "sigma.Intercept",
        "sigma.z",
    ]
    assert torch.isfinite(inference.covariance_matrix).all()

    residuals = model.quantile_residuals_data(data)
    assert residuals.shape == (len(data),)
    assert torch.isfinite(residuals).all()
