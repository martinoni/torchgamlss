import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from tools.run_parity import (
    ParityConfigurationError,
    compare_results,
    load_spec,
    run_case,
)

ROOT = Path(__file__).parents[1]
EXAMPLE_DIR = ROOT / "examples" / "normal_location_scale"


def test_manifest_rejects_numeric_tolerance_for_an_undeclared_key(tmp_path):
    manifest = {
        "schema_version": 1,
        "case": "invalid",
        "commands": {"r": ["Rscript"], "python": ["python"]},
        "artifacts": [
            {
                "name": "values",
                "file": "values.csv",
                "keys": ["row"],
                "numeric_keys": {"probability": {"atol": 1e-12}},
                "numeric": {"value": {"atol": 1e-8}},
            }
        ],
    }
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ParityConfigurationError,
        match="numeric_keys are not declared keys",
    ):
        load_spec(path)


def test_numeric_keys_align_with_tolerance_and_mismatches_report_first_key(
    tmp_path,
):
    r_results = tmp_path / "r"
    python_results = tmp_path / "python"
    r_results.mkdir()
    python_results.mkdir()
    pd.DataFrame(
        {
            "observation": [0, 0],
            "probability": [0.03, 0.97],
            "value": [1.0, 2.0],
        }
    ).to_csv(r_results / "values.csv", index=False)
    pd.DataFrame(
        {
            "observation": [0, 0],
            "probability": [0.030000000000000002, 0.9700000000000001],
            "value": [1.0, 2.5],
        }
    ).to_csv(python_results / "values.csv", index=False)
    spec = {
        "artifacts": [
            {
                "name": "values",
                "file": "values.csv",
                "keys": ["observation", "probability"],
                "numeric_keys": {
                    "probability": {"atol": 1e-14, "rtol": 1e-14}
                },
                "numeric": {"value": {"atol": 1e-12, "rtol": 0.0}},
            }
        ]
    }

    report = compare_results(spec, r_results, python_results)

    assert not report["passed"]
    assert report["artifacts"][0]["rows"] == 2
    assert "numeric column 'value' differs at" in report["failures"][0]
    assert "'probability': 0.97" in report["failures"][0]
    assert "Python=2.5" in report["failures"][0]


def test_normal_location_scale_example_matches_committed_r_reference(tmp_path):
    report = run_case(
        EXAMPLE_DIR / "parity.json",
        tmp_path / "results",
        r_reference=EXAMPLE_DIR / "reference" / "r",
        python_executable=sys.executable,
    )

    assert report["passed"]
    assert report["engines"]["r"]["mode"] == "reference"
    assert report["engines"]["python"]["mode"] == "run"
    assert {
        artifact["name"]: artifact["rows"]
        for artifact in report["artifacts"]
    } == {
        "fit": 1,
        "coefficients": 4,
        "fitted parameters": 12,
        "response quantiles": 36,
        "quantile residuals": 12,
    }
    assert (tmp_path / "results" / "report.json").is_file()
    assert (
        tmp_path
        / "results"
        / "python"
        / "location_scale_fit.png"
    ).is_file()


def test_non_finite_mismatch_still_produces_valid_json(tmp_path):
    r_results = tmp_path / "r"
    python_results = tmp_path / "python"
    r_results.mkdir()
    python_results.mkdir()
    pd.DataFrame({"value": [float("nan")]}).to_csv(
        r_results / "values.csv",
        index=False,
    )
    pd.DataFrame({"value": [1.0]}).to_csv(
        python_results / "values.csv",
        index=False,
    )
    spec = {
        "artifacts": [
            {
                "name": "values",
                "file": "values.csv",
                "numeric": {"value": {"atol": 1e-12, "rtol": 0.0}},
            }
        ]
    }

    report = compare_results(spec, r_results, python_results)

    assert not report["passed"]
    assert report["artifacts"][0]["columns"][0]["max_absolute_error"] is None
    json.dumps(report, allow_nan=False)
