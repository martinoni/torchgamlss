"""Run declarative end-to-end parity cases for R and TorchGAMLSS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ParityConfigurationError(ValueError):
    """Raised when a parity manifest is malformed."""


def load_spec(path: Path) -> dict[str, Any]:
    """Load and validate one parity-case manifest."""
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParityConfigurationError(f"cannot read parity manifest {path}") from error
    if not isinstance(spec, dict):
        raise ParityConfigurationError("parity manifest must contain a JSON object")
    if spec.get("schema_version") != 1:
        raise ParityConfigurationError("parity manifest schema_version must be 1")
    if not isinstance(spec.get("case"), str) or not spec["case"].strip():
        raise ParityConfigurationError("parity manifest requires a non-empty case")
    working_directory = spec.get("working_directory", ".")
    if not isinstance(working_directory, str) or not working_directory:
        raise ParityConfigurationError(
            "parity manifest working_directory must be a non-empty string"
        )
    timeout = spec.get("timeout_seconds", 300)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not np.isfinite(timeout)
        or timeout <= 0
    ):
        raise ParityConfigurationError(
            "parity manifest timeout_seconds must be finite and positive"
        )
    commands = spec.get("commands")
    if not isinstance(commands, dict):
        raise ParityConfigurationError("parity manifest requires commands")
    for engine in ("r", "python"):
        command = commands.get(engine)
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ParityConfigurationError(
                f"parity command {engine!r} must be a non-empty string list"
            )
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ParityConfigurationError("parity manifest requires artifacts")
    names: set[str] = set()
    for artifact in artifacts:
        _validate_artifact(artifact, names)
    return spec


def _validate_artifact(artifact: Any, names: set[str]) -> None:
    if not isinstance(artifact, dict):
        raise ParityConfigurationError("each parity artifact must be an object")
    name = artifact.get("name")
    file_name = artifact.get("file")
    if not isinstance(name, str) or not name:
        raise ParityConfigurationError("each parity artifact requires a name")
    if name in names:
        raise ParityConfigurationError(f"duplicate parity artifact name {name!r}")
    names.add(name)
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).is_absolute()
        or ".." in Path(file_name).parts
    ):
        raise ParityConfigurationError(
            f"artifact {name!r} requires a safe relative file path"
        )
    keys = artifact.get("keys", [])
    numeric_keys = artifact.get("numeric_keys", {})
    exact = artifact.get("exact", [])
    numeric = artifact.get("numeric", {})
    if not _is_string_list(keys):
        raise ParityConfigurationError(f"artifact {name!r} keys must be strings")
    if not isinstance(numeric_keys, dict):
        raise ParityConfigurationError(
            f"artifact {name!r} numeric_keys must be an object"
        )
    unknown_numeric_keys = set(numeric_keys) - set(keys)
    if unknown_numeric_keys:
        raise ParityConfigurationError(
            f"artifact {name!r} numeric_keys are not declared keys: "
            f"{sorted(unknown_numeric_keys)}"
        )
    if not _is_string_list(exact):
        raise ParityConfigurationError(f"artifact {name!r} exact must be strings")
    if not isinstance(numeric, dict):
        raise ParityConfigurationError(f"artifact {name!r} numeric must be an object")
    compared = [*keys, *exact, *numeric]
    if len(compared) != len(set(compared)):
        raise ParityConfigurationError(
            f"artifact {name!r} repeats a key or comparison column"
        )
    if not exact and not numeric:
        raise ParityConfigurationError(
            f"artifact {name!r} must compare at least one column"
        )
    for column, tolerance in [*numeric_keys.items(), *numeric.items()]:
        if not isinstance(column, str) or not column:
            raise ParityConfigurationError(
                f"artifact {name!r} has an invalid numeric column"
            )
        _validate_tolerance(name, column, tolerance)


def _validate_tolerance(
    artifact_name: str,
    column: str,
    tolerance: Any,
) -> None:
    if not isinstance(tolerance, dict):
        raise ParityConfigurationError(
            f"numeric tolerance for {artifact_name}.{column} must be an object"
        )
    unknown = set(tolerance) - {"atol", "rtol"}
    if unknown:
        raise ParityConfigurationError(
            f"unknown tolerance options for {artifact_name}.{column}: "
            f"{sorted(unknown)}"
        )
    for tolerance_name in ("atol", "rtol"):
        value = tolerance.get(tolerance_name, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ParityConfigurationError(
                f"{artifact_name}.{column} {tolerance_name} "
                "must be finite and non-negative"
            )


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    )


def compare_results(
    spec: Mapping[str, Any],
    r_results: Path,
    python_results: Path,
) -> dict[str, Any]:
    """Compare all declared result artifacts."""
    artifact_reports = [
        _compare_artifact(artifact, r_results, python_results)
        for artifact in spec["artifacts"]
    ]
    failures = [
        f"{artifact['name']}: {failure}"
        for artifact in artifact_reports
        for failure in artifact["failures"]
    ]
    return {
        "passed": not failures,
        "artifacts": artifact_reports,
        "failures": failures,
    }


def _compare_artifact(
    artifact: Mapping[str, Any],
    r_results: Path,
    python_results: Path,
) -> dict[str, Any]:
    name = artifact["name"]
    file_name = artifact["file"]
    report: dict[str, Any] = {
        "name": name,
        "file": file_name,
        "passed": False,
        "rows": None,
        "columns": [],
        "failures": [],
    }
    try:
        r_table = pd.read_csv(r_results / file_name)
        python_table = pd.read_csv(python_results / file_name)
    except (OSError, pd.errors.ParserError) as error:
        report["failures"].append(f"cannot read results: {error}")
        return report

    keys = list(artifact.get("keys", []))
    numeric_keys = dict(artifact.get("numeric_keys", {}))
    exact = list(artifact.get("exact", []))
    numeric = dict(artifact.get("numeric", {}))
    required = [*keys, *exact, *numeric]
    for engine, table in (("R", r_table), ("Python", python_table)):
        missing = [column for column in required if column not in table.columns]
        if missing:
            report["failures"].append(
                f"{engine} result is missing columns {missing!r}"
            )
    if report["failures"]:
        return report

    if keys:
        for engine, table in (("R", r_table), ("Python", python_table)):
            duplicates = table.duplicated(keys, keep=False)
            if duplicates.any():
                report["failures"].append(
                    f"{engine} result has duplicate keys "
                    f"{_row_key(table.loc[duplicates].iloc[0], keys)!r}"
                )
        if report["failures"]:
            return report
        r_selected = r_table[required]
        python_selected = python_table[required]
    else:
        keys = ["_row"]
        r_selected = r_table[required].assign(_row=np.arange(len(r_table)))
        python_selected = python_table[required].assign(
            _row=np.arange(len(python_table))
        )
        required = [*keys, *exact, *numeric]

    merged, alignment_failure = _align_results(
        r_selected,
        python_selected,
        keys=keys,
        compared_columns=[*exact, *numeric],
        numeric_keys=numeric_keys,
    )
    if alignment_failure is not None:
        report["failures"].append(alignment_failure)
        return report

    assert merged is not None
    report["rows"] = len(merged)
    if numeric_keys:
        report["numeric_keys"] = numeric_keys
    for column in exact:
        column_report = _compare_exact_column(merged, keys, column)
        report["columns"].append(column_report)
        if not column_report["passed"]:
            report["failures"].append(column_report["failure"])
    for column, tolerance in numeric.items():
        column_report = _compare_numeric_column(
            merged,
            keys,
            column,
            atol=float(tolerance.get("atol", 0.0)),
            rtol=float(tolerance.get("rtol", 0.0)),
        )
        report["columns"].append(column_report)
        if not column_report["passed"]:
            report["failures"].append(column_report["failure"])
    report["passed"] = not report["failures"]
    return report


def _align_results(
    r_table: pd.DataFrame,
    python_table: pd.DataFrame,
    *,
    keys: Sequence[str],
    compared_columns: Sequence[str],
    numeric_keys: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame | None, str | None]:
    if not numeric_keys:
        merged = r_table.merge(
            python_table,
            how="outer",
            on=list(keys),
            suffixes=("_r", "_python"),
            indicator=True,
            sort=True,
            validate="one_to_one",
        )
        unmatched = merged["_merge"] != "both"
        if unmatched.any():
            row = merged.loc[unmatched].iloc[0]
            return None, (
                f"result keys differ at {_row_key(row, keys)!r}: {row['_merge']}"
            )
        return merged, None

    if len(r_table) != len(python_table):
        return None, (
            f"result row counts differ: R={len(r_table)}, "
            f"Python={len(python_table)}"
        )
    exact_keys = [key for key in keys if key not in numeric_keys]
    numeric_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in numeric_keys:
        try:
            r_values = pd.to_numeric(r_table[key], errors="raise").to_numpy(
                dtype=float
            )
            python_values = pd.to_numeric(
                python_table[key], errors="raise"
            ).to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            return None, f"numeric key {key!r} is not numeric: {error}"
        if not np.isfinite(r_values).all() or not np.isfinite(python_values).all():
            return None, f"numeric key {key!r} contains non-finite values"
        numeric_values[key] = (r_values, python_values)

    available = np.ones(len(python_table), dtype=bool)
    matches: list[int] = []
    for r_index, r_row in r_table.reset_index(drop=True).iterrows():
        candidates = available.copy()
        for key in exact_keys:
            candidates &= python_table[key].to_numpy() == r_row[key]
        for key, tolerance in numeric_keys.items():
            r_values, python_values = numeric_values[key]
            candidates &= np.isclose(
                python_values,
                r_values[r_index],
                atol=float(tolerance.get("atol", 0.0)),
                rtol=float(tolerance.get("rtol", 0.0)),
            )
        candidate_indices = np.flatnonzero(candidates)
        if len(candidate_indices) != 1:
            reason = "no Python key within tolerance"
            if len(candidate_indices) > 1:
                reason = f"{len(candidate_indices)} ambiguous Python keys"
            return None, (
                f"result keys differ at {_row_key(r_row, keys)!r}: {reason}"
            )
        python_index = int(candidate_indices[0])
        matches.append(python_index)
        available[python_index] = False

    r_aligned = r_table.reset_index(drop=True)
    python_aligned = python_table.iloc[matches].reset_index(drop=True)
    merged = r_aligned[list(keys)].copy()
    for column in compared_columns:
        merged[f"{column}_r"] = r_aligned[column]
        merged[f"{column}_python"] = python_aligned[column]
    return merged, None


def _compare_exact_column(
    merged: pd.DataFrame,
    keys: Sequence[str],
    column: str,
) -> dict[str, Any]:
    r_values = merged[f"{column}_r"].map(_normalize_exact)
    python_values = merged[f"{column}_python"].map(_normalize_exact)
    mismatches = r_values != python_values
    report: dict[str, Any] = {
        "name": column,
        "kind": "exact",
        "passed": not bool(mismatches.any()),
    }
    if mismatches.any():
        index = int(np.flatnonzero(mismatches.to_numpy())[0])
        report["failure"] = (
            f"exact column {column!r} differs at "
            f"{_row_key(merged.iloc[index], keys)!r}: "
            f"R={r_values.iloc[index]!r}, Python={python_values.iloc[index]!r}"
        )
    return report


def _normalize_exact(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    return str(value)


def _compare_numeric_column(
    merged: pd.DataFrame,
    keys: Sequence[str],
    column: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    try:
        r_values = pd.to_numeric(merged[f"{column}_r"], errors="raise").to_numpy(
            dtype=float
        )
        python_values = pd.to_numeric(
            merged[f"{column}_python"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        return {
            "name": column,
            "kind": "numeric",
            "passed": False,
            "atol": atol,
            "rtol": rtol,
            "failure": f"numeric column {column!r} is not numeric: {error}",
        }
    finite = np.isfinite(r_values) & np.isfinite(python_values)
    absolute_error = np.abs(python_values - r_values)
    allowed_error = atol + rtol * np.abs(r_values)
    passed = finite & (absolute_error <= allowed_error)
    nonzero = np.abs(r_values) > 0
    relative_error = np.zeros_like(absolute_error)
    np.divide(
        absolute_error,
        np.abs(r_values),
        out=relative_error,
        where=nonzero,
    )
    report: dict[str, Any] = {
        "name": column,
        "kind": "numeric",
        "passed": bool(passed.all()),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": _finite_max(absolute_error),
        "max_relative_error": _finite_max(relative_error),
    }
    if not passed.all():
        index = int(np.flatnonzero(~passed)[0])
        report["failure"] = (
            f"numeric column {column!r} differs at "
            f"{_row_key(merged.iloc[index], keys)!r}: "
            f"R={r_values[index]:.17g}, Python={python_values[index]:.17g}, "
            f"absolute_error={absolute_error[index]:.17g}, "
            f"allowed={allowed_error[index]:.17g}"
        )
    return report


def _finite_max(values: np.ndarray) -> float | None:
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return None
    return float(finite_values.max(initial=0.0))


def _row_key(row: pd.Series, keys: Sequence[str]) -> dict[str, Any]:
    return {key: _json_value(row[key]) for key in keys}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_case(
    spec_path: Path,
    output_dir: Path,
    *,
    r_reference: Path | None = None,
    python_reference: Path | None = None,
    rscript: str | None = None,
    python_executable: str | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a parity case and write its machine-readable report."""
    spec_path = spec_path.resolve()
    output_dir = output_dir.resolve()
    spec = load_spec(spec_path)
    case_dir = spec_path.parent
    working_dir = (case_dir / spec.get("working_directory", ".")).resolve()
    if not working_dir.is_dir():
        raise ParityConfigurationError(
            f"parity working directory does not exist: {working_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_reports: dict[str, Any] = {}
    result_dirs: dict[str, Path] = {}
    references = {"r": r_reference, "python": python_reference}
    for engine in ("r", "python"):
        reference = references[engine]
        if reference is not None:
            result_dir = reference.resolve()
            if not result_dir.is_dir():
                raise ParityConfigurationError(
                    f"{engine} reference directory does not exist: {result_dir}"
                )
            engine_reports[engine] = {
                "mode": "reference",
                "result_dir": str(result_dir),
                "passed": True,
            }
        else:
            result_dir = output_dir / engine
            result_dir.mkdir(parents=True, exist_ok=True)
            command_context = {
                "case_dir": str(case_dir),
                "output_dir": str(result_dir),
                "python": python_executable or sys.executable,
            }
            if engine == "r":
                command_context["rscript"] = rscript or _discover_rscript()
            command = _render_command(spec["commands"][engine], command_context)
            engine_reports[engine] = _run_command(
                command,
                working_dir=working_dir,
                timeout_seconds=float(spec.get("timeout_seconds", 300)),
            )
            engine_reports[engine]["mode"] = "run"
            engine_reports[engine]["result_dir"] = str(result_dir)
        result_dirs[engine] = result_dir

    if all(engine["passed"] for engine in engine_reports.values()):
        comparison = compare_results(
            spec,
            result_dirs["r"],
            result_dirs["python"],
        )
    else:
        comparison = {
            "passed": False,
            "artifacts": [],
            "failures": [
                f"{engine} command failed: {report.get('error', 'unknown error')}"
                for engine, report in engine_reports.items()
                if not report["passed"]
            ],
        }
    report = {
        "schema_version": 1,
        "case": spec["case"],
        "description": spec.get("description", ""),
        "passed": comparison["passed"],
        "engines": engine_reports,
        "artifacts": comparison["artifacts"],
        "failures": comparison["failures"],
    }
    destination = (report_path or output_dir / "report.json").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(destination)
    return report


def _render_command(
    template: Sequence[str],
    context: Mapping[str, str],
) -> list[str]:
    try:
        return [part.format_map(context) for part in template]
    except KeyError as error:
        raise ParityConfigurationError(
            f"unknown command placeholder {error.args[0]!r}"
        ) from error


def _run_command(
    command: Sequence[str],
    *,
    working_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=working_dir,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(error),
        }
    report = {
        "command": list(command),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        report["error"] = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit code {completed.returncode}"
        )
    return report


def _discover_rscript() -> str:
    configured = os.environ.get("RSCRIPT")
    if configured:
        return configured
    discovered = shutil.which("Rscript")
    if discovered:
        return discovered
    if os.name == "nt":
        candidates = sorted(
            Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"),
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    raise ParityConfigurationError(
        "Rscript was not found; provide --rscript or set RSCRIPT"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one declarative R-to-Python parity case."
    )
    parser.add_argument("spec", type=Path, help="Path to parity.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for engine results and report.json",
    )
    parser.add_argument(
        "--r-reference",
        type=Path,
        help="Use committed R results instead of executing R",
    )
    parser.add_argument(
        "--python-reference",
        type=Path,
        help="Use committed Python results instead of executing Python",
    )
    parser.add_argument("--rscript", help="Path to Rscript")
    parser.add_argument("--python", dest="python_executable", help="Python executable")
    parser.add_argument("--report", type=Path, help="Custom JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_case(
            args.spec,
            args.output_dir,
            r_reference=args.r_reference,
            python_reference=args.python_reference,
            rscript=args.rscript,
            python_executable=args.python_executable,
            report_path=args.report,
        )
    except ParityConfigurationError as error:
        print(f"Parity configuration error: {error}", file=sys.stderr)
        return 2
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"{report['case']}: {status}")
    for artifact in report["artifacts"]:
        artifact_status = "PASSED" if artifact["passed"] else "FAILED"
        print(
            f"  {artifact['name']}: {artifact_status} "
            f"({artifact['rows']} rows)"
        )
    for failure in report["failures"]:
        print(f"  failure: {failure}", file=sys.stderr)
    print(f"Report: {report['report_path']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
