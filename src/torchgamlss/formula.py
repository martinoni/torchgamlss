"""Wilkinson-formula compilation for tabular GAMLSS inputs."""

from __future__ import annotations

import ast
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from formulaic import ModelSpec, model_matrix
from formulaic.errors import DataMismatchWarning, FormulaicError
from formulaic.model_matrix import ModelMatrices, ModelMatrix
from torch import Tensor


def _identity_transform(value: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return value


_FORMULA_CONTEXT = {
    "offset": _identity_transform,
    "pb": _identity_transform,
}


@dataclass(frozen=True)
class SmoothFormulaSpec:
    """Configuration extracted from one ``pb(...)`` formula factor."""

    name: str
    covariate: str
    options: Mapping[str, Any]


@dataclass(frozen=True)
class FormulaData:
    """Tensor inputs materialized from one tabular dataset."""

    response: Tensor | None
    design_matrices: Mapping[str, Tensor]
    offsets: Mapping[str, Tensor]
    smooth_covariates: Mapping[str, Mapping[str, Tensor]]


class FormulaEncoder:
    """Store fitted formula encodings and reproduce them for new data."""

    def __init__(
        self,
        *,
        parameter_names: tuple[str, ...],
        formulas: Mapping[str, str],
        response_name: str,
        model_specs: Mapping[str, ModelSpec],
        design_columns: Mapping[str, tuple[str, ...]],
        smooth_specs: Mapping[str, tuple[SmoothFormulaSpec, ...]],
        offset_columns: Mapping[str, str],
    ) -> None:
        self.parameter_names = parameter_names
        self.formulas = dict(formulas)
        self.response_name = response_name
        self.model_specs = dict(model_specs)
        self.design_columns = dict(design_columns)
        self.smooth_specs = dict(smooth_specs)
        self.offset_columns = dict(offset_columns)

    @classmethod
    def fit(
        cls,
        parameter_names: tuple[str, ...],
        formulas: Mapping[str, str],
        data: Any,
    ) -> FormulaEncoder:
        expected = set(parameter_names)
        received = set(formulas)
        if expected != received:
            raise ValueError(
                "Formulas do not match family parameters: "
                f"missing={sorted(expected - received)}, "
                f"extra={sorted(received - expected)}"
            )
        if any(
            not isinstance(formula, str) or not formula.strip()
            for formula in formulas.values()
        ):
            raise ValueError("Every parameter formula must be a non-empty string")

        frame = _as_frame(data)
        response_name: str | None = None
        model_specs: dict[str, ModelSpec] = {}
        design_columns: dict[str, tuple[str, ...]] = {}
        smooth_specs: dict[str, tuple[SmoothFormulaSpec, ...]] = {}
        offset_columns: dict[str, str] = {}

        for index, parameter in enumerate(parameter_names):
            materialized = _materialize_formula(formulas[parameter], frame)
            if isinstance(materialized, ModelMatrices):
                if index != 0:
                    raise ValueError(
                        "Only the first family-parameter formula may contain a response"
                    )
                response_name = _response_column(materialized, frame)
                matrix = materialized.rhs
            else:
                if index == 0:
                    raise ValueError(
                        "The first family-parameter formula must contain a response"
                    )
                matrix = materialized

            parameter_smooths, parameter_offset_factors = _special_factors(
                matrix.model_spec
            )
            if len(parameter_offset_factors) > 1:
                raise ValueError(
                    f"formula for {parameter!r} may contain at most one offset()"
                )
            columns = tuple(
                str(column)
                for column in matrix.columns
                if str(column) not in parameter_offset_factors
            )
            if not columns:
                raise ValueError(
                    f"formula for {parameter!r} produced no design-matrix columns"
                )

            model_specs[parameter] = matrix.model_spec
            design_columns[parameter] = columns
            smooth_specs[parameter] = tuple(parameter_smooths)
            if parameter_offset_factors:
                factor_expression = next(iter(parameter_offset_factors))
                offset_columns[parameter] = parameter_offset_factors[factor_expression]

        assert response_name is not None
        return cls(
            parameter_names=parameter_names,
            formulas=formulas,
            response_name=response_name,
            model_specs=model_specs,
            design_columns=design_columns,
            smooth_specs=smooth_specs,
            offset_columns=offset_columns,
        )

    def transform(
        self,
        data: Any,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        include_response: bool,
    ) -> FormulaData:
        frame = _as_frame(data)
        design_matrices: dict[str, Tensor] = {}
        offsets: dict[str, Tensor] = {}
        smooth_covariates: dict[str, dict[str, Tensor]] = {}

        for parameter in self.parameter_names:
            matrix = _materialize_spec(self.model_specs[parameter], frame)
            design_matrices[parameter] = _frame_tensor(
                matrix.loc[:, self.design_columns[parameter]],
                dtype=dtype,
                device=device,
                context=f"design matrix for {parameter!r}",
            )
            if parameter in self.offset_columns:
                column = self.offset_columns[parameter]
                offsets[parameter] = _column_tensor(
                    frame,
                    column,
                    dtype=dtype,
                    device=device,
                    context=f"offset for {parameter!r}",
                )
            smooth_covariates[parameter] = {
                spec.name: _column_tensor(
                    frame,
                    spec.covariate,
                    dtype=dtype,
                    device=device,
                    context=f"smooth covariate {spec.name!r} for {parameter!r}",
                )
                for spec in self.smooth_specs[parameter]
            }

        response = None
        if include_response:
            response = _column_tensor(
                frame,
                self.response_name,
                dtype=dtype,
                device=device,
                context="formula response",
            )
        return FormulaData(
            response=response,
            design_matrices=design_matrices,
            offsets=offsets,
            smooth_covariates=smooth_covariates,
        )

    def tensor(
        self,
        data: Any,
        value: Any,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        context: str,
    ) -> Tensor | None:
        if value is None:
            return None
        if isinstance(value, str):
            return _column_tensor(
                _as_frame(data),
                value,
                dtype=dtype,
                device=device,
                context=context,
            )
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(column, str) for column in value)
        ):
            frame = _as_frame(data)
            missing = set(value).difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{context} columns are missing: {sorted(missing)}"
                )
            return _frame_tensor(
                frame.loc[:, list(value)],
                dtype=dtype,
                device=device,
                context=context,
            )
        if isinstance(value, Tensor):
            return value.to(dtype=dtype, device=device)
        try:
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{context} cannot be converted to a tensor") from error
        return tensor


def _as_frame(data: Any) -> pd.DataFrame:
    try:
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "formula data must be convertible to a pandas DataFrame"
        ) from error
    if len(frame.index) == 0:
        raise ValueError("formula data must contain at least one observation")
    return frame


def _materialize_formula(
    formula: str, frame: pd.DataFrame
) -> ModelMatrix | ModelMatrices:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataMismatchWarning)
            return model_matrix(
                formula,
                frame,
                context=_FORMULA_CONTEXT,
                output="pandas",
                na_action="raise",
            )
    except (
        FormulaicError,
        DataMismatchWarning,
        KeyError,
        SyntaxError,
        ValueError,
    ) as error:
        raise ValueError(
            f"could not materialize formula {formula!r}: {error}"
        ) from error


def _materialize_spec(spec: ModelSpec, frame: pd.DataFrame) -> pd.DataFrame:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataMismatchWarning)
            return spec.get_model_matrix(
                frame,
                context=_FORMULA_CONTEXT,
                output="pandas",
                na_action="raise",
            )
    except (
        FormulaicError,
        DataMismatchWarning,
        KeyError,
        SyntaxError,
        ValueError,
    ) as error:
        raise ValueError(f"could not materialize new formula data: {error}") from error


def _response_column(materialized: ModelMatrices, frame: pd.DataFrame) -> str:
    columns = tuple(str(column) for column in materialized.lhs.columns)
    if len(columns) != 1 or columns[0] not in frame.columns:
        raise ValueError("the formula response must be one untransformed data column")
    return columns[0]


def _special_factors(
    model_spec: ModelSpec,
) -> tuple[list[SmoothFormulaSpec], dict[str, str]]:
    smooths: list[SmoothFormulaSpec] = []
    offsets: dict[str, str] = {}
    smooth_names: set[str] = set()

    for term in model_spec.formula:
        for factor in term.factors:
            expression = factor.expr
            call = _special_call(expression)
            nested_special = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"offset", "pb"}
                for node in ast.walk(ast.parse(expression, mode="eval"))
            )
            if call is None:
                if nested_special:
                    raise ValueError(
                        "pb() and offset() must be standalone additive formula terms"
                    )
                continue
            name, parsed_call = call
            if len(term.factors) != 1:
                raise ValueError(
                    "pb() and offset() cannot be used in formula interactions"
                )
            if name == "offset":
                covariate = _simple_covariate(parsed_call, "offset")
                if parsed_call.keywords or len(parsed_call.args) != 1:
                    raise ValueError("offset() accepts exactly one column name")
                offsets[expression] = covariate
                continue

            covariate = _simple_covariate(parsed_call, "pb")
            options = _pb_options(parsed_call)
            smooth_name = options.pop("name", covariate)
            if (
                not isinstance(smooth_name, str)
                or not smooth_name
                or "." in smooth_name
            ):
                raise ValueError(
                    "pb() name must be a non-empty string containing no dots"
                )
            if smooth_name in smooth_names:
                raise ValueError(f"duplicate pb() smooth name: {smooth_name!r}")
            smooth_names.add(smooth_name)
            smooths.append(
                SmoothFormulaSpec(
                    name=smooth_name,
                    covariate=covariate,
                    options=options,
                )
            )
    return smooths, offsets


def _special_call(expression: str) -> tuple[str, ast.Call] | None:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"invalid formula factor {expression!r}") from error
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"offset", "pb"}
    ):
        return node.func.id, node
    return None


def _simple_covariate(call: ast.Call, function_name: str) -> str:
    if not call.args or not isinstance(call.args[0], ast.Name):
        raise ValueError(f"{function_name}() requires a simple numeric column name")
    return call.args[0].id


def _pb_options(call: ast.Call) -> dict[str, Any]:
    if len(call.args) != 1:
        raise ValueError("pb() accepts one column followed by keyword options")
    aliases = {
        "df": "degrees_of_freedom",
        "inter": "intervals",
        "k": "criterion_penalty",
        "lambda_": "smoothing_parameter",
        "method": "smoothing_method",
    }
    allowed = {
        "criterion_penalty",
        "degree",
        "degrees_of_freedom",
        "initial_smoothing_parameter",
        "intervals",
        "name",
        "penalty_order",
        "smoothing_method",
        "smoothing_parameter",
    }
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("pb() does not accept expanded keyword arguments")
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(f"unsupported pb() option: {keyword.arg!r}")
        if name in options:
            raise ValueError(f"duplicate pb() option: {name!r}")
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"pb() option {name!r} must be a literal") from error
    return options


def _frame_tensor(
    frame: pd.DataFrame,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    context: str,
) -> Tensor:
    try:
        values = frame.to_numpy(dtype=float, copy=True)
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{context} must contain only numeric values") from error
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{context} must be finite")
    return tensor


def _column_tensor(
    frame: pd.DataFrame,
    column: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    context: str,
) -> Tensor:
    if column not in frame.columns:
        raise ValueError(f"{context} column {column!r} is missing")
    try:
        values = frame[column].to_numpy(dtype=float, copy=True)
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{context} column {column!r} must be numeric") from error
    if tensor.ndim != 1:
        raise ValueError(f"{context} column {column!r} must be one-dimensional")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{context} column {column!r} must be finite")
    return tensor
