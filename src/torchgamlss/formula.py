"""Wilkinson-formula compilation for tabular GAMLSS inputs."""

from __future__ import annotations

import ast
import warnings
from collections.abc import Mapping, Sequence
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


def _mrf_region_labels(
    *,
    adjacency: Any = None,
    edges: Any = None,
    levels: Any = None,
    polygons: Any = None,
) -> tuple[object, ...] | None:
    """Resolve the categorical region order declared by an MRF formula."""
    if levels is not None:
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            raise ValueError("mrf() levels must be a sequence of region labels")
        normalized = tuple(levels)
        if len(normalized) < 2:
            raise ValueError("mrf() levels must contain at least two labels")
        return normalized
    if adjacency is not None:
        if not isinstance(adjacency, Mapping):
            raise ValueError("mrf() adjacency must be a mapping of named neighbors")
        normalized = tuple(adjacency)
        if len(normalized) < 2:
            raise ValueError("mrf() adjacency must name at least two regions")
        return normalized
    if polygons is not None:
        if not isinstance(polygons, Mapping):
            raise ValueError("mrf() polygons must be a mapping of region shapes")
        normalized = tuple(polygons)
        if len(normalized) < 2:
            raise ValueError("mrf() polygons must name at least two regions")
        return normalized
    if edges is None:
        return None
    if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
        raise ValueError("mrf() edges must contain two-region pairs")
    endpoints: list[object] = []
    for index, edge in enumerate(edges):
        if (
            isinstance(edge, (str, bytes))
            or not isinstance(edge, Sequence)
            or len(edge) != 2
        ):
            raise ValueError(f"mrf() edge {index} must contain two regions")
        endpoints.extend(edge)
    if all(
        isinstance(endpoint, int) and not isinstance(endpoint, bool)
        for endpoint in endpoints
    ):
        return None
    normalized = tuple(dict.fromkeys(endpoints))
    if len(normalized) < 2:
        raise ValueError("mrf() labelled edges must contain at least two regions")
    return normalized


def _mrf_transform(
    value: Any,
    *args: Any,
    adjacency: Any = None,
    nb: Any = None,
    edges: Any = None,
    graph_penalty: Any = None,
    levels: Any = None,
    penalty: Any = None,
    polygons: Any = None,
    **kwargs: Any,
) -> Any:
    """Encode named MRF regions before Formulaic expands categorical data."""
    del args, kwargs
    if adjacency is not None and nb is not None:
        raise ValueError("mrf() accepts either adjacency or nb, not both")
    declared_adjacency = adjacency if adjacency is not None else nb
    spatial_source_count = sum(
        (
            declared_adjacency is not None,
            edges is not None,
            polygons is not None,
            graph_penalty is not None or penalty is not None,
        )
    )
    if spatial_source_count > 1:
        raise ValueError(
            "mrf() accepts exactly one of edges, adjacency, polygons, or penalty"
        )
    region_labels = _mrf_region_labels(
        adjacency=declared_adjacency,
        edges=edges,
        levels=levels,
        polygons=polygons,
    )
    if region_labels is None:
        if graph_penalty is not None or penalty is not None:
            try:
                pd.to_numeric(pd.Series(value))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "mrf() levels are required when a supplied penalty is "
                    "used with categorical regions"
                ) from error
        return value
    label_codes = {label: index for index, label in enumerate(region_labels)}
    series = pd.Series(
        value,
        index=getattr(value, "index", None),
        name=getattr(value, "name", None),
    )
    encoded = series.map(label_codes)
    if encoded.isna().any():
        unknown = tuple(dict.fromkeys(series.loc[encoded.isna()].tolist()))
        raise ValueError(f"mrf() contains unknown region labels: {unknown!r}")
    return encoded.astype(float)


_FORMULA_CONTEXT = {
    "offset": _identity_transform,
    "pb": _identity_transform,
    "pbc": _identity_transform,
    "pbz": _identity_transform,
    "ad": _identity_transform,
    "mrf": _mrf_transform,
    "tp": _identity_transform,
    "te": _identity_transform,
    "ti": _identity_transform,
}


@dataclass(frozen=True)
class SmoothFormulaSpec:
    """Configuration extracted from one scalar smooth formula factor."""

    name: str
    covariate: str
    kind: str
    options: Mapping[str, Any]
    factor_expression: str


@dataclass(frozen=True)
class TensorFormulaSpec:
    """Configuration extracted from a multivariate smooth formula factor."""

    name: str
    covariates: tuple[str, ...]
    kind: str
    interaction: bool
    options: Mapping[str, Any]
    factor_expression: str


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
        tensor_specs: Mapping[str, tuple[TensorFormulaSpec, ...]],
        offset_columns: Mapping[str, str],
    ) -> None:
        self.parameter_names = parameter_names
        self.formulas = dict(formulas)
        self.response_name = response_name
        self.model_specs = dict(model_specs)
        self.design_columns = dict(design_columns)
        self.smooth_specs = dict(smooth_specs)
        self.tensor_specs = dict(tensor_specs)
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
        tensor_specs: dict[str, tuple[TensorFormulaSpec, ...]] = {}
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

            (
                parameter_smooths,
                parameter_tensors,
                parameter_offset_factors,
            ) = _special_factors(matrix.model_spec)
            if len(parameter_offset_factors) > 1:
                raise ValueError(
                    f"formula for {parameter!r} may contain at most one offset()"
                )
            tensor_factors = {
                spec.factor_expression for spec in parameter_tensors
            }
            standalone_smooth_factors = {
                spec.factor_expression
                for spec in parameter_smooths
                if spec.kind in {"pbc", "pbz", "ad", "mrf"}
            }
            columns = tuple(
                str(column)
                for column in matrix.columns
                if str(column) not in parameter_offset_factors
                and str(column) not in tensor_factors
                and str(column) not in standalone_smooth_factors
            )
            if not columns:
                raise ValueError(
                    f"formula for {parameter!r} produced no design-matrix columns"
                )

            model_specs[parameter] = matrix.model_spec
            design_columns[parameter] = columns
            smooth_specs[parameter] = tuple(parameter_smooths)
            tensor_specs[parameter] = tuple(parameter_tensors)
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
            tensor_specs=tensor_specs,
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
                spec.name: (
                    _mrf_column_tensor(
                        frame,
                        spec.covariate,
                        tuple(spec.options["region_labels"]),
                        dtype=dtype,
                        device=device,
                        context=(
                            f"smooth covariate {spec.name!r} for {parameter!r}"
                        ),
                    )
                    if spec.kind == "mrf" and "region_labels" in spec.options
                    else _column_tensor(
                        frame,
                        spec.covariate,
                        dtype=dtype,
                        device=device,
                        context=(
                            f"smooth covariate {spec.name!r} for {parameter!r}"
                        ),
                    )
                )
                for spec in self.smooth_specs[parameter]
            }
            smooth_covariates[parameter].update(
                {
                    spec.name: _frame_tensor(
                        frame.loc[:, list(spec.covariates)],
                        dtype=dtype,
                        device=device,
                        context=(
                            f"tensor covariates {spec.name!r} "
                            f"for {parameter!r}"
                        ),
                    )
                    for spec in self.tensor_specs[parameter]
                }
            )

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
) -> tuple[
    list[SmoothFormulaSpec],
    list[TensorFormulaSpec],
    dict[str, str],
]:
    smooths: list[SmoothFormulaSpec] = []
    tensors: list[TensorFormulaSpec] = []
    offsets: dict[str, str] = {}
    smooth_names: set[str] = set()

    for term in model_spec.formula:
        for factor in term.factors:
            expression = factor.expr
            call = _special_call(expression)
            nested_special = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id
                in {
                    "offset",
                    "pb",
                    "pbc",
                    "pbz",
                    "ad",
                    "mrf",
                    "tp",
                    "te",
                    "ti",
                }
                for node in ast.walk(ast.parse(expression, mode="eval"))
            )
            if call is None:
                if nested_special:
                    raise ValueError(
                        "pb(), pbc(), pbz(), ad(), mrf(), tp(), te(), ti(), "
                        "and "
                        "offset() must be "
                        "standalone "
                        "additive formula terms"
                    )
                continue
            name, parsed_call = call
            if len(term.factors) != 1:
                raise ValueError(
                    "pb(), pbc(), pbz(), ad(), mrf(), tp(), te(), ti(), and "
                    "offset() cannot be "
                    "used in formula interactions"
                )
            if name == "offset":
                covariate = _simple_covariate(parsed_call, "offset")
                if parsed_call.keywords or len(parsed_call.args) != 1:
                    raise ValueError("offset() accepts exactly one column name")
                offsets[expression] = covariate
                continue

            if name in {"tp", "te", "ti"} or (
                name == "ad" and len(parsed_call.args) == 2
            ):
                covariates = (
                    _thin_plate_covariates(parsed_call)
                    if name == "tp"
                    else _tensor_covariates(parsed_call, name)
                )
                options = (
                    _thin_plate_options(parsed_call, len(covariates))
                    if name == "tp"
                    else (
                        _adaptive_options(parsed_call, len(covariates))
                        if name == "ad"
                        else _tensor_options(
                            parsed_call,
                            name,
                            len(covariates),
                        )
                    )
                )
                default_name = f"{name}_{'_'.join(covariates)}"
                smooth_name = options.pop("name", default_name)
                _validate_smooth_name(smooth_name, name)
                if smooth_name in smooth_names:
                    raise ValueError(
                        f"duplicate smooth name: {smooth_name!r}"
                    )
                smooth_names.add(smooth_name)
                tensors.append(
                    TensorFormulaSpec(
                        name=smooth_name,
                        covariates=covariates,
                        kind=name,
                        interaction=name == "ti",
                        options=options,
                        factor_expression=expression,
                    )
                )
                continue

            covariate = _simple_covariate(parsed_call, name)
            options = (
                _pbz_options(parsed_call)
                if name == "pbz"
                else (
                    _mrf_options(parsed_call)
                    if name == "mrf"
                    else (
                        _adaptive_options(parsed_call, 1)
                        if name == "ad"
                        else _pb_options(parsed_call, function_name=name)
                    )
                )
            )
            smooth_name = options.pop("name", covariate)
            _validate_smooth_name(smooth_name, name)
            if smooth_name in smooth_names:
                raise ValueError(f"duplicate smooth name: {smooth_name!r}")
            smooth_names.add(smooth_name)
            smooths.append(
                SmoothFormulaSpec(
                    name=smooth_name,
                    covariate=covariate,
                    kind=name,
                    options=options,
                    factor_expression=expression,
                )
            )
    return smooths, tensors, offsets


def _special_call(expression: str) -> tuple[str, ast.Call] | None:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"invalid formula factor {expression!r}") from error
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {"offset", "pb", "pbc", "pbz", "ad", "mrf", "tp", "te", "ti"}
    ):
        return node.func.id, node
    return None


def _simple_covariate(call: ast.Call, function_name: str) -> str:
    if not call.args or not isinstance(call.args[0], ast.Name):
        raise ValueError(f"{function_name}() requires a simple numeric column name")
    return call.args[0].id


def _tensor_covariates(call: ast.Call, function_name: str) -> tuple[str, ...]:
    if len(call.args) < 2 or any(
        not isinstance(argument, ast.Name) for argument in call.args
    ):
        raise ValueError(
            f"{function_name}() requires at least two simple numeric "
            "column names followed by keyword options"
        )
    return tuple(argument.id for argument in call.args)


def _thin_plate_covariates(call: ast.Call) -> tuple[str, ...]:
    if not call.args or any(
        not isinstance(argument, ast.Name) for argument in call.args
    ):
        raise ValueError(
            "tp() requires one or more simple numeric column names followed "
            "by keyword options"
        )
    return tuple(argument.id for argument in call.args)


def _validate_smooth_name(value: Any, function_name: str) -> None:
    if not isinstance(value, str) or not value or "." in value:
        raise ValueError(
            f"{function_name}() name must be a non-empty string containing no dots"
        )


def _pb_options(
    call: ast.Call,
    *,
    function_name: str = "pb",
) -> dict[str, Any]:
    if len(call.args) != 1:
        raise ValueError(
            f"{function_name}() accepts one column followed by keyword options"
        )
    aliases = {
        "df": "degrees_of_freedom",
        "inter": "intervals",
        "k": "criterion_penalty",
        "lambda_": "smoothing_parameter",
        "method": "smoothing_method",
    }
    if function_name == "pbc":
        aliases["sin"] = "sine_penalty"
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
    if function_name == "pbc":
        allowed.add("sine_penalty")
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError(
                f"{function_name}() does not accept expanded keyword arguments"
            )
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(
                f"unsupported {function_name}() option: {keyword.arg!r}"
            )
        if name in options:
            raise ValueError(f"duplicate {function_name}() option: {name!r}")
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{function_name}() option {name!r} must be a literal"
            ) from error
    return options


def _pbz_options(call: ast.Call) -> dict[str, Any]:
    if len(call.args) != 1:
        raise ValueError("pbz() accepts one column followed by keyword options")
    aliases = {
        "inter": "intervals",
        "initial_lambda_": "initial_smoothing_parameters",
        "lambda_": "smoothing_parameters",
        "order": "penalty_order",
        "start": "initial_smoothing_parameters",
    }
    allowed = {
        "degree",
        "initial_smoothing_parameters",
        "intervals",
        "name",
        "penalty_order",
        "smoothing_parameters",
    }
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("pbz() does not accept expanded keyword arguments")
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(f"unsupported pbz() option: {keyword.arg!r}")
        if name in options:
            raise ValueError(f"duplicate pbz() option: {name!r}")
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"pbz() option {name!r} must be a literal") from error

    fixed = "smoothing_parameters" in options
    if fixed and "initial_smoothing_parameters" in options:
        raise ValueError(
            "pbz() accepts either smoothing_parameters or "
            "initial_smoothing_parameters, not both"
        )
    smoothing_parameters = options.pop(
        "initial_smoothing_parameters",
        options.get("smoothing_parameters", (1e-4, 1e-4)),
    )
    if (
        not isinstance(smoothing_parameters, (list, tuple))
        or len(smoothing_parameters) != 2
    ):
        raise ValueError("pbz() smoothing_parameters must contain two values")
    options["smoothing_parameters"] = tuple(smoothing_parameters)
    options["estimate_smoothing"] = not fixed
    return options


def _thin_plate_options(
    call: ast.Call,
    covariate_count: int,
) -> dict[str, Any]:
    if len(call.args) != covariate_count:
        raise ValueError(
            "tp() accepts covariate columns followed by keyword options"
        )
    aliases = {
        "initial_lambda_": "initial_smoothing_parameter",
        "k": "basis_size",
        "lambda_": "smoothing_parameter",
        "m": "penalty_order",
    }
    allowed = {
        "basis_size",
        "initial_smoothing_parameter",
        "max_knots",
        "name",
        "penalty_order",
        "smoothing_parameter",
    }
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("tp() does not accept expanded keyword arguments")
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(f"unsupported tp() option: {keyword.arg!r}")
        if name in options:
            raise ValueError(f"duplicate tp() option: {name!r}")
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"tp() option {name!r} must be a literal") from error
    if (
        "smoothing_parameter" in options
        and "initial_smoothing_parameter" in options
    ):
        raise ValueError(
            "tp() accepts either lambda_ or initial_lambda_, not both"
        )
    return options


def _adaptive_options(
    call: ast.Call,
    covariate_count: int,
) -> dict[str, Any]:
    if len(call.args) != covariate_count or covariate_count not in {1, 2}:
        raise ValueError(
            "ad() accepts one or two columns followed by keyword options"
        )
    aliases = {
        "initial_lambda_": "initial_smoothing_parameters",
        "k": "basis_size",
        "lambda_": "smoothing_parameters",
        "m": "penalty_basis_size",
    }
    allowed = {
        "basis_size",
        "initial_smoothing_parameters",
        "name",
        "penalty_basis_size",
        "smoothing_parameters",
    }
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("ad() does not accept expanded keyword arguments")
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(f"unsupported ad() option: {keyword.arg!r}")
        if name in options:
            raise ValueError(f"duplicate ad() option: {name!r}")
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"ad() option {name!r} must be a literal") from error
    if (
        "smoothing_parameters" in options
        and "initial_smoothing_parameters" in options
    ):
        raise ValueError(
            "ad() accepts either lambda_ or initial_lambda_, not both"
        )
    return options


def _mrf_options(call: ast.Call) -> dict[str, Any]:
    if len(call.args) != 1:
        raise ValueError("mrf() accepts one region column and keyword options")
    aliases = {
        "initial_lambda_": "initial_smoothing_parameter",
        "k": "basis_size",
        "lambda_": "smoothing_parameter",
        "nb": "adjacency",
        "penalty": "graph_penalty",
    }
    allowed = {
        "adjacency",
        "area_count",
        "basis_size",
        "edges",
        "graph_penalty",
        "initial_smoothing_parameter",
        "levels",
        "name",
        "polygons",
        "smoothing_parameter",
    }
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("mrf() does not accept expanded keyword arguments")
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(f"unsupported mrf() option: {keyword.arg!r}")
        if name in options:
            raise ValueError(f"duplicate mrf() option: {name!r}")
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"mrf() option {name!r} must be a literal") from error
    graph_options = {
        "adjacency",
        "edges",
        "graph_penalty",
        "polygons",
    }.intersection(options)
    if not graph_options:
        raise ValueError(
            "mrf() requires edges, adjacency, polygons, or a graph penalty"
        )
    if len(graph_options) > 1:
        raise ValueError(
            "mrf() accepts exactly one of edges, adjacency, polygons, or penalty"
        )
    adjacency = options.pop("adjacency", None)
    edges = options.get("edges")
    polygons = options.get("polygons")
    levels = options.pop("levels", None)
    region_labels = _mrf_region_labels(
        adjacency=adjacency,
        edges=edges,
        levels=levels,
        polygons=polygons,
    )
    if adjacency is not None:
        options["edges"] = adjacency
    if region_labels is not None:
        if len(set(region_labels)) != len(region_labels):
            raise ValueError("mrf() region labels must be unique")
        options["region_labels"] = region_labels
        supplied_area_count = options.get("area_count")
        if supplied_area_count is None:
            options["area_count"] = len(region_labels)
        elif supplied_area_count != len(region_labels):
            raise ValueError(
                "mrf() area_count must equal the number of region labels"
            )
    if (
        "smoothing_parameter" in options
        and "initial_smoothing_parameter" in options
    ):
        raise ValueError(
            "mrf() accepts either lambda_ or initial_lambda_, not both"
        )
    return options


def _tensor_options(
    call: ast.Call,
    function_name: str,
    marginal_count: int,
) -> dict[str, Any]:
    aliases = {
        "inter": "intervals",
        "initial_lambda_": "initial_smoothing_parameters",
        "lambda_": "smoothing_parameters",
    }
    allowed = {
        "center",
        "degree",
        "initial_smoothing_parameters",
        "intervals",
        "name",
        "penalty_order",
        "smoothing_parameters",
    }
    options: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError(
                f"{function_name}() does not accept expanded keyword arguments"
            )
        name = aliases.get(keyword.arg, keyword.arg)
        if name not in allowed:
            raise ValueError(
                f"unsupported {function_name}() option: {keyword.arg!r}"
            )
        if name in options:
            raise ValueError(
                f"duplicate {function_name}() option: {name!r}"
            )
        try:
            options[name] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{function_name}() option {name!r} must be a literal"
            ) from error

    fixed = "smoothing_parameters" in options
    if fixed and "initial_smoothing_parameters" in options:
        raise ValueError(
            f"{function_name}() accepts either smoothing_parameters or "
            "initial_smoothing_parameters, not both"
        )
    smoothing_parameters = options.pop(
        "initial_smoothing_parameters",
        options.get(
            "smoothing_parameters",
            (10.0,) * marginal_count,
        ),
    )
    if (
        not isinstance(smoothing_parameters, (list, tuple))
        or len(smoothing_parameters) != marginal_count
    ):
        raise ValueError(
            f"{function_name}() smoothing_parameters must contain one "
            "value per marginal covariate"
        )
    options["smoothing_parameters"] = tuple(smoothing_parameters)
    options["estimate_smoothing"] = not fixed
    if function_name == "ti" and "center" in options:
        raise ValueError("ti() does not accept the center option")
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


def _mrf_column_tensor(
    frame: pd.DataFrame,
    column: str,
    region_labels: tuple[object, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    context: str,
) -> Tensor:
    """Encode a named MRF column with its fitted graph-level ordering."""
    if column not in frame.columns:
        raise ValueError(f"{context} column {column!r} is missing")
    label_codes = {label: index for index, label in enumerate(region_labels)}
    values = frame[column]
    encoded = values.map(label_codes)
    if encoded.isna().any():
        unknown = tuple(dict.fromkeys(values.loc[encoded.isna()].tolist()))
        raise ValueError(
            f"{context} column {column!r} contains unknown MRF regions: "
            f"{unknown!r}"
        )
    tensor = torch.as_tensor(
        encoded.to_numpy(dtype=float, copy=True),
        dtype=dtype,
        device=device,
    )
    if tensor.ndim != 1:
        raise ValueError(f"{context} column {column!r} must be one-dimensional")
    return tensor
