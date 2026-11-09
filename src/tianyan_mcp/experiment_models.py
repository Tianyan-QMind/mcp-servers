import json
from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tianyan_mcp.models import BackendType


class UpstreamExperimentSubmission(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    lab_id: Literal[""] = ""
    shots: int = Field(ge=1, le=1000)
    is_verify: Literal[True] = True
    circuit: tuple[str, ...] = Field(min_length=1, max_length=20)
    name: str
    lab_name: str | None = None
    computer_code: str = Field(alias="computerCode")
    quantum_state: Literal[0] | None = Field(default=None, alias="quantumState")
    language: Literal["qcis"] = "qcis"
    priority: Literal[0] = 0


class UpstreamExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    result_status: tuple[tuple[int, ...], ...] = Field(alias="resultStatus")
    probability: dict[str, float]
    experiment_task_id: str = Field(alias="experimentTaskId", min_length=1)

    @field_validator("result_status", mode="before")
    @classmethod
    def validate_result_status(cls, value: Any) -> tuple[tuple[int, ...], ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("resultStatus must be a list of rows")

        rows: list[tuple[int, ...]] = []
        for row in cast(list[object] | tuple[object, ...], value):
            if not isinstance(row, (list, tuple)):
                raise ValueError("resultStatus rows must be lists")
            normalized_row: list[int] = []
            for item in cast(list[object] | tuple[object, ...], row):
                if type(item) is not int:
                    raise ValueError("resultStatus values must be exact ints")
                normalized_row.append(item)
            rows.append(tuple(normalized_row))
        return tuple(rows)

    @field_validator("probability", mode="before")
    @classmethod
    def validate_probability(cls, value: Any) -> dict[str, float]:
        if type(value) is str:
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise ValueError("probability must be a mapping")

        normalized: dict[str, float] = {}
        for key, probability in cast(Mapping[object, object], value).items():
            if type(key) is not str:
                raise ValueError("probability keys must be strings")
            if type(probability) not in {int, float}:
                raise ValueError("probability values must be exact numbers")
            probability_value = float(cast(int | float, probability))
            if not isfinite(probability_value):
                raise ValueError("probability values must be finite")
            normalized[key] = probability_value
        return normalized

    @field_validator("experiment_task_id", mode="before")
    @classmethod
    def validate_experiment_task_id(cls, value: Any) -> str:
        if type(value) is int:
            if value <= 0:
                raise ValueError("experimentTaskId integer must be positive")
            return str(value)
        if type(value) is not str or value == "":
            raise ValueError("experimentTaskId must be a non-empty string or positive integer")
        return value


class ExperimentResultQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    status: Literal["processing", "completed"]
    upstream_status: Literal[0, 1]
    probability: dict[str, float] | None = None
    experiment_task_id: str | None = None
    result_status: tuple[tuple[int, ...], ...] | None = None


class ExperimentSubmissionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    backend_code: str
    resource_type: BackendType
    shots: int
    name: str
    lab_name: str | None = None
    status: str = "submitted"


class ExperimentBatchSubmissionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_ids: tuple[str, ...]
    backend_code: str
    resource_type: BackendType
    shots: int
    name: str
    lab_name: str | None = None
    status: str = "submitted"
