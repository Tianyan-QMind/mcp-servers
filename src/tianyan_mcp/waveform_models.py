from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_task_id(value: Any) -> str:
    """Normalize snowflake task ids (JSON number or numeric string) to a string."""
    if type(value) is int and not isinstance(value, bool):
        if value <= 0:
            raise ValueError("waveform task id integer must be positive")
        return str(value)
    if type(value) is not str or not value.strip():
        raise ValueError("waveform task id must be a non-empty string or positive integer")
    return value.strip()


class UpstreamWaveformSubmission(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    circuit: str = Field(min_length=1, max_length=20_000)
    qc_code: str = Field(alias="qcCode", min_length=1, max_length=64)
    circuit_name: str | None = Field(
        default=None, alias="circuitName", min_length=1, max_length=100
    )
    is_verify: Literal["true", "false"] = Field(default="true", alias="isVerify")


class UpstreamWaveformTask(BaseModel):
    """Response data of POST generateWaveformDiagram (PulseControllerData)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    id: str
    circuit_name: str = Field(alias="circuitName")
    circuit_content: str = Field(alias="circuitContent")
    status: Literal[0, 1, 2]
    create_time: str | None = Field(default=None, alias="createTime")
    visible_url: str | None = Field(default=None, alias="visibleUrl")
    measure_qubit_list: str | None = Field(default=None, alias="measureQubitList")

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        return _coerce_task_id(value)


class UpstreamWaveformDiagram(BaseModel):
    """Response data of GET getWaveformDiagram.

    Live payloads only carry ``id``/``circuit_name``/``visibleUrl`` (the terminal
    failure is normally signalled by the top-level ``code=500`` envelope), but the
    documented status field 0/1/2 is accepted defensively whenever the upstream
    includes it.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    id: str
    circuit_name: str = Field(alias="circuit_name")
    visible_url: str = Field(alias="visibleUrl")
    status: Literal[0, 1, 2] | None = None

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        return _coerce_task_id(value)


class WaveformGenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    qc_code: str
    circuit_name: str
    circuit_content: str
    status: Literal["generating", "completed", "failed"]
    upstream_status: Literal[0, 1, 2]
    create_time: str | None = None
    visible_url: str | None = None
    measure_qubits: tuple[str, ...] = ()


class WaveformDiagramResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    circuit_name: str
    status: Literal["generating", "completed"]
    visible_url: str | None = None
