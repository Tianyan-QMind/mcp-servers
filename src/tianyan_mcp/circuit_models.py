"""Pydantic models for cqlib circuit-analysis MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CircuitFormat = Literal["qasm2", "qasm3", "qcis"]


class ConvertCircuitFormatResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_format: CircuitFormat
    target_format: CircuitFormat
    circuit: str = Field(min_length=1, max_length=200_000)


class CircuitOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    qubits: list[int]
    params: list[float | str]
    instruction_type: str


class CircuitAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depth: int
    depth_recursive: int
    dag_depth: int
    total_operations: int
    operation_counts: dict[str, int]
    operation_counts_recursive: dict[str, int]
    layers: list[list[int]]
    operations: list[CircuitOperation]


class ComplexNumber(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    real: float
    imag: float


class CircuitMatrixResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_qubits: int
    matrix: list[list[ComplexNumber]]


class SymbolicComplexNumber(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    real: str
    imag: str


class CircuitSymbolicMatrixResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_qubits: int
    symbols: list[str]
    matrix: list[list[SymbolicComplexNumber]]


class DrawCircuitResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text", "svg", "png"]
    content: str = Field(min_length=1, max_length=5_000_000)
    encoding: Literal["utf-8", "base64"]
    media_type: Literal["text/plain", "image/svg+xml", "image/png"]


class PlotMeasurementResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["histogram", "distribution"]
    image_format: Literal["svg", "png"]
    content: str = Field(min_length=1, max_length=5_000_000)
    encoding: Literal["utf-8", "base64"]
    media_type: Literal["image/svg+xml", "image/png"]


class CircuitEquivalenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: bool
    max_abs_diff: float
    global_phase_ignored: bool
