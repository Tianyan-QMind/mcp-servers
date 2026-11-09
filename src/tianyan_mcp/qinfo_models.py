"""Pydantic models for cqlib quantum-information MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClassicalRuntimeValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    kind: Literal["bit", "bool", "uint", "bit_vec"]
    value: bool | int | str | None
    qubits: list[int] | None = None


class ClassicalData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    values: list[ClassicalRuntimeValue]
    vars: list[ClassicalRuntimeValue]


class SimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    simulator: Literal["statevector", "density_matrix", "density_matrix_noise", "stabilizer"]
    num_qubits: int
    probabilities: dict[str, float]
    state_vector: list[dict[str, float]] | None = None
    density_matrix: list[list[dict[str, float]]] | None = None
    readout_probabilities: dict[str, float] | None = None
    classical_data: ClassicalData | None = None
    state_svg: str | None = None
    state_png_base64: str | None = None


class PlotStateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["bloch_vector", "bloch_multivector", "city", "paulivec"]
    image_format: Literal["svg", "png"]
    content: str = Field(min_length=1, max_length=5_000_000)
    encoding: Literal["utf-8", "base64"]
    media_type: Literal["image/svg+xml", "image/png"]


class PauliAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pauli_string: str
    phase: str
    support: list[int]
    matrix: list[list[dict[str, float]]] | None = None
    commutes_with: bool | None = None
    product: str | None = None


class HamiltonianTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pauli: str
    coefficient: float


class HamiltonianResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_qubits: int
    terms: list[HamiltonianTerm]
    num_terms: int
    all_terms_commute: bool
    matrix: list[list[dict[str, float]]] | None = None
    expectation: float | None = None
    variance: float | None = None
    trotter_circuit: str | None = None


class StateMetricsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    value: float | None = None
    # Populated only by partial_transpose, which returns a matrix instead of a scalar.
    density_matrix: list[list[dict[str, float]]] | None = None
