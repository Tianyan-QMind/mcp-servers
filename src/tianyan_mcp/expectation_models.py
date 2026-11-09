"""Bounded, credential-free inputs and stateless expectation task manifests."""

import re
from dataclasses import dataclass
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Qubit = Annotated[int, Field(strict=True, ge=1, le=999_999_999)]
Index = Annotated[int, Field(strict=True, ge=0, le=7)]
QueryId = Annotated[str, Field(pattern=r"^[0-9]{1,128}$")]


def parse_pauli(value: str) -> dict[int, str]:
    if value == "I":
        return {}
    tokens = value.split()
    if not tokens or len(value) > 1024:
        raise ValueError("invalid Pauli term")
    axes: dict[int, str] = {}
    for token in tokens:
        match = re.fullmatch(r"([XYZ])([1-9][0-9]{0,8})", token)
        if match is None:
            raise ValueError("Pauli terms require X/Y/Z and positive physical qubit IDs")
        qubit = int(match[2])
        if qubit in axes:
            raise ValueError("repeated physical qubit in Pauli term")
        axes[qubit] = match[1]
    return dict(sorted(axes.items()))


class ExpectationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObservableTerm(ExpectationModel):
    pauli: str = Field(min_length=1, max_length=1024)
    coefficient: float = Field(strict=True, ge=-1_000_000, le=1_000_000, allow_inf_nan=False)

    @field_validator("pauli")
    @classmethod
    def canonical_pauli(cls, value: str) -> str:
        axes = parse_pauli(value.strip())
        return " ".join(f"{axis}{qubit}" for qubit, axis in axes.items()) or "I"


def normalize_observable(
    terms: list[ObservableTerm] | tuple[ObservableTerm, ...],
) -> tuple[ObservableTerm, ...]:
    if not 1 <= len(terms) <= 16:
        raise ValueError("observable requires 1..16 terms")
    coefficients: dict[str, float] = {}
    qubits: set[int] = set()
    for term in terms:
        qubits.update(parse_pauli(term.pauli))
        coefficients[term.pauli] = coefficients.get(term.pauli, 0) + term.coefficient
    if len(qubits) > 32:
        raise ValueError("observable exceeds 32 physical qubits")
    normalized = tuple(
        ObservableTerm(pauli=pauli, coefficient=coefficient)
        for pauli, coefficient in coefficients.items()
        if coefficient != 0
    )
    return normalized or (ObservableTerm(pauli="I", coefficient=0),)


class ManifestJob(ExpectationModel):
    circuit_index: Index
    pauli: str = Field(min_length=1, max_length=1024)
    qubits: tuple[Qubit, ...] = Field(min_length=1, max_length=32)
    status: Literal["not_submitted", "submitted", "rejected", "unknown"] = "not_submitted"
    query_id: QueryId | None = None
    error_code: str | None = Field(default=None, pattern=r"^[A-Z_]{1,64}$")

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if (self.status == "submitted") != (self.query_id is not None):
            raise ValueError("only submitted jobs have query IDs")
        if (self.status in {"rejected", "unknown"}) != (self.error_code is not None):
            raise ValueError("failed submissions require an error code")
        return self


class ExpectationManifest(ExpectationModel):
    version: Literal[1] = 1
    circuit_count: Annotated[int, Field(strict=True, ge=1, le=8)]
    observable: tuple[ObservableTerm, ...] = Field(min_length=1, max_length=16)
    jobs: tuple[ManifestJob, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        canonical = normalize_observable(self.observable)
        if canonical != self.observable:
            raise ValueError("manifest observable must be normalized")
        expected = [
            (index, term.pauli, tuple(parse_pauli(term.pauli)))
            for index in range(self.circuit_count)
            for term in canonical
            if term.pauli != "I"
        ]
        actual = [(job.circuit_index, job.pauli, job.qubits) for job in self.jobs]
        if actual != expected:
            raise ValueError("manifest measurement coverage does not match observable")
        ids = [job.query_id for job in self.jobs if job.query_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate query IDs in manifest")
        return self


class PreparedJob(ExpectationModel):
    circuit_index: Index
    pauli: str
    qubits: tuple[Qubit, ...]
    circuit: str = Field(min_length=1, max_length=20_000)


class PreparedMeasurements(ExpectationModel):
    observable: tuple[ObservableTerm, ...]
    qubits: tuple[Qubit, ...]
    jobs: tuple[PreparedJob, ...]


class ExpectationSubmission(ExpectationModel):
    dry_run: bool
    computer_code: str
    shots_per_job: int
    job_count: int
    total_shots: int
    measurements: tuple[PreparedJob, ...]
    manifest: ExpectationManifest


@dataclass(frozen=True)
class Measurement:
    pauli: str
    qubits: tuple[int, ...]
    header: tuple[int, ...]
    probability: dict[str, float]


class CircuitExpectation(ExpectationModel):
    circuit_index: Index
    status: Literal[
        "completed", "processing", "query_error", "invalid_result", "incomplete_submission"
    ]
    value: float | None = None
    query_ids: tuple[str, ...]
    error_code: str | None = None

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("expectation must be finite")
        return value


class ExpectationResults(ExpectationModel):
    results: tuple[CircuitExpectation, ...]
