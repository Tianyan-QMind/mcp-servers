from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BackendType(StrEnum):
    ALL = "all"
    QUANTUM_COMPUTER = "quantum_computer"
    SIMULATOR = "simulator"


class UpstreamGate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    qcis: str | None = None


def _empty_gate_list() -> list[UpstreamGate]:
    return []


class UpstreamBackend(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = None
    code: str | None = None
    name: str | None = None
    name_en: str | None = Field(default=None, alias="nameEn")
    bit_width: Any = Field(default=None, alias="bitWidth")
    labels: str | None = None
    label_name: str | None = Field(default=None, alias="labelName")
    label_name_en: str | None = Field(default=None, alias="labelNameEn")
    status: int | None = None
    qubit_num: Any = Field(default=None, alias="qubitNum")
    coupler_num: Any = Field(default=None, alias="couplerNum")
    readout_num: Any = Field(default=None, alias="readOutNum")
    disabled_qubits: str | None = Field(default=None, alias="disabledQubits")
    disabled_couplers: str | None = Field(default=None, alias="disabledCouplers")
    processor_name: str | None = Field(default=None, alias="processorCn")
    processor_name_en: str | None = Field(default=None, alias="processorEn")
    running_task_count: Any = Field(default=None, alias="taskNum")
    calibration_time: str | None = Field(default=None, alias="readTime")
    single_qubit_gate_error: Any = Field(default=None, alias="oneqGateError")
    two_qubit_gate_error: Any = Field(default=None, alias="czError")
    readout_error: Any = Field(default=None, alias="readoutError")
    fsim_error: Any = Field(default=None, alias="fsimError")
    t1: Any = Field(default=None, alias="tone")
    t2: Any = Field(default=None, alias="ttwo")
    base_gates: list[UpstreamGate] = Field(default_factory=_empty_gate_list, alias="baseGate")
    derivative_gates: list[UpstreamGate] = Field(
        default_factory=_empty_gate_list, alias="derivativeGate"
    )


class BackendSnapshot(BaseModel):
    fetched_at: datetime
    backends: list[UpstreamBackend]


class CoherenceTime(BaseModel):
    t1: float | None
    t2: float | None


class ErrorRates(BaseModel):
    readout: float | None
    single_qubit_gate: float | None
    two_qubit_gate: float | None
    fsim: float | None


class QuantumComputerSpecifications(BaseModel):
    processor_name: str | None
    processor_name_en: str | None
    qubit_count: int | None
    bit_width: int | None
    coupler_count: int | None
    readout_count: int | None
    running_task_count: int | None
    calibration_time: str | None
    disabled_qubits: list[str]
    disabled_couplers: list[str]
    coherence_time_us: CoherenceTime
    error_rates_percent: ErrorRates
    supported_gates: list[str]


class SimulatorSpecifications(BaseModel):
    simulation_type: str
    max_qubit_count: int | None
    supported_gates: list[str]


class BackendDetail(BaseModel):
    fetched_at: datetime
    id: str
    code: str
    name: str | None
    name_en: str | None
    resource_type: BackendType
    subtype: str
    operational_status: str
    available: bool
    raw_status: int
    raw_label: str
    specifications: QuantumComputerSpecifications | SimulatorSpecifications


class BackendSummary(BaseModel):
    id: str
    code: str
    name: str | None
    name_en: str | None
    resource_type: BackendType
    subtype: str
    operational_status: str
    available: bool
    raw_status: int
    raw_label: str
    qubit_count: int | None
    max_qubit_count: int | None
    running_task_count: int | None


class BackendListResult(BaseModel):
    fetched_at: datetime
    backend_type_filter: BackendType
    total: int
    backends: list[BackendSummary]
