from __future__ import annotations

from datetime import datetime
from typing import Any

from tianyan_mcp.models import (
    BackendDetail,
    BackendSummary,
    BackendType,
    CoherenceTime,
    ErrorRates,
    QuantumComputerSpecifications,
    SimulatorSpecifications,
    UpstreamBackend,
)

LABELS: dict[str, tuple[BackendType, str]] = {
    "1": (BackendType.QUANTUM_COMPUTER, "超导量子计算机"),
    "3": (BackendType.SIMULATOR, "全振幅状态向量仿真机"),
    "4": (BackendType.SIMULATOR, "稳定子仿真机"),
    "5": (BackendType.SIMULATOR, "带噪声仿真机"),
    "6": (BackendType.SIMULATOR, "单振幅仿真机"),
    "7": (BackendType.SIMULATOR, "张量网络仿真机"),
}
STATUSES = {0: "running", 1: "calibrating", 4: "upgrading"}


def normalise_backend(
    raw: UpstreamBackend,
    fetched_at: datetime,
) -> BackendDetail | None:
    if (
        raw.id is None
        or raw.code is None
        or raw.labels is None
        or raw.status is None
        or raw.labels not in LABELS
    ):
        return None

    resource_type, fallback_subtype = LABELS[raw.labels]
    subtype = raw.label_name or fallback_subtype
    gates = sorted(
        {
            gate.qcis.strip()
            for gate in [*raw.base_gates, *raw.derivative_gates]
            if gate.qcis and gate.qcis.strip()
        }
    )
    if resource_type is BackendType.QUANTUM_COMPUTER:
        specifications = QuantumComputerSpecifications(
            processor_name=raw.processor_name,
            processor_name_en=raw.processor_name_en,
            qubit_count=_to_int(raw.qubit_num),
            bit_width=_to_int(raw.bit_width),
            coupler_count=_to_int(raw.coupler_num),
            readout_count=_to_int(raw.readout_num),
            running_task_count=_to_int(raw.running_task_count),
            calibration_time=raw.calibration_time,
            disabled_qubits=_split_csv(raw.disabled_qubits),
            disabled_couplers=_split_csv(raw.disabled_couplers),
            coherence_time_us=CoherenceTime(
                t1=_to_float(raw.t1),
                t2=_to_float(raw.t2),
            ),
            error_rates_percent=ErrorRates(
                readout=_to_float(raw.readout_error),
                single_qubit_gate=_to_float(raw.single_qubit_gate_error),
                two_qubit_gate=_to_float(raw.two_qubit_gate_error),
                fsim=_to_float(raw.fsim_error),
            ),
            supported_gates=gates,
        )
    else:
        specifications = SimulatorSpecifications(
            simulation_type=subtype,
            max_qubit_count=_to_int(raw.bit_width),
            supported_gates=gates,
        )

    return BackendDetail(
        fetched_at=fetched_at,
        id=raw.id,
        code=raw.code,
        name=raw.name,
        name_en=raw.name_en,
        resource_type=resource_type,
        subtype=subtype,
        operational_status=STATUSES.get(raw.status, "unknown"),
        available=raw.status == 0,
        raw_status=raw.status,
        raw_label=raw.labels,
        specifications=specifications,
    )


def summary_from_detail(detail: BackendDetail) -> BackendSummary:
    specs = detail.specifications
    return BackendSummary(
        id=detail.id,
        code=detail.code,
        name=detail.name,
        name_en=detail.name_en,
        resource_type=detail.resource_type,
        subtype=detail.subtype,
        operational_status=detail.operational_status,
        available=detail.available,
        raw_status=detail.raw_status,
        raw_label=detail.raw_label,
        qubit_count=(
            specs.qubit_count if isinstance(specs, QuantumComputerSpecifications) else None
        ),
        max_qubit_count=(
            specs.max_qubit_count if isinstance(specs, SimulatorSpecifications) else None
        ),
        running_task_count=(
            specs.running_task_count if isinstance(specs, QuantumComputerSpecifications) else None
        ),
    )


def _split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
