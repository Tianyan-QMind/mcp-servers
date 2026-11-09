from datetime import UTC, datetime
from typing import Any

from tianyan_mcp.mapping import normalise_backend, summary_from_detail
from tianyan_mcp.models import (
    BackendType,
    QuantumComputerSpecifications,
    SimulatorSpecifications,
    UpstreamBackend,
)

FETCHED_AT = datetime(2026, 7, 29, 8, 30, tzinfo=UTC)


def test_maps_running_quantum_computer(upstream_payload: dict[str, Any]) -> None:
    raw_payload = dict(upstream_payload["data"][0])
    raw_payload["imageUrl"] = "https://example.invalid/backend.jpg"
    raw = UpstreamBackend.model_validate(raw_payload)
    detail = normalise_backend(raw, FETCHED_AT)

    assert detail is not None
    serialized = detail.model_dump()
    assert "image_url" not in serialized
    assert "imageUrl" not in serialized
    assert detail.resource_type is BackendType.QUANTUM_COMPUTER
    assert detail.operational_status == "running"
    assert detail.available is True
    assert isinstance(detail.specifications, QuantumComputerSpecifications)
    assert detail.specifications.disabled_qubits == ["Q90", "Q35"]
    assert detail.specifications.coherence_time_us.t1 == 47.02
    assert detail.specifications.supported_gates == ["X", "Y"]


def test_maps_simulator_capacity(upstream_payload: dict[str, Any]) -> None:
    raw = UpstreamBackend.model_validate(upstream_payload["data"][1])
    detail = normalise_backend(raw, FETCHED_AT)

    assert detail is not None
    assert detail.resource_type is BackendType.SIMULATOR
    assert isinstance(detail.specifications, SimulatorSpecifications)
    assert detail.specifications.max_qubit_count == 5000


def test_maps_upgrade_status_as_unavailable(
    upstream_payload: dict[str, Any],
) -> None:
    raw = UpstreamBackend.model_validate(upstream_payload["data"][2])
    detail = normalise_backend(raw, FETCHED_AT)

    assert detail is not None
    assert detail.operational_status == "upgrading"
    assert detail.available is False


def test_rejects_unknown_label() -> None:
    raw = UpstreamBackend(id="x", code="x", labels="99", status=0)
    assert normalise_backend(raw, FETCHED_AT) is None


def test_summary_keeps_physical_and_simulator_capacity_separate(
    upstream_payload: dict[str, Any],
) -> None:
    physical = normalise_backend(
        UpstreamBackend.model_validate(upstream_payload["data"][0]), FETCHED_AT
    )
    simulator = normalise_backend(
        UpstreamBackend.model_validate(upstream_payload["data"][1]), FETCHED_AT
    )

    assert physical is not None and simulator is not None
    assert summary_from_detail(physical).qubit_count == 105
    assert summary_from_detail(physical).max_qubit_count is None
    assert summary_from_detail(simulator).qubit_count is None
    assert summary_from_detail(simulator).max_qubit_count == 5000
