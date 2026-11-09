from datetime import UTC, datetime
from typing import cast

import pytest

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import TianyanExperimentClient
from tianyan_mcp.experiment_models import UpstreamExperimentSubmission
from tianyan_mcp.experiment_service import ExperimentService
from tianyan_mcp.models import BackendDetail, BackendType, SimulatorSpecifications
from tianyan_mcp.service import BackendService


class FakeBackendService:
    def __init__(self, detail: BackendDetail) -> None:
        self.detail = detail
        self.identifiers: list[str] = []

    async def get_backend(self, identifier: str) -> BackendDetail:
        self.identifiers.append(identifier)
        return self.detail


class FakeExperimentClient:
    def __init__(self) -> None:
        self.requests: list[UpstreamExperimentSubmission] = []
        self.batch_requests: list[UpstreamExperimentSubmission] = []

    async def submit(self, request: UpstreamExperimentSubmission) -> str:
        self.requests.append(request)
        return "query-1"

    async def submit_batch(self, request: UpstreamExperimentSubmission) -> tuple[str, ...]:
        self.batch_requests.append(request)
        return tuple(f"query-{index}" for index, _ in enumerate(request.circuit, start=1))


def make_backend_detail(
    *, resource_type: BackendType = BackendType.SIMULATOR, available: bool = True
) -> BackendDetail:
    return BackendDetail(
        fetched_at=datetime.now(UTC),
        id="backend-id",
        code="backend-code",
        name="Test backend",
        name_en="Test backend",
        resource_type=resource_type,
        subtype="test",
        operational_status="available" if available else "unavailable",
        available=available,
        raw_status=1 if available else 0,
        raw_label="test",
        specifications=SimulatorSpecifications(
            simulation_type="state-vector",
            max_qubit_count=10,
            supported_gates=["H"],
        ),
    )


def make_service(
    detail: BackendDetail,
    *,
    allow_hardware_submission: bool = False,
) -> tuple[ExperimentService, FakeBackendService, FakeExperimentClient]:
    backends = FakeBackendService(detail)
    client = FakeExperimentClient()
    service = ExperimentService(
        cast(BackendService, backends),
        cast(TianyanExperimentClient, client),
        allow_hardware_submission=allow_hardware_submission,
    )
    return service, backends, client


async def assert_rejected(
    service: ExperimentService,
    client: FakeExperimentClient,
    *,
    code: str,
    **kwargs: object,
) -> None:
    with pytest.raises(ServiceError) as exc_info:
        await service.submit(**kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == code
    assert client.requests == []


@pytest.mark.asyncio
async def test_submits_one_simulator_circuit_after_live_check() -> None:
    backends_detail = make_backend_detail()
    service, backends, client = make_service(backends_detail)

    result = await service.submit(
        computer_code="tianyan_sw",
        circuit="H Q1\nM Q1",
        name="bell-test",
        lab_name="mcp-test",
        shots=100,
        allow_hardware=False,
    )

    assert backends.identifiers == ["tianyan_sw"]
    assert result.query_id == "query-1"
    assert result.backend_code == "tianyan_sw"
    request = client.requests[0]
    assert request.circuit == ("H Q1\nM Q1",)
    assert request.lab_id == ""
    assert request.is_verify is True
    assert request.language == "qcis"
    assert request.priority == 0


@pytest.mark.asyncio
async def test_submits_batch_of_circuits_in_one_upstream_request() -> None:
    service, backends, client = make_service(make_backend_detail())

    result = await service.submit_batch(
        computer_code="tianyan_sw",
        circuits=["H Q1\nM Q1", "X Q1\nM Q1"],
        name="batch-test",
        lab_name="mcp-test",
        shots=100,
        allow_hardware=False,
    )

    assert backends.identifiers == ["tianyan_sw"]
    assert client.requests == []
    assert len(client.batch_requests) == 1
    assert client.batch_requests[0].circuit == ("H Q1\nM Q1", "X Q1\nM Q1")
    assert result.query_ids == ("query-1", "query-2")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "circuits",
    [["H Q1"], [f"H Q{index}" for index in range(21)]],
    ids=["one-circuit", "twenty-one-circuits"],
)
async def test_batch_rejects_size_outside_two_to_twenty_before_backend_or_sdk(
    circuits: list[str],
) -> None:
    service, backends, client = make_service(make_backend_detail())

    with pytest.raises(ServiceError, match="INVALID_EXPERIMENT_ARGUMENT"):
        await service.submit_batch(
            computer_code="tianyan_sw",
            circuits=circuits,
            name="batch-test",
            lab_name=None,
            shots=100,
            allow_hardware=False,
        )

    assert backends.identifiers == []
    assert client.requests == []
    assert client.batch_requests == []


@pytest.mark.asyncio
async def test_tianyan_sa_batch_defaults_quantum_state_to_zero() -> None:
    service, _, client = make_service(make_backend_detail())

    await service.submit_batch(
        computer_code="tianyan_sa",
        circuits=["H Q1\nY Q1", "X Q1\nY Q1"],
        name="single-amplitude-batch",
        lab_name=None,
        shots=100,
        allow_hardware=False,
    )

    assert client.batch_requests[0].quantum_state == 0


@pytest.mark.asyncio
async def test_tianyan_sa_submission_defaults_quantum_state_to_zero() -> None:
    service, _, client = make_service(make_backend_detail())

    await service.submit(
        computer_code="tianyan_sa",
        circuit="H Q1\nY Q1",
        name="single-amplitude",
        lab_name="mcp-test",
        shots=100,
        allow_hardware=False,
    )

    request = client.requests[0]
    assert request.computer_code == "tianyan_sa"
    assert request.quantum_state == 0


@pytest.mark.asyncio
async def test_submits_without_lab_name_and_returns_null() -> None:
    service, backends, client = make_service(make_backend_detail())

    result = await service.submit(
        computer_code="tianyan_sw",
        circuit="H Q1\nM Q1",
        name="bell-test",
        lab_name=None,
        shots=100,
        allow_hardware=False,
    )

    assert backends.identifiers == ["tianyan_sw"]
    assert len(client.requests) == 1
    assert client.requests[0].lab_name is None
    assert result.lab_name is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("computer_code", "resource_type", "allow_hardware"),
    [
        ("tianyan_sw", BackendType.SIMULATOR, False),
        ("tianyan_s", BackendType.SIMULATOR, False),
        ("tianyan_tn", BackendType.SIMULATOR, False),
        ("tianyan_swn", BackendType.SIMULATOR, False),
        ("tianyan176", BackendType.QUANTUM_COMPUTER, True),
    ],
)
async def test_non_sa_submissions_omit_quantum_state(
    computer_code: str,
    resource_type: BackendType,
    allow_hardware: bool,
) -> None:
    service, _, client = make_service(
        make_backend_detail(resource_type=resource_type),
        allow_hardware_submission=allow_hardware,
    )

    await service.submit(
        computer_code=computer_code,
        circuit="H Q1\nM Q1",
        name="control",
        lab_name="mcp-test",
        shots=100,
        allow_hardware=allow_hardware,
    )

    assert client.requests[0].quantum_state is None


@pytest.mark.asyncio
async def test_preserves_original_circuit_and_trims_names_before_submission() -> None:
    service, _, client = make_service(make_backend_detail())

    result = await service.submit(
        computer_code="tianyan_sw",
        circuit="  H Q1\nM Q1  ",
        name="  bell-test  ",
        lab_name="  mcp-test  ",
        shots=100,
        allow_hardware=False,
    )

    assert client.requests[0].circuit == ("  H Q1\nM Q1  ",)
    assert result.name == "bell-test"
    assert result.lab_name == "mcp-test"


@pytest.mark.asyncio
@pytest.mark.parametrize("circuit", ["", " \t\n", "H Q1\x00", "H" * 20_001])
async def test_rejects_invalid_circuit_without_submitting(circuit: str) -> None:
    service, _, client = make_service(make_backend_detail())

    await assert_rejected(
        service,
        client,
        code="INVALID_EXPERIMENT_ARGUMENT",
        computer_code="tianyan_sw",
        circuit=circuit,
        name="valid-name",
        lab_name="valid-lab",
        shots=100,
        allow_hardware=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shots", [0, 1001, True])
async def test_rejects_invalid_shots_without_submitting(shots: int | bool) -> None:
    service, _, client = make_service(make_backend_detail())

    await assert_rejected(
        service,
        client,
        code="INVALID_EXPERIMENT_ARGUMENT",
        computer_code="tianyan_sw",
        circuit="H Q1",
        name="valid-name",
        lab_name="valid-lab",
        shots=shots,
        allow_hardware=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "lab_name"),
    [
        ("", "valid-lab"),
        ("invalid/name", "valid-lab"),
        ("valid-name", "invalid\nlab"),
        ("valid-name", "\x00lab"),
    ],
)
async def test_rejects_invalid_names_without_submitting(name: str, lab_name: str) -> None:
    service, _, client = make_service(make_backend_detail())

    await assert_rejected(
        service,
        client,
        code="INVALID_EXPERIMENT_ARGUMENT",
        computer_code="tianyan_sw",
        circuit="H Q1",
        name=name,
        lab_name=lab_name,
        shots=100,
        allow_hardware=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lab_name", ["", " \t\n"])
async def test_rejects_explicit_empty_lab_name_before_backend_or_sdk(lab_name: str) -> None:
    service, backends, client = make_service(make_backend_detail())

    await assert_rejected(
        service,
        client,
        code="INVALID_EXPERIMENT_ARGUMENT",
        computer_code="tianyan_sw",
        circuit="H Q1",
        name="valid-name",
        lab_name=lab_name,
        shots=100,
        allow_hardware=False,
    )

    assert backends.identifiers == []


@pytest.mark.asyncio
async def test_uses_policy_lookup_code_and_canonical_code_for_hardware() -> None:
    detail = make_backend_detail(resource_type=BackendType.QUANTUM_COMPUTER)
    service, backends, client = make_service(detail, allow_hardware_submission=True)

    result = await service.submit(
        computer_code="tianyan-176",
        circuit="H Q1",
        name="hardware-test",
        lab_name="mcp-test",
        shots=100,
        allow_hardware=True,
    )

    assert backends.identifiers == ["tianyan176"]
    assert client.requests[0].computer_code == "tianyan176"
    assert result.backend_code == "tianyan176"
    assert result.resource_type is BackendType.QUANTUM_COMPUTER
    assert result.shots == 100
    assert result.status == "submitted"


@pytest.mark.asyncio
async def test_rejects_unavailable_backend_without_submitting() -> None:
    service, _, client = make_service(make_backend_detail(available=False))

    await assert_rejected(
        service,
        client,
        code="BACKEND_UNAVAILABLE",
        computer_code="tianyan_sw",
        circuit="H Q1",
        name="valid-name",
        lab_name="valid-lab",
        shots=100,
        allow_hardware=False,
    )


@pytest.mark.asyncio
async def test_rejects_backend_policy_type_mismatch_without_submitting() -> None:
    detail = make_backend_detail(resource_type=BackendType.QUANTUM_COMPUTER)
    service, _, client = make_service(detail)

    await assert_rejected(
        service,
        client,
        code="BACKEND_POLICY_MISMATCH",
        computer_code="tianyan_sw",
        circuit="H Q1",
        name="valid-name",
        lab_name="valid-lab",
        shots=100,
        allow_hardware=False,
    )


@pytest.mark.asyncio
async def test_rejects_hardware_when_server_flag_is_disabled_without_submitting() -> None:
    detail = make_backend_detail(resource_type=BackendType.QUANTUM_COMPUTER)
    service, _, client = make_service(detail, allow_hardware_submission=False)

    await assert_rejected(
        service,
        client,
        code="HARDWARE_SUBMISSION_DISABLED",
        computer_code="tianyan176",
        circuit="H Q1",
        name="valid-name",
        lab_name="valid-lab",
        shots=100,
        allow_hardware=True,
    )


@pytest.mark.asyncio
async def test_rejects_hardware_without_explicit_confirmation_without_submitting() -> None:
    detail = make_backend_detail(resource_type=BackendType.QUANTUM_COMPUTER)
    service, _, client = make_service(detail, allow_hardware_submission=True)

    await assert_rejected(
        service,
        client,
        code="HARDWARE_CONFIRMATION_REQUIRED",
        computer_code="tianyan176",
        circuit="H Q1",
        name="valid-name",
        lab_name="valid-lab",
        shots=100,
        allow_hardware=False,
    )


@pytest.mark.asyncio
async def test_rejects_unsupported_backend_without_calling_client() -> None:
    service, backends, client = make_service(make_backend_detail())

    await assert_rejected(
        service,
        client,
        code="UNSUPPORTED_BACKEND",
        computer_code="not-allowed",
        circuit="H Q1",
        name="valid-name",
        lab_name="valid-lab",
        shots=100,
        allow_hardware=False,
    )
    assert backends.identifiers == []
