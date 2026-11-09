import json
from collections.abc import Callable
from typing import Protocol, cast

import httpx
import pytest

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import SUBMIT_URL, TianyanExperimentClient
from tianyan_mcp.experiment_models import UpstreamExperimentSubmission


class MutableCircuit(Protocol):
    def append(self, value: str) -> None: ...


class FakeTokenProvider:
    def __init__(self) -> None:
        self.invalidated = False

    async def get_access_token(self) -> str:
        return "fake-token"

    def invalidate(self) -> None:
        self.invalidated = True


def request_model() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1\nM Q1",),
        name="bell-test",
        lab_name="mcp-test",
        computerCode="tianyan_sw",
    )


def batch_request_model() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1\nM Q1", "X Q1\nM Q1"),
        name="batch-test",
        lab_name="mcp-test",
        computerCode="tianyan_sw",
    )


def tianyan_sa_request_model() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1\nY Q1",),
        name="single-amplitude",
        lab_name="mcp-test",
        computerCode="tianyan_sa",
        quantumState=0,
    )


def request_model_without_lab_name() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1\nM Q1",),
        name="bell-test",
        computerCode="tianyan_sw",
    )


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], provider: FakeTokenProvider | None = None
) -> TianyanExperimentClient:
    return TianyanExperimentClient(
        provider or FakeTokenProvider(),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_submits_exactly_once_with_bearer_token_and_fixed_fields() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url == SUBMIT_URL
        assert request.headers["Authorization"] == "Bearer fake-token"
        body = json.loads(request.content)
        assert body == {
            "lab_id": "",
            "shots": 100,
            "is_verify": True,
            "circuit": ["H Q1\nM Q1"],
            "name": "bell-test",
            "lab_name": "mcp-test",
            "computerCode": "tianyan_sw",
            "language": "qcis",
            "priority": 0,
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "message": None,
                "data": {"query_ids": ["query-1"]},
                "success": True,
            },
        )

    client = client_for(handler)

    assert await client.submit(request_model()) == "query-1"
    assert calls == 1


@pytest.mark.asyncio
async def test_submits_batch_once_and_returns_query_ids_in_upstream_order() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url == SUBMIT_URL
        body = json.loads(request.content)
        assert body["circuit"] == ["H Q1\nM Q1", "X Q1\nM Q1"]
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["query-1", "query-2"]}},
        )

    assert await client_for(handler).submit_batch(batch_request_model()) == ("query-1", "query-2")
    assert calls == 1


@pytest.mark.asyncio
async def test_batch_rejects_mismatched_query_id_count_without_retry() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["query-1"]}},
        )

    with pytest.raises(ServiceError, match="SUBMISSION_OUTCOME_UNKNOWN"):
        await client_for(handler).submit_batch(batch_request_model())
    assert calls == 1


@pytest.mark.asyncio
async def test_serializes_tianyan_sa_quantum_state_as_exact_integer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["computerCode"] == "tianyan_sa"
        assert body["quantumState"] == 0
        assert type(body["quantumState"]) is int
        assert "noise" not in body
        assert "nm" not in body
        return httpx.Response(
            200,
            json={
                "code": 0,
                "message": None,
                "data": {"query_ids": ["query-1"]},
                "success": True,
            },
        )

    client = client_for(handler)

    assert await client.submit(tianyan_sa_request_model()) == "query-1"


@pytest.mark.asyncio
async def test_omits_absent_lab_name_from_submission_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "lab_name" not in body
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["query-1"]}},
        )

    assert await client_for(handler).submit(request_model_without_lab_name()) == "query-1"


def raising(error: httpx.RequestError) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_: httpx.Request) -> httpx.Response:
        raise error

    return handler


def responding(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_: httpx.Request) -> httpx.Response:
        return response

    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "expected_code"),
    [
        (raising(httpx.ReadTimeout("timeout")), "SUBMISSION_OUTCOME_UNKNOWN"),
        (raising(httpx.ConnectError("connection error")), "SUBMISSION_OUTCOME_UNKNOWN"),
        (responding(httpx.Response(500)), "SUBMISSION_OUTCOME_UNKNOWN"),
        (responding(httpx.Response(200, content=b"not-json")), "SUBMISSION_OUTCOME_UNKNOWN"),
        (
            responding(httpx.Response(302, headers={"Location": str(SUBMIT_URL)})),
            "SUBMISSION_REJECTED",
        ),
        (responding(httpx.Response(200, json=["not-a-payload"])), "SUBMISSION_OUTCOME_UNKNOWN"),
        (
            responding(
                httpx.Response(200, json={"code": 0, "success": True, "data": {"query_ids": []}})
            ),
            "SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(
                httpx.Response(
                    200, json={"code": 0, "success": True, "data": {"query_ids": ["a", "b"]}}
                )
            ),
            "SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(
                httpx.Response(200, json={"code": 0, "success": True, "data": {"query_ids": [" "]}})
            ),
            "SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(
                httpx.Response(200, json={"code": 0, "success": True, "data": {"query_ids": [123]}})
            ),
            "SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (responding(httpx.Response(400)), "SUBMISSION_REJECTED"),
        (
            responding(
                httpx.Response(
                    200, json={"code": 0, "success": False, "data": {"query_ids": ["query-1"]}}
                )
            ),
            "SUBMISSION_REJECTED",
        ),
        (
            responding(
                httpx.Response(200, json={"code": 1, "success": False, "data": {"query_ids": []}})
            ),
            "SUBMISSION_REJECTED",
        ),
    ],
    ids=[
        "timeout",
        "connection-error",
        "server-error",
        "non-json-success",
        "redirect-not-followed",
        "non-object-payload",
        "missing-query-id",
        "multiple-query-ids",
        "blank-query-id",
        "non-string-query-id",
        "bad-request",
        "rejected-success-flag",
        "rejected-code",
    ],
)
async def test_submission_failure_uses_one_call_only(
    handler: Callable[[httpx.Request], httpx.Response], expected_code: str
) -> None:
    calls = 0

    def counting_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return handler(request)

    client = client_for(counting_handler)

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == expected_code
    assert captured.value.retryable is False
    assert calls == 1


def test_submission_models_are_immutable() -> None:
    from tianyan_mcp.experiment_models import ExperimentSubmissionResult
    from tianyan_mcp.models import BackendType

    request = request_model()
    result = ExperimentSubmissionResult(
        query_id="query-1",
        backend_code="tianyan_sw",
        resource_type=BackendType.SIMULATOR,
        shots=100,
        name="bell-test",
        lab_name="mcp-test",
    )

    with pytest.raises(ValueError):
        request.shots = 200
    with pytest.raises(ValueError):
        result.status = "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_auth_failure_invalidates_token_without_retry(status_code: int) -> None:
    provider = FakeTokenProvider()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    client = client_for(handler, provider)

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == "SUBMISSION_OUTCOME_UNKNOWN"
    assert captured.value.retryable is False
    assert provider.invalidated is True
    assert calls == 1


@pytest.mark.asyncio
async def test_rejection_error_does_not_expose_token_request_or_upstream_body() -> None:
    client = client_for(
        responding(httpx.Response(500, text="upstream diagnostic must remain private"))
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    error_text = str(captured.value)
    assert "fake-token" not in error_text
    assert "Authorization" not in error_text
    assert "H Q1" not in error_text
    assert "upstream diagnostic must remain private" not in error_text


@pytest.mark.asyncio
async def test_business_rejection_surfaces_upstream_message_field() -> None:
    client = client_for(
        responding(
            httpx.Response(
                200,
                json={
                    "code": 500,
                    "success": False,
                    "message": "该账号无真实硬件提交权限",
                },
            )
        )
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == "SUBMISSION_REJECTED"
    assert "天衍云拒绝了实验提交" in captured.value.message
    assert "该账号无真实硬件提交权限" in captured.value.message


@pytest.mark.asyncio
async def test_business_rejection_decodes_msg_cn_en_envelope() -> None:
    client = client_for(
        responding(
            httpx.Response(
                200,
                json={
                    "code": 500,
                    "msg": '{"CN":"shots 超出上限","EN":"shots exceed the limit"}',
                },
            )
        )
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == "SUBMISSION_REJECTED"
    assert "shots 超出上限" in captured.value.message
    assert "shots exceed the limit" in captured.value.message


@pytest.mark.asyncio
async def test_http_4xx_json_envelope_message_is_surfaced() -> None:
    client = client_for(
        responding(
            httpx.Response(
                400,
                json={"code": 400, "message": "computerCode 不存在", "success": False},
            )
        )
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == "SUBMISSION_REJECTED"
    assert "computerCode 不存在" in captured.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text="upstream secret diagnostic"),
        httpx.Response(400, content=b"\xff\xfe not json"),
        httpx.Response(302, headers={"Location": str(SUBMIT_URL)}),
        httpx.Response(200, json={"code": 1, "success": False}),
    ],
    ids=["plain-text-body", "non-json-body", "redirect", "envelope-without-message"],
)
async def test_rejection_without_structured_message_keeps_generic_text(
    response: httpx.Response,
) -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(responding(response)).submit(request_model())

    assert captured.value.code == "SUBMISSION_REJECTED"
    assert captured.value.message == "天衍云拒绝了实验提交"
    assert "upstream secret diagnostic" not in str(captured.value)
    assert "fake-token" not in str(captured.value)


@pytest.mark.asyncio
async def test_business_rejection_redacts_upstream_server_internals() -> None:
    client = client_for(
        responding(
            httpx.Response(
                200,
                json={
                    "code": 500,
                    "success": False,
                    "message": (
                        "java.lang.IllegalStateException: cannot read "
                        "/opt/app/config/server.jks\n"
                        "\tat com.zdxlz.qccp.SubmitService.submit(SubmitService.java:88)"
                    ),
                },
            )
        )
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    message = captured.value.message
    assert "[redacted]" in message
    assert "IllegalStateException" not in message
    assert "/opt/app/config/server.jks" not in message
    assert "com.zdxlz" not in message
    assert "SubmitService.java:88" not in message


@pytest.mark.asyncio
async def test_surfaced_upstream_message_is_sanitized_and_bounded() -> None:
    long_message = "硬件提交被拒 " * 80
    assert len(long_message) > 300
    raw_message = long_message.replace("硬件提交被拒 ", "硬件\x00提交被拒 ", 1)
    client = client_for(
        responding(httpx.Response(200, json={"code": 1, "success": False, "message": raw_message}))
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    message = captured.value.message
    assert "\x00" not in message
    assert len(message) <= len("天衍云拒绝了实验提交：") + 301
    assert message.startswith("天衍云拒绝了实验提交：")


FIXED_FIELD_OVERRIDES: list[dict[str, object]] = [
    {"lab_id": "unexpected-lab"},
    {"is_verify": False},
    {"language": "openqasm"},
    {"priority": 1},
]


@pytest.mark.parametrize("overrides", FIXED_FIELD_OVERRIDES)
def test_submission_rejects_overrides_of_fixed_upstream_fields(
    overrides: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "shots": 100,
        "circuit": ("H Q1\nM Q1",),
        "name": "bell-test",
        "lab_name": "mcp-test",
        "computerCode": "tianyan_sw",
    }
    payload.update(overrides)

    with pytest.raises(ValueError):
        UpstreamExperimentSubmission.model_validate(payload)


def test_submission_circuit_cannot_be_mutated_in_place() -> None:
    request = request_model()

    with pytest.raises(AttributeError):
        mutable_circuit = cast(MutableCircuit, request.circuit)
        mutable_circuit.append("X Q1")


@pytest.mark.parametrize(
    "circuit",
    [(), tuple(f"H Q{index}" for index in range(1, 22))],
    ids=["empty", "more-than-twenty"],
)
def test_submission_rejects_circuit_outside_model_limits(
    circuit: tuple[str, ...],
) -> None:
    payload: dict[str, object] = {
        "shots": 100,
        "circuit": circuit,
        "name": "bell-test",
        "lab_name": "mcp-test",
        "computerCode": "tianyan_sw",
    }

    with pytest.raises(ValueError):
        UpstreamExperimentSubmission.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [False, "0", 0.0, 1.5])
async def test_rejects_invalid_submission_business_code_type(code: object) -> None:
    client = client_for(
        responding(
            httpx.Response(
                200,
                json={"code": code, "success": True, "data": {"query_ids": ["query-1"]}},
            )
        )
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == "SUBMISSION_OUTCOME_UNKNOWN"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [-1, 1])
async def test_rejects_nonzero_integer_submission_business_code(code: int) -> None:
    client = client_for(
        responding(
            httpx.Response(
                200,
                json={"code": code, "success": True, "data": {"query_ids": ["query-1"]}},
            )
        )
    )

    with pytest.raises(ServiceError) as captured:
        await client.submit(request_model())

    assert captured.value.code == "SUBMISSION_REJECTED"
