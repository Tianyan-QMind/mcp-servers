import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.waveform_client import DIAGRAM_URL, GENERATE_URL, TianyanWaveformClient
from tianyan_mcp.waveform_models import UpstreamWaveformSubmission, UpstreamWaveformTask

SNOWFLAKE_ID = 1823156489274675200


class FakeTokenProvider:
    def __init__(self) -> None:
        self.invalidated = False

    async def get_access_token(self) -> str:
        return "fake-waveform-token"

    def invalidate(self) -> None:
        self.invalidated = True


def submission(
    *, circuit_name: str | None = "bell-state", is_verify: bool = True
) -> UpstreamWaveformSubmission:
    return UpstreamWaveformSubmission(
        circuit="H Q0\nM Q0",
        qcCode="QPU-01",
        circuitName=circuit_name,
        isVerify="true" if is_verify else "false",
    )


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
    provider: FakeTokenProvider | None = None,
) -> TianyanWaveformClient:
    return TianyanWaveformClient(
        provider or FakeTokenProvider(),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )


def success_task_payload() -> dict[str, object]:
    return {
        "id": SNOWFLAKE_ID,
        "circuitName": "bell-state",
        "circuitContent": "H Q0\nM Q0",
        "status": 0,
        "userId": 1001,
        "username": "zhangsan",
        "createTime": "2026-09-28 15:00:00",
        "visibleUrl": None,
        "measureQubitList": "Q0, Q1",
    }


@pytest.mark.asyncio
async def test_generate_posts_exact_body_with_bearer_and_parses_task() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.method == "POST"
        assert str(request.url) == GENERATE_URL
        assert request.headers["Authorization"] == "Bearer fake-waveform-token"
        assert json.loads(request.content) == {
            "circuit": "H Q0\nM Q0",
            "qcCode": "QPU-01",
            "circuitName": "bell-state",
            "isVerify": "true",
        }
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "message": None, "data": success_task_payload()},
        )

    task = await client_for(handler).generate(submission())

    assert calls == 1
    assert task.id == str(SNOWFLAKE_ID)
    assert task.circuit_name == "bell-state"
    assert task.circuit_content == "H Q0\nM Q0"
    assert task.status == 0
    assert task.create_time == "2026-09-28 15:00:00"
    assert task.visible_url is None
    assert task.measure_qubit_list == "Q0, Q1"


@pytest.mark.asyncio
async def test_generate_omits_absent_name_and_serializes_false_is_verify() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "circuit": "H Q0\nM Q0",
            "qcCode": "QPU-01",
            "isVerify": "false",
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {
                    "id": "123",
                    "circuitName": "example123",
                    "circuitContent": "H Q0",
                    "status": 1,
                    "visibleUrl": "https://files.example.com/waveform/x?token=t",
                    "measureQubitList": None,
                },
            },
        )

    task = await client_for(handler).generate(submission(circuit_name=None, is_verify=False))

    assert task.id == "123"
    assert task.status == 1
    assert task.visible_url == "https://files.example.com/waveform/x?token=t"


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
        (raising(httpx.ReadTimeout("timeout")), "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN"),
        (raising(httpx.ConnectError("connection error")), "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN"),
        (responding(httpx.Response(500)), "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN"),
        (
            responding(httpx.Response(200, content=b"not-json")),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(httpx.Response(200, json=["not-a-payload"])),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(httpx.Response(200, json={"code": "0", "data": {}})),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(
                httpx.Response(200, json={"code": 0, "success": True, "data": "not-an-object"})
            ),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(
                httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "success": True,
                        "data": {"id": "123", "circuitName": "n"},
                    },
                )
            ),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(httpx.Response(302, headers={"Location": GENERATE_URL})),
            "WAVEFORM_SUBMISSION_REJECTED",
        ),
        (responding(httpx.Response(400)), "WAVEFORM_SUBMISSION_REJECTED"),
        (
            responding(
                httpx.Response(
                    200,
                    json={"code": 500, "message": "qcCode 不存在", "data": None},
                )
            ),
            "WAVEFORM_SUBMISSION_REJECTED",
        ),
        (
            responding(
                httpx.Response(
                    200,
                    json={
                        "code": 500,
                        "msg": '{"CN":"未知异常","EN":"Unknown exception"}',
                    },
                )
            ),
            "WAVEFORM_SUBMISSION_REJECTED",
        ),
        (
            responding(httpx.Response(200, json={"code": 0, "data": {}})),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
        (
            responding(httpx.Response(200, json={"code": 0, "success": False, "data": None})),
            "WAVEFORM_SUBMISSION_REJECTED",
        ),
        (
            responding(
                httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "success": True,
                        "data": {"id": 0, "circuitName": "n", "status": 0},
                    },
                )
            ),
            "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN",
        ),
    ],
    ids=[
        "timeout",
        "connection-error",
        "server-error",
        "non-json",
        "non-object",
        "code-wrong-type",
        "data-not-object",
        "data-malformed",
        "redirect",
        "bad-request",
        "business-rejection-message",
        "business-rejection-msg-json",
        "success-flag-missing",
        "explicit-success-false",
        "non-positive-id",
    ],
)
async def test_generate_failures_raise_stable_errors_without_retry(
    handler: Callable[[httpx.Request], httpx.Response], expected_code: str
) -> None:
    calls = 0

    def counting_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return handler(request)

    with pytest.raises(ServiceError) as captured:
        await client_for(counting_handler).generate(submission())

    assert captured.value.code == expected_code
    assert captured.value.retryable is False
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_generate_auth_failure_invalidates_token_without_retry(
    status_code: int,
) -> None:
    provider = FakeTokenProvider()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    with pytest.raises(ServiceError) as captured:
        await client_for(handler, provider).generate(submission())

    assert captured.value.code == "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN"
    assert provider.invalidated is True
    assert calls == 1


@pytest.mark.asyncio
async def test_generate_error_hides_token_and_upstream_detail() -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(
            responding(httpx.Response(500, text="upstream secret diagnostic"))
        ).generate(submission())

    detail = str(captured.value)
    assert "fake-waveform-token" not in detail
    assert "upstream secret diagnostic" not in detail


@pytest.mark.asyncio
async def test_generate_business_rejection_surfaces_upstream_message() -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(
            responding(
                httpx.Response(
                    200,
                    json={"code": 500, "message": "qcCode 不存在", "data": None},
                )
            )
        ).generate(submission())

    assert captured.value.code == "WAVEFORM_SUBMISSION_REJECTED"
    assert "天衍云拒绝了波形生成任务" in captured.value.message
    assert "qcCode 不存在" in captured.value.message


@pytest.mark.asyncio
async def test_generate_business_rejection_decodes_msg_cn_en_envelope() -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(
            responding(
                httpx.Response(
                    200,
                    json={
                        "code": 500,
                        "msg": '{"CN":"门校验未通过","EN":"gate verification failed"}',
                    },
                )
            )
        ).generate(submission())

    assert captured.value.code == "WAVEFORM_SUBMISSION_REJECTED"
    assert "门校验未通过" in captured.value.message
    assert "gate verification failed" in captured.value.message


@pytest.mark.asyncio
async def test_generate_http_4xx_json_envelope_message_is_surfaced() -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(
            responding(
                httpx.Response(
                    400,
                    json={"code": 400, "message": "qcCode 不存在", "success": False},
                )
            )
        ).generate(submission())

    assert captured.value.code == "WAVEFORM_SUBMISSION_REJECTED"
    assert "qcCode 不存在" in captured.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text="upstream secret diagnostic"),
        httpx.Response(400),
        httpx.Response(302, headers={"Location": GENERATE_URL}),
        httpx.Response(200, json={"code": 0, "success": False, "data": None}),
    ],
    ids=["plain-text-body", "empty-body", "redirect", "envelope-without-message"],
)
async def test_generate_rejection_without_structured_message_keeps_generic_text(
    response: httpx.Response,
) -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(responding(response)).generate(submission())

    assert captured.value.code == "WAVEFORM_SUBMISSION_REJECTED"
    assert captured.value.message == "天衍云拒绝了波形生成任务"
    assert "upstream secret diagnostic" not in str(captured.value)
    assert "fake-waveform-token" not in str(captured.value)


@pytest.mark.asyncio
async def test_get_diagram_uses_query_param_and_parses_generating_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url).startswith(DIAGRAM_URL)
        assert request.url.params["id"] == str(SNOWFLAKE_ID)
        assert request.headers["Authorization"] == "Bearer fake-waveform-token"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {
                    "id": str(SNOWFLAKE_ID),
                    "circuit_name": "bell-state",
                    "visibleUrl": "",
                },
            },
        )

    diagram = await client_for(handler).get_diagram(str(SNOWFLAKE_ID))

    assert diagram.id == str(SNOWFLAKE_ID)
    assert diagram.circuit_name == "bell-state"
    assert diagram.visible_url == ""
    # Live generating/completed payloads omit status entirely.
    assert diagram.status is None


@pytest.mark.asyncio
async def test_get_diagram_accepts_integer_id_and_completed_url() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {
                    "id": SNOWFLAKE_ID,
                    "circuit_name": "bell-state",
                    "visibleUrl": "https://files.example.com/waveform/xxx?token=tok",
                    "status": 1,
                },
            },
        )

    diagram = await client_for(handler).get_diagram(str(SNOWFLAKE_ID))

    assert diagram.id == str(SNOWFLAKE_ID)
    assert diagram.visible_url == "https://files.example.com/waveform/xxx?token=tok"
    assert diagram.status == 1


@pytest.mark.asyncio
async def test_get_diagram_network_error_is_retryable() -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(raising(httpx.ReadTimeout("timeout"))).get_diagram("123")

    assert captured.value.code == "WAVEFORM_QUERY_UNAVAILABLE"
    assert captured.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_get_diagram_auth_failure_invalidates_token(status_code: int) -> None:
    provider = FakeTokenProvider()

    with pytest.raises(ServiceError) as captured:
        await client_for(responding(httpx.Response(status_code)), provider).get_diagram("123")

    assert captured.value.code == "SDK_AUTH_FAILED"
    assert provider.invalidated is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 500])
async def test_get_diagram_rejects_non_success_http_status(status_code: int) -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(responding(httpx.Response(status_code))).get_diagram("123")

    assert captured.value.code == "WAVEFORM_QUERY_REJECTED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"code": 500, "message": "id不能为空"}, "INVALID_WAVEFORM_TASK_ID"),
        ({"code": 500, "message": "未查询到波形线路数据"}, "WAVEFORM_TASK_NOT_FOUND"),
        ({"code": 500, "message": "不是该用户的波形图"}, "WAVEFORM_TASK_FORBIDDEN"),
        ({"code": 500, "message": "线路波形图生成失败"}, "WAVEFORM_GENERATION_FAILED"),
        ({"code": 500, "message": "something else went wrong"}, "WAVEFORM_QUERY_REJECTED"),
        ({"code": 500}, "WAVEFORM_QUERY_REJECTED"),
        # Live upstream puts errors into msg, often as a JSON-encoded CN/EN envelope.
        (
            {
                "code": 500,
                "msg": '{"CN":"未查询到波形线路数据","EN":"waveform task not found"}',
            },
            "WAVEFORM_TASK_NOT_FOUND",
        ),
        (
            {"code": 500, "msg": '{"CN":"未知异常","EN":"Unknown exception"}'},
            "WAVEFORM_QUERY_REJECTED",
        ),
        ({"code": 500, "msg": "线路波形图生成失败"}, "WAVEFORM_GENERATION_FAILED"),
        (
            {"code": 401, "msg": '{"CN":"令牌不能为空","EN":"The token cannot be empty"}'},
            "WAVEFORM_QUERY_REJECTED",
        ),
        (
            {"code": 500, "msg": json.dumps("线路波形图生成失败")},
            "WAVEFORM_GENERATION_FAILED",
        ),
        ({"code": 500, "msg": "123"}, "WAVEFORM_QUERY_REJECTED"),
    ],
    ids=[
        "empty-id",
        "not-found",
        "not-owner",
        "failed",
        "unknown-message",
        "missing-message",
        "not-found-msg-json",
        "unknown-msg-json",
        "failed-msg-plain",
        "auth-error-msg-json",
        "msg-json-string",
        "msg-json-scalar",
    ],
)
async def test_get_diagram_classifies_documented_business_errors(
    payload: dict[str, object], expected_code: str
) -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(responding(httpx.Response(200, json=payload))).get_diagram("123")

    assert captured.value.code == expected_code
    assert captured.value.retryable is False
    # Documented phrases are the fixed MCP-layer messages; only unknown upstream
    # text must never be passed through verbatim.
    detail = str(captured.value)
    if expected_code == "WAVEFORM_QUERY_REJECTED":
        assert "Unknown exception" not in detail
        assert "something else went wrong" not in detail
        assert "令牌不能为空" not in detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=["not-a-payload"]),
        httpx.Response(200, json={"code": "0", "data": {}}),
        httpx.Response(200, json={"code": 0, "success": True, "data": "not-an-object"}),
        httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"id": "123", "visibleUrl": ""}},
        ),
        httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"id": "123", "circuit_name": "n", "visibleUrl": 42},
            },
        ),
        httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {
                    "id": "123",
                    "circuit_name": "n",
                    "visibleUrl": "",
                    "status": 3,
                },
            },
        ),
        httpx.Response(200, json={"code": 0, "success": "not-a-bool", "data": {}}),
    ],
    ids=[
        "non-json",
        "non-object",
        "code-wrong-type",
        "data-not-object",
        "missing-circuit-name",
        "url-wrong-type",
        "status-out-of-range",
        "success-wrong-type",
    ],
)
async def test_get_diagram_rejects_malformed_envelope(response: httpx.Response) -> None:
    with pytest.raises(ServiceError) as captured:
        await client_for(responding(response)).get_diagram("123")

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"


def test_submission_model_rejects_unknown_and_non_string_is_verify() -> None:
    with pytest.raises(ValueError):
        UpstreamWaveformSubmission.model_validate(
            {
                "circuit": "H Q0",
                "qcCode": "QPU-01",
                "isVerify": True,
            }
        )
    with pytest.raises(ValueError):
        UpstreamWaveformSubmission.model_validate(
            {
                "circuit": "H Q0",
                "qcCode": "QPU-01",
                "unexpected": "x",
            }
        )


def test_waveform_models_are_immutable() -> None:
    request = submission()
    with pytest.raises(ValidationError):
        request.qc_code = "QPU-02"  # type: ignore[misc]


@pytest.mark.parametrize("bad_id", ["", "   ", True, 0, -1])
def test_upstream_waveform_task_rejects_invalid_ids(bad_id: object) -> None:
    with pytest.raises(ValidationError):
        UpstreamWaveformTask.model_validate(
            {
                "id": bad_id,
                "circuitName": "n",
                "circuitContent": "H Q0",
                "status": 0,
            }
        )
