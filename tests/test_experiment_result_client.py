import json
from collections.abc import Awaitable, Callable
from typing import cast

import httpx
import pytest

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import RESULT_URL, STATUS_URL, TianyanExperimentClient
from tianyan_mcp.experiment_models import UpstreamExperimentSubmission


class FakeTokenProvider:
    def __init__(self) -> None:
        self.invalidate_calls = 0

    async def get_access_token(self) -> str:
        return "access-token"

    def invalidate(self) -> None:
        self.invalidate_calls += 1


ResponseAction = httpx.Response | Exception


def make_client(
    actions: list[ResponseAction],
) -> tuple[TianyanExperimentClient, list[httpx.Request], FakeTokenProvider]:
    requests: list[httpx.Request] = []
    provider = FakeTokenProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        action = actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    return (
        TianyanExperimentClient(
            provider,
            timeout_seconds=1,
            transport=httpx.MockTransport(handler),
        ),
        requests,
        provider,
    )


def result_response(
    *,
    experiment_task_id: str = "query-1",
    probability: dict[str, float] | None = None,
    result_status: list[list[int]] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "code": 0,
            "message": None,
            "data": {
                "experimentResultModelList": [
                    {
                        "resultStatus": result_status if result_status is not None else [[0]],
                        "probability": probability if probability is not None else {"0": 1.0},
                        "experimentTaskId": experiment_task_id,
                    }
                ]
            },
            "success": True,
        },
    )


def query_response(*, data: object | None = None, content: bytes | None = None) -> httpx.Response:
    if content is not None:
        return httpx.Response(200, content=content)
    return httpx.Response(200, json={"code": 0, "success": True, "data": data})


@pytest.mark.asyncio
async def test_get_status_posts_one_query_id_to_fixed_production_url() -> None:
    client, requests, provider = make_client(
        [
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "message": None,
                    "data": {"status": 1},
                    "success": True,
                },
            )
        ]
    )

    assert await client.get_status("query-1") == 1
    assert len(requests) == 1
    assert str(requests[0].url) == STATUS_URL
    assert json.loads(requests[0].content) == {"query_ids": ["query-1"]}
    assert requests[0].headers["Authorization"] == "Bearer access-token"
    assert provider.invalidate_calls == 0


@pytest.mark.asyncio
async def test_get_status_accepts_completed_status_zero() -> None:
    client, requests, _ = make_client(
        [httpx.Response(200, json={"code": 0, "success": True, "data": {"status": 0}})]
    )

    assert await client.get_status("query-1") == 0
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_get_result_parses_exactly_the_documented_fields() -> None:
    client, requests, _ = make_client(
        [
            result_response(
                experiment_task_id="query-1",
                probability={"0": 0.6, "1": 0.4},
                result_status=[[0], [1]],
            )
        ]
    )

    results = await client.get_result("query-1")

    assert len(results) == 1
    assert results[0].experiment_task_id == "query-1"
    assert results[0].probability == {"0": 0.6, "1": 0.4}
    assert results[0].result_status == ((0,), (1,))
    assert isinstance(results, tuple)
    assert len(requests) == 1
    assert str(requests[0].url) == RESULT_URL
    assert json.loads(requests[0].content) == {"query_ids": ["query-1"]}


@pytest.mark.asyncio
async def test_get_result_parses_probability_json_string_from_upstream() -> None:
    client, requests, _ = make_client(
        [
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[1], [0], [1]],
                            "probability": '{"0":0.4,"1":0.6}',
                            "experimentTaskId": "task-1",
                        }
                    ]
                }
            )
        ]
    )

    results = await client.get_result("query-1")

    assert len(results) == 1
    assert results[0].probability == {"0": 0.4, "1": 0.6}
    assert results[0].result_status == ((1,), (0,), (1,))
    assert results[0].experiment_task_id == "task-1"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_get_result_converts_numeric_experiment_task_id_without_precision_loss() -> None:
    numeric_task_id = 2094304697921560578
    client, requests, _ = make_client(
        [
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[1], [0], [0]],
                            "probability": '{"0":1.0}',
                            "experimentTaskId": numeric_task_id,
                        }
                    ]
                }
            )
        ]
    )

    results = await client.get_result("2094304697921560578")

    assert len(results) == 1
    assert results[0].experiment_task_id == "2094304697921560578"
    assert isinstance(results[0].experiment_task_id, str)
    assert results[0].probability == {"0": 1.0}
    assert results[0].result_status == ((1,), (0,), (0,))
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"status": 2},
        {"status": "0"},
        {"status": False},
        [],
    ],
    ids=["missing-status", "unknown-status", "string-status", "boolean-status", "wrong-data-type"],
)
async def test_get_status_rejects_invalid_data_with_one_request(data: object) -> None:
    client, requests, _ = make_client(
        [httpx.Response(200, json={"code": 0, "success": True, "data": data})]
    )

    with pytest.raises(ServiceError) as captured:
        await client.get_status("query-1")

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert captured.value.retryable is False
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"experimentResultModelList": []},
        {"experimentResultModelList": {}},
        {"experimentResultModelList": [{"experimentTaskId": "query-1"}]},
    ],
    ids=["missing-list", "empty-list", "wrong-list-type", "invalid-result-entry"],
)
async def test_get_result_rejects_invalid_data_with_one_request(data: object) -> None:
    client, requests, _ = make_client(
        [httpx.Response(200, json={"code": 0, "success": True, "data": data})]
    )

    with pytest.raises(ServiceError) as captured:
        await client.get_result("query-1")

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert captured.value.retryable is False
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_status", "get_result"])
async def test_query_rejects_malformed_json_with_one_request(method: str) -> None:
    client, requests, _ = make_client([httpx.Response(200, content=b"not-json")])

    with pytest.raises(ServiceError) as captured:
        await cast(Callable[[str], Awaitable[object]], getattr(client, method))("query-1")

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert captured.value.retryable is False
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.parametrize("method", ["get_status", "get_result"])
async def test_query_auth_failure_invalidates_once_without_retry(
    status_code: int, method: str
) -> None:
    client, requests, provider = make_client([httpx.Response(status_code)])

    with pytest.raises(ServiceError) as captured:
        await cast(Callable[[str], Awaitable[object]], getattr(client, method))("query-1")

    assert captured.value.code == "SDK_AUTH_FAILED"
    assert captured.value.retryable is False
    assert provider.invalidate_calls == 1
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 500])
async def test_query_http_failure_is_rejected_with_one_request(status_code: int) -> None:
    client, requests, _ = make_client([httpx.Response(status_code)])

    with pytest.raises(ServiceError) as captured:
        await client.get_result("query-1")

    assert captured.value.code == "EXPERIMENT_QUERY_REJECTED"
    assert captured.value.retryable is False
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("timeout"), httpx.ConnectError("connection error")],
    ids=["timeout", "connection-error"],
)
async def test_query_transport_failure_is_retryable_with_one_request(error: Exception) -> None:
    client, requests, _ = make_client([error])

    with pytest.raises(ServiceError) as captured:
        await client.get_status("query-1")

    assert captured.value.code == "EXPERIMENT_QUERY_UNAVAILABLE"
    assert captured.value.retryable is True
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected_code",
    [
        ({"code": 1, "success": True, "data": {"status": 0}}, "EXPERIMENT_QUERY_REJECTED"),
        ({"code": 0, "success": False, "data": {"status": 0}}, "EXPERIMENT_QUERY_REJECTED"),
        ({"code": False, "success": True, "data": {"status": 0}}, "UPSTREAM_INVALID_RESPONSE"),
        ({"code": 0, "success": 1, "data": {"status": 0}}, "UPSTREAM_INVALID_RESPONSE"),
    ],
    ids=["business-code", "business-success", "invalid-code-type", "invalid-success-type"],
)
async def test_query_strictly_checks_business_envelope_with_one_request(
    payload: dict[str, object], expected_code: str
) -> None:
    client, requests, _ = make_client([httpx.Response(200, json=payload)])

    with pytest.raises(ServiceError) as captured:
        await client.get_status("query-1")

    assert captured.value.code == expected_code
    assert captured.value.retryable is False
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "case_id"),
    [
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [["0"]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": "query-1",
                        }
                    ]
                }
            ),
            "measurement-string",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[True]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": "query-1",
                        }
                    ]
                }
            ),
            "measurement-bool",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[1.5]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": "query-1",
                        }
                    ]
                }
            ),
            "measurement-float",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": "0.5"},
                            "experimentTaskId": "query-1",
                        }
                    ]
                }
            ),
            "probability-string",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": True},
                            "experimentTaskId": "query-1",
                        }
                    ]
                }
            ),
            "probability-bool",
        ),
        (
            query_response(
                content=(
                    b'{"code":0,"success":true,"data":{"experimentResultModelList":'
                    b'[{"resultStatus":[[0]],"probability":{"0":NaN},"experimentTaskId":"query-1"}]}}'
                )
            ),
            "probability-nan",
        ),
        (
            query_response(
                content=(
                    b'{"code":0,"success":true,"data":{"experimentResultModelList":'
                    b'[{"resultStatus":[[0]],"probability":{"0":Infinity},"experimentTaskId":"query-1"}]}}'
                )
            ),
            "probability-infinity",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": 0,
                        }
                    ]
                }
            ),
            "task-id-zero",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": -1,
                        }
                    ]
                }
            ),
            "task-id-negative",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": 1.0,
                        }
                    ]
                }
            ),
            "task-id-float",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": True,
                        }
                    ]
                }
            ),
            "task-id-bool",
        ),
        (
            query_response(
                data={
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": "",
                        }
                    ]
                }
            ),
            "task-id-empty",
        ),
    ],
    ids=lambda case: cast(str, case),
)
async def test_get_result_rejects_invalid_nested_scalars_with_one_request(
    response: httpx.Response, case_id: str
) -> None:
    del case_id
    client, requests, _ = make_client([response])

    with pytest.raises(ServiceError) as captured:
        await client.get_result("query-1")

    assert captured.value.code == "UPSTREAM_INVALID_RESPONSE"
    assert captured.value.retryable is False
    assert len(requests) == 1


def submission_request() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("H Q1\nM Q1",),
        name="bell-test",
        lab_name="mcp-test",
        computerCode="tianyan_sw",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (
            httpx.Response(
                200,
                json={"code": 0, "success": True, "data": {"query_ids": ["query-1"]}},
            ),
            None,
        ),
        (httpx.Response(400), "SUBMISSION_REJECTED"),
        (httpx.Response(500), "SUBMISSION_OUTCOME_UNKNOWN"),
    ],
    ids=["success", "rejected", "outcome-unknown"],
)
async def test_submit_keeps_existing_single_post_semantics(
    response: httpx.Response, expected_code: str | None
) -> None:
    client, requests, _ = make_client([response])

    if expected_code is None:
        assert await client.submit(submission_request()) == "query-1"
    else:
        with pytest.raises(ServiceError) as captured:
            await client.submit(submission_request())
        assert captured.value.code == expected_code

    assert len(requests) == 1
