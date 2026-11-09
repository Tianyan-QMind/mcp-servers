import json
from typing import Any

import httpx
import pytest
import respx
from mcp.types import TextContent

from tests.mcp_helper import running_mcp_client
from tianyan_mcp.app import create_app
from tianyan_mcp.client import TIANYAN_QUERY, TIANYAN_URL
from tianyan_mcp.config import Settings
from tianyan_mcp.experiment_client import RESULT_URL, STATUS_URL, SUBMIT_URL
from tianyan_mcp.sdk_auth import SDK_LOGIN_URL
from tianyan_mcp.waveform_client import DIAGRAM_URL, GENERATE_URL

API_KEY = "test-key-" + "x" * 23


@respx.mock
async def test_expectation_http_preview_submit_and_query_with_fake_upstream(
    upstream_payload: dict[str, Any],
) -> None:
    respx.route(url__regex=r"^http://127\.0\.0\.1:").pass_through()
    backend = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).respond(200, json=upstream_payload)
    login = respx.post(SDK_LOGIN_URL).respond(
        200,
        json={
            "code": 0,
            "success": True,
            "data": {"access_token": "fake-expectation-token", "expires_in": 720},
        },
    )
    captured_submits: list[httpx.Request] = []

    def capture_submit(request: httpx.Request) -> httpx.Response:
        captured_submits.append(request)
        return httpx.Response(
            200, json={"code": 0, "success": True, "data": {"query_ids": ["123"]}}
        )

    submitted_route = respx.post(SUBMIT_URL).mock(side_effect=capture_submit)
    status = respx.post(STATUS_URL).respond(
        200, json={"code": 0, "success": True, "data": {"status": 0}}
    )
    result = respx.post(RESULT_URL).respond(
        200,
        json={
            "code": 0,
            "success": True,
            "data": {
                "experimentResultModelList": [
                    {
                        "resultStatus": [[1], [0]],
                        "probability": {"0": 1.0},
                        "experimentTaskId": "123",
                    }
                ]
            },
        },
    )
    settings = Settings.from_env(
        {"TIANYAN_MCP_API_KEY": API_KEY, "TIANYAN_SDK_OPEN_ID": "fake-expectation-open-id"}
    )
    arguments = {
        "computer_code": "tianyan_s",
        "circuit": "H Q1",
        "name": "test",
        "shots": 10,
        "observable": [{"pauli": "X1", "coefficient": 1.0}],
    }
    async with running_mcp_client(create_app(settings), API_KEY) as session:
        preview = await session.call_tool("estimate_quantum_expectation", arguments)
        assert preview.isError is not True
        assert backend.call_count == login.call_count == submitted_route.call_count == 0
        submitted = await session.call_tool(
            "estimate_quantum_expectation", {**arguments, "dry_run": False}
        )
        assert submitted.isError is not True
        assert submitted.structuredContent is not None
        manifest = submitted.structuredContent["manifest"]
        queried = await session.call_tool("get_quantum_expectation_result", {"manifest": manifest})
    assert queried.isError is not True
    assert queried.structuredContent is not None
    assert queried.structuredContent["results"][0]["value"] == 1.0
    assert (
        backend.call_count
        == login.call_count
        == submitted_route.call_count
        == status.call_count
        == result.call_count
        == 1
    )
    assert json.loads(captured_submits[0].content)["circuit"] == ["H Q1\nH Q1\nM Q1"]
    assert "result_status" not in json.dumps(queried.structuredContent)


@pytest.mark.asyncio
@respx.mock
async def test_waveform_http_generate_then_poll_with_fake_upstream() -> None:
    respx.route(url__regex=r"^http://127\.0\.0\.1:").pass_through()
    login = respx.post(SDK_LOGIN_URL).respond(
        200,
        json={
            "code": 0,
            "success": True,
            "data": {"access_token": "fake-waveform-token", "expires_in": 720},
        },
    )
    generated_requests: list[httpx.Request] = []

    def capture_generate(request: httpx.Request) -> httpx.Response:
        generated_requests.append(request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {
                    "id": 1823156489274675200,
                    "circuitName": "bell-state",
                    "circuitContent": "H Q0\nM Q0",
                    "status": 0,
                    "createTime": "2026-09-28 15:00:00",
                    "visibleUrl": None,
                    "measureQubitList": "Q0",
                },
            },
        )

    generate_route = respx.post(GENERATE_URL).mock(side_effect=capture_generate)
    diagram_route = respx.get(DIAGRAM_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "success": True,
                    "message": None,
                    "data": {
                        "id": "1823156489274675200",
                        "circuit_name": "bell-state",
                        "visibleUrl": "",
                    },
                },
            ),
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "success": True,
                    "message": None,
                    "data": {
                        "id": "1823156489274675200",
                        "circuit_name": "bell-state",
                        "visibleUrl": "https://files.example.com/waveform/xxx?token=tok",
                    },
                },
            ),
        ]
    )
    settings = Settings.from_env(
        {"TIANYAN_MCP_API_KEY": API_KEY, "TIANYAN_SDK_OPEN_ID": "fake-waveform-open-id"}
    )

    async with running_mcp_client(create_app(settings), API_KEY) as session:
        generated = await session.call_tool(
            "generate_waveform_diagram",
            {
                "qc_code": "QPU-01",
                "circuit": "H Q0\nM Q0",
                "circuit_name": "bell-state",
            },
        )
        generating = await session.call_tool(
            "get_waveform_diagram", {"task_id": "1823156489274675200"}
        )
        completed = await session.call_tool(
            "get_waveform_diagram", {"task_id": "1823156489274675200"}
        )

    assert generated.isError is not True
    assert generated.structuredContent is not None
    assert generated.structuredContent["status"] == "generating"
    assert generated.structuredContent["measure_qubits"] == ["Q0"]
    assert json.loads(generated_requests[0].content) == {
        "circuit": "H Q0\nM Q0",
        "qcCode": "QPU-01",
        "circuitName": "bell-state",
        "isVerify": "true",
    }
    assert generating.structuredContent is not None
    assert generating.structuredContent["status"] == "generating"
    assert generating.structuredContent["visible_url"] is None
    assert completed.structuredContent is not None
    assert completed.structuredContent["status"] == "completed"
    assert completed.structuredContent["visible_url"] == (
        "https://files.example.com/waveform/xxx?token=tok"
    )
    assert login.call_count == 1
    assert generate_route.call_count == 1
    assert diagram_route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_tool_flow(
    upstream_payload: dict[str, Any],
) -> None:
    respx.route(url__regex=r"^http://127\.0\.0\.1:").pass_through()
    upstream = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        return_value=httpx.Response(200, json=upstream_payload)
    )
    settings = Settings.from_env({"TIANYAN_MCP_API_KEY": API_KEY})

    async with running_mcp_client(create_app(settings), API_KEY) as session:
        catalog = await session.list_tools()
        assert {tool.name for tool in catalog.tools} == {
            "list_available_quantum_backends",
            "get_quantum_backend",
            "submit_quantum_experiment",
            "submit_quantum_experiment_batch",
            "get_quantum_experiment_result",
            "estimate_quantum_expectation",
            "batch_estimate_quantum_expectation",
            "get_quantum_expectation_result",
            "generate_waveform_diagram",
            "get_waveform_diagram",
        }

        listed = await session.call_tool(
            "list_available_quantum_backends",
            {"backend_type": "simulator"},
        )
        assert listed.isError is not True
        assert listed.structuredContent is not None
        assert listed.structuredContent["total"] == 1

        detail = await session.call_tool(
            "get_quantum_backend",
            {"identifier": "TIANYAN-287"},
        )
        assert detail.isError is not True
        assert detail.structuredContent is not None
        assert detail.structuredContent["id"] == "1968635345322422274"

        missing = await session.call_tool(
            "get_quantum_backend",
            {"identifier": "missing"},
        )
        assert missing.isError is True
        assert isinstance(missing.content[0], TextContent)
        assert "RESOURCE_NOT_FOUND" in missing.content[0].text

        call_count_before_invalid_input = upstream.call_count
        invalid = await session.call_tool(
            "get_quantum_backend",
            {"identifier": ""},
        )
        assert invalid.isError is True
        assert upstream.call_count == call_count_before_invalid_input


@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_single_user_submit_omits_absent_lab_name(
    upstream_payload: dict[str, Any],
) -> None:
    respx.route(url__regex=r"^http://127\.0\.0\.1:").pass_through()
    backend = respx.get(TIANYAN_URL, params=TIANYAN_QUERY).mock(
        return_value=httpx.Response(200, json=upstream_payload)
    )
    login = respx.post(SDK_LOGIN_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"access_token": "single-user-submit-token", "expires_in": 720},
            },
        )
    )
    captured_submits: list[httpx.Request] = []

    def submit_handler(request: httpx.Request) -> httpx.Response:
        captured_submits.append(request)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"query_ids": ["single-user-submit-query"]},
            },
        )

    submit = respx.post(SUBMIT_URL).mock(side_effect=submit_handler)
    settings = Settings.from_env(
        {
            "TIANYAN_MCP_API_KEY": API_KEY,
            "TIANYAN_SDK_OPEN_ID": "single-user-submit-open-id",
        }
    )

    async with running_mcp_client(create_app(settings), API_KEY) as session:
        submitted = await session.call_tool(
            "submit_quantum_experiment",
            {
                "computer_code": "tianyan_s",
                "circuit": "H q[0]",
                "name": "single-user-submit",
                "shots": 100,
            },
        )

    assert submitted.isError is not True
    assert submitted.structuredContent is not None
    assert submitted.structuredContent["query_id"] == "single-user-submit-query"
    assert submitted.structuredContent["lab_name"] is None
    assert backend.call_count == login.call_count == submit.call_count == 1
    captured_submit = captured_submits[0]
    assert captured_submit.headers["Authorization"] == "Bearer single-user-submit-token"
    assert "lab_name" not in json.loads(captured_submit.content)


@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_queries_completed_experiment_with_mocked_sdk_login() -> None:
    respx.route(url__regex=r"^http://127\.0\.0\.1:").pass_through()
    login = respx.post(SDK_LOGIN_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"access_token": "single-user-token", "expires_in": 720},
            },
        )
    )
    status = respx.post(STATUS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"status": 0}},
        )
    )
    result = respx.post(RESULT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0], [1]],
                            "probability": {"0": 0.5, "1": 0.5},
                            "experimentTaskId": "123",
                        }
                    ]
                },
            },
        )
    )
    settings = Settings.from_env(
        {
            "TIANYAN_MCP_API_KEY": API_KEY,
            "TIANYAN_SDK_OPEN_ID": "single-user-open-id",
        }
    )

    async with running_mcp_client(create_app(settings), API_KEY) as session:
        queried = await session.call_tool(
            "get_quantum_experiment_result",
            {"query_id": "123", "include_result_status": True},
        )

    assert queried.structuredContent is not None
    assert queried.structuredContent == {
        "query_id": "123",
        "status": "completed",
        "upstream_status": 0,
        "probability": {"0": 0.5, "1": 0.5},
        "experiment_task_id": "123",
        "result_status": [[0], [1]],
    }
    assert login.call_count == status.call_count == result.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_processing_result_does_not_call_find() -> None:
    respx.route(url__regex=r"^http://127\.0\.0\.1:").pass_through()
    login = respx.post(SDK_LOGIN_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"access_token": "processing-token", "expires_in": 720},
            },
        )
    )
    status = respx.post(STATUS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"status": 1}},
        )
    )
    result = respx.post(RESULT_URL).mock(
        return_value=httpx.Response(500, json={"must": "not be called"})
    )
    settings = Settings.from_env(
        {
            "TIANYAN_MCP_API_KEY": API_KEY,
            "TIANYAN_SDK_OPEN_ID": "processing-open-id",
        }
    )

    async with running_mcp_client(create_app(settings), API_KEY) as session:
        queried = await session.call_tool(
            "get_quantum_experiment_result",
            {"query_id": "456", "include_result_status": False},
        )

    assert queried.structuredContent is not None
    assert queried.structuredContent["status"] == "processing"
    assert login.call_count == status.call_count == 1
    assert result.call_count == 0
