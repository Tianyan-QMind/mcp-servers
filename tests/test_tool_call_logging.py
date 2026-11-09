from __future__ import annotations

import logging
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult

from tianyan_mcp.server import (
    _logging_call_tool_handler,  # pyright: ignore[reportPrivateUsage]
    create_mcp_server,
)

BELL_CIRCUIT = "H Q0\nCX Q0 Q1\nM Q0 Q1"


def _build_server() -> FastMCP:
    # 本地 cqlib 工具不触碰云端 service，注册阶段仅保存闭包引用，
    # 因此这里用极简占位对象即可。
    return create_mcp_server(
        service=object(),  # type: ignore[arg-type]
        experiment_service=object(),  # type: ignore[arg-type]
        experiment_result_service=object(),  # type: ignore[arg-type]
        waveform_service=object(),  # type: ignore[arg-type]
    )


def _low_level(mcp: FastMCP) -> Server[Any, Any]:
    # FastMCP 1.x 未公开底层 Server 访问器，测试中定点豁免。
    return mcp._mcp_server  # pyright: ignore[reportPrivateUsage]


async def _dispatch(
    mcp: FastMCP, name: str, arguments: dict[str, Any] | None = None
) -> CallToolResult:
    handler = _low_level(mcp).request_handlers[CallToolRequest]
    request = CallToolRequest(params=CallToolRequestParams(name=name, arguments=arguments or {}))
    result = await handler(request)
    assert isinstance(result.root, CallToolResult)
    return result.root


async def test_successful_tool_call_logs_tool_name(caplog: pytest.LogCaptureFixture):
    mcp = _build_server()

    with caplog.at_level(logging.INFO, logger="tianyan_mcp.server"):
        payload = await _dispatch(
            mcp, "analyze_circuit", {"circuit": BELL_CIRCUIT, "format": "qcis"}
        )

    assert payload.isError is False
    messages = [record.getMessage() for record in caplog.records]
    assert "event=tool_call_start tool=analyze_circuit" in messages
    assert "event=tool_call_done tool=analyze_circuit" in messages
    assert not any("event=tool_call_failed" in message for message in messages)


async def test_failed_tool_call_logs_tool_name(caplog: pytest.LogCaptureFixture):
    mcp = _build_server()

    with caplog.at_level(logging.INFO, logger="tianyan_mcp.server"):
        payload = await _dispatch(
            mcp, "analyze_circuit", {"circuit": "CNOT Q0 Q1", "format": "qcis"}
        )

    assert payload.isError is True
    messages = [record.getMessage() for record in caplog.records]
    assert "event=tool_call_start tool=analyze_circuit" in messages
    assert "event=tool_call_failed tool=analyze_circuit" in messages
    assert not any("event=tool_call_done" in message for message in messages)


async def test_handler_exception_is_logged_and_reraised(
    caplog: pytest.LogCaptureFixture,
):
    async def broken_handler(_: CallToolRequest) -> CallToolResult:
        raise RuntimeError("boom")

    handler = _logging_call_tool_handler(broken_handler)
    request = CallToolRequest(params=CallToolRequestParams(name="broken_tool", arguments={}))

    with caplog.at_level(logging.INFO, logger="tianyan_mcp.server"):
        with pytest.raises(RuntimeError, match="boom"):
            await handler(request)

    messages = [record.getMessage() for record in caplog.records]
    assert "event=tool_call_start tool=broken_tool" in messages
    assert "event=tool_call_exception tool=broken_tool" in messages
    assert not any("event=tool_call_done" in message for message in messages)
    assert not any("event=tool_call_failed" in message for message in messages)
