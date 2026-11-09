import json
from typing import Any, cast
from unittest.mock import patch

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from tests.test_expectation_service import FakeReader, FakeSubmitter, service
from tianyan_mcp.expectation_tools import register_expectation_tools


@pytest.mark.parametrize("batch", [False, True])
async def test_expectation_tool_defaults_to_preview(batch: bool) -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    mcp = FastMCP("expectation-test")
    register_expectation_tools(mcp, service(writer, reader))
    arguments: dict[str, Any] = {
        "computer_code": "tianyan_sw",
        "name": "test",
        "shots": 10,
        "observable": [{"pauli": "X1", "coefficient": 1.0}],
        "circuits" if batch else "circuit": ["H Q1", "X Q1"] if batch else "H Q1",
    }
    result = await mcp.call_tool(
        "batch_estimate_quantum_expectation" if batch else "estimate_quantum_expectation", arguments
    )
    assert isinstance(result, tuple)
    structured = cast(dict[str, Any], result[1])
    assert structured["dry_run"] is True
    assert structured["job_count"] == (2 if batch else 1)
    assert writer.calls == reader.calls == []


async def test_tool_submission_and_manifest_query_roundtrip() -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    mcp = FastMCP("expectation-test")
    register_expectation_tools(mcp, service(writer, reader))
    submitted = await mcp.call_tool(
        "estimate_quantum_expectation",
        {
            "computer_code": "tianyan_sw",
            "circuit": "H Q1",
            "name": "test",
            "shots": 10,
            "observable": [{"pauli": "Z1", "coefficient": 1.0}],
            "dry_run": False,
        },
    )
    assert isinstance(submitted, tuple)
    manifest = cast(dict[str, Any], submitted[1])["manifest"]
    result = await mcp.call_tool("get_quantum_expectation_result", {"manifest": manifest})
    assert isinstance(result, tuple)
    assert cast(dict[str, Any], result[1])["results"][0]["value"] == 1.0
    assert len(writer.calls) == 1 and reader.calls == ["1"]


async def test_tool_domain_errors_and_unknown_errors_are_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mcp = FastMCP("expectation-test")
    estimator = service(FakeSubmitter(), FakeReader())
    register_expectation_tools(mcp, estimator)
    arguments = {
        "computer_code": "tianyan_sw",
        "circuit": "M Q1",
        "name": "test",
        "shots": 10,
        "observable": [{"pauli": "Z1", "coefficient": 1.0}],
    }
    with pytest.raises(ToolError, match="INVALID_EXPECTATION_ARGUMENT"):
        await mcp.call_tool("estimate_quantum_expectation", arguments)
    with patch.object(estimator, "submit", side_effect=RuntimeError("private-detail")):
        with pytest.raises(ToolError, match="INTERNAL_ERROR") as captured:
            await mcp.call_tool("estimate_quantum_expectation", arguments)
    assert "private-detail" not in str(captured.value) + caplog.text


@pytest.mark.parametrize(
    "updates", [{"shots": True}, {"dry_run": "false"}, {"shots": 1001}, {"observable": []}]
)
async def test_invalid_tool_schema_never_submits(updates: dict[str, object]) -> None:
    writer, reader = FakeSubmitter(), FakeReader()
    mcp = FastMCP("expectation-test")
    register_expectation_tools(mcp, service(writer, reader))
    arguments: dict[str, object] = {
        "computer_code": "tianyan_sw",
        "circuit": "H Q1",
        "name": "test",
        "shots": 10,
        "observable": [{"pauli": "Z1", "coefficient": 1.0}],
        "dry_run": False,
    }
    arguments.update(updates)
    with pytest.raises(ToolError):
        await mcp.call_tool("estimate_quantum_expectation", json.loads(json.dumps(arguments)))
    assert writer.calls == []
