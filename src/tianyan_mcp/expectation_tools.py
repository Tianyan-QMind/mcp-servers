"""MCP tool entry points exposed over the HTTP transport."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.expectation_models import (
    ExpectationManifest,
    ExpectationResults,
    ExpectationSubmission,
    ObservableTerm,
)
from tianyan_mcp.expectation_service import ExpectationService

LOGGER = logging.getLogger(__name__)
Circuit = Annotated[str, Field(min_length=1, max_length=20_000)]
Circuits = Annotated[list[Circuit], Field(min_length=1, max_length=8)]
Observable = Annotated[list[ObservableTerm], Field(min_length=1, max_length=16)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]
Name = Annotated[str, Field(min_length=1, max_length=100)]
Shots = Annotated[int, Field(strict=True, ge=1, le=1000)]
Switch = Annotated[bool, Field(strict=True)]


@contextmanager
def tool_errors() -> Generator[None]:
    try:
        yield
    except ServiceError as exc:
        LOGGER.warning("event=expectation_tool_error code=%s", exc.code)
        raise RuntimeError(exc.as_json()) from None
    except Exception:
        LOGGER.error("event=expectation_tool_error code=INTERNAL_ERROR")
        raise RuntimeError(
            ServiceError("INTERNAL_ERROR", "服务内部错误", False).as_json()
        ) from None


def register_expectation_tools(mcp: FastMCP, service: ExpectationService) -> None:
    async def estimate_quantum_expectation(
        computer_code: Identifier,
        circuit: Circuit,
        observable: Observable,
        name: Name,
        shots: Shots,
        lab_name: Name | None = None,
        allow_hardware: Switch = False,
        dry_run: Switch = True,
    ) -> ExpectationSubmission:
        """Preview or submit measurement-based expectation estimation for one QCIS preparation.

        X1/Y1/Z1 refer to physical Q1. Input must have no measurements or resets.
        dry_run=true creates no cloud tasks. Before dry_run=false, the host MUST show
        backend, generated measurement circuits, job count and total shots, and obtain
        user confirmation. shots applies to EACH measurement circuit. Repeated submission
        can duplicate charged tasks; never automatically retry. Save the full manifest.
        """
        with tool_errors():
            return await service.submit(
                computer_code=computer_code,
                circuits=[circuit],
                observable=observable,
                name=name,
                shots=shots,
                lab_name=lab_name,
                allow_hardware=allow_hardware,
                dry_run=dry_run,
            )

    async def batch_estimate_quantum_expectation(
        computer_code: Identifier,
        circuits: Circuits,
        observable: Observable,
        name: Name,
        shots: Shots,
        lab_name: Name | None = None,
        allow_hardware: Switch = False,
        dry_run: Switch = True,
    ) -> ExpectationSubmission:
        """Preview or submit up to 8 QCIS preparations using the SAME observable.

        At most 16 expanded measurement tasks. shots is PER measurement, not per batch.
        Default dry_run=true does not submit. Before dry_run=false, the host MUST show
        backend, generated circuits, job count and total shots and obtain confirmation.
        Submission stops on the first error or unknown outcome; do not retry the batch.
        Save the full manifest, including partial submissions, for the read-only query.
        """
        with tool_errors():
            return await service.submit(
                computer_code=computer_code,
                circuits=circuits,
                observable=observable,
                name=name,
                shots=shots,
                lab_name=lab_name,
                allow_hardware=allow_hardware,
                dry_run=dry_run,
            )

    async def get_quantum_expectation_result(manifest: ExpectationManifest) -> ExpectationResults:
        """Query a complete saved manifest once; return per-circuit results in input order.

        This does not submit or poll. An expectation is returned only when every required
        measurement for that circuit is valid and complete. The manifest is explicit,
        unsigned input, not proof of ownership; queries use the caller's current identity.
        """
        with tool_errors():
            return await service.query(manifest)

    for tool in (estimate_quantum_expectation, batch_estimate_quantum_expectation):
        mcp.tool(
            annotations=ToolAnnotations(
                readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
            )
        )(tool)
    mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
        )
    )(get_quantum_expectation_result)
