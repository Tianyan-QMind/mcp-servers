"""Pydantic models for cqlib compilation and optimization MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CommutationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commutes: bool
    exact: bool
    phase: str | None = None
    method: Literal["check", "algebraic"]


class CompileForDeviceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    circuit: str = Field(min_length=1, max_length=200_000)
    changed: bool
    mode: str
    initial_layout: dict[int, int] | None = None
    final_layout: dict[int, int] | None = None


class CompileWithTargetResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    circuit: str = Field(min_length=1, max_length=200_000)
    changed: bool
    mode: str
    target_kind: Literal["logical", "basis", "device", "topology_basis"]
    initial_layout: dict[int, int] | None = None
    final_layout: dict[int, int] | None = None


class DecomposeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    circuit: str = Field(min_length=1, max_length=200_000)
    changed: bool
    transform: str


class NativeOptimizationSummary(BaseModel):
    """Native-gate statistics for one side of a NativeOptimizer run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native_total_ops: int
    native_two_qubit_ops: int
    native_two_qubit_depth: int
    total_native_depth: int
    predicted_log_error: float | None = None
    imputed_error_count: int
    unavailable_error_count: int


class NativeOptimizationStats(BaseModel):
    """Before/after native-gate statistics reported only by native_optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    before: NativeOptimizationSummary
    after: NativeOptimizationSummary
    rounds: int
    restored_best: bool


class OptimizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    circuit: str = Field(min_length=1, max_length=200_000)
    changed: bool
    pass_name: str
    summary: NativeOptimizationStats | None = None
