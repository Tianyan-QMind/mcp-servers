from __future__ import annotations

import re
from typing import Protocol

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.waveform_models import (
    UpstreamWaveformDiagram,
    UpstreamWaveformSubmission,
    UpstreamWaveformTask,
    WaveformDiagramResult,
    WaveformGenerationResult,
)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\- \u4e00-\u9fff]+$")
QC_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
TASK_STATUS_LABELS = ("generating", "completed", "failed")


class WaveformGenerator(Protocol):
    async def generate(self, request: UpstreamWaveformSubmission) -> UpstreamWaveformTask: ...


class WaveformDiagramReader(Protocol):
    async def get_diagram(self, task_id: str) -> UpstreamWaveformDiagram: ...


class WaveformService:
    def __init__(self, generator: WaveformGenerator, reader: WaveformDiagramReader) -> None:
        self._generator = generator
        self._reader = reader

    async def generate_waveform(
        self,
        *,
        qc_code: str,
        circuit: str,
        circuit_name: str | None,
        is_verify: bool,
    ) -> WaveformGenerationResult:
        clean_qc_code = qc_code.strip()
        if not 1 <= len(clean_qc_code) <= 64 or QC_CODE_PATTERN.fullmatch(clean_qc_code) is None:
            raise ServiceError("INVALID_WAVEFORM_ARGUMENT", "qc_code 格式无效", False)
        if not circuit.strip() or len(circuit) > 20_000 or "\x00" in circuit:
            raise ServiceError("INVALID_WAVEFORM_ARGUMENT", "QCIS 线路内容无效", False)
        clean_name = self._clean_name(circuit_name)

        request = UpstreamWaveformSubmission(
            circuit=circuit,
            qcCode=clean_qc_code,
            circuitName=clean_name,
            isVerify="true" if is_verify else "false",
        )
        try:
            task = await self._generator.generate(request)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("INTERNAL_ERROR", "服务内部错误", False) from exc

        return WaveformGenerationResult(
            id=task.id,
            qc_code=clean_qc_code,
            circuit_name=task.circuit_name,
            circuit_content=task.circuit_content,
            status=TASK_STATUS_LABELS[task.status],
            upstream_status=task.status,
            create_time=task.create_time,
            visible_url=task.visible_url,
            measure_qubits=self._parse_measure_qubits(task.measure_qubit_list),
        )

    async def get_waveform(self, task_id: str) -> WaveformDiagramResult:
        clean_id = task_id.strip()
        if len(clean_id) > 32 or not clean_id.isascii() or not clean_id.isdigit():
            raise ServiceError("INVALID_WAVEFORM_TASK_ID", "波形任务 id 必须是数字字符串", False)

        try:
            diagram = await self._reader.get_diagram(clean_id)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("INTERNAL_ERROR", "服务内部错误", False) from exc

        if type(diagram.visible_url) is not str:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云波形图查询响应无效", False)
        if diagram.visible_url == "":
            # Live upstreams normally signal render failure via a code=500 envelope
            # (already classified in the client); status=2 inside data is the
            # documented in-band equivalent, e.g. when the machine-side compiler
            # rejects pulse-level gates.
            if diagram.status == 2:
                raise ServiceError("WAVEFORM_GENERATION_FAILED", "线路波形图生成失败", False)
            return WaveformDiagramResult(
                id=diagram.id,
                circuit_name=diagram.circuit_name,
                status="generating",
                visible_url=None,
            )
        return WaveformDiagramResult(
            id=diagram.id,
            circuit_name=diagram.circuit_name,
            status="completed",
            visible_url=diagram.visible_url,
        )

    @staticmethod
    def _clean_name(circuit_name: str | None) -> str | None:
        if circuit_name is None:
            return None
        cleaned = circuit_name.strip()
        if not cleaned:
            return None
        if len(cleaned) > 100 or NAME_PATTERN.fullmatch(cleaned) is None:
            raise ServiceError("INVALID_WAVEFORM_ARGUMENT", "circuit_name 格式无效", False)
        return cleaned

    @staticmethod
    def _parse_measure_qubits(raw: str | None) -> tuple[str, ...]:
        if raw is None or not raw.strip():
            return ()
        return tuple(qubit.strip() for qubit in raw.split(",") if qubit.strip())
