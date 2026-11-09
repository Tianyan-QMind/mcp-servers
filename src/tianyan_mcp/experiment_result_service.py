from __future__ import annotations

from typing import Protocol

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_models import (
    ExperimentResultQueryResult,
    UpstreamExperimentResult,
)


class ExperimentResultReader(Protocol):
    async def get_status(self, query_id: str) -> int: ...

    async def get_result(self, query_id: str) -> tuple[UpstreamExperimentResult, ...]: ...


class ExperimentResultService:
    def __init__(self, reader: ExperimentResultReader) -> None:
        self._reader = reader

    async def query(
        self, query_id: str, include_result_status: bool
    ) -> ExperimentResultQueryResult:
        clean_id = query_id.strip()
        if len(clean_id) > 128 or not clean_id.isascii() or not clean_id.isdigit():
            raise ServiceError("INVALID_EXPERIMENT_QUERY_ID", "query_id 必须是数字字符串", False)

        try:
            status = await self._reader.get_status(clean_id)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("INTERNAL_ERROR", "服务内部错误", False) from exc

        if type(status) is not int or status not in {0, 1}:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验状态响应无效", False)
        if status == 1:
            return ExperimentResultQueryResult(
                query_id=clean_id, status="processing", upstream_status=1
            )

        try:
            results = await self._reader.get_result(clean_id)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("INTERNAL_ERROR", "服务内部错误", False) from exc

        try:
            result_count = len(results)
        except Exception as exc:
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE", "天衍云实验结果响应无效", False
            ) from exc
        if result_count != 1:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验结果响应无效", False)
        item = results[0]
        try:
            return ExperimentResultQueryResult(
                query_id=clean_id,
                status="completed",
                upstream_status=0,
                probability=dict(item.probability),
                experiment_task_id=item.experiment_task_id,
                result_status=item.result_status if include_result_status else None,
            )
        except Exception as exc:
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE", "天衍云实验结果响应无效", False
            ) from exc
