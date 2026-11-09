from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

import httpx
from pydantic import ValidationError

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.models import BackendSnapshot, UpstreamBackend

TIANYAN_URL = "https://qc.zdxlz.com/qccp-quantum/experiments/quantum/computer/by/user"
TIANYAN_QUERY = {"isToll": "", "labels": "", "apiCode": "byUser"}
RETRYABLE_STATUS = {502, 503, 504}
LOGGER = logging.getLogger(__name__)


class TianyanApiClient:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = timeout_seconds

    async def fetch_backends(self) -> BackendSnapshot:
        response = await self._request_with_retry()
        payload = self._parse_payload(response)
        raw_data_value = payload.get("data")
        if not isinstance(raw_data_value, list):
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE",
                "天衍云响应 data 不是数组",
                False,
            )
        raw_data = cast(list[Any], raw_data_value)

        backends: list[UpstreamBackend] = []
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                LOGGER.warning("event=skip_invalid_backend index=%s", index)
                continue
            try:
                backend = UpstreamBackend.model_validate(item)
            except ValidationError:
                LOGGER.warning("event=skip_invalid_backend index=%s", index)
                continue
            if (
                backend.id is None
                or backend.code is None
                or backend.labels is None
                or backend.status is None
            ):
                LOGGER.warning("event=skip_backend_missing_key index=%s", index)
                continue
            backends.append(backend)

        if raw_data and not backends:
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE",
                "天衍云响应不包含有效资源",
                False,
            )
        return BackendSnapshot(fetched_at=datetime.now(UTC), backends=backends)

    async def _request_with_retry(self) -> httpx.Response:
        headers = {
            "Accept": "application/json",
            "User-Agent": "tianyan-mcp-server/0.1.0",
        }
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    follow_redirects=False,
                    headers=headers,
                ) as client:
                    response = await client.get(TIANYAN_URL, params=TIANYAN_QUERY)
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    await asyncio.sleep(0.2)
                    continue
                raise ServiceError("UPSTREAM_TIMEOUT", "天衍云请求超时", True) from exc
            except httpx.RequestError as exc:
                if attempt == 0:
                    await asyncio.sleep(0.2)
                    continue
                raise ServiceError("UPSTREAM_UNAVAILABLE", "无法连接天衍云接口", True) from exc

            if response.status_code in RETRYABLE_STATUS and attempt == 0:
                await asyncio.sleep(0.2)
                continue
            if response.status_code in RETRYABLE_STATUS:
                raise ServiceError(
                    "UPSTREAM_UNAVAILABLE",
                    f"天衍云接口返回 HTTP {response.status_code}",
                    True,
                )
            if response.status_code != 200:
                raise ServiceError(
                    "UPSTREAM_REJECTED",
                    f"天衍云接口返回 HTTP {response.status_code}",
                    False,
                )
            return response
        raise AssertionError("retry loop did not return or raise")

    @staticmethod
    def _parse_payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE", "天衍云响应不是有效 JSON", False
            ) from exc
        if not isinstance(payload, dict):
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云业务状态异常", False)
        typed_payload = cast(dict[str, Any], payload)
        if typed_payload.get("code") != 200:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云业务状态异常", False)
        return typed_payload
