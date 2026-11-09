from __future__ import annotations

from typing import cast

import httpx
from pydantic import ValidationError

from tianyan_mcp.errors import (
    ServiceError,
    rejection_message,
    upstream_business_message,
)
from tianyan_mcp.sdk_auth import AccessTokenProvider
from tianyan_mcp.waveform_models import (
    UpstreamWaveformDiagram,
    UpstreamWaveformSubmission,
    UpstreamWaveformTask,
)

GENERATE_URL = "https://qc.zdxlz.com/qccp-quantum/sdk/generateWaveformDiagram"
DIAGRAM_URL = "https://qc.zdxlz.com/qccp-quantum/sdk/getWaveformDiagram"
SUCCESS_CODE = 0
UNKNOWN_GENERATION_MESSAGE = "天衍云可能已经接收波形生成任务，请先在平台中核查，不要直接重复提交"
WAVEFORM_SUBMISSION_REJECTED_MESSAGE = "天衍云拒绝了波形生成任务"

_BUSINESS_ERROR_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("id不能为空", "INVALID_WAVEFORM_TASK_ID", "波形任务 id 无效"),
    ("未查询到", "WAVEFORM_TASK_NOT_FOUND", "未查询到波形线路数据"),
    ("不是该用户", "WAVEFORM_TASK_FORBIDDEN", "不是该用户的波形图"),
    ("生成失败", "WAVEFORM_GENERATION_FAILED", "线路波形图生成失败"),
)


class TianyanWaveformClient:
    def __init__(
        self,
        token_provider: AccessTokenProvider,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._timeout = timeout_seconds
        self._transport = transport

    async def generate(self, request: UpstreamWaveformSubmission) -> UpstreamWaveformTask:
        token = await self._token_provider.get_access_token()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = await client.post(
                    GENERATE_URL,
                    json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
                    headers={"Authorization": f"Bearer {token}"},
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ServiceError(
                "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
            ) from exc

        if response.status_code in {401, 403}:
            self._token_provider.invalidate()
            raise ServiceError(
                "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
            )
        if response.status_code >= 500:
            raise ServiceError(
                "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
            )
        if not 200 <= response.status_code < 300:
            raise ServiceError(
                "WAVEFORM_SUBMISSION_REJECTED",
                rejection_message(WAVEFORM_SUBMISSION_REJECTED_MESSAGE, _json_object(response)),
                False,
            )
        payload = self._read_envelope(response, unknown_outcome=True)
        if payload["code"] != SUCCESS_CODE or payload.get("success") is False:
            raise ServiceError(
                "WAVEFORM_SUBMISSION_REJECTED",
                rejection_message(WAVEFORM_SUBMISSION_REJECTED_MESSAGE, payload),
                False,
            )
        if payload.get("success") is not True:
            # code == 0 but without an explicit success flag on a write request is
            # ambiguous: the task may have been created, so never signal retry.
            raise ServiceError(
                "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ServiceError(
                "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
            )
        try:
            return UpstreamWaveformTask.model_validate(data)
        except ValidationError as exc:
            raise ServiceError(
                "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
            ) from exc

    async def get_diagram(self, task_id: str) -> UpstreamWaveformDiagram:
        token = await self._token_provider.get_access_token()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = await client.get(
                    DIAGRAM_URL,
                    params={"id": task_id},
                    headers={"Authorization": f"Bearer {token}"},
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ServiceError(
                "WAVEFORM_QUERY_UNAVAILABLE", "天衍云波形图查询服务不可用", True
            ) from exc

        if response.status_code in {401, 403}:
            self._token_provider.invalidate()
            raise ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)
        if not 200 <= response.status_code < 300:
            raise ServiceError("WAVEFORM_QUERY_REJECTED", "天衍云拒绝了波形图查询", False)
        payload = self._read_envelope(response, unknown_outcome=False)
        if payload["code"] != SUCCESS_CODE or payload.get("success") is False:
            raise self._business_error(payload)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云波形图查询响应无效", False)
        try:
            return UpstreamWaveformDiagram.model_validate(data)
        except ValidationError as exc:
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE", "天衍云波形图查询响应无效", False
            ) from exc

    @staticmethod
    def _read_envelope(response: httpx.Response, *, unknown_outcome: bool) -> dict[str, object]:
        try:
            raw: object = response.json()
        except ValueError as exc:
            if unknown_outcome:
                raise ServiceError(
                    "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
                ) from exc
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE", "天衍云波形图查询响应无效", False
            ) from exc
        if not isinstance(raw, dict):
            if unknown_outcome:
                raise ServiceError(
                    "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
                )
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云波形图查询响应无效", False)
        payload = cast(dict[str, object], raw)
        if type(payload.get("code")) is not int:
            if unknown_outcome:
                raise ServiceError(
                    "WAVEFORM_SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_GENERATION_MESSAGE, False
                )
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云波形图查询响应无效", False)
        return payload

    @staticmethod
    def _business_error(payload: dict[str, object]) -> ServiceError:
        message = _extract_error_text(payload)
        for marker, code, stable_message in _BUSINESS_ERROR_MARKERS:
            if marker in message:
                return ServiceError(code, stable_message, False)
        return ServiceError("WAVEFORM_QUERY_REJECTED", "天衍云拒绝了波形图查询", False)


def _json_object(response: httpx.Response) -> dict[str, object] | None:
    """Return a parsed JSON object body, or None for non-JSON/non-object bodies.

    Used only to surface structured business error text on explicit HTTP
    rejections; raw response bodies are never returned to callers.
    """
    try:
        raw: object = response.json()
    except ValueError:
        return None
    return cast(dict[str, object], raw) if isinstance(raw, dict) else None


def _extract_error_text(payload: dict[str, object]) -> str:
    """Collect human-readable text from the upstream error envelope.

    Live upstreams emit errors either in ``message`` or in ``msg``; ``msg`` is
    frequently a JSON-encoded ``{"CN": ..., "EN": ...}`` string.
    """
    return upstream_business_message(payload)
