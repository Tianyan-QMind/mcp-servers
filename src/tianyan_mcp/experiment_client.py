from __future__ import annotations

from typing import cast

import httpx
from pydantic import ValidationError

from tianyan_mcp.errors import ServiceError, rejection_message
from tianyan_mcp.experiment_models import UpstreamExperimentResult, UpstreamExperimentSubmission
from tianyan_mcp.sdk_auth import AccessTokenProvider

SUBMIT_URL = "https://qc.zdxlz.com/qccp-quantum/sdk/experiment/submit"
STATUS_URL = "https://qc.zdxlz.com/qccp-quantum/sdk/experiment/result/findStatus"
RESULT_URL = "https://qc.zdxlz.com/qccp-quantum/sdk/experiment/result/find"
UNKNOWN_MESSAGE = "天衍云可能已经接收实验，请先在平台中核查，不要直接重复提交"
SUBMISSION_REJECTED_MESSAGE = "天衍云拒绝了实验提交"


class TianyanExperimentClient:
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

    async def submit(self, request: UpstreamExperimentSubmission) -> str:
        return (await self.submit_batch(request))[0]

    async def submit_batch(self, request: UpstreamExperimentSubmission) -> tuple[str, ...]:
        token = await self._token_provider.get_access_token()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = await client.post(
                    SUBMIT_URL,
                    json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
                    headers={"Authorization": f"Bearer {token}"},
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False) from exc

        if response.status_code in {401, 403}:
            self._token_provider.invalidate()
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        if response.status_code >= 500:
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        if not 200 <= response.status_code < 300:
            raise ServiceError(
                "SUBMISSION_REJECTED",
                rejection_message(SUBMISSION_REJECTED_MESSAGE, _json_object(response)),
                False,
            )
        try:
            raw: object = response.json()
        except ValueError as exc:
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False) from exc
        if not isinstance(raw, dict):
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        payload = cast(dict[str, object], raw)
        code = payload.get("code")
        if type(code) is not int:
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        if code != 0 or payload.get("success") is not True:
            raise ServiceError(
                "SUBMISSION_REJECTED",
                rejection_message(SUBMISSION_REJECTED_MESSAGE, payload),
                False,
            )
        data = payload.get("data")
        raw_query_ids = (
            cast(dict[str, object], data).get("query_ids") if isinstance(data, dict) else None
        )
        if not isinstance(raw_query_ids, list):
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        query_ids = cast(list[object], raw_query_ids)
        if len(query_ids) != len(request.circuit):
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        if any(not isinstance(query_id, str) or not query_id.strip() for query_id in query_ids):
            raise ServiceError("SUBMISSION_OUTCOME_UNKNOWN", UNKNOWN_MESSAGE, False)
        return tuple(cast(str, query_id).strip() for query_id in query_ids)

    async def get_status(self, query_id: str) -> int:
        data = await self._post_query_json(STATUS_URL, query_id)
        if not isinstance(data, dict):
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False)
        payload_data = cast(dict[str, object], data)
        status = payload_data.get("status")
        if type(status) is not int or status not in {0, 1}:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验状态无效", False)
        return status

    async def get_result(self, query_id: str) -> tuple[UpstreamExperimentResult, ...]:
        data = await self._post_query_json(RESULT_URL, query_id)
        if not isinstance(data, dict):
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False)
        payload_data = cast(dict[str, object], data)
        raw_results = payload_data.get("experimentResultModelList")
        if not isinstance(raw_results, list) or not raw_results:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验结果无效", False)
        try:
            results = cast(list[object], raw_results)
            return tuple(UpstreamExperimentResult.model_validate(result) for result in results)
        except ValidationError as exc:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验结果无效", False) from exc

    async def _post_query_json(self, url: str, query_id: str) -> object:
        token = await self._token_provider.get_access_token()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
                headers={"Accept": "application/json"},
            ) as client:
                response = await client.post(
                    url,
                    json={"query_ids": [query_id]},
                    headers={"Authorization": f"Bearer {token}"},
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise ServiceError(
                "EXPERIMENT_QUERY_UNAVAILABLE", "天衍云实验查询服务不可用", True
            ) from exc

        if response.status_code in {401, 403}:
            self._token_provider.invalidate()
            raise ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)
        if not 200 <= response.status_code < 300:
            raise ServiceError("EXPERIMENT_QUERY_REJECTED", "天衍云拒绝了实验查询", False)
        try:
            raw: object = response.json()
        except ValueError as exc:
            raise ServiceError(
                "UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False
            ) from exc
        if not isinstance(raw, dict):
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False)
        payload = cast(dict[str, object], raw)
        code = payload.get("code")
        if type(code) is not int:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False)
        success = payload.get("success")
        if type(success) is not bool:
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False)
        if code != 0 or success is not True:
            raise ServiceError("EXPERIMENT_QUERY_REJECTED", "天衍云拒绝了实验查询", False)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ServiceError("UPSTREAM_INVALID_RESPONSE", "天衍云实验查询响应无效", False)
        return cast(dict[str, object], data)


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
