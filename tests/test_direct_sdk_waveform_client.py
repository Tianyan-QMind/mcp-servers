from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_waveform_client import DirectSdkWaveformClient
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import DirectSdkPrincipal, direct_sdk_scope
from tianyan_mcp.waveform_client import DIAGRAM_URL, GENERATE_URL
from tianyan_mcp.waveform_models import UpstreamWaveformSubmission


class FakeLoginClient:
    def __init__(self, tokens: dict[str, str]) -> None:
        self._tokens = tokens

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        return SecretStr(self._tokens[open_id.get_secret_value()]), 720.0


def principal(open_id: str, fingerprint: bytes, request_id: str) -> DirectSdkPrincipal:
    return DirectSdkPrincipal(SecretStr(open_id), fingerprint, request_id)


def submission() -> UpstreamWaveformSubmission:
    return UpstreamWaveformSubmission(circuit="H Q0", qcCode="QPU-01")


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
    manager: DirectSdkAuthManager,
) -> DirectSdkWaveformClient:
    return DirectSdkWaveformClient(
        manager,
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )


def success_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "code": 0,
            "success": True,
            "message": None,
            "data": {
                "id": 1823156489274675200,
                "circuitName": "example1",
                "circuitContent": "H Q0",
                "status": 0,
                "measureQubitList": None,
            },
        },
    )


@pytest.mark.asyncio
async def test_generate_uses_current_principal_token_and_logs_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    open_id = "secret-waveform-open-id"
    token = "opaque-waveform-token"
    manager = DirectSdkAuthManager(FakeLoginClient({open_id: token}))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == GENERATE_URL
        assert request.headers["Authorization"] == f"Bearer {token}"
        return success_response()

    fingerprint = b"waveform-fingerprint!!"
    user = principal(open_id, fingerprint, "request-waveform")
    caplog.set_level(logging.INFO, logger="tianyan_mcp.direct_sdk_waveform_client")
    client = client_for(handler, manager)

    with direct_sdk_scope(user):
        task = await client.generate(submission())

    assert task.id == "1823156489274675200"
    assert open_id not in str(requests[0].content)
    expected_log = (
        "event=waveform_generate "
        f"request_id=request-waveform sdk_fingerprint={fingerprint.hex()[:12]} result=success"
    )
    assert expected_log in caplog.text
    assert open_id not in caplog.text
    assert token not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "result"),
    [
        (httpx.Response(400), "denied"),
        (httpx.Response(500), "unknown"),
    ],
    ids=["denied", "unknown"],
)
async def test_generate_logs_failure_outcomes(
    caplog: pytest.LogCaptureFixture, response: httpx.Response, result: str
) -> None:
    open_id = "secret-waveform-open-id"
    fingerprint = b"waveform-fingerprint!!"
    manager = DirectSdkAuthManager(FakeLoginClient({open_id: "opaque-waveform-token"}))
    user = principal(open_id, fingerprint, "request-waveform-log")
    caplog.set_level(logging.INFO, logger="tianyan_mcp.direct_sdk_waveform_client")

    with direct_sdk_scope(user):
        with pytest.raises(ServiceError):
            await client_for(lambda _: response, manager).generate(submission())

    expected_log = (
        "event=waveform_generate "
        f"request_id=request-waveform-log sdk_fingerprint={fingerprint.hex()[:12]} "
        f"result={result}"
    )
    assert expected_log in caplog.text


@pytest.mark.asyncio
async def test_get_diagram_uses_current_principal_token() -> None:
    alice_open_id = "secret-alice-open-id"
    bob_open_id = "secret-bob-open-id"
    tokens = {alice_open_id: "token-alice", bob_open_id: "token-bob"}
    manager = DirectSdkAuthManager(FakeLoginClient(tokens))
    seen_tokens: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(DIAGRAM_URL)
        seen_tokens.append(request.headers["Authorization"])
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {"id": "123", "circuit_name": "n", "visibleUrl": ""},
            },
        )

    client = client_for(handler, manager)
    alice = principal(alice_open_id, b"alice-fingerprint1", "request-alice")
    bob = principal(bob_open_id, b"bob-fingerprint12", "request-bob")

    with direct_sdk_scope(alice):
        alice_diagram = await client.get_diagram("123")
    with direct_sdk_scope(bob):
        bob_diagram = await client.get_diagram("123")

    assert alice_diagram.visible_url == ""
    assert bob_diagram.id == "123"
    assert seen_tokens == ["Bearer token-alice", "Bearer token-bob"]


class CountingLoginClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.logins = 0

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        self.logins += 1
        return SecretStr(self.token), 720.0


@pytest.mark.asyncio
async def test_unauthorized_response_invalidates_cached_token() -> None:
    open_id = "secret-waveform-open-id"
    login = CountingLoginClient("opaque-waveform-token")
    manager = DirectSdkAuthManager(login)
    user = principal(open_id, b"waveform-fingerprint!!", "request-waveform-401")
    statuses: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(DIAGRAM_URL)
        status = 401 if not statuses else 200
        statuses.append(status)
        if status == 401:
            return httpx.Response(401)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "message": None,
                "data": {"id": "123", "circuit_name": "n", "visibleUrl": ""},
            },
        )

    client = client_for(handler, manager)

    with direct_sdk_scope(user):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await client.get_diagram("123")
        diagram = await client.get_diagram("123")

    assert login.logins == 2
    assert diagram.id == "123"
