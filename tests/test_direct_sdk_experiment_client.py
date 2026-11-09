from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager
from tianyan_mcp.direct_sdk_experiment_client import DirectSdkExperimentClient
from tianyan_mcp.errors import ServiceError
from tianyan_mcp.experiment_client import RESULT_URL, STATUS_URL
from tianyan_mcp.experiment_models import UpstreamExperimentSubmission
from tianyan_mcp.identity import DirectSdkPrincipal, direct_sdk_scope


class FakeLoginClient:
    def __init__(
        self,
        tokens: dict[str, str],
        *,
        expected_logins: int | None = None,
        all_logins_started: asyncio.Event | None = None,
        release_logins: asyncio.Event | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._tokens = tokens
        self._expected_logins = expected_logins
        self._all_logins_started = all_logins_started
        self._release_logins = release_logins

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        value = open_id.get_secret_value()
        self.calls.append(value)
        if self._all_logins_started is not None and len(self.calls) == self._expected_logins:
            self._all_logins_started.set()
        if self._release_logins is not None:
            await self._release_logins.wait()
        return SecretStr(self._tokens[value]), 720.0


def principal(open_id: str, fingerprint: bytes, request_id: str) -> DirectSdkPrincipal:
    return DirectSdkPrincipal(SecretStr(open_id), fingerprint, request_id)


def submission(computer_code: str = "tianyan_sw") -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("SECRET-QCIS-CIRCUIT",),
        name="direct-sdk-test",
        lab_name="direct-sdk-lab",
        computerCode=computer_code,
    )


def batch_submission() -> UpstreamExperimentSubmission:
    return UpstreamExperimentSubmission(
        shots=100,
        circuit=("SECRET-QCIS-CIRCUIT-1", "SECRET-QCIS-CIRCUIT-2"),
        name="direct-sdk-batch-test",
        computerCode="tianyan_sw",
    )


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
    manager: DirectSdkAuthManager,
) -> DirectSdkExperimentClient:
    return DirectSdkExperimentClient(
        manager,
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )


def assert_request_excludes_open_id(request: httpx.Request, open_id: str) -> None:
    assert open_id not in str(request.url)
    assert all(open_id not in value for value in request.headers.values())
    assert open_id.encode() not in request.content


async def submit_as(
    client: DirectSdkExperimentClient,
    user: DirectSdkPrincipal,
    request: UpstreamExperimentSubmission,
) -> str:
    with direct_sdk_scope(user):
        return await client.submit(request)


async def submit_batch_as(
    client: DirectSdkExperimentClient,
    user: DirectSdkPrincipal,
    request: UpstreamExperimentSubmission,
) -> tuple[str, ...]:
    with direct_sdk_scope(user):
        return await client.submit_batch(request)


@pytest.mark.asyncio
async def test_batch_submission_uses_current_principal_token_and_returns_all_ids() -> None:
    open_id = "secret-batch-open-id"
    token = "opaque-sdk-token-batch"
    manager = DirectSdkAuthManager(FakeLoginClient({open_id: token}))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert_request_excludes_open_id(request, open_id)
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["query-1", "query-2"]}},
        )

    user = principal(open_id, b"batch-fingerprint", "request-batch")
    assert await submit_batch_as(client_for(handler, manager), user, batch_submission()) == (
        "query-1",
        "query-2",
    )


@pytest.mark.asyncio
async def test_result_queries_use_current_principal_token_once_without_leaking_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    open_id = "secret-query-open-id"
    token = "opaque-query-access-token"
    login = FakeLoginClient({open_id: token})
    manager = DirectSdkAuthManager(login)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert_request_excludes_open_id(request, open_id)
        if str(request.url) == STATUS_URL:
            return httpx.Response(
                200,
                json={"code": 0, "success": True, "data": {"status": 0}},
            )
        assert str(request.url) == RESULT_URL
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {
                    "experimentResultModelList": [
                        {
                            "resultStatus": [[0]],
                            "probability": {"0": 1.0},
                            "experimentTaskId": "123",
                        }
                    ]
                },
            },
        )

    user = principal(open_id, bytes.fromhex("0123456789abcdef" * 4), "request-query")
    client = client_for(handler, manager)
    caplog.set_level(logging.INFO, logger="tianyan_mcp.direct_sdk_experiment_client")

    with direct_sdk_scope(user):
        assert await client.get_status("123") == 0
        results = await client.get_result("123")

    assert results[0].experiment_task_id == "123"
    assert [str(request.url) for request in requests] == [STATUS_URL, RESULT_URL]
    assert login.calls == [open_id]
    assert open_id not in caplog.text
    assert token not in caplog.text


@pytest.mark.asyncio
async def test_current_open_id_is_used_only_for_login_and_sdk_token_reaches_upstream() -> None:
    open_id = "secret-alice-open-id"
    token = "opaque-sdk-token-alpha"
    login = FakeLoginClient({open_id: token})
    manager = DirectSdkAuthManager(login)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert_request_excludes_open_id(request, open_id)
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["query-alice"]}},
        )

    user = principal(open_id, b"alice-fingerprint", "request-alice")
    assert await submit_as(client_for(handler, manager), user, submission()) == "query-alice"
    assert login.calls == [open_id]


@pytest.mark.asyncio
async def test_concurrent_principals_keep_login_and_submission_tokens_isolated() -> None:
    alice_open_id = "secret-alice-open-id"
    bob_open_id = "secret-bob-open-id"
    expected_tokens = {
        alice_open_id: "opaque-sdk-token-alpha",
        bob_open_id: "opaque-sdk-token-beta",
    }
    both_logins_started = asyncio.Event()
    release_logins = asyncio.Event()
    login = FakeLoginClient(
        expected_tokens,
        expected_logins=2,
        all_logins_started=both_logins_started,
        release_logins=release_logins,
    )
    manager = DirectSdkAuthManager(login)
    token_to_query_id = {
        f"Bearer {expected_tokens[alice_open_id]}": "query-alice",
        f"Bearer {expected_tokens[bob_open_id]}": "query-bob",
    }
    received_tokens: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["Authorization"]
        received_tokens.append(authorization)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "success": True,
                "data": {"query_ids": [token_to_query_id[authorization]]},
            },
        )

    client = client_for(handler, manager)
    alice = principal(alice_open_id, b"alice-fingerprint", "request-alice")
    bob = principal(bob_open_id, b"bob-fingerprint", "request-bob")

    submissions = asyncio.gather(
        submit_as(client, alice, submission()), submit_as(client, bob, submission())
    )
    await both_logins_started.wait()
    assert set(login.calls) == {alice_open_id, bob_open_id}
    release_logins.set()
    assert await submissions == ["query-alice", "query-bob"]
    assert set(received_tokens) == {
        f"Bearer {expected_tokens[alice_open_id]}",
        f"Bearer {expected_tokens[bob_open_id]}",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_auth_failure_invalidates_only_current_fingerprint(status_code: int) -> None:
    alice_token = "opaque-sdk-token-alpha"
    bob_token = "opaque-sdk-token-beta"
    login = FakeLoginClient(
        {
            "secret-alice-open-id": alice_token,
            "secret-bob-open-id": bob_token,
        }
    )
    manager = DirectSdkAuthManager(login)
    alice = principal("secret-alice-open-id", b"alice-fingerprint", "request-alice")
    bob = principal("secret-bob-open-id", b"bob-fingerprint", "request-bob")
    assert await manager.get_access_token(alice) == alice_token
    assert await manager.get_access_token(bob) == bob_token
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    with pytest.raises(ServiceError, match="SUBMISSION_OUTCOME_UNKNOWN"):
        await submit_as(client_for(handler, manager), alice, submission())

    assert calls == 1
    assert alice.sdk_fingerprint not in manager._tokens  # pyright: ignore[reportPrivateUsage]
    assert bob.sdk_fingerprint in manager._tokens  # pyright: ignore[reportPrivateUsage]
    assert await manager.get_access_token(alice) == alice_token
    assert await manager.get_access_token(bob) == bob_token
    assert login.calls == [
        "secret-alice-open-id",
        "secret-bob-open-id",
        "secret-alice-open-id",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "result"),
    [
        (
            httpx.Response(
                200,
                json={"code": 0, "success": True, "data": {"query_ids": ["query-ok"]}},
            ),
            "success",
        ),
        (httpx.Response(400), "denied"),
        (httpx.Response(500), "unknown"),
    ],
    ids=["success", "denied", "unknown"],
)
async def test_submission_logs_are_structured_and_secret_free(
    caplog: pytest.LogCaptureFixture,
    response: httpx.Response,
    result: str,
) -> None:
    open_id = "secret-log-open-id"
    token = "opaque-sdk-token-log"
    circuit = "SECRET-QCIS-CIRCUIT"
    fingerprint = bytes.fromhex("0123456789abcdef" * 4)
    user = principal(open_id, fingerprint, "request-log")
    manager = DirectSdkAuthManager(FakeLoginClient({open_id: token}))
    caplog.set_level(logging.INFO, logger="tianyan_mcp.direct_sdk_experiment_client")
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    client = client_for(handler, manager)
    if result == "success":
        assert await submit_as(client, user, submission()) == "query-ok"
    else:
        with pytest.raises(ServiceError):
            await submit_as(client, user, submission())

    assert calls == 1
    expected_log = (
        "event=experiment_submit "
        f"request_id=request-log sdk_fingerprint={fingerprint.hex()[:12]} result={result}"
    )
    assert expected_log in caplog.text
    assert open_id not in caplog.text
    assert token not in caplog.text
    assert circuit not in caplog.text


@pytest.mark.asyncio
async def test_adapter_does_not_reject_hardware_submission() -> None:
    open_id = "secret-hardware-open-id"
    token = "opaque-sdk-token-hardware"
    login = FakeLoginClient({open_id: token})
    manager = DirectSdkAuthManager(login)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Authorization"] == f"Bearer {token}"
        return httpx.Response(
            200,
            json={"code": 0, "success": True, "data": {"query_ids": ["hardware-query"]}},
        )

    user = principal(open_id, b"hardware-fingerprint", "request-hardware")
    assert (
        await submit_as(client_for(handler, manager), user, submission("tianyan176"))
        == "hardware-query"
    )
    assert calls == 1
