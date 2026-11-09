from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.identity import DirectSdkPrincipal

if TYPE_CHECKING:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager


def principal(open_id: str, fingerprint: bytes) -> DirectSdkPrincipal:
    return DirectSdkPrincipal(
        sdk_open_id=SecretStr(open_id), sdk_fingerprint=fingerprint, request_id="test-request"
    )


class FakeLoginClient:
    def __init__(
        self,
        login: Callable[[SecretStr], Awaitable[tuple[SecretStr, float]]] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._login = login

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        self.calls.append(open_id.get_secret_value())
        if self._login is not None:
            return await self._login(open_id)
        return SecretStr(f"token-{len(self.calls)}"), 720.0


@pytest.mark.asyncio
async def test_cache_hit_reuses_token_without_login() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    login = FakeLoginClient()
    manager = DirectSdkAuthManager(login, clock=lambda: 100.0)
    user = principal("fake-open-id", b"fingerprint")

    assert await manager.get_access_token(user) == "token-1"
    assert await manager.get_access_token(user) == "token-1"
    assert login.calls == ["fake-open-id"]


@pytest.mark.asyncio
async def test_same_fingerprint_callers_share_one_inflight_login() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_login(_: SecretStr) -> tuple[SecretStr, float]:
        started.set()
        await release.wait()
        return SecretStr("shared-token"), 720.0

    login = FakeLoginClient(delayed_login)
    manager = DirectSdkAuthManager(login)
    user = principal("fake-open-id", b"fingerprint")

    pending = [asyncio.create_task(manager.get_access_token(user)) for _ in range(2)]
    await started.wait()
    await asyncio.sleep(0)
    assert login.calls == ["fake-open-id"]
    release.set()

    assert await asyncio.gather(*pending) == ["shared-token", "shared-token"]


@pytest.mark.asyncio
async def test_different_fingerprints_can_login_concurrently() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0

    async def delayed_login(open_id: SecretStr) -> tuple[SecretStr, float]:
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await release.wait()
        return SecretStr(f"token-{open_id.get_secret_value()}"), 720.0

    login = FakeLoginClient(delayed_login)
    manager = DirectSdkAuthManager(login)

    pending = asyncio.gather(
        manager.get_access_token(principal("fake-alice", b"alice")),
        manager.get_access_token(principal("fake-bob", b"bob")),
    )
    await both_started.wait()
    release.set()

    assert set(await pending) == {"token-fake-alice", "token-fake-bob"}


@pytest.mark.asyncio
async def test_exact_fingerprint_isolates_cached_identities() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    login = FakeLoginClient()
    manager = DirectSdkAuthManager(login, clock=lambda: 100.0)
    first = principal("fake-open-id", b"fingerprint-one")
    second = principal("fake-open-id", b"fingerprint-two")

    assert await manager.get_access_token(first) == "token-1"
    assert await manager.get_access_token(second) == "token-2"
    assert await manager.get_access_token(first) == "token-1"
    assert login.calls == ["fake-open-id", "fake-open-id"]


@pytest.mark.asyncio
async def test_expired_token_is_removed_before_refresh() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    now = 100.0
    login = FakeLoginClient()
    manager = DirectSdkAuthManager(login, clock=lambda: now)
    user = principal("fake-open-id", b"fingerprint")

    assert await manager.get_access_token(user) == "token-1"
    now = 760.0
    assert manager._valid_cached(user.sdk_fingerprint) is None  # pyright: ignore[reportPrivateUsage]
    assert user.sdk_fingerprint not in manager._tokens  # pyright: ignore[reportPrivateUsage]
    assert await manager.get_access_token(user) == "token-2"


@pytest.mark.asyncio
async def test_login_errors_are_propagated_and_not_cached() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    async def failing_login(_: SecretStr) -> tuple[SecretStr, float]:
        raise RuntimeError("fake login failure")

    manager = DirectSdkAuthManager(FakeLoginClient(failing_login))
    user = principal("fake-open-id", b"fingerprint")

    with pytest.raises(RuntimeError, match="fake login failure"):
        await manager.get_access_token(user)
    assert manager._tokens == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_cancelled_login_cleans_up_per_fingerprint_lock() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_login(_: SecretStr) -> tuple[SecretStr, float]:
        started.set()
        await release.wait()
        return SecretStr("fresh-token"), 720.0

    login = FakeLoginClient(delayed_login)
    manager = DirectSdkAuthManager(login)
    user = principal("fake-open-id", b"fingerprint")
    pending = asyncio.create_task(manager.get_access_token(user))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert manager._locks == {}  # pyright: ignore[reportPrivateUsage]

    release.set()
    assert await manager.get_access_token(user) == "fresh-token"
    assert manager._locks == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_invalidate_removes_only_requested_fingerprint() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    login = FakeLoginClient()
    manager = DirectSdkAuthManager(login, clock=lambda: 100.0)
    first = principal("fake-first", b"first")
    second = principal("fake-second", b"second")

    assert await manager.get_access_token(first) == "token-1"
    assert await manager.get_access_token(second) == "token-2"
    manager.invalidate(first.sdk_fingerprint)
    assert await manager.get_access_token(first) == "token-3"
    assert await manager.get_access_token(second) == "token-2"


@pytest.mark.asyncio
async def test_rejects_get_access_token_from_a_non_owner_event_loop_and_thread() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    login = FakeLoginClient()
    manager = DirectSdkAuthManager(login, clock=lambda: 100.0)
    user = principal("fake-open-id", b"fingerprint")

    assert await manager.get_access_token(user) == "token-1"

    async def foreign_get() -> str:
        with pytest.raises(RuntimeError, match="owner event loop and thread"):
            await manager.get_access_token(user)
        return "rejected"

    assert await asyncio.to_thread(lambda: asyncio.run(foreign_get())) == "rejected"
    assert login.calls == ["fake-open-id"]
    assert user.sdk_fingerprint in manager._tokens  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_rejects_invalidate_from_a_non_owner_thread_before_cache_access() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    manager = DirectSdkAuthManager(FakeLoginClient(), clock=lambda: 100.0)
    user = principal("fake-open-id", b"fingerprint")

    assert await manager.get_access_token(user) == "token-1"

    def foreign_invalidate() -> str:
        with pytest.raises(RuntimeError, match="owner event loop and thread"):
            manager.invalidate(user.sdk_fingerprint)
        return "rejected"

    assert await asyncio.to_thread(foreign_invalidate) == "rejected"
    assert user.sdk_fingerprint in manager._tokens  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_cache_retains_secrets_without_open_id_or_plain_token_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    open_id = "fake-sensitive-open-id"
    token = "fake-sensitive-token"

    async def fixed_login(_: SecretStr) -> tuple[SecretStr, float]:
        return SecretStr(token), 720.0

    caplog.set_level(logging.DEBUG)
    manager = DirectSdkAuthManager(FakeLoginClient(fixed_login), clock=lambda: 100.0)
    user = principal(open_id, b"fingerprint")

    assert await manager.get_access_token(user) == token
    assert list(manager._tokens) == [user.sdk_fingerprint]  # pyright: ignore[reportPrivateUsage]
    cached = manager._tokens[user.sdk_fingerprint]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(cached.value, SecretStr)
    assert open_id not in repr(manager._tokens)  # pyright: ignore[reportPrivateUsage]
    assert token not in repr(manager._tokens)  # pyright: ignore[reportPrivateUsage]
    assert open_id not in caplog.text
    assert token not in caplog.text


@pytest.mark.asyncio
async def test_token_cache_is_lru_bounded_to_1024_successful_identities() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    manager = DirectSdkAuthManager(FakeLoginClient(), clock=lambda: 100.0)
    principals = [
        principal(f"fake-open-id-{index}", index.to_bytes(4, "big")) for index in range(1025)
    ]

    for user in principals:
        await manager.get_access_token(user)

    assert len(manager._tokens) == 1024  # pyright: ignore[reportPrivateUsage]
    assert principals[0].sdk_fingerprint not in manager._tokens  # pyright: ignore[reportPrivateUsage]
    assert principals[-1].sdk_fingerprint in manager._tokens  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_cache_hit_promotes_entry_before_lru_eviction() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    manager = DirectSdkAuthManager(FakeLoginClient(), clock=lambda: 100.0)
    principals = [
        principal(f"fake-open-id-{index}", index.to_bytes(4, "big")) for index in range(1025)
    ]

    for user in principals[:1024]:
        await manager.get_access_token(user)
    assert await manager.get_access_token(principals[0]) == "token-1"
    await manager.get_access_token(principals[-1])

    assert principals[0].sdk_fingerprint in manager._tokens  # pyright: ignore[reportPrivateUsage]
    assert principals[1].sdk_fingerprint not in manager._tokens  # pyright: ignore[reportPrivateUsage]
    assert principals[-1].sdk_fingerprint in manager._tokens  # pyright: ignore[reportPrivateUsage]


class ScriptedLoginClient:
    """Login client that replays scripted outcomes, repeating the last one."""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    async def login(self, open_id: SecretStr) -> tuple[SecretStr, float]:
        self.calls.append(open_id.get_secret_value())
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, ServiceError):
            raise outcome
        token, expires_in = outcome  # type: ignore[misc]
        return SecretStr(token), float(expires_in)  # type: ignore[arg-type]


_FAILED = ServiceError("SDK_AUTH_FAILED", "天衍云 SDK 登录失败", False)
_UNAVAILABLE = ServiceError("SDK_AUTH_UNAVAILABLE", "天衍云 SDK 登录服务不可用", True)


def windowed_manager(login: ScriptedLoginClient, now: list[float]) -> DirectSdkAuthManager:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    return DirectSdkAuthManager(
        login,
        clock=lambda: now[0],
        reject_ttl_seconds=60.0,
        max_verify_attempts=5,
    )


@pytest.mark.asyncio
async def test_verify_success_prewarms_token_for_later_get_access_token() -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    login = FakeLoginClient()
    manager = DirectSdkAuthManager(login, clock=lambda: 100.0)
    user = principal("fake-open-id", b"fingerprint")

    await manager.verify(user)
    assert await manager.get_access_token(user) == "token-1"
    assert login.calls == ["fake-open-id"]


@pytest.mark.asyncio
async def test_verify_reuses_positive_token_cache_without_login() -> None:
    login = ScriptedLoginClient([("shared-token", 720.0)])
    manager = windowed_manager(login, [100.0])
    user = principal("fake-open-id", b"fingerprint")

    await manager.verify(user)
    await manager.verify(user)

    assert login.calls == ["fake-open-id"]


@pytest.mark.asyncio
async def test_verify_allows_only_five_failed_logins_per_window() -> None:
    login = ScriptedLoginClient([_FAILED])
    manager = windowed_manager(login, [100.0])
    user = principal("forged-open-id", b"fingerprint")

    for _ in range(7):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)

    assert login.calls == ["forged-open-id"] * 5


@pytest.mark.asyncio
async def test_verify_window_resets_after_reject_ttl() -> None:
    login = ScriptedLoginClient([_FAILED])
    now = [100.0]
    manager = windowed_manager(login, now)
    user = principal("forged-open-id", b"fingerprint")

    for _ in range(5):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)
    assert len(login.calls) == 5

    now[0] = 159.0
    with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
        await manager.verify(user)
    assert len(login.calls) == 5

    now[0] = 160.0
    with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
        await manager.verify(user)
    assert len(login.calls) == 6


@pytest.mark.asyncio
async def test_verify_unavailable_does_not_consume_failure_budget() -> None:
    login = ScriptedLoginClient([_UNAVAILABLE, _UNAVAILABLE, _FAILED])
    manager = windowed_manager(login, [100.0])
    user = principal("forged-open-id", b"fingerprint")

    for _ in range(2):
        with pytest.raises(ServiceError, match="SDK_AUTH_UNAVAILABLE"):
            await manager.verify(user)
    assert manager._verify_windows == {}  # pyright: ignore[reportPrivateUsage]

    for _ in range(5):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)
    assert len(login.calls) == 7
    with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
        await manager.verify(user)
    assert len(login.calls) == 7


@pytest.mark.asyncio
async def test_verify_success_clears_failure_window() -> None:
    login = ScriptedLoginClient([_FAILED, _FAILED, ("good-token", 720.0), _FAILED])
    manager = windowed_manager(login, [100.0])
    user = principal("fake-open-id", b"fingerprint")

    for _ in range(2):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)
    await manager.verify(user)
    assert manager._verify_windows == {}  # pyright: ignore[reportPrivateUsage]

    manager.invalidate(user.sdk_fingerprint)
    for _ in range(5):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)
    assert len(login.calls) == 8
    with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
        await manager.verify(user)
    assert len(login.calls) == 8


@pytest.mark.asyncio
async def test_concurrent_verify_callers_share_one_inflight_login() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_failed_login(_: SecretStr) -> tuple[SecretStr, float]:
        started.set()
        await release.wait()
        raise _FAILED

    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    login = FakeLoginClient(delayed_failed_login)
    manager = DirectSdkAuthManager(login)
    user = principal("forged-open-id", b"fingerprint")

    pending = [asyncio.create_task(manager.verify(user)) for _ in range(10)]
    await started.wait()
    await asyncio.sleep(0)
    assert login.calls == ["forged-open-id"]
    release.set()

    results = await asyncio.gather(*pending, return_exceptions=True)
    assert len(results) == 10
    assert all(isinstance(result, ServiceError) for result in results)
    assert login.calls == ["forged-open-id"]
    window = manager._verify_windows[user.sdk_fingerprint]  # pyright: ignore[reportPrivateUsage]
    assert window.failures == 1


@pytest.mark.asyncio
async def test_invalidate_clears_verify_window() -> None:
    login = ScriptedLoginClient([_FAILED])
    manager = windowed_manager(login, [100.0])
    user = principal("forged-open-id", b"fingerprint")

    for _ in range(5):
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)
    assert len(login.calls) == 5
    manager.invalidate(user.sdk_fingerprint)

    with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
        await manager.verify(user)
    assert len(login.calls) == 6


@pytest.mark.asyncio
async def test_verify_window_cache_is_lru_bounded_to_4096_failed_identities() -> None:
    login = ScriptedLoginClient([_FAILED])
    manager = windowed_manager(login, [100.0])
    users = [
        principal(f"forged-open-id-{index}", index.to_bytes(4, "big")) for index in range(4097)
    ]

    for user in users:
        with pytest.raises(ServiceError, match="SDK_AUTH_FAILED"):
            await manager.verify(user)

    assert len(manager._verify_windows) == 4096  # pyright: ignore[reportPrivateUsage]
    assert users[0].sdk_fingerprint not in manager._verify_windows  # pyright: ignore[reportPrivateUsage]
    assert users[-1].sdk_fingerprint in manager._verify_windows  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("ttl", "attempts"),
    [(0.0, 5), (-1.0, 5), (60.0, 0), (60.0, -1)],
)
def test_verify_window_parameters_must_be_positive(ttl: float, attempts: int) -> None:
    from tianyan_mcp.direct_sdk_auth import DirectSdkAuthManager

    with pytest.raises(ValueError):
        DirectSdkAuthManager(
            FakeLoginClient(), reject_ttl_seconds=ttl, max_verify_attempts=attempts
        )
