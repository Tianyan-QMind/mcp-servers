import asyncio
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from tianyan_mcp.errors import ServiceError
from tianyan_mcp.sdk_auth import SDK_LOGIN_URL, TianyanSdkAuthProvider

FAKE_OPEN_ID = "fake-open-id"
FAKE_TOKEN = "fake-access-token"


def successful_login(token: str = FAKE_TOKEN, expires_in: object = 720) -> dict[str, Any]:
    return {
        "code": 0,
        "message": None,
        "data": {"access_token": token, "expires_in": expires_in},
        "success": True,
    }


@pytest.mark.asyncio
async def test_logs_in_and_reuses_cached_token() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.method == "POST"
        assert request.url == httpx.URL(SDK_LOGIN_URL, params={"openId": FAKE_OPEN_ID})
        return httpx.Response(200, json=successful_login())

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
        clock=lambda: 100.0,
    )

    assert await provider.get_access_token() == FAKE_TOKEN
    assert await provider.get_access_token() == FAKE_TOKEN
    assert calls == 1


@pytest.mark.asyncio
async def test_rejects_http_200_business_failure_without_leaking_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"msg": f"{FAKE_OPEN_ID} {FAKE_TOKEN}", "code": 500})
    )
    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID), timeout_seconds=1, transport=transport
    )

    with pytest.raises(ServiceError) as captured:
        await provider.get_access_token()

    assert captured.value.code == "SDK_AUTH_FAILED"
    assert_secret_is_not_exposed(caplog, captured.value)


@pytest.mark.asyncio
async def test_refreshes_inside_sixty_second_buffer() -> None:
    now = 100.0
    issued = iter(["first-token", "second-token"])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=successful_login(next(issued)))

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
        clock=lambda: now,
    )

    assert await provider.get_access_token() == "first-token"
    now = 761.0
    assert await provider.get_access_token() == "second-token"


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_login() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return httpx.Response(200, json=successful_login("shared-token"))

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    tokens = await asyncio.gather(*(provider.get_access_token() for _ in range(5)))

    assert tokens == ["shared-token"] * 5
    assert calls == 1


@pytest.mark.asyncio
async def test_invalidate_discards_cached_token() -> None:
    issued = iter(["first-token", "second-token"])

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=successful_login(next(issued)))
        ),
    )

    assert await provider.get_access_token() == "first-token"
    provider.invalidate()
    assert await provider.get_access_token() == "second-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expires_in",
    [True, float("nan"), float("inf"), float("-inf")],
)
async def test_rejects_boolean_and_non_finite_expiry(
    expires_in: object,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        if isinstance(expires_in, float) and not math.isfinite(expires_in):
            literal = (
                "NaN" if math.isnan(expires_in) else "Infinity" if expires_in > 0 else "-Infinity"
            )
            return httpx.Response(
                200,
                content=(
                    '{"code":0,"message":null,"data":{"access_token":"fake-access-token",'
                    f'"expires_in":{literal}'
                    '},"success":true}'
                ).encode(),
            )
        return httpx.Response(200, json=successful_login(expires_in=expires_in))

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ServiceError) as captured:
        await provider.get_access_token()

    assert captured.value.code == "SDK_AUTH_UNAVAILABLE"


@pytest.mark.asyncio
async def test_refreshes_when_remaining_validity_is_exactly_sixty_seconds() -> None:
    now = 100.0
    issued = iter(["first-token", "second-token"])
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=successful_login(next(issued)))

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
        clock=lambda: now,
    )

    assert await provider.get_access_token() == "first-token"
    now = 760.0
    assert await provider.get_access_token() == "second-token"
    assert calls == 2


@pytest.mark.asyncio
async def test_invalidate_during_login_does_not_restore_stale_cache() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    issued = iter(["stale-token", "fresh-token"])
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json=successful_login(next(issued)))

    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    pending_login = asyncio.create_task(provider.get_access_token())
    await started.wait()
    provider.invalidate()
    release.set()
    assert await pending_login == "stale-token"
    assert await provider.get_access_token() == "fresh-token"
    assert calls == 2


def http_error_handler(_: httpx.Request) -> httpx.Response:
    return httpx.Response(503, text="server error")


def non_json_handler(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="not json")


def empty_token_handler(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=successful_login("   "))


def zero_expiry_handler(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=successful_login(expires_in=0))


def negative_expiry_handler(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=successful_login(expires_in=-1))


def timeout_handler(_: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("upstream timed out")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("open_id", "handler", "expected_code"),
    [
        (None, None, "SDK_AUTH_FAILED"),
        (SecretStr(FAKE_OPEN_ID), http_error_handler, "SDK_AUTH_UNAVAILABLE"),
        (SecretStr(FAKE_OPEN_ID), timeout_handler, "SDK_AUTH_UNAVAILABLE"),
        (SecretStr(FAKE_OPEN_ID), non_json_handler, "SDK_AUTH_UNAVAILABLE"),
        (SecretStr(FAKE_OPEN_ID), empty_token_handler, "SDK_AUTH_UNAVAILABLE"),
        (SecretStr(FAKE_OPEN_ID), negative_expiry_handler, "SDK_AUTH_UNAVAILABLE"),
        (SecretStr(FAKE_OPEN_ID), zero_expiry_handler, "SDK_AUTH_UNAVAILABLE"),
    ],
    ids=[
        "missing-open-id",
        "http-error",
        "timeout",
        "non-json",
        "empty-token",
        "zero-expiry",
        "negative-expiry",
    ],
)
async def test_login_failures_do_not_expose_credentials(
    open_id: SecretStr | None,
    handler: Callable[[httpx.Request], httpx.Response] | None,
    expected_code: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = TianyanSdkAuthProvider(
        open_id,
        timeout_seconds=1,
        transport=httpx.MockTransport(handler) if handler is not None else None,
    )

    with pytest.raises(ServiceError) as captured:
        await provider.get_access_token()

    assert captured.value.code == expected_code
    assert_secret_is_not_exposed(caplog, captured.value)


def assert_secret_is_not_exposed(caplog: pytest.LogCaptureFixture, error: Exception) -> None:
    rendered = f"{error}\n{caplog.text}"
    assert FAKE_OPEN_ID not in rendered
    assert FAKE_TOKEN not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [False, "0", 0.0, 1.5])
async def test_rejects_invalid_login_business_code_type(code: object) -> None:
    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={**successful_login(), "code": code})
        ),
    )

    with pytest.raises(ServiceError) as captured:
        await provider.get_access_token()

    assert captured.value.code == "SDK_AUTH_UNAVAILABLE"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [-1, 1])
async def test_rejects_nonzero_integer_login_business_code(code: int) -> None:
    provider = TianyanSdkAuthProvider(
        SecretStr(FAKE_OPEN_ID),
        timeout_seconds=1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={**successful_login(), "code": code})
        ),
    )

    with pytest.raises(ServiceError) as captured:
        await provider.get_access_token()

    assert captured.value.code == "SDK_AUTH_FAILED"
