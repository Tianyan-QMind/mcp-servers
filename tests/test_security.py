import logging

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tianyan_mcp.app import create_app
from tianyan_mcp.config import Settings
from tianyan_mcp.security import SecurityMiddleware

API_KEY = "test-key-" + "x" * 23


async def endpoint(_: Request) -> JSONResponse:
    return JSONResponse({"reached": True})


def make_settings() -> Settings:
    return Settings.from_env(
        {
            "TIANYAN_MCP_API_KEY": API_KEY,
            "TIANYAN_MCP_ALLOWED_ORIGINS": "https://agent.example.com",
        }
    )


def secured_app() -> SecurityMiddleware:
    inner = Starlette(
        routes=[
            Route("/healthz", endpoint, methods=["GET"]),
            Route("/mcp", endpoint, methods=["POST"]),
        ]
    )
    return SecurityMiddleware(inner, make_settings())


@pytest.mark.asyncio
async def test_health_is_public() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=secured_app()),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        (
            {
                "Authorization": f"Bearer {API_KEY}",
                "Origin": "https://evil.example.com",
            },
            403,
        ),
        (
            {
                "Authorization": f"Bearer {API_KEY}",
                "Host": "evil.example.com",
            },
            421,
        ),
    ],
)
async def test_rejects_invalid_access(
    headers: dict[str, str],
    status: int,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=secured_app()),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post("/mcp", headers=headers)
    assert response.status_code == status
    if status == 401:
        assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_rejects_malformed_host() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=secured_app()),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {API_KEY}", "Host": "["},
        )
    assert response.status_code == 421


@pytest.mark.asyncio
async def test_rejects_duplicate_origin() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=secured_app()),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post(
            "/mcp",
            headers=[
                ("Authorization", f"Bearer {API_KEY}"),
                ("Origin", "https://agent.example.com"),
                ("Origin", "https://evil.example.com"),
            ],
        )
    assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_headers",
    [
        {},
        {"Origin": "https://agent.example.com"},
        {"Host": "127.0.0.1:8000"},
    ],
)
async def test_accepts_valid_access(extra_headers: dict[str, str]) -> None:
    headers = {"Authorization": f"Bearer {API_KEY}", **extra_headers}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=secured_app()),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post("/mcp", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"reached": True}


@pytest.mark.asyncio
async def test_credentials_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=secured_app()),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    assert response.status_code == 200
    assert API_KEY not in caplog.text


@pytest.mark.asyncio
async def test_app_factory_exposes_public_health() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(make_settings())),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
