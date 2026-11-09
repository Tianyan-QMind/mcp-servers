from __future__ import annotations

import getpass

import httpx
import pytest

from tianyan_mcp.client_cli import main


def test_configure_uses_hidden_inputs_and_never_echoes_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key = "tymcp_0000000000000000_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    open_id = "fake-open-id"
    monkeypatch.setattr(getpass, "getpass", iter([key, open_id]).__next__)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"configured": True}, request=request)

    assert (
        main(
            ["configure", "--server", "https://mcp.example.com"],
            transport=httpx.MockTransport(handler),
        )
        == 0
    )
    output = capsys.readouterr().out
    assert open_id not in output and key not in output
    assert seen[0].url == "https://mcp.example.com/account/sdk-credential"
    assert seen[0].headers["authorization"] == f"Bearer {key}"


@pytest.mark.parametrize(
    "server", ["http://mcp.example.com", "http://localhost:8000", "ftp://mcp.example.com"]
)
def test_client_rejects_non_https_public_servers(server: str) -> None:
    with pytest.raises(SystemExit, match="HTTPS"):
        main(["status", "--server", server])


@pytest.mark.parametrize(
    "server",
    [
        "https://mcp.example.com/path",
        "https://user@mcp.example.com",
        "https://mcp.example.com?x=1",
        "https://mcp.example.com:99999",
    ],
)
def test_client_rejects_server_url_components_outside_fixed_endpoint(server: str) -> None:
    with pytest.raises(SystemExit, match="server URL"):
        main(["status", "--server", server])


def test_client_rejects_redirect_without_following_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(getpass, "getpass", lambda: "key")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"Location": "https://evil.example"}, request=request)

    assert (
        main(
            ["status", "--server", "https://mcp.example.com"],
            transport=httpx.MockTransport(handler),
        )
        == 2
    )
    assert len(seen) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"configured": "true", "updated_at": None},
        {"configured": True, "updated_at": None, "credential_revision": 1},
    ],
)
def test_status_rejects_malformed_or_extra_response_without_echoing_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    body: dict[str, object],
) -> None:
    key = "tymcp_0000000000000000_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    monkeypatch.setattr(getpass, "getpass", lambda: key)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    assert (
        main(
            ["status", "--server", "https://mcp.example.com"],
            transport=httpx.MockTransport(handler),
        )
        == 2
    )
    captured = capsys.readouterr()
    assert key not in captured.out and key not in captured.err
