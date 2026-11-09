import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_COMPOSE = """\
name: tianyan-mcp

services:
  tianyan-mcp-server:
    build:
      context: .
      dockerfile: Dockerfile
    image: tianyan-mcp-server:local-arm64
    container_name: tianyan-mcp-server
    network_mode: "container:nginx-test"
    restart: unless-stopped
    init: true
    user: "10001:10001"
    read_only: true
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    env_file:
      - /mnt/sdc/MCPserver/secrets/runtime.env
    volumes:
      - /mnt/sdc/MCPserver/data:/data
      - /mnt/sdc/MCPserver/backup:/backup
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=64m
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - >-
          import urllib.request;
          urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5).read()
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    stop_grace_period: 30s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
"""


def _compose() -> str:
    return (ROOT / "compose.yaml").read_text(encoding="utf-8")


def test_compose_exactly_matches_approved_runtime_contract() -> None:
    assert _compose() == _EXPECTED_COMPOSE


def _service(compose: str) -> str:
    match = re.search(
        r"(?ms)^  tianyan-mcp-server:\n(?P<service>(?:^    .*\n|^\n)*)\Z",
        compose,
    )
    assert match, "compose must contain one block-style tianyan-mcp-server service"
    return match.group("service")


def _list_setting(service: str, name: str) -> list[str]:
    match = re.search(rf"(?m)^    {name}:\n((?:      - .*\n)+)", service)
    assert match, f"{name} must use a block-style list"
    return [line.strip()[2:] for line in match.group(1).splitlines()]


def _nested_setting(service: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^    {name}:\n(?P<setting>(?:^      .*\n|^\n)*)"
        rf"(?=^    [a-z_][a-z_]*:|\Z)",
        service,
    )
    assert match, f"{name} must use a block-style mapping"
    return match.group("setting")


def test_compose_shares_nginx_namespace_without_publishing_ports() -> None:
    compose = _compose()
    service = _service(compose)

    assert "container_name: tianyan-mcp-server" in compose
    assert 'network_mode: "container:nginx-test"' in compose
    assert not re.search(r"(?m)^    ['\"]?(ports|expose|privileged|networks)['\"]?\s*:", service)
    assert "/var/run/docker.sock" not in compose


def test_compose_mounts_only_persistent_runtime_data() -> None:
    compose = _compose()
    service = _service(compose)

    assert _list_setting(service, "env_file") == ["/mnt/sdc/MCPserver/secrets/runtime.env"]
    assert _list_setting(service, "volumes") == [
        "/mnt/sdc/MCPserver/data:/data",
        "/mnt/sdc/MCPserver/backup:/backup",
    ]
    assert "/mnt/sdc/MCPserver/secrets/runtime.env" not in _list_setting(service, "volumes")
    assert "/mnt/sdc/MCPserver/app:/app" not in compose


def test_compose_applies_runtime_hardening() -> None:
    compose = _compose()
    service = _service(compose)

    assert 'user: "10001:10001"' in service
    assert "read_only: true" in service
    assert _list_setting(service, "cap_drop") == ["ALL"]
    assert _list_setting(service, "security_opt") == ["no-new-privileges:true"]
    assert _list_setting(service, "tmpfs") == ["/tmp:rw,noexec,nosuid,size=64m"]  # noqa: S108
    assert "restart: unless-stopped" in service
    assert 'max-size: "10m"' in _nested_setting(service, "logging")
    assert 'max-file: "3"' in _nested_setting(service, "logging")


def test_compose_healthcheck_is_local_and_read_only() -> None:
    compose = _compose()
    healthcheck = _nested_setting(_service(compose), "healthcheck")

    assert "http://127.0.0.1:8000/healthz" in healthcheck
    assert "urllib.request.urlopen" in healthcheck
    assert "TIANYAN_LIVE_TEST" not in compose
    assert "TIANYAN_LIVE_WRITE_TEST" not in compose
    assert "TIANYAN_SDK_OPEN_ID" not in compose
    assert "TIANYAN_MCP_API_KEY" not in compose
