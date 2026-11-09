from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, StrictBool


class _CredentialStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    configured: StrictBool
    updated_at: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tianyan-mcp-client")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("configure", "status", "remove"):
        child = commands.add_parser(command)
        child.add_argument("--server", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    args = build_parser().parse_args(argv)
    status: _CredentialStatus | None = None
    try:
        server = _validated_server(args.server)
        api_key = getpass.getpass()
        if not api_key:
            raise ValueError("MCP API Key is required")
        headers = {"Authorization": f"Bearer {api_key}"}
        with httpx.Client(
            transport=transport, follow_redirects=False, trust_env=False, headers=headers
        ) as client:
            if args.command == "configure":
                open_id = getpass.getpass()
                response = client.put(
                    f"{server}/account/sdk-credential", json={"sdk_open_id": open_id}
                )
            elif args.command == "status":
                response = client.get(f"{server}/account/sdk-credential")
            else:
                response = client.delete(f"{server}/account/sdk-credential")
            if response.is_redirect:
                raise ValueError("server redirect is not allowed")
            if response.status_code >= 400:
                raise ValueError("server rejected credential request")
            if args.command == "status":
                status = _CredentialStatus.model_validate(response.json())
    except (ValueError, httpx.HTTPError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.command == "status":
        if status is None:
            print("server returned an invalid credential status", file=sys.stderr)
            return 2
        print(f"configured: {str(status.configured).lower()}")
        print(f"updated_at: {status.updated_at or 'none'}")
    else:
        print("credential request completed")
    return 0


def _validated_server(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise SystemExit("server URL is invalid") from error
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or not parsed.hostname
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise SystemExit("server URL is invalid")
    if parsed.scheme == "https":
        return value.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}:
        return value.rstrip("/")
    raise SystemExit("public credential server must use HTTPS")
