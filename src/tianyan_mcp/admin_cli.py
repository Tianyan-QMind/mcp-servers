from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from tianyan_mcp.config import Settings
from tianyan_mcp.credential_crypto import ApiKeyCodec
from tianyan_mcp.identity import IdentityMode
from tianyan_mcp.tenant_store import AuditEventType, TenantStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tianyan-mcp-admin")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user")
    create.add_argument("--name", required=True)
    commands.add_parser("list-users")
    for command in ("disable-user", "enable-user", "rotate-key", "revoke-key"):
        child = commands.add_parser(command)
        child.add_argument("--tenant-id", required=True)
        child.add_argument("--confirm", action="store_true")
    backup = commands.add_parser("backup")
    backup.add_argument("--output", type=Path, required=True)
    backup.add_argument("--confirm", action="store_true")
    return parser


def build_admin_dependencies(settings: Settings) -> tuple[TenantStore, ApiKeyCodec]:
    if settings.database_path is None or not settings.database_path.is_absolute():
        raise SystemExit("admin CLI requires an absolute database path")
    if settings.api_key_pepper is None:
        raise SystemExit("admin CLI requires API key pepper")
    codec = ApiKeyCodec(settings.api_key_pepper.get_secret_value())
    store = TenantStore(settings.database_path, codec)
    store.initialize()
    return store, codec


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in _CONFIRMATION_REQUIRED_COMMANDS and not args.confirm:
        print("explicit confirmation is required; re-run with --confirm", file=sys.stderr)
        return 2
    try:
        settings = Settings.from_env(environ, require_api_key=False)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if settings.identity_mode is not IdentityMode.MULTI_TENANT:
        raise SystemExit("admin CLI requires multi_tenant mode")
    store, codec = build_admin_dependencies(settings)
    try:
        return _dispatch(args, store, codec)
    except (KeyError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, store: TenantStore, codec: ApiKeyCodec) -> int:
    command: str = args.command
    if command == "create-user":
        generated = codec.generate()
        tenant = store.create_tenant(
            args.name, generated, audit_event=AuditEventType.TENANT_CREATED
        )
        print(f"tenant_id: {tenant.id}")
        print(f"api_key: {generated.plaintext}")
        return 0
    if command == "list-users":
        for tenant in store.list_tenants():
            print(
                f"tenant_id: {tenant.id} name: {tenant.display_name} "
                f"status: {tenant.status.value} auth_revision: {tenant.auth_revision}"
            )
        return 0
    if command == "disable-user":
        store.set_tenant_status(
            args.tenant_id, enabled=False, audit_event=AuditEventType.TENANT_STATUS_CHANGED
        )
        print("tenant disabled")
        return 0
    if command == "enable-user":
        store.set_tenant_status(
            args.tenant_id, enabled=True, audit_event=AuditEventType.TENANT_STATUS_CHANGED
        )
        print("tenant enabled")
        return 0
    if command == "rotate-key":
        generated = codec.generate()
        store.rotate_key(args.tenant_id, generated, audit_event=AuditEventType.API_KEY_ROTATED)
        print(f"api_key: {generated.plaintext}")
        return 0
    if command == "revoke-key":
        store.revoke_key(args.tenant_id, audit_event=AuditEventType.API_KEY_REVOKED)
        print("tenant API key revoked")
        return 0
    if command == "backup":
        if not args.output.is_absolute():
            print("backup output path must be absolute", file=sys.stderr)
            return 2
        store.backup_with_audit(args.output)
        print("backup created")
        return 0
    raise RuntimeError("unsupported admin command")


_CONFIRMATION_REQUIRED_COMMANDS = frozenset(
    {"disable-user", "enable-user", "rotate-key", "revoke-key", "backup"}
)
