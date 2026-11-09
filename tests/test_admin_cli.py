from __future__ import annotations

import base64
import sqlite3
from pathlib import Path

import pytest

from tianyan_mcp.admin_cli import main


@pytest.fixture
def cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        "TIANYAN_IDENTITY_MODE": "multi_tenant",
        "TIANYAN_DATABASE_PATH": str((tmp_path / "tenants.sqlite").resolve()),
        "TIANYAN_API_KEY_PEPPER": "p" * 32,
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION": "1",
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V1": base64.b64encode(b"k" * 32).decode(),
    }


def _created_tenant(cli_env: dict[str, str], capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    assert main(["create-user", "--name", "alice"], environ=cli_env) == 0
    output = capsys.readouterr().out
    plaintext = next(part for part in output.split() if part.startswith("tymcp_"))
    with sqlite3.connect(cli_env["TIANYAN_DATABASE_PATH"]) as connection:
        tenant_id = connection.execute("SELECT id FROM tenants").fetchone()[0]
    return tenant_id, plaintext


def test_create_user_prints_key_once_without_persisting_it(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["create-user", "--name", "alice"], environ=cli_env) == 0

    output = capsys.readouterr().out
    plaintext = next(part for part in output.split() if part.startswith("tymcp_"))
    assert output.count(plaintext) == 1
    assert plaintext.encode() not in Path(cli_env["TIANYAN_DATABASE_PATH"]).read_bytes()


def test_list_users_never_prints_credentials(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    _created_tenant(cli_env, capsys)

    assert main(["list-users"], environ=cli_env) == 0
    output = capsys.readouterr().out.lower()
    assert "tymcp_" not in output
    assert "sdk_open_id" not in output
    assert "ciphertext" not in output
    assert "token" not in output
    assert all(all(character.isprintable() for character in line) for line in output.splitlines())


@pytest.mark.parametrize(
    "command",
    ["disable-user", "enable-user", "rotate-key", "revoke-key"],
)
def test_sensitive_commands_require_explicit_confirmation_without_audit_changes(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str], command: str
) -> None:
    tenant_id, _ = _created_tenant(cli_env, capsys)
    database = Path(cli_env["TIANYAN_DATABASE_PATH"])
    with sqlite3.connect(database) as connection:
        before = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    assert main([command, "--tenant-id", tenant_id], environ=cli_env) == 2

    assert "confirmation" in capsys.readouterr().err.lower()
    with sqlite3.connect(database) as connection:
        after = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    assert after == before


@pytest.mark.parametrize(
    ("argv",),
    [
        (["disable-user", "--tenant-id", "00000000-0000-0000-0000-000000000000"],),
        (["enable-user", "--tenant-id", "00000000-0000-0000-0000-000000000000"],),
        (["rotate-key", "--tenant-id", "00000000-0000-0000-0000-000000000000"],),
        (["revoke-key", "--tenant-id", "00000000-0000-0000-0000-000000000000"],),
        (["backup", "--output", "C:\\not-used.sqlite"],),
    ],
)
def test_unconfirmed_command_does_not_initialize_new_database(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    database = Path(cli_env["TIANYAN_DATABASE_PATH"])
    assert not database.exists()

    assert main(argv, environ=cli_env) == 2

    assert "confirmation" in capsys.readouterr().err.lower()
    assert not database.exists()


def test_rotate_revokes_old_key_and_prints_replacement_once(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    tenant_id, old_key = _created_tenant(cli_env, capsys)

    assert main(["rotate-key", "--tenant-id", tenant_id, "--confirm"], environ=cli_env) == 0
    output = capsys.readouterr().out
    new_key = next(part for part in output.split() if part.startswith("tymcp_"))
    assert output.count(new_key) == 1

    from tianyan_mcp.admin_cli import build_admin_dependencies
    from tianyan_mcp.config import Settings

    store, _ = build_admin_dependencies(Settings.from_env(cli_env, require_api_key=False))
    assert store.authenticate(old_key) is None
    assert store.authenticate(new_key) is not None


def test_admin_cli_rejects_single_user_and_relative_database_path(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    single_user_env = {"TIANYAN_MCP_API_KEY": "k" * 32}
    with pytest.raises(SystemExit, match="multi_tenant"):
        main(["list-users"], environ=single_user_env)

    cli_env["TIANYAN_DATABASE_PATH"] = "relative.sqlite"
    with pytest.raises(SystemExit, match="absolute"):
        main(["list-users"], environ=cli_env)
    assert "p" * 32 not in capsys.readouterr().err


def test_backup_refuses_unsafe_destination_and_requires_confirmation(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _created_tenant(cli_env, capsys)
    source = Path(cli_env["TIANYAN_DATABASE_PATH"])

    assert main(["backup", "--output", str(source)], environ=cli_env) == 2
    assert "confirmation" in capsys.readouterr().err.lower()
    assert main(["backup", "--output", str(source), "--confirm"], environ=cli_env) == 2
    assert "invalid" in capsys.readouterr().err.lower()

    assert main(["backup", "--output", "relative.sqlite", "--confirm"], environ=cli_env) == 2
    assert "absolute" in capsys.readouterr().err.lower()


def test_admin_audits_are_written_with_state_changes(
    cli_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    tenant_id, _ = _created_tenant(cli_env, capsys)
    assert main(["disable-user", "--tenant-id", tenant_id, "--confirm"], environ=cli_env) == 0

    with sqlite3.connect(cli_env["TIANYAN_DATABASE_PATH"]) as connection:
        status = connection.execute(
            "SELECT status FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        events = connection.execute(
            "SELECT event_type, outcome FROM audit_events WHERE tenant_id = ? ORDER BY occurred_at",
            (tenant_id,),
        ).fetchall()
    assert status == ("disabled",)
    assert events == [("tenant_created", "success"), ("tenant_status_changed", "success")]


def test_failed_admin_audit_rolls_back_tenant_state(
    cli_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tianyan_mcp.tenant_store import TenantStore

    def fail_audit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(TenantStore, "_append_audit", staticmethod(fail_audit))
    with pytest.raises(RuntimeError, match="audit unavailable"):
        main(["create-user", "--name", "alice"], environ=cli_env)

    with sqlite3.connect(cli_env["TIANYAN_DATABASE_PATH"]) as connection:
        count = connection.execute("SELECT COUNT(*) FROM tenants").fetchone()
    assert count == (0,)
