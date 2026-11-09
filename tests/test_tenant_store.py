from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import SecretStr

from tianyan_mcp.credential_crypto import ApiKeyCodec, CredentialCipher
from tianyan_mcp.identity import TenantPrincipal
from tianyan_mcp.tenant_store import (
    AuditEventType,
    AuditOutcome,
    CredentialStatus,
    TenantDisabledError,
    TenantStore,
)


@pytest.fixture
def store(tmp_path: Path) -> TenantStore:
    result = TenantStore(tmp_path / "tenants.sqlite", ApiKeyCodec(b"p" * 32))
    result.initialize()
    return result


@pytest.fixture
def codec() -> ApiKeyCodec:
    return ApiKeyCodec(b"p" * 32)


@pytest.fixture
def cipher() -> CredentialCipher:
    return CredentialCipher(active_version=1, keys={1: b"k" * 32})


def test_migrations_are_idempotent_and_enable_sqlite_safety(store: TenantStore) -> None:
    store.initialize()

    with store._connect() as connection:  # pyright: ignore[reportPrivateUsage]
        version = connection.execute("SELECT version FROM schema_migrations").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()

    assert version == (1,)
    assert foreign_keys == (1,)
    assert journal_mode == ("wal",)
    assert busy_timeout == (5000,)


def test_create_and_authenticate_one_active_key(store: TenantStore, codec: ApiKeyCodec) -> None:
    generated = codec.generate()
    tenant = store.create_tenant("alice", generated)

    assert store.authenticate(generated.plaintext) == TenantPrincipal(
        tenant.id, generated.key_id, 1
    )
    assert generated.plaintext.encode() not in store.path.read_bytes()


def test_failed_authentication_does_not_update_last_used_at(
    store: TenantStore, codec: ApiKeyCodec
) -> None:
    generated = codec.generate()
    tenant = store.create_tenant("alice", generated)

    replacement = "A" if generated.plaintext[-1] != "A" else "B"
    invalid_but_well_formed_key = generated.plaintext[:-1] + replacement
    assert store.authenticate(invalid_but_well_formed_key) is None

    with sqlite3.connect(store.path) as connection:
        last_used_at = connection.execute(
            "SELECT last_used_at FROM api_keys WHERE tenant_id = ?", (tenant.id,)
        ).fetchone()
    assert last_used_at == (None,)


def test_rotate_key_revokes_old_key_atomically(store: TenantStore, codec: ApiKeyCodec) -> None:
    first = codec.generate()
    tenant = store.create_tenant("alice", first)
    second = codec.generate()

    replacement = store.rotate_key(tenant.id, second)

    assert replacement.tenant_id == tenant.id
    assert store.authenticate(first.plaintext) is None
    assert store.authenticate(second.plaintext) == TenantPrincipal(tenant.id, second.key_id, 2)


def test_partial_unique_constraint_allows_only_one_active_key(
    store: TenantStore, codec: ApiKeyCodec
) -> None:
    first = codec.generate()
    tenant = store.create_tenant("alice", first)
    second = codec.generate()

    with sqlite3.connect(store.path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO api_keys (id, tenant_id, verification_tag, created_at)
            VALUES (?, ?, ?, ?)""",
            (second.key_id, tenant.id, second.verification_tag, "2026-08-11T00:00:00+00:00"),
        )


def test_disabled_tenant_is_distinguished(store: TenantStore, codec: ApiKeyCodec) -> None:
    generated = codec.generate()
    tenant = store.create_tenant("alice", generated)
    store.set_tenant_status(tenant.id, enabled=False)

    with pytest.raises(TenantDisabledError):
        store.authenticate(generated.plaintext)


def test_credential_revision_increments_and_never_persists_plaintext(
    store: TenantStore, codec: ApiKeyCodec, cipher: CredentialCipher
) -> None:
    tenant = store.create_tenant("alice", codec.generate())
    first = cipher.encrypt(tenant.id, SecretStr("first-open-id"))
    second = cipher.encrypt(tenant.id, SecretStr("second-open-id"))

    first_status = store.put_credential(tenant.id, first)
    second_status = store.put_credential(tenant.id, second)
    record = store.get_credential(tenant.id)
    removed = store.delete_credential(tenant.id)
    third_status = store.put_credential(tenant.id, first)

    assert first_status == CredentialStatus(
        configured=True, credential_revision=1, updated_at=first_status.updated_at
    )
    assert second_status.credential_revision == 2
    assert record is not None and record.credential_revision == 2
    assert removed.credential_revision == 3
    assert third_status.credential_revision == 4
    contents = store.path.read_bytes()
    assert b"first-open-id" not in contents
    assert b"second-open-id" not in contents


@pytest.mark.parametrize(
    ("operation", "event"),
    [
        ("put", AuditEventType.CREDENTIAL_UPDATED),
        ("delete", AuditEventType.CREDENTIAL_DELETED),
    ],
)
def test_credential_state_and_audit_roll_back_together_when_audit_fails(
    store: TenantStore,
    codec: ApiKeyCodec,
    cipher: CredentialCipher,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    event: AuditEventType,
) -> None:
    generated = codec.generate()
    tenant = store.create_tenant("alice", generated)
    if operation == "delete":
        store.put_credential(tenant.id, cipher.encrypt(tenant.id, SecretStr("old")))
    before = store.get_credential(tenant.id)

    def fail_audit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(TenantStore, "_append_audit", staticmethod(fail_audit))
    with pytest.raises(RuntimeError, match="audit unavailable"):
        if operation == "put":
            store.put_credential(
                tenant.id,
                cipher.encrypt(tenant.id, SecretStr("new")),
                audit_event=event,
                audit_key_id=generated.key_id,
                request_id="33c635fb-3dc7-4b47-aa93-7fead7c6a0d7",
            )
        else:
            store.delete_credential(
                tenant.id,
                audit_event=event,
                audit_key_id=generated.key_id,
                request_id="33c635fb-3dc7-4b47-aa93-7fead7c6a0d7",
            )

    assert store.get_credential(tenant.id) == before
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone() == (0,)


def test_audit_only_persists_whitelisted_non_secret_fields(
    store: TenantStore, codec: ApiKeyCodec
) -> None:
    generated = codec.generate()
    tenant = store.create_tenant("alice", generated)
    request_id = "33c635fb-3dc7-4b47-aa93-7fead7c6a0d7"

    store.append_audit(
        tenant.id,
        None,
        AuditEventType.CREDENTIAL_UPDATED,
        AuditOutcome.SUCCESS,
        request_id,
    )

    with sqlite3.connect(store.path) as connection:
        before_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()

    forbidden_values = (
        ("not-a-uuid", None, AuditEventType.CREDENTIAL_UPDATED, AuditOutcome.SUCCESS, None),
        (tenant.id, "not-a-key-id", AuditEventType.CREDENTIAL_UPDATED, AuditOutcome.SUCCESS, None),
        (tenant.id, None, "Authorization: Bearer secret", AuditOutcome.SUCCESS, None),
        (tenant.id, None, AuditEventType.CREDENTIAL_UPDATED, "sdk-open-id", None),
        (tenant.id, None, AuditEventType.CREDENTIAL_UPDATED, AuditOutcome.SUCCESS, "not-a-uuid"),
    )
    for values in forbidden_values:
        with pytest.raises((TypeError, ValueError)):
            store.append_audit(*values)

    with sqlite3.connect(store.path) as connection:
        after_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        stored = connection.execute(
            "SELECT tenant_id, key_id, event_type, outcome, request_id FROM audit_events"
        ).fetchone()
    assert before_count == (1,)
    assert after_count == before_count
    assert stored == (tenant.id, None, "credential_updated", "success", request_id)

    secret_bearing_values = (
        generated.plaintext,
        "sdk-open-id-value",
        "Authorization: Bearer secret-value",
    )
    valid_values: list[object] = [
        tenant.id,
        None,
        AuditEventType.CREDENTIAL_UPDATED,
        AuditOutcome.SUCCESS,
        request_id,
    ]
    for secret in secret_bearing_values:
        for index in range(len(valid_values)):
            invalid_values = valid_values.copy()
            invalid_values[index] = secret
            with pytest.raises((TypeError, ValueError)):
                store.append_audit(*invalid_values)

    with sqlite3.connect(store.path) as connection:
        final_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()
    assert final_count == before_count


def test_backup_rejects_source_and_existing_destination_without_overwrite(
    store: TenantStore, codec: ApiKeyCodec, tmp_path: Path
) -> None:
    store.create_tenant("alice", codec.generate())
    source_snapshot = store.path.read_bytes()
    existing_backup = tmp_path / "existing.sqlite"
    existing_backup.write_bytes(b"sqlite-sentinel")
    existing_directory = tmp_path / "existing-directory"
    existing_directory.mkdir()

    with pytest.raises(ValueError, match="backup destination is invalid"):
        store.backup(store.path.parent / "." / store.path.name)
    with pytest.raises(ValueError, match="backup destination is invalid"):
        store.backup(existing_backup)
    with pytest.raises(ValueError, match="backup destination is invalid"):
        store.backup(existing_directory)

    assert store.path.read_bytes() == source_snapshot
    assert existing_backup.read_bytes() == b"sqlite-sentinel"
    assert existing_directory.is_dir()


def test_backup_rejects_dangling_symlink_before_path_resolution(
    store: TenantStore, codec: ApiKeyCodec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store.create_tenant("alice", codec.generate())
    dangling_symlink = tmp_path / "dangling-backup-link"
    resolved_target = tmp_path / "resolved-backup-target.sqlite"
    original_resolve = Path.resolve
    original_is_symlink = Path.is_symlink

    def resolve_with_dangling_link(path: Path, strict: bool = False) -> Path:
        if path == dangling_symlink:
            return resolved_target
        return original_resolve(path, strict=strict)

    def identify_dangling_link(path: Path) -> bool:
        return path == dangling_symlink or original_is_symlink(path)

    monkeypatch.setattr(Path, "resolve", resolve_with_dangling_link)
    monkeypatch.setattr(Path, "is_symlink", identify_dangling_link)

    with pytest.raises(ValueError, match="backup destination is invalid"):
        store.backup(dangling_symlink)

    assert not resolved_target.exists()


def test_audit_schema_has_no_payload_column(store: TenantStore) -> None:
    with store._connect() as connection:  # pyright: ignore[reportPrivateUsage]
        column_names = {
            row[1] for row in connection.execute("PRAGMA table_info(audit_events)").fetchall()
        }

    assert "payload" not in column_names


def test_foreign_keys_reject_orphaned_api_key(store: TenantStore, codec: ApiKeyCodec) -> None:
    generated = codec.generate()

    with store._connect() as connection, pytest.raises(sqlite3.IntegrityError):  # pyright: ignore[reportPrivateUsage]
        connection.execute(
            """INSERT INTO api_keys (id, tenant_id, verification_tag, created_at)
            VALUES (?, ?, ?, ?)""",
            (
                generated.key_id,
                "missing-tenant",
                generated.verification_tag,
                "2026-08-11T00:00:00+00:00",
            ),
        )


def test_concurrent_short_writes_remain_consistent(store: TenantStore, codec: ApiKeyCodec) -> None:
    def create(index: int) -> str:
        return store.create_tenant(f"tenant-{index}", codec.generate()).id

    with ThreadPoolExecutor(max_workers=6) as executor:
        ids = list(executor.map(create, range(12)))

    assert len(set(ids)) == 12


def test_online_backup_is_a_consistent_database(
    store: TenantStore, codec: ApiKeyCodec, tmp_path: Path
) -> None:
    tenant = store.create_tenant("alice", codec.generate())
    backup_path = tmp_path / "backup.sqlite"

    store.backup(backup_path)

    with sqlite3.connect(backup_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        copied = connection.execute("SELECT id FROM tenants").fetchone()
    assert integrity == ("ok",)
    assert copied == (tenant.id,)


def test_admin_backup_removes_new_destination_when_audit_fails(
    store: TenantStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "backup.sqlite"

    def fail_audit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(TenantStore, "_append_audit", staticmethod(fail_audit))
    with pytest.raises(RuntimeError, match="audit unavailable"):
        store.backup_with_audit(destination)

    assert not destination.exists()
    with sqlite3.connect(store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()
    assert count == (0,)


def test_admin_backup_removes_new_destination_when_copy_fails(
    store: TenantStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "backup.sqlite"

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise sqlite3.DatabaseError("copy unavailable")

    monkeypatch.setattr(TenantStore, "_copy_backup", fail_copy)
    with pytest.raises(sqlite3.DatabaseError, match="copy unavailable"):
        store.backup_with_audit(destination)

    assert not destination.exists()
    with sqlite3.connect(store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()
    assert count == (0,)


def test_admin_backup_creates_snapshot_and_audits_success(
    store: TenantStore, codec: ApiKeyCodec, tmp_path: Path
) -> None:
    tenant = store.create_tenant("alice", codec.generate())
    destination = tmp_path / "backup.sqlite"

    store.backup_with_audit(destination)

    with sqlite3.connect(destination) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        copied = connection.execute("SELECT id FROM tenants").fetchone()
    with sqlite3.connect(store.path) as connection:
        events = connection.execute("SELECT event_type, outcome FROM audit_events").fetchall()
    assert integrity == ("ok",)
    assert copied == (tenant.id,)
    assert events == [("backup_created", "success")]


@pytest.mark.parametrize(
    "display_name",
    [
        "",
        "a" * 129,
        "alice\nadmin",
        "alice\x1b[2J",
        "alice\u202eadmin",
        "tenant tymcp_0000000000000000_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ],
)
def test_create_tenant_rejects_unsafe_display_names(
    store: TenantStore, codec: ApiKeyCodec, display_name: str
) -> None:
    with pytest.raises(ValueError, match="display name"):
        store.create_tenant(display_name, codec.generate())


def test_normal_chinese_display_name_is_safe_to_list(
    store: TenantStore, codec: ApiKeyCodec
) -> None:
    tenant = store.create_tenant("  量子 用户  ", codec.generate())

    listed = store.list_tenants()[0]
    assert listed.id == tenant.id
    assert listed.display_name == "量子 用户"
    assert all(character.isprintable() for character in listed.display_name)
    assert "tymcp_" not in listed.display_name
