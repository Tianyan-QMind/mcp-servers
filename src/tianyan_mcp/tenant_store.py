from __future__ import annotations

import os
import re
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from tianyan_mcp.credential_crypto import ApiKeyCodec, EncryptedCredential, GeneratedApiKey
from tianyan_mcp.identity import TenantPrincipal
from tianyan_mcp.tenant_migrations import apply_migrations

_DISPLAY_NAME_KEY_PATTERN = re.compile(r"tymcp_[0-9a-f]{16}_[A-Za-z0-9_-]{43}")


class TenantStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class AuditEventType(StrEnum):
    TENANT_CREATED = "tenant_created"
    TENANT_STATUS_CHANGED = "tenant_status_changed"
    API_KEY_ROTATED = "api_key_rotated"
    API_KEY_REVOKED = "api_key_revoked"
    MCP_AUTHENTICATED = "mcp_authenticated"
    MCP_AUTH_FAILED = "mcp_auth_failed"
    CREDENTIAL_UPDATED = "credential_updated"
    CREDENTIAL_DELETED = "credential_deleted"
    SDK_AUTH_FAILED = "sdk_auth_failed"
    EXPERIMENT_SUBMITTED = "experiment_submitted"
    BACKUP_CREATED = "backup_created"


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    UNKNOWN = "unknown"


class TenantDisabledError(Exception):
    """Raised only after a valid key identifies a disabled tenant."""

    def __init__(self, principal: TenantPrincipal) -> None:
        super().__init__("tenant is disabled")
        self.principal = principal


@dataclass(frozen=True, slots=True)
class TenantRecord:
    id: str
    display_name: str
    status: TenantStatus
    auth_revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    id: str
    tenant_id: str
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


@dataclass(frozen=True, slots=True)
class CredentialRecord:
    tenant_id: str
    encrypted: EncryptedCredential
    credential_revision: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    configured: bool
    credential_revision: int
    updated_at: str


class TenantStore:
    def __init__(self, path: Path, codec: ApiKeyCodec) -> None:
        self.path = path
        self._codec = codec

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                apply_migrations(connection)
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def create_tenant(
        self,
        display_name: str,
        generated_key: GeneratedApiKey,
        *,
        audit_event: AuditEventType | None = None,
    ) -> TenantRecord:
        name = _validated_display_name(display_name)
        now = _utc_now()
        tenant = TenantRecord(
            id=str(uuid.uuid4()),
            display_name=name,
            status=TenantStatus.ACTIVE,
            auth_revision=1,
            created_at=now,
            updated_at=now,
        )
        with self._write() as connection:
            connection.execute(
                """INSERT INTO tenants
                (id, display_name, status, auth_revision, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    tenant.id,
                    tenant.display_name,
                    tenant.status.value,
                    tenant.auth_revision,
                    tenant.created_at,
                    tenant.updated_at,
                ),
            )
            connection.execute(
                """INSERT INTO api_keys
                (id, tenant_id, verification_tag, created_at)
                VALUES (?, ?, ?, ?)""",
                (generated_key.key_id, tenant.id, generated_key.verification_tag, now),
            )
            if audit_event is not None:
                self._append_audit(
                    connection,
                    tenant.id,
                    generated_key.key_id,
                    audit_event,
                    AuditOutcome.SUCCESS,
                    None,
                )
        return tenant

    def list_tenants(self) -> list[TenantRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, display_name, status, auth_revision, created_at, updated_at
                FROM tenants ORDER BY created_at, id"""
            ).fetchall()
        return [
            TenantRecord(
                id=row[0],
                display_name=row[1],
                status=TenantStatus(row[2]),
                auth_revision=row[3],
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        ]

    def authenticate(self, plaintext_key: str) -> TenantPrincipal | None:
        key_id = self._codec.parse_key_id(plaintext_key)
        if key_id is None:
            return None
        with self._write() as connection:
            row = connection.execute(
                """SELECT tenants.id, tenants.status, tenants.auth_revision,
                api_keys.verification_tag
                FROM api_keys JOIN tenants ON tenants.id = api_keys.tenant_id
                WHERE api_keys.id = ? AND api_keys.revoked_at IS NULL""",
                (key_id,),
            ).fetchone()
            if row is None or not self._codec.verify(plaintext_key, row[3]):
                return None
            principal = TenantPrincipal(tenant_id=row[0], key_id=key_id, auth_revision=row[2])
            if row[1] == TenantStatus.DISABLED.value:
                raise TenantDisabledError(principal)
            now = _utc_now()
            connection.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, key_id))
            return principal

    def set_tenant_status(
        self,
        tenant_id: str,
        *,
        enabled: bool,
        audit_event: AuditEventType | None = None,
    ) -> TenantRecord:
        status = TenantStatus.ACTIVE if enabled else TenantStatus.DISABLED
        with self._write() as connection:
            current = connection.execute(
                "SELECT display_name, auth_revision, created_at FROM tenants WHERE id = ?",
                (tenant_id,),
            ).fetchone()
            if current is None:
                raise KeyError("tenant does not exist")
            now = _utc_now()
            revision = current[1] + 1
            connection.execute(
                "UPDATE tenants SET status = ?, auth_revision = ?, updated_at = ? WHERE id = ?",
                (status.value, revision, now, tenant_id),
            )
            if audit_event is not None:
                self._append_audit(
                    connection,
                    tenant_id,
                    None,
                    audit_event,
                    AuditOutcome.SUCCESS,
                    None,
                )
            return TenantRecord(tenant_id, current[0], status, revision, current[2], now)

    def rotate_key(
        self,
        tenant_id: str,
        generated_key: GeneratedApiKey,
        *,
        audit_event: AuditEventType | None = None,
    ) -> ApiKeyRecord:
        with self._write() as connection:
            self._require_tenant(connection, tenant_id)
            now = _utc_now()
            connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE tenant_id = ? AND revoked_at IS NULL",
                (now, tenant_id),
            )
            connection.execute(
                """INSERT INTO api_keys
                (id, tenant_id, verification_tag, created_at)
                VALUES (?, ?, ?, ?)""",
                (generated_key.key_id, tenant_id, generated_key.verification_tag, now),
            )
            connection.execute(
                "UPDATE tenants SET auth_revision = auth_revision + 1, updated_at = ? WHERE id = ?",
                (now, tenant_id),
            )
            if audit_event is not None:
                self._append_audit(
                    connection,
                    tenant_id,
                    generated_key.key_id,
                    audit_event,
                    AuditOutcome.SUCCESS,
                    None,
                )
            return ApiKeyRecord(generated_key.key_id, tenant_id, now, None, None)

    def revoke_key(self, tenant_id: str, *, audit_event: AuditEventType | None = None) -> None:
        with self._write() as connection:
            self._require_tenant(connection, tenant_id)
            now = _utc_now()
            connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE tenant_id = ? AND revoked_at IS NULL",
                (now, tenant_id),
            )
            connection.execute(
                "UPDATE tenants SET auth_revision = auth_revision + 1, updated_at = ? WHERE id = ?",
                (now, tenant_id),
            )
            if audit_event is not None:
                self._append_audit(
                    connection,
                    tenant_id,
                    None,
                    audit_event,
                    AuditOutcome.SUCCESS,
                    None,
                )

    def put_credential(
        self,
        tenant_id: str,
        encrypted: EncryptedCredential,
        *,
        audit_event: AuditEventType | None = None,
        audit_key_id: str | None = None,
        request_id: str | None = None,
    ) -> CredentialStatus:
        with self._write() as connection:
            self._require_tenant(connection, tenant_id)
            current = connection.execute(
                "SELECT credential_revision FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()
            if current is None:
                raise RuntimeError("tenant disappeared during credential update")
            revision = current[0] + 1
            now = _utc_now()
            connection.execute(
                "UPDATE tenants SET credential_revision = ?, updated_at = ? WHERE id = ?",
                (revision, now, tenant_id),
            )
            connection.execute(
                """INSERT INTO sdk_credentials
                (tenant_id, ciphertext, nonce, key_version, credential_revision, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    ciphertext = excluded.ciphertext,
                    nonce = excluded.nonce,
                    key_version = excluded.key_version,
                    credential_revision = excluded.credential_revision,
                    updated_at = excluded.updated_at""",
                (
                    tenant_id,
                    encrypted.ciphertext,
                    encrypted.nonce,
                    encrypted.key_version,
                    revision,
                    now,
                ),
            )
            if audit_event is not None:
                self._append_audit(
                    connection,
                    tenant_id,
                    audit_key_id,
                    audit_event,
                    AuditOutcome.SUCCESS,
                    request_id,
                )
            return CredentialStatus(True, revision, now)

    def get_credential(self, tenant_id: str) -> CredentialRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT ciphertext, nonce, key_version, credential_revision, updated_at
                FROM sdk_credentials WHERE tenant_id = ?""",
                (tenant_id,),
            ).fetchone()
        if row is None:
            return None
        return CredentialRecord(
            tenant_id=tenant_id,
            encrypted=EncryptedCredential(ciphertext=row[0], nonce=row[1], key_version=row[2]),
            credential_revision=row[3],
            updated_at=row[4],
        )

    def delete_credential(
        self,
        tenant_id: str,
        *,
        audit_event: AuditEventType | None = None,
        audit_key_id: str | None = None,
        request_id: str | None = None,
    ) -> CredentialStatus:
        with self._write() as connection:
            self._require_tenant(connection, tenant_id)
            current = connection.execute(
                "SELECT credential_revision FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchone()
            if current is None:
                raise RuntimeError("tenant disappeared during credential deletion")
            revision = current[0] + 1
            now = _utc_now()
            connection.execute(
                "UPDATE tenants SET credential_revision = ?, updated_at = ? WHERE id = ?",
                (revision, now, tenant_id),
            )
            connection.execute("DELETE FROM sdk_credentials WHERE tenant_id = ?", (tenant_id,))
            if audit_event is not None:
                self._append_audit(
                    connection,
                    tenant_id,
                    audit_key_id,
                    audit_event,
                    AuditOutcome.SUCCESS,
                    request_id,
                )
            return CredentialStatus(False, revision, now)

    def append_audit(
        self,
        tenant_id: object | None,
        key_id: object | None,
        event_type: object,
        outcome: object,
        request_id: object | None,
    ) -> None:
        tenant_id = _validate_audit_uuid(tenant_id)
        key_id = _validate_audit_key_id(key_id)
        request_id = _validate_audit_uuid(request_id)
        if not isinstance(event_type, AuditEventType):
            raise TypeError("audit event type is invalid")
        if not isinstance(outcome, AuditOutcome):
            raise TypeError("audit outcome is invalid")
        with self._write() as connection:
            self._append_audit(connection, tenant_id, key_id, event_type, outcome, request_id)

    def backup(self, destination: Path) -> None:
        destination_path, _ = self._reserve_backup_destination(destination)
        self._copy_backup(destination_path)

    def backup_with_audit(self, destination: Path) -> None:
        destination_path, identity = self._reserve_backup_destination(destination)
        try:
            self._copy_backup(destination_path)
            with self._write() as connection:
                self._append_audit(
                    connection,
                    None,
                    None,
                    AuditEventType.BACKUP_CREATED,
                    AuditOutcome.SUCCESS,
                    None,
                )
        except BaseException:
            self._cleanup_reserved_backup(destination_path, identity)
            raise

    def _reserve_backup_destination(self, destination: Path) -> tuple[Path, tuple[int, int]]:
        if destination.is_symlink():
            raise ValueError("backup destination is invalid")
        source_path = self.path.resolve()
        destination_path = destination.resolve()
        if (
            destination_path == source_path
            or destination_path.exists()
            or destination_path.is_symlink()
        ):
            raise ValueError("backup destination is invalid")
        try:
            descriptor = os.open(destination_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            raise ValueError("backup destination is invalid") from None
        try:
            state = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        return destination_path, (state.st_dev, state.st_ino)

    def _copy_backup(self, destination_path: Path) -> None:
        target = sqlite3.connect(destination_path)
        try:
            with self._connect() as source:
                source.backup(target)
        finally:
            target.close()

    @staticmethod
    def _cleanup_reserved_backup(destination_path: Path, identity: tuple[int, int]) -> None:
        try:
            state = destination_path.lstat()
        except FileNotFoundError:
            return
        if (state.st_dev, state.st_ino) != identity:
            return
        destination_path.unlink()

    @staticmethod
    def _append_audit(
        connection: sqlite3.Connection,
        tenant_id: object | None,
        key_id: object | None,
        event_type: object,
        outcome: object,
        request_id: object | None,
    ) -> None:
        tenant_id = _validate_audit_uuid(tenant_id)
        key_id = _validate_audit_key_id(key_id)
        request_id = _validate_audit_uuid(request_id)
        if not isinstance(event_type, AuditEventType):
            raise TypeError("audit event type is invalid")
        if not isinstance(outcome, AuditOutcome):
            raise TypeError("audit outcome is invalid")
        connection.execute(
            """INSERT INTO audit_events
            (id, occurred_at, tenant_id, key_id, event_type, outcome, request_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                _utc_now(),
                tenant_id,
                key_id,
                event_type.value,
                outcome.value,
                request_id,
            ),
        )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self) -> Generator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _require_tenant(connection: sqlite3.Connection, tenant_id: str) -> None:
        if (
            connection.execute("SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
            is None
        ):
            raise KeyError("tenant does not exist")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validated_display_name(display_name: str) -> str:
    name = display_name.strip()
    if not 1 <= len(name) <= 128:
        raise ValueError("display name must contain 1 to 128 characters")
    if not all(character.isprintable() for character in name):
        raise ValueError("display name contains invalid characters")
    if _DISPLAY_NAME_KEY_PATTERN.search(name) is not None:
        raise ValueError("display name contains sensitive key material")
    return name


_AUDIT_KEY_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def _validate_audit_uuid(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 36:
        raise ValueError("audit identifier is invalid")
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise ValueError("audit identifier is invalid") from None
    if str(parsed) != value:
        raise ValueError("audit identifier is invalid")
    return value


def _validate_audit_key_id(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _AUDIT_KEY_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("audit key identifier is invalid")
    return value
