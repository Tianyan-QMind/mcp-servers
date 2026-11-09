from __future__ import annotations

import sqlite3
from collections.abc import Callable

Migration = Callable[[sqlite3.Connection], None]


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply every ordered tenant-store schema migration exactly once."""
    connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
    applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    for version, migration in enumerate(_MIGRATIONS, start=1):
        if version in applied:
            continue
        migration(connection)
        connection.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))


def _migration_1(connection: sqlite3.Connection) -> None:
    for statement in _MIGRATION_1_STATEMENTS:
        connection.execute(statement)


_MIGRATIONS: tuple[Migration, ...] = (_migration_1,)


_MIGRATION_1_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE tenants (
        id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
        auth_revision INTEGER NOT NULL,
        credential_revision INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE api_keys (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL REFERENCES tenants(id),
        verification_tag BLOB NOT NULL,
        created_at TEXT NOT NULL,
        last_used_at TEXT,
        revoked_at TEXT
    )""",
    "CREATE UNIQUE INDEX one_active_key_per_tenant ON api_keys(tenant_id) WHERE revoked_at IS NULL",
    """CREATE TABLE sdk_credentials (
        tenant_id TEXT PRIMARY KEY REFERENCES tenants(id),
        ciphertext BLOB NOT NULL,
        nonce BLOB NOT NULL,
        key_version INTEGER NOT NULL,
        credential_revision INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE audit_events (
        id TEXT PRIMARY KEY,
        occurred_at TEXT NOT NULL,
        tenant_id TEXT REFERENCES tenants(id),
        key_id TEXT,
        event_type TEXT NOT NULL,
        outcome TEXT NOT NULL,
        request_id TEXT
    )""",
)
