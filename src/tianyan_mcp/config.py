from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretBytes, SecretStr, field_validator, model_validator

from tianyan_mcp.identity import IdentityMode


def _empty_credential_keys() -> dict[int, SecretBytes]:
    return {}


class Settings(BaseModel):
    api_key: SecretStr | None = None
    sdk_open_id: SecretStr | None = None
    allow_hardware_submission: bool = False
    allowed_origins: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
    http_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    live_test: bool = False
    identity_mode: IdentityMode = IdentityMode.SINGLE_USER
    database_path: Path | None = None
    api_key_pepper: SecretBytes | None = None
    credential_key_version: int | None = None
    credential_keys: dict[int, SecretBytes] = Field(default_factory=_empty_credential_keys)
    direct_sdk_fingerprint_key: SecretBytes | None = None
    direct_sdk_verify_openid: bool = True
    direct_sdk_verify_reject_ttl_seconds: float = Field(default=60.0, gt=0, le=300)
    direct_sdk_max_verify_attempts: int = Field(default=5, ge=1, le=20)
    direct_sdk_auth_enforcement: Literal["enforce", "shadow"] = "enforce"
    direct_sdk_rate_limit_enabled: bool = True
    direct_sdk_rate_limit_window_seconds: float = Field(default=60.0, gt=0, le=3600)
    direct_sdk_rate_limit_max_requests: int = Field(default=120, ge=1, le=100_000)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value().strip()) < 32:
            raise ValueError("TIANYAN_MCP_API_KEY must contain at least 32 characters")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def reject_wildcard_origin(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if "*" in value:
            raise ValueError("wildcard Origin is not allowed")
        return value

    @model_validator(mode="after")
    def validate_multi_tenant_contract(self) -> Settings:
        if self.identity_mode is IdentityMode.DIRECT_SDK:
            if self.api_key is not None or self.sdk_open_id is not None:
                raise ValueError("single-user credential is not allowed in direct_sdk mode")
            if self.direct_sdk_fingerprint_key is None:
                raise ValueError(
                    "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY is required in direct_sdk mode"
                )
            if len(self.direct_sdk_fingerprint_key.get_secret_value()) < 32:
                raise ValueError(
                    "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY must contain at least 32 bytes"
                )
            if not _is_loopback_bind_host(self.host):
                raise ValueError("TIANYAN_MCP_HOST must be a loopback address in direct_sdk mode")
        if self.identity_mode is IdentityMode.MULTI_TENANT:
            if self.api_key is not None or self.sdk_open_id is not None:
                raise ValueError("single-user credential is not allowed in multi_tenant mode")
            if self.database_path is None:
                raise ValueError("TIANYAN_DATABASE_PATH is required in multi_tenant mode")
            if self.api_key_pepper is None:
                raise ValueError("TIANYAN_API_KEY_PEPPER is required in multi_tenant mode")
            if len(self.api_key_pepper.get_secret_value()) < 32:
                raise ValueError("TIANYAN_API_KEY_PEPPER must contain at least 32 bytes")
            if self.credential_key_version not in self.credential_keys:
                raise ValueError("active credential key material is required in multi_tenant mode")
            if not self.database_path.is_absolute():
                raise ValueError(
                    "TIANYAN_DATABASE_PATH must be an absolute path in multi_tenant mode"
                )
            if not _is_loopback_bind_host(self.host):
                raise ValueError("TIANYAN_MCP_HOST must be a loopback address in multi_tenant mode")
        return self

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        require_api_key: bool = True,
        default_identity_mode: IdentityMode | None = IdentityMode.SINGLE_USER,
    ) -> Settings:
        env = os.environ if environ is None else environ
        raw_key = env.get("TIANYAN_MCP_API_KEY", "").strip()
        raw_allow_hardware_submission = env.get("TIANYAN_ALLOW_HARDWARE_SUBMISSION")
        raw_open_id = env.get("TIANYAN_SDK_OPEN_ID", "").strip()
        raw_identity_mode = env.get("TIANYAN_IDENTITY_MODE", "").strip()
        if raw_identity_mode:
            identity_mode = IdentityMode(raw_identity_mode)
        elif default_identity_mode is not None:
            identity_mode = default_identity_mode
        else:
            raise ValueError("TIANYAN_IDENTITY_MODE is required")

        database_path: Path | None = None
        api_key_pepper: SecretBytes | None = None
        credential_key_version: int | None = None
        credential_keys: dict[int, SecretBytes] = {}
        direct_sdk_fingerprint_key: SecretBytes | None = None

        if identity_mode is IdentityMode.MULTI_TENANT:
            if raw_key or raw_open_id:
                raise ValueError("single-user credential is not allowed in multi_tenant mode")
            raw_database_path = env.get("TIANYAN_DATABASE_PATH", "").strip()
            if not raw_database_path:
                raise ValueError("TIANYAN_DATABASE_PATH is required in multi_tenant mode")
            database_path = Path(raw_database_path)
            raw_api_key_pepper = env.get("TIANYAN_API_KEY_PEPPER", "").strip()
            if not raw_api_key_pepper:
                raise ValueError("TIANYAN_API_KEY_PEPPER is required in multi_tenant mode")
            api_key_pepper = SecretBytes(raw_api_key_pepper.encode())
            raw_credential_key_version = env.get(
                "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION", ""
            ).strip()
            credential_key_version = _positive_integer(
                "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION", raw_credential_key_version
            )
            credential_keys = _credential_keys(env)
            if credential_key_version not in credential_keys:
                raise ValueError(
                    f"TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V{credential_key_version} is required"
                )
            raw_key = ""
            raw_open_id = ""

        if identity_mode is IdentityMode.DIRECT_SDK:
            if raw_key or raw_open_id:
                raise ValueError("single-user credential is not allowed in direct_sdk mode")
            raw_fingerprint_key = env.get("TIANYAN_DIRECT_SDK_FINGERPRINT_KEY", "").strip()
            if not raw_fingerprint_key:
                raise ValueError(
                    "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY is required in direct_sdk mode"
                )
            direct_sdk_fingerprint_key = SecretBytes(raw_fingerprint_key.encode("utf-8"))
            raw_key = ""
            raw_open_id = ""

        if identity_mode is IdentityMode.SINGLE_USER and require_api_key and not raw_key:
            raise ValueError("TIANYAN_MCP_API_KEY is required")
        raw_verify_openid = env.get("TIANYAN_DIRECT_SDK_VERIFY_OPENID", "true").strip()
        raw_enforcement = env.get("TIANYAN_DIRECT_SDK_AUTH_ENFORCEMENT", "enforce").strip()
        if raw_enforcement not in ("enforce", "shadow"):
            raise ValueError("TIANYAN_DIRECT_SDK_AUTH_ENFORCEMENT must be 'enforce' or 'shadow'")
        raw_rate_limit_enabled = env.get("TIANYAN_DIRECT_SDK_RATE_LIMIT_ENABLED", "true").strip()
        return cls(
            api_key=SecretStr(raw_key) if raw_key else None,
            identity_mode=identity_mode,
            database_path=database_path,
            api_key_pepper=api_key_pepper,
            credential_key_version=credential_key_version,
            credential_keys=credential_keys,
            direct_sdk_fingerprint_key=direct_sdk_fingerprint_key,
            sdk_open_id=SecretStr(raw_open_id) if raw_open_id else None,
            allow_hardware_submission=(
                False
                if raw_allow_hardware_submission is None
                else _strict_bool(
                    "TIANYAN_ALLOW_HARDWARE_SUBMISSION",
                    raw_allow_hardware_submission,
                )
            ),
            allowed_origins=_csv(env.get("TIANYAN_MCP_ALLOWED_ORIGINS", "")),
            allowed_hosts=_csv(env.get("TIANYAN_MCP_ALLOWED_HOSTS", "127.0.0.1,localhost,::1")),
            http_timeout_seconds=float(env.get("TIANYAN_HTTP_TIMEOUT_SECONDS", "10")),
            host=env.get("TIANYAN_MCP_HOST", "127.0.0.1").strip(),
            port=int(env.get("TIANYAN_MCP_PORT", "8000")),
            live_test=env.get("TIANYAN_LIVE_TEST", "0") == "1",
            direct_sdk_verify_openid=_strict_bool(
                "TIANYAN_DIRECT_SDK_VERIFY_OPENID", raw_verify_openid
            ),
            direct_sdk_verify_reject_ttl_seconds=float(
                env.get("TIANYAN_DIRECT_SDK_VERIFY_REJECT_TTL_SECONDS", "60")
            ),
            direct_sdk_max_verify_attempts=int(
                env.get("TIANYAN_DIRECT_SDK_MAX_VERIFY_ATTEMPTS", "5")
            ),
            direct_sdk_auth_enforcement=raw_enforcement,
            direct_sdk_rate_limit_enabled=_strict_bool(
                "TIANYAN_DIRECT_SDK_RATE_LIMIT_ENABLED", raw_rate_limit_enabled
            ),
            direct_sdk_rate_limit_window_seconds=float(
                env.get("TIANYAN_DIRECT_SDK_RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
            direct_sdk_rate_limit_max_requests=int(
                env.get("TIANYAN_DIRECT_SDK_RATE_LIMIT_MAX_REQUESTS", "120")
            ),
        )

    def api_key_value(self) -> str:
        if self.api_key is None:
            raise RuntimeError("API key is not configured")
        return self.api_key.get_secret_value().strip()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _is_loopback_bind_host(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _strict_bool(name: str, value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false'")


_CREDENTIAL_KEY_PATTERN = re.compile(r"^TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V([1-9]\d*)$")


def _positive_integer(name: str, value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _credential_keys(environ: Mapping[str, str]) -> dict[int, SecretBytes]:
    keys: dict[int, SecretBytes] = {}
    for name, value in environ.items():
        match = _CREDENTIAL_KEY_PATTERN.fullmatch(name)
        if match is None or not value.strip():
            continue
        try:
            decoded = base64.b64decode(value.strip(), validate=True)
        except binascii.Error as error:
            raise ValueError(f"{name} must be a Base64-encoded 32-byte key") from error
        if len(decoded) != 32:
            raise ValueError(f"{name} must be a Base64-encoded 32-byte key")
        keys[int(match.group(1))] = SecretBytes(decoded)
    return keys
