import base64
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretBytes, SecretStr, ValidationError

from tianyan_mcp.config import Settings
from tianyan_mcp.identity import IdentityMode

VALID_KEY = "k" * 32


def test_requires_api_key_for_http_mode() -> None:
    with pytest.raises(ValueError, match="TIANYAN_MCP_API_KEY is required"):
        Settings.from_env({})


def test_rejects_short_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"TIANYAN_MCP_API_KEY": "short"})


def test_parses_security_and_runtime_settings() -> None:
    settings = Settings.from_env(
        {
            "TIANYAN_MCP_API_KEY": VALID_KEY,
            "TIANYAN_MCP_ALLOWED_ORIGINS": ("https://agent.example.com,http://127.0.0.1:6274"),
            "TIANYAN_MCP_ALLOWED_HOSTS": "mcp.example.com,localhost",
            "TIANYAN_HTTP_TIMEOUT_SECONDS": "8",
            "TIANYAN_MCP_HOST": "127.0.0.1",
            "TIANYAN_MCP_PORT": "9000",
        }
    )

    assert settings.api_key_value() == VALID_KEY
    assert settings.allowed_origins == (
        "https://agent.example.com",
        "http://127.0.0.1:6274",
    )
    assert settings.allowed_hosts == ("mcp.example.com", "localhost")
    assert settings.http_timeout_seconds == 8
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000


def test_rejects_wildcard_origin() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env(
            {
                "TIANYAN_MCP_API_KEY": VALID_KEY,
                "TIANYAN_MCP_ALLOWED_ORIGINS": "*",
            }
        )


def test_api_key_requirement_can_be_disabled() -> None:
    settings = Settings.from_env({}, require_api_key=False)
    assert settings.api_key is None


def test_parses_optional_sdk_submission_settings() -> None:
    settings = Settings.from_env(
        {
            "TIANYAN_MCP_API_KEY": VALID_KEY,
            "TIANYAN_SDK_OPEN_ID": "fake-sdk-open-id",
            "TIANYAN_ALLOW_HARDWARE_SUBMISSION": "true",
        }
    )

    assert settings.sdk_open_id is not None
    assert settings.sdk_open_id.get_secret_value() == "fake-sdk-open-id"
    assert settings.allow_hardware_submission is True


def test_hardware_submission_defaults_to_disabled() -> None:
    settings = Settings.from_env({"TIANYAN_MCP_API_KEY": VALID_KEY})
    assert settings.sdk_open_id is None
    assert settings.allow_hardware_submission is False


@pytest.mark.parametrize("value", ["", " ", "1", "yes", "TRUE", "TRUE ", "enabled"])
def test_rejects_ambiguous_hardware_flag(value: str) -> None:
    with pytest.raises(ValueError, match="TIANYAN_ALLOW_HARDWARE_SUBMISSION"):
        Settings.from_env(
            {
                "TIANYAN_MCP_API_KEY": VALID_KEY,
                "TIANYAN_ALLOW_HARDWARE_SUBMISSION": value,
            }
        )


@pytest.fixture
def valid_multi_env(tmp_path: Path) -> dict[str, str]:
    return {
        "TIANYAN_IDENTITY_MODE": "multi_tenant",
        "TIANYAN_DATABASE_PATH": str(tmp_path / "tenant.db"),
        "TIANYAN_API_KEY_PEPPER": "p" * 32,
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION": "1",
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V1": base64.b64encode(b"k" * 32).decode(),
    }


def test_multi_tenant_settings_require_store_secrets(tmp_path: Path) -> None:
    env = {
        "TIANYAN_IDENTITY_MODE": "multi_tenant",
        "TIANYAN_DATABASE_PATH": str(tmp_path / "tenant.db"),
    }
    with pytest.raises(ValueError, match="TIANYAN_API_KEY_PEPPER"):
        Settings.from_env(env)


def test_multi_tenant_settings_reject_short_api_key_pepper_from_env(
    valid_multi_env: dict[str, str],
) -> None:
    valid_multi_env["TIANYAN_API_KEY_PEPPER"] = "p" * 31

    with pytest.raises(ValidationError, match="at least 32 bytes"):
        Settings.from_env(valid_multi_env)


def test_direct_multi_tenant_settings_reject_short_api_key_pepper() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        Settings(
            identity_mode=IdentityMode.MULTI_TENANT,
            database_path=Path("tenant.db"),
            api_key_pepper=SecretBytes(b"p" * 31),
            credential_key_version=1,
            credential_keys={1: SecretBytes(b"k" * 32)},
        )


def test_multi_tenant_rejects_global_credentials(valid_multi_env: dict[str, str]) -> None:
    valid_multi_env["TIANYAN_SDK_OPEN_ID"] = "must-not-be-used"
    with pytest.raises(ValueError, match="single-user credential"):
        Settings.from_env(valid_multi_env)


def test_multi_tenant_parses_versioned_credential_keys(valid_multi_env: dict[str, str]) -> None:
    settings = Settings.from_env(valid_multi_env)
    assert settings.credential_key_version == 1
    assert settings.credential_keys[1].get_secret_value() == b"k" * 32


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": SecretStr(VALID_KEY)}, "single-user credential"),
        ({"sdk_open_id": SecretStr("fake-sdk-open-id")}, "single-user credential"),
        ({"database_path": None}, "TIANYAN_DATABASE_PATH"),
        ({"api_key_pepper": None}, "TIANYAN_API_KEY_PEPPER"),
        ({"credential_keys": {}}, "active credential key material"),
    ],
)
def test_direct_multi_tenant_settings_require_complete_isolated_configuration(
    overrides: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "identity_mode": IdentityMode.MULTI_TENANT,
        "database_path": Path("tenant.db"),
        "api_key_pepper": SecretBytes(b"p" * 32),
        "credential_key_version": 1,
        "credential_keys": {1: SecretBytes(b"k" * 32)},
    }
    base.update(overrides)
    with pytest.raises(ValidationError, match=message):
        Settings(**cast(Any, base))


def test_direct_sdk_parses_isolated_fingerprint_key() -> None:
    settings = Settings.from_env(
        {
            "TIANYAN_IDENTITY_MODE": "direct_sdk",
            "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
            "TIANYAN_MCP_HOST": "127.0.0.1",
        }
    )

    assert settings.identity_mode is IdentityMode.DIRECT_SDK
    assert settings.direct_sdk_fingerprint_key is not None
    assert settings.direct_sdk_fingerprint_key.get_secret_value() == b"f" * 32
    assert settings.database_path is None
    assert settings.direct_sdk_verify_openid is True
    assert settings.direct_sdk_verify_reject_ttl_seconds == 60.0
    assert settings.direct_sdk_max_verify_attempts == 5
    assert settings.direct_sdk_auth_enforcement == "enforce"
    assert settings.direct_sdk_rate_limit_enabled is True
    assert settings.direct_sdk_rate_limit_window_seconds == 60.0
    assert settings.direct_sdk_rate_limit_max_requests == 120


def test_direct_sdk_parses_openid_verification_settings() -> None:
    settings = Settings.from_env(
        {
            "TIANYAN_IDENTITY_MODE": "direct_sdk",
            "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
            "TIANYAN_MCP_HOST": "127.0.0.1",
            "TIANYAN_DIRECT_SDK_VERIFY_OPENID": "false",
            "TIANYAN_DIRECT_SDK_VERIFY_REJECT_TTL_SECONDS": "120",
            "TIANYAN_DIRECT_SDK_MAX_VERIFY_ATTEMPTS": "3",
            "TIANYAN_DIRECT_SDK_AUTH_ENFORCEMENT": "shadow",
        }
    )

    assert settings.direct_sdk_verify_openid is False
    assert settings.direct_sdk_verify_reject_ttl_seconds == 120.0
    assert settings.direct_sdk_max_verify_attempts == 3
    assert settings.direct_sdk_auth_enforcement == "shadow"


def test_direct_sdk_parses_rate_limit_settings() -> None:
    settings = Settings.from_env(
        {
            "TIANYAN_IDENTITY_MODE": "direct_sdk",
            "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
            "TIANYAN_MCP_HOST": "127.0.0.1",
            "TIANYAN_DIRECT_SDK_RATE_LIMIT_ENABLED": "false",
            "TIANYAN_DIRECT_SDK_RATE_LIMIT_WINDOW_SECONDS": "10",
            "TIANYAN_DIRECT_SDK_RATE_LIMIT_MAX_REQUESTS": "30",
        }
    )

    assert settings.direct_sdk_rate_limit_enabled is False
    assert settings.direct_sdk_rate_limit_window_seconds == 10.0
    assert settings.direct_sdk_rate_limit_max_requests == 30


@pytest.mark.parametrize(
    "extra",
    [
        {"TIANYAN_DIRECT_SDK_RATE_LIMIT_ENABLED": "yes"},
        {"TIANYAN_DIRECT_SDK_RATE_LIMIT_WINDOW_SECONDS": "0"},
        {"TIANYAN_DIRECT_SDK_RATE_LIMIT_WINDOW_SECONDS": "3601"},
        {"TIANYAN_DIRECT_SDK_RATE_LIMIT_MAX_REQUESTS": "0"},
        {"TIANYAN_DIRECT_SDK_RATE_LIMIT_MAX_REQUESTS": "100001"},
    ],
)
def test_direct_sdk_rejects_invalid_rate_limit_settings(extra: dict[str, str]) -> None:
    env = {
        "TIANYAN_IDENTITY_MODE": "direct_sdk",
        "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
        "TIANYAN_MCP_HOST": "127.0.0.1",
        **extra,
    }
    with pytest.raises((ValueError, ValidationError)):
        Settings.from_env(env)


@pytest.mark.parametrize(
    "extra",
    [
        {"TIANYAN_DIRECT_SDK_VERIFY_OPENID": "yes"},
        {"TIANYAN_DIRECT_SDK_VERIFY_REJECT_TTL_SECONDS": "0"},
        {"TIANYAN_DIRECT_SDK_VERIFY_REJECT_TTL_SECONDS": "301"},
        {"TIANYAN_DIRECT_SDK_MAX_VERIFY_ATTEMPTS": "0"},
        {"TIANYAN_DIRECT_SDK_MAX_VERIFY_ATTEMPTS": "21"},
        {"TIANYAN_DIRECT_SDK_AUTH_ENFORCEMENT": "permissive"},
    ],
)
def test_direct_sdk_rejects_invalid_openid_verification_settings(extra: dict[str, str]) -> None:
    env = {
        "TIANYAN_IDENTITY_MODE": "direct_sdk",
        "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
        "TIANYAN_MCP_HOST": "127.0.0.1",
        **extra,
    }
    with pytest.raises((ValueError, ValidationError)):
        Settings.from_env(env)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({}, "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY"),
        ({"TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 31}, "at least 32 bytes"),
        (
            {
                "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
                "TIANYAN_MCP_API_KEY": VALID_KEY,
            },
            "single-user credential",
        ),
        (
            {
                "TIANYAN_DIRECT_SDK_FINGERPRINT_KEY": "f" * 32,
                "TIANYAN_SDK_OPEN_ID": "fake-open-id",
            },
            "single-user credential",
        ),
    ],
)
def test_direct_sdk_rejects_incomplete_or_conflicting_settings(
    extra: dict[str, str], message: str
) -> None:
    env = {"TIANYAN_IDENTITY_MODE": "direct_sdk", **extra}
    with pytest.raises((ValueError, ValidationError), match=message):
        Settings.from_env(env)


def test_direct_sdk_settings_reject_public_bind_host() -> None:
    with pytest.raises(ValidationError, match="loopback"):
        Settings(
            identity_mode=IdentityMode.DIRECT_SDK,
            direct_sdk_fingerprint_key=SecretBytes(b"f" * 32),
            host=".".join(("0", "0", "0", "0")),
        )
